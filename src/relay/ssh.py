"""How relay talks to the cluster: one place that builds every `ssh` argv.

Klone requires Duo for every new SSH connection. That makes connections
expensive in the one way that matters for a polling daemon: a human has to
tap a phone. So relay never lets `ssh` open a fresh connection on its own.
Instead it runs its *own* ControlMaster -- a long-lived background connection
that every later `ssh` invocation tunnels through -- and every argv relay
builds says so explicitly with `-o` options.

Why not rely on the user's `~/.ssh/config`? Because most users will not have
set ControlMaster up, and the ones who have set it up differently. Command
line `-o` options override the config file, so building the options here
works whether or not the user did anything. The alias still comes from
`~/.ssh/config`: it supplies the username, hostname, keys and any jump host.
relay never stores or constructs any of those.

Two things in here are load-bearing and easy to get wrong:

  * `BatchMode=yes` on every non-interactive call. Without it, an ssh with no
    live master would stop and wait for a Duo prompt that nobody will ever
    see, and the daemon would hang forever. With it, ssh fails immediately,
    and the daemon can report "run `relay connect`" instead.

  * `ControlPath` under `~/.relay/`, not under the XDG data directory. A Unix
    socket path is limited to about 104 bytes on macOS (108 on Linux), and
    the XDG path plus a hash can exceed that. `%C` expands to a hash of
    host, port and user, so the path stays short and unique per host.

`relay connect` is the one deliberately interactive call: it starts the
master with `-M -N -f` and *without* BatchMode so the user can answer Duo.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# How long the master keeps running after its last client disconnects. The
# daemon's own polling keeps it busy, so in practice this is the ceiling on
# how long relay can go without a Duo tap. Eight hours is a working day.
CONTROL_PERSIST = "8h"

# The control socket template. `%C` is expanded by ssh itself into a hash of
# the connection parameters; relay passes it through verbatim.
CONTROL_PATH_TEMPLATE = "~/.relay/cm-%C"

# Keepalives. A laptop that sleeps or changes networks kills the TCP
# connection underneath the master; these make ssh notice within
# 30 * 3 = 90 seconds instead of hanging on a dead socket.
SERVER_ALIVE_INTERVAL = "30"
SERVER_ALIVE_COUNT_MAX = "3"

# The failure kinds the daemon distinguishes. See `classify_failure`.
AUTH_REQUIRED = "auth_required"
UNREACHABLE = "unreachable"

# How long one remote round trip may take before relay gives up on it.
#
# This is not a network number. Through a live ControlMaster there is no
# handshake at all; what takes the time is the login node starting a shell
# for the command, and on Klone that measured 4 to 5 seconds with the master
# already open (a `conda init` in ~/.bashrc, run by a shared and busy node).
# A loaded login node is much worse. So the floor to plan for is about 5
# seconds per round trip, and 30 leaves room for a bad day without letting
# the daemon hang on a dead socket for a minute. Configurable as
# `ssh_timeout`; every remote call in relay uses that one value, never a
# literal of its own, so nothing can be accidentally tighter than the rest.
DEFAULT_SSH_TIMEOUT = 30.0

# How relay runs its *own* commands on the login node. `-T` (no pty: these are
# probes, not sessions) plus a bare bash that reads no startup files. This
# skips whatever the user's rc files would do for the inner shell; it cannot
# skip the login shell sshd itself spawns to run this command, which on some
# bash builds (SSH_SOURCE_BASHRC) still sources ~/.bashrc. That outer cost is
# what the doctor's slow-round-trip warning points at. Never applied to the
# training job: that runs under Slurm, in the user's own environment.
REMOTE_SHELL = ("bash", "--noprofile", "--norc", "-c")


def remote_shell_argv(command: str) -> list[str]:
    """Wrap one already-quoted shell command line in relay's clean shell.

    `command` is a string a POSIX shell would execute (the caller has done any
    quoting of paths and arguments). The result is the remote argv:
    `bash --noprofile --norc -c '<command>'`, ready for `build_ssh_argv`.
    """
    return [*REMOTE_SHELL, command]


def relay_dir() -> Path:
    """`~/.relay/`, the directory the control socket lives in. Not created."""
    return Path(os.path.expanduser("~/.relay"))


def ensure_control_dir() -> Path:
    """Make sure `~/.relay/` exists as a directory with mode 0700.

    ssh does not create the ControlPath's parent directory. It fails with
    "unix_listener: cannot bind to path ...: No such file or directory" --
    *after* the user has answered Duo, which is the worst possible moment.
    The first real run on Klone hit exactly that, and a fake ssh runner can
    never catch it, because the bind happens inside the real ssh binary.

    Called by every argv builder below, so any code path that is about to run
    ssh -- `relay connect`, the daemon, doctor, the backend -- has the
    directory in place first. The daemon and doctor can run before `connect`
    ever has, so putting this only in `connect` would not be enough.

    Mode 0700 is chmod'ed explicitly even when the directory already exists:
    `mkdir(mode=)` only applies to a directory it creates, and a socket other
    users can connect to is a socket other users can run commands through.
    """
    path = relay_dir()
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
    except FileExistsError:
        # exist_ok only forgives an existing *directory*; a plain file at the
        # path still raises. Fall through to the is_dir check so both shapes
        # of "something is in the way" produce the same sentence.
        pass
    if not path.is_dir():
        raise NotADirectoryError(
            f"{path} exists but is not a directory. relay keeps its SSH control "
            "socket there; move the file out of the way and try again."
        )
    if (path.stat().st_mode & 0o777) != 0o700:
        os.chmod(path, 0o700)
    return path


def _common_options() -> list[str]:
    """Options shared by every relay ssh call, interactive or not.

    Also the one place that guarantees the ControlPath directory exists: every
    argv relay builds passes through here, so no ssh invocation can happen
    without it.
    """
    ensure_control_dir()
    return [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={CONTROL_PATH_TEMPLATE}",
        "-o", f"ControlPersist={CONTROL_PERSIST}",
        "-o", f"ServerAliveInterval={SERVER_ALIVE_INTERVAL}",
        "-o", f"ServerAliveCountMax={SERVER_ALIVE_COUNT_MAX}",
    ]


def build_ssh_argv(alias: str, remote_argv: list[str]) -> list[str]:
    """The argv for one non-interactive remote command.

    `remote_argv` is passed through *as given*. Callers that need shell
    quoting on the remote side must apply it themselves, because ssh joins
    its trailing arguments with spaces and hands the string to the remote
    login shell. `SlurmBackend._ssh` does that quoting in one place.

    Never builds `user@host`: `alias` is passed bare and the user's own ssh
    config resolves it.
    """
    _check_alias(alias)
    return [
        "ssh",
        # No pseudo-terminal: these are commands whose output relay parses,
        # not interactive sessions. A pty would turn "\n" into "\r\n" and
        # let the remote shell think somebody is watching.
        "-T",
        *_common_options(),
        "-o", "BatchMode=yes",
        alias,
        *remote_argv,
    ]


def connect_argv(alias: str) -> list[str]:
    """The argv `relay connect` runs to start the master interactively.

    `-M` makes this ssh *be* the master rather than look for one, `-N` runs
    no remote command, `-f` backgrounds it once authentication succeeds --
    which is exactly the moment Duo has been answered. No BatchMode here,
    on purpose: this is the one call that is allowed to prompt.
    """
    _check_alias(alias)
    return ["ssh", *_common_options(), "-M", "-N", "-f", alias]


def master_check_argv(alias: str) -> list[str]:
    """The argv for `ssh -O check`: "is a master alive for this alias?"

    Talks only to the local control socket, so it costs nothing and can
    never trigger Duo. Exit 0 means a live master; anything else means none.
    """
    _check_alias(alias)
    ensure_control_dir()
    return ["ssh", "-o", f"ControlPath={CONTROL_PATH_TEMPLATE}", "-O", "check", alias]


def master_alive(alias: str, runner=None, timeout: float = DEFAULT_SSH_TIMEOUT) -> bool:
    """Run `ssh -O check` and report whether a master is up.

    `runner` is the same `(argv, input_bytes, timeout) -> (rc, out, err)`
    seam the backends use, so tests never touch a real socket. The check
    talks only to the local socket and normally answers in milliseconds; the
    timeout is the shared `ssh_timeout` anyway, so nothing in relay waits
    less than anything else.
    """
    run = runner or _default_runner
    try:
        rc, _out, _err = run(master_check_argv(alias), None, timeout)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return rc == 0


_PERMISSION_DENIED = re.compile(
    r"permission denied|authentication failed|too many authentication failures",
    re.IGNORECASE,
)


def classify_failure(returncode: int, stderr: str, master_alive: bool) -> str:
    """Decide whether an ssh failure needs a human or just a retry.

    Two kinds, because the daemon reacts differently to each:

      * `auth_required` -- there is no live master and ssh could not make a
        new connection without a Duo prompt (exit 255 with the master gone),
        or the server refused the credentials outright. Retrying will not
        help; the user has to run `relay connect`. The daemon stops polling
        the cluster and instead checks the socket once a minute.

      * `unreachable` -- anything else at the transport level: network down,
        host not resolving, connection reset, timeout. Ordinary exponential
        backoff applies; it usually fixes itself.

    Exit 255 is ssh's own "I failed" code, as opposed to the remote command's
    exit status. A 255 *with* a live master is a network problem between the
    master and the server, not an auth problem.
    """
    if _PERMISSION_DENIED.search(stderr or ""):
        return AUTH_REQUIRED
    if returncode == 255 and not master_alive:
        return AUTH_REQUIRED
    return UNREACHABLE


def _check_alias(alias: str) -> None:
    """Refuse anything that looks like `user@host` rather than an alias."""
    if not alias or "@" in alias or any(c.isspace() for c in alias):
        raise ValueError(
            f"ssh_alias must be a bare alias from ~/.ssh/config, got {alias!r}. "
            "relay never builds user@host itself."
        )


def _default_runner(argv: list[str], input_bytes: bytes | None, timeout: float):
    """subprocess.run in the shape the backends' `runner` seam expects.

    With no input to send, stdin is /dev/null rather than inherited. ssh
    otherwise forwards relay's own stdin to the remote command, and a
    command that reads it (or a shell startup file that does) would block
    until the timeout for no reason anyone could see.
    """
    proc = subprocess.run(
        argv,
        input=input_bytes,
        stdin=None if input_bytes is not None else subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr
