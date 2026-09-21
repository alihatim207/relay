"""Tests for `relay.ssh`: the one module that builds every ssh command line.

Nothing here opens a connection. Every function under test is either pure
(it returns an argv) or takes the same `runner` seam the backends use, so the
whole file runs in CI with no cluster, no network and no control socket.

Two things are worth testing this carefully in a module this small.

**The options are load-bearing, individually.** `BatchMode=yes` is the
difference between a daemon that reports "run `relay connect`" and a daemon
that hangs forever on a Duo prompt nobody can see. `ControlPath` under
`~/.relay/` is the difference between a working socket and one whose path
exceeds the ~104-byte Unix limit. A test that only checked "some options are
present" would not notice either going missing.

**The alias must stay bare.** relay stores the name of a `Host` block and
hands it to ssh untouched, so the user's own config supplies the username,
hostname, key and jump host. A `user@host` sneaking in here would defeat the
whole arrangement, so the builders refuse one outright.
"""

from __future__ import annotations

import subprocess

import pytest

from relay import ssh

ALIAS = "klone"


# --------------------------------------------------------------------------
# build_ssh_argv
# --------------------------------------------------------------------------

# The exact option list, in order, that every non-interactive relay ssh call
# carries. Spelled out here rather than imported from the module, so that a
# change to the module has to be a deliberate change to this list too.
EXPECTED_OPTIONS = [
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.relay/cm-%C",
    "-o", "ControlPersist=8h",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-o", "BatchMode=yes",
]


def test_build_ssh_argv_has_the_six_options_in_order_then_the_alias():
    argv = ssh.build_ssh_argv(ALIAS, ["squeue", "--me"])

    assert argv[0] == "ssh"
    assert argv[1:13] == EXPECTED_OPTIONS
    # BatchMode is last of the options and the alias comes straight after it,
    # bare, with the remote command trailing.
    assert argv[11:13] == ["-o", "BatchMode=yes"]
    assert argv[13] == ALIAS
    assert argv[14:] == ["squeue", "--me"]


def test_build_ssh_argv_passes_remote_words_through_untouched():
    """Quoting is the caller's job; this function must not second-guess it."""
    words = ["sh", "-c", "'tail -c +1 /mmfs1/a b/events.jsonl'"]
    assert ssh.build_ssh_argv(ALIAS, words)[-3:] == words


def test_build_ssh_argv_never_contains_a_user_at_host():
    for word in ssh.build_ssh_argv(ALIAS, ["true"]):
        assert "@" not in word


def test_control_path_is_short_and_under_the_relay_directory():
    """A Unix socket path is capped near 104 bytes, so this cannot live in XDG."""
    assert ssh.CONTROL_PATH_TEMPLATE == "~/.relay/cm-%C"
    assert len(ssh.CONTROL_PATH_TEMPLATE) < 40


# --------------------------------------------------------------------------
# connect_argv
# --------------------------------------------------------------------------


def test_connect_argv_omits_batchmode_and_adds_the_master_flags():
    """`relay connect` is the one call allowed to prompt for Duo."""
    argv = ssh.connect_argv(ALIAS)

    assert argv[0] == "ssh"
    assert "BatchMode=yes" not in argv, "connect must be allowed to prompt"
    # -M be the master, -N run no command, -f background once authenticated.
    assert argv[-4:] == ["-M", "-N", "-f", ALIAS]


def test_connect_argv_shares_the_control_path_with_build_ssh_argv():
    """Otherwise `relay connect` would open a master nothing else can find."""
    connect = ssh.connect_argv(ALIAS)
    run = ssh.build_ssh_argv(ALIAS, ["true"])
    path_option = f"ControlPath={ssh.CONTROL_PATH_TEMPLATE}"
    assert path_option in connect
    assert path_option in run
    assert f"ControlPersist={ssh.CONTROL_PERSIST}" in connect


# --------------------------------------------------------------------------
# master_check_argv / master_alive
# --------------------------------------------------------------------------


def test_master_check_argv_only_asks_the_local_socket():
    argv = ssh.master_check_argv(ALIAS)
    assert argv == [
        "ssh",
        "-o",
        f"ControlPath={ssh.CONTROL_PATH_TEMPLATE}",
        "-O",
        "check",
        ALIAS,
    ]
    # No BatchMode needed: -O check never touches the network, so it can never
    # reach the point of prompting.
    assert "BatchMode=yes" not in argv


def test_master_alive_is_true_on_exit_zero():
    calls = []

    def runner(argv, input_bytes, timeout):
        calls.append(argv)
        return 0, b"Master running (pid=1234)\n", b""

    assert ssh.master_alive(ALIAS, runner=runner) is True
    assert calls == [ssh.master_check_argv(ALIAS)]


def test_master_alive_is_false_on_any_other_exit():
    def runner(argv, input_bytes, timeout):
        return 255, b"", b"Control socket connect(...): No such file or directory\n"

    assert ssh.master_alive(ALIAS, runner=runner) is False


@pytest.mark.parametrize(
    "boom",
    [
        FileNotFoundError("ssh"),
        subprocess.TimeoutExpired(cmd="ssh", timeout=10),
        OSError("cannot fork"),
    ],
)
def test_master_alive_is_false_when_the_check_itself_cannot_run(boom):
    """No answer is not a live master. This is called on a failure path; it
    must never raise on top of the failure it is trying to explain."""

    def runner(argv, input_bytes, timeout):
        raise boom

    assert ssh.master_alive(ALIAS, runner=runner) is False


# --------------------------------------------------------------------------
# classify_failure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("returncode", "stderr", "alive", "expected"),
    [
        # A refusal is a refusal whatever ssh exited with: retrying cannot fix
        # credentials, so this is auth_required even at a non-255 code.
        (255, "klone: Permission denied (publickey,keyboard-interactive).", False, ssh.AUTH_REQUIRED),
        (1, "Permission denied (publickey).", True, ssh.AUTH_REQUIRED),
        (255, "ssh: Too many authentication failures", True, ssh.AUTH_REQUIRED),
        (255, "Authentication failed.", True, ssh.AUTH_REQUIRED),
        # 255 with no master: BatchMode stopped ssh from prompting for Duo.
        # Only a human can fix that, so do not back off -- say so.
        (255, "ssh: connect to host klone port 22: Connection refused", False, ssh.AUTH_REQUIRED),
        (255, "", False, ssh.AUTH_REQUIRED),
        # 255 *with* a live master is a network problem between the master and
        # the server, not an auth problem. Ordinary backoff applies.
        (255, "ssh: connect to host klone port 22: Connection refused", True, ssh.UNREACHABLE),
        (255, "client_loop: send disconnect: Broken pipe", True, ssh.UNREACHABLE),
        # Anything that is not ssh's own 255 is the remote command's status.
        (1, "", False, ssh.UNREACHABLE),
        (44, "", False, ssh.UNREACHABLE),
    ],
)
def test_classify_failure_table(returncode, stderr, alive, expected):
    assert ssh.classify_failure(returncode, stderr, alive) == expected


def test_classify_failure_is_case_insensitive_about_permission_denied():
    assert (
        ssh.classify_failure(1, "PERMISSION DENIED (publickey)", True)
        == ssh.AUTH_REQUIRED
    )


def test_classify_failure_tolerates_empty_stderr():
    assert ssh.classify_failure(1, "", True) == ssh.UNREACHABLE


# --------------------------------------------------------------------------
# Alias validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "ahatim@klone.hyak.uw.edu",  # the thing relay exists to avoid
        "klone hyak",  # a space would split into two ssh arguments
        "klone\t2",
        "klone\n",
        "",
        None,
    ],
)
@pytest.mark.parametrize(
    "builder",
    [ssh.build_ssh_argv, ssh.connect_argv, ssh.master_check_argv],
)
def test_every_builder_refuses_a_bad_alias(builder, bad):
    with pytest.raises(ValueError) as excinfo:
        if builder is ssh.build_ssh_argv:
            builder(bad, ["true"])
        else:
            builder(bad)
    assert "~/.ssh/config" in str(excinfo.value)


def test_a_plain_alias_is_accepted():
    for good in ("klone", "hyak-login", "cluster_2"):
        assert ssh.build_ssh_argv(good, ["true"])[-2] == good
