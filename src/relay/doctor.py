"""`relay doctor`: run every preflight check and report all of them.

The rule that shapes this module is in CLAUDE.md: **run every check, never
fail fast, report all results.** A doctor that stops at the first problem makes
you fix one thing, re-run, discover the next thing, and re-run again. On a
cluster where each new SSH connection costs a Duo push, that loop is genuinely
expensive. So every check runs, every check produces a `CheckResult`, and a
check that cannot run because something before it failed returns the status
`"skip"` with a sentence saying why.

The second rule is that warnings are not failures. `worst_status()` returns 0
when the worst thing we found was a warning and 1 when anything errored, so
`relay doctor` can sit in a CI job as a smoke test without going red because
the daemon happens not to be running on the build machine.

Three things are worth knowing before reading the checks themselves.

**Every remote command goes through the user's own `ssh` binary with the
configured alias.** relay never builds a `user@host` string -- see the docstring
in `config.py` for why. Everything funnels through the single `_ssh()` helper,
which means there is exactly one place that knows how to shell out, exactly one
place to monkeypatch in the tests, and no way for a check to accidentally
invent its own connection. `_ssh()` does not build its own argv either: it asks
`relay.ssh.build_ssh_argv`, the one function in relay that knows which
ControlMaster options every invocation carries. doctor must test the same
connection the daemon actually uses, so a doctor that built its own slightly
different argv would be testing the wrong thing.

The one exception is `ssh -O check`, which takes no remote command and talks
only to the local control socket. It gets its own thin wrapper,
`_ssh_master_check()`, over the same subprocess helper -- a second seam of the
same shape, so tests script both the same way.

**Transport failure is a normal outcome, not an exception.** `_ssh()` never
raises. A refused connection, a timeout, a missing `ssh` binary -- all of them
come back as an `SshResult` with `ok=False` and a message. The check that asked
turns that into an `"error"` result and the *next* check still gets to run.

**Checks are timed, but latency never names its own cause.** `duration_ms` is
on every result, and for the SSH round-trip check the number is worth printing
-- it is not worth drawing a conclusion from on its own. The first real Klone
run measured 4.8 seconds through a *live* ControlMaster, because the login node
runs `conda init` out of ~/.bashrc before it will run anything at all. doctor
read that number by itself and announced that ssh had "opened a fresh
connection instead of reusing relay's ControlMaster socket", which was flatly
untrue, and then its own 15 second timeout failed the check outright and
skipped all ten remote checks behind it.

So the ssh check reads `ctx.master_alive` -- the answer the master check, which
runs first, already has -- and only mentions a fresh connection when there is
genuinely no master to reuse. Latency is a symptom; the master check is the
cause. And the timeout it runs under is generous enough that a slow login node
produces a *warning* you can read rather than an error that hides everything
else.
"""

from __future__ import annotations

import errno
import fcntl
import re
import shlex
import sqlite3
import stat
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from relay import config as cfg
from relay import ssh as ssh_mod
from relay import store as store_mod

# The scontrol parsers are pure functions with no ssh in them, and the rule is
# one parser per output format: if doctor grew its own it would drift from the
# one the backend submits with, and the two would disagree about the partition
# relay is about to use. `resolve_requeue_on_term` is the same idea -- doctor
# must report exactly what submit will decide, so it asks submit's own code.
from relay.backends.slurm import (
    parse_scontrol_kv,
    resolve_requeue_on_term,
    split_scontrol_entries,
)
from relay.store import Store

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

# doctor has no timeout constant of its own, on purpose.
#
# There used to be two: a 15 second cap on every remote command and a 2 second
# "this is slow" threshold. Both were wrong on a real login node. Klone answers
# a trivial command in 4 to 5 seconds *through a live master* -- the time goes
# on starting a shell, not on the network -- so 2 seconds warned about
# something that was working and 15 seconds failed a check that would have
# succeeded.
#
# The number now comes from `conf.ssh_timeout` (see `_ssh_timeout` below),
# which defaults to `ssh.DEFAULT_SSH_TIMEOUT`. One value for the whole of
# relay: the daemon, the backend and doctor all wait the same length of time,
# so no check here can be accidentally tighter than the thing it is checking
# on behalf of. That one value is also the threshold above which a *successful*
# round trip is reported as a warning -- "slower than relay is willing to wait"
# is the honest definition of too slow, and it needs no second constant.

# Free space on remote_root below which we warn. Checkpoints are large and a
# job that dies at 3am because the filesystem filled up is a bad way to find out.
LOW_DISK_GB = 10.0

# The daemon records a timestamp after each completed cycle. Its slowest normal
# interval is 60s (see the adaptive interval in daemon.py), so five minutes of
# silence means it is wedged, not merely idle. Same threshold the daemon uses
# for flagging a run stale, for the same reason.
STALE_SYNC_SECONDS = 300

# Status values. Constants rather than bare strings so a typo is an
# AttributeError here instead of a check that silently never matches in the CLI.
OK = "ok"
WARN = "warn"
ERROR = "error"
SKIP = "skip"


# --------------------------------------------------------------------------
# The result type
# --------------------------------------------------------------------------


@dataclass
class CheckResult:
    """One line of `relay doctor` output.

    `name` is a short slug (`ssh`, `remote_root`) so the CLI can align a column
    and `--json` consumers can key off something stable.

    `detail` is a plain sentence. For `warn` and `error` it must say what
    happened *and what to do next* -- a diagnostic that tells you something is
    broken without telling you which lever to pull is only half a diagnostic.

    `duration_ms` is None for checks that did no work (anything skipped).
    """

    name: str
    status: str
    detail: str
    duration_ms: float | None = None

    def as_dict(self) -> dict:
        """Plain dict for `relay doctor --json`."""
        return asdict(self)


def worst_status(results: list[CheckResult]) -> int:
    """Process exit code for a list of results: 1 if anything errored, else 0.

    Warnings deliberately do not fail. CLAUDE.md wants `relay doctor` usable as
    a CI smoke test, and in CI there is no cluster, no daemon and no database --
    all of which are warnings by design, not errors.
    """
    return 1 if any(r.status == ERROR for r in results) else 0


# --------------------------------------------------------------------------
# The one place that runs ssh
# --------------------------------------------------------------------------


@dataclass
class SshResult:
    """What came back from one `ssh` invocation.

    `ok` means *the ssh process ran and exited*. It says nothing about whether
    the remote command succeeded -- that is `returncode`. The distinction
    matters: `command -v sbatch` exiting 1 is a real answer ("not installed"),
    while ssh itself failing to connect is a transport failure, and those two
    deserve different sentences.
    """

    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: float
    error: str | None = None  # set only when ssh itself could not run

    def succeeded(self) -> bool:
        """True when ssh ran *and* the remote command exited 0."""
        return self.ok and self.returncode == 0

    def message(self) -> str:
        """The most useful one-line explanation of a failure we can produce."""
        if self.error:
            return self.error
        first = _first_line(self.stderr) or _first_line(self.stdout)
        if first:
            return first
        return f"exit code {self.returncode}"


def _ssh(
    alias: str, args: list[str], timeout: float = ssh_mod.DEFAULT_SSH_TIMEOUT
) -> SshResult:
    """Run one remote command over relay's ssh options, and never raise.

    `alias` is a Host block name from the user's `~/.ssh/config`, handed to the
    ssh binary untouched so that their own configuration supplies the username,
    hostname, key and jump host.

    `args` are the words of the remote command. They are joined with spaces
    into one command line -- which is why anything containing a path gets
    `shlex.quote()`d by the caller, and why `$(whoami)` in a command string
    expands on the cluster rather than here (`subprocess.run` with a list does
    no local shell expansion at all).

    That command line is then wrapped by `ssh.remote_shell_argv`, so every
    probe doctor builds runs under `bash --noprofile --norc -c`. This is the
    one place that wrapping happens, so no check can forget it. The point is
    the check the login node failed: a `conda init` in ~/.bashrc costs seconds
    on every single round trip, and none of these probes -- `true`,
    `command -v sbatch`, `df` -- needs anything a startup file sets up. It does
    not touch the training job, which runs under Slurm in the user's own
    environment.

    The wrapped command is `shlex.quote`d because ssh joins its trailing
    arguments with spaces and hands the string to the *remote* login shell:
    without the quoting, that shell would parse the probe's own `;` and `|`
    before bash ever saw them.

    The argv comes from `relay.ssh.build_ssh_argv`, so doctor exercises exactly
    the connection the daemon and the backend use: relay's own ControlMaster,
    relay's ControlPath, `-T`, and `BatchMode=yes`. BatchMode is what keeps this
    check honest -- without it a doctor run against a host with no live master
    sits forever on a Duo prompt nobody will see, and a health check that hangs
    is worse than one that fails.

    Returns an `SshResult`; transport failures are values, not exceptions,
    because the caller's job is to report them and let later checks run.
    """
    try:
        remote_argv = ssh_mod.remote_shell_argv(shlex.quote(" ".join(args)))
        argv = ssh_mod.build_ssh_argv(alias, remote_argv)
    except ValueError as exc:
        # Raised for an empty alias or anything shaped like `user@host`.
        return SshResult(
            ok=False, returncode=None, stdout="", stderr="", duration_ms=0.0, error=str(exc)
        )
    return _run_ssh(argv, timeout)


def _ssh_master_check(
    alias: str, timeout: float = ssh_mod.DEFAULT_SSH_TIMEOUT
) -> SshResult:
    """Run `ssh -O check`: "is relay's master alive for this alias?"

    A second seam of the same shape as `_ssh`, because the argv is a different
    shape: `-O check` takes no remote command (so there is nothing to wrap in a
    clean shell) and speaks only to the local control socket, so it costs
    nothing and can never trigger Duo. That is why this check still runs when
    the round-trip check failed -- it is usually the explanation.

    The timeout is the shared `ssh_timeout` even though a local socket answers
    in milliseconds. Nothing in relay waits less than that one value; a second,
    tighter number here would be one more thing that can be wrong on its own.
    """
    try:
        argv = ssh_mod.master_check_argv(alias)
    except ValueError as exc:
        return SshResult(
            ok=False, returncode=None, stdout="", stderr="", duration_ms=0.0, error=str(exc)
        )
    return _run_ssh(argv, timeout)


def _run_ssh(argv: list[str], timeout: float) -> SshResult:
    """The one place doctor shells out. Times the call and swallows every error."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SshResult(
            ok=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_ms=_ms_since(start),
            # Deliberately says only what happened, not why. This helper has no
            # idea whether a master is alive, and the sentence it used to add
            # ("run `relay connect`") was pure guesswork that the calling check
            # then repeated to a user whose master was up the whole time. The
            # ssh check knows the answer and adds the cause itself.
            error=f"ssh did not answer within {timeout:.0f} seconds",
        )
    except FileNotFoundError:
        return SshResult(
            ok=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_ms=_ms_since(start),
            error="no `ssh` binary was found on your PATH",
        )
    except OSError as exc:
        return SshResult(
            ok=False,
            returncode=None,
            stdout="",
            stderr="",
            duration_ms=_ms_since(start),
            error=f"could not start ssh: {exc}",
        )

    return SshResult(
        ok=True,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        duration_ms=_ms_since(start),
    )


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _last_line(text: str) -> str:
    """The last non-blank line of some output.

    ssh prints its own diagnostics *after* the ones from the connection it was
    setting up, so when ssh itself fails (exit 255) the sentence that says why
    is usually the last line, not the first. `_first_line` is still right for a
    remote command's own error output, which is why both exist.
    """
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _skipped(name: str, reason: str) -> CheckResult:
    """A check that did not run, and the sentence explaining why."""
    return CheckResult(name=name, status=SKIP, detail=f"Skipped: {reason}.")


def _ssh_timeout(conf: cfg.Config) -> float:
    """The one number every remote call in doctor waits, in seconds.

    `getattr` rather than `conf.ssh_timeout` so that a Config built by older
    code -- or by a test that constructs one by hand -- still works and simply
    gets the shared default. A nonsensical value (zero, negative, not a number)
    falls back too: a health check may not be the thing that crashes.
    """
    raw = getattr(conf, "ssh_timeout", ssh_mod.DEFAULT_SSH_TIMEOUT)
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return ssh_mod.DEFAULT_SSH_TIMEOUT
    return seconds if seconds > 0 else ssh_mod.DEFAULT_SSH_TIMEOUT


def _human_duration(ms: float) -> str:
    """"30 ms" or "4.8s" -- seconds once milliseconds stop being readable."""
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000.0:.1f}s"


@dataclass
class _Context:
    """The few facts one check learns that a later check needs.

    Checks are otherwise independent, which is the point: each one asks the
    cluster its own question and reports its own sentence. But three of them --
    the requeue policy, the preemption window and the time limit -- are all
    readings of the *same* `scontrol show partition` output, and asking for it
    three times would mean three SSH round trips for one answer. So the
    partition check parses it once and leaves the fields here.

    A small mutable bag rather than a return-value chain, because every check
    still has to fit the `(conf, ctx) -> CheckResult` shape that `run_checks`
    calls them all through. The sophisticated alternative is a dependency graph
    between checks; with exactly two dependencies it would be more machinery
    than it saves.
    """

    # Did `ssh -O check` find a live master? Read by the ssh round-trip check,
    # which otherwise cannot tell "the cluster is down" from "nobody has run
    # `relay connect` yet" -- nor, and this is the one that bit us, "the login
    # node takes five seconds to start a shell" from "ssh re-authenticated".
    # Every sentence that check writes about *why* it was slow or why it failed
    # comes from this flag, never from the clock.
    master_alive: bool = False

    # Parsed `scontrol show partition <p>`, or None when that check did not get
    # an answer. Keys are scontrol's own: PreemptMode, GraceTime, MaxTime, ...
    partition: dict[str, str] | None = None


def _parse_iso(ts: str) -> datetime | None:
    """Parse one of relay's ISO-8601 UTC timestamps, or return None.

    `fromisoformat` understands the trailing `Z` from Python 3.11 on, which is
    also relay's minimum version.
    """
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_seconds(ts: str) -> float | None:
    parsed = _parse_iso(ts)
    if parsed is None:
        return None
    return (datetime.now(timezone.utc) - parsed).total_seconds()


def slurm_time_to_seconds(value: str | None) -> int | None:
    """Turn any of Slurm's six `--time` spellings into seconds. Pure function.

    Slurm accepts, and `scontrol` prints, all of these:

        MM            "30"          thirty minutes, not thirty seconds
        MM:SS         "30:00"
        HH:MM:SS      "04:00:00"
        D-HH          "2-00"
        D-HH:MM       "2-12:30"
        D-HH:MM:SS    "2-12:30:15"

    The hyphen is what disambiguates the colon forms: *with* days, the first
    field after it is hours; *without*, a bare two fields mean minutes and
    seconds. That is the whole trick, and it is the one thing an eyeballed
    comparison of two time limits gets wrong.

    `UNLIMITED` (and `INFINITE`, which older Slurms print) return None, meaning
    "no ceiling". So does anything unparseable: a caller comparing against None
    should report "cannot tell", never "fits".
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in ("UNLIMITED", "INFINITE", "NONE", "N/A"):
        return None

    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        if not day_part.isdigit():
            return None
        days = int(day_part)
        fields = text.split(":")
        # With days present the fields are hours, then minutes, then seconds.
        hours, minutes, seconds = (fields + ["0", "0"])[:3]
    else:
        fields = text.split(":")
        if len(fields) == 1:
            hours, minutes, seconds = "0", fields[0], "0"  # bare MM
        elif len(fields) == 2:
            hours, minutes, seconds = "0", fields[0], fields[1]  # MM:SS
        elif len(fields) == 3:
            hours, minutes, seconds = fields
        else:
            return None

    try:
        total = days * 86400 + int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    except ValueError:
        return None
    return total


def parse_kill_wait(text: str) -> int | None:
    """Pull `KillWait` out of `scontrol show config` output, in seconds.

    That command prints aligned `Key = Value` pairs rather than the packed
    `Key=Value` tokens `scontrol show partition` uses, and the value carries a
    unit: `KillWait                = 10 sec`. The regex accepts both spacings
    and stops at the number, so the unit is ignored -- Slurm only ever reports
    this one in seconds.

    Returns None when the key is absent, which the caller reports as "unknown"
    rather than guessing a number the sidecar would then budget against.
    """
    match = re.search(r"^\s*KillWait\s*=\s*(\d+)", text or "", re.IGNORECASE | re.MULTILINE)
    return int(match.group(1)) if match else None


def _seconds_field(raw: str | None) -> int | None:
    """Read a scontrol field that holds seconds, e.g. `GraceTime=0`.

    Partition GraceTime is a plain integer of seconds, but accepting a clock
    form too costs one line and saves a surprise on a Slurm that prints one.
    """
    if raw is None:
        return None
    text = raw.strip()
    if text.isdigit():
        return int(text)
    return slurm_time_to_seconds(text)


def _humanise_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f} days ago"


# --------------------------------------------------------------------------
# Check 1: the config file
# --------------------------------------------------------------------------


def _check_config(path: Path | None) -> tuple[cfg.Config | None, CheckResult]:
    """Load the config. Returns (config or None, result).

    Every other check depends on this one, so it is the only check that hands
    something back besides a result.
    """
    try:
        conf = cfg.load(path)
    except cfg.ConfigNotFound as exc:
        # The message from config.py already says "run `relay init`", so pass
        # it through rather than writing a second, competing sentence.
        return None, CheckResult("config", ERROR, str(exc))
    except cfg.ConfigError as exc:
        return None, CheckResult("config", ERROR, str(exc))

    where = path if path is not None else cfg.config_path()
    if conf.backend == "local":
        detail = (
            f"Config at {where} parses. Backend is local, so jobs run as "
            f"subprocesses under {conf.local_root}."
        )
    else:
        detail = (
            f"Config at {where} parses. Backend is slurm via SSH alias "
            f"{conf.ssh_alias!r}, remote root {conf.remote_root}."
        )
    return conf, CheckResult("config", OK, detail)


# --------------------------------------------------------------------------
# Check 2: the directory relay's control socket lives in
# --------------------------------------------------------------------------


def _check_control_dir(conf: cfg.Config, ctx: _Context) -> CheckResult:
    """Does `~/.relay/` exist, as a directory, with mode 0700?

    This is the check the first real Klone run needed and did not have. ssh
    does not create the parent directory of its ControlPath. If `~/.relay/` is
    missing, ssh authenticates -- Duo push, phone tap, the lot -- and only
    *then* fails with "unix_listener: cannot bind to path ...: No such file or
    directory". Every relay command that touches the cluster failed that way,
    after the most annoying possible delay.

    `ssh.ensure_control_dir()` now fixes that at the source: every argv builder
    calls it, so any ssh relay runs has the directory in place first. This check
    exists anyway for two reasons. It runs *before* the master check, which is
    the first thing in doctor that would build an ssh argv, so it reports the
    state the user actually had rather than the state the repair left behind.
    And it is the only place that tells the user the directory was missing or
    world-readable at all, which is the difference between "relay fixed
    something for you" and silence.

    Mode matters as much as existence: anyone who can connect to relay's
    control socket can run commands on the cluster as the user, with no
    authentication of their own, because the socket *is* an authenticated
    session. That is a warning rather than an ok even though relay corrects it,
    because a permission that was wrong once was set by something, and the user
    should know.
    """
    path = ssh_mod.relay_dir()

    # Look before touching. `ensure_control_dir()` is deliberately idempotent
    # and silent, so after it runs there is no way to tell what it had to do.
    existed = path.exists() or path.is_symlink()
    was_dir = path.is_dir()
    mode_before: int | None = None
    if existed:
        try:
            mode_before = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            mode_before = None

    try:
        ssh_mod.ensure_control_dir()
    except NotADirectoryError as exc:
        # ensure_control_dir()'s own sentence already says what to do.
        return CheckResult("control_dir", ERROR, str(exc))
    except OSError as exc:
        if existed and not was_dir:
            # `os.makedirs` reports a plain file in the way as FileExistsError
            # rather than NotADirectoryError, so say the useful thing here too.
            return CheckResult(
                "control_dir",
                ERROR,
                f"{path} exists but is not a directory. relay keeps its SSH "
                f"control socket there; move the file out of the way and "
                f"re-run `relay doctor`.",
            )
        reason = exc.strerror or str(exc)
        return CheckResult(
            "control_dir",
            ERROR,
            f"cannot create {path}: {reason}. relay's SSH control socket lives "
            f"there, and ssh fails after you have already answered Duo if it is "
            f"missing.",
        )

    if not existed:
        return CheckResult(
            "control_dir",
            OK,
            f"created {path} with mode 0700 (ssh does not create it, and fails "
            f"after Duo if it is missing).",
        )
    if mode_before == 0o700:
        return CheckResult("control_dir", OK, f"{path} exists with mode 0700.")
    shown = "unknown" if mode_before is None else f"{mode_before:04o}"
    return CheckResult(
        "control_dir",
        WARN,
        f"{path} had mode {shown}; corrected to 0700 (a control socket other "
        f"users can reach lets them run commands as you).",
    )


# --------------------------------------------------------------------------
# Check 2: relay's own SSH master
# --------------------------------------------------------------------------


def _check_master(conf: cfg.Config, ctx: _Context) -> CheckResult:
    """Is relay's multiplexing socket alive for this alias?

    This runs *before* the round-trip check, and deliberately so: a dead master
    is the cause of nearly every other remote failure on a Duo cluster, and
    knowing the answer first lets the ssh check say "run `relay connect`"
    instead of leaving the user to guess.

    It is a warning, never an error. relay without a master still works, it
    just cannot open a new connection non-interactively -- `BatchMode=yes`
    means every later check fails fast rather than hanging on a Duo prompt.
    """
    alias = conf.ssh_alias
    if not alias:
        return _skipped("master", "no ssh_alias is configured")

    result = _ssh_master_check(alias, timeout=_ssh_timeout(conf))

    if not result.ok:
        return CheckResult(
            "master",
            WARN,
            f"Could not ask ssh about a master connection for {alias}: "
            f"{result.message()}. Run `relay connect` to start one.",
            result.duration_ms,
        )
    if result.returncode == 0:
        ctx.master_alive = True
        return CheckResult(
            "master",
            OK,
            f"relay's SSH master is alive for {alias}, so every relay command "
            f"rides one already-authenticated connection instead of starting a "
            f"fresh login (a Duo push, on Hyak) each time.",
            result.duration_ms,
        )
    return CheckResult(
        "master",
        WARN,
        f"No live SSH master for {alias}. Run `relay connect`. Until then every "
        f"ssh relay runs carries BatchMode=yes and fails immediately rather than "
        f"waiting for a two-factor prompt nobody is watching.",
        result.duration_ms,
    )


# --------------------------------------------------------------------------
# Check 3: SSH reachability, and how long it took
# --------------------------------------------------------------------------


def _check_ssh(conf: cfg.Config, ctx: _Context) -> CheckResult:
    """Can relay reach the login node, and how long did it take?

    Which sentence you get depends on *two* facts, never one: whether the round
    trip worked, and whether `ctx.master_alive` -- filled in by the master
    check, which runs first -- says relay has a live multiplexed connection.
    The clock alone cannot tell these cases apart, and pretending it can is what
    produced a confident, false diagnosis on the first real Klone run.

        fast (either)          -> ok
        slow, master alive     -> warn: the login node is slow (shell startup)
        slow, no master        -> warn: ssh negotiated a fresh connection
        failed, master alive   -> error: it did not answer; raise ssh_timeout
        failed, no master      -> error: run `relay connect`

    Only the two "no master" rows may blame the connection, because only there
    is there actually no connection to reuse. "Slow" means slower than
    `ssh_timeout`, which is also how long relay is willing to wait anywhere
    else: below that, a slow login node is a fact about the cluster rather than
    a problem with relay, and doctor says nothing about it.
    """
    alias = conf.ssh_alias
    if not alias:
        # config.load() already refuses a slurm backend with no alias, so this
        # is belt-and-braces for a hand-built Config object.
        return CheckResult(
            "ssh",
            ERROR,
            f"No ssh_alias is set in {cfg.config_path()}. Add the name of a Host "
            f"block from your ~/.ssh/config, for example 'klone'.",
        )

    timeout = _ssh_timeout(conf)

    # Twice the timeout, and this is the one call in doctor that gets more than
    # the shared value. `timeout` is *also* the threshold above which a
    # successful round trip warns, so if the subprocess were cut off at exactly
    # that number the warn branch below could never fire: every round trip slow
    # enough to be worth warning about would come back as a timeout *error*
    # instead, and the user would be told the host is unreachable when it is
    # merely slow. The subprocess timeout has to exceed the warn threshold for
    # the warning to exist at all.
    round_trip_timeout = 2 * timeout

    # `true` is the cheapest possible remote command: it proves the whole path
    # (ssh config, network, authentication, remote shell) without producing
    # output or side effects.
    #
    # This still runs when there is no live master. BatchMode makes it fail
    # fast instead of hanging, and a fast failure with the right sentence
    # attached is more useful than a skipped check.
    result = _ssh(alias, ["true"], timeout=round_trip_timeout)

    # The master check ran first, so we can name the likeliest cause instead of
    # leaving the user to guess which of two problems they have. With a master
    # alive there is nothing to say here: `relay connect` has already been run.
    hint = (
        ""
        if ctx.master_alive
        else " There is no live SSH master either, which is the likeliest cause: "
        "run `relay connect` to start one, then re-run `relay doctor`."
    )

    if not result.ok:
        # ssh itself could not run, could not connect, or ran out of time.
        if ctx.master_alive:
            # Not an authentication problem: relay is holding an authenticated
            # connection to this host right now. Something on the far end is
            # not answering, and the only levers the user has are patience (a
            # bigger ssh_timeout) and a look at the login node itself.
            return CheckResult(
                "ssh",
                ERROR,
                f"Could not reach {alias}: {result.message()}. relay's SSH master "
                f"is alive, so this is not an authentication problem -- the login "
                f"node did not answer within ssh_timeout ({timeout:.0f}s). Raise "
                f"ssh_timeout in {cfg.config_path()} if the node is merely slow, "
                f"or check that it is healthy with `ssh {alias} true` by hand.",
                result.duration_ms,
            )
        return CheckResult(
            "ssh",
            ERROR,
            f"Could not reach {alias}: {result.message()}. Check that `ssh {alias}` "
            f"works from a terminal, and that the Host block exists in your "
            f"~/.ssh/config.{hint}",
            result.duration_ms,
        )
    if result.returncode != 0:
        # 255 is ssh's own "I failed" code rather than the remote command's
        # exit status, and ssh prints its reason last -- after whatever the
        # connection attempt itself said. Quoting that line is the difference
        # between "exited 255" and "unix_listener: cannot bind to path
        # /Users/you/.relay/cm-abc: No such file or directory", which names the
        # actual problem.
        detail = result.message()
        if result.returncode == 255:
            detail = _last_line(result.stderr) or detail
        return CheckResult(
            "ssh",
            ERROR,
            f"`ssh {alias} true` exited {result.returncode}: {detail}. "
            f"Fix the connection by hand, then re-run `relay doctor`.{hint}",
            result.duration_ms,
        )

    seconds = result.duration_ms / 1000.0
    if seconds > timeout:
        if ctx.master_alive:
            # The case the old code got wrong. Nothing was re-authenticated:
            # relay's master carried this command, and the time went somewhere
            # on the far end -- on Klone, into a `conda init` that every shell
            # runs before it will run anything. Say that, suggest the two real
            # levers, and make no claim about the connection.
            return CheckResult(
                "ssh",
                WARN,
                f"Round trip to {alias} took {seconds:.1f}s through relay's live "
                f"SSH master. The login node is slow to respond; remote shell "
                f"startup is the usual reason (for example `conda init` in "
                f"~/.bashrc). Trim it, or raise ssh_timeout in "
                f"{cfg.config_path()}.",
                result.duration_ms,
            )
        # No master, so ssh really did negotiate a connection of its own, and
        # here the seconds are a fair thing to point at.
        return CheckResult(
            "ssh",
            WARN,
            f"Reached {alias}, but the round trip took {seconds:.1f}s with no live "
            f"SSH master, so ssh negotiated a fresh connection of its own. On a "
            f"cluster with two-factor login relay polls far too often to "
            f"re-authenticate each time: run `relay connect` to start the master.",
            result.duration_ms,
        )
    return CheckResult(
        "ssh",
        OK,
        f"Reached {alias} in {_human_duration(result.duration_ms)}.",
        result.duration_ms,
    )


# --------------------------------------------------------------------------
# Check 4: the remote Python
# --------------------------------------------------------------------------


def _check_remote_python(conf: cfg.Config, ctx: _Context) -> CheckResult:
    alias = conf.ssh_alias or ""
    python = conf.remote_python

    # quote(): remote_python is often a full path into a conda environment and
    # those paths sometimes contain spaces. ssh hands the words to a remote
    # shell, so quoting has to happen here.
    result = _ssh(alias, [shlex.quote(python), "--version"], timeout=_ssh_timeout(conf))

    if not result.ok:
        return CheckResult(
            "remote_python",
            ERROR,
            f"Could not check {python} on {alias}: {result.message()}.",
            result.duration_ms,
        )
    if result.returncode != 0:
        return CheckResult(
            "remote_python",
            ERROR,
            f"{python} is not runnable on {alias}: {result.message()}. Set "
            f"remote_python in {cfg.config_path()} to a Python that exists there "
            f"(often a full path into a module or conda environment). relay's "
            f"sidecar needs only the standard library, so any Python 3.11+ will do.",
            result.duration_ms,
        )

    # Pythons before 3.4 printed --version to stderr; harmless to keep handling
    # both, and it costs one `or`.
    version = _first_line(result.stdout) or _first_line(result.stderr)
    return CheckResult(
        "remote_python",
        OK,
        f"{python} on {alias} is {version or 'runnable'}.",
        result.duration_ms,
    )


# --------------------------------------------------------------------------
# Check 5: sbatch on the remote PATH
# --------------------------------------------------------------------------


def _check_sbatch(conf: cfg.Config, ctx: _Context) -> CheckResult:
    alias = conf.ssh_alias or ""

    # `command -v` is the POSIX builtin for "where would the shell find this";
    # `which` is not standardised and behaves differently between systems.
    result = _ssh(alias, ["command", "-v", "sbatch"], timeout=_ssh_timeout(conf))

    if not result.ok:
        return CheckResult(
            "sbatch",
            ERROR,
            f"Could not look for sbatch on {alias}: {result.message()}.",
            result.duration_ms,
        )
    if result.returncode != 0:
        # Exit 1 from `command -v` is a real answer, not a transport failure.
        return CheckResult(
            "sbatch",
            ERROR,
            f"sbatch is not on the PATH of a non-interactive shell on {alias}. "
            f"relay submits over `ssh {alias} sbatch ...`, which does not read "
            f"your interactive shell profile. Make sure Slurm's binaries are on "
            f"the PATH for non-login shells, or that your ~/.bashrc puts them "
            f"there before its interactive-only section.",
            result.duration_ms,
        )
    return CheckResult(
        "sbatch",
        OK,
        f"sbatch is at {_first_line(result.stdout) or 'a location on PATH'} on {alias}.",
        result.duration_ms,
    )


# --------------------------------------------------------------------------
# Check 6: the Slurm account
# --------------------------------------------------------------------------


def _check_account(conf: cfg.Config, ctx: _Context) -> CheckResult:
    alias = conf.ssh_alias or ""
    account = conf.slurm.account

    if not account:
        return CheckResult(
            "slurm_account",
            WARN,
            f"No slurm.account is set in {cfg.config_path()}, so submitted jobs use "
            f"whatever default account the cluster picks. On most clusters that "
            f"works; on some it is a submission error. Set slurm.account to be sure.",
        )

    # `$(whoami)` is expanded by the *remote* shell, so we never need to know or
    # store a username locally -- same reason the config stores an alias rather
    # than user@host. One command, one round trip, which matters when every new
    # connection may cost a Duo push.
    #
    # -n drops the header, -P prints pipe-separated fields with no padding.
    result = _ssh(
        alias,
        ["sacctmgr -n -P show assoc user=$(whoami) format=account"],
        timeout=_ssh_timeout(conf),
    )

    if not result.ok or result.returncode != 0:
        # A warning, not an error: sacctmgr is not installed on every cluster,
        # and being unable to *verify* the account is not the same as the
        # account being wrong.
        return CheckResult(
            "slurm_account",
            WARN,
            f"Could not list your Slurm accounts on {alias}: {result.message()}. "
            f"relay cannot confirm that account {account!r} is valid; check it by "
            f"hand with `sacctmgr show assoc user=$USER`.",
            result.duration_ms,
        )

    accounts = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    if account in accounts:
        return CheckResult(
            "slurm_account",
            OK,
            f"Account {account!r} is one of your associations on {alias}.",
            result.duration_ms,
        )
    return CheckResult(
        "slurm_account",
        WARN,
        f"Account {account!r} is not in your Slurm associations on {alias} "
        f"({', '.join(accounts) if accounts else 'none found'}). Submissions will "
        f"most likely be rejected. Correct slurm.account in {cfg.config_path()}.",
        result.duration_ms,
    )


# --------------------------------------------------------------------------
# Check 7: the partition
# --------------------------------------------------------------------------


def _check_partition(conf: cfg.Config, ctx: _Context) -> CheckResult:
    alias = conf.ssh_alias or ""
    partition = conf.slurm.partition

    if not partition:
        return CheckResult(
            "partition",
            WARN,
            f"No slurm.partition is set in {cfg.config_path()}, so jobs go to the "
            f"cluster's default partition. Set it if you need a particular set of "
            f"nodes (a GPU partition, say).",
        )

    result = _ssh(
        alias,
        ["scontrol", "show", "partition", shlex.quote(partition)],
        timeout=_ssh_timeout(conf),
    )

    if not result.ok:
        return CheckResult(
            "partition",
            ERROR,
            f"Could not ask {alias} about partition {partition!r}: {result.message()}.",
            result.duration_ms,
        )
    if result.returncode != 0:
        return CheckResult(
            "partition",
            ERROR,
            f"Partition {partition!r} does not exist on {alias}: {result.message()}. "
            f"Run `sinfo -s` there to see the partitions you can use, then correct "
            f"slurm.partition in {cfg.config_path()}.",
            result.duration_ms,
        )

    # Parse once, here, and leave the fields on the context: three more checks
    # below are readings of this same output, and asking scontrol three times
    # would be three SSH round trips for one answer.
    #
    # `split_scontrol_entries` first because scontrol separates entries with
    # blank lines; we asked for one partition by name, so we want the first.
    entries = split_scontrol_entries(result.stdout)
    info = parse_scontrol_kv(entries[0] if entries else result.stdout)
    ctx.partition = info

    state = info.get("State")
    preempt_mode = info.get("PreemptMode")
    priority = info.get("PriorityTier") or info.get("Priority")

    notes = []
    if preempt_mode and preempt_mode.upper() not in ("OFF", "(NULL)"):
        notes.append(
            "jobs here can be preempted, which is exactly the case relay's "
            "requeue handling is for"
        )
    elif preempt_mode:
        notes.append("preemption is off")
    if priority:
        notes.append(f"priority tier {priority}")

    # The four fields the three follow-up checks reason about, printed verbatim
    # so the user can compare them against their own config without a second
    # trip to the login node.
    fields = ", ".join(
        f"{key}={info.get(key) or 'unknown'}"
        for key in ("PreemptMode", "GraceTime", "MaxTime", "DefaultTime")
    )

    detail = f"Partition {partition!r} exists on {alias}"
    if notes:
        detail += "; " + ", ".join(notes)
    detail += f". {fields}."

    if state and state.upper() not in ("UP", "UP*"):
        return CheckResult(
            "partition",
            WARN,
            f"{detail} Its state is {state}, so it may not accept or start jobs "
            f"right now. Check with `sinfo -p {partition}`.",
            result.duration_ms,
        )
    return CheckResult("partition", OK, detail, result.duration_ms)


# --------------------------------------------------------------------------
# Checks 8-10: what the partition's settings mean for this config
# --------------------------------------------------------------------------
#
# All three read `ctx.partition`, which the check above filled in, and all
# three are skipped by `run_checks` when it did not. They are separate checks
# rather than three sentences on the partition line because they answer three
# unrelated questions and any one of them can be wrong on its own.


def _check_requeue_policy(conf: cfg.Config, ctx: _Context) -> CheckResult:
    """Does `requeue_on_term` agree with who actually requeues here?

    Two ways to get this wrong, in opposite directions:

      * `always` on a `PreemptMode=REQUEUE` partition means Slurm requeues the
        job *and* the sidecar asks for a requeue on top. Klone is this case.
      * `never` on a partition that does not requeue means a preempted run
        simply never comes back, and the user finds out hours later.

    `auto` is right on both and is the default, so the ok line says which way
    it resolves here -- "auto" on its own tells you nothing about what will
    happen tonight.
    """
    info = ctx.partition or {}
    preempt_mode = info.get("PreemptMode") or ""
    printed_mode = preempt_mode or "unknown"
    has_requeue = "REQUEUE" in preempt_mode.upper()
    configured = (conf.requeue_on_term or "auto").strip().lower()

    if has_requeue and configured == "always":
        return CheckResult(
            "requeue_policy",
            WARN,
            f"PreemptMode={printed_mode}, but requeue_on_term is 'always'. Slurm "
            f"already requeues preempted jobs on this partition; set "
            f"requeue_on_term: auto or never. As it stands a preempted run is put "
            f"back by Slurm and asked for again by the sidecar, which costs an "
            f"extra attempt that starts and dies immediately.",
        )
    if not has_requeue and configured == "never":
        return CheckResult(
            "requeue_policy",
            WARN,
            f"PreemptMode={printed_mode} does not include REQUEUE, and "
            f"requeue_on_term is 'never', so preempted runs will not come back; "
            f"set requeue_on_term: auto or always. Nothing else on this partition "
            f"puts an evicted job back in the queue.",
        )

    # Ask submit's own resolver, so doctor can never disagree with what the
    # backend will actually write into sidecar.json.
    resolved = resolve_requeue_on_term(configured, preempt_mode)
    behaviour = (
        "calls `scontrol requeue` itself after a SIGTERM"
        if resolved == "always"
        else "does not call `scontrol requeue` after a SIGTERM"
    )
    return CheckResult(
        "requeue_policy",
        OK,
        f"requeue_on_term: {configured} resolves to {resolved!r} on this partition "
        f"(PreemptMode={printed_mode}), so the sidecar {behaviour}.",
    )


def _check_preempt_window(conf: cfg.Config, ctx: _Context) -> CheckResult:
    """How many seconds really pass between SIGTERM and SIGKILL?

    The partition's `GraceTime` wins when it is set. `GraceTime=0` means "no
    partition-specific window", and the cluster-wide `KillWait` applies
    instead -- which is Klone's case, and the reason for the one extra round
    trip here. That trip only happens when GraceTime is 0, because when it is
    not, KillWait is irrelevant and a second SSH call would buy nothing.

    The number matters because the sidecar budgets against `preempt_grace`: it
    waits `preempt_grace - 2` seconds for the training script to checkpoint.
    Configure 30 on a cluster that gives 10 and the sidecar is still waiting
    when SIGKILL lands, so nothing gets written -- not the checkpoint, and not
    the `run_ended` event either.
    """
    info = ctx.partition or {}
    partition = conf.slurm.partition or "the configured partition"
    grace_raw = info.get("GraceTime")
    grace = _seconds_field(grace_raw)

    kill_wait: int | None = None
    kill_wait_note = "not consulted"
    duration_ms: float | None = None

    if grace is not None and grace > 0:
        effective: int | None = grace
    else:
        result = _ssh(
            conf.ssh_alias or "",
            ["scontrol", "show", "config"],
            timeout=_ssh_timeout(conf),
        )
        duration_ms = result.duration_ms
        if result.succeeded():
            kill_wait = parse_kill_wait(result.stdout)
            kill_wait_note = str(kill_wait) if kill_wait is not None else "not reported"
        else:
            kill_wait_note = f"unreadable ({result.message()})"
        effective = kill_wait

    if effective is None:
        return CheckResult(
            "preempt_window",
            WARN,
            f"Could not work out the preemption window on {partition!r} "
            f"(GraceTime={grace_raw or 'unknown'} KillWait={kill_wait_note}). relay "
            f"will budget against preempt_grace: {conf.preempt_grace} from your "
            f"config; confirm it with `scontrol show config | grep KillWait`.",
            duration_ms,
        )

    detail = (
        f"{partition!r}: effective preemption window {effective}s "
        f"(GraceTime={grace_raw or 'unknown'} KillWait={kill_wait_note})."
    )

    if conf.preempt_grace > effective:
        return CheckResult(
            "preempt_window",
            WARN,
            f"{detail} preempt_grace is {conf.preempt_grace} but the cluster only "
            f"gives {effective} seconds; set preempt_grace: {effective} in "
            f"{cfg.config_path()}. Budgeting for time you do not have means the "
            f"sidecar is still waiting for the checkpoint when SIGKILL arrives.",
            duration_ms,
        )
    return CheckResult(
        "preempt_window",
        OK,
        f"{detail} preempt_grace is {conf.preempt_grace}, which fits.",
        duration_ms,
    )


def _check_time_limit(conf: cfg.Config, ctx: _Context) -> CheckResult:
    """Will this config's `--time` be accepted, and is there one at all?

    Both halves are worth a warning. A limit above `MaxTime` is rejected by
    sbatch at submit -- annoying, but loud. No limit at all is the quiet one:
    relay refuses to submit without one (exit 2), and on a cluster whose
    `DefaultTime` is an hour that refusal is the only thing standing between
    the user and a twelve-hour job killed at sixty minutes.
    """
    info = ctx.partition or {}
    partition = conf.slurm.partition or "the configured partition"
    max_raw = info.get("MaxTime")
    default_raw = info.get("DefaultTime") or "unknown"

    # `slurm.time` is the only place a config can set a time limit; `--time`
    # on `relay submit` is the other, and doctor cannot see that one. A
    # `--time` inside `sbatch_extra` is refused at submit, so it is not a
    # third source.
    configured = conf.slurm.time

    if not configured:
        return CheckResult(
            "time_limit",
            WARN,
            f"There is no slurm.time in config; every submit will need --time "
            f"(DefaultTime here is {default_raw}). relay refuses to submit a job "
            f"with no time limit rather than let the partition's default silently "
            f"cap it, so a `relay submit` without --time exits 2.",
        )

    max_seconds = slurm_time_to_seconds(max_raw)
    if max_seconds is None:
        # UNLIMITED, absent, or a spelling we do not recognise. Nothing to
        # compare against, and a QOS limit may still apply -- say so rather
        # than imply the time limit has been validated.
        return CheckResult(
            "time_limit",
            OK,
            f"slurm.time is {configured}; MaxTime={max_raw or 'unknown'} on "
            f"{partition!r} does not cap it. A QOS may still impose its own limit.",
        )

    configured_seconds = slurm_time_to_seconds(configured)
    if configured_seconds is None:
        return CheckResult(
            "time_limit",
            WARN,
            f"slurm.time is {configured!r}, which relay could not read as a Slurm "
            f"time limit, so it cannot be compared against MaxTime={max_raw} on "
            f"{partition!r}. Use one of MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM or "
            f"D-HH:MM:SS.",
        )
    if configured_seconds > max_seconds:
        return CheckResult(
            "time_limit",
            WARN,
            f"slurm.time is {configured} but partition {partition!r} has "
            f"MaxTime={max_raw}, so sbatch will reject every submission. Lower "
            f"slurm.time to {max_raw} or less in {cfg.config_path()}.",
        )
    return CheckResult(
        "time_limit",
        OK,
        f"slurm.time {configured} is within MaxTime={max_raw} on {partition!r}.",
    )


# --------------------------------------------------------------------------
# Check 11: remote_root exists, is writable, has room
# --------------------------------------------------------------------------

# Markers, not exit codes. The script always exits 0, so a non-zero return from
# ssh can only mean a transport failure -- which keeps "the directory is not
# writable" and "the connection dropped" from looking identical to the caller.
_NO_DIR = "RELAY_NO_DIR"
_NO_WRITE = "RELAY_NO_WRITE"
_ROOT_OK = "RELAY_OK"

# One round trip does all three questions: exists, writable, how much room.
# `df -Pk` is the POSIX-portable form: one line per filesystem, sizes in 1K
# blocks, columns that do not wrap however long the device name is.
_REMOTE_ROOT_PROBE = (
    'd={root}; '
    'if [ ! -d "$d" ]; then echo {no_dir}; exit 0; fi; '
    'p="$d/.relay-doctor-probe.$$"; '
    'if ! touch "$p" 2>/dev/null; then echo {no_write}; exit 0; fi; '
    'rm -f "$p"; '
    'echo {ok}; '
    'df -Pk "$d" | tail -1'
)


def _check_remote_root(conf: cfg.Config, ctx: _Context) -> CheckResult:
    alias = conf.ssh_alias or ""
    root = conf.remote_root or ""
    if not root:
        return CheckResult(
            "remote_root",
            ERROR,
            f"No remote_root is set in {cfg.config_path()}. relay needs a directory "
            f"on the cluster's shared filesystem to write run directories into.",
        )

    command = _REMOTE_ROOT_PROBE.format(
        root=shlex.quote(root), no_dir=_NO_DIR, no_write=_NO_WRITE, ok=_ROOT_OK
    )
    result = _ssh(alias, [command], timeout=_ssh_timeout(conf))

    if not result.ok or result.returncode != 0:
        return CheckResult(
            "remote_root",
            ERROR,
            f"Could not inspect {root} on {alias}: {result.message()}.",
            result.duration_ms,
        )

    output = result.stdout
    if _NO_DIR in output:
        return CheckResult(
            "remote_root",
            ERROR,
            f"{root} does not exist on {alias}. Create it with "
            f"`ssh {alias} mkdir -p {shlex.quote(root)}`. It must be on the shared "
            f"filesystem, visible from the compute nodes -- that directory is the "
            f"only channel a running job has back to your laptop.",
            result.duration_ms,
        )
    if _NO_WRITE in output:
        return CheckResult(
            "remote_root",
            ERROR,
            f"{root} exists on {alias} but relay could not create a file in it. "
            f"Check the directory's ownership and permissions, and that your "
            f"group's quota is not full.",
            result.duration_ms,
        )
    if _ROOT_OK not in output:
        return CheckResult(
            "remote_root",
            ERROR,
            f"The check on {root} returned output relay did not understand: "
            f"{_first_line(output) or '(nothing)'}.",
            result.duration_ms,
        )

    free_gb = _parse_df_available_gb(output)
    if free_gb is None:
        return CheckResult(
            "remote_root",
            OK,
            f"{root} exists on {alias} and is writable. Free space could not be "
            f"read from df output.",
            result.duration_ms,
        )
    if free_gb < LOW_DISK_GB:
        return CheckResult(
            "remote_root",
            WARN,
            f"{root} exists on {alias} and is writable, but only {free_gb:.1f} GB "
            f"are free. Checkpoints are large; clear space or move remote_root to a "
            f"filesystem with more room before starting a long run.",
            result.duration_ms,
        )
    return CheckResult(
        "remote_root",
        OK,
        f"{root} exists on {alias}, is writable, and has {free_gb:.1f} GB free.",
        result.duration_ms,
    )


def _parse_df_available_gb(output: str) -> float | None:
    """Pull the "available" column out of `df -Pk` output, in GB, or None.

    POSIX df columns are: filesystem, 1K-blocks, used, available, capacity,
    mount point. We take field 3 (zero-based) from the last line that has at
    least six fields. Best effort: an unparseable df is worth reporting as
    "unknown", never worth crashing a health check over.
    """
    for line in reversed(output.splitlines()):
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            available_kb = float(fields[3])
        except ValueError:
            continue
        # 1024-based, because that is what df -k reports and what the filesystem
        # quota tools on the cluster show.
        return available_kb / (1024.0 * 1024.0)
    return None


# --------------------------------------------------------------------------
# Check 12: the database
# --------------------------------------------------------------------------


def _check_database() -> CheckResult:
    path = cfg.db_path()

    # Note the deliberate ordering: we test for the file *before* constructing a
    # Store, because Store.__init__ creates the database if it is missing. A
    # health check must not manufacture the thing it is checking for.
    if not path.exists():
        return CheckResult(
            "database",
            WARN,
            f"No database at {path} yet. The daemon creates it on its first cycle; "
            f"start it with `relay daemon`. (Nothing is wrong if you have not "
            f"submitted anything.)",
        )

    try:
        store = Store(path)
    except sqlite3.Error as exc:
        return CheckResult(
            "database",
            ERROR,
            f"Could not open {path}: {exc}. If the file is corrupt, stop the daemon "
            f"and delete it -- relay rebuilds it by re-reading the event logs, "
            f"because every insert is idempotent.",
        )
    except OSError as exc:
        return CheckResult("database", ERROR, f"Could not open {path}: {exc}.")

    # Note that opening a Store *sets* journal_mode=WAL, and journal mode is a
    # property of the file that persists, so this check quietly repairs a
    # database some other tool left in rollback-journal mode. What it can still
    # catch is a database that will not accept WAL at all: SQLite needs shared
    # memory for it, which some network filesystems do not provide, and there
    # it silently stays in the old mode.
    try:
        mode = store.journal_mode()
    finally:
        store.close()

    if mode != "wal":
        return CheckResult(
            "database",
            ERROR,
            f"{path} is in journal mode {mode!r} and would not switch to WAL. "
            f"Three processes share this file -- daemon, CLI and dashboard -- and "
            f"without WAL a reader and a writer block each other, so polling fails "
            f"with 'database is locked'. This usually means the database is on a "
            f"network filesystem; move it to local disk by setting XDG_DATA_HOME.",
        )
    return CheckResult("database", OK, f"{path} opens and is in WAL mode.")


# --------------------------------------------------------------------------
# Check 13: the usage table, and when it was last filled
# --------------------------------------------------------------------------


def _check_usage_table() -> CheckResult:
    """Does the `usage` table exist, and has the daemon ever filled it?

    Separate from the database check because they fail for different reasons.
    The table missing means a database created before usage accounting existed
    and never migrated -- a schema problem. The table being empty of fetches
    means the daemon is running but its usage calls are failing or have never
    had a run to report on, which is a *daemon* problem. One line saying "the
    database is fine" would hide both.
    """
    path = cfg.db_path()
    if not path.exists():
        return _skipped("usage_table", "there is no database yet")

    try:
        store = Store(path)
    except (sqlite3.Error, OSError) as exc:
        return _skipped("usage_table", f"the database could not be opened ({exc})")

    try:
        # sqlite_master is SQLite's own catalogue: one row per table. Asking it
        # is how you test for a table without touching it.
        row = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'usage'"
        ).fetchone()
        last = store.get_daemon_state().get("last_usage_fetch_ts")
    finally:
        store.close()

    if row is None:
        return CheckResult(
            "usage_table",
            ERROR,
            f"{path} has no `usage` table, so `relay usage` has nowhere to read "
            f"GPU-hours from. Stop the daemon and delete the file; relay rebuilds "
            f"it by re-reading the event logs, because every insert is idempotent.",
        )

    if not last:
        return CheckResult(
            "usage_table",
            WARN,
            f"The `usage` table exists, but usage has never been fetched (the "
            f"daemon fetches every 5 minutes while runs exist). Nothing is wrong "
            f"if you have not submitted anything yet; otherwise check that the "
            f"daemon is running and that `sacct` works on the cluster.",
        )

    age = _age_seconds(last)
    if age is None:
        return CheckResult(
            "usage_table",
            WARN,
            f"The `usage` table exists, but the stored last-fetch timestamp "
            f"{last!r} is unreadable.",
        )
    return CheckResult(
        "usage_table",
        OK,
        f"The `usage` table exists; usage last fetched {_humanise_age(age)}.",
    )


# --------------------------------------------------------------------------
# Check 14: the daemon's lock, and its last cycle
# --------------------------------------------------------------------------


def _check_daemon() -> CheckResult:
    """Is a daemon running? Answered by trying to take its lock.

    The trick is that we *want* to fail. The daemon holds an exclusive `flock`
    on this file for as long as it lives. If our non-blocking attempt is
    refused with EWOULDBLOCK, somebody else holds it, and that somebody is the
    daemon. If we get the lock, nobody is running.

    Why flock rather than a PID file: a PID file outlives the process that
    wrote it, so a crashed daemon leaves a stale file that looks exactly like a
    running one. A flock is owned by the kernel and released the instant the
    process dies, however it dies.

    Subtlety worth remembering: flock locks belong to the *open file
    description*, not to the process. Two separate `open()` calls on the same
    file -- even inside one process -- create two descriptions and therefore do
    conflict. (POSIX `fcntl`/`lockf` record locks are the opposite: per-process,
    so a second attempt inside the same process silently succeeds. That
    difference is why the test for this check can hold the lock on one file
    object and observe a real EWOULDBLOCK on another.)
    """
    path = cfg.daemon_lock_path()

    # Do not create the file. If it is missing, the daemon has never run here,
    # and that is already the answer -- no need to leave a lock file behind in
    # an otherwise untouched data directory.
    if not path.exists():
        return CheckResult(
            "daemon",
            WARN,
            f"No daemon lock file at {path}, so the daemon has never run on this "
            f"machine. Start it with `relay daemon`; without it nothing tails your "
            f"runs and `relay ls` will show stale data.",
        )

    try:
        handle = open(path, "a+")
    except OSError as exc:
        return CheckResult(
            "daemon",
            ERROR,
            f"Could not open the daemon lock file {path}: {exc}.",
        )

    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                return CheckResult(
                    "daemon",
                    OK,
                    f"The daemon is running: something holds the lock on {path}.",
                )
            return CheckResult(
                "daemon",
                ERROR,
                f"Could not test the daemon lock at {path}: {exc}.",
            )
        # We got it, so nobody was holding it. Let go at once: doctor must not
        # keep a lock that would stop a daemon from starting a moment later.
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return CheckResult(
            "daemon",
            WARN,
            f"No daemon is running (the lock on {path} was free). Start it with "
            f"`relay daemon`; until then, no new events reach the database.",
        )
    finally:
        handle.close()


def _check_last_sync() -> CheckResult:
    """When did the daemon last finish a full cycle?

    Split out from the lock check because the two can disagree in a way worth
    seeing: a daemon that holds its lock but has not completed a cycle in an
    hour is wedged, and folding that into one line would hide it behind the
    reassuring "daemon is running".
    """
    path = cfg.db_path()
    if not path.exists():
        return _skipped("last_sync", "there is no database yet")

    try:
        store = Store(path)
    except (sqlite3.Error, OSError) as exc:
        return _skipped("last_sync", f"the database could not be opened ({exc})")

    try:
        last = store.last_synced()
        started = store.meta_get(store_mod.META_DAEMON_STARTED_TS)
    finally:
        store.close()

    if last is None:
        detail = "The daemon has never recorded a completed cycle."
        if started:
            detail += f" It last started at {started}."
        return CheckResult(
            "last_sync",
            WARN,
            detail + " Start it with `relay daemon` and re-run this check.",
        )

    age = _age_seconds(last)
    if age is None:
        return CheckResult(
            "last_sync", WARN, f"The stored last-sync timestamp {last!r} is unreadable."
        )
    if age > STALE_SYNC_SECONDS:
        return CheckResult(
            "last_sync",
            WARN,
            f"The daemon last completed a cycle {_humanise_age(age)} ({last}). Its "
            f"slowest normal interval is 60 seconds, so anything older than "
            f"{STALE_SYNC_SECONDS // 60} minutes means it is stopped or stuck. "
            f"Restart it with `relay daemon`.",
        )
    return CheckResult("last_sync", OK, f"The daemon last synced {_humanise_age(age)}.")


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------

# Remote checks in the order they run. Listing them as data rather than as ten
# hand-written call sites means the "skip everything remote" path below cannot
# forget one.
_REMOTE_CHECKS: tuple[tuple[str, object], ...] = (
    # `control_dir` goes first because it is the only check that must run
    # before anything builds an ssh argv: every builder calls
    # `ensure_control_dir()`, so a check placed later would always find the
    # directory already repaired and report a problem the user never sees.
    ("control_dir", _check_control_dir),
    # `master` next: it costs nothing (a local socket probe), and knowing the
    # answer lets every check after it name the right cause.
    ("master", _check_master),
    ("ssh", _check_ssh),
    ("remote_python", _check_remote_python),
    ("sbatch", _check_sbatch),
    ("slurm_account", _check_account),
    ("partition", _check_partition),
    ("requeue_policy", _check_requeue_policy),
    ("preempt_window", _check_preempt_window),
    ("time_limit", _check_time_limit),
    ("remote_root", _check_remote_root),
)

# Which remote checks need a working SSH round trip. `master` is absent on
# purpose: `ssh -O check` talks to a local socket and is often the very
# explanation for a failing round trip, so it is worth running anyway.
_NEEDS_SSH = frozenset(
    {
        "remote_python",
        "sbatch",
        "slurm_account",
        "partition",
        "requeue_policy",
        "preempt_window",
        "time_limit",
        "remote_root",
    }
)

# Which checks read the partition fields that `_check_partition` parsed. With
# no partition to reason about they have nothing to say, so they skip rather
# than warn about a comparison they could not make.
_NEEDS_PARTITION = frozenset({"requeue_policy", "preempt_window", "time_limit"})


def _timed(name: str, func, *args) -> CheckResult:
    """Run one check, time it, and never let it raise.

    A check that throws an unexpected exception becomes an error *result*, so
    the remaining checks still run. This is the mechanical half of "never fail
    fast"; the other half is the skip logic in `run_checks`.
    """
    start = time.perf_counter()
    try:
        result = func(*args)
    except Exception as exc:  # noqa: BLE001 - a health check may not crash
        return CheckResult(
            name,
            ERROR,
            f"The {name} check failed unexpectedly: {exc!r}. This is a bug in relay; "
            f"the other checks still ran.",
            _ms_since(start),
        )
    if result.duration_ms is None and result.status != SKIP:
        result.duration_ms = _ms_since(start)
    return result


def run_checks(config_path: Path | None = None) -> list[CheckResult]:
    """Run every check and return one `CheckResult` each, in a stable order.

    Nothing raises out of here. A check that cannot run returns `"skip"` with
    the reason; a check that blows up returns `"error"` with the exception.
    `config_path` exists so tests (and a user with more than one cluster) can
    point at a specific file.
    """
    results: list[CheckResult] = []

    # Timed by hand rather than through `_timed`, because this is the one check
    # that returns a value as well as a result.
    start = time.perf_counter()
    try:
        conf, config_result = _check_config(config_path)
    except Exception as exc:  # noqa: BLE001 - a health check may not crash
        conf, config_result = None, CheckResult(
            "config", ERROR, f"The config check failed unexpectedly: {exc!r}."
        )
    config_result.duration_ms = _ms_since(start)
    results.append(config_result)

    # One reason, computed once, for every remote check to share.
    if conf is None:
        skip_reason: str | None = (
            "the config could not be loaded, so relay does not know which host to check"
        )
    elif conf.backend == "local":
        skip_reason = "backend is local, so there is no cluster to check"
    else:
        skip_reason = None

    # The two facts checks hand forward: whether a master is up, and what the
    # partition said. Everything else a check learns, it reports and forgets.
    ctx = _Context()
    ssh_ok = False
    partition_ok = False

    for name, func in _REMOTE_CHECKS:
        if skip_reason is not None:
            results.append(_skipped(name, skip_reason))
            continue
        if name in _NEEDS_SSH and not ssh_ok:
            # Do not hammer a host we already know we cannot reach: on a
            # two-factor cluster, five more doomed connections is five more
            # timeouts and possibly five more Duo pushes.
            results.append(_skipped(name, "the SSH check did not succeed"))
            continue
        if name in _NEEDS_PARTITION and not partition_ok:
            results.append(_skipped(name, "the partition check did not succeed"))
            continue

        result = _timed(name, func, conf, ctx)
        if name == "ssh":
            # A slow connection still works, so a warning here does not stop the
            # remote checks that follow.
            ssh_ok = result.status in (OK, WARN)
        if name == "partition":
            # A partition that is merely DOWN still reported its settings, so a
            # warning is enough to reason about; what we need is the fields.
            partition_ok = result.status in (OK, WARN) and bool(ctx.partition)
        results.append(result)

    # Local state: always checked, because it is local state. These are the
    # checks that matter in CI, where there is no cluster at all.
    results.append(_timed("database", _check_database))
    results.append(_timed("usage_table", _check_usage_table))
    results.append(_timed("daemon", _check_daemon))
    results.append(_timed("last_sync", _check_last_sync))

    return results
