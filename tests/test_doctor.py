"""Tests for relay.doctor.

Two rules hold for every test in this file.

**No test may make a real SSH connection.** An autouse fixture replaces
`doctor._ssh` and `doctor._ssh_master_check` with stubs that fail the test
loudly if they are called without being told what to answer. Tests that
exercise a remote check install their own fakes in their place. This is why
those two are thin helpers over one subprocess call: two seams to patch, and
nowhere else in the module shells out.

**No test may touch the real home directory.** HOME and both XDG variables are
redirected into `tmp_path`, so `cfg.config_path()`, `cfg.db_path()` and
`cfg.daemon_lock_path()` all resolve inside the throwaway directory. Without
that, a test for the database check would happily open (and a test for `init`
would overwrite) the developer's real state.
"""

from __future__ import annotations

import dataclasses
import fcntl
import os
import stat
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest

from relay import config as cfg
from relay import doctor
from relay import ssh as ssh_mod
from relay.store import Store, utc_now_iso

# Captured before any test can monkeypatch the module attribute, so the tests
# for the helpers themselves can still reach the real implementations while the
# autouse guard below protects everything else.
REAL_SSH = doctor._ssh
REAL_MASTER_CHECK = doctor._ssh_master_check

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point HOME and both XDG variables at a throwaway directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    return home


@pytest.fixture(autouse=True)
def no_real_ssh(monkeypatch):
    """Make an unmocked SSH call a test failure rather than a network call."""

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "doctor._ssh was called without a fake installed; a test tried to "
            "make a real SSH connection"
        )

    monkeypatch.setattr(doctor, "_ssh", _forbidden)
    monkeypatch.setattr(doctor, "_ssh_master_check", _forbidden)


def write_config(text: str) -> Path:
    path = cfg.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def write_slurm_config(**overrides) -> Path:
    """A minimally valid slurm config, so the remote checks actually run.

    Built line by line rather than from one f-string because three of the keys
    are optional: `time` can be omitted entirely (pass `time=None`), and the
    two SIGTERM settings only appear when a test cares about them. A config
    that always carried them could not exercise the defaults.
    """
    lines = [
        "backend: slurm",
        f"ssh_alias: {overrides.get('ssh_alias', 'testcluster')}",
        f"remote_root: {overrides.get('remote_root', '/mmfs1/gscratch/mlopt/you/relay')}",
        f"remote_python: {overrides.get('remote_python', 'python3')}",
    ]
    if "requeue_on_term" in overrides:
        lines.append(f"requeue_on_term: {overrides['requeue_on_term']}")
    if "preempt_grace" in overrides:
        lines.append(f"preempt_grace: {overrides['preempt_grace']}")
    lines += [
        "slurm:",
        f"  account: {overrides.get('account', 'mlopt')}",
        f"  partition: {overrides.get('partition', 'gpu-a40')}",
    ]
    time_limit = overrides.get("time", "04:00:00")
    if time_limit:
        lines.append(f'  time: "{time_limit}"')
    return write_config("\n".join(lines) + "\n")


def ssh_ok(stdout: str = "", duration_ms: float = 30.0) -> doctor.SshResult:
    return doctor.SshResult(
        ok=True, returncode=0, stdout=stdout, stderr="", duration_ms=duration_ms
    )


def ssh_rc(returncode: int, stderr: str = "", duration_ms: float = 30.0) -> doctor.SshResult:
    """ssh ran fine; the *remote command* failed."""
    return doctor.SshResult(
        ok=True, returncode=returncode, stdout="", stderr=stderr, duration_ms=duration_ms
    )


def ssh_transport_failure(message: str = "Connection refused") -> doctor.SshResult:
    """ssh itself could not run or connect."""
    return doctor.SshResult(
        ok=False, returncode=None, stdout="", stderr="", duration_ms=12.0, error=message
    )


# A df -Pk last line: filesystem, 1K-blocks, used, available, capacity, mount.
# 209715200 KB available is 200 GiB.
DF_ROOMY = "gpfs1 1073741824 863986624 209715200 81% /mmfs1"
DF_TIGHT = "gpfs1 1073741824 1068547072 5194752 99% /mmfs1"  # ~5 GiB


class SshCall(NamedTuple):
    """One recorded call to the faked `_ssh` / `_ssh_master_check`.

    `timeout` is recorded because it is now load-bearing: every remote call in
    doctor must wait at least the configured `ssh_timeout`, and a check that
    quietly kept a tighter literal of its own is exactly the bug that made a
    healthy-but-slow login node look unreachable.
    """

    alias: str
    command: str
    timeout: float


def commands_of(calls: list[SshCall], needle: str = "") -> list[str]:
    """The recorded command lines, optionally only those containing `needle`."""
    return [call.command for call in calls if needle in call.command]


def install_ssh(monkeypatch, table: dict[str, doctor.SshResult], default=None, calls=None):
    """Install fake `_ssh` and `_ssh_master_check` that answer by substring.

    The match string is the remote command's words joined with spaces, so
    "command -v sbatch" selects the sbatch lookup and "show config" selects the
    KillWait probe. The master check has no remote command at all, so it is
    recorded and matched as the literal "-O check".

    These stubs stand in for `_ssh` itself, so what they see is the *unwrapped*
    command -- the `bash --noprofile --norc -c` wrapping happens inside the real
    `_ssh`, which is where its own tests look for it.

    `table` is an ordinary dict and matching walks it in insertion order, so a
    narrow needle ("show config") must be listed before a broad one that also
    matches it ("scontrol").
    """

    def _answer(alias, command, timeout):
        if calls is not None:
            calls.append(SshCall(alias, command, timeout))
        for needle, result in table.items():
            if needle in command:
                return result
        if default is not None:
            return default
        raise AssertionError(f"no fake SSH response for command: {command!r}")

    def _fake(alias, args, timeout=ssh_mod.DEFAULT_SSH_TIMEOUT):
        return _answer(alias, " ".join(args), timeout)

    def _fake_master(alias, timeout=ssh_mod.DEFAULT_SSH_TIMEOUT):
        return _answer(alias, "-O check", timeout)

    monkeypatch.setattr(doctor, "_ssh", _fake)
    monkeypatch.setattr(doctor, "_ssh_master_check", _fake_master)


# A Klone-shaped partition: preemptible, no partition-specific grace window
# (so KillWait decides it), no ceiling on --time, one hour if you forget one.
KLONE_PARTITION = (
    "PartitionName=gpu-a40 State=UP PreemptMode=REQUEUE PriorityTier=1 "
    "GraceTime=0 MaxTime=UNLIMITED DefaultTime=01:00:00"
)

# `scontrol show config` prints aligned `Key = Value`, unlike the packed
# `Key=Value` of `show partition`. Both spellings have to parse.
KLONE_SLURM_CONFIG = (
    "PreemptMode              = REQUEUE\n"
    "PreemptType              = preempt/qos\n"
    "KillWait                = 10 sec\n"
)


def healthy_ssh_table() -> dict[str, doctor.SshResult]:
    """Answers that make every remote check pass."""
    return {
        "true": ssh_ok(),
        "-O check": ssh_ok("Master running (pid=1234)"),
        "--version": ssh_ok("Python 3.11.9"),
        "command -v sbatch": ssh_ok("/usr/bin/sbatch"),
        "sacctmgr": ssh_ok("mlopt\nstf\n"),
        # Before the broad "scontrol" needle, which would otherwise swallow it.
        "show config": ssh_ok(KLONE_SLURM_CONFIG),
        "scontrol": ssh_ok(KLONE_PARTITION),
        "RELAY_NO_DIR": ssh_ok(f"{doctor._ROOT_OK}\n{DF_ROOMY}\n"),
    }


def partition_table(blob: str, **extra: doctor.SshResult) -> dict[str, doctor.SshResult]:
    """The healthy table with a different `scontrol show partition` answer."""
    table = healthy_ssh_table()
    table["scontrol"] = ssh_ok(blob)
    table.update(extra)
    return table


def by_name(results: list[doctor.CheckResult]) -> dict[str, doctor.CheckResult]:
    return {r.name: r for r in results}


ALL_CHECK_NAMES = [
    "config",
    "control_dir",
    "master",
    "ssh",
    "remote_python",
    "sbatch",
    "slurm_account",
    "partition",
    "requeue_policy",
    "preempt_window",
    "time_limit",
    "remote_root",
    "database",
    "usage_table",
    "daemon",
    "last_sync",
]

REMOTE_CHECK_NAMES = [
    "control_dir",
    "master",
    "ssh",
    "remote_python",
    "sbatch",
    "slurm_account",
    "partition",
    "requeue_policy",
    "preempt_window",
    "time_limit",
    "remote_root",
]

# The checks that read fields out of `scontrol show partition`.
PARTITION_DERIVED = ["requeue_policy", "preempt_window", "time_limit"]


# --------------------------------------------------------------------------
# run_checks: never fails fast
# --------------------------------------------------------------------------


def test_all_checks_run_when_config_is_missing():
    """No config at all must still produce one result per check, and no raise."""
    results = doctor.run_checks()

    assert [r.name for r in results] == ALL_CHECK_NAMES
    assert all(r.status in (doctor.OK, doctor.WARN, doctor.ERROR, doctor.SKIP) for r in results)

    checks = by_name(results)
    assert checks["config"].status == doctor.ERROR
    assert "relay init" in checks["config"].detail
    for name in REMOTE_CHECK_NAMES:
        assert checks[name].status == doctor.SKIP, name
        assert "config" in checks[name].detail
    # Local checks still run: they do not depend on the config file.
    assert checks["database"].status == doctor.WARN
    assert checks["daemon"].status == doctor.WARN


def test_local_backend_skips_every_remote_check():
    write_config("backend: local\n")

    checks = by_name(doctor.run_checks())

    assert checks["config"].status == doctor.OK
    for name in REMOTE_CHECK_NAMES:
        assert checks[name].status == doctor.SKIP, name
        assert "backend is local" in checks[name].detail
        # A skipped check did no work, so it reports no duration.
        assert checks[name].duration_ms is None, name


def test_an_exploding_check_becomes_an_error_and_others_still_run(monkeypatch):
    write_config("backend: local\n")

    def _boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(doctor, "_check_database", _boom)

    results = doctor.run_checks()
    checks = by_name(results)

    assert checks["database"].status == doctor.ERROR
    assert "disk on fire" in checks["database"].detail
    # The checks after it still ran.
    assert checks["daemon"].status in (doctor.OK, doctor.WARN)
    assert [r.name for r in results] == ALL_CHECK_NAMES


def test_explicit_config_path_is_used(tmp_path):
    elsewhere = tmp_path / "other.yaml"
    elsewhere.write_text("backend: local\n", encoding="utf-8")

    checks = by_name(doctor.run_checks(config_path=elsewhere))

    assert checks["config"].status == doctor.OK
    assert str(elsewhere) in checks["config"].detail


def test_unparseable_config_is_an_error_not_a_crash():
    write_config("backend: [this is not a string]\n")

    checks = by_name(doctor.run_checks())

    assert checks["config"].status == doctor.ERROR
    assert checks["ssh"].status == doctor.SKIP


# --------------------------------------------------------------------------
# The SSH helper itself
# --------------------------------------------------------------------------


class FakeProc:
    returncode = 0
    stdout = "hi"
    stderr = ""


def record_subprocess(monkeypatch) -> dict:
    """Capture the argv doctor would have handed to subprocess.run."""
    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    return seen


def test_ssh_carries_relays_control_options_and_a_bare_alias(monkeypatch):
    """The argv must be the one relay.ssh builds: relay's own ControlMaster."""
    seen = record_subprocess(monkeypatch)

    result = REAL_SSH("klone", ["true"])

    argv = seen["argv"]
    # Identical to what the backend and the daemon use. doctor testing a
    # different connection from the one relay actually opens would be useless.
    assert argv == ssh_mod.build_ssh_argv("klone", ssh_mod.remote_shell_argv("true"))
    assert argv[0] == "ssh"
    assert argv[-6:] == ["klone", "bash", "--noprofile", "--norc", "-c", "true"]
    assert "@" not in " ".join(argv)

    options = [argv[i + 1] for i, word in enumerate(argv) if word == "-o"]
    assert options == [
        "ControlMaster=auto",
        "ControlPath=~/.relay/cm-%C",
        "ControlPersist=8h",
        "ServerAliveInterval=30",
        "ServerAliveCountMax=3",
        "BatchMode=yes",
    ]
    assert seen["kwargs"]["timeout"] == ssh_mod.DEFAULT_SSH_TIMEOUT
    assert result.succeeded()
    assert result.duration_ms >= 0


def test_doctor_probes_run_under_a_clean_remote_shell(monkeypatch):
    """`bash --noprofile --norc -c`, so no startup file runs for a probe.

    The whole reason a round trip to Klone costs seconds is that a shell starts
    first, and `conda init` in ~/.bashrc is not free. None of relay's probes
    (`true`, `command -v sbatch`, `df`) needs anything a startup file sets up,
    so none of them should pay for one. This cannot skip the login shell sshd
    itself spawns, which is why the slow-round-trip warning still exists.
    """
    seen = record_subprocess(monkeypatch)

    REAL_SSH("klone", ["command", "-v", "sbatch"])

    argv = seen["argv"]
    assert "bash --noprofile --norc -c" in " ".join(argv)
    # The whole command is one argv word, quoted: ssh joins these with spaces
    # and the remote login shell re-parses the result.
    assert argv[-5:] == ["bash", "--noprofile", "--norc", "-c", "'command -v sbatch'"]


def test_the_remote_root_probe_survives_being_wrapped(monkeypatch):
    """The probe is a whole shell script, semicolons and all.

    Without quoting, the remote *login* shell would eat the `;` and the `if`
    before bash ever saw them, and the probe would run in pieces.
    """
    seen = record_subprocess(monkeypatch)
    probe = doctor._REMOTE_ROOT_PROBE.format(
        root="'/mmfs1/gscratch/mlopt/you/relay'",
        no_dir=doctor._NO_DIR,
        no_write=doctor._NO_WRITE,
        ok=doctor._ROOT_OK,
    )

    REAL_SSH("klone", [probe])

    argv = seen["argv"]
    assert argv[-5:-1] == ["bash", "--noprofile", "--norc", "-c"]
    # One word, and the script -- semicolons included -- is inside it.
    assert doctor._NO_DIR in argv[-1]
    assert argv[-1].startswith("'") and argv[-1].endswith("'")


def test_master_check_uses_relays_control_path_and_no_remote_command(monkeypatch):
    seen = record_subprocess(monkeypatch)

    REAL_MASTER_CHECK("klone")

    assert seen["argv"] == ssh_mod.master_check_argv("klone")
    assert seen["argv"] == [
        "ssh",
        "-o",
        "ControlPath=~/.relay/cm-%C",
        "-O",
        "check",
        "klone",
    ]


def test_a_user_at_host_alias_is_a_value_not_an_exception(monkeypatch):
    """relay.ssh refuses these; doctor must report, not crash."""
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **kw: FakeProc())

    result = REAL_SSH("me@klone.hyak.uw.edu", ["true"])

    assert result.ok is False
    assert "alias" in result.message()


def test_ssh_timeout_is_a_value_not_an_exception(monkeypatch):
    def fake_run(argv, **kwargs):
        raise doctor.subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 15))

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)

    result = REAL_SSH("klone", ["true"], timeout=15)

    assert result.ok is False
    assert result.succeeded() is False
    assert "did not answer" in result.message()


def test_ssh_missing_binary_is_a_value_not_an_exception(monkeypatch):
    def fake_run(argv, **kwargs):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)

    result = REAL_SSH("klone", ["true"])

    assert result.ok is False
    assert "ssh" in result.message()


# --------------------------------------------------------------------------
# Check 2: ssh round trip
# --------------------------------------------------------------------------


def test_fast_ssh_is_ok(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    checks = by_name(doctor.run_checks())

    assert checks["ssh"].status == doctor.OK
    assert "testcluster" in checks["ssh"].detail
    assert checks["ssh"].duration_ms is not None


# The four branches of the ssh check. Which one fires depends on *two* facts,
# never on the clock alone: did the round trip work, and is relay's master
# alive? Reading the clock by itself is what produced a confident, false
# "ssh opened a fresh connection" on a host whose master was up the whole time.


def test_a_round_trip_slower_than_a_second_but_under_the_timeout_is_ok(monkeypatch):
    """Klone's real number, roughly: 4.8s through a live master, and fine.

    This is the measurement that used to be reported as a broken
    ControlMaster. Five seconds is what a login node costs when every shell
    runs `conda init` first; it is slow, not wrong, and well inside the 30s
    relay is willing to wait.
    """
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_ok(duration_ms=5000.0)
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["ssh"]

    assert result.status == doctor.OK
    assert "5.0s" in result.detail
    assert "ControlMaster" not in result.detail


def test_twenty_seconds_still_passes_on_the_default_timeout(monkeypatch):
    """The threshold is ssh_timeout itself, so 20 < 30 is not worth a word."""
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_ok(duration_ms=20_000.0)
    install_ssh(monkeypatch, table)

    assert by_name(doctor.run_checks())["ssh"].status == doctor.OK


def test_slow_round_trip_with_a_live_master_blames_shell_startup(monkeypatch):
    """The fix. With a master up, the only honest culprit is the far end."""
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_ok(duration_ms=31_000.0)
    install_ssh(monkeypatch, table)

    checks = by_name(doctor.run_checks())
    detail = checks["ssh"].detail

    assert checks["ssh"].status == doctor.WARN
    assert "31.0s" in detail
    assert "shell startup" in detail
    assert "~/.bashrc" in detail
    assert "ssh_timeout" in detail
    # The two claims that were false: nothing was re-negotiated, and there is
    # nothing for `relay connect` to fix.
    assert "fresh connection" not in detail
    assert "relay connect" not in detail
    # Slow is not broken: the checks behind it still ran.
    assert checks["sbatch"].status == doctor.OK
    assert checks["remote_root"].status == doctor.OK
    # And a warning does not fail the run.
    assert doctor.worst_status(list(checks.values())) == 0


def test_slow_round_trip_with_no_master_does_blame_the_connection(monkeypatch):
    """Here the old sentence was right, and it stays."""
    write_slurm_config()
    table = no_master_table()
    table["true"] = ssh_ok(duration_ms=31_000.0)
    install_ssh(monkeypatch, table)

    checks = by_name(doctor.run_checks())
    detail = checks["ssh"].detail

    assert checks["ssh"].status == doctor.WARN
    assert "fresh connection" in detail
    assert "relay connect" in detail
    # Still only a warning, so the remote checks behind it are not skipped.
    assert checks["sbatch"].status == doctor.OK


def test_ssh_failure_with_a_live_master_points_at_the_timeout(monkeypatch):
    """A master is up, so a failure cannot be an authentication problem.

    Nothing here may say `relay connect` -- including the sentence the timeout
    itself produces, which used to carry that guess inside its own message.
    """
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_transport_failure("ssh did not answer within 30 seconds")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["ssh"]

    assert result.status == doctor.ERROR
    assert "ssh_timeout" in result.detail
    assert "did not answer" in result.detail
    assert "relay connect" not in result.detail


def test_the_timeout_message_itself_does_not_guess_at_a_cause(monkeypatch):
    """`_run_ssh` has no idea whether a master is alive, so it must not say."""

    def fake_run(argv, **kwargs):
        raise doctor.subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)

    result = REAL_SSH("klone", ["true"], timeout=30)

    assert "did not answer within 30 seconds" in result.message()
    assert "relay connect" not in result.message()


def test_failed_ssh_is_an_error_and_later_remote_checks_skip(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_transport_failure("Connection timed out")
    install_ssh(monkeypatch, table)

    checks = by_name(doctor.run_checks())

    assert checks["ssh"].status == doctor.ERROR
    assert "Connection timed out" in checks["ssh"].detail
    for name in ("remote_python", "sbatch", "slurm_account", "partition", "remote_root"):
        assert checks[name].status == doctor.SKIP, name
        assert "SSH check" in checks[name].detail
    for name in PARTITION_DERIVED:
        assert checks[name].status == doctor.SKIP, name
    # The master check still runs: `ssh -O check` is local and often the reason.
    assert checks["master"].status == doctor.OK
    # And the local checks are entirely unaffected.
    assert checks["database"].status == doctor.WARN


def test_ssh_nonzero_exit_is_an_error(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_rc(255, stderr="Permission denied (publickey).")
    install_ssh(monkeypatch, table)

    checks = by_name(doctor.run_checks())

    assert checks["ssh"].status == doctor.ERROR
    assert "Permission denied" in checks["ssh"].detail


def test_ssh_255_quotes_the_last_line_of_stderr(monkeypatch):
    """Exit 255 is ssh's own failure, and ssh prints its reason last.

    The first lines belong to the connection attempt ("Authenticated to ...");
    the line that names the problem comes after them. Reporting only the first
    line, or only "exited 255", hides the one sentence that would have told the
    user what was wrong.
    """
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_rc(
        255,
        stderr=(
            "debug1: Connecting to klone-login03.hyak.local port 22.\n"
            "Authenticated to klone-login03 using \"keyboard-interactive\".\n"
            "unix_listener: cannot bind to path /home/you/.relay/cm-9f2a: "
            "No such file or directory\n"
        ),
    )
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["ssh"]

    assert result.status == doctor.ERROR
    assert "unix_listener: cannot bind" in result.detail
    assert "Authenticated to" not in result.detail


# --------------------------------------------------------------------------
# Check 2: the ControlPath directory
# --------------------------------------------------------------------------
#
# Why this check exists, given that `ssh.ensure_control_dir()` already fixes
# the problem: no test in this file has ever run the real ssh binary (the
# autouse `no_real_ssh` fixture sees to that), and the original bug --
# "unix_listener: cannot bind to path ~/.relay/cm-...: No such file or
# directory" -- happens *inside* that binary, at socket-bind time, after Duo
# has already been answered. A stubbed `_ssh` seam could never have caught it,
# and neither could any assertion about the argv, because the argv was correct.
# So these tests are not really "does the fix work"; they are "does doctor tell
# the user, before a Duo push is spent finding out".


def mode_of(path: Path) -> int:
    """The permission bits of a path, without the file-type bits."""
    return stat.S_IMODE(path.stat().st_mode)


def test_relay_dir_resolves_inside_the_test_home(isolated_home):
    """Guard for every test below: `~` must expand to the throwaway HOME.

    `os.path.expanduser` reads $HOME on POSIX, which the autouse fixture has
    redirected -- so `ssh.relay_dir()` points into tmp_path and a test that
    creates it cannot touch the developer's real `~/.relay`.
    """
    assert ssh_mod.relay_dir() == isolated_home / ".relay"


def test_missing_control_dir_is_created_and_reported(monkeypatch, isolated_home):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())
    relay_dir = isolated_home / ".relay"
    assert not relay_dir.exists()

    result = by_name(doctor.run_checks())["control_dir"]

    assert result.status == doctor.OK
    assert "created" in result.detail
    assert str(relay_dir) in result.detail
    # The sentence has to say why it mattered, or "created a directory" reads
    # like noise rather than the thing that broke the first real run.
    assert "Duo" in result.detail
    assert relay_dir.is_dir()
    assert mode_of(relay_dir) == 0o700


def test_loose_control_dir_is_corrected_and_warned_about(monkeypatch, isolated_home):
    """0755 is a warning, not an ok: relay fixed it, but it was wrong."""
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())
    relay_dir = isolated_home / ".relay"
    relay_dir.mkdir()
    os.chmod(relay_dir, 0o755)  # mkdir(mode=) is masked by the umask; this is not

    result = by_name(doctor.run_checks())["control_dir"]

    assert result.status == doctor.WARN
    assert "0755" in result.detail
    assert "corrected" in result.detail
    assert mode_of(relay_dir) == 0o700
    # A warning must not fail the run as a whole.
    assert doctor.worst_status([result]) == 0


def test_correct_control_dir_is_left_alone(monkeypatch, isolated_home):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())
    relay_dir = isolated_home / ".relay"
    relay_dir.mkdir()
    os.chmod(relay_dir, 0o700)
    before = relay_dir.stat().st_mtime

    result = by_name(doctor.run_checks())["control_dir"]

    assert result.status == doctor.OK
    assert "exists with mode 0700" in result.detail
    assert "created" not in result.detail
    assert mode_of(relay_dir) == 0o700
    assert relay_dir.stat().st_mtime == before


def test_a_file_where_the_control_dir_belongs_is_an_error(monkeypatch, isolated_home):
    """And the checks after it still run: doctor never fails fast."""
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())
    relay_dir = isolated_home / ".relay"
    relay_dir.write_text("something else lives here", encoding="utf-8")

    checks = by_name(doctor.run_checks())

    assert checks["control_dir"].status == doctor.ERROR
    assert "not a directory" in checks["control_dir"].detail
    assert str(relay_dir) in checks["control_dir"].detail
    # doctor never deletes anything it found in the way.
    assert relay_dir.read_text(encoding="utf-8") == "something else lives here"
    # Never fail fast: the remote checks behind it still ran.
    assert checks["master"].status == doctor.OK
    assert checks["ssh"].status == doctor.OK
    assert checks["database"].status == doctor.WARN


def test_control_dir_skips_on_the_local_backend(isolated_home):
    """No cluster, no ssh, no reason to create a control socket directory."""
    write_config("backend: local\n")

    result = by_name(doctor.run_checks())["control_dir"]

    assert result.status == doctor.SKIP
    assert "backend is local" in result.detail
    assert not (isolated_home / ".relay").exists()


def test_control_dir_runs_before_the_master_check(monkeypatch):
    """Order is the whole point.

    Every argv builder in `ssh.py` calls `ensure_control_dir()`, so the first
    check that touches ssh repairs the directory as a side effect. Run after
    that and this check would always find a tidy 0700 directory and report a
    problem the user never got told about.
    """
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    names = [r.name for r in doctor.run_checks()]

    assert names.index("control_dir") < names.index("master")
    assert names.index("control_dir") == names.index("config") + 1


# --------------------------------------------------------------------------
# Check 2: relay's own SSH master
# --------------------------------------------------------------------------


def no_master_table() -> dict[str, doctor.SshResult]:
    """The healthy cluster, but with nobody having run `relay connect`."""
    table = healthy_ssh_table()
    # What ssh -O check actually says when the socket is not there.
    table["-O check"] = ssh_rc(255, stderr="Control socket connect(...): No such file")
    return table


def test_master_alive_is_ok(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    result = by_name(doctor.run_checks())["master"]

    assert result.status == doctor.OK
    assert "relay's SSH master is alive" in result.detail


def test_master_absent_warns_and_says_to_run_relay_connect(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, no_master_table())

    result = by_name(doctor.run_checks())["master"]

    assert result.status == doctor.WARN  # not an error: `relay connect` fixes it
    assert "No live SSH master for testcluster" in result.detail
    assert "relay connect" in result.detail
    # A missing master must not fail the run as a whole.
    assert doctor.worst_status([result]) == 0


def test_master_runs_before_the_ssh_round_trip(monkeypatch):
    """Order matters: the ssh check reads the master check's answer."""
    write_slurm_config()
    calls: list[SshCall] = []
    install_ssh(monkeypatch, healthy_ssh_table(), calls=calls)

    doctor.run_checks()

    commands = commands_of(calls)
    assert commands.index("-O check") < commands.index("true")


def test_failed_ssh_points_at_relay_connect_when_no_master_is_up(monkeypatch):
    """With no master, ssh fails fast (BatchMode) -- and we can name the cause."""
    write_slurm_config()
    table = no_master_table()
    table["true"] = ssh_rc(255, stderr="Permission denied (publickey,keyboard-interactive).")
    install_ssh(monkeypatch, table)

    checks = by_name(doctor.run_checks())

    # The round trip still ran; BatchMode makes it fail rather than hang.
    assert checks["ssh"].status == doctor.ERROR
    assert "relay connect" in checks["ssh"].detail
    # ssh prints its own reason last, and that line is what names the problem.
    assert "Permission denied (publickey,keyboard-interactive)." in checks["ssh"].detail
    assert checks["master"].status == doctor.WARN


def test_failed_ssh_does_not_blame_the_master_when_one_is_live(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["true"] = ssh_transport_failure("Connection timed out")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["ssh"]

    assert result.status == doctor.ERROR
    assert "relay connect" not in result.detail


# --------------------------------------------------------------------------
# Checks 4 and 5: remote python and sbatch
# --------------------------------------------------------------------------


def test_remote_python_reports_its_version(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    checks = by_name(doctor.run_checks())

    assert checks["remote_python"].status == doctor.OK
    assert "Python 3.11.9" in checks["remote_python"].detail


def test_missing_remote_python_is_an_error(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["--version"] = ssh_rc(127, stderr="bash: python3: command not found")
    install_ssh(monkeypatch, table)

    checks = by_name(doctor.run_checks())

    assert checks["remote_python"].status == doctor.ERROR
    assert "remote_python" in checks["remote_python"].detail


def test_sbatch_present_is_ok_and_missing_is_an_error(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())
    assert by_name(doctor.run_checks())["sbatch"].status == doctor.OK

    table = healthy_ssh_table()
    table["command -v sbatch"] = ssh_rc(1)
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["sbatch"]
    assert result.status == doctor.ERROR
    assert "PATH" in result.detail


# --------------------------------------------------------------------------
# Check 6: the Slurm account
# --------------------------------------------------------------------------


def test_configured_account_found_in_associations(monkeypatch):
    write_slurm_config(account="mlopt")
    install_ssh(monkeypatch, healthy_ssh_table())

    assert by_name(doctor.run_checks())["slurm_account"].status == doctor.OK


def test_account_mismatch_warns_and_lists_what_was_found(monkeypatch):
    write_slurm_config(account="wrong-account")
    install_ssh(monkeypatch, healthy_ssh_table())

    result = by_name(doctor.run_checks())["slurm_account"]
    assert result.status == doctor.WARN
    assert "wrong-account" in result.detail
    assert "mlopt" in result.detail


def test_account_check_uses_remote_whoami_not_a_local_username(monkeypatch):
    write_slurm_config()
    calls: list[SshCall] = []
    install_ssh(monkeypatch, healthy_ssh_table(), calls=calls)

    doctor.run_checks()

    account_commands = commands_of(calls, "sacctmgr")
    assert account_commands, "the account check never ran"
    # $(whoami) is expanded by the remote shell; nothing local is baked in.
    assert "$(whoami)" in account_commands[0]


def test_sacctmgr_failure_warns_rather_than_errors(monkeypatch):
    """Not every cluster has sacctmgr; being unable to verify is not a failure."""
    write_slurm_config()
    table = healthy_ssh_table()
    table["sacctmgr"] = ssh_rc(127, stderr="sacctmgr: command not found")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["slurm_account"]
    assert result.status == doctor.WARN


# --------------------------------------------------------------------------
# Check 7: the partition
# --------------------------------------------------------------------------


def test_partition_reports_that_it_is_preemptible(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    result = by_name(doctor.run_checks())["partition"]
    assert result.status == doctor.OK
    assert "preempted" in result.detail
    assert "PreemptMode=REQUEUE" in result.detail


def test_partition_detail_lists_the_four_fields_the_follow_ups_use(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    detail = by_name(doctor.run_checks())["partition"].detail

    assert "PreemptMode=REQUEUE" in detail
    assert "GraceTime=0" in detail
    assert "MaxTime=UNLIMITED" in detail
    assert "DefaultTime=01:00:00" in detail


def test_partition_reports_unknown_for_fields_scontrol_did_not_print(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, partition_table("PartitionName=gpu-a40 State=UP"))

    detail = by_name(doctor.run_checks())["partition"].detail

    assert "PreemptMode=unknown" in detail
    assert "GraceTime=unknown" in detail


def test_partition_with_preemption_off_says_so(monkeypatch):
    write_slurm_config()
    install_ssh(
        monkeypatch,
        partition_table(
            "PartitionName=gpu-a40 State=UP PreemptMode=OFF GraceTime=30 "
            "MaxTime=UNLIMITED DefaultTime=01:00:00"
        ),
    )

    result = by_name(doctor.run_checks())["partition"]
    assert result.status == doctor.OK
    assert "preemption is off" in result.detail


def test_unknown_partition_is_an_error(monkeypatch):
    write_slurm_config(partition="typo-partition")
    table = healthy_ssh_table()
    table["scontrol"] = ssh_rc(1, stderr="Partition typo-partition not found")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["partition"]
    assert result.status == doctor.ERROR
    assert "typo-partition" in result.detail


def test_down_partition_warns(monkeypatch):
    write_slurm_config()
    install_ssh(
        monkeypatch,
        partition_table("PartitionName=gpu-a40 State=DOWN PreemptMode=REQUEUE GraceTime=20"),
    )

    result = by_name(doctor.run_checks())["partition"]
    assert result.status == doctor.WARN
    assert "DOWN" in result.detail


def test_partition_derived_checks_skip_when_the_partition_is_unknown(monkeypatch):
    write_slurm_config(partition="typo-partition")
    table = healthy_ssh_table()
    table["scontrol"] = ssh_rc(1, stderr="Partition typo-partition not found")
    install_ssh(monkeypatch, table)

    checks = by_name(doctor.run_checks())

    assert checks["partition"].status == doctor.ERROR
    for name in PARTITION_DERIVED:
        assert checks[name].status == doctor.SKIP, name
        assert "partition check" in checks[name].detail


def test_partition_derived_checks_still_run_for_a_down_partition(monkeypatch):
    """DOWN is a warning, and the settings it printed are still true."""
    write_slurm_config()
    install_ssh(
        monkeypatch,
        partition_table(
            "PartitionName=gpu-a40 State=DOWN PreemptMode=REQUEUE GraceTime=20 "
            "MaxTime=UNLIMITED DefaultTime=01:00:00"
        ),
    )

    checks = by_name(doctor.run_checks())

    for name in PARTITION_DERIVED:
        assert checks[name].status != doctor.SKIP, name


# --------------------------------------------------------------------------
# Check 8: requeue_on_term against the partition's PreemptMode
# --------------------------------------------------------------------------

PREEMPTIBLE = (
    "PartitionName=gpu-a40 State=UP PreemptMode=REQUEUE GraceTime=20 "
    "MaxTime=UNLIMITED DefaultTime=01:00:00"
)
NOT_PREEMPTIBLE = (
    "PartitionName=gpu-a40 State=UP PreemptMode=OFF GraceTime=20 "
    "MaxTime=UNLIMITED DefaultTime=01:00:00"
)


def test_always_on_a_requeueing_partition_warns_about_double_requeue(monkeypatch):
    write_slurm_config(requeue_on_term="always")
    install_ssh(monkeypatch, partition_table(PREEMPTIBLE))

    result = by_name(doctor.run_checks())["requeue_policy"]

    assert result.status == doctor.WARN
    assert "Slurm already requeues preempted jobs on this partition" in result.detail
    assert "set requeue_on_term: auto or never" in result.detail


def test_never_on_a_partition_nothing_requeues_warns_about_lost_runs(monkeypatch):
    write_slurm_config(requeue_on_term="never")
    install_ssh(monkeypatch, partition_table(NOT_PREEMPTIBLE))

    result = by_name(doctor.run_checks())["requeue_policy"]

    assert result.status == doctor.WARN
    assert "preempted runs will not come back" in result.detail
    assert "set requeue_on_term: auto or always" in result.detail


def test_auto_says_which_way_it_resolves_on_a_requeueing_partition(monkeypatch):
    """Klone's case: Slurm requeues, so the sidecar must not."""
    write_slurm_config(requeue_on_term="auto")
    install_ssh(monkeypatch, partition_table(PREEMPTIBLE))

    result = by_name(doctor.run_checks())["requeue_policy"]

    assert result.status == doctor.OK
    assert "never" in result.detail


def test_auto_resolves_to_always_where_nothing_else_requeues(monkeypatch):
    write_slurm_config(requeue_on_term="auto")
    install_ssh(monkeypatch, partition_table(NOT_PREEMPTIBLE))

    result = by_name(doctor.run_checks())["requeue_policy"]

    assert result.status == doctor.OK
    assert "always" in result.detail


def test_the_safe_combinations_are_ok(monkeypatch):
    """`never` where Slurm requeues, and `always` where nothing does."""
    write_slurm_config(requeue_on_term="never")
    install_ssh(monkeypatch, partition_table(PREEMPTIBLE))
    assert by_name(doctor.run_checks())["requeue_policy"].status == doctor.OK

    write_slurm_config(requeue_on_term="always")
    install_ssh(monkeypatch, partition_table(NOT_PREEMPTIBLE))
    assert by_name(doctor.run_checks())["requeue_policy"].status == doctor.OK


# --------------------------------------------------------------------------
# Check 9: the preemption window
# --------------------------------------------------------------------------


def config_commands(calls: list[SshCall]) -> list[str]:
    return commands_of(calls, "show config")


def test_grace_time_zero_falls_back_to_cluster_kill_wait(monkeypatch):
    """Klone's shape: GraceTime=0, KillWait=10, so the window is 10 seconds."""
    write_slurm_config(preempt_grace=10)
    calls: list[SshCall] = []
    install_ssh(monkeypatch, healthy_ssh_table(), calls=calls)

    result = by_name(doctor.run_checks())["preempt_window"]

    assert result.status == doctor.OK
    assert "effective preemption window 10s" in result.detail
    assert "GraceTime=0" in result.detail
    assert "KillWait=10" in result.detail
    assert len(config_commands(calls)) == 1


def test_preempt_grace_longer_than_the_window_warns_with_the_value_to_set(monkeypatch):
    write_slurm_config(preempt_grace=30)
    install_ssh(monkeypatch, healthy_ssh_table())

    result = by_name(doctor.run_checks())["preempt_window"]

    assert result.status == doctor.WARN
    assert "the cluster only gives 10 seconds" in result.detail
    assert "set preempt_grace: 10" in result.detail


def test_a_real_grace_time_wins_and_kill_wait_is_never_asked_for(monkeypatch):
    """One fewer SSH round trip when the partition already answered."""
    write_slurm_config(preempt_grace=10)
    calls: list[SshCall] = []
    install_ssh(monkeypatch, partition_table(PREEMPTIBLE), calls=calls)

    result = by_name(doctor.run_checks())["preempt_window"]

    assert result.status == doctor.OK
    assert "effective preemption window 20s" in result.detail
    assert config_commands(calls) == []


def test_unreadable_kill_wait_warns_rather_than_guessing(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["show config"] = ssh_rc(1, stderr="scontrol: error: Unable to contact slurm")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["preempt_window"]

    assert result.status == doctor.WARN
    assert "Could not work out the preemption window" in result.detail


def test_kill_wait_parser_handles_both_spacings():
    assert doctor.parse_kill_wait("KillWait                = 10 sec") == 10
    assert doctor.parse_kill_wait("KillWait=30") == 30
    assert doctor.parse_kill_wait("killwait = 45 sec\n") == 45
    assert doctor.parse_kill_wait("PreemptMode = REQUEUE\n") is None
    assert doctor.parse_kill_wait("") is None


# --------------------------------------------------------------------------
# Check 10: the time limit against MaxTime
# --------------------------------------------------------------------------

CAPPED_AT_FOUR_HOURS = (
    "PartitionName=gpu-a40 State=UP PreemptMode=REQUEUE GraceTime=20 "
    "MaxTime=04:00:00 DefaultTime=01:00:00"
)


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("30", 1800),  # MM: minutes, not seconds
        ("30:00", 1800),  # MM:SS
        ("04:00:00", 14400),  # HH:MM:SS
        ("2-00", 172800),  # D-HH
        ("2-12", 216000),  # D-HH
        ("1-00:30", 88200),  # D-HH:MM
        ("1-00:00:30", 86430),  # D-HH:MM:SS
        ("UNLIMITED", None),
        ("INFINITE", None),
        ("", None),
        (None, None),
        ("not-a-time", None),
    ],
)
def test_slurm_time_to_seconds(text, seconds):
    assert doctor.slurm_time_to_seconds(text) == seconds


def test_time_limit_over_max_time_warns(monkeypatch):
    write_slurm_config(time="12:00:00")
    install_ssh(monkeypatch, partition_table(CAPPED_AT_FOUR_HOURS))

    result = by_name(doctor.run_checks())["time_limit"]

    assert result.status == doctor.WARN
    assert "MaxTime=04:00:00" in result.detail
    assert "12:00:00" in result.detail


def test_time_limit_within_max_time_is_ok(monkeypatch):
    write_slurm_config(time="02:00:00")
    install_ssh(monkeypatch, partition_table(CAPPED_AT_FOUR_HOURS))

    result = by_name(doctor.run_checks())["time_limit"]

    assert result.status == doctor.OK
    assert "02:00:00" in result.detail


def test_no_time_limit_at_all_warns_and_names_the_default(monkeypatch):
    write_slurm_config(time=None)
    install_ssh(monkeypatch, healthy_ssh_table())

    result = by_name(doctor.run_checks())["time_limit"]

    assert result.status == doctor.WARN
    assert "every submit will need --time" in result.detail
    assert "DefaultTime here is 01:00:00" in result.detail


def test_unlimited_max_time_cannot_be_exceeded(monkeypatch):
    """Klone's partition-level answer: nothing to compare against."""
    write_slurm_config(time="2-00")
    install_ssh(monkeypatch, healthy_ssh_table())

    result = by_name(doctor.run_checks())["time_limit"]

    assert result.status == doctor.OK
    assert "UNLIMITED" in result.detail


# --------------------------------------------------------------------------
# Check 11: remote_root
# --------------------------------------------------------------------------


def test_remote_root_parses_free_space_from_df(monkeypatch):
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    result = by_name(doctor.run_checks())["remote_root"]
    assert result.status == doctor.OK
    assert "writable" in result.detail
    assert "200.0 GB" in result.detail


def test_remote_root_low_space_warns(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["RELAY_NO_DIR"] = ssh_ok(f"{doctor._ROOT_OK}\n{DF_TIGHT}\n")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["remote_root"]
    assert result.status == doctor.WARN
    assert "5.0 GB" in result.detail


def test_remote_root_missing_directory_is_an_error(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["RELAY_NO_DIR"] = ssh_ok(f"{doctor._NO_DIR}\n")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["remote_root"]
    assert result.status == doctor.ERROR
    assert "mkdir -p" in result.detail


def test_remote_root_not_writable_is_an_error(monkeypatch):
    write_slurm_config()
    table = healthy_ssh_table()
    table["RELAY_NO_DIR"] = ssh_ok(f"{doctor._NO_WRITE}\n")
    install_ssh(monkeypatch, table)

    result = by_name(doctor.run_checks())["remote_root"]
    assert result.status == doctor.ERROR
    assert "permissions" in result.detail


def test_remote_root_probe_quotes_the_path(monkeypatch):
    """A path with a space must survive being re-parsed by the remote shell."""
    write_slurm_config(remote_root="/mmfs1/gscratch/my lab/relay")
    calls: list[SshCall] = []
    install_ssh(monkeypatch, healthy_ssh_table(), calls=calls)

    doctor.run_checks()

    probes = commands_of(calls, doctor._NO_DIR)
    assert probes
    assert "'/mmfs1/gscratch/my lab/relay'" in probes[0]


def test_df_parser_ignores_junk_lines():
    assert doctor._parse_df_available_gb("nonsense\n") is None
    assert doctor._parse_df_available_gb(f"Filesystem blah\n{DF_ROOMY}\n") == pytest.approx(
        200.0, rel=1e-3
    )


# --------------------------------------------------------------------------
# Check 9: the database
# --------------------------------------------------------------------------


def test_missing_database_is_a_warning_and_is_not_created():
    write_config("backend: local\n")

    result = by_name(doctor.run_checks())["database"]

    assert result.status == doctor.WARN
    assert "relay daemon" in result.detail
    # A health check must not create the thing it is checking for.
    assert not cfg.db_path().exists()


def test_real_database_is_reported_as_wal():
    write_config("backend: local\n")
    store = Store(cfg.db_path())
    store.close()

    result = by_name(doctor.run_checks())["database"]

    assert result.status == doctor.OK
    assert "WAL" in result.detail


def test_database_left_in_delete_mode_is_switched_back_to_wal():
    """Opening a Store sets journal_mode=WAL, so this check also repairs it.

    Journal mode is a property of the *file* and persists, so a database left
    in DELETE mode by some other tool comes back to WAL the moment doctor
    opens it. The check therefore reports ok -- and the situation really is
    fixed, which is the useful outcome.
    """
    write_config("backend: local\n")
    store = Store(cfg.db_path())
    store.conn.execute("PRAGMA journal_mode=DELETE")
    assert store.journal_mode() == "delete"
    store.close()

    result = by_name(doctor.run_checks())["database"]

    assert result.status == doctor.OK
    reopened = Store(cfg.db_path())
    assert reopened.journal_mode() == "wal"
    reopened.close()


def test_database_that_refuses_wal_is_an_error(monkeypatch):
    """The branch that survives: a file that cannot be put into WAL at all.

    SQLite silently declines WAL on some network filesystems (it needs shared
    memory), so journal_mode can read back as something else even after the
    pragma. Simulated here, because producing a real one needs an NFS mount.
    """
    write_config("backend: local\n")
    Store(cfg.db_path()).close()
    monkeypatch.setattr(Store, "journal_mode", lambda self: "delete")

    result = by_name(doctor.run_checks())["database"]

    assert result.status == doctor.ERROR
    assert "WAL" in result.detail


def test_corrupt_database_is_an_error_not_a_crash():
    write_config("backend: local\n")
    path = cfg.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is definitely not a sqlite file" * 10)

    result = by_name(doctor.run_checks())["database"]

    assert result.status == doctor.ERROR


# --------------------------------------------------------------------------
# Check 13: the usage table
# --------------------------------------------------------------------------


def test_usage_table_skips_when_there_is_no_database():
    write_config("backend: local\n")

    result = by_name(doctor.run_checks())["usage_table"]

    assert result.status == doctor.SKIP
    assert "database" in result.detail


def test_usage_table_warns_when_usage_has_never_been_fetched():
    write_config("backend: local\n")
    Store(cfg.db_path()).close()

    result = by_name(doctor.run_checks())["usage_table"]

    assert result.status == doctor.WARN
    assert "never been fetched" in result.detail
    assert "every 5 minutes" in result.detail


def test_usage_table_reports_how_long_ago_usage_was_fetched():
    write_config("backend: local\n")
    store = Store(cfg.db_path())
    store.set_daemon_state(last_usage_fetch_ts=utc_now_iso())
    store.close()

    result = by_name(doctor.run_checks())["usage_table"]

    assert result.status == doctor.OK
    assert "usage last fetched" in result.detail
    assert "ago" in result.detail


def test_usage_table_reports_a_stale_fetch_with_its_age():
    write_config("backend: local\n")
    store = Store(cfg.db_path())
    store.set_daemon_state(last_usage_fetch_ts="2020-01-01T00:00:00.000Z")
    store.close()

    result = by_name(doctor.run_checks())["usage_table"]

    # Old is still ok: usage is not live data, and the daemon check is what
    # says whether anything is running. This line only reports the age.
    assert result.status == doctor.OK
    assert "days ago" in result.detail


# --------------------------------------------------------------------------
# Check 14: the daemon lock and the last sync
# --------------------------------------------------------------------------


def test_no_lock_file_means_the_daemon_has_never_run():
    write_config("backend: local\n")

    result = by_name(doctor.run_checks())["daemon"]

    assert result.status == doctor.WARN
    assert "never run" in result.detail
    assert not cfg.daemon_lock_path().exists()  # no droppings left behind


def test_free_lock_means_no_daemon_is_running():
    write_config("backend: local\n")
    lock = cfg.daemon_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()

    result = by_name(doctor.run_checks())["daemon"]

    assert result.status == doctor.WARN
    assert "No daemon is running" in result.detail


def test_held_lock_means_the_daemon_is_running():
    """Hold the lock on one file object and let doctor open its own.

    flock locks belong to the *open file description*, not to the process, so
    two separate open() calls conflict even inside a single process. (POSIX
    fcntl/lockf record locks are per-process and would silently let the second
    attempt succeed -- which is exactly why daemon.py uses flock.)
    """
    write_config("backend: local\n")
    lock = cfg.daemon_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()

    holder = open(lock, "a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        result = by_name(doctor.run_checks())["daemon"]

        assert result.status == doctor.OK
        assert "daemon is running" in result.detail
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_doctor_releases_the_probe_lock_it_takes():
    """After reporting "not running", the lock must be free for a real daemon."""
    write_config("backend: local\n")
    lock = cfg.daemon_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()

    assert doctor._check_daemon().status == doctor.WARN

    after = open(lock, "a+")
    try:
        fcntl.flock(after.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
    finally:
        fcntl.flock(after.fileno(), fcntl.LOCK_UN)
        after.close()


def test_last_sync_skips_when_there_is_no_database():
    write_config("backend: local\n")

    result = by_name(doctor.run_checks())["last_sync"]

    assert result.status == doctor.SKIP
    assert "database" in result.detail


def test_last_sync_warns_when_the_daemon_never_completed_a_cycle():
    write_config("backend: local\n")
    Store(cfg.db_path()).close()

    result = by_name(doctor.run_checks())["last_sync"]

    assert result.status == doctor.WARN
    assert "never" in result.detail


def test_recent_sync_is_ok():
    write_config("backend: local\n")
    store = Store(cfg.db_path())
    store.mark_synced()
    store.close()

    result = by_name(doctor.run_checks())["last_sync"]

    assert result.status == doctor.OK
    assert "synced" in result.detail


def test_old_sync_warns():
    write_config("backend: local\n")
    store = Store(cfg.db_path())
    store.mark_synced("2020-01-01T00:00:00.000Z")
    store.close()

    result = by_name(doctor.run_checks())["last_sync"]

    assert result.status == doctor.WARN
    assert "2020-01-01" in result.detail


# --------------------------------------------------------------------------
# worst_status
# --------------------------------------------------------------------------


def test_worst_status_is_zero_for_ok_and_warn_and_skip():
    results = [
        doctor.CheckResult("a", doctor.OK, "fine"),
        doctor.CheckResult("b", doctor.WARN, "not ideal"),
        doctor.CheckResult("c", doctor.SKIP, "did not run"),
    ]
    assert doctor.worst_status(results) == 0
    assert doctor.worst_status([]) == 0


def test_worst_status_is_one_when_anything_errored():
    results = [
        doctor.CheckResult("a", doctor.OK, "fine"),
        doctor.CheckResult("b", doctor.ERROR, "broken"),
    ]
    assert doctor.worst_status(results) == 1


def test_a_fully_healthy_slurm_run_exits_zero(monkeypatch):
    """The whole thing, end to end, against a Klone-shaped healthy cluster.

    Every check ok, including the four that spec change round 1 added: a live
    master, `requeue_on_term: auto` resolving correctly against
    PreemptMode=REQUEUE, `preempt_grace: 10` fitting the 10 second window
    KillWait gives, and a time limit inside MaxTime.

    `control_dir` is ok here too, by the "created it" path rather than the
    "found it" one: the test HOME is fresh, so there is no `~/.relay` yet, and
    creating one is a normal healthy outcome rather than a complaint.
    """
    write_slurm_config()
    install_ssh(monkeypatch, healthy_ssh_table())

    store = Store(cfg.db_path())
    store.mark_synced()
    store.set_daemon_state(last_usage_fetch_ts=utc_now_iso())
    store.close()

    lock = cfg.daemon_lock_path()
    lock.touch()
    holder = open(lock, "a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        results = doctor.run_checks()

        assert [r.status for r in results] == [doctor.OK] * len(ALL_CHECK_NAMES)
        assert doctor.worst_status(results) == 0
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_no_remote_call_waits_less_than_the_configured_ssh_timeout(monkeypatch):
    """One number for the whole of relay, and every check must use it.

    This is the regression guard for the shape of the original bug: doctor kept
    a 15 second literal of its own, tighter than anything else in relay, and a
    login node that answered in 4.8 seconds through a live master still failed
    the check outright -- which skipped all ten remote checks behind it. A
    per-check literal is the only way that can come back, so assert on every
    recorded call rather than on the two that happen to be interesting.
    """
    write_slurm_config()
    calls: list[SshCall] = []
    install_ssh(monkeypatch, healthy_ssh_table(), calls=calls)

    doctor.run_checks()

    expected = ssh_mod.DEFAULT_SSH_TIMEOUT
    assert expected >= 30.0  # the floor the Klone measurement set
    assert calls, "no remote call was recorded"
    for call in calls:
        assert call.timeout >= expected, call

    # The round trip gets double, because the same number is the threshold
    # above which it *warns*. Cut it off at the threshold and a slow-but-alive
    # login node could only ever be reported as an error.
    round_trips = [call for call in calls if call.command == "true"]
    assert len(round_trips) == 1
    assert round_trips[0].timeout >= 2 * expected

    # Including the local socket probe: it answers in milliseconds, but nothing
    # in relay waits less than the shared value.
    master_checks = [call for call in calls if call.command == "-O check"]
    assert master_checks and master_checks[0].timeout >= expected


def test_a_configured_ssh_timeout_is_what_every_call_uses(monkeypatch):
    """Raising `ssh_timeout` has to reach the checks, not just the daemon."""
    if "ssh_timeout" not in {f.name for f in dataclasses.fields(cfg.Config)}:
        pytest.skip("Config.ssh_timeout has not landed yet")

    write_config(
        "backend: slurm\n"
        "ssh_alias: testcluster\n"
        "remote_root: /mmfs1/gscratch/mlopt/you/relay\n"
        "remote_python: python3\n"
        "ssh_timeout: 90\n"
        "slurm:\n"
        "  account: mlopt\n"
        "  partition: gpu-a40\n"
        '  time: "04:00:00"\n'
    )
    calls: list[SshCall] = []
    install_ssh(monkeypatch, healthy_ssh_table(), calls=calls)

    doctor.run_checks()

    assert calls
    for call in calls:
        assert call.timeout >= 90.0, call
    assert [c for c in calls if c.command == "true"][0].timeout >= 180.0


def test_a_config_without_ssh_timeout_falls_back_to_the_shared_default():
    """`getattr`, so a hand-built Config still works and is never tighter."""

    class Bare:
        pass

    assert doctor._ssh_timeout(Bare()) == ssh_mod.DEFAULT_SSH_TIMEOUT
    # Nonsense is a fallback too: a health check may not be the thing that dies.
    for bad in (None, 0, -5, "soon"):
        holder = Bare()
        holder.ssh_timeout = bad
        assert doctor._ssh_timeout(holder) == ssh_mod.DEFAULT_SSH_TIMEOUT


def test_results_are_json_friendly(monkeypatch):
    write_config("backend: local\n")

    for result in doctor.run_checks():
        as_dict = result.as_dict()
        assert set(as_dict) == {"name", "status", "detail", "duration_ms"}
        assert isinstance(as_dict["detail"], str)
        assert as_dict["duration_ms"] is None or isinstance(as_dict["duration_ms"], float)
