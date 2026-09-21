"""Tests for the sidecar.

The sidecar is a process, not a library, so most of these run it the way the
cluster will: `python3 -m relay.sidecar --run ... --dir ... -- <command>`, with a
fake training script written into tmp_path. Testing the real subprocess is the
only way to cover the parts that matter -- pipe readers, signal handling, exit
codes -- because none of those exist inside a single-process function call.
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from relay import sidecar as sidecar_module

SIDECAR_PATH = Path(sidecar_module.__file__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def read_events(run_dir: Path) -> list[dict]:
    """Parse events.jsonl into a list of dicts."""
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def of_type(events: list[dict], type_: str) -> list[dict]:
    return [event for event in events if event["type"] == type_]


def write_script(path: Path, body: str) -> Path:
    """Write a fake training script and return its path."""
    path.write_text(body, encoding="utf-8")
    return path


def sidecar_argv(
    run_dir: Path,
    command: list[str],
    run: str = "vr_s0_test",
    grace: float = 10.0,
    heartbeat: float = 3600.0,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "relay.sidecar",
        "--run",
        run,
        "--dir",
        str(run_dir),
        "--grace",
        str(grace),
        "--heartbeat",
        str(heartbeat),
        "--",
        *command,
    ]


def run_sidecar(
    run_dir: Path,
    command: list[str],
    run: str = "vr_s0_test",
    grace: float = 10.0,
    heartbeat: float = 3600.0,
    timeout: float = 60.0,
    env: dict | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run the sidecar to completion as a subprocess.

    The heartbeat defaults to an hour so it never fires in tests that do not
    care about it; the tests that do care pass something small.

    `cwd` is the job's working directory, which is what the resume glob is
    resolved against -- so the resume tests set it the way sbatch would.
    """
    return subprocess.run(
        sidecar_argv(run_dir, command, run=run, grace=grace, heartbeat=heartbeat),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_env(env),
        cwd=None if cwd is None else str(cwd),
    )


def child_env(extra: dict | None = None) -> dict:
    """This process's environment plus `extra`.

    Always a full copy, never just `extra`: the sidecar is started with
    `python -m relay.sidecar`, which needs PATH, PYTHONPATH and the virtualenv
    to still be there.
    """
    env = dict(os.environ)
    # Tests must not inherit a real Slurm job's identity from whatever shell
    # they were started in -- that would make the requeue guard fire for real.
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_RESTART_COUNT", None)
    if extra:
        env.update(extra)
    return env


def write_sidecar_config(run_dir: Path, **values) -> None:
    """Write the `sidecar.json` a backend would have written at submit time."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sidecar.json").write_text(json.dumps(values), encoding="utf-8")


def fake_scontrol(tmp_path: Path) -> tuple[Path, Path]:
    """Put a fake `scontrol` on PATH and return `(bin dir, call log)`.

    The sidecar runs `scontrol requeue $SLURM_JOB_ID` by PATH lookup, exactly
    as it would on a cluster, so a shell script earlier in PATH is all it takes
    to observe whether the requeue happened -- and how many times. Counting the
    calls is the point: the bug this whole change exists to fix was a requeue
    that should not have happened at all.
    """
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "scontrol_calls.log"
    script = bin_dir / "scontrol"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{log}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bin_dir, log


def scontrol_calls(log: Path) -> list[str]:
    """Every argument line the fake `scontrol` was invoked with."""
    try:
        return [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    except FileNotFoundError:
        return []


def scontrol_env(tmp_path: Path, job_id: str = "424242") -> tuple[dict, Path]:
    """Environment that makes a self-requeue both possible and observable."""
    bin_dir, log = fake_scontrol(tmp_path)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        # The sidecar refuses to requeue without this, and rightly: on a laptop
        # there is no job to put back in a queue.
        "SLURM_JOB_ID": job_id,
    }
    return env, log


def sidecar_pid(run_dir: Path) -> int:
    """The sidecar's own PID, which it records in `run_started`.

    Needed because these tests signal the sidecar the way Slurm's
    `--signal=B:` does -- the batch process only, not the whole process group,
    so the training script gets the forwarded SIGTERM and nothing else.
    """
    started = of_type(read_events(run_dir), "run_started")
    assert started, "the sidecar has not written run_started yet"
    return int(started[-1]["payload"]["pid"])


def wait_for(predicate, timeout: float = 20.0, interval: float = 0.1) -> bool:
    """Poll until predicate() is true or the timeout expires.

    Timing-based tests are the flakiest kind, so nothing here sleeps a fixed
    amount and hopes. Timeouts are generous: a slow CI box taking ten seconds is
    normal, a hung sidecar never finishes at all.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# --------------------------------------------------------------------------
# The stdlib-only rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["sidecar.py", "relay_ckpt.py"])
def test_shipped_files_import_only_stdlib(filename):
    """Both files are copied to the cluster alone; neither can have dependencies.

    `sidecar.py` wraps the job and `relay_ckpt.py` is imported by the user's
    training script (the sidecar puts the run directory on its PYTHONPATH). Both
    land on a compute node with nothing but a bare Python, so neither may import
    anything that is not in the standard library -- and neither may import
    `relay`, which is not installed there.

    AST-parsing the imports is the enforcement CLAUDE.md asks for, and it is
    stronger than importing the module and hoping: this catches an import that
    only happens inside a function, or one added in a branch nobody exercises.
    """
    path = SIDECAR_PATH.parent / filename
    if not path.exists():
        pytest.skip(f"{filename} does not exist yet")

    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pytest.fail(
                    f"{filename} uses a relative import; it must be self-contained"
                )
            if node.module:
                imported.add(node.module.split(".")[0])

    # __future__ is not in stdlib_module_names but is always available.
    imported.discard("__future__")

    non_stdlib = sorted(imported - sys.stdlib_module_names)
    assert non_stdlib == [], f"{filename} imports non-stdlib modules: {non_stdlib}"
    assert "relay" not in imported


# --------------------------------------------------------------------------
# Line parsing (pure function, no subprocess needed)
# --------------------------------------------------------------------------


def test_parse_relay_line_metric():
    parsed = sidecar_module.parse_relay_line(
        '##relay## {"step": 1000, "loss": 0.412, "sps": 18400}'
    )
    assert parsed == ("metric", {"step": 1000, "values": {"loss": 0.412, "sps": 18400.0}})


def test_parse_relay_line_skips_non_numeric_and_bools():
    _, payload = sidecar_module.parse_relay_line(
        '##relay## {"step": 5, "loss": 0.1, "note": "warmup", "converged": true}'
    )
    assert payload == {"step": 5, "values": {"loss": 0.1}}


def test_parse_relay_line_checkpoint():
    parsed = sidecar_module.parse_relay_line(
        '##relay## {"checkpoint": "ckpt/step_1000.pt", "step": 1000}'
    )
    assert parsed == ("checkpoint", {"path": "ckpt/step_1000.pt", "step": 1000})


@pytest.mark.parametrize(
    "line",
    [
        "ordinary log line",
        "##relay##no-space-so-not-a-marker",
        "##relay## not json at all",
        "##relay## [1, 2, 3]",
        '##relay## {"loss": 0.4}',  # no step
        '##relay## {"step": "ten", "loss": 0.4}',  # step is not a number
        '  ##relay## {"step": 1}',  # marker must start the line
    ],
)
def test_parse_relay_line_rejects(line):
    assert sidecar_module.parse_relay_line(line) is None


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


TRAINER = """\
import sys, time

print("starting up", flush=True)
for step in (100, 200, 300):
    print('##relay## {"step": %d, "loss": %f, "sps": 18400}' % (step, 1.0 / step),
          flush=True)
    print("did step", step, flush=True)
print("to stderr", file=sys.stderr, flush=True)
print('##relay## {"checkpoint": "ckpt/step_300.pt", "step": 300}', flush=True)
print("done", flush=True)
"""


def test_happy_path(tmp_path):
    script = write_script(tmp_path / "train.py", TRAINER)
    run_dir = tmp_path / "run"

    result = run_sidecar(run_dir, [sys.executable, str(script)])
    assert result.returncode == 0

    events = read_events(run_dir)

    # Envelope: seq starts at 1 and increases by exactly one each time.
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    for event in events:
        assert event["v"] == 2
        assert event["run"] == "vr_s0_test"
        assert event["ts"].endswith("Z")
        assert set(event) == {"v", "run", "seq", "ts", "type", "payload"}

    # run_started first, run_ended last.
    assert events[0]["type"] == "run_started"
    assert events[0]["payload"]["command"] == [sys.executable, str(script)]
    assert events[0]["payload"]["pid"] > 0
    assert events[0]["payload"]["hostname"]
    # First attempt, so SLURM_RESTART_COUNT is absent and attempt is 0.
    assert events[0]["payload"]["attempt"] == 0
    # Always present, null when there was nothing to resume from.
    assert events[0]["payload"]["resumed_from"] is None

    assert events[-1]["type"] == "run_ended"
    assert events[-1]["payload"] == {"exit_code": 0, "status": "completed"}

    # Metrics: numbers nested under "values", step alongside.
    metrics = of_type(events, "metric")
    assert [m["payload"]["step"] for m in metrics] == [100, 200, 300]
    assert metrics[0]["payload"]["values"]["sps"] == 18400.0
    assert metrics[0]["payload"]["values"]["loss"] == pytest.approx(0.01)

    # Checkpoints are observed, not written, by the sidecar.
    checkpoints = of_type(events, "checkpoint")
    assert len(checkpoints) == 1
    assert checkpoints[0]["payload"] == {"path": "ckpt/step_300.pt", "step": 300}

    # Everything else became a log event, tagged with its stream.
    logs = of_type(events, "log")
    stdout_lines = [
        log["payload"]["line"] for log in logs if log["payload"]["stream"] == "stdout"
    ]
    stderr_lines = [
        log["payload"]["line"] for log in logs if log["payload"]["stream"] == "stderr"
    ]
    assert "starting up" in stdout_lines
    assert "did step 200" in stdout_lines
    assert "done" in stdout_lines
    assert stderr_lines == ["to stderr"]

    # ...and was mirrored to the sidecar's own stdout/stderr, so Slurm's output
    # file still has it.
    assert "starting up" in result.stdout
    assert "to stderr" in result.stderr


def test_creates_run_directory(tmp_path):
    run_dir = tmp_path / "deeply" / "nested" / "run"
    result = run_sidecar(run_dir, [sys.executable, "-c", "print('hi')"])
    assert result.returncode == 0
    assert (run_dir / "events.jsonl").exists()


def test_heartbeat_carries_last_step(tmp_path):
    """A heartbeat reports the last step seen, so the daemon can spot a hang."""
    script = write_script(
        tmp_path / "train.py",
        'import time\n'
        'print(\'##relay## {"step": 42, "loss": 0.5}\', flush=True)\n'
        'time.sleep(1.2)\n',
    )
    run_dir = tmp_path / "run"

    result = run_sidecar(
        run_dir, [sys.executable, str(script)], heartbeat=0.2, timeout=60.0
    )
    assert result.returncode == 0

    heartbeats = of_type(read_events(run_dir), "heartbeat")
    assert heartbeats, "expected at least one heartbeat"
    assert heartbeats[-1]["payload"] == {"last_step": 42}


def test_heartbeat_last_step_is_null_before_any_metric(tmp_path):
    script = write_script(tmp_path / "train.py", "import time\ntime.sleep(1.2)\n")
    run_dir = tmp_path / "run"

    run_sidecar(run_dir, [sys.executable, str(script)], heartbeat=0.2)

    heartbeats = of_type(read_events(run_dir), "heartbeat")
    assert heartbeats
    assert heartbeats[0]["payload"] == {"last_step": None}


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------


def test_nonzero_exit_is_propagated(tmp_path):
    """The sidecar exits with the child's code, never a helpful 0."""
    script = write_script(
        tmp_path / "train.py",
        "import sys\nprint('crashing', flush=True)\nsys.exit(17)\n",
    )
    run_dir = tmp_path / "run"

    result = run_sidecar(run_dir, [sys.executable, str(script)])
    assert result.returncode == 17

    ended = of_type(read_events(run_dir), "run_ended")
    assert len(ended) == 1
    assert ended[0]["payload"] == {"exit_code": 17, "status": "failed"}


def test_missing_command_is_a_failed_run_not_a_crash(tmp_path):
    run_dir = tmp_path / "run"
    result = run_sidecar(run_dir, ["/nonexistent/binary/relay-test"])

    assert result.returncode == 127
    events = read_events(run_dir)
    assert events[0]["type"] == "run_started"
    assert events[-1]["payload"]["status"] == "failed"


def test_bad_usage_is_rejected(tmp_path):
    """No command after `--` is a usage error, not a silent empty run."""
    result = subprocess.run(
        [sys.executable, "-m", "relay.sidecar", "--run", "r", "--dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2  # argparse's usage-error code


# --------------------------------------------------------------------------
# Requeue continuation
# --------------------------------------------------------------------------


def test_seq_continues_across_attempts(tmp_path):
    """The second attempt must not restart seq at 1.

    If it did, its events would collide on the store's (run_id, seq) primary key
    and INSERT OR IGNORE would drop them silently -- the first attempt's history
    would quietly overwrite the second's.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    existing = [
        {"v": 1, "run": "r1", "seq": 1, "ts": "2026-09-16T00:00:00.000Z",
         "type": "run_started", "payload": {}},
        {"v": 1, "run": "r1", "seq": 7, "ts": "2026-09-16T00:00:01.000Z",
         "type": "metric", "payload": {"step": 900, "values": {"loss": 0.5}}},
        {"v": 1, "run": "r1", "seq": 4, "ts": "2026-09-16T00:00:02.000Z",
         "type": "log", "payload": {"stream": "stdout", "line": "out of order"}},
    ]
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in existing:
            handle.write(json.dumps(event) + "\n")

    result = run_sidecar(
        run_dir, [sys.executable, "-c", "print('resumed')"], run="r1"
    )
    assert result.returncode == 0

    events = read_events(run_dir)
    new_events = events[len(existing) :]
    # Highest existing seq is 7 (not the last line's 4), so the next one is 8.
    assert new_events[0]["type"] == "run_started"
    assert [event["seq"] for event in new_events] == list(
        range(8, 8 + len(new_events))
    )


def test_truncated_tail_does_not_block_startup(tmp_path):
    """A SIGKILL mid-write leaves half a line. Starting must still work.

    And the stump must not swallow our first event: the new attempt terminates
    the torn line before appending, so the damage stays one unparseable line
    instead of corrupting a real record too.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"v": 1, "run": "r1", "seq": 3, "ts": "2026-09-16T00:00:00.000Z",
                 "type": "log", "payload": {"stream": "stdout", "line": "x"}}
            )
            + "\n"
        )
        handle.write('{"v":1,"run":"r1","seq":4,"ts":"2026-09-1')  # truncated

    result = run_sidecar(run_dir, [sys.executable, "-c", "pass"], run="r1")
    assert result.returncode == 0

    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    # The stump is still there, still unparseable, and still on a line of its own.
    assert lines[1] == '{"v":1,"run":"r1","seq":4,"ts":"2026-09-1'
    with pytest.raises(ValueError):
        json.loads(lines[1])

    # Every other line parses, and the run continued from the highest
    # *parseable* seq (3), so the new events start at 4.
    good = [json.loads(line) for i, line in enumerate(lines) if i != 1]
    assert [event["seq"] for event in good[1:]] == list(range(4, 3 + len(good)))
    assert good[1]["type"] == "run_started"


def test_highest_seq_on_missing_file(tmp_path):
    assert sidecar_module.highest_seq(str(tmp_path / "nope.jsonl")) == 0


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
#
# Two signals, two meanings, and the difference is the bug this section exists
# to prevent. SIGUSR1 can only be Slurm's time-limit warning, so requeueing is
# always right. SIGTERM is preemption *or* `scancel`, and they are the same
# signal -- so the sidecar records `interrupted` and decides about requeue from
# facts that were on disk before the signal arrived.


SLEEPER = """\
import signal, sys, time

print("training", flush=True)
# No SIGTERM handler: this script is the "user did not implement checkpointing"
# case, so the sidecar's forwarded SIGTERM kills it outright.
time.sleep(300)
"""

# Ignores SIGTERM entirely, so the sidecar has to wait out its window and then
# SIGKILL. That is the only way to observe how long the window actually was.
STUBBORN = """\
import signal, time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("stubborn", flush=True)
time.sleep(300)
"""


def start_sidecar(
    run_dir: Path,
    command: list[str],
    run: str,
    env: dict | None = None,
    grace: float = 10.0,
) -> subprocess.Popen:
    """Start the sidecar in its own session and return the Popen.

    Its own session so that a signal sent to this test's process group does not
    reach it, and so the tests can choose precisely who gets signalled.
    """
    return subprocess.Popen(
        sidecar_argv(run_dir, command, run=run, grace=grace, heartbeat=3600.0),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(env),
        start_new_session=True,
    )


def wait_for_child_running(run_dir: Path, marker: str) -> None:
    """Block until the training script has printed its first line.

    Signalling before the child is up would test a different code path (the one
    where there is nothing to forward to) and would do it intermittently, which
    is the worst kind of test.
    """
    assert wait_for(
        lambda: (run_dir / "events.jsonl").exists()
        and any(
            event["type"] == "log" and event["payload"]["line"] == marker
            for event in read_events(run_dir)
        )
    ), "the child never produced its first log line"


def signal_and_wait(proc: subprocess.Popen, run_dir: Path, signum: int) -> int:
    """Send `signum` to the sidecar itself and wait for it to exit.

    To the sidecar process, not the process group: Slurm's `--signal=B:` form
    signals the batch process alone, and that is the case worth testing. If the
    group were signalled, the child would get its own copy and we could not
    tell the forward from the original.
    """
    try:
        os.kill(sidecar_pid(run_dir), signum)
        return proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_sigusr1_is_a_time_limit_and_self_requeues(tmp_path):
    """The unambiguous path: only the time-limit warning sends USR1."""
    script = write_script(tmp_path / "train.py", SLEEPER)
    run_dir = tmp_path / "run"
    env, calls = scontrol_env(tmp_path)

    proc = start_sidecar(run_dir, [sys.executable, str(script)], "vr_usr1", env=env)
    wait_for_child_running(run_dir, "training")
    returncode = signal_and_wait(proc, run_dir, signal.SIGUSR1)

    # Exit 0: the job did not fail, and a nonzero exit would stop Slurm from
    # requeueing it.
    assert returncode == 0

    events = read_events(run_dir)
    signals = of_type(events, "signal")
    assert signals[0]["payload"] == {"signal": "SIGUSR1", "reason": "time_limit"}

    ended = of_type(events, "run_ended")
    assert len(ended) == 1
    assert ended[0]["payload"] == {"exit_code": None, "status": "requeued"}
    # run_ended is the last event of the attempt, after the signal event.
    assert events[-1]["type"] == "run_ended"
    assert signals[0]["seq"] < ended[0]["seq"]

    # Exactly one requeue. Not zero (the job would never come back) and not two
    # (Slurm would run the job twice).
    assert scontrol_calls(calls) == ["requeue 424242"]


def test_sigterm_is_interrupted_and_does_not_requeue_by_default(tmp_path):
    """No sidecar.json at all: the conservative default is `never`.

    This is the bug fix in one test. The old sidecar treated every SIGTERM as a
    preemption and requeued, so `scancel` -- which is also a SIGTERM -- brought
    cancelled jobs back from the dead.
    """
    script = write_script(tmp_path / "train.py", SLEEPER)
    run_dir = tmp_path / "run"
    env, calls = scontrol_env(tmp_path)

    proc = start_sidecar(run_dir, [sys.executable, str(script)], "vr_term", env=env)
    wait_for_child_running(run_dir, "training")
    returncode = signal_and_wait(proc, run_dir, signal.SIGTERM)

    assert returncode == 0
    assert not (run_dir / "sidecar.json").exists(), "this test is about the default"

    events = read_events(run_dir)
    assert of_type(events, "signal")[0]["payload"] == {
        "signal": "SIGTERM",
        "reason": "term",
    }
    ended = of_type(events, "run_ended")
    assert len(ended) == 1
    assert ended[0]["payload"] == {"exit_code": None, "status": "interrupted"}
    assert scontrol_calls(calls) == []


def test_sigterm_with_requeue_always_requeues_once(tmp_path):
    """`requeue_on_term: always` is for a partition Slurm does not requeue for."""
    script = write_script(tmp_path / "train.py", SLEEPER)
    run_dir = tmp_path / "run"
    write_sidecar_config(run_dir, requeue_on_term="always", preempt_grace=10)
    env, calls = scontrol_env(tmp_path)

    proc = start_sidecar(run_dir, [sys.executable, str(script)], "vr_always", env=env)
    wait_for_child_running(run_dir, "training")
    assert signal_and_wait(proc, run_dir, signal.SIGTERM) == 0

    # Still `interrupted`: requeueing is what we did about it, not what happened.
    ended = of_type(read_events(run_dir), "run_ended")
    assert ended[0]["payload"]["status"] == "interrupted"
    assert scontrol_calls(calls) == ["requeue 424242"]


def test_cancel_requested_beats_requeue_always(tmp_path):
    """The user asked for this run to stop. Nothing may bring it back.

    `relay cancel` writes this file before sending the signal, which is the
    only ordering that works: the sidecar looks for it *after* its child is
    dead, but by then it has to already be there.
    """
    script = write_script(tmp_path / "train.py", SLEEPER)
    run_dir = tmp_path / "run"
    write_sidecar_config(run_dir, requeue_on_term="always", preempt_grace=10)
    (run_dir / "CANCEL_REQUESTED").touch()
    env, calls = scontrol_env(tmp_path)

    proc = start_sidecar(run_dir, [sys.executable, str(script)], "vr_cancel", env=env)
    wait_for_child_running(run_dir, "training")
    assert signal_and_wait(proc, run_dir, signal.SIGTERM) == 0

    ended = of_type(read_events(run_dir), "run_ended")
    assert ended[0]["payload"]["status"] == "interrupted"
    assert scontrol_calls(calls) == []


def test_sigusr1_ignores_cancel_requested(tmp_path):
    """A time-limit requeue is not a cancel, even if a cancel is pending.

    If both happen at once, the job goes back in the queue and the next attempt
    is cancelled there. Dropping the requeue instead would lose a run whose
    cancel may still be beaten by the scheduler.
    """
    script = write_script(tmp_path / "train.py", SLEEPER)
    run_dir = tmp_path / "run"
    write_sidecar_config(run_dir, requeue_on_term="never", preempt_grace=10)
    (run_dir / "CANCEL_REQUESTED").touch()
    env, calls = scontrol_env(tmp_path)

    proc = start_sidecar(run_dir, [sys.executable, str(script)], "vr_both", env=env)
    wait_for_child_running(run_dir, "training")
    assert signal_and_wait(proc, run_dir, signal.SIGUSR1) == 0

    assert of_type(read_events(run_dir), "run_ended")[0]["payload"]["status"] == (
        "requeued"
    )
    assert scontrol_calls(calls) == ["requeue 424242"]


def test_child_that_checkpoints_on_sigterm_gets_its_window(tmp_path):
    """The grace window is the point: the child must get time to save."""
    script = write_script(
        tmp_path / "train.py",
        """\
import signal, sys, time

def on_term(signum, frame):
    print('##relay## {"checkpoint": "ckpt/step_5.pt", "step": 5}', flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, on_term)
print('##relay## {"step": 5, "loss": 0.25}', flush=True)
print("training", flush=True)
time.sleep(300)
""",
    )
    run_dir = tmp_path / "run"
    write_sidecar_config(run_dir, requeue_on_term="never", preempt_grace=20)

    proc = start_sidecar(run_dir, [sys.executable, str(script)], "vr_ckpt")
    wait_for_child_running(run_dir, "training")
    assert signal_and_wait(proc, run_dir, signal.SIGTERM) == 0

    events = read_events(run_dir)
    checkpoints = of_type(events, "checkpoint")
    assert checkpoints, "the child's checkpoint line was lost during shutdown"
    assert checkpoints[0]["payload"] == {"path": "ckpt/step_5.pt", "step": 5}
    # And it landed before the ending, which is what joining the reader threads
    # before writing run_ended buys us.
    assert checkpoints[0]["seq"] < of_type(events, "run_ended")[0]["seq"]
    assert of_type(events, "run_ended")[0]["payload"]["status"] == "interrupted"


def test_preempt_grace_bounds_how_long_a_stubborn_child_survives(tmp_path):
    """A child that ignores SIGTERM is SIGKILLed when its window runs out.

    preempt_grace 3 leaves the child 3 - 2 = 1 second, which is the floor. The
    assertion is deliberately loose at the top end -- this is a real process on
    a possibly-loaded machine -- but tight enough to fail if the sidecar were
    using the 10-second default, or waiting forever.
    """
    script = write_script(tmp_path / "train.py", STUBBORN)
    run_dir = tmp_path / "run"
    write_sidecar_config(run_dir, requeue_on_term="never", preempt_grace=3)

    proc = start_sidecar(run_dir, [sys.executable, str(script)], "vr_stubborn")
    wait_for_child_running(run_dir, "stubborn")

    started = time.monotonic()
    assert signal_and_wait(proc, run_dir, signal.SIGTERM) == 0
    elapsed = time.monotonic() - started

    assert elapsed >= 0.5, "the child was killed without getting its window"
    assert elapsed < 8.0, f"waited {elapsed:.1f}s; the 1s window was not honoured"
    assert of_type(read_events(run_dir), "run_ended")[0]["payload"] == {
        "exit_code": None,
        "status": "interrupted",
    }


def test_requeue_is_skipped_without_slurm_job_id(tmp_path, monkeypatch):
    """`scontrol` does not exist on a laptop; the guard keeps local runs sane."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    sidecar = sidecar_module.Sidecar(run="r", run_dir=str(tmp_path), command=["true"])
    sidecar._requeue()  # must not raise, must not shell out


# --------------------------------------------------------------------------
# sidecar.json and the attempt counter
# --------------------------------------------------------------------------


def test_sidecar_config_defaults_when_the_file_is_absent(tmp_path):
    config = sidecar_module.load_sidecar_config(str(tmp_path))
    assert config.requeue_on_term == "never"
    assert config.preempt_grace == 10.0
    # No config file means no resume configured and no mtime floor.
    assert config.resume is None
    assert config.first_attempt_start == 0.0


@pytest.mark.parametrize(
    "contents",
    [
        "not json at all",
        "[1, 2, 3]",  # valid JSON, wrong shape
        "{}",
        '{"requeue_on_term": "auto"}',  # the backend should have resolved this
        '{"requeue_on_term": 7, "preempt_grace": "soon"}',
        '{"preempt_grace": -4}',
    ],
)
def test_sidecar_config_falls_back_to_defaults_on_nonsense(tmp_path, contents):
    """Never a reason to refuse to start a training job.

    `never` is the conservative default: a job that does not come back wastes a
    queue slot, while one that requeues when it should not can loop forever.
    """
    (tmp_path / "sidecar.json").write_text(contents, encoding="utf-8")
    assert sidecar_module.load_sidecar_config(str(tmp_path)) == (
        "never",
        10.0,
        0.0,
        None,
    )


def test_sidecar_config_is_read(tmp_path):
    write_sidecar_config(tmp_path, requeue_on_term="always", preempt_grace=45)
    config = sidecar_module.load_sidecar_config(str(tmp_path))
    assert config.requeue_on_term == "always"
    assert config.preempt_grace == 45.0
    # A v1-shaped file has neither of the new keys, and must still work: no
    # resume configured, no mtime floor.
    assert config.resume is None
    assert config.first_attempt_start == 0.0


def test_sidecar_config_reads_the_resume_block(tmp_path):
    write_sidecar_config(
        tmp_path,
        requeue_on_term="always",
        preempt_grace=10,
        first_attempt_start=1758412929.5,
        resume={"checkpoint_glob": "ckpt/*.pt", "arg": "--resume {path}"},
    )
    config = sidecar_module.load_sidecar_config(str(tmp_path))
    assert config.first_attempt_start == 1758412929.5
    assert config.resume == ("ckpt/*.pt", "--resume {path}")


@pytest.mark.parametrize(
    "resume",
    [
        None,
        "ckpt/*.pt",  # not an object
        {"checkpoint_glob": "ckpt/*.pt"},  # no arg
        {"arg": "--resume {path}"},  # no glob
        {"checkpoint_glob": "", "arg": "--resume {path}"},
        {"checkpoint_glob": "ckpt/*.pt", "arg": ""},
        # An arg with no placeholder would append a bare flag with no value and
        # break the training script in a way that looks like relay's fault.
        {"checkpoint_glob": "ckpt/*.pt", "arg": "--resume"},
        {"checkpoint_glob": 7, "arg": "--resume {path}"},
    ],
)
def test_nonsense_resume_blocks_switch_the_feature_off(tmp_path, resume):
    write_sidecar_config(tmp_path, preempt_grace=10, resume=resume)
    assert sidecar_module.load_sidecar_config(str(tmp_path)).resume is None


def test_child_window_leaves_a_margin_and_has_a_floor():
    assert sidecar_module.child_window(10.0) == 8.0
    assert sidecar_module.child_window(60.0) == 58.0
    # Floored at 1: SIGTERM and SIGKILL in the same breath helps nobody.
    assert sidecar_module.child_window(3.0) == 1.0
    assert sidecar_module.child_window(1.0) == 1.0
    assert sidecar_module.child_window(0.0) == 1.0


@pytest.mark.parametrize(
    "value,expected",
    [(None, 0), ("", 0), ("0", 0), ("1", 1), ("12", 12), ("garbage", 0), ("-3", 0)],
)
def test_restart_count_is_read_from_the_environment(monkeypatch, value, expected):
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    if value is not None:
        monkeypatch.setenv("SLURM_RESTART_COUNT", value)
    assert sidecar_module.read_restart_count() == expected


def test_run_started_carries_the_attempt_number(tmp_path):
    """A single append-only log holds every attempt, so each says which it is."""
    run_dir = tmp_path / "run"

    run_sidecar(run_dir, [sys.executable, "-c", "pass"], run="r1")
    assert of_type(read_events(run_dir), "run_started")[0]["payload"]["attempt"] == 0

    run_sidecar(
        run_dir,
        [sys.executable, "-c", "pass"],
        run="r1",
        env={"SLURM_RESTART_COUNT": "2"},
    )
    started = of_type(read_events(run_dir), "run_started")
    assert [event["payload"]["attempt"] for event in started] == [0, 2]


# --------------------------------------------------------------------------
# Resuming from the previous attempt's checkpoint
# --------------------------------------------------------------------------
#
# The sidecar's whole job here is to put a path on the next attempt's command
# line. So these tests check the command line the training script actually saw,
# from inside the training script, rather than trusting the sidecar's own
# account of it.


ECHO = """\
import json, os, sys

print("argv " + json.dumps(sys.argv[1:]), flush=True)
print("env " + json.dumps({
    "RELAY_RUN_ID": os.environ.get("RELAY_RUN_ID"),
    "RELAY_RUN_DIR": os.environ.get("RELAY_RUN_DIR"),
    "PYTHONPATH": os.environ.get("PYTHONPATH"),
}), flush=True)
"""


def logged_json(run_dir: Path, prefix: str):
    """Pull back the JSON the ECHO script printed on its `prefix` line."""
    for event in of_type(read_events(run_dir), "log"):
        line = event["payload"]["line"]
        if line.startswith(prefix + " "):
            return json.loads(line[len(prefix) + 1 :])
    raise AssertionError(f"the child never logged a {prefix!r} line")


def child_argv(run_dir: Path) -> list[str]:
    return logged_json(run_dir, "argv")


def relay_lines(run_dir: Path) -> list[str]:
    """Log events the sidecar wrote about itself, not ones it copied."""
    return [
        event["payload"]["line"]
        for event in of_type(read_events(run_dir), "log")
        if event["payload"]["stream"] == "relay"
    ]


def resumed_from(run_dir: Path):
    return of_type(read_events(run_dir), "run_started")[-1]["payload"]["resumed_from"]


def touch_at(path: Path, mtime: float) -> str:
    """Create a fake checkpoint file with an exact mtime. Returns its path.

    mtimes are set explicitly rather than by writing the files in order: "these
    two were written a second apart" is not something a test can rely on a
    filesystem to agree with.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake checkpoint bytes", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return str(path)


def resume_job(tmp_path: Path, resume: dict | None, first_attempt_start: float = 0.0):
    """Set up a job directory and a run directory. Returns `(job_dir, run_dir)`.

    The two are separate directories, as they are on the cluster: the glob is
    resolved against the job's working directory, and nothing in the run
    directory (`events.jsonl`, `sidecar.json`) should ever be in its way.

    `.resolve()` because the sidecar resolves matches against `os.getcwd()`,
    which is the physical path -- and on macOS the temp directory is reached
    through a symlink, so an unresolved path would not compare equal.
    """
    job_dir = (tmp_path / "job").resolve()
    job_dir.mkdir()
    run_dir = (tmp_path / "run").resolve()
    write_sidecar_config(
        run_dir,
        requeue_on_term="never",
        preempt_grace=10,
        first_attempt_start=first_attempt_start,
        resume=resume,
    )
    return job_dir, run_dir


def run_attempt(run_dir: Path, job_dir: Path, attempt: int) -> subprocess.CompletedProcess:
    """Run the ECHO script under the sidecar as attempt number `attempt`."""
    script = write_script(job_dir / "train.py", ECHO)
    env = {"SLURM_RESTART_COUNT": str(attempt)} if attempt else None
    result = run_sidecar(
        run_dir,
        [sys.executable, str(script)],
        run="vr_resume",
        env=env,
        cwd=job_dir,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_attempt_zero_never_resumes(tmp_path):
    """The first attempt starts from scratch even with a checkpoint sitting there.

    A file matching the glob on attempt 0 is somebody else's -- a previous run
    in the same directory, or a hand-run experiment. Picking it up would make
    the run silently continue from a state it never reached.
    """
    job_dir, run_dir = resume_job(
        tmp_path, {"checkpoint_glob": "ckpt/*.pt", "arg": "--resume {path}"}
    )
    touch_at(job_dir / "ckpt" / "step_100.pt", time.time())

    run_attempt(run_dir, job_dir, attempt=0)

    assert child_argv(run_dir) == []
    assert resumed_from(run_dir) is None
    # Not even the "nothing matched" warning: there is nothing to resume yet.
    assert relay_lines(run_dir) == []


def test_attempt_one_appends_the_newest_checkpoint(tmp_path):
    """A requeued attempt is handed the newest checkpoint, by mtime."""
    job_dir, run_dir = resume_job(
        tmp_path, {"checkpoint_glob": "ckpt/*.pt", "arg": "--resume {path}"}
    )
    now = time.time()
    touch_at(job_dir / "ckpt" / "step_100.pt", now - 600)
    newest = touch_at(job_dir / "ckpt" / "step_200.pt", now - 10)
    # Named so that alphabetical order and mtime order disagree -- mtime wins.
    touch_at(job_dir / "ckpt" / "zzz_stale.pt", now - 900)

    run_attempt(run_dir, job_dir, attempt=1)

    assert child_argv(run_dir) == ["--resume", newest]
    assert resumed_from(run_dir) == newest
    assert relay_lines(run_dir) == [f"resuming from {newest} (matched ckpt/*.pt)"]


def test_checkpoints_from_before_the_run_started_are_ignored(tmp_path):
    """Only files the run itself wrote count. Older ones are somebody else's.

    `first_attempt_start` is the floor: a checkpoint left in the working
    directory by a previous run would otherwise be resumed from silently, and
    the resulting curves would look plausible and be wrong.
    """
    start = time.time()
    job_dir, run_dir = resume_job(
        tmp_path,
        {"checkpoint_glob": "ckpt/*.pt", "arg": "--resume {path}"},
        first_attempt_start=start,
    )
    touch_at(job_dir / "ckpt" / "old_run.pt", start - 3600)
    touch_at(job_dir / "ckpt" / "step_100.pt", start + 5)
    newest = touch_at(job_dir / "ckpt" / "step_200.pt", start + 60)

    run_attempt(run_dir, job_dir, attempt=1)

    assert child_argv(run_dir) == ["--resume", newest]
    assert resumed_from(run_dir) == newest


def test_only_stale_checkpoints_counts_as_no_match(tmp_path):
    start = time.time()
    job_dir, run_dir = resume_job(
        tmp_path,
        {"checkpoint_glob": "ckpt/*.pt", "arg": "--resume {path}"},
        first_attempt_start=start,
    )
    touch_at(job_dir / "ckpt" / "old_run.pt", start - 3600)

    run_attempt(run_dir, job_dir, attempt=1)

    assert child_argv(run_dir) == []
    assert resumed_from(run_dir) is None
    assert relay_lines(run_dir) == [
        "warning: no checkpoint matched ckpt/*.pt newer than the run's first "
        "start; starting from scratch"
    ]


def test_no_match_runs_the_command_unchanged_and_warns(tmp_path):
    """Nothing matched: run anyway, but say so.

    Starting over is the right behaviour -- refusing to run would turn a
    missing file into a dead experiment -- but an attempt that quietly threw
    away its progress is the thing the user most needs to find in the log.
    """
    job_dir, run_dir = resume_job(
        tmp_path, {"checkpoint_glob": "ckpt/*.pt", "arg": "--resume {path}"}
    )
    # No ckpt directory at all: the glob cannot even see the directory.

    run_attempt(run_dir, job_dir, attempt=2)

    assert child_argv(run_dir) == []
    assert resumed_from(run_dir) is None
    assert relay_lines(run_dir) == [
        "warning: no checkpoint matched ckpt/*.pt newer than the run's first "
        "start; starting from scratch"
    ]


@pytest.mark.parametrize(
    "arg,expected",
    [
        # Two argv words...
        ("--resume {path}", lambda path: ["--resume", path]),
        # ...or one, depending on what the user's script accepts. The sidecar
        # does not have to know which: quoting and then splitting reproduces
        # whichever shape the arg was written in.
        ("--resume={path}", lambda path: [f"--resume={path}"]),
        ("--ckpt {path} --strict", lambda path: ["--ckpt", path, "--strict"]),
    ],
)
def test_resume_arg_shapes_and_paths_with_spaces(tmp_path, arg, expected):
    """A path with a space in it must arrive as one argv word, not two."""
    job_dir, run_dir = resume_job(
        tmp_path, {"checkpoint_glob": "my ckpts/*.pt", "arg": arg}
    )
    path = touch_at(job_dir / "my ckpts" / "step 200.pt", time.time())

    run_attempt(run_dir, job_dir, attempt=1)

    assert " " in path, "this test is pointless without a space in the path"
    assert child_argv(run_dir) == expected(path)
    assert resumed_from(run_dir) == path


def test_recursive_glob_finds_a_nested_checkpoint(tmp_path):
    """`**` works, because the glob is run with recursive=True."""
    job_dir, run_dir = resume_job(
        tmp_path, {"checkpoint_glob": "out/**/*.pt", "arg": "--resume {path}"}
    )
    path = touch_at(job_dir / "out" / "run" / "deep" / "step_900.pt", time.time())

    run_attempt(run_dir, job_dir, attempt=1)

    assert child_argv(run_dir) == ["--resume", path]
    assert resumed_from(run_dir) == path


def test_no_resume_block_means_no_resume_behaviour(tmp_path):
    """A v1-style sidecar.json: nothing appended, nothing logged, nothing null-less."""
    job_dir, run_dir = resume_job(tmp_path, None)
    touch_at(job_dir / "ckpt" / "step_100.pt", time.time())

    run_attempt(run_dir, job_dir, attempt=1)

    assert child_argv(run_dir) == []
    assert resumed_from(run_dir) is None
    assert relay_lines(run_dir) == []


# --------------------------------------------------------------------------
# The child's environment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attempt", [0, 1])
def test_child_gets_the_relay_environment_on_every_attempt(tmp_path, attempt):
    """RELAY_RUN_ID, RELAY_RUN_DIR, and run_dir first on PYTHONPATH.

    Every attempt, identically: the training script's `import relay_ckpt` has
    to work on the requeued attempt exactly as it did on the first, and the
    values are derived only from the run ID and the run directory, both of
    which are fixed for the life of the run.
    """
    job_dir, run_dir = resume_job(tmp_path, None)
    existing = str(tmp_path / "my-project")

    script = write_script(job_dir / "train.py", ECHO)
    env = {"PYTHONPATH": existing}
    if attempt:
        env["SLURM_RESTART_COUNT"] = str(attempt)
    result = run_sidecar(
        run_dir,
        [sys.executable, str(script)],
        run="vr_env",
        env=env,
        cwd=job_dir,
    )
    assert result.returncode == 0, result.stderr

    seen = logged_json(run_dir, "env")
    assert seen["RELAY_RUN_ID"] == "vr_env"
    assert seen["RELAY_RUN_DIR"] == str(run_dir)

    parts = seen["PYTHONPATH"].split(os.pathsep)
    # First, so relay_ckpt in the run directory wins over anything else with
    # that name...
    assert parts[0] == str(run_dir)
    # ...and the user's own PYTHONPATH is still there behind it. Replacing it
    # would break the very training script we are trying to run.
    assert existing in parts[1:]


def test_child_environment_without_an_existing_pythonpath(monkeypatch):
    """No PYTHONPATH set: we set it to just the run directory, no stray separator."""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = sidecar_module.child_environment("vr_x", "/runs/vr_x")
    assert env["PYTHONPATH"] == "/runs/vr_x"
    assert env["RELAY_RUN_ID"] == "vr_x"
    assert env["RELAY_RUN_DIR"] == "/runs/vr_x"


# --------------------------------------------------------------------------
# Durability details
# --------------------------------------------------------------------------


def test_events_are_appended_not_rewritten(tmp_path):
    """O_APPEND plus one write per line: existing bytes are never touched."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {"v": 1, "run": "r1", "seq": 2, "ts": "2026-09-16T00:00:00.000Z",
             "type": "log", "payload": {"stream": "stdout", "line": "first attempt"}}
        )
        + "\n",
        encoding="utf-8",
    )
    before = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    run_sidecar(run_dir, [sys.executable, "-c", "print('second attempt')"], run="r1")

    after = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before)
    assert after.endswith("\n")


def test_every_line_is_complete_json(tmp_path):
    """One os.write per complete line means no half-records in the file."""
    script = write_script(
        tmp_path / "train.py",
        "for i in range(200):\n"
        "    print('##relay## {\"step\": %d, \"loss\": 0.1}' % i, flush=True)\n"
        "    print('log line %d' % i, flush=True)\n",
    )
    run_dir = tmp_path / "run"
    run_sidecar(run_dir, [sys.executable, str(script)])

    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        json.loads(line)  # raises if any line is truncated or interleaved

    events = read_events(run_dir)
    assert len(of_type(events, "metric")) == 200
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))


def test_runnable_as_a_copied_file(tmp_path):
    """On the cluster it is a lone file next to the sbatch script, not a package."""
    copied = tmp_path / "sidecar.py"
    copied.write_text(SIDECAR_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    run_dir = tmp_path / "run"

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # make sure `relay` is not importable by accident

    result = subprocess.run(
        [
            sys.executable, str(copied),
            "--run", "vr_copied",
            "--dir", str(run_dir),
            "--", sys.executable, "-c", "print('hello from a copy')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr

    events = read_events(run_dir)
    assert events[0]["type"] == "run_started"
    assert events[-1]["payload"] == {"exit_code": 0, "status": "completed"}
