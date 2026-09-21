"""Tests for `LocalBackend`.

These start real subprocesses. There is no way around that: the things worth
testing here -- that a PID is alive, that a process group dies on SIGTERM, that
an exit code survives the process that produced it -- do not exist inside a
single-process function call, and a mock of `subprocess.Popen` would only prove
that the mock behaves like the mock.

Everything that waits does so with a deadline loop rather than a fixed sleep.
Process startup on a loaded CI machine can take an order of magnitude longer
than on a laptop, so `sleep(0.2); assert done` is a test that fails on Tuesdays.
`wait_until(predicate)` returns as soon as the condition holds and only spends
the full timeout when something is genuinely wrong.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from relay.backends.base import STATUSES, JobSpec, UsageRow
from relay.backends.local import (
    ATTEMPT_FILENAME,
    CANCEL_FILENAME,
    CKPT_HELPER_FILENAME,
    EXIT_CODE_FILENAME,
    OUTPUT_FILENAME,
    PID_FILENAME,
    SIDECAR_CONFIG_FILENAME,
    SIDECAR_FILENAME,
    LocalBackend,
    _render_wrapper,
    default_ckpt_helper_path,
)

# Generous: these are upper bounds on "something has gone wrong", not on how
# long the test normally takes.
TIMEOUT = 30.0
POLL = 0.05


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def wait_until(predicate, timeout: float = TIMEOUT, what: str = "condition"):
    """Poll `predicate` until it returns something truthy, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(POLL)
    pytest.fail(f"timed out after {timeout}s waiting for {what}")


def write_script(path: Path, body: str) -> Path:
    """Write a fake training script and return its path."""
    path.write_text(body, encoding="utf-8")
    return path


# A training script that emits two metric lines the sidecar will recognise, one
# ordinary log line, and exits 0.
GOOD_SCRIPT = """\
import sys, time
print("starting up")
for step in (100, 200):
    print('##relay## {"step": %d, "loss": 0.5}' % step, flush=True)
print("done")
sys.exit(0)
"""

FAILING_SCRIPT = """\
import sys
print("about to fail", flush=True)
sys.exit(3)
"""

# Sleeps effectively forever, but handles SIGTERM the way a training script with
# checkpointing would: notice, say so, exit cleanly.
SLEEPING_SCRIPT = """\
import signal, sys, time

def on_term(signum, frame):
    print('##relay## {"checkpoint": "ckpt/last.pt", "step": 7}', flush=True)
    sys.exit(0)

signal.signal(signal.SIGTERM, on_term)
print("sleeping", flush=True)
for _ in range(600):
    time.sleep(0.1)
"""


# Ignores SIGTERM, so the sidecar has to wait out its preemption window and
# then SIGKILL. Used to check that the window is the one `sidecar.json` says.
STUBBORN_SCRIPT = """\
import signal, time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("stubborn", flush=True)
for _ in range(600):
    time.sleep(0.1)
"""


def fake_scontrol_env(tmp_path: Path, job_id: str = "777001") -> tuple[dict, Path]:
    """A fake `scontrol` on PATH plus a SLURM_JOB_ID, as `spec.env`.

    The sidecar looks `scontrol` up on PATH, exactly as it would on a cluster,
    so a shell script earlier in PATH is all it takes to see whether a requeue
    happened -- and how many times. The count is the whole point: the bug being
    fixed here was a requeue that should never have been made.

    SLURM_JOB_ID has to be there too. Without it the sidecar declines to
    requeue at all, which is right on a laptop and would hide the behaviour
    under test.
    """
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "scontrol_calls.log"
    script = bin_dir / "scontrol"
    script.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\n', encoding="utf-8")
    script.chmod(0o755)
    return (
        {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "SLURM_JOB_ID": job_id,
        },
        log,
    )


def scontrol_calls(log: Path) -> list[str]:
    try:
        return [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    except FileNotFoundError:
        return []


def sidecar_pid(backend: LocalBackend, run_dir: Path) -> int:
    """The sidecar's own PID, which it records in `run_started`.

    The job ID is the *wrapper shell's* PID. Signalling the sidecar alone is
    how Slurm's `--signal=B:` form behaves, and it keeps the training script's
    only SIGTERM the one the sidecar forwards.
    """
    started = [
        event
        for event in read_events(backend, run_dir)
        if event["type"] == "run_started"
    ]
    assert started, "the sidecar has not written run_started yet"
    return int(started[-1]["payload"]["pid"])


def wait_for_run_started(backend: LocalBackend, run_dir: Path) -> None:
    wait_until(
        lambda: (run_dir / "events.jsonl").exists()
        and any(
            event["type"] == "run_started"
            for event in read_events(backend, run_dir)
        ),
        what="the sidecar to write run_started",
    )


def ended_status(backend: LocalBackend, run_dir: Path) -> str:
    ended = [
        event for event in read_events(backend, run_dir) if event["type"] == "run_ended"
    ]
    assert ended, "the run never wrote an ending"
    return ended[-1]["payload"]["status"]


def make_backend(tmp_path: Path) -> LocalBackend:
    """A backend rooted inside the test's tmp_path, not the real ~/.local."""
    return LocalBackend(root=tmp_path / "backend_root")


def make_spec(tmp_path: Path, script: Path, run_id: str = "vr_s1_test") -> JobSpec:
    return JobSpec(
        run_id=run_id,
        # sys.executable so the child runs under the same interpreter pytest
        # does, which matters inside a virtualenv.
        command=[sys.executable, str(script)],
        run_dir=str(tmp_path / "runs" / run_id),
        # Short, so a test that wants to see a heartbeat or a grace window does
        # not have to wait half a minute for one.
        grace_seconds=10,
        heartbeat_seconds=1,
    )


def read_events(backend: LocalBackend, run_dir: Path) -> list[dict]:
    """Read events.jsonl through `read_bytes`, exercising the real read path."""
    data, _ = backend.read_bytes(str(run_dir / "events.jsonl"), 0)
    events = []
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def event_types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


def test_submit_starts_a_live_process_and_lays_out_the_run_dir(tmp_path):
    """Submit returns a live PID, and the run directory looks the way Slurm's will."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", SLEEPING_SCRIPT)
    spec = make_spec(tmp_path, script)
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)

    try:
        # The job ID is a PID, as a string, and it names a process that exists.
        pid = int(job_id)
        os.kill(pid, 0)  # raises if it is not there

        # The sidecar was copied next to the job, exactly as SlurmBackend will
        # copy it next to the sbatch script on shared storage.
        assert (run_dir / SIDECAR_FILENAME).is_file()

        # job.pid is for a human with a terminal; it should agree with the ID.
        assert (run_dir / PID_FILENAME).read_text().strip() == job_id

        # The sidecar writes run_started before anything else, so the event log
        # appearing is proof the sidecar actually got going.
        wait_until(
            lambda: (run_dir / "events.jsonl").exists(), what="events.jsonl to appear"
        )
        wait_until(
            lambda: "run_started" in event_types(read_events(backend, run_dir)),
            what="a run_started event",
        )

        # While the process is alive and has written no exit code, it is running.
        assert backend.status([job_id]) == {job_id: "running"}
    finally:
        backend.cancel(job_id)


def test_completed_job_reports_completed_with_a_full_event_log(tmp_path):
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script)
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)

    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed",
        what="the job to report completed",
    )

    # The exit code file is what makes that verdict survive a restart.
    assert (run_dir / EXIT_CODE_FILENAME).read_text().strip() == "0"

    events = read_events(backend, run_dir)
    types = event_types(events)
    assert types[0] == "run_started"
    assert types[-1] == "run_ended"
    assert "metric" in types

    ended = events[-1]["payload"]
    assert ended["exit_code"] == 0
    assert ended["status"] == "completed"

    # seq is a per-run monotonic counter starting at 1.
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))

    # The human-readable fallback copy of stdout exists and has the job's output.
    assert "starting up" in (run_dir / OUTPUT_FILENAME).read_text()


def test_failing_job_reports_failed(tmp_path):
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "boom.py", FAILING_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_s2_fail")
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)

    wait_until(
        lambda: backend.status([job_id])[job_id] == "failed",
        what="the job to report failed",
    )
    # The sidecar exits with the child's code, and the wrapper records it, so
    # the user's 3 makes it all the way out.
    assert (run_dir / EXIT_CODE_FILENAME).read_text().strip() == "3"

    events = read_events(backend, run_dir)
    assert events[-1]["payload"]["status"] == "failed"
    assert events[-1]["payload"]["exit_code"] == 3


def test_status_covers_every_requested_id_with_a_known_vocabulary(tmp_path):
    """Batched status: one call, one key per job, all values legal."""
    backend = make_backend(tmp_path)
    good = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    bad = write_script(tmp_path / "boom.py", FAILING_SCRIPT)

    a = backend.submit(make_spec(tmp_path, good, run_id="vr_a"))
    b = backend.submit(make_spec(tmp_path, bad, run_id="vr_b"))

    def both_finished():
        result = backend.status([a, b])
        return result if result[a] != "running" and result[b] != "running" else None

    result = wait_until(both_finished, what="both jobs to finish")

    assert set(result) == {a, b}
    assert result == {a: "completed", b: "failed"}
    assert all(value in STATUSES for value in result.values())


# --------------------------------------------------------------------------
# The pid -> run_dir index (the "daemon restarted" case)
# --------------------------------------------------------------------------


def test_a_fresh_backend_on_the_same_root_still_resolves_the_exit_code(tmp_path):
    """A restarted daemon has no Popen object, only the index file and exit_code.

    This is the whole reason both files exist. The kernel forgets a subprocess's
    exit status the moment it is reaped, and the new process was never its
    parent anyway -- so if the verdict were not written down, every run that
    finished while the daemon was down would be permanently undecidable.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_restart")

    job_id = backend.submit(spec)
    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed",
        what="the first backend to see the job finish",
    )

    # A brand new backend object on the same root: no _spawned entry, no cached
    # index, nothing in memory at all. Same root directory is the only link.
    restarted = make_backend(tmp_path)
    assert restarted.status([job_id]) == {job_id: "completed"}


def test_index_miss_is_re_read_from_disk(tmp_path):
    """A backend that cached the index still picks up a later submit.

    Two relay processes share a root: the daemon polls while `relay submit`
    adds rows. If the daemon cached the index at startup it would never resolve
    anything submitted afterwards.
    """
    watcher = make_backend(tmp_path)
    submitter = make_backend(tmp_path)

    # Prime the watcher's cache with an empty index.
    assert watcher.status(["999999999"]) == {"999999999": "unknown"}

    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    job_id = submitter.submit(make_spec(tmp_path, script, run_id="vr_late"))

    wait_until(
        lambda: watcher.status([job_id])[job_id] == "completed",
        what="the watcher to resolve a job it never submitted",
    )


def test_unknown_pid_is_unknown_not_an_error(tmp_path):
    backend = make_backend(tmp_path)
    # Not a PID at all, and a PID that cannot exist. Both are "unknown": the
    # daemon reads that as "no new information" and keeps going.
    result = backend.status(["not-a-pid", "4294967295"])
    assert result == {"not-a-pid": "unknown", "4294967295": "unknown"}


def test_empty_status_request_is_an_empty_dict(tmp_path):
    assert make_backend(tmp_path).status([]) == {}


# --------------------------------------------------------------------------
# read_bytes
# --------------------------------------------------------------------------


def test_read_bytes_is_incremental(tmp_path):
    """Read, remember the offset, read again, get only what is new.

    This is exactly the daemon's tail loop, isolated. Written against a plain
    file rather than a live run so the test controls the bytes precisely.
    """
    backend = make_backend(tmp_path)
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"seq":1}\n')

    first, offset = backend.read_bytes(str(path), 0)
    assert first == b'{"seq":1}\n'
    assert offset == len(first)

    # Nothing new: no bytes, same offset. Not an error, just a quiet cycle.
    nothing, same = backend.read_bytes(str(path), offset)
    assert nothing == b""
    assert same == offset

    with open(path, "ab") as handle:
        handle.write(b'{"seq":2}\n')

    second, new_offset = backend.read_bytes(str(path), offset)
    assert second == b'{"seq":2}\n'
    assert new_offset == offset + len(second)

    # And reading the whole file from zero gives both records back, so the
    # incremental path and the full path agree.
    whole, end = backend.read_bytes(str(path), 0)
    assert whole == b'{"seq":1}\n{"seq":2}\n'
    assert end == new_offset


def test_read_bytes_on_a_missing_path_raises_file_not_found(tmp_path):
    """FileNotFoundError must escape the backend.

    The daemon reads it as "this job has not created its run directory yet, so
    it is still queued". If the backend swallowed it and returned empty bytes,
    a queued job would be indistinguishable from a started one that has gone
    silent -- and the staleness check would start firing on jobs that have not
    begun.
    """
    backend = make_backend(tmp_path)
    with pytest.raises(FileNotFoundError):
        backend.read_bytes(str(tmp_path / "nope" / "events.jsonl"), 0)


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------


def test_cancel_stops_the_process_group_and_the_sidecar_records_it(tmp_path):
    """Cancel signals the group; the sidecar's signal path writes it down.

    Note what `run_ended` says: `interrupted`, not `cancelled` and not
    `requeued`. The sidecar cannot tell a `relay cancel` from a Slurm
    preemption -- both are a bare SIGTERM -- so it claims neither, and the
    daemon resolves it from the scheduler afterwards. That is the standing
    rule: the log is authoritative about what the training did, the scheduler
    about what the job did. The backend just delivers the signal.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "sleepy.py", SLEEPING_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_cancel")
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    pid = int(job_id)

    # Wait until the training script is genuinely up, so we are cancelling a
    # running job rather than racing the interpreter's startup.
    wait_until(
        lambda: any(
            event["type"] == "log" and "sleeping" in event["payload"].get("line", "")
            for event in read_events(backend, run_dir)
        )
        if (run_dir / "events.jsonl").exists()
        else False,
        what="the training script to start",
    )

    backend.cancel(job_id)

    # The whole group goes away: wrapper shell, sidecar, training script.
    def group_is_gone():
        try:
            os.killpg(pid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        return False

    wait_until(group_is_gone, what="the process group to exit")

    events = read_events(backend, run_dir)
    types = event_types(events)
    assert "signal" in types
    assert types[-1] == "run_ended"
    assert events[-1]["payload"] == {"exit_code": None, "status": "interrupted"}

    # The signal event names the signal we actually sent, and why it mattered.
    signal_event = next(event for event in events if event["type"] == "signal")
    assert signal_event["payload"] == {"signal": "SIGTERM", "reason": "term"}

    # seq stayed monotonic through the signal path.
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))


def test_cancelling_a_finished_job_is_a_no_op(tmp_path):
    """The daemon acts on a snapshot that may be seconds old; losing that race
    is normal, not an error."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    job_id = backend.submit(make_spec(tmp_path, script, run_id="vr_done"))

    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed",
        what="the job to finish",
    )

    backend.cancel(job_id)  # must not raise
    assert backend.status([job_id]) == {job_id: "completed"}


def test_cancel_rejects_a_job_id_that_is_not_a_pid(tmp_path):
    """Unlike status, a malformed ID here is a programming error worth a shout."""
    backend = make_backend(tmp_path)
    with pytest.raises(ValueError):
        backend.cancel("slurm-12345")


# --------------------------------------------------------------------------
# Odds and ends
# --------------------------------------------------------------------------


def test_stale_exit_code_from_a_previous_attempt_is_cleared_on_submit(tmp_path):
    """A reused run directory must not announce the new attempt as finished."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "sleepy.py", SLEEPING_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_reuse")
    run_dir = Path(spec.run_dir)

    # Pretend a previous attempt in this directory failed.
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / EXIT_CODE_FILENAME).write_text("1", encoding="utf-8")

    job_id = backend.submit(spec)
    try:
        assert not (run_dir / EXIT_CODE_FILENAME).exists()
        assert backend.status([job_id]) == {job_id: "running"}
    finally:
        backend.cancel(job_id)


def test_spec_env_reaches_the_training_script(tmp_path):
    """JobSpec.env is added on top of the inherited environment."""
    backend = make_backend(tmp_path)
    script = write_script(
        tmp_path / "env.py",
        "import os\nprint('RELAY_TEST_VAR=' + os.environ.get('RELAY_TEST_VAR', ''))\n",
    )
    spec = make_spec(tmp_path, script, run_id="vr_env")
    spec.env = {"RELAY_TEST_VAR": "hello"}
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed",
        what="the env job to finish",
    )

    lines = [
        event["payload"]["line"]
        for event in read_events(backend, run_dir)
        if event["type"] == "log"
    ]
    assert "RELAY_TEST_VAR=hello" in lines


def test_slurm_fields_are_ignored_rather_than_rejected(tmp_path, caplog):
    """One JobSpec shape serves both backends; local just ignores the Slurm half.

    Ignored, but not silently: `--mem=64G` doing nothing is a surprise worth
    being able to see, so the backend says so once at DEBUG. DEBUG and not
    WARNING because a spec carrying Slurm fields to LocalBackend is the normal
    case -- the CLI builds one spec from one config -- not a mistake.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_slurmfields")
    spec.account = "mlopt"
    spec.partition = "gpu-a40"
    spec.sbatch_extra = ["--mem=64G"]
    spec.gres = "gpu:1"

    with caplog.at_level(logging.DEBUG, logger="relay.backends.local"):
        job_id = backend.submit(spec)

    messages = [record.getMessage() for record in caplog.records]
    ignored = [line for line in messages if "ignores Slurm-only fields" in line]
    assert len(ignored) == 1, f"expected exactly one debug line, got {messages}"
    assert "--mem=64G" in ignored[0]
    assert "gpu:1" in ignored[0]

    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed",
        what="the job to finish despite the Slurm fields",
    )


def test_no_debug_line_when_there_are_no_slurm_fields(tmp_path, caplog):
    """The line is about something being dropped; with nothing to drop, silence."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_noslurmfields")

    with caplog.at_level(logging.DEBUG, logger="relay.backends.local"):
        job_id = backend.submit(spec)

    assert not [
        record
        for record in caplog.records
        if "ignores Slurm-only fields" in record.getMessage()
    ]
    wait_until(
        lambda: backend.status([job_id])[job_id] != "running", what="the job to finish"
    )


def test_custom_sidecar_path_is_the_file_that_gets_copied(tmp_path):
    """The tests' escape hatch, and the one thing SlurmBackend will reuse."""
    fake = tmp_path / "my_sidecar.py"
    fake.write_text("# not the real sidecar\n", encoding="utf-8")
    backend = LocalBackend(root=tmp_path / "root", sidecar_path=fake)

    spec = make_spec(tmp_path, tmp_path / "unused.py", run_id="vr_sidecar")
    run_dir = Path(spec.run_dir)
    job_id = backend.submit(spec)

    # The copy is made before the process starts, so it is there immediately.
    assert (run_dir / SIDECAR_FILENAME).read_text() == "# not the real sidecar\n"

    # That "sidecar" does nothing and exits 0, so the wrapper still records a
    # code -- proof the wrapper is independent of what it wraps.
    wait_until(
        lambda: backend.status([job_id])[job_id] in ("completed", "failed"),
        what="the fake sidecar to finish",
    )


def test_index_file_maps_job_ids_to_run_dirs(tmp_path):
    """The index is a small, readable JSON file, not an opaque blob."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_index")
    job_id = backend.submit(spec)

    index = json.loads(
        (tmp_path / "backend_root" / "local_jobs.json").read_text(encoding="utf-8")
    )
    assert index[job_id]["run_dir"] == spec.run_dir
    assert index[job_id]["run_id"] == "vr_index"

    wait_until(
        lambda: backend.status([job_id])[job_id] != "running", what="the job to finish"
    )


# --------------------------------------------------------------------------
# sidecar.json: the settings the sidecar must have before a signal arrives
# --------------------------------------------------------------------------


def test_submit_writes_sidecar_json(tmp_path):
    """The backend hands the sidecar its signal policy as a file at submit time.

    It has to be a file, and it has to be written now. The sidecar is a
    separate process on a machine we cannot reach, and by the time the setting
    matters -- SIGTERM has landed and SIGKILL is ten seconds away -- there is
    no time to go and look anything up.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_cfg")
    spec.requeue_on_term = "always"
    spec.preempt_grace = 25
    run_dir = Path(spec.run_dir)

    before = time.time()
    job_id = backend.submit(spec)
    try:
        config = json.loads((run_dir / SIDECAR_CONFIG_FILENAME).read_text())
        assert config["requeue_on_term"] == "always"
        assert config["preempt_grace"] == 25

        # The clock the resume path filters checkpoint mtimes against. A
        # float, and it is *now*, because this is the first attempt in this
        # directory.
        assert isinstance(config["first_attempt_start"], float)
        assert before <= config["first_attempt_start"] <= time.time()

        # No resume asked for, so no resume offered. Null rather than a
        # missing key: the sidecar reads one shape either way.
        assert config["resume"] is None
    finally:
        backend.cancel(job_id)


def test_sidecar_json_carries_the_resume_pair_when_the_spec_has_one(tmp_path):
    """Both halves travel together, under the names the sidecar reads."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_resume")
    spec.resume_glob = "checkpoints/**/*.pt"
    spec.resume_arg = "--resume-from {path}"
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    try:
        config = json.loads((run_dir / SIDECAR_CONFIG_FILENAME).read_text())
        assert config["resume"] == {
            "checkpoint_glob": "checkpoints/**/*.pt",
            "arg": "--resume-from {path}",
        }
    finally:
        backend.cancel(job_id)


@pytest.mark.parametrize(
    "resume_glob, resume_arg",
    [
        ("checkpoints/*.pt", None),
        (None, "--resume-from {path}"),
    ],
)
def test_half_a_resume_spec_is_no_resume_spec(tmp_path, resume_glob, resume_arg):
    """A glob with no argument has nothing to say about the file it found."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_half_resume")
    spec.resume_glob = resume_glob
    spec.resume_arg = resume_arg
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    try:
        config = json.loads((run_dir / SIDECAR_CONFIG_FILENAME).read_text())
        assert config["resume"] is None
    finally:
        backend.cancel(job_id)


def test_a_resubmit_keeps_the_original_first_attempt_start(tmp_path):
    """The one key a resubmit must not rewrite, and the whole reason resume works.

    Attempt 0 writes a checkpoint at some moment T. The sidecar finds it again
    by globbing and keeping only files modified at or after
    `first_attempt_start`. If attempt 1's submit reset that to its own
    `time.time()` -- necessarily later than T -- the filter would discard
    exactly the file the resume exists to find, and every attempt would start
    from step zero.

    Slurm never hits this: it requeues in place and never comes back through
    `submit`. LocalBackend has no scheduler, so a resubmit does, and it has to
    carry the value forward itself.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_first_start")
    spec.preempt_grace = 10
    run_dir = Path(spec.run_dir)

    first_job = backend.submit(spec)
    wait_until(
        lambda: backend.status([first_job])[first_job] == "completed",
        what="attempt 0 to finish",
    )
    first_config = json.loads((run_dir / SIDECAR_CONFIG_FILENAME).read_text())
    original_start = first_config["first_attempt_start"]

    # Enough of a gap that a rewritten timestamp could not be mistaken for the
    # original one through float noise.
    time.sleep(0.05)

    # Everything *else* is rewritten from the new spec, so the test also shows
    # that preserving one key does not freeze the rest of the file.
    spec.preempt_grace = 42
    spec.requeue_on_term = "always"
    spec.resume_glob = "ckpt/*.pt"
    spec.resume_arg = "--resume {path}"

    second_job = backend.submit(spec)
    wait_until(
        lambda: backend.status([second_job])[second_job] == "completed",
        what="attempt 1 to finish",
    )
    second_config = json.loads((run_dir / SIDECAR_CONFIG_FILENAME).read_text())

    assert second_config["first_attempt_start"] == original_start
    assert second_config["preempt_grace"] == 42
    assert second_config["requeue_on_term"] == "always"
    assert second_config["resume"] == {
        "checkpoint_glob": "ckpt/*.pt",
        "arg": "--resume {path}",
    }


def test_a_nonsense_first_attempt_start_is_replaced_rather_than_trusted(tmp_path):
    """A hand-edited or truncated sidecar.json must not stop a submit.

    Being wrong here costs a resume, not a run, so the backend starts the
    clock again rather than raising.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_bad_start")
    run_dir = Path(spec.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SIDECAR_CONFIG_FILENAME).write_text(
        '{"first_attempt_start": "yesterday"}', encoding="utf-8"
    )

    before = time.time()
    job_id = backend.submit(spec)
    try:
        config = json.loads((run_dir / SIDECAR_CONFIG_FILENAME).read_text())
        assert isinstance(config["first_attempt_start"], float)
        assert config["first_attempt_start"] >= before
    finally:
        backend.cancel(job_id)


def test_the_checkpoint_helper_is_copied_next_to_the_sidecar(tmp_path):
    """`relay_ckpt.py` travels with the job the same way `sidecar.py` does.

    The compute node may have no relay installed, so a training script that
    wants the helper has to find it in the run directory. Skipped when the
    installation has no helper: it is optional, and its absence must not fail
    a submit or this suite.
    """
    source = default_ckpt_helper_path()
    if not source.is_file():
        pytest.skip(f"{source.name} is not part of this installation yet")

    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_ckpt_helper")
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    try:
        copied = run_dir / CKPT_HELPER_FILENAME
        assert copied.is_file()
        assert copied.read_bytes() == source.read_bytes()
        # Beside the sidecar, so one `sys.path` entry reaches both.
        assert copied.parent == (run_dir / SIDECAR_FILENAME).parent
    finally:
        backend.cancel(job_id)


def test_auto_resolves_to_never_locally(tmp_path):
    """"auto" means "let the scheduler handle it", and there is no scheduler here.

    The sidecar is never shown "auto": resolving it is the backend's job,
    because the backend is the one that can ask Slurm about the partition.
    Locally the answer is always `never` -- there is no queue to go back into.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_auto")
    assert spec.requeue_on_term == "auto", "auto should be the JobSpec default"
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    config = json.loads((run_dir / SIDECAR_CONFIG_FILENAME).read_text())
    assert config["requeue_on_term"] == "never"
    # Klone is the opposite case and the reason "auto" exists at all: there
    # PreemptMode=REQUEUE, so Slurm requeues preempted jobs itself and a
    # sidecar that also called `scontrol requeue` would do it twice.
    wait_until(
        lambda: backend.status([job_id])[job_id] != "running",
        what="the job to finish",
    )


# --------------------------------------------------------------------------
# Signal semantics, end to end through a real sidecar
# --------------------------------------------------------------------------
#
# These are the tests for the bug: `scancel` sends SIGTERM, and a sidecar that
# treats every SIGTERM as a preemption puts the cancelled job back in the
# queue. Each one runs a real job, signals it, and then counts how many times a
# fake `scontrol` on PATH was called.


def run_until_signalled(
    tmp_path: Path,
    run_id: str,
    signum: int,
    *,
    requeue_on_term: str = "auto",
    preempt_grace: int = 10,
    cancel_via_backend: bool = False,
    script_body: str = SLEEPING_SCRIPT,
) -> tuple[LocalBackend, Path, list[str]]:
    """Submit a job, let it get going, signal it, wait for it to die.

    Returns `(backend, run_dir, scontrol call lines)`.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / f"{run_id}.py", script_body)
    spec = make_spec(tmp_path, script, run_id=run_id)
    spec.requeue_on_term = requeue_on_term
    spec.preempt_grace = preempt_grace
    env, calls = fake_scontrol_env(tmp_path)
    spec.env = env
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    wait_for_run_started(backend, run_dir)

    if cancel_via_backend:
        # The real `relay cancel` path: the CANCEL_REQUESTED file first, the
        # signal second.
        backend.cancel(job_id, run_dir=str(run_dir))
    else:
        # Slurm's `--signal=B:` form signals the batch process, not the group,
        # so the training script's only SIGTERM is the one the sidecar
        # forwards. That is also the preemption case: nobody wrote a cancel
        # file, because nobody asked for this.
        os.kill(sidecar_pid(backend, run_dir), signum)

    wait_until(
        lambda: (run_dir / EXIT_CODE_FILENAME).is_file(),
        what="the job to die and record its exit code",
    )
    wait_until(
        lambda: any(
            event["type"] == "run_ended" for event in read_events(backend, run_dir)
        ),
        what="the sidecar to write its ending",
    )
    return backend, run_dir, scontrol_calls(calls)


def test_sigusr1_ends_requeued_and_calls_scontrol_once(tmp_path):
    """Time limit approaching. Only Slurm sends USR1, so requeue is unambiguous."""
    backend, run_dir, calls = run_until_signalled(
        tmp_path, "vr_usr1", signal.SIGUSR1
    )

    events = read_events(backend, run_dir)
    signal_event = next(event for event in events if event["type"] == "signal")
    assert signal_event["payload"] == {"signal": "SIGUSR1", "reason": "time_limit"}
    assert ended_status(backend, run_dir) == "requeued"

    # Exactly one. Zero would strand the run; two would run it twice.
    assert calls == ["requeue 777001"]

    # Exit 0 on the requeue path: a nonzero exit would stop Slurm requeueing it.
    assert (run_dir / EXIT_CODE_FILENAME).read_text().strip() == "0"


def test_sigterm_with_cancel_requested_never_requeues(tmp_path):
    """`relay cancel` writes the file first, so the cancelled job stays dead.

    This is the bug in one test. Without the file, a job cancelled on a
    partition configured `requeue_on_term: always` would come straight back.
    """
    backend, run_dir, calls = run_until_signalled(
        tmp_path,
        "vr_cancelled",
        signal.SIGTERM,
        requeue_on_term="always",
        cancel_via_backend=True,
    )

    assert (run_dir / CANCEL_FILENAME).is_file()
    assert ended_status(backend, run_dir) == "interrupted"
    assert calls == []


def test_sigterm_with_requeue_never_does_not_requeue(tmp_path):
    backend, run_dir, calls = run_until_signalled(
        tmp_path, "vr_never", signal.SIGTERM, requeue_on_term="never"
    )
    assert ended_status(backend, run_dir) == "interrupted"
    assert calls == []


def test_sigterm_with_requeue_always_requeues_once(tmp_path):
    """For a partition whose PreemptMode does not requeue for us."""
    backend, run_dir, calls = run_until_signalled(
        tmp_path, "vr_always", signal.SIGTERM, requeue_on_term="always"
    )

    events = read_events(backend, run_dir)
    signal_event = next(event for event in events if event["type"] == "signal")
    assert signal_event["payload"] == {"signal": "SIGTERM", "reason": "term"}
    # `interrupted` either way: requeueing is what we did about it, not what
    # happened to us.
    assert ended_status(backend, run_dir) == "interrupted"
    assert calls == ["requeue 777001"]


def test_preempt_grace_bounds_a_stubborn_child(tmp_path):
    """preempt_grace 3 leaves the child 3 - 2 = 1 second, the floor.

    Loose at the top end -- these are real processes on a possibly-loaded
    machine -- but tight enough to fail if the sidecar used the 10-second
    default, or waited for a child that will never exit on its own.
    """
    started = time.monotonic()
    backend, run_dir, _ = run_until_signalled(
        tmp_path,
        "vr_stubborn",
        signal.SIGTERM,
        preempt_grace=3,
        script_body=STUBBORN_SCRIPT,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, f"took {elapsed:.1f}s; the 1s window was not honoured"
    assert ended_status(backend, run_dir) == "interrupted"


def test_cancel_without_a_run_dir_writes_no_flag(tmp_path):
    """`run_dir` is optional, and omitting it must not invent a path.

    A caller that does not know the run directory still gets a working cancel;
    it just does not get the protection against a self-requeue.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "sleepy.py", SLEEPING_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_nodir")
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    wait_for_run_started(backend, run_dir)
    backend.cancel(job_id)

    assert not (run_dir / CANCEL_FILENAME).exists()


def test_cancel_is_still_idempotent_with_a_run_dir(tmp_path):
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_twice")
    job_id = backend.submit(spec)
    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed", what="the job to finish"
    )

    backend.cancel(job_id, run_dir=spec.run_dir)
    backend.cancel(job_id, run_dir=spec.run_dir)  # neither call may raise
    assert (Path(spec.run_dir) / CANCEL_FILENAME).is_file()


# --------------------------------------------------------------------------
# Attempts
# --------------------------------------------------------------------------


def attempts_reported(backend: LocalBackend, run_dir: Path) -> list[int]:
    return [
        event["payload"]["attempt"]
        for event in read_events(backend, run_dir)
        if event["type"] == "run_started"
    ]


def test_attempt_counts_up_across_resubmits_into_the_same_run_dir(tmp_path):
    """A requeued run reuses its directory, and each attempt says which it is.

    The counter is a file this backend owns rather than a count of
    `run_started` events. Counting events would work and is the trap: a backend
    that parses `events.jsonl` knows the event schema, and the rule that keeps
    relay's pieces apart is that it does not.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_attempts")
    run_dir = Path(spec.run_dir)

    first = backend.submit(spec)
    wait_until(
        lambda: backend.status([first])[first] == "completed", what="attempt 0 to finish"
    )
    assert attempts_reported(backend, run_dir) == [0]
    assert (run_dir / ATTEMPT_FILENAME).read_text().strip() == "1"

    second = backend.submit(spec)
    wait_until(
        lambda: backend.status([second])[second] == "completed",
        what="attempt 1 to finish",
    )
    assert attempts_reported(backend, run_dir) == [0, 1]
    assert (run_dir / ATTEMPT_FILENAME).read_text().strip() == "2"

    # Same run directory, one append-only log, seq never restarted.
    seqs = [event["seq"] for event in read_events(backend, run_dir)]
    assert seqs == list(range(1, len(seqs) + 1))


def test_an_explicit_restart_count_in_spec_env_wins(tmp_path):
    """Precedence: whatever the caller puts in `spec.env` beats the counter.

    `spec.env` is applied on top of the environment the backend builds, so a
    test (or a caller standing in for a scheduler) can pin an attempt number
    without submitting the job that many times. Nobody sets it in normal use.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_pinned")
    spec.env = {"SLURM_RESTART_COUNT": "5"}
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed", what="the job to finish"
    )
    assert attempts_reported(backend, run_dir) == [5]


def test_a_resubmit_clears_a_stale_cancel_flag(tmp_path):
    """Otherwise a new attempt would start out already cancelled."""
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "sleepy.py", SLEEPING_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_stale_cancel")
    run_dir = Path(spec.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CANCEL_FILENAME).touch()

    job_id = backend.submit(spec)
    try:
        assert not (run_dir / CANCEL_FILENAME).exists()
    finally:
        backend.cancel(job_id)


# --------------------------------------------------------------------------
# setup lines
# --------------------------------------------------------------------------
#
# `spec.setup` is `module load cuda`, `conda activate rl`, `export FOO=bar`:
# shell lines that have to have happened before the training command runs. On
# Slurm they go into the sbatch script. Locally they go into the same `sh`
# wrapper that runs the sidecar, in the *same shell process*, which is the only
# arrangement where an `export` survives to reach the child.


def test_setup_exports_reach_the_training_script(tmp_path):
    """`export` in setup is visible three processes down.

    shell -> sidecar -> training script. Nothing copies the variable along; it
    is inherited, which only works because the setup lines run in the shell
    that goes on to exec the sidecar rather than in a subshell of it.
    """
    backend = make_backend(tmp_path)
    script = write_script(
        tmp_path / "setup_env.py",
        "import os\nprint('SETUP_VAR=' + os.environ.get('RELAY_SETUP_VAR', ''))\n",
    )
    spec = make_spec(tmp_path, script, run_id="vr_setup_env")
    spec.setup = ["export RELAY_SETUP_VAR=from_setup"]
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)
    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed",
        what="the setup job to finish",
    )

    lines = [
        event["payload"]["line"]
        for event in read_events(backend, run_dir)
        if event["type"] == "log"
    ]
    assert "SETUP_VAR=from_setup" in lines


def test_setup_lines_run_in_the_order_they_were_given(tmp_path):
    """Verbatim and in order: relay does not reorder or deduplicate them.

    `module load cuda` before `conda activate rl` is not the same as the other
    way round, and relay has no idea which is which -- so it must not shuffle.
    """
    backend = make_backend(tmp_path)
    marker = tmp_path / "setup_order.txt"
    script = write_script(tmp_path / "noop.py", "print('done')\n")
    spec = make_spec(tmp_path, script, run_id="vr_setup_order")
    spec.setup = [f"echo first >> {marker}", f"echo second >> {marker}"]

    job_id = backend.submit(spec)
    wait_until(
        lambda: backend.status([job_id])[job_id] == "completed",
        what="the ordered-setup job to finish",
    )

    assert marker.read_text(encoding="utf-8").split() == ["first", "second"]


def test_a_failing_setup_line_fails_the_run_and_never_starts_the_sidecar(tmp_path):
    """A failed `conda activate` must end the run, not train in the wrong env.

    The subtle part is the *verdict*. Under `set -e` the shell exits the moment
    a setup line fails, which is before the wrapper's usual `printf` at the
    bottom -- so without the EXIT trap there would be no `exit_code` file at
    all, and `status()` would say "unknown" (a process that vanished) instead
    of "failed". "unknown" is not terminal to the daemon, so the run would
    never finish; this test is really about that distinction.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_setup_fail")
    # A second line that would leave a trace, to prove nothing after the
    # failure ran either.
    marker = tmp_path / "must_not_exist.txt"
    spec.setup = ["false", f"echo reached >> {marker}"]
    run_dir = Path(spec.run_dir)

    job_id = backend.submit(spec)

    wait_until(
        lambda: backend.status([job_id])[job_id] == "failed",
        what="the run to report failed",
    )

    # A real, nonzero status was written down, so the verdict survives a
    # daemon restart exactly as an ordinary failure would.
    code = (run_dir / EXIT_CODE_FILENAME).read_text(encoding="utf-8").strip()
    assert code != "0" and int(code) != 0

    # The sidecar never started: it writes `run_started` as its first act, so
    # the absence of the event log is proof it was never reached.
    assert not (run_dir / "events.jsonl").exists()
    assert not marker.exists()

    # And cancelling the wreckage is a no-op, like cancelling any finished job.
    backend.cancel(job_id, run_dir=str(run_dir))
    assert backend.status([job_id]) == {job_id: "failed"}


def test_the_wrapper_is_unchanged_when_there_is_no_setup():
    """No setup, no machinery: the script is the one this backend always ran.

    `set -e` is the thing to check for. The sidecar exits with the training
    script's code, which is routinely nonzero and is *data* -- a wrapper left
    in `set -e` would abort on it before recording it.
    """
    plain = _render_wrapper([])
    assert _render_wrapper(None) == plain
    assert "set -e" not in plain
    assert "EXIT" not in plain
    assert plain.splitlines() == [
        "trap ':' TERM",
        '"$@"',
        "code=$?",
        'printf %s "$code" > "$0.tmp" && mv "$0.tmp" "$0"',
        'exit "$code"',
    ]

    # With setup, the lines land between the guard rails, above the command,
    # and `set -e` is turned back off before the sidecar runs.
    with_setup = _render_wrapper(["module load cuda", "conda activate rl"]).splitlines()
    assert with_setup.index("set -e") < with_setup.index("module load cuda")
    assert with_setup.index("module load cuda") < with_setup.index("conda activate rl")
    assert with_setup.index("conda activate rl") < with_setup.index("set +e")
    assert with_setup.index("set +e") < with_setup.index('"$@"')
    assert with_setup.index("trap - EXIT") < with_setup.index('"$@"')


# --------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------


def test_usage_reports_a_running_job_then_a_completed_one(tmp_path):
    """Wall-clock accounting: the only thing a laptop can honestly measure.

    `sacct` on a cluster is a database that remembers every attempt for weeks.
    Here the start time is recorded at submit because after the process is
    reaped there is nobody left to ask, and the end time is the mtime of the
    file the wrapper writes the moment the sidecar finishes.
    """
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "sleepy.py", SLEEPING_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_usage")
    job_id = backend.submit(spec)

    try:
        rows = backend.usage([job_id])
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, UsageRow)
        assert row.job_id == job_id
        assert row.state == "RUNNING"
        assert row.end is None
        assert row.start.endswith("Z")
        assert row.elapsed_s >= 0
        # No scheduler, so no GPUs were allocated and no partition was landed
        # on. Zero and None are the honest answers, not placeholders.
        assert row.gpus == 0
        assert row.partition is None
        assert row.cpus >= 1

        # A running attempt's elapsed time grows between calls. That is exactly
        # why the store upserts usage rows instead of INSERT OR IGNORE.
        time.sleep(1.1)
        assert backend.usage([job_id])[0].elapsed_s > row.elapsed_s
    finally:
        backend.cancel(job_id)

    wait_until(
        lambda: backend.status([job_id])[job_id] != "running", what="the job to finish"
    )

    finished = backend.usage([job_id])[0]
    # The sidecar exits 0 on the signal path, so the wrapper recorded a 0.
    assert finished.state == "COMPLETED"
    assert finished.end is not None
    assert finished.end >= finished.start
    # And it stops growing once it has stopped.
    assert backend.usage([job_id])[0].elapsed_s == finished.elapsed_s


def test_usage_reports_a_failed_job(tmp_path):
    backend = make_backend(tmp_path)
    script = write_script(tmp_path / "boom.py", FAILING_SCRIPT)
    spec = make_spec(tmp_path, script, run_id="vr_usage_fail")
    job_id = backend.submit(spec)

    wait_until(
        lambda: backend.status([job_id])[job_id] == "failed", what="the job to fail"
    )
    row = backend.usage([job_id])[0]
    assert row.state == "FAILED"
    assert row.end is not None


def test_usage_skips_job_ids_it_has_never_heard_of(tmp_path):
    """No record is not the same claim as no usage.

    A row of zeroes would say "this job ran and consumed nothing", which is a
    different and wronger thing than saying nothing at all.
    """
    backend = make_backend(tmp_path)
    assert backend.usage(["999999999", "not-a-pid"]) == []
    assert backend.usage([]) == []


def test_usage_covers_several_jobs_in_one_call(tmp_path):
    """Batched, for the same reason `status` is: one `sacct` covers every job."""
    backend = make_backend(tmp_path)
    good = write_script(tmp_path / "train.py", GOOD_SCRIPT)
    bad = write_script(tmp_path / "boom.py", FAILING_SCRIPT)
    a = backend.submit(make_spec(tmp_path, good, run_id="vr_u_a"))
    b = backend.submit(make_spec(tmp_path, bad, run_id="vr_u_b"))

    wait_until(
        lambda: all(
            backend.status([a, b])[job] != "running" for job in (a, b)
        ),
        what="both jobs to finish",
    )

    rows = {row.job_id: row for row in backend.usage([a, b, "999999999"])}
    assert set(rows) == {a, b}
    assert rows[a].state == "COMPLETED"
    assert rows[b].state == "FAILED"
