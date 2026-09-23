"""Tests for the daemon loop.

Almost everything here drives the daemon one cycle at a time with
`run_once()` and a scripted `FakeBackend`. That is the whole reason `run_once`
is public: a loop you can only start and stop can only be tested with sleeps,
while a single cycle is an ordinary function call whose effects are visible in
the database the moment it returns.

`FakeBackend` is a dict of path -> bytes plus a dict of job_id -> status, both
of which the test mutates between cycles. It gives us things a real backend
cannot be asked for on demand: a read that fails exactly once, a file that
stops halfway through a JSON object, a scheduler that disagrees with the event
log.

Two tests use the real `LocalBackend` and a real subprocess, because the point
of them is that the daemon works against something it did not stage itself.
Those wait with deadline loops rather than fixed sleeps -- `sleep(0.2); assert`
is a test that fails on a loaded CI machine.
"""

from __future__ import annotations

import inspect
import logging
import stat
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from relay import config as config_module
from relay import daemon as daemon_module
from relay import ssh as ssh_module
from relay.backends.base import Backend, JobSpec, UsageRow
from relay.backends.local import LocalBackend
from relay.backends.slurm import SlurmBackend, TransportError
from relay.daemon import (
    AUTH_CHECK_INTERVAL,
    INTERVAL_ACTIVE,
    INTERVAL_IDLE,
    INTERVAL_QUEUED,
    AlreadyRunning,
    CycleStats,
    Daemon,
    acquire_lock,
    next_interval,
    release_lock,
)
from relay.events import build_event, serialize
from relay.store import TERMINAL, Store, utc_now_iso

TIMEOUT = 30.0
POLL = 0.05


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def wait_until(predicate, timeout: float = TIMEOUT, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(POLL)
    pytest.fail(f"timed out after {timeout}s waiting for {what}")


def ts_ago(seconds: float) -> str:
    """An ISO-8601 UTC timestamp `seconds` in the past, in the schema's format."""
    moment = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def line(run: str, seq: int, type: str, payload: dict | None = None, ts: str | None = None) -> bytes:
    """One complete event log line, newline included."""
    return serialize(build_event(run, seq, type, payload, ts=ts)).encode("utf-8") + b"\n"


def metric_line(run: str, seq: int, step: int, **values) -> bytes:
    return line(run, seq, "metric", {"step": step, "values": values})


class FakeBackend(Backend):
    """A backend made of two dicts, so a test can script the cluster.

    `files` maps a path to its full contents; `read_bytes` slices from the
    offset exactly as a real one does. `statuses` maps a job ID to what the
    scheduler is currently saying. Anything not in `statuses` comes back as
    "unknown", which is what both real backends do for a job they cannot find.
    """

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.statuses: dict[str, str] = {}
        # path -> exception to raise instead of reading. Set it to simulate a
        # dropped SSH connection, clear it to simulate the network coming back.
        self.read_errors: dict[str, BaseException] = {}
        self.status_error: BaseException | None = None
        self.status_calls: list[list[str]] = []
        self.cancelled: list[str] = []
        # Scripted accounting: job_id -> the UsageRows sacct would report for
        # it. A job with several rows is a job that was requeued.
        self.usage_rows: dict[str, list[UsageRow]] = {}
        self.usage_calls: list[list[str]] = []
        self.usage_error: BaseException | None = None

    # -- test-side helpers --

    def append(self, path: str, data: bytes) -> None:
        self.files[path] = self.files.get(path, b"") + data

    def set_usage(self, job_id: str, *, elapsed_s: int = 60, gpus: int = 1, **fields) -> None:
        """Script one attempt for `job_id`, with plausible defaults."""
        row = UsageRow(
            job_id=job_id,
            state=fields.pop("state", "RUNNING"),
            partition=fields.pop("partition", "ckpt-all"),
            start=fields.pop("start", "2026-09-16T18:00:00Z"),
            end=fields.pop("end", None),
            elapsed_s=elapsed_s,
            gpus=gpus,
            cpus=fields.pop("cpus", 4),
        )
        assert not fields, f"unexpected fields {sorted(fields)}"
        self.usage_rows.setdefault(job_id, []).append(row)

    # -- Backend interface --

    def submit(self, spec: JobSpec) -> str:  # pragma: no cover - unused here
        raise NotImplementedError

    def status(self, job_ids: list[str]) -> dict[str, str]:
        self.status_calls.append(list(job_ids))
        if self.status_error is not None:
            raise self.status_error
        return {job_id: self.statuses.get(job_id, "unknown") for job_id in job_ids}

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def usage(self, job_ids: list[str]) -> list[UsageRow]:
        self.usage_calls.append(list(job_ids))
        if self.usage_error is not None:
            raise self.usage_error
        rows: list[UsageRow] = []
        for job_id in job_ids:
            rows.extend(self.usage_rows.get(job_id, []))
        return rows

    def read_bytes(self, path: str, offset: int) -> tuple[bytes, int]:
        error = self.read_errors.get(path)
        if error is not None:
            raise error
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path][offset:]
        return data, offset + len(data)


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "relay.db") as store:
        yield store


@pytest.fixture()
def backend() -> FakeBackend:
    return FakeBackend()


def seed_run(
    store: Store,
    tmp_path: Path,
    run_id: str = "vr_s1_test",
    *,
    job_id: str = "1001",
    status: str = "queued",
) -> tuple[str, str]:
    """Create a run row and return (run_id, events path). Mirrors what submit does."""
    run_dir = tmp_path / run_id
    store.create_run(
        run_id,
        backend="fake",
        job_id=job_id,
        status=status,
        run_dir=str(run_dir),
    )
    return run_id, str(run_dir / daemon_module.EVENTS_FILENAME)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_one_cycle_ingests_events_metrics_and_offset(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.statuses["1001"] = "running"
    backend.append(path, line(run_id, 1, "run_started", {"cmd": ["python", "train.py"]}))
    backend.append(path, metric_line(run_id, 2, 100, loss=0.5, sps=1800))
    backend.append(path, metric_line(run_id, 3, 200, loss=0.4, sps=1900))

    stats = Daemon(store, backend).run_once()

    assert stats.runs_checked == 1
    assert stats.events_parsed == 3
    assert stats.events_inserted == 3
    assert stats.metrics_inserted == 4  # two names x two steps
    assert stats.errors == 0
    assert stats.bad_lines == 0
    assert not stats.failed

    assert store.count_events(run_id) == 3
    assert [e["type"] for e in store.list_events(run_id)] == ["run_started", "metric", "metric"]
    assert store.metric_series(run_id, "loss") == [
        {"step": 100, "value": 0.5, "ts": store.list_events(run_id)[1]["ts"]},
        {"step": 200, "value": 0.4, "ts": store.list_events(run_id)[2]["ts"]},
    ]
    # The whole file was complete, so the offset is its full length.
    assert store.get_offset(run_id, path) == len(backend.files[path])
    assert store.get_run(run_id)["last_step"] == 200
    assert store.last_synced() is not None


def test_only_one_status_call_for_all_runs(store, backend, tmp_path):
    seed_run(store, tmp_path, "vr_a", job_id="1")
    seed_run(store, tmp_path, "vr_b", job_id="2")
    seed_run(store, tmp_path, "vr_c", job_id="3")

    Daemon(store, backend).run_once()

    # One call, every job in it. Per-run calls would be one SSH round trip each.
    assert len(backend.status_calls) == 1
    assert sorted(backend.status_calls[0]) == ["1", "2", "3"]


def test_second_cycle_reads_only_what_is_new(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))
    daemon = Daemon(store, backend)
    daemon.run_once()

    backend.append(path, metric_line(run_id, 2, 10, loss=1.0))
    stats = daemon.run_once()

    assert stats.events_parsed == 1
    assert stats.events_inserted == 1
    assert store.count_events(run_id) == 2


def test_unparseable_lines_are_counted_and_skipped(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))
    backend.append(path, b"this is not json at all\n")
    backend.append(path, b'{"v":1,"run":"x"}\n')  # valid JSON, bad envelope
    backend.append(path, metric_line(run_id, 2, 10, loss=1.0))

    stats = Daemon(store, backend).run_once()

    assert stats.bad_lines == 2
    assert stats.events_inserted == 2
    # A corrupt line must not stop the good lines behind it from loading.
    assert store.count_events(run_id) == 2
    assert store.get_offset(run_id, path) == len(backend.files[path])


# --------------------------------------------------------------------------
# Restart: killed mid-cycle, then started again
# --------------------------------------------------------------------------


def test_restarted_daemon_reingests_without_duplicating(store, backend, tmp_path):
    """A daemon killed mid-cycle re-reads bytes it already saw. That must be free.

    We model the kill by rewinding the stored byte offset back to zero, which
    is exactly the state a crash between the ingest and the offset update would
    leave behind if they were not in one transaction. A brand new `Daemon`
    (with no memory of the first one) then re-reads the whole file plus the
    lines that arrived in the meantime.
    """
    run_id, path = seed_run(store, tmp_path)
    for seq in range(1, 4):
        backend.append(path, metric_line(run_id, seq, seq * 10, loss=1.0 / seq))

    Daemon(store, backend).run_once()
    assert store.count_events(run_id) == 3

    # "The daemon died before its offset update committed."
    store.ingest(run_id, path, [], 0)
    assert store.get_offset(run_id, path) == 0

    backend.append(path, metric_line(run_id, 4, 40, loss=0.25))

    restarted = Daemon(store, backend)
    stats = restarted.run_once()

    # It parsed all four lines, but only one of them was new to the database.
    assert stats.events_parsed == 4
    assert stats.events_inserted == 1
    assert store.count_events(run_id) == 4
    assert [e["seq"] for e in store.list_events(run_id)] == [1, 2, 3, 4]
    assert store.count_metrics(run_id) == 4
    assert store.get_offset(run_id, path) == len(backend.files[path])


# --------------------------------------------------------------------------
# Torn lines
# --------------------------------------------------------------------------


def test_incomplete_final_line_rewinds_the_offset(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    complete = line(run_id, 1, "run_started") + metric_line(run_id, 2, 10, loss=1.0)
    torn = metric_line(run_id, 3, 20, loss=0.5)
    # The filesystem has only made the first fifteen bytes of the third line
    # visible to us -- mid-JSON-object.
    backend.append(path, complete + torn[:15])

    daemon = Daemon(store, backend)
    stats = daemon.run_once()

    assert stats.events_inserted == 2
    assert store.count_events(run_id) == 2
    # The offset stops at the end of the last *complete* line, not at the end
    # of what we read. Storing the further offset would lose event 3 forever.
    assert store.get_offset(run_id, path) == len(complete)

    # The rest of the line lands.
    backend.append(path, torn[15:])
    stats = daemon.run_once()

    assert stats.events_parsed == 1
    assert stats.events_inserted == 1
    assert stats.bad_lines == 0
    assert store.count_events(run_id) == 3
    assert store.get_offset(run_id, path) == len(backend.files[path])


def test_chunk_with_no_newline_at_all_makes_no_progress(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, b'{"v":1,"run":"vr_s1_test","seq":1,')

    stats = Daemon(store, backend).run_once()

    assert stats.events_parsed == 0
    assert stats.bad_lines == 0  # a partial line is not a bad line
    assert store.get_offset(run_id, path) == 0


# --------------------------------------------------------------------------
# Transport failures are normal
# --------------------------------------------------------------------------


def test_read_failure_on_one_run_does_not_stop_the_others(store, backend, tmp_path):
    broken_id, broken_path = seed_run(store, tmp_path, "vr_broken", job_id="1")
    ok_id, ok_path = seed_run(store, tmp_path, "vr_ok", job_id="2")
    backend.append(broken_path, line(broken_id, 1, "run_started"))
    backend.append(ok_path, line(ok_id, 1, "run_started"))
    backend.read_errors[broken_path] = ConnectionError("ssh: connection closed")

    daemon = Daemon(store, backend)
    stats = daemon.run_once()

    assert stats.errors == 1
    assert stats.read_failures == 1
    # One run failing is not a failed cycle: the channel is clearly up.
    assert not stats.failed
    assert store.count_events(broken_id) == 0
    assert store.count_events(ok_id) == 1

    # The network comes back; nothing else had to happen for recovery.
    backend.read_errors.clear()
    stats = daemon.run_once()

    assert stats.errors == 0
    assert store.count_events(broken_id) == 1


def test_status_failure_still_lets_events_be_read(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))
    backend.status_error = ConnectionError("squeue: unreachable")

    stats = Daemon(store, backend).run_once()

    assert stats.status_failed
    assert stats.errors == 1
    assert store.count_events(run_id) == 1
    # The event log lives on the filesystem; a broken scheduler command says
    # nothing about whether we can read it.
    assert not stats.failed


def test_missing_event_file_is_not_an_error(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.statuses["1001"] = "queued"
    # No file: the job is still in the queue and has not made its directory.

    stats = Daemon(store, backend).run_once()

    assert stats.errors == 0
    assert stats.read_failures == 0
    assert not stats.failed
    run = store.get_run(run_id)
    assert run["status"] == "queued"
    assert run["stale"] is False
    assert store.get_offset(run_id, path) == 0


def test_ingest_failure_leaves_the_offset_alone(store, backend, tmp_path, monkeypatch):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))

    def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(store, "ingest", boom)
    stats = Daemon(store, backend).run_once()

    assert stats.errors == 1
    assert stats.events_inserted == 0

    monkeypatch.undo()
    # Because the offset never moved, the next cycle re-reads the same bytes.
    stats = Daemon(store, backend).run_once()
    assert stats.events_inserted == 1


# --------------------------------------------------------------------------
# Stale detection
# --------------------------------------------------------------------------


def test_stale_is_flagged_after_the_threshold_and_cleared_by_a_heartbeat(
    store, backend, tmp_path
):
    run_id, path = seed_run(store, tmp_path)
    backend.statuses["1001"] = "running"
    backend.append(path, line(run_id, 1, "run_started"))
    backend.append(path, line(run_id, 2, "heartbeat", {"last_step": 10}, ts=ts_ago(600)))

    daemon = Daemon(store, backend, stale_after_seconds=60)
    stats = daemon.run_once()

    run = store.get_run(run_id)
    assert stats.stale_flagged == 1
    assert run["stale"] is True
    # Stale is a flag, not a status. The scheduler still owns the status.
    assert run["status"] == "running"

    backend.append(path, line(run_id, 3, "heartbeat", {"last_step": 20}))
    stats = daemon.run_once()

    assert stats.stale_cleared == 1
    assert store.get_run(run_id)["stale"] is False
    assert store.get_run(run_id)["status"] == "running"


def test_a_run_with_no_heartbeat_yet_is_not_stale(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.statuses["1001"] = "running"
    backend.append(path, line(run_id, 1, "run_started"))

    stats = Daemon(store, backend, stale_after_seconds=0.0).run_once()

    # We have no evidence either way, and "I cannot tell" must not raise an
    # alarm. Measuring from submission instead would flag every job that
    # waited in the queue longer than the threshold.
    assert stats.stale_flagged == 0
    assert store.get_run(run_id)["stale"] is False


def test_a_queued_run_is_never_stale(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.statuses["1001"] = "queued"
    backend.append(path, line(run_id, 1, "heartbeat", {"last_step": 1}, ts=ts_ago(9999)))

    stats = Daemon(store, backend, stale_after_seconds=1).run_once()

    assert stats.stale_flagged == 0
    assert store.get_run(run_id)["stale"] is False


# --------------------------------------------------------------------------
# Status: the scheduler is the authority
# --------------------------------------------------------------------------


def test_scheduler_ends_a_run_that_never_wrote_run_ended(store, backend, tmp_path):
    """A SIGKILLed job writes no `run_ended`. Only Slurm can tell us it is over."""
    run_id, path = seed_run(store, tmp_path, status="running")
    backend.statuses["1001"] = "completed"
    backend.append(path, line(run_id, 1, "run_started"))
    backend.append(path, metric_line(run_id, 2, 10, loss=1.0))

    stats = Daemon(store, backend).run_once()

    assert stats.status_changes == 1
    run = store.get_run(run_id)
    assert run["status"] == "completed"
    assert run["terminal"] is True
    # The final bytes were read in the same cycle that ended the run: status is
    # sampled before the read, so the log was already complete when we read it.
    assert store.count_events(run_id) == 2
    # And the run drops out of the working set from here on.
    assert store.list_runs(active_only=True) == []


def test_scheduler_beats_a_run_ended_in_the_log(store, backend, tmp_path):
    """The log says `requeued`; the scheduler says cancelled. The scheduler wins.

    This is what `relay cancel` looks like from the daemon's side: the sidecar
    catches SIGTERM, writes `run_ended` with status "requeued" and exits, but
    the job was cancelled, not preempted, and it is not coming back.
    """
    run_id, path = seed_run(store, tmp_path, status="running")
    backend.statuses["1001"] = "cancelled"
    backend.append(path, line(run_id, 1, "run_started"))
    backend.append(path, line(run_id, 2, "signal", {"signal": "SIGTERM"}))
    backend.append(path, line(run_id, 3, "run_ended", {"status": "requeued"}))

    Daemon(store, backend).run_once()

    run = store.get_run(run_id)
    assert run["status"] == "cancelled"
    # The log is still the faithful history of what the training did.
    assert [e["type"] for e in store.list_events(run_id)] == [
        "run_started",
        "signal",
        "run_ended",
    ]


def test_unknown_status_does_not_overwrite_a_known_one(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path, status="running")
    backend.statuses["1001"] = "unknown"
    backend.append(path, line(run_id, 1, "run_started"))

    stats = Daemon(store, backend).run_once()

    assert stats.status_changes == 0
    # "unknown" is the absence of information, not a state to record.
    assert store.get_run(run_id)["status"] == "running"


def test_queued_becomes_running_when_the_scheduler_says_so(store, backend, tmp_path):
    run_id, _ = seed_run(store, tmp_path)
    backend.statuses["1001"] = "running"

    stats = Daemon(store, backend).run_once()

    assert stats.status_changes == 1
    assert stats.running == 1
    assert store.get_run(run_id)["status"] == "running"


def test_a_run_with_no_job_id_is_left_alone(store, backend, tmp_path):
    store.create_run("vr_nojob", backend="fake", status="queued")

    stats = Daemon(store, backend).run_once()

    assert backend.status_calls == []  # nothing to ask about
    assert stats.status_changes == 0
    assert store.get_run("vr_nojob")["status"] == "queued"


# --------------------------------------------------------------------------
# Adaptive interval
# --------------------------------------------------------------------------


def test_next_interval_active():
    assert next_interval(CycleStats(runs_checked=1, running=1)) == INTERVAL_ACTIVE
    # Bytes arrived, even if the scheduler has not caught up yet.
    assert next_interval(CycleStats(runs_checked=1, bytes_read=512)) == INTERVAL_ACTIVE


def test_next_interval_queued():
    assert next_interval(CycleStats(runs_checked=3, queued=3)) == INTERVAL_QUEUED


def test_next_interval_idle():
    assert next_interval(CycleStats()) == INTERVAL_IDLE


def test_next_interval_backs_off_exponentially_and_caps():
    intervals = [next_interval(CycleStats(consecutive_failures=n)) for n in range(1, 5)]
    assert intervals == [4.0, 8.0, 16.0, 32.0]
    assert next_interval(CycleStats(consecutive_failures=99)) == daemon_module.BACKOFF_CAP
    # Backoff outranks "something is running": if the transport is down we
    # cannot see what is running anyway.
    assert next_interval(CycleStats(running=1, consecutive_failures=2)) == 8.0


def test_consecutive_failures_grow_then_reset(store, backend, tmp_path):
    # A run with no run_dir: nothing to read, so the only thing we can do this
    # cycle is ask the scheduler -- and that is what is broken.
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = ConnectionError("ssh: Operation timed out")

    # Every cycle must actually reach the scheduler for this test to be about
    # backoff rather than about scheduling: with relay's real 30s floor, cycles
    # two and three would skip the poll and so neither fail nor succeed.
    daemon = Daemon(store, backend, scheduler_interval_min=0, scheduler_interval_max=0)
    sleeps = []
    for _ in range(3):
        stats = daemon.run_once()
        assert stats.failed
        sleeps.append(next_interval(stats))

    assert sleeps == [4.0, 8.0, 16.0]
    assert daemon.consecutive_failures == 3
    # A failed cycle must not claim to have synced.
    assert store.last_synced() is None

    backend.status_error = None
    stats = daemon.run_once()

    assert not stats.failed
    assert stats.consecutive_failures == 0
    assert daemon.consecutive_failures == 0
    assert next_interval(stats) == INTERVAL_QUEUED
    assert store.last_synced() is not None


def test_next_interval_auth_required_outranks_everything():
    # Not a retry schedule: we are looking at a local socket, not the cluster,
    # and when the user finally runs `relay connect` we want to notice fast.
    assert next_interval(CycleStats(auth_required=True)) == AUTH_CHECK_INTERVAL
    assert next_interval(CycleStats(auth_required=True, consecutive_failures=5)) == 60.0


# --------------------------------------------------------------------------
# Usage accounting: one batched call, every five minutes, once more at the end
# --------------------------------------------------------------------------


def test_usage_is_fetched_on_the_first_cycle_and_filed_by_run(store, backend, tmp_path):
    seed_run(store, tmp_path, "vr_a", job_id="1")
    seed_run(store, tmp_path, "vr_b", job_id="2")
    backend.set_usage("1", elapsed_s=120, gpus=2)
    backend.set_usage("2", elapsed_s=60, gpus=1)
    # A row for a job we never asked about. There is no run to file it under,
    # and inventing one would corrupt somebody's hours, so it is dropped.
    backend.usage_rows["1"].append(
        UsageRow(
            job_id="999",
            state="COMPLETED",
            partition="ckpt",
            start="2026-09-16T17:00:00Z",
            end="2026-09-16T17:30:00Z",
            elapsed_s=1800,
            gpus=8,
            cpus=8,
        )
    )

    stats = Daemon(store, backend).run_once()

    # One call for both runs, same reason `status` is batched: one SSH round
    # trip, not one per run.
    assert len(backend.usage_calls) == 1
    assert sorted(backend.usage_calls[0]) == ["1", "2"]
    assert stats.usage_fetched
    assert stats.usage_rows == 2  # the third row had nowhere to go

    rows_a = store.list_usage("vr_a")
    assert [(r["job_id"], r["run_id"], r["gpus"]) for r in rows_a] == [("1", "vr_a", 2)]
    assert [(r["job_id"], r["run_id"]) for r in store.list_usage("vr_b")] == [("2", "vr_b")]
    assert store.get_daemon_state()["last_usage_fetch_ts"] is not None


def test_usage_is_not_fetched_again_within_the_interval(store, backend, tmp_path):
    seed_run(store, tmp_path, "vr_a", job_id="1")
    backend.set_usage("1")

    daemon = Daemon(store, backend)
    daemon.run_once()
    daemon.run_once()
    daemon.run_once()

    # Usage is accounting, not live data. Three cycles, one sacct.
    assert len(backend.usage_calls) == 1


def test_usage_is_fetched_again_once_the_interval_has_passed(store, backend, tmp_path):
    seed_run(store, tmp_path, "vr_a", job_id="1")
    backend.set_usage("1")

    daemon = Daemon(store, backend)
    daemon.run_once()

    # Rather than waiting five minutes, move the recorded time backwards: the
    # daemon reads it out of the database, so an old timestamp is
    # indistinguishable from time having passed.
    store.set_daemon_state(last_usage_fetch_ts=ts_ago(600))
    daemon.run_once()

    assert len(backend.usage_calls) == 2
    # ISO-8601 UTC sorts correctly as text, which is why we store it that way.
    assert store.get_daemon_state()["last_usage_fetch_ts"] > ts_ago(60)


def test_a_tiny_usage_interval_fetches_every_cycle(store, backend, tmp_path):
    """The interval is a constructor argument purely so tests can shrink it."""
    seed_run(store, tmp_path, "vr_a", job_id="1")
    backend.set_usage("1")

    daemon = Daemon(store, backend, usage_interval_seconds=0.0)
    daemon.run_once()
    time.sleep(0.01)
    daemon.run_once()

    assert len(backend.usage_calls) == 2


def test_a_run_going_terminal_is_charged_once_more_then_never_again(store, backend, tmp_path):
    finished_id, _ = seed_run(store, tmp_path, "vr_done", job_id="1", status="running")
    live_id, _ = seed_run(store, tmp_path, "vr_live", job_id="2", status="running")
    backend.statuses["1"] = "completed"
    backend.statuses["2"] = "running"
    backend.set_usage("1", elapsed_s=300, gpus=2, state="COMPLETED", end="2026-09-16T18:05:00Z")
    backend.set_usage("2")

    daemon = Daemon(store, backend)
    # Pretend a fetch happened a moment ago, so the five-minute timer is not
    # due: the only thing that can trigger this fetch is the run finishing.
    store.set_daemon_state(last_usage_fetch_ts=utc_now_iso())

    daemon.run_once()

    assert len(backend.usage_calls) == 1
    assert sorted(backend.usage_calls[0]) == ["1", "2"]
    assert store.list_usage(finished_id)[0]["elapsed_s"] == 300
    # A finished job's sacct numbers never change again, so that reading was
    # the last one and the flag says so.
    assert store.get_run(finished_id)["usage_final"] is True
    assert store.get_run(live_id)["usage_final"] is False

    # And from here on the finished run is not even mentioned in the call.
    store.set_daemon_state(last_usage_fetch_ts=ts_ago(600))
    daemon.run_once()

    assert backend.usage_calls[-1] == ["2"]


def test_a_failed_usage_call_does_not_block_ingest(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))
    backend.append(path, metric_line(run_id, 2, 10, loss=1.0))
    backend.usage_error = ConnectionError("sacct: connection closed by remote host")

    stats = Daemon(store, backend).run_once()

    assert stats.usage_failed
    assert not stats.usage_fetched
    assert stats.usage_rows == 0
    # The live data is what the user is watching; accounting can wait.
    assert store.count_events(run_id) == 2
    assert store.count_metrics(run_id) == 1
    # No timestamp was written, so the next cycle tries again rather than
    # waiting out an interval for a fetch that never happened.
    assert store.get_daemon_state()["last_usage_fetch_ts"] is None
    assert store.get_daemon_state()["last_error_kind"] == "unreachable"


# --------------------------------------------------------------------------
# Classifying ssh failures
# --------------------------------------------------------------------------


@pytest.fixture()
def master(monkeypatch):
    """Control what `ssh -O check` reports, and record every call to it.

    The daemon calls `ssh.master_alive`, so patching the attribute on the
    module object is enough -- no socket, no subprocess, no Duo.

    The recorder keeps each call's `timeout`, and the teardown below asserts
    that no test in this file ever saw the daemon pass one under
    `ssh.DEFAULT_SSH_TIMEOUT`. That is the whole point of consolidating on a
    single `ssh_timeout`: a call that quietly used a tighter literal of its own
    would give up on a login node that every other call in relay is still
    waiting patiently for, and nothing but an assertion like this would notice.
    """

    class FakeMaster:
        def __init__(self) -> None:
            self.alive = False
            self.calls = 0
            # One entry per call, in order: the `timeout` the daemon passed.
            self.timeouts: list[float | None] = []

        def __call__(self, alias, runner=None, timeout=None):
            self.calls += 1
            self.timeouts.append(timeout)
            return self.alive

    fake = FakeMaster()
    monkeypatch.setattr(ssh_module, "master_alive", fake)
    yield fake

    assert all(
        isinstance(t, (int, float)) and t >= ssh_module.DEFAULT_SSH_TIMEOUT
        for t in fake.timeouts
    ), (
        f"the daemon called master_alive with {fake.timeouts}; every remote "
        f"call must use the backend's ssh_timeout, never a literal tighter "
        f"than {ssh_module.DEFAULT_SSH_TIMEOUT}s"
    )


def in_auth_required(store: Store) -> None:
    """Put the stored daemon state where a Duo-expired connection would."""
    store.set_daemon_state(
        last_error_kind="auth_required",
        last_error_ts=utc_now_iso(),
        last_error_detail="ssh exited 255",
    )


def test_exit_255_with_no_live_master_is_auth_required(store, backend, master):
    backend.ssh_alias = "klone"
    master.alive = False
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = TransportError(
        "ssh exited 255 running squeue: " + "x" * 500, returncode=255, stderr=""
    )

    Daemon(store, backend).run_once()

    state = store.get_daemon_state()
    assert state["last_error_kind"] == "auth_required"
    assert state["last_error_ts"] is not None
    assert state["last_error_detail"].startswith("ssh exited 255")
    # A state row is not a log file.
    assert len(state["last_error_detail"]) <= daemon_module.ERROR_DETAIL_CHARS


def test_exit_255_with_a_live_master_is_unreachable(store, backend, master):
    backend.ssh_alias = "klone"
    master.alive = True  # the master is up, so this is the network, not Duo
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = TransportError("ssh exited 255", returncode=255, stderr="")

    Daemon(store, backend).run_once()

    assert store.get_daemon_state()["last_error_kind"] == "unreachable"


def test_permission_denied_is_auth_required_whatever_the_master_says(store, backend, master):
    backend.ssh_alias = "klone"
    master.alive = True
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = TransportError(
        "ssh failed",
        returncode=255,
        stderr="klone: Permission denied (publickey,keyboard-interactive).",
    )

    Daemon(store, backend).run_once()

    # The server refused the credentials outright. No amount of retrying fixes
    # that, whatever the control socket currently thinks.
    assert store.get_daemon_state()["last_error_kind"] == "auth_required"


def test_a_non_ssh_backend_failure_is_unreachable_and_checks_no_socket(store, backend, master):
    # FakeBackend has no `ssh_alias`, exactly like LocalBackend.
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = ConnectionError("connection reset by peer")

    Daemon(store, backend).run_once()

    assert store.get_daemon_state()["last_error_kind"] == "unreachable"
    # There is no master to check, so we must not have gone looking for one.
    assert master.calls == 0


def test_only_one_classification_per_cycle(store, backend, tmp_path, master):
    backend.ssh_alias = "klone"
    master.alive = True
    for n in (1, 2, 3):
        _, path = seed_run(store, tmp_path, f"vr_{n}", job_id=str(n))
        backend.append(path, line(f"vr_{n}", 1, "run_started"))
        backend.read_errors[path] = TransportError("ssh died", returncode=255, stderr="")

    Daemon(store, backend).run_once()

    # Three failed reads, one verdict: the second failure this cycle has the
    # same cause as the first, and `ssh -O check` is a process spawn.
    assert master.calls == 1
    assert store.get_daemon_state()["last_error_kind"] == "unreachable"


def test_a_good_cycle_clears_a_previous_error_and_records_last_ok(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))
    store.set_daemon_state(
        last_error_kind="unreachable",
        last_error_ts=ts_ago(60),
        last_error_detail="ssh: connection reset",
    )

    stats = Daemon(store, backend).run_once()

    # `unreachable` never stopped us polling, so the cycle ran normally...
    assert len(backend.status_calls) == 1
    assert stats.error_kind is None
    state = store.get_daemon_state()
    # ...and clearing has to be as prompt as setting, or the CLI keeps showing
    # an error that is over.
    assert state["last_error_kind"] is None
    assert state["last_error_ts"] is None
    assert state["last_error_detail"] is None
    assert state["last_ok_ts"] is not None


# --------------------------------------------------------------------------
# auth_required: stop talking to the cluster, watch the socket instead
# --------------------------------------------------------------------------


def test_auth_required_makes_a_cycle_touch_nothing(store, backend, tmp_path, master):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))
    backend.ssh_alias = "klone"
    master.alive = False
    in_auth_required(store)

    stats = Daemon(store, backend).run_once()

    assert stats.auth_required
    # Zero backend calls: every one of them would fail the same way, and
    # without BatchMode they would hang waiting for a Duo prompt nobody sees.
    assert backend.status_calls == []
    assert backend.usage_calls == []
    assert store.count_events(run_id) == 0
    # The one thing we do is look at our own socket.
    assert master.calls == 1
    assert next_interval(stats) == AUTH_CHECK_INTERVAL == 60.0
    assert store.get_daemon_state()["last_error_kind"] == "auth_required"
    # A cycle that did nothing must not claim to have synced.
    assert store.last_synced() is None


def test_a_live_master_clears_the_lockout_and_the_same_cycle_continues(
    store, backend, tmp_path, master
):
    run_id, path = seed_run(store, tmp_path)
    backend.append(path, line(run_id, 1, "run_started"))
    backend.statuses["1001"] = "running"
    backend.ssh_alias = "klone"
    master.alive = True  # the user just ran `relay connect`
    in_auth_required(store)

    stats = Daemon(store, backend).run_once()

    assert not stats.auth_required
    # Not "we will start again next minute" -- this cycle does the work.
    assert len(backend.status_calls) == 1
    assert store.count_events(run_id) == 1
    assert store.get_run(run_id)["status"] == "running"
    state = store.get_daemon_state()
    assert state["last_error_kind"] is None
    assert state["last_ok_ts"] is not None


def test_the_socket_check_is_rate_limited_to_once_a_minute(store, backend, master):
    backend.ssh_alias = "klone"
    master.alive = False
    in_auth_required(store)

    daemon = Daemon(store, backend)
    first = daemon.run_once()
    second = daemon.run_once()

    assert first.auth_required and second.auth_required
    assert master.calls == 1


def test_shrinking_the_check_interval_checks_every_cycle(store, backend, master):
    backend.ssh_alias = "klone"
    master.alive = False
    in_auth_required(store)

    daemon = Daemon(store, backend, auth_check_interval=0.0)
    daemon.run_once()
    daemon.run_once()

    assert master.calls == 2


# --------------------------------------------------------------------------
# One timeout everywhere: the socket check uses the backend's ssh_timeout
# --------------------------------------------------------------------------


def test_the_auth_gate_checks_the_socket_with_the_backends_ssh_timeout(
    store, backend, master
):
    """The lockout gate's `ssh -O check` waits as long as the config says.

    `ssh_timeout` is one number for the whole of relay, and the backend is
    where it lives (it is the thing that owns the connection). The daemon must
    read it off the backend rather than pass a number of its own, so a user who
    raises the timeout for a slow login node raises it for every call relay
    makes, this one included.
    """
    backend.ssh_alias = "klone"
    backend.ssh_timeout = 45
    master.alive = False
    in_auth_required(store)

    stats = Daemon(store, backend).run_once()

    assert stats.auth_required
    assert master.timeouts == [45]


def test_classifying_a_failure_checks_the_socket_with_the_backends_ssh_timeout(
    store, backend, master
):
    """The other `master_alive` call site -- failure classification -- too.

    This is the call that decides between `auth_required` and `unreachable`,
    and it runs in exactly the circumstances where the login node is unwell. A
    shorter timeout here would report "no master" for a master that was merely
    slow to answer, and send the user off to run `relay connect` for nothing.
    """
    backend.ssh_alias = "klone"
    backend.ssh_timeout = 45
    master.alive = True
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = TransportError("ssh exited 255", returncode=255, stderr="")

    Daemon(store, backend).run_once()

    assert store.get_daemon_state()["last_error_kind"] == "unreachable"
    assert master.timeouts == [45]


def test_a_backend_with_no_ssh_timeout_falls_back_to_the_default(store, backend, master):
    """A backend that never heard of `ssh_timeout` still gets a sane number.

    `SlurmBackend` carries the attribute; a hand-rolled or older backend need
    not, and the daemon must not raise `AttributeError` on one. The fallback is
    the same default the config uses, so nothing silently gets its own idea of
    how long is too long.
    """
    backend.ssh_alias = "klone"
    assert not hasattr(backend, "ssh_timeout")
    master.alive = False
    in_auth_required(store)

    Daemon(store, backend).run_once()

    assert master.timeouts == [ssh_module.DEFAULT_SSH_TIMEOUT]
    assert ssh_module.DEFAULT_SSH_TIMEOUT == 30.0


# --------------------------------------------------------------------------
# `interrupted`: the sidecar refuses to guess, and so does the daemon
# --------------------------------------------------------------------------


def test_an_interrupted_run_that_slurm_requeued_stays_in_the_working_set(store, backend, tmp_path):
    """SIGTERM could have been preemption or `scancel`. Slurm knows which."""
    run_id, path = seed_run(store, tmp_path, status="running")
    backend.statuses["1001"] = "queued"  # preempted, and put back in the queue
    backend.append(path, line(run_id, 1, "run_started", {"attempt": 0}))
    backend.append(path, line(run_id, 2, "signal", {"signal": "SIGTERM", "reason": "term"}))
    backend.append(path, line(run_id, 3, "run_ended", {"status": "interrupted"}))

    Daemon(store, backend).run_once()

    run = store.get_run(run_id)
    # `run_ended` in the log does not end a run. The same run ID is about to
    # start writing to this same file again under a new attempt.
    assert run["status"] == "queued"
    assert run["terminal"] is False
    assert [r["run_id"] for r in store.list_runs(active_only=True)] == [run_id]
    # The log is still the faithful history of what the training did.
    assert [e["type"] for e in store.list_events(run_id)] == [
        "run_started",
        "signal",
        "run_ended",
    ]


def test_an_interrupted_run_that_slurm_cancelled_is_terminal(store, backend, tmp_path):
    run_id, path = seed_run(store, tmp_path, status="running")
    backend.statuses["1001"] = "cancelled"  # the same log, a different verdict
    backend.append(path, line(run_id, 1, "run_started", {"attempt": 0}))
    backend.append(path, line(run_id, 2, "run_ended", {"status": "interrupted"}))

    Daemon(store, backend).run_once()

    run = store.get_run(run_id)
    assert run["status"] == "cancelled"
    assert run["terminal"] is True
    assert store.list_runs(active_only=True) == []
    # Terminal this cycle, so it got its final usage reading immediately.
    assert store.get_run(run_id)["usage_final"] is True


# --------------------------------------------------------------------------
# The single-instance lock
# --------------------------------------------------------------------------


def test_lock_is_exclusive(tmp_path):
    path = tmp_path / "daemon.lock"
    first = acquire_lock(path)
    try:
        # Two `open()` calls create two open file descriptions, and flock locks
        # belong to the description rather than to the process -- so this
        # really does conflict even though it is the same interpreter. (POSIX
        # record locks, fcntl.lockf, are per process and would silently
        # succeed, which is why the daemon uses flock.)
        with pytest.raises(AlreadyRunning):
            acquire_lock(path)
    finally:
        release_lock(first)


def test_lock_is_reusable_after_release(tmp_path):
    path = tmp_path / "daemon.lock"
    first = acquire_lock(path)
    release_lock(first)
    # A daemon that exits -- or crashes -- leaves the lock free. A PID file
    # would not.
    second = acquire_lock(path)
    release_lock(second)


def test_lock_file_records_the_pid(tmp_path):
    import os

    path = tmp_path / "daemon.lock"
    handle = acquire_lock(path)
    try:
        assert path.read_text().strip() == str(os.getpid())
    finally:
        release_lock(handle)


def test_lock_file_is_rewritten_not_appended(tmp_path):
    path = tmp_path / "daemon.lock"
    path.write_text("99999\n99999\n99999\n")
    handle = acquire_lock(path)
    try:
        assert len(path.read_text().strip().splitlines()) == 1
    finally:
        release_lock(handle)


# --------------------------------------------------------------------------
# run_forever, against a real local job
# --------------------------------------------------------------------------

FAKE_TRAINING_SCRIPT = """\
import sys, time
for step in range(5):
    print('##relay## {"step": %d, "loss": %f}' % (step, 1.0 / (step + 1)), flush=True)
    time.sleep(0.02)
print("training finished", flush=True)
"""


def test_run_forever_tails_a_real_local_run_until_stopped(tmp_path):
    """A miniature end-to-end: submit locally, let the loop run, stop it cleanly.

    Everything is real here except the interval overrides -- a real sidecar in
    a real subprocess, a real `LocalBackend`, a real SQLite file.

    The daemon opens its `Store` *inside* its own thread. A Store owns one
    sqlite3 connection, and sqlite3 refuses to let a connection be used from a
    thread other than the one that created it -- which is the same rule the
    real program obeys by having the daemon be its own process. The test reads
    through a second Store on the same file, which is exactly the arrangement
    WAL mode exists for.
    """
    script = tmp_path / "train.py"
    script.write_text(FAKE_TRAINING_SCRIPT, encoding="utf-8")

    db_path = tmp_path / "relay.db"
    backend = LocalBackend(root=tmp_path / "backend")
    run_id = "vr_local_1"
    run_dir = tmp_path / "runs" / run_id

    import sys

    job_id = backend.submit(
        JobSpec(
            run_id=run_id,
            command=[sys.executable, str(script)],
            run_dir=str(run_dir),
            heartbeat_seconds=1,
        )
    )

    reader = Store(db_path)
    reader.create_run(
        run_id,
        backend="local",
        job_id=job_id,
        status="queued",
        run_dir=str(run_dir),
    )

    stop = threading.Event()
    daemons: list[Daemon] = []

    def loop() -> None:
        with Store(db_path) as daemon_store:
            daemon = Daemon(
                daemon_store,
                backend,
                interval_active=0.05,
                interval_queued=0.05,
                interval_idle=0.05,
                # The scheduler poll is what moves a run to a terminal status,
                # and its real floor is 30 seconds. Shrink it along with the
                # tail intervals, or this test waits out the floor.
                scheduler_interval_min=0.05,
                scheduler_interval_max=0.05,
            )
            daemons.append(daemon)
            daemon.run_forever(stop)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    try:
        wait_until(
            lambda: (reader.get_run(run_id) or {}).get("status") in TERMINAL,
            what="the run to reach a terminal status",
        )
        wait_until(
            lambda: reader.count_metrics(run_id) >= 5,
            what="all five metric points to be ingested",
        )
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not thread.is_alive(), "run_forever did not return after stop_event was set"
    assert daemons and daemons[0].cycles > 1

    run = reader.get_run(run_id)
    assert run["status"] == "completed"
    assert run["last_step"] == 4
    assert reader.metric_series(run_id, "loss")[0]["step"] == 0
    types = {e["type"] for e in reader.list_events(run_id)}
    assert "run_started" in types and "run_ended" in types
    assert reader.last_synced() is not None

    reader.close()


def test_run_forever_returns_immediately_if_already_stopped(store, backend):
    stop = threading.Event()
    stop.set()
    daemon = Daemon(store, backend)

    assert daemon.run_forever(stop) == 0
    assert daemon.cycles == 0


# --------------------------------------------------------------------------
# The ControlPath directory, from the daemon's side
# --------------------------------------------------------------------------


def test_a_daemon_cycle_creates_the_ssh_control_directory(
    store, tmp_path, monkeypatch
):
    """The daemon's first ssh invocation must create `~/.relay` by itself.

    The first real run on Klone failed with `unix_listener: cannot bind to
    path ~/.relay/cm-...: No such file or directory`, because nothing had ever
    created the ControlPath's parent directory. The fix lives in
    `ssh.ensure_control_dir()`, called from every argv builder rather than from
    `relay connect` alone -- and this test is why it cannot live only in
    `connect`. The daemon and `doctor` can both run on a machine where
    `relay connect` has never run: a daemon started at login, a `doctor` run as
    a CI smoke test. Either would hit the same bind failure.

    So: a real `SlurmBackend` (fake `runner`, nothing is spawned), HOME
    redirected at an empty temporary directory, one `run_once()`, and the
    directory has to be there afterwards.

    Note what this test does *not* prove. The runner is a fake, so no socket is
    ever bound; a fake can only ever show that the argv was built and the
    directory appeared. The real bind is exercised in
    `tests/test_ssh_integration.py`, which needs a local sshd and skips
    without one.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    control_dir = tmp_path / ".relay"
    assert not control_dir.exists()

    calls: list[list[str]] = []

    def runner(argv, input_bytes, timeout):
        # Success with no output: squeue lists no live jobs, sacct knows
        # nothing, the event log read comes back empty. A quiet cycle that
        # still has to talk to the cluster, which is all this test needs.
        calls.append(list(argv))
        return 0, b"", b""

    backend = SlurmBackend(
        ssh_alias="klone",
        remote_root="/r",
        runner=runner,
    )
    store.create_run(
        "vr_s1_ctl",
        backend="slurm",
        job_id="4242",
        status="queued",
        run_dir="/r/vr_s1_ctl",
    )

    Daemon(store, backend).run_once()

    assert calls, "the cycle never reached ssh, so it proves nothing"
    assert control_dir.is_dir(), (
        "a daemon cycle ran ssh without creating the ControlPath directory; "
        "a real ssh would have failed to bind its control socket"
    )
    assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700


# --------------------------------------------------------------------------
# Two schedules: the tail and the scheduler poll
# --------------------------------------------------------------------------
#
# The daemon reads event logs off the shared filesystem every couple of
# seconds, and asks `squeue` what the jobs are doing at most every thirty.
# Those are two different questions put to two different pieces of
# infrastructure, and the whole point of splitting them is that the cheap one
# can be fast without dragging the expensive one along with it.
#
# Everything below drives both schedules with a fake clock. Proving a
# thirty-second floor by waiting thirty seconds would give us a test suite
# nobody runs, and proving a five-minute ceiling by waiting five minutes would
# give us one nobody can run.


class FakeClock:
    """A monotonic clock the test moves by hand.

    `Daemon` takes its clock as a constructor argument for exactly this
    reason: both schedules are elapsed-time decisions, so the only two ways to
    test them are to sleep through them or to inject the clock. The fake reads
    like the real thing -- a zero-argument callable returning seconds that only
    ever goes forwards -- because that is the contract `time.monotonic`
    satisfies and the daemon relies on.
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = float(now)

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now

    def __call__(self) -> float:
        return self.now


def test_a_cycle_can_tail_without_polling_the_scheduler(store, backend, tmp_path):
    """Metrics keep flowing on a cycle that asks slurmctld nothing.

    This is the reason the two schedules exist. A user watching a loss curve
    wants a new point every couple of seconds; slurmctld, which every person
    on the cluster shares, wants to be asked about a job's state once every
    thirty at most. Before the split, one interval served both and there was no
    way to have the first without the second.
    """
    run_id, path = seed_run(store, tmp_path)
    # The job has just started, so the first poll finds a change and leaves the
    # interval at its floor rather than backing off.
    backend.statuses["1001"] = "running"
    backend.append(path, line(run_id, 1, "run_started"))

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    first = daemon.run_once()
    assert first.tailed
    assert first.scheduler_polled
    assert len(backend.status_calls) == 1
    assert first.scheduler_interval == pytest.approx(daemon_module.SCHEDULER_INTERVAL_MIN)

    # Two seconds later: a tail's worth of time, nowhere near a poll's worth.
    backend.append(path, metric_line(run_id, 2, 100, loss=0.5))
    clock.advance(2)
    second = daemon.run_once()

    assert second.tailed
    assert second.bytes_read > 0
    assert second.scheduler_polled is False
    assert len(backend.status_calls) == 1
    assert store.count_metrics(run_id) == 1

    # Thirty seconds after the poll, and only then, the scheduler is asked again.
    clock.advance(28)
    third = daemon.run_once()

    assert third.scheduler_polled
    assert len(backend.status_calls) == 2


def test_the_scheduler_floor_holds_however_many_runs_are_active(store, backend, tmp_path):
    """Twenty-five busy runs do not buy a single extra `squeue`.

    How much the user's jobs are printing says nothing about how much load
    slurmctld should be asked to carry, so run activity is deliberately absent
    from `_scheduler_due`. Getting this wrong is not a small mistake: a daemon
    polling at the tail's cadence for a twelve-hour run would put roughly
    twenty thousand requests a day into shared infrastructure.
    """
    paths = {}
    for n in range(25):
        run_id, path = seed_run(store, tmp_path, f"vr_{n:02d}", job_id=str(n))
        backend.statuses[str(n)] = "running"
        backend.append(path, line(run_id, 1, "run_started"))
        paths[run_id] = path

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    for cycle in range(10):
        for n, (run_id, path) in enumerate(paths.items()):
            # seq is per-run and monotonic; every run emits one point a cycle.
            backend.append(path, metric_line(run_id, cycle + 2, cycle, loss=1.0 / (n + 1)))
        stats = daemon.run_once()
        assert stats.tailed
        assert stats.bytes_read > 0
        clock.advance(2)

    # Ten cycles, twenty-five running runs, one question put to the scheduler.
    assert len(backend.status_calls) == 1
    assert sorted(backend.status_calls[0], key=int) == [str(n) for n in range(25)]
    # And the tail did its job throughout: every run has ten points.
    assert all(store.count_metrics(run_id) == 10 for run_id in paths)

    # Past the floor, one more -- still exactly one, not twenty-five.
    clock.advance(11)
    daemon.run_once()
    assert len(backend.status_calls) == 2


def test_the_scheduler_interval_backs_off_to_the_ceiling(store, backend, tmp_path):
    """Nothing changing means asking less often, up to five minutes and no further.

    Twenty minutes into a twelve-hour run that nobody has touched, the answer
    to "what is this job doing" has been the same for twenty minutes and will
    be the same for hours. The backoff is what turns that observation into
    fewer requests; the ceiling is what stops it turning into a daemon that
    notices a finished job ten minutes late.
    """
    seed_run(store, tmp_path)
    backend.statuses["1001"] = "queued"  # nothing ever changes

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    seen = []
    for _ in range(7):
        stats = daemon.run_once()
        assert stats.scheduler_polled
        assert stats.status_changes == 0
        seen.append(stats.scheduler_interval)
        # Sleep exactly as long as the daemon just decided to, so the next
        # cycle is due to the millisecond -- the schedule tests itself.
        clock.advance(daemon._scheduler_interval)

    # 30 x 1.5 each time, then held at the ceiling.
    assert seen == pytest.approx([45.0, 67.5, 101.25, 151.875, 227.8125, 300.0, 300.0])
    assert daemon.scheduler_interval_max == 300.0


def test_a_status_change_snaps_the_interval_back_to_the_floor(store, backend, tmp_path):
    """The moment anything moves, pay attention again.

    A job starting, finishing, being preempted or being cancelled all arrive as
    a status change, and any of them means the picture is moving. Staying at a
    five-minute poll through the interesting part of a run would be the worst
    of both worlds: we backed off because nothing was happening, and now
    something is.
    """
    run_id, _ = seed_run(store, tmp_path)
    backend.statuses["1001"] = "queued"

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    for _ in range(3):
        stats = daemon.run_once()
        clock.advance(stats.scheduler_interval)
    assert daemon._scheduler_interval == pytest.approx(101.25)

    backend.statuses["1001"] = "running"  # the job started
    stats = daemon.run_once()

    assert stats.scheduler_polled
    assert stats.status_changes == 1
    assert stats.scheduler_interval == pytest.approx(30.0)
    assert store.get_run(run_id)["status"] == "running"


def test_a_poke_polls_at_once_and_resets_the_backoff(store, backend, tmp_path):
    """`submit` and `cancel` say "look now", and the daemon looks now.

    Those are the two moments a user has just done something and is watching
    for the result -- and they are also, by construction, the moments the
    backoff is at its most relaxed, because nothing had been happening, which
    is why it backed off. Waiting out five minutes to show a job you just
    submitted would make relay feel broken.

    The interval afterwards is exactly the floor, even though the poll itself
    found nothing new. A poke counts as a change for the backoff, the same as
    a status change would: the user has just told us something is about to
    move, and taking a "nothing changed" step up in the very same cycle would
    mean the next poll comes at 45 seconds instead of 30.
    """
    seed_run(store, tmp_path)
    backend.statuses["1001"] = "queued"

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    for _ in range(3):
        stats = daemon.run_once()
        clock.advance(stats.scheduler_interval)
    assert daemon._scheduler_interval == pytest.approx(101.25)
    calls_before = len(backend.status_calls)

    store.poke_scheduler()  # what `relay submit` does, from another process
    poked = daemon.run_once()  # no clock advance at all

    assert poked.scheduler_polled is True
    assert poked.scheduler_poked is True
    assert len(backend.status_calls) == calls_before + 1
    assert poked.scheduler_interval == pytest.approx(30.0)

    # The poke is one-shot: it is a timestamp the daemon compares against the
    # last one it acted on, so reading it again is not another poke.
    again = daemon.run_once()
    assert again.scheduler_polled is False
    assert len(backend.status_calls) == calls_before + 1


def test_a_daemon_started_after_a_poke_does_not_treat_it_as_news(store, backend, tmp_path):
    """An hour-old `relay submit` must not poke a daemon that has just started.

    The poke is a persistent timestamp rather than a flag that gets cleared, so
    a new daemon has to seed itself with whatever is already there. Otherwise
    every restart would find an ancient poke, treat it as fresh, and poll a
    second time for no reason -- and on a busy laptop, restart-loop its way
    into exactly the polling rate the floor exists to prevent.
    """
    seed_run(store, tmp_path)
    backend.statuses["1001"] = "queued"
    store.poke_scheduler()

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    # Cycle one polls regardless -- a daemon that has just started should look
    # immediately rather than make the user wait out an interval.
    first = daemon.run_once()
    assert first.scheduler_polled

    second = daemon.run_once()  # no advance, and no new poke
    assert second.scheduler_polled is False
    assert len(backend.status_calls) == 1


def test_a_failed_poll_neither_grows_nor_resets_the_interval(store, backend, tmp_path):
    """A network blip is not evidence about how fast the jobs are moving.

    Growing the interval would punish the user for a dropped connection by
    making recovery slower; resetting it would reward them for one. Neither
    reading is supported by a call that never got an answer, so the interval is
    left exactly where it was and the whole-cycle backoff handles a transport
    that is genuinely down.
    """
    seed_run(store, tmp_path)
    backend.statuses["1001"] = "queued"

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    first = daemon.run_once()
    assert first.scheduler_interval == pytest.approx(45.0)

    backend.status_error = ConnectionError("squeue: connection closed by remote host")
    clock.advance(45)
    failed = daemon.run_once()

    assert failed.scheduler_polled
    assert failed.status_failed
    assert failed.scheduler_interval == pytest.approx(45.0)
    assert daemon._scheduler_interval == pytest.approx(45.0)

    backend.status_error = None
    clock.advance(45)
    recovered = daemon.run_once()

    # Not reset to the floor (nothing changed), not stuck (the poll worked).
    assert not recovered.status_failed
    assert recovered.scheduler_interval == pytest.approx(67.5)


def test_the_sleep_is_the_sooner_of_the_two_schedules(store, backend, tmp_path):
    """One thread, two schedules, so the nap is however long the nearer one is.

    A second thread for the scheduler poll would be the sophisticated
    alternative, and it would also mean two things writing to one SQLite
    connection. Instead the loop sleeps until whichever schedule comes first
    and the next cycle works out which one it woke up for.
    """
    run_id, path = seed_run(store, tmp_path)
    backend.statuses["1001"] = "running"
    backend.append(path, line(run_id, 1, "run_started"))

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    stats = daemon.run_once()
    assert stats.running == 1
    # A running run makes the tail the nearer of the two, by a long way.
    assert daemon._interval_for(stats) == INTERVAL_ACTIVE == 2.0

    clock.advance(29)
    stats = daemon.run_once()

    assert stats.scheduler_polled is False
    # One second of the scheduler's thirty is left, which is now sooner than
    # the tail's two. Sleeping the full two would overshoot the poll.
    assert daemon._interval_for(stats) == pytest.approx(1.0)


def test_an_idle_daemon_still_wakes_up_for_the_scheduler(store, backend):
    """Nothing to tail is not nothing to do.

    With no runs at all the tail would happily sleep a minute, but a job may be
    sitting in the queue about to start, and the only way to find out is to
    ask. The scheduler's own schedule is what gets the daemon out of bed.
    """
    clock = FakeClock()
    daemon = Daemon(
        store,
        backend,
        # A ceiling equal to the floor means a fixed 30s poll with no backoff,
        # which keeps this test about the `min` and not about the backoff.
        scheduler_interval_min=30.0,
        scheduler_interval_max=30.0,
        clock=clock,
    )

    stats = daemon.run_once()

    assert stats.runs_checked == 0
    assert next_interval(stats) == INTERVAL_IDLE == 60.0
    assert daemon._interval_for(stats) == pytest.approx(30.0)


def test_the_scheduler_freshness_stamp_moves_only_when_the_scheduler_answers(
    store, backend, tmp_path
):
    """Two schedules need two "last synced" numbers, or one of them lies.

    `relay ls` prints both: how fresh the metrics are and how fresh the job
    state is. They are minutes apart by design, and stamping the scheduler's
    number on a cycle that only tailed would tell the user their job state was
    two seconds old when it was five minutes old. Stale data is fine; stale
    data presented as current is the one thing relay refuses to do.
    """
    run_id, path = seed_run(store, tmp_path)
    backend.statuses["1001"] = "running"
    backend.append(path, line(run_id, 1, "run_started"))

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)

    assert store.last_scheduler_synced() is None

    daemon.run_once()
    stamped = store.last_scheduler_synced()
    assert stamped is not None

    backend.append(path, metric_line(run_id, 2, 10, loss=1.0))
    clock.advance(2)
    tail_only = daemon.run_once()

    assert tail_only.scheduler_polled is False
    assert tail_only.bytes_read > 0
    # The tail's stamp is fresh; the scheduler's has not moved.
    assert store.last_synced() is not None
    assert store.last_scheduler_synced() == stamped


def test_a_failed_poll_does_not_stamp_the_scheduler_freshness(store, backend, tmp_path):
    """Asking and getting nothing is not the same as the scheduler answering."""
    seed_run(store, tmp_path)
    backend.status_error = ConnectionError("squeue: unreachable")

    stats = Daemon(store, backend, clock=FakeClock()).run_once()

    assert stats.scheduler_polled
    assert stats.status_failed
    assert store.last_scheduler_synced() is None


def test_a_cycle_that_asks_nothing_mid_outage_is_neither_success_nor_failure(
    store, backend
):
    """A cycle that made no call learned nothing, so it must not clear the streak.

    This case only exists because the schedules are independent. Before the
    split every cycle made at least one call, so "did not fail" really did mean
    "something worked". Now a cycle can have nothing to tail (a queued run with
    no run directory yet) and no poll due, and if that counted as a success it
    would reset the backoff in the middle of an outage and put a fresh
    "metrics 0s ago" under `relay ls` on the strength of having done nothing.
    """
    # No run_dir, so there is nothing to tail: the scheduler poll is the only
    # call this daemon can make, and it is the one that is broken.
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = ConnectionError("ssh: Operation timed out")

    clock = FakeClock()
    daemon = Daemon(store, backend, clock=clock)  # the real 30s floor

    first = daemon.run_once()
    assert first.failed
    assert first.scheduler_polled
    assert daemon.consecutive_failures == 1
    assert store.last_synced() is None

    # Same instant: the poll is not due, and there is nothing else to do.
    neutral = daemon.run_once()

    assert neutral.scheduler_polled is False
    assert neutral.read_attempts == 0
    assert not neutral.failed
    assert daemon.consecutive_failures == 1
    assert store.last_synced() is None

    clock.advance(30)
    third = daemon.run_once()

    # The streak kept counting rather than starting over, so the whole-cycle
    # backoff is where an outage of this length should have put it.
    assert third.failed
    assert daemon.consecutive_failures == 2
    assert next_interval(third) == 8.0


def test_a_cycle_that_asks_nothing_does_not_claim_the_channel_is_healthy(store, backend):
    """The neutral-cycle rule covers the daemon_state row too, not just the streak.

    `run_once` already refuses to let a cycle that made no call reset the
    failure streak or stamp `last_synced`. But `_record_cycle_health` runs
    afterwards on the same cycle and, seeing no error *this* cycle, clears the
    stored `unreachable` verdict and writes a fresh `last_ok_ts` -- which is
    supposed to mean "the last time the channel demonstrably worked". Nothing
    was demonstrated. `relay ls` and the dashboard read those fields, so during
    an outage the error banner blinks off on every in-between cycle and comes
    back on the next real attempt.
    """
    store.create_run("vr_x", backend="fake", job_id="7", status="queued")
    backend.status_error = ConnectionError("ssh: Operation timed out")

    daemon = Daemon(store, backend, clock=FakeClock())
    daemon.run_once()
    assert store.get_daemon_state()["last_error_kind"] == "unreachable"

    neutral = daemon.run_once()
    assert neutral.read_attempts == 0 and not neutral.scheduler_polled

    state = store.get_daemon_state()
    assert state["last_error_kind"] == "unreachable"
    assert state["last_ok_ts"] is None


# --------------------------------------------------------------------------
# Merging the intervals: the config file and `relay daemon`'s flags
# --------------------------------------------------------------------------


DAEMON_INTERVAL_KEYS = {
    "interval_active",
    "interval_queued",
    "interval_idle",
    "scheduler_interval_min",
    "scheduler_interval_max",
    "usage_interval_seconds",
}


def a_config(**daemon_fields) -> config_module.Config:
    return config_module.Config(daemon=config_module.DaemonConfig(**daemon_fields))


def test_daemon_intervals_reads_the_config_when_no_flags_were_given():
    """The `daemon:` section reaches the daemon under the daemon's own names.

    Two vocabularies meet here: the config file says `tail_interval_active`
    and `usage_interval`, `Daemon.__init__` says `interval_active` and
    `usage_interval_seconds`. The translation has to live in exactly one place
    or the two drift apart silently -- a renamed key that nothing reads looks
    identical to a key that works.
    """
    conf = a_config(
        tail_interval_active=5,
        scheduler_interval_min=45,
        scheduler_interval_max=90,
        usage_interval=120,
    )

    values = daemon_module.daemon_intervals(conf, None)

    assert values["interval_active"] == 5
    assert values["scheduler_interval_min"] == 45
    assert values["scheduler_interval_max"] == 90
    assert values["usage_interval_seconds"] == 120
    # Untouched keys keep the config's defaults rather than disappearing.
    assert values["interval_queued"] == 30.0
    assert values["interval_idle"] == 60.0


def test_a_flag_beats_the_file_and_an_absent_flag_changes_nothing():
    """argparse leaves an unpassed flag as None, and None must mean "not given".

    If a None went through as a value it would overwrite the config with
    nothing, so a user who passed one flag would silently lose every other
    setting in their `daemon:` section.
    """
    conf = a_config(tail_interval_active=5, scheduler_interval_min=45)

    values = daemon_module.daemon_intervals(
        conf, {"scheduler_interval_min": 60, "interval_active": None}
    )

    assert values["scheduler_interval_min"] == 60
    assert values["interval_active"] == 5


def test_the_scheduler_floor_is_reapplied_to_the_flag(caplog):
    """`--scheduler-interval-min 3` is clamped, not obeyed.

    The flag never goes through the config validator, and a floor enforced in
    one of the two places a number can arrive from is not a floor. The warning
    has to name the flag, because that is what the user has to go and change.
    """
    conf = a_config(scheduler_interval_min=45, scheduler_interval_max=90)

    with caplog.at_level(logging.WARNING, logger="relay.config"):
        values = daemon_module.daemon_intervals(conf, {"scheduler_interval_min": 3})

    assert values["scheduler_interval_min"] == config_module.MIN_SCHEDULER_INTERVAL == 10.0
    assert "floor" in caplog.text
    assert "--scheduler-interval-min" in caplog.text


def test_a_flag_that_pushes_the_floor_above_the_ceiling_raises_the_ceiling(caplog):
    """`--scheduler-interval-min 120` against a config ceiling of 90 still starts.

    A ceiling under the floor would make the backoff shrink the interval
    instead of growing it. The config validator rejects that combination
    outright, but a flag can still produce it, and refusing to start the daemon
    -- leaving the user with no view of their running jobs at all -- would be a
    poor trade for a number we can simply raise.
    """
    conf = a_config(scheduler_interval_min=45, scheduler_interval_max=90)

    with caplog.at_level(logging.WARNING, logger="relay.daemon"):
        values = daemon_module.daemon_intervals(conf, {"scheduler_interval_min": 120})

    assert values["scheduler_interval_min"] == 120
    assert values["scheduler_interval_max"] == 120
    assert "scheduler_interval_max" in caplog.text


def test_daemon_intervals_returns_exactly_the_daemon_kwargs(store, backend):
    """The dict is passed to `Daemon(**...)`, so a stray key is a TypeError at startup.

    `relay daemon` builds its daemon from whatever this returns. A key that is
    not a constructor argument would not be caught by anything until the moment
    a user runs the command, which is the worst possible place to find out.
    """
    values = daemon_module.daemon_intervals(a_config(), None)

    assert set(values) == DAEMON_INTERVAL_KEYS
    accepted = set(inspect.signature(Daemon.__init__).parameters)
    assert DAEMON_INTERVAL_KEYS <= accepted

    daemon = Daemon(store, backend, **values)

    assert daemon.interval_active == 2.0
    assert daemon.scheduler_interval_min == 30.0
    assert daemon.scheduler_interval_max == 300.0
    assert daemon.usage_interval_seconds == 300.0
