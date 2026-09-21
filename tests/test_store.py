"""Tests for the store.

The properties worth protecting here are not "does INSERT work" -- they are the
two guarantees the rest of relay leans on:

  * ingest is atomic: events, derived metrics, run state and the byte offset
    move together or not at all;
  * ingest is idempotent: re-reading bytes we already read changes nothing.

Everything else (queries, filters, defaults) is checked because the CLI, the
daemon and the dashboard all call it and none of them should have to guess.

These tests build event dicts by hand rather than importing `relay.events`, so
the store is tested against the *schema*, not against another module.
"""

import json
import sqlite3
import threading
import time

import pytest

from relay.store import (
    META_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TERMINAL,
    Store,
    utc_now_iso,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def event(seq, type, payload=None, ts=None, run="r1"):
    """One event dict in the wire shape the daemon hands to the store."""
    return {
        "v": 1,
        "run": run,
        "seq": seq,
        "ts": ts or f"2026-09-16T18:00:{seq:02d}.000Z",
        "type": type,
        "payload": payload or {},
    }


def metric_event(seq, step, values, ts=None):
    return event(seq, "metric", {"step": step, "values": values}, ts=ts)


def usage_row(
    job_id="100",
    start="2026-09-16T18:00:00.000Z",
    run_id="r1",
    state="COMPLETED",
    partition="ckpt-all",
    end="2026-09-16T19:00:00.000Z",
    elapsed_s=3600,
    gpus=1,
    cpus=4,
):
    """One usage row in the dict shape the daemon hands the store.

    Written out by hand rather than built from `backends.base.UsageRow`, for the
    same reason the event helper above does not import `relay.events`: the store
    is tested against the *contract*, not against another module.
    """
    return {
        "job_id": job_id,
        "start": start,
        "run_id": run_id,
        "state": state,
        "partition": partition,
        "end": end,
        "elapsed_s": elapsed_s,
        "gpus": gpus,
        "cpus": cpus,
    }


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "relay.db"


@pytest.fixture
def store(db_path):
    s = Store(db_path)
    yield s
    s.close()


@pytest.fixture
def run(store):
    """A store with one registered run, `r1`."""
    store.create_run("r1", backend="local", name="demo", run_dir="/tmp/r1")
    return store


# --------------------------------------------------------------------------
# Schema and connection setup
# --------------------------------------------------------------------------


def test_schema_created_on_fresh_file(db_path):
    assert not db_path.exists()
    s = Store(db_path)
    names = {
        row[0]
        for row in s.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"runs", "events", "metrics", "sources", "meta"} <= names
    s.close()
    assert db_path.exists()


def test_parent_directory_is_created(tmp_path):
    nested = tmp_path / "does" / "not" / "exist" / "relay.db"
    Store(nested).close()
    assert nested.exists()


def test_wal_mode_is_active(store):
    assert store.journal_mode() == "wal"
    assert store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    # synchronous=NORMAL is 1 in SQLite's numbering (OFF=0, FULL=2).
    assert store.conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_opening_an_existing_database_is_harmless(db_path):
    first = Store(db_path)
    first.create_run("r1", backend="local")
    first.close()

    second = Store(db_path)
    assert second.get_run("r1") is not None
    second.close()


def test_requeued_is_not_terminal():
    # A preempted job comes back and keeps writing to the same log, so the
    # daemon must keep watching it.
    assert "requeued" not in TERMINAL
    assert TERMINAL == {"completed", "failed", "cancelled"}


# --------------------------------------------------------------------------
# Ingest: atomicity, derivation, idempotence
# --------------------------------------------------------------------------


def test_ingest_inserts_events_metrics_and_offset(run):
    events = [
        event(1, "run_started", {"cmd": "python train.py"}),
        metric_event(2, 100, {"loss": 0.5, "sps": 18400}),
        event(3, "log", {"stream": "stdout", "text": "epoch 1"}),
    ]
    result = run.ingest("r1", "/tmp/r1/events.jsonl", events, new_offset=512)

    assert result == {"events_inserted": 3, "metrics_inserted": 2, "byte_offset": 512}
    assert run.count_events("r1") == 3
    assert run.count_metrics("r1") == 2
    assert run.get_offset("r1", "/tmp/r1/events.jsonl") == 512

    # Payload survives the round trip as a dict, not as text.
    stored = run.list_events("r1")
    assert stored[0]["payload"] == {"cmd": "python train.py"}
    assert [e["seq"] for e in stored] == [1, 2, 3]


def test_ingest_is_idempotent(run):
    events = [
        event(1, "run_started"),
        metric_event(2, 100, {"loss": 0.5}),
        event(3, "heartbeat", {"last_step": 100}),
    ]
    run.ingest("r1", "log", events, new_offset=300)
    before = _snapshot(run)

    second = run.ingest("r1", "log", events, new_offset=300)

    assert second["events_inserted"] == 0
    assert second["metrics_inserted"] == 0
    assert _snapshot(run) == before


def test_duplicate_seq_keeps_the_first_row(run):
    run.ingest("r1", "log", [event(1, "log", {"text": "original"})], new_offset=50)
    run.ingest("r1", "log", [event(1, "log", {"text": "impostor"})], new_offset=50)

    rows = run.list_events("r1")
    assert len(rows) == 1
    assert rows[0]["payload"]["text"] == "original"


def test_partial_reread_yields_exact_totals(run):
    """The daemon's real failure mode: it crashes after reading, before committing.

    On restart it re-reads from the last committed offset, so the store sees the
    old events again followed by new ones. The totals must be exactly right.
    """
    first_batch = [metric_event(seq, seq * 10, {"loss": 1.0 / seq}) for seq in range(1, 6)]
    run.ingest("r1", "log", first_batch, new_offset=400)

    overlap = first_batch[3:]  # seq 4, 5 seen again
    new = [metric_event(seq, seq * 10, {"loss": 1.0 / seq}) for seq in range(6, 9)]
    result = run.ingest("r1", "log", overlap + new, new_offset=700)

    assert result["events_inserted"] == 3
    assert run.count_events("r1") == 8
    assert run.count_metrics("r1") == 8
    assert [e["seq"] for e in run.list_events("r1")] == list(range(1, 9))
    assert run.get_offset("r1", "log") == 700


def test_ingest_is_one_transaction(run):
    """A failure anywhere in ingest must leave the database exactly as it was.

    A malformed event (no "ts") blows up partway through building the rows; if
    events, metrics and the offset were written separately, we would be left
    with some of them.
    """
    run.ingest("r1", "log", [event(1, "log", {"text": "good"})], new_offset=100)

    bad = [metric_event(2, 10, {"loss": 0.1}), {"v": 1, "run": "r1", "seq": 3, "type": "log"}]
    with pytest.raises(KeyError):
        run.ingest("r1", "log", bad, new_offset=999)

    assert run.count_events("r1") == 1
    assert run.count_metrics("r1") == 0
    assert run.get_offset("r1", "log") == 100


def test_ingest_with_no_events_still_moves_the_offset(run):
    run.ingest("r1", "log", [], new_offset=42)
    assert run.get_offset("r1", "log") == 42
    assert run.count_events("r1") == 0


def test_non_numeric_metric_values_are_skipped(run):
    run.ingest(
        "r1",
        "log",
        [metric_event(1, 10, {"loss": 0.5, "phase": "warmup", "converged": False})],
        new_offset=100,
    )
    # The event keeps everything; the chart table keeps only numbers. A bool is
    # not a number here, even though Python says isinstance(True, int).
    assert run.metric_names("r1") == ["loss"]
    assert run.list_events("r1")[0]["payload"]["values"]["phase"] == "warmup"


def test_events_for_an_unknown_run_are_still_stored(store):
    """No foreign key: ingest must not depend on who wrote the run row first."""
    store.ingest("ghost", "log", [event(1, "log", {"text": "hi"}, run="ghost")], new_offset=10)
    assert store.count_events("ghost") == 1
    assert store.get_run("ghost") is None


# --------------------------------------------------------------------------
# Run state derived during ingest
# --------------------------------------------------------------------------


def test_heartbeat_updates_last_heartbeat_and_step(run):
    assert run.get_run("r1")["last_heartbeat_ts"] is None

    run.ingest(
        "r1",
        "log",
        [event(1, "heartbeat", {"last_step": 250}, ts="2026-09-16T18:22:09.003Z")],
        new_offset=120,
    )

    row = run.get_run("r1")
    assert row["last_heartbeat_ts"] == "2026-09-16T18:22:09.003Z"
    assert row["last_step"] == 250


def test_metric_events_advance_last_step(run):
    run.ingest("r1", "log", [metric_event(1, 100, {"loss": 1.0})], new_offset=50)
    run.ingest("r1", "log", [metric_event(2, 300, {"loss": 0.5})], new_offset=100)
    assert run.get_run("r1")["last_step"] == 300


def test_replayed_events_never_drag_state_backwards(run):
    run.ingest("r1", "log", [metric_event(1, 100, {"loss": 1.0})], new_offset=50)
    run.ingest("r1", "log", [metric_event(2, 900, {"loss": 0.2})], new_offset=100)
    # Re-ingesting the older batch must not reset last_step to 100.
    run.ingest("r1", "log", [metric_event(1, 100, {"loss": 1.0})], new_offset=100)
    assert run.get_run("r1")["last_step"] == 900


# --------------------------------------------------------------------------
# Runs table
# --------------------------------------------------------------------------


def test_create_run_round_trips_spec_and_defaults(store):
    store.create_run(
        "r1",
        backend="slurm",
        name="ppo-seed0",
        job_id="12345",
        run_dir="/mmfs1/.../r1",
        spec={"script": "train.py", "seed": 0},
    )
    row = store.get_run("r1")
    assert row["backend"] == "slurm"
    assert row["job_id"] == "12345"
    assert row["status"] == "queued"
    assert row["stale"] is False
    assert row["terminal"] is False
    assert row["spec"] == {"script": "train.py", "seed": 0}
    assert row["created_ts"] == row["updated_ts"]


def test_create_run_twice_keeps_the_original(store):
    store.create_run("r1", backend="local", name="first")
    store.create_run("r1", backend="local", name="second")
    assert store.get_run("r1")["name"] == "first"


def test_get_run_missing_returns_none(store):
    assert store.get_run("nope") is None


def test_upsert_run_creates_then_updates(store):
    store.upsert_run("r1", backend="local", status="running", job_id="991")
    row = store.get_run("r1")
    assert (row["status"], row["job_id"]) == ("running", "991")

    store.upsert_run("r1", backend="local", status="completed")
    assert store.get_run("r1")["status"] == "completed"


def test_update_run_status_rejects_unknown_status(run):
    with pytest.raises(ValueError):
        run.update_run_status("r1", "vibing")


def test_update_run_rejects_unknown_column(run):
    with pytest.raises(ValueError):
        run.update_run("r1", typo="x")


def test_set_stale_does_not_touch_status(run):
    run.update_run_status("r1", "running")
    run.set_stale("r1", True)
    row = run.get_run("r1")
    assert row["stale"] is True
    assert row["status"] == "running"

    run.set_stale("r1", False)
    assert run.get_run("r1")["stale"] is False


def test_list_runs_active_only_excludes_terminal(store):
    store.create_run("a", backend="local", created_ts="2026-09-16T00:00:00.000Z")
    store.create_run("b", backend="local", created_ts="2026-09-16T01:00:00.000Z")
    store.create_run("c", backend="local", created_ts="2026-09-16T02:00:00.000Z")
    store.update_run_status("a", "running")
    store.update_run_status("b", "completed")
    store.update_run_status("c", "requeued")

    assert [r["run_id"] for r in store.list_runs()] == ["c", "b", "a"]
    # requeued stays in the working set; completed drops out.
    assert {r["run_id"] for r in store.list_runs(active_only=True)} == {"a", "c"}
    assert [r["run_id"] for r in store.list_runs(limit=1)] == ["c"]


# --------------------------------------------------------------------------
# Reads: events, logs, metrics
# --------------------------------------------------------------------------


def test_list_events_filters_by_type_and_since_seq(run):
    run.ingest(
        "r1",
        "log",
        [
            event(1, "run_started"),
            event(2, "log", {"text": "a"}),
            metric_event(3, 10, {"loss": 1.0}),
            event(4, "log", {"text": "b"}),
        ],
        new_offset=400,
    )

    assert [e["seq"] for e in run.list_events("r1", since_seq=2)] == [3, 4]
    assert [e["seq"] for e in run.list_events("r1", type="log")] == [2, 4]
    assert [e["seq"] for e in run.list_events("r1", limit=2)] == [1, 2]
    assert [e["payload"]["text"] for e in run.list_logs("r1", since_seq=2)] == ["b"]
    assert run.max_seq("r1") == 4
    assert run.max_seq("unknown") == 0


def test_metric_series_since_step_returns_only_newer_points(run):
    run.ingest(
        "r1",
        "log",
        [metric_event(i + 1, i * 100, {"loss": 1.0 / (i + 1)}) for i in range(5)],
        new_offset=900,
    )

    everything = run.metric_series("r1", "loss")
    assert [p["step"] for p in everything] == [0, 100, 200, 300, 400]
    # since_step is exclusive, matching the dashboard's "I have up to step N".
    assert [p["step"] for p in run.metric_series("r1", "loss", since_step=200)] == [300, 400]
    assert run.metric_series("r1", "loss", since_step=400) == []
    # Default None, not 0: step 0 is a real point and must survive a first fetch.
    assert run.metric_series("r1", "loss", since_step=0)[0]["step"] == 100


def test_metric_names_and_latest(run):
    run.ingest(
        "r1",
        "log",
        [
            metric_event(1, 10, {"loss": 1.0, "sps": 100}),
            metric_event(2, 20, {"loss": 0.4, "sps": 120}),
        ],
        new_offset=200,
    )
    assert run.metric_names("r1") == ["loss", "sps"]
    assert run.latest_metrics("r1") == {"loss": 0.4, "sps": 120.0}


def test_same_metric_name_and_step_is_written_once(run):
    run.ingest("r1", "log", [metric_event(1, 10, {"loss": 1.0})], new_offset=50)
    # A different event (seq 2) reporting the same name/step: the metrics table
    # is keyed on (run_id, name, step), so it stays a single point.
    run.ingest("r1", "log", [metric_event(2, 10, {"loss": 2.0})], new_offset=100)
    series = run.metric_series("r1", "loss")
    assert len(series) == 1
    assert series[0]["value"] == 1.0


def test_metrics_are_scoped_per_run(store):
    store.ingest("r1", "log", [metric_event(1, 10, {"loss": 1.0})], new_offset=50)
    store.ingest("r2", "log", [metric_event(1, 10, {"loss": 9.0})], new_offset=50)
    assert store.metric_series("r1", "loss")[0]["value"] == 1.0
    assert store.metric_series("r2", "loss")[0]["value"] == 9.0
    assert store.count_events() == 2


# --------------------------------------------------------------------------
# Sources and meta
# --------------------------------------------------------------------------


def test_get_offset_defaults_to_zero(store):
    assert store.get_offset("r1", "/never/read") == 0


def test_offsets_are_per_path(run):
    run.ingest("r1", "a.jsonl", [event(1, "log")], new_offset=10)
    run.ingest("r1", "b.jsonl", [event(2, "log")], new_offset=99)
    assert run.get_offset("r1", "a.jsonl") == 10
    assert run.get_offset("r1", "b.jsonl") == 99
    assert [s["path"] for s in run.list_sources("r1")] == ["a.jsonl", "b.jsonl"]


def test_meta_get_set_and_sync_stamp(store):
    assert store.meta_get("nothing") is None
    assert store.meta_get("nothing", "fallback") == "fallback"

    store.meta_set("k", "v1")
    store.meta_set("k", "v2")  # meta overwrites, unlike events
    assert store.meta_get("k") == "v2"

    stamp = store.mark_synced()
    assert store.last_synced() == stamp
    assert stamp.endswith("Z")


def test_utc_now_iso_shape():
    ts = utc_now_iso()
    assert ts.endswith("Z") and "T" in ts and "+" not in ts


# --------------------------------------------------------------------------
# Schema migration: old databases must keep working
# --------------------------------------------------------------------------

# The `runs` table exactly as schema version 1 created it -- no `attempts`, no
# `usage_final`. Copied here on purpose rather than imported: the point of the
# test is that a database written by *last month's relay* still opens, and the
# only way to have one of those is to write the old DDL down.
OLD_RUNS_DDL = """
CREATE TABLE runs (
    run_id            TEXT PRIMARY KEY,
    name              TEXT,
    backend           TEXT NOT NULL,
    job_id            TEXT,
    status            TEXT NOT NULL,
    stale             INTEGER NOT NULL DEFAULT 0,
    created_ts        TEXT NOT NULL,
    updated_ts        TEXT NOT NULL,
    last_heartbeat_ts TEXT,
    last_step         INTEGER,
    run_dir           TEXT,
    spec              TEXT
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO meta (key, value) VALUES ('schema_version', '1');
INSERT INTO runs (run_id, backend, status, created_ts, updated_ts, job_id)
VALUES ('old', 'slurm', 'running', '2026-01-01T00:00:00.000Z',
        '2026-01-01T00:00:00.000Z', '42');
"""


@pytest.fixture
def old_db(db_path):
    """A database in the shape relay used to write, before usage accounting."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(OLD_RUNS_DDL)
    conn.commit()
    conn.close()
    return db_path


def test_old_database_gains_the_new_columns(old_db):
    store = Store(old_db)
    columns = {row["name"] for row in store.conn.execute("PRAGMA table_info(runs)")}
    assert "attempts" in columns and "usage_final" in columns
    store.close()


def test_old_database_gains_the_new_tables(old_db):
    store = Store(old_db)
    tables = {
        row[0] for row in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"usage", "daemon_state", "events", "metrics", "sources"} <= tables
    store.close()


def test_migration_keeps_existing_rows_and_defaults_the_new_columns(old_db):
    store = Store(old_db)
    run = store.get_run("old")
    assert run["status"] == "running"  # the row survived untouched
    assert run["attempts"] == 0  # SQLite backfilled the DEFAULT
    assert run["usage_final"] is False
    store.close()


def test_migration_bumps_the_recorded_schema_version(old_db):
    store = Store(old_db)
    assert store.meta_get(META_SCHEMA_VERSION) == str(SCHEMA_VERSION)
    assert SCHEMA_VERSION == 2
    store.close()


def test_migration_is_safe_to_run_twice(old_db):
    Store(old_db).close()
    second = Store(old_db)  # would raise "duplicate column name" if unguarded
    assert second.get_run("old")["attempts"] == 0
    second.close()


def test_fresh_database_has_the_usage_tables(db_path):
    store = Store(db_path)
    tables = {
        row[0] for row in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"usage", "daemon_state"} <= tables
    store.close()


# --------------------------------------------------------------------------
# Usage: the one table that updates in place
# --------------------------------------------------------------------------


def test_upsert_usage_inserts_then_updates_in_place(run):
    run.upsert_usage([usage_row(state="RUNNING", end=None, elapsed_s=600)])
    assert len(run.list_usage("r1")) == 1

    # Same attempt, later look: it has been running longer and has now ended.
    run.upsert_usage([usage_row(state="COMPLETED", end="2026-09-16T19:00:00.000Z", elapsed_s=3600)])

    rows = run.list_usage("r1")
    assert len(rows) == 1, "an attempt that grew must update, not duplicate"
    assert rows[0]["elapsed_s"] == 3600
    assert rows[0]["state"] == "COMPLETED"
    assert rows[0]["end"] == "2026-09-16T19:00:00.000Z"


def test_upsert_usage_of_identical_rows_changes_nothing(run):
    rows = [usage_row(start="2026-09-16T18:00:00.000Z"), usage_row(start="2026-09-16T20:00:00.000Z")]
    run.upsert_usage(rows)
    before = run.list_usage("r1")

    run.upsert_usage(rows)  # re-fetching the same sacct output
    assert run.list_usage("r1") == before


def test_upsert_usage_with_no_rows_is_a_no_op(run):
    assert run.upsert_usage([]) == 0
    assert run.list_usage("r1") == []


def test_attempts_of_one_job_are_separate_rows_and_sum(run):
    # One job ID, three attempts: preempted twice, then finished. Each burned
    # half an hour on two GPUs.
    run.upsert_usage(
        [
            usage_row(start="2026-09-16T18:00:00.000Z", state="PREEMPTED", elapsed_s=1800, gpus=2),
            usage_row(start="2026-09-16T19:00:00.000Z", state="PREEMPTED", elapsed_s=1800, gpus=2),
            usage_row(start="2026-09-16T20:00:00.000Z", state="COMPLETED", elapsed_s=1800, gpus=2),
        ]
    )
    by_run = run.usage_by("run")
    assert len(by_run) == 1
    assert by_run[0]["key"] == "r1"
    assert by_run[0]["attempts"] == 3
    assert by_run[0]["runs"] == 1
    assert by_run[0]["gpu_hours"] == pytest.approx(3 * 1800 * 2 / 3600)
    assert by_run[0]["cpu_hours"] == pytest.approx(3 * 1800 * 4 / 3600)


# --------------------------------------------------------------------------
# Attempt counting at ingest
# --------------------------------------------------------------------------


def test_run_started_attempt_sets_the_attempt_count(run):
    # `attempt` is 0-based, so the third start is attempts = 3.
    run.ingest("r1", "log", [event(1, "run_started", {"attempt": 2})], new_offset=10)
    assert run.get_run("r1")["attempts"] == 3


def test_a_replayed_earlier_attempt_does_not_lower_the_count(run):
    run.ingest("r1", "log", [event(1, "run_started", {"attempt": 2})], new_offset=10)
    run.ingest("r1", "log", [event(2, "run_started", {"attempt": 0})], new_offset=20)
    assert run.get_run("r1")["attempts"] == 3


def test_v1_run_started_without_attempt_counts_as_the_first(run):
    # Schema v1 had no requeue-aware sidecar, so a run_started with no
    # `attempt` field was always a first start.
    run.ingest("r1", "log", [event(1, "run_started", {"cmd": ["python", "train.py"]})], new_offset=9)
    assert run.get_run("r1")["attempts"] == 1


def test_attempts_starts_at_zero_before_any_run_started(run):
    assert run.get_run("r1")["attempts"] == 0


# --------------------------------------------------------------------------
# Attempt history: what `relay show` prints per attempt
# --------------------------------------------------------------------------


def test_attempt_history_is_empty_for_an_unknown_run(store):
    assert store.attempt_history("nobody") == []


def test_attempt_history_lists_each_start_in_seq_order(run):
    run.ingest(
        "r1",
        "log",
        [
            event(1, "run_started", {"attempt": 0}),
            metric_event(2, 100, {"loss": 1.0}),
            event(
                3,
                "run_started",
                {"attempt": 1, "resumed_from": "/mmfs1/scratch/r1/ckpt_100.pt"},
            ),
        ],
        new_offset=300,
    )

    history = run.attempt_history("r1")
    assert [a["attempt"] for a in history] == [0, 1]
    assert [a["seq"] for a in history] == [1, 3]
    assert history[0]["ts"] == "2026-09-16T18:00:01.000Z"
    assert history[0]["resumed_from"] is None
    assert history[1]["resumed_from"] == "/mmfs1/scratch/r1/ckpt_100.pt"
    # Present but always None for now, so the CLI's table has a stable shape.
    assert all(a["job_id"] is None for a in history)


def test_attempt_history_reads_a_v1_start_without_attempt_as_zero(run):
    run.ingest("r1", "log", [event(1, "run_started", {"cmd": ["python", "train.py"]})], new_offset=9)
    history = run.attempt_history("r1")
    assert len(history) == 1
    assert history[0]["attempt"] == 0
    assert history[0]["resumed_from"] is None


def test_attempt_history_reads_a_null_resumed_from_as_none(run):
    # The sidecar writes the key explicitly as null when the attempt started
    # from scratch; that must read the same as the key being absent.
    run.ingest("r1", "log", [event(1, "run_started", {"attempt": 0, "resumed_from": None})], 9)
    assert run.attempt_history("r1")[0]["resumed_from"] is None


# --------------------------------------------------------------------------
# Which runs the daemon should fetch usage for
# --------------------------------------------------------------------------


def test_runs_needing_usage_covers_live_and_unfinalised_runs(store):
    store.create_run("live", backend="slurm", job_id="1", status="running")
    store.create_run("done_open", backend="slurm", job_id="2", status="completed")
    store.create_run("done_final", backend="slurm", job_id="3", status="completed")
    store.mark_usage_final("done_final")
    store.create_run("never_submitted", backend="slurm", status="queued")  # no job_id

    ids = {r["run_id"] for r in store.runs_needing_usage()}
    assert ids == {"live", "done_open"}


def test_mark_usage_final_is_visible_on_the_run(store):
    store.create_run("r", backend="slurm", job_id="7", status="completed")
    assert store.get_run("r")["usage_final"] is False
    store.mark_usage_final("r")
    assert store.get_run("r")["usage_final"] is True
    assert store.list_runs()[0]["usage_final"] is True


# --------------------------------------------------------------------------
# Derived numbers: by group, by partition, totals
# --------------------------------------------------------------------------


@pytest.fixture
def usage_fixture(store):
    """Two accounts, two partitions, several attempts. The shape of a report."""
    store.create_run("a1", backend="slurm", job_id="1", spec={"account": "mlopt"})
    store.create_run("a2", backend="slurm", job_id="2", spec={"account": "mlopt"})
    store.create_run("b1", backend="slurm", job_id="3", spec={"account": "other"})
    store.create_run("local", backend="local", job_id="999")  # no spec at all

    store.upsert_usage(
        [
            usage_row(job_id="1", run_id="a1", partition="ckpt-all", elapsed_s=3600, gpus=2, cpus=8),
            usage_row(
                job_id="2",
                run_id="a2",
                partition="gpu-mlopt",
                start="2026-09-16T18:30:00.000Z",
                elapsed_s=1800,
                gpus=1,
                cpus=4,
            ),
            usage_row(
                job_id="3",
                run_id="b1",
                partition="ckpt-all",
                start="2026-09-16T19:00:00.000Z",
                elapsed_s=7200,
                gpus=4,
                cpus=16,
            ),
            usage_row(
                job_id="999",
                run_id="local",
                partition=None,
                start="2026-09-16T20:00:00.000Z",
                elapsed_s=3600,
                gpus=0,
                cpus=2,
            ),
        ]
    )
    return store


def test_usage_by_group_reads_the_account_out_of_the_spec(usage_fixture):
    rows = {r["key"]: r for r in usage_fixture.usage_by("group")}
    assert set(rows) == {"mlopt", "other", "(none)"}
    assert rows["mlopt"]["gpu_hours"] == pytest.approx(2.0 + 0.5)
    assert rows["mlopt"]["runs"] == 2
    assert rows["other"]["gpu_hours"] == pytest.approx(8.0)
    # A run with no account is still hours somebody spent; it must not vanish.
    assert rows["(none)"]["cpu_hours"] == pytest.approx(2.0)


def test_usage_by_partition_splits_ckpt_from_the_group_allocation(usage_fixture):
    rows = {r["key"]: r for r in usage_fixture.usage_by("partition")}
    assert rows["ckpt-all"]["gpu_hours"] == pytest.approx(2.0 + 8.0)
    assert rows["gpu-mlopt"]["gpu_hours"] == pytest.approx(0.5)
    assert rows["(none)"]["gpu_hours"] == pytest.approx(0.0)


def test_usage_by_orders_biggest_first(usage_fixture):
    keys = [r["key"] for r in usage_fixture.usage_by("run")]
    assert keys[0] == "b1"  # 8 GPU-hours


def test_group_filter_applies_to_every_grouping(usage_fixture):
    rows = usage_fixture.usage_by("run", group="mlopt")
    assert {r["key"] for r in rows} == {"a1", "a2"}
    assert usage_fixture.usage_totals(group="mlopt")["gpu_hours"] == pytest.approx(2.5)


def test_usage_totals_include_the_partition_split(usage_fixture):
    totals = usage_fixture.usage_totals()
    assert totals["gpu_hours"] == pytest.approx(2.0 + 0.5 + 8.0 + 0.0)
    assert totals["cpu_hours"] == pytest.approx(8.0 + 2.0 + 32.0 + 2.0)
    assert totals["attempts"] == 4
    assert totals["runs"] == 4

    split = {p["partition"]: p["gpu_hours"] for p in totals["by_partition"]}
    assert split["ckpt-all"] == pytest.approx(10.0)
    assert split["gpu-mlopt"] == pytest.approx(0.5)
    assert sum(split.values()) == pytest.approx(totals["gpu_hours"])


def test_usage_totals_on_an_empty_database_are_zero(store):
    totals = store.usage_totals()
    assert totals == {
        "gpu_hours": 0.0,
        "cpu_hours": 0.0,
        "attempts": 0,
        "runs": 0,
        "by_partition": [],
    }


def test_since_filters_on_the_attempt_start(usage_fixture):
    rows = usage_fixture.usage_by("run", since="2026-09-16T19:00:00.000Z")
    assert {r["key"] for r in rows} == {"b1", "local"}
    # A plain date works too, which is what `relay usage --since 2026-09-17`
    # will hand us.
    assert usage_fixture.usage_by("run", since="2026-09-17") == []


def test_unknown_grouping_and_unparseable_since_are_rejected(usage_fixture):
    with pytest.raises(ValueError):
        usage_fixture.usage_by("node")
    with pytest.raises(ValueError):
        usage_fixture.usage_by("run", since="last tuesday")


# --------------------------------------------------------------------------
# Wasted hours
# --------------------------------------------------------------------------


def test_failed_runs_waste_every_one_of_their_attempts(store):
    store.create_run("bad", backend="slurm", job_id="1", status="failed")
    store.create_run("good", backend="slurm", job_id="2", status="completed")
    store.upsert_usage(
        [
            usage_row(job_id="1", run_id="bad", start="2026-09-16T18:00:00.000Z", elapsed_s=3600),
            usage_row(job_id="1", run_id="bad", start="2026-09-16T19:00:00.000Z", elapsed_s=1800),
            usage_row(job_id="2", run_id="good", start="2026-09-16T18:00:00.000Z", elapsed_s=3600),
        ]
    )

    failed = store.wasted_hours()["failed"]
    assert failed["runs"] == ["bad"]
    assert failed["gpu_hours"] == pytest.approx(1.5)  # both attempts, not just the last
    assert failed["cpu_hours"] == pytest.approx(6.0)


def test_preempted_attempt_loses_only_the_tail_after_its_last_checkpoint(run):
    run.upsert_usage(
        [
            usage_row(
                state="PREEMPTED",
                start="2026-09-16T18:00:00.000Z",
                end="2026-09-16T19:00:00.000Z",
                elapsed_s=3600,
                gpus=2,
                cpus=8,
            )
        ]
    )
    run.ingest(
        "r1",
        "log",
        [
            event(1, "checkpoint", {"step": 100}, ts="2026-09-16T18:10:00.000Z"),
            event(2, "checkpoint", {"step": 400}, ts="2026-09-16T18:40:00.000Z"),
            # Belongs to the *next* attempt, after this one died. Counting it
            # would make the lost time look like zero.
            event(3, "checkpoint", {"step": 700}, ts="2026-09-16T19:30:00.000Z"),
        ],
        new_offset=300,
    )

    lost = run.wasted_hours()["lost_to_preemption"]
    attempt = lost["attempts"][0]
    assert attempt["no_checkpoints"] is False
    assert attempt["lost_s"] == 20 * 60  # 18:40 -> 19:00
    assert attempt["run_id"] == "r1" and attempt["job_id"] == "100"
    assert lost["gpu_hours"] == pytest.approx(20 * 60 * 2 / 3600)
    assert lost["cpu_hours"] == pytest.approx(20 * 60 * 8 / 3600)


def test_an_attempt_with_no_checkpoints_loses_all_of_itself(run):
    run.upsert_usage(
        [
            usage_row(
                state="PREEMPTED",
                start="2026-09-16T18:00:00.000Z",
                end="2026-09-16T19:00:00.000Z",
                elapsed_s=3600,
                gpus=1,
            )
        ]
    )
    # Metrics but no checkpoint events: the script is training and not saving,
    # or not reporting that it saves. Either way the user should be told.
    run.ingest("r1", "log", [metric_event(1, 10, {"loss": 1.0})], new_offset=80)

    lost = run.wasted_hours()["lost_to_preemption"]
    assert lost["attempts"][0]["no_checkpoints"] is True
    assert lost["attempts"][0]["lost_s"] == 3600
    assert lost["gpu_hours"] == pytest.approx(1.0)


def test_requeued_counts_as_interrupted_too(run):
    run.upsert_usage([usage_row(state="REQUEUED", elapsed_s=3600, gpus=1)])
    assert len(run.wasted_hours()["lost_to_preemption"]["attempts"]) == 1


def test_a_running_attempt_has_not_lost_anything_yet(run):
    run.upsert_usage([usage_row(state="RUNNING", end=None, elapsed_s=1200, gpus=1)])
    lost = run.wasted_hours()["lost_to_preemption"]
    assert lost["attempts"] == []
    assert lost["gpu_hours"] == 0.0


def test_completed_attempts_are_not_wasted(run):
    run.upsert_usage([usage_row(state="COMPLETED", elapsed_s=3600, gpus=1)])
    wasted = run.wasted_hours()
    assert wasted["lost_to_preemption"]["attempts"] == []
    assert wasted["failed"]["runs"] == []


def test_wasted_hours_respects_since_and_group(store):
    store.create_run("r", backend="slurm", job_id="1", status="failed", spec={"account": "mlopt"})
    store.upsert_usage(
        [
            usage_row(job_id="1", run_id="r", start="2026-09-01T00:00:00.000Z", elapsed_s=3600),
            usage_row(job_id="1", run_id="r", start="2026-09-20T00:00:00.000Z", elapsed_s=3600),
        ]
    )
    assert store.wasted_hours()["failed"]["gpu_hours"] == pytest.approx(2.0)
    assert store.wasted_hours(since="2026-09-10")["failed"]["gpu_hours"] == pytest.approx(1.0)
    assert store.wasted_hours(group="mlopt")["failed"]["runs"] == ["r"]
    assert store.wasted_hours(group="nobody")["failed"]["runs"] == []


# --------------------------------------------------------------------------
# Daemon state
# --------------------------------------------------------------------------


def test_daemon_state_is_all_none_before_anything_is_written(store):
    assert store.get_daemon_state() == {
        "last_error_kind": None,
        "last_error_ts": None,
        "last_error_detail": None,
        "last_usage_fetch_ts": None,
        "last_ok_ts": None,
    }


def test_daemon_state_round_trips(store):
    store.set_daemon_state(
        last_error_kind="auth_required",
        last_error_ts="2026-09-16T18:00:00.000Z",
        last_error_detail="Permission denied (publickey)",
    )
    state = store.get_daemon_state()
    assert state["last_error_kind"] == "auth_required"
    assert state["last_error_detail"] == "Permission denied (publickey)"


def test_partial_updates_leave_the_other_fields_alone(store):
    store.set_daemon_state(last_usage_fetch_ts="2026-09-16T18:00:00.000Z")
    store.set_daemon_state(last_error_kind="unreachable")
    store.set_daemon_state(last_ok_ts="2026-09-16T18:05:00.000Z")

    state = store.get_daemon_state()
    assert state["last_usage_fetch_ts"] == "2026-09-16T18:00:00.000Z"
    assert state["last_error_kind"] == "unreachable"
    assert state["last_ok_ts"] == "2026-09-16T18:05:00.000Z"


def test_clear_daemon_error_clears_only_the_error(store):
    store.set_daemon_state(
        last_error_kind="unreachable",
        last_error_ts="2026-09-16T18:00:00.000Z",
        last_error_detail="ssh: connect to host: Operation timed out",
        last_usage_fetch_ts="2026-09-16T17:00:00.000Z",
    )
    store.clear_daemon_error()

    state = store.get_daemon_state()
    assert state["last_error_kind"] is None
    assert state["last_error_ts"] is None
    assert state["last_error_detail"] is None
    assert state["last_usage_fetch_ts"] == "2026-09-16T17:00:00.000Z"


def test_daemon_state_stays_one_row(store):
    store.set_daemon_state(last_ok_ts="a")
    store.set_daemon_state(last_ok_ts="b")
    count = store.conn.execute("SELECT count(*) FROM daemon_state").fetchone()[0]
    assert count == 1
    assert store.get_daemon_state()["last_ok_ts"] == "b"


def test_daemon_state_rejects_unknown_fields(store):
    with pytest.raises(ValueError):
        store.set_daemon_state(mood="anxious")


# --------------------------------------------------------------------------
# Concurrency: this is why WAL and busy_timeout exist
# --------------------------------------------------------------------------


def test_two_stores_on_one_file_read_and_write(db_path):
    writer = Store(db_path)
    reader = Store(db_path)

    writer.create_run("r1", backend="local")
    writer.ingest("r1", "log", [metric_event(1, 10, {"loss": 0.5})], new_offset=64)

    assert reader.get_run("r1") is not None
    assert reader.metric_series("r1", "loss")[0]["value"] == 0.5

    # And the second connection can write too.
    reader.update_run_status("r1", "running")
    assert writer.get_run("r1")["status"] == "running"

    writer.close()
    reader.close()


def test_reader_is_not_blocked_by_an_open_write_transaction(db_path):
    """WAL's headline property: a writer in flight does not stall readers."""
    writer = Store(db_path)
    reader = Store(db_path)
    writer.create_run("r1", backend="local", name="visible")

    writer.conn.execute("BEGIN IMMEDIATE")
    writer.conn.execute("UPDATE runs SET name = 'not yet' WHERE run_id = 'r1'")
    try:
        # Reads the last committed state immediately, no waiting, no error.
        assert reader.get_run("r1")["name"] == "visible"
    finally:
        writer.conn.execute("COMMIT")

    assert reader.get_run("r1")["name"] == "not yet"
    writer.close()
    reader.close()


def test_second_writer_waits_instead_of_failing(db_path):
    """busy_timeout=5000 turns a lock collision into a short wait, not a crash."""
    other = Store(db_path)
    other.create_run("r1", backend="local")

    locked = threading.Event()

    def hold_the_write_lock():
        # A sqlite3 connection belongs to the thread that made it, so the
        # holder opens its own Store here -- which is also how the real system
        # works: separate processes, separate connections.
        holder = Store(db_path)
        holder.conn.execute("BEGIN IMMEDIATE")
        locked.set()
        time.sleep(0.3)
        holder.conn.execute("COMMIT")
        holder.close()

    thread = threading.Thread(target=hold_the_write_lock)
    thread.start()
    assert locked.wait(timeout=5), "helper thread never took the write lock"

    started = time.monotonic()
    try:
        # Would raise "database is locked" instantly without busy_timeout.
        other.ingest("r1", "log", [event(1, "log", {"text": "x"})], new_offset=20)
    except sqlite3.OperationalError as exc:  # pragma: no cover - failure path
        pytest.fail(f"write should have waited for the lock, got: {exc}")
    waited = time.monotonic() - started
    thread.join()

    # It really did wait for the other writer, and it really did finish.
    assert 0.1 < waited < 5.0
    assert other.count_events("r1") == 1
    other.close()


# --------------------------------------------------------------------------


def _snapshot(store):
    """Everything that should be identical after a repeated ingest."""
    return {
        "events": [json.dumps(e, sort_keys=True) for e in store.list_events("r1")],
        "metrics": store.metric_series("r1", "loss"),
        "offset": store.get_offset("r1", "log"),
        "last_step": store.get_run("r1")["last_step"],
        "heartbeat": store.get_run("r1")["last_heartbeat_ts"],
    }
