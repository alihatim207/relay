"""The store: relay's local SQLite database, and the only place SQL lives.

Three processes share one file on the laptop:

  * the daemon writes (it tails `events.jsonl` over SSH and ingests new bytes),
  * the CLI reads (`relay ls`, `relay show`, `relay logs`),
  * the dashboard reads (FastAPI polling every two seconds).

Nothing here talks to the cluster and nothing here parses an event line. The
daemon hands us event dicts that are already parsed and validated; our job is
to write them down in a way that survives being killed at any moment.

Two ideas do most of the work:

1. **Composite primary keys plus `INSERT OR IGNORE`.** Re-inserting bytes we
   have already seen is a no-op rather than a duplicate or an error. That is
   what makes "just restart it" a valid recovery strategy everywhere else in
   relay: the daemon may re-read the same region of the file as many times as
   it likes.

   The `usage` table is the one exception, and it is explained where it lives:
   a running attempt's elapsed seconds grow, so its rows update in place. It is
   still idempotent, which is the property that actually matters.

2. **One transaction per ingest.** Events, the metric rows derived from them,
   the run's live state, and the file byte offset all commit together. If they
   were separate statements, a crash in between would leave the offset saying
   "I have read 4000 bytes" while the events from those bytes were missing,
   and we would never go back for them.

Everything is plain `sqlite3` from the standard library. The sophisticated
alternative would be an ORM (SQLAlchemy) or a migration framework (alembic).
For seven tables that only ever grow new columns, neither pays for itself, and
hand-written SQL keeps the storage behaviour readable in one file. Adding a
column to an existing database is handled by `_migrate`, which is fifteen lines.

GPU-hours, CPU-hours and wasted hours are *queries*, never columns. Storing a
total would mean keeping a second copy of the truth in step with the rows it
came from, and those rows change while a job runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

# --------------------------------------------------------------------------
# Constants shared with the rest of relay
# --------------------------------------------------------------------------

# Where the database lives when nobody says otherwise. Follows the XDG spec:
# ~/.local/share is for application state the user does not edit by hand
# (config goes elsewhere, under ~/.config). Tests always pass an explicit path.
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "relay" / "relay.db"

# The statuses a run may have. `status` is owned by the *scheduler* -- Slurm
# decides whether a job is queued, running or finished -- so the daemon writes
# it from `squeue`/`sacct` output, never from the event log.
STATUSES = frozenset(
    {
        "queued",
        "running",
        "completed",
        "failed",
        "requeued",
        "cancelled",
    }
)

# A terminal run is one we will never hear from again, so the daemon stops
# polling it. Note what is *not* here: "requeued". A preempted job goes back
# into the queue and keeps writing to the same event log with the same run ID,
# so it must stay in the daemon's working set.
TERMINAL = frozenset({"completed", "failed", "cancelled"})

# Well-known keys in the `meta` table. They are constants rather than string
# literals scattered across modules so the daemon, the CLI and doctor cannot
# drift apart on spelling.
META_LAST_SYNC_TS = "last_sync_ts"  # when the daemon last finished a full cycle
META_DAEMON_STARTED_TS = "daemon_started_ts"  # when the running daemon booted
META_SCHEMA_VERSION = "schema_version"  # bumped only if the schema changes shape

# The daemon polls the *filesystem* (the event logs) and the *scheduler*
# (`squeue`) on two different schedules, because they cost two very different
# things: a tail is a read on shared storage, while a squeue is a question put
# to slurmctld, which every user on the cluster shares. So there are two
# freshness timestamps, not one, and `relay ls` shows both -- metrics two
# seconds old alongside job state thirty seconds old is the normal, healthy
# state of affairs, and a single "last synced" number would hide it.
META_LAST_SCHEDULER_TS = "last_scheduler_ts"  # last successful scheduler poll

# Set by `submit` and `cancel` to tell the daemon "something just changed that
# you will want to see". The daemon's scheduler poll backs off while nothing is
# happening (up to five minutes), and without this a freshly submitted run
# could sit invisible for that whole time. It lives in `meta` rather than in a
# `daemon_state` column precisely because `meta` is key/value: a new well-known
# key needs no ALTER TABLE, so an existing database picks this up for free.
META_SCHEDULER_POKE_TS = "scheduler_poke_ts"

# 1 -> 2 added usage accounting: the `usage` and `daemon_state` tables and the
# `runs.attempts` / `runs.usage_final` columns. Nothing was removed or renamed,
# so an old database is migrated forward in place (see `_migrate`).
SCHEMA_VERSION = 2

# The scheduler state words that mean "this attempt was cut short while it was
# still working". They are the raw strings sacct prints, not relay's own status
# vocabulary, because usage accounting has to tell a preempted attempt from a
# failed one and `STATUSES` deliberately folds those together.
INTERRUPTED_STATES = ("PREEMPTED", "REQUEUED")

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
#
# Kept as one string of plain DDL so the whole shape of the database is visible
# at a glance. Every statement is `IF NOT EXISTS`, so running it on an existing
# database is harmless -- which means every process can call it on startup and
# nobody has to own "who creates the tables".

SCHEMA = """
-- One row per run. `run_id` is relay's own ID (e.g. vr_s1_7f3a), never Slurm's
-- job ID; Slurm's lives in `job_id` and changes on requeue, which is exactly
-- why we do not key anything on it.
CREATE TABLE IF NOT EXISTS runs (
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
    spec              TEXT,
    -- How many times the job has started, counted from `run_started` events:
    -- 1 on a first run, 2 after one requeue, and so on. The sidecar reports a
    -- 0-based `attempt` (from SLURM_RESTART_COUNT) and we store the count.
    attempts          INTEGER NOT NULL DEFAULT 0,
    -- 1 once we have fetched usage for this run after it went terminal. A
    -- finished job's sacct numbers never change again, so one final fetch is
    -- enough and the daemon can stop asking about it forever after.
    usage_final       INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS runs_by_status ON runs (status);

-- The faithful history: one row per event line, payload kept as the JSON text
-- we were given. We never rewrite an event. (run_id, seq) is the primary key,
-- so re-reading the same bytes cannot duplicate anything.
CREATE TABLE IF NOT EXISTS events (
    run_id  TEXT NOT NULL,
    seq     INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    type    TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);

-- `relay logs` and the dashboard both want "events of one type, after seq N",
-- which this index answers without scanning the run's whole history.
CREATE INDEX IF NOT EXISTS events_by_type ON events (run_id, type, seq);

-- The denormalised read path for charting: one row per metric name per step,
-- written at ingest time. Without it, drawing a 200k-point loss curve would
-- mean JSON-parsing 200k payloads on every dashboard poll.
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    name   TEXT NOT NULL,
    step   INTEGER NOT NULL,
    value  REAL NOT NULL,
    ts     TEXT,
    PRIMARY KEY (run_id, name, step)
);

-- How far we have read each file we tail. One run could in principle have more
-- than one source file, so the key is (run_id, path), not run_id alone.
CREATE TABLE IF NOT EXISTS sources (
    run_id      TEXT NOT NULL,
    path        TEXT NOT NULL,
    byte_offset INTEGER NOT NULL DEFAULT 0,
    updated_ts  TEXT,
    PRIMARY KEY (run_id, path)
);

-- Small key/value scratchpad for process-wide facts that belong to no run:
-- when the daemon last completed a cycle (the CLI prints "last synced Ns ago"),
-- when it started (doctor reports on it), the schema version.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- What the scheduler charged us, one row per *attempt*. A job preempted twice
-- and requeued has three attempts under one job ID, and each burned real
-- GPU-hours, so the key is (job_id, start): attempts share a job ID but never
-- a start time.
--
-- `state` is the scheduler's own word ("COMPLETED", "PREEMPTED", "REQUEUED",
-- "RUNNING", ...), kept verbatim rather than mapped onto relay's statuses.
-- `end` is NULL while an attempt is still running. Both `end` and `partition`
-- are SQL keywords, which is why every statement below quotes them.
CREATE TABLE IF NOT EXISTS usage (
    job_id      TEXT NOT NULL,
    start       TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    state       TEXT NOT NULL,
    "partition" TEXT,
    "end"       TEXT,
    elapsed_s   INTEGER NOT NULL,
    gpus        INTEGER NOT NULL DEFAULT 0,
    cpus        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job_id, start)
);

-- Every usage report is "for these runs", so the run ID is the column we
-- filter and group on far more often than the job ID.
CREATE INDEX IF NOT EXISTS usage_by_run ON usage (run_id);

-- One row, ever: the daemon's own health, which belongs to no run. The CHECK
-- makes that structural rather than a convention -- there is no way to insert
-- a second row by accident, so readers never have to ask "which one?".
--
-- `last_error_kind` is "auth_required" or "unreachable". The CLI and the
-- dashboard read it to print "Authentication needed. Run `relay connect`."
-- instead of a generic transport error.
CREATE TABLE IF NOT EXISTS daemon_state (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    last_error_kind     TEXT,
    last_error_ts       TEXT,
    last_error_detail   TEXT,
    last_usage_fetch_ts TEXT,
    last_ok_ts          TEXT
);
"""

# Columns added to `runs` after version 1 shipped, as (name, DDL) pairs.
#
# SQLite has no `ADD COLUMN IF NOT EXISTS`, and `CREATE TABLE IF NOT EXISTS`
# above does nothing at all to a table that already exists -- so a database
# created before this change would keep the old five-column `runs` and every
# query mentioning `attempts` would fail. `_migrate` therefore asks
# `PRAGMA table_info(runs)` what is actually there and adds what is missing.
#
# The rule for anything added here: it must have a default, because SQLite
# fills existing rows with that default and cannot do otherwise.
_RUNS_ADDED_COLUMNS = (
    ("attempts", "attempts INTEGER NOT NULL DEFAULT 0"),
    ("usage_final", "usage_final INTEGER NOT NULL DEFAULT 0"),
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing Z, matching the event log.

    All timestamps in the database are strings in this format. That is a
    deliberate choice: ISO-8601 UTC sorts correctly as plain text, so SQLite can
    compare and order timestamps with no date functions and no timezone
    ambiguity, and the values are readable when you open the database by hand.
    """
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _row(row: sqlite3.Row | None) -> dict | None:
    """Turn one `sqlite3.Row` into a plain dict (or pass None through).

    Callers get dicts, never Row objects, so that nothing outside this module
    depends on sqlite3 types -- the CLI can `json.dumps` a result directly.
    """
    return None if row is None else dict(row)


def _rows(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


def _is_number(value: object) -> bool:
    """True for real numbers only.

    `isinstance(True, int)` is True in Python, so booleans have to be excluded
    explicitly or a user logging `{"converged": true}` would get a metric series
    of 1.0s and 0.0s in their loss chart.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# Columns of `runs` that callers are allowed to set. An allowlist rather than
# f-string interpolation of whatever key was passed: column names cannot be
# parameterised in SQL, so this is the only safe way to build an UPDATE from
# keyword arguments.
_RUN_COLUMNS = (
    "name",
    "backend",
    "job_id",
    "status",
    "stale",
    "last_heartbeat_ts",
    "last_step",
    "run_dir",
    "spec",
)
# `attempts` and `usage_final` are deliberately absent: they are derived state,
# owned by `ingest` and `mark_usage_final` respectively. Nothing else should be
# able to set them by hand.

# The same idea for the single `daemon_state` row.
_DAEMON_STATE_COLUMNS = (
    "last_error_kind",
    "last_error_ts",
    "last_error_detail",
    "last_usage_fetch_ts",
    "last_ok_ts",
)

# Columns of `usage` that an upsert overwrites when the row already exists.
# Everything except the primary key (job_id, start), which is what identifies
# the attempt in the first place.
_USAGE_UPDATED_COLUMNS = ("run_id", "state", '"partition"', '"end"', "elapsed_s", "gpus", "cpus")

# The run's Slurm account, dug out of the JSON spec we stored at submit time.
# Two guards worth explaining:
#   * `json_valid` -- `json_extract` raises on text that is not JSON, and a
#     malformed spec should not take down the whole usage report.
#   * `COALESCE(..., '(none)')` -- a run with no account (every local run) has
#     to land in some group, and grouping on NULL would drop it from the report
#     entirely, which looks like the hours simply vanished.
# `r` is the alias every query below gives the `runs` table.
_ACCOUNT_SQL = (
    "COALESCE(CASE WHEN json_valid(r.spec) THEN json_extract(r.spec, '$.account') END, '(none)')"
)

# One place for the arithmetic, so per-run, per-group and total numbers cannot
# drift apart. Seconds -> hours is /3600.0; the `.0` matters, because integer
# division would silently floor every short attempt to zero hours.
_GPU_HOURS_SQL = "COALESCE(SUM(u.elapsed_s * u.gpus), 0) / 3600.0"
_CPU_HOURS_SQL = "COALESCE(SUM(u.elapsed_s * u.cpus), 0) / 3600.0"


class Store:
    """A connection to relay's SQLite database.

    One `Store` owns one connection and is *not* thread-safe; give each thread
    (or each process) its own. That is fine because SQLite itself, in WAL mode,
    is the thing coordinating the three processes.
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        # sqlite3 will happily create the file but not the directory above it.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # isolation_level=None turns OFF the sqlite3 module's habit of opening
        # implicit transactions for us. We want transaction boundaries to be
        # explicit and visible (`BEGIN IMMEDIATE` ... `COMMIT`), because the
        # atomicity of ingest is the single most important property here.
        self.conn = sqlite3.connect(
            str(self.path),
            isolation_level=None,
            timeout=5.0,
        )
        self.conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._create_schema()

    # -- setup ------------------------------------------------------------

    def _apply_pragmas(self) -> None:
        """Pragmas are per-connection, so every connection must set them.

        journal_mode is the exception: it is a property of the database file
        and persists, but setting it again costs nothing.
        """
        # WAL: readers and writers stop blocking each other. With the default
        # rollback journal, the dashboard polling every 2s would collide with
        # the daemon writing every 2s and one of them would see "database is
        # locked". WAL is required, not an optimisation.
        self.conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL: fsync at WAL checkpoints rather than on every commit. The
        # worst case is losing the last few commits in an OS crash -- and we
        # would simply re-read those bytes from the event log, because the
        # offset was in the same lost transaction. Correctness is preserved by
        # idempotence, so we can afford the faster setting.
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # Wait up to 5 seconds for a lock instead of failing instantly. Three
        # processes means brief contention is normal, not exceptional.
        self.conn.execute("PRAGMA busy_timeout=5000")
        # Enforce (run_id) references implicitly by convention rather than with
        # real foreign keys: the daemon can legitimately ingest events for a run
        # whose row is written by a different process a moment later, and a
        # foreign key would reject them. Ordering across processes is not
        # something we want the database to police.

    def _create_schema(self) -> None:
        # Not wrapped in our own transaction: `executescript` issues an implicit
        # COMMIT before it runs, so a BEGIN around it would be undone anyway.
        # Every statement is CREATE ... IF NOT EXISTS, so running this on an
        # existing database, from any process, is a no-op.
        self.conn.executescript(SCHEMA)
        self._migrate()
        with self._writing() as conn:
            # Upsert, not INSERT OR IGNORE: a database created by an older
            # relay already has a `schema_version` row saying 1, and having
            # just migrated it we want the number to be true.
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (META_SCHEMA_VERSION, str(SCHEMA_VERSION)),
            )

    def _migrate(self) -> None:
        """Add columns that a database created by an older relay is missing.

        This is the whole migration system, and it is deliberately tiny: ask
        SQLite which columns `runs` has, and `ALTER TABLE ADD COLUMN` the ones
        that are not there. Existing databases from before usage accounting
        must keep working -- the user has months of runs in them -- and SQLite
        has no `ADD COLUMN IF NOT EXISTS` to lean on.

        The sophisticated alternative is a real migration framework (alembic)
        with numbered up/down scripts. That pays for itself when columns get
        renamed or dropped; for a schema that only ever grows a column with a
        default, it would be more machinery than schema.
        """
        present = {row["name"] for row in self.conn.execute("PRAGMA table_info(runs)")}
        for column, ddl in _RUNS_ADDED_COLUMNS:
            if column not in present:
                # No placeholders: SQL cannot parameterise an identifier, and
                # these strings are module constants, not user input.
                self.conn.execute(f"ALTER TABLE runs ADD COLUMN {ddl}")

    # -- transactions -----------------------------------------------------

    @contextmanager
    def _writing(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside one write transaction. Commit, or roll it all back.

        `BEGIN IMMEDIATE` (rather than a plain `BEGIN`) takes the write lock up
        front. The subtle reason: a deferred transaction that reads first and
        writes later has to *upgrade* its lock at the first write, and SQLite
        does not apply busy_timeout to that upgrade -- it fails immediately with
        SQLITE_BUSY. Taking the lock at the start means the 5-second timeout
        actually protects us.
        """
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- diagnostics ------------------------------------------------------

    def journal_mode(self) -> str:
        """The file's journal mode, lowercased. `doctor` checks this says 'wal'."""
        row = self.conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    # ------------------------------------------------------------------
    # Ingest: the one method where atomicity matters
    # ------------------------------------------------------------------

    def ingest(
        self,
        run_id: str,
        path: str,
        events: list[dict],
        new_offset: int,
    ) -> dict:
        """Write a batch of parsed events and the new byte offset, atomically.

        `events` are dicts as produced by the daemon's parser: keys `v`, `run`,
        `seq`, `ts`, `type`, `payload`. `new_offset` is where the daemon should
        resume reading `path` next cycle -- the daemon has already rewound it
        past any trailing incomplete line, so we store exactly what we are told.

        Four things happen here, and they happen together or not at all:

          1. event rows are inserted,
          2. metric rows are derived from `metric` events and inserted,
          3. the run's live state (last_step, last_heartbeat_ts) is refreshed,
          4. the source byte offset is moved forward.

        Step 3 lives here, rather than in the daemon, precisely *because* of the
        single-transaction rule. The daemon is the one that knows a heartbeat
        arrived, but if it wrote `last_heartbeat_ts` in a separate statement, a
        crash in between would leave a run looking stale while the heartbeat
        that proves otherwise is sitting in the events table. Derived state must
        commit with the data it is derived from.

        Returns a small summary dict: how many event and metric rows were
        actually new, and the offset now stored.
        """
        metric_rows: list[tuple] = []
        last_step: int | None = None
        last_heartbeat_ts: str | None = None
        attempts: int | None = None

        # Work out everything we want to write *before* opening the transaction,
        # so the write lock is held for as short a time as possible.
        for event in events:
            if event["type"] == "metric":
                payload = event.get("payload") or {}
                step = payload.get("step")
                values = payload.get("values") or {}
                if isinstance(step, int) and isinstance(values, dict):
                    for name, value in values.items():
                        # Anything non-numeric (a string label, a bool) is kept
                        # in the event payload but left out of `metrics`, which
                        # is a chart-shaped table of REALs.
                        if _is_number(value):
                            metric_rows.append(
                                (run_id, str(name), step, float(value), event["ts"])
                            )
                    last_step = step if last_step is None else max(last_step, step)
            elif event["type"] == "heartbeat":
                payload = event.get("payload") or {}
                # ISO-8601 UTC strings compare correctly with `max`, which is
                # the whole reason we store timestamps as text.
                ts = event["ts"]
                last_heartbeat_ts = (
                    ts if last_heartbeat_ts is None else max(last_heartbeat_ts, ts)
                )
                # The sidecar spells the heartbeat payload {"last_step": N},
                # not {"step": N} like metric events — it reports the last
                # step it *saw*, which may be None before the first metric.
                step = payload.get("last_step")
                if isinstance(step, int):
                    last_step = step if last_step is None else max(last_step, step)
            elif event["type"] == "run_started":
                payload = event.get("payload") or {}
                # `attempt` is 0-based, straight from SLURM_RESTART_COUNT: 0 on
                # the first start, 1 after one requeue. We store a *count*, so
                # the third attempt (attempt=2) means attempts=3.
                #
                # A v1 `run_started` predates the field entirely. Treating a
                # missing `attempt` as 0 is the right reading: schema v1 had no
                # requeue-aware sidecar, so every one of those was a first start.
                attempt = payload.get("attempt")
                if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
                    attempt = 0
                count = attempt + 1
                attempts = count if attempts is None else max(attempts, count)

        event_rows = [
            (
                run_id,
                event["seq"],
                event["ts"],
                event["type"],
                json.dumps(event.get("payload") or {}, separators=(",", ":")),
            )
            for event in events
        ]

        now = utc_now_iso()

        with self._writing() as conn:
            before_events = _scalar(conn, "SELECT count(*) FROM events WHERE run_id = ?", (run_id,))
            before_metrics = _scalar(
                conn, "SELECT count(*) FROM metrics WHERE run_id = ?", (run_id,)
            )

            # INSERT OR IGNORE plus the (run_id, seq) primary key is what makes
            # re-reading harmless. A duplicate seq keeps the row we already had
            # rather than overwriting it -- events are immutable history, and
            # the first version we saw is the one that was actually on disk.
            conn.executemany(
                "INSERT OR IGNORE INTO events (run_id, seq, ts, type, payload)"
                " VALUES (?, ?, ?, ?, ?)",
                event_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO metrics (run_id, name, step, value, ts)"
                " VALUES (?, ?, ?, ?, ?)",
                metric_rows,
            )

            # Advance the run's live state. These are CASE expressions rather
            # than plain assignments so that an out-of-order or replayed batch
            # can never drag last_step or the heartbeat backwards in time.
            if last_step is not None or last_heartbeat_ts is not None or attempts is not None:
                conn.execute(
                    "UPDATE runs SET"
                    "   last_step = CASE"
                    "     WHEN ?1 IS NULL THEN last_step"
                    "     WHEN last_step IS NULL OR last_step < ?1 THEN ?1"
                    "     ELSE last_step END,"
                    "   last_heartbeat_ts = CASE"
                    "     WHEN ?2 IS NULL THEN last_heartbeat_ts"
                    "     WHEN last_heartbeat_ts IS NULL OR last_heartbeat_ts < ?2 THEN ?2"
                    "     ELSE last_heartbeat_ts END,"
                    # Same never-backwards shape as last_step: re-reading an old
                    # attempt's `run_started` must not un-count the later ones.
                    "   attempts = CASE"
                    "     WHEN ?3 IS NULL THEN attempts"
                    "     WHEN attempts IS NULL OR attempts < ?3 THEN ?3"
                    "     ELSE attempts END,"
                    "   updated_ts = ?4"
                    " WHERE run_id = ?5",
                    (last_step, last_heartbeat_ts, attempts, now, run_id),
                )
                # If no `runs` row exists yet this UPDATE matches nothing, which
                # is correct: the row is created by `relay submit`, and events
                # for a run we have never heard of are still worth keeping.

            # The offset is the one value that must *not* use INSERT OR IGNORE:
            # it is mutable state that has to move forward, so we upsert it.
            conn.execute(
                "INSERT INTO sources (run_id, path, byte_offset, updated_ts)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(run_id, path) DO UPDATE SET"
                "   byte_offset = excluded.byte_offset,"
                "   updated_ts  = excluded.updated_ts",
                (run_id, path, int(new_offset), now),
            )

            after_events = _scalar(conn, "SELECT count(*) FROM events WHERE run_id = ?", (run_id,))
            after_metrics = _scalar(
                conn, "SELECT count(*) FROM metrics WHERE run_id = ?", (run_id,)
            )

        return {
            "events_inserted": after_events - before_events,
            "metrics_inserted": after_metrics - before_metrics,
            "byte_offset": int(new_offset),
        }

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self,
        run_id: str,
        *,
        backend: str,
        name: str | None = None,
        job_id: str | None = None,
        status: str = "queued",
        run_dir: str | None = None,
        spec: dict | None = None,
        created_ts: str | None = None,
    ) -> dict:
        """Insert a run row if it is not already there, and return the row.

        `INSERT OR IGNORE` again: submitting is not supposed to be retried, but
        if it is (a flaky SSH connection, a user re-running a command), we would
        rather keep the original row than clobber it.
        """
        ts = created_ts or utc_now_iso()
        with self._writing() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs"
                " (run_id, name, backend, job_id, status, stale,"
                "  created_ts, updated_ts, run_dir, spec)"
                " VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
                (
                    run_id,
                    name,
                    backend,
                    job_id,
                    status,
                    ts,
                    ts,
                    run_dir,
                    json.dumps(spec) if spec is not None else None,
                ),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def upsert_run(self, run_id: str, *, backend: str, **fields) -> dict:
        """Create the run if new, then apply any given fields to it.

        Used by code that does not know or care whether the run already exists
        (the daemon reconciling against `squeue`, for instance).
        """
        self.create_run(run_id, backend=backend)
        if fields:
            self.update_run(run_id, **fields)
        return self.get_run(run_id)  # type: ignore[return-value]

    def update_run(self, run_id: str, **fields) -> None:
        """Set any subset of the mutable run columns, plus `updated_ts`.

        Column names come from the `_RUN_COLUMNS` allowlist because SQL cannot
        parameterise an identifier -- only values get `?` placeholders.
        """
        unknown = set(fields) - set(_RUN_COLUMNS)
        if unknown:
            raise ValueError(f"unknown run column(s): {sorted(unknown)}")
        if not fields:
            return

        values = dict(fields)
        if "spec" in values and isinstance(values["spec"], dict):
            values["spec"] = json.dumps(values["spec"])
        if "stale" in values:
            values["stale"] = 1 if values["stale"] else 0

        assignments = ", ".join(f"{column} = ?" for column in values)
        params = list(values.values()) + [utc_now_iso(), run_id]
        with self._writing() as conn:
            conn.execute(
                f"UPDATE runs SET {assignments}, updated_ts = ? WHERE run_id = ?",
                params,
            )

    def update_run_status(
        self,
        run_id: str,
        status: str,
        *,
        job_id: str | None = None,
    ) -> None:
        """Record what the scheduler says about a run.

        Status is scheduler-owned. The event log tells us what the *training*
        did; only Slurm knows whether the *job* is queued, running or dead --
        a SIGKILLed job never gets to write `run_ended` at all.
        """
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}; expected one of {sorted(STATUSES)}")
        fields: dict = {"status": status}
        if job_id is not None:
            fields["job_id"] = job_id
        self.update_run(run_id, **fields)

    def set_stale(self, run_id: str, stale: bool = True) -> None:
        """Flag or unflag a run as stale (heartbeat older than the threshold).

        Stale is a *flag*, not a status, and lives in its own column for that
        reason: a stale run is still whatever Slurm says it is, and marking it
        must never overwrite the scheduler's view or kill anything.
        """
        self.update_run(run_id, stale=1 if stale else 0)

    def get_run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _decode_run(_row(row))

    def list_runs(self, *, active_only: bool = False, limit: int | None = None) -> list[dict]:
        """Newest run first. `active_only` drops the terminal ones.

        The daemon calls this with `active_only=True` every cycle: those are
        exactly the runs worth asking Slurm about and reading bytes for.
        """
        sql = "SELECT * FROM runs"
        params: list = []
        if active_only:
            placeholders = ", ".join("?" for _ in TERMINAL)
            sql += f" WHERE status NOT IN ({placeholders})"
            params.extend(sorted(TERMINAL))
        sql += " ORDER BY created_ts DESC, run_id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [_decode_run(r) for r in _rows(self.conn.execute(sql, params))]

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def list_events(
        self,
        run_id: str,
        *,
        type: str | None = None,
        since_seq: int = 0,
        limit: int | None = None,
    ) -> list[dict]:
        """Events for a run in seq order, with the payload decoded back to a dict.

        `since_seq` is exclusive: pass the highest seq you already have and you
        get only what is new, which is how the dashboard polls cheaply.
        """
        sql = "SELECT run_id, seq, ts, type, payload FROM events WHERE run_id = ? AND seq > ?"
        params: list = [run_id, since_seq]
        if type is not None:
            sql += " AND type = ?"
            params.append(type)
        sql += " ORDER BY seq"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        out = []
        for row in self.conn.execute(sql, params):
            event = dict(row)
            event["payload"] = json.loads(event["payload"])
            out.append(event)
        return out

    def list_logs(self, run_id: str, *, since_seq: int = 0, limit: int | None = None) -> list[dict]:
        """Just the log events, for `relay logs`. A thin wrapper, on purpose.

        The store deliberately does not reach into a log payload to pull out
        `text` or `stream`: payload shape belongs to the event schema, and if it
        gains a field the store should not need editing.
        """
        return self.list_events(run_id, type="log", since_seq=since_seq, limit=limit)

    def attempt_history(self, run_id: str) -> list[dict]:
        """One row per `run_started` event, oldest first, for `relay show`.

        Each attempt of a run writes exactly one `run_started`, so this is the
        story of the run as the *job* lived it: when each attempt began and,
        under config-driven resume, which checkpoint file it picked up from.
        The `events_by_type` index answers it directly -- (run_id, type, seq) --
        so we never scan a long run's whole history to find a handful of rows.

        Every row has the same four-and-a-bit keys:

          * `attempt`     -- 0-based, straight from the payload. A v1
            `run_started` predates the field; missing means 0, for the same
            reason `ingest` reads it that way: schema v1 had no requeue-aware
            sidecar, so every one of those was a first start.
          * `seq`, `ts`   -- from the envelope.
          * `resumed_from`-- the absolute path of the checkpoint this attempt
            resumed from, or None (both "the key is absent" and "the sidecar
            wrote null" mean the same thing: it started from scratch).
          * `job_id`      -- always None for now. `run_started` does not carry
            Slurm's job ID, and rather than leave the key out we hand back a
            column the CLI's table can rely on; a future `run_started` may fill
            it in, and nothing above here will need changing when it does.

        This is a pure read. It derives nothing and writes nothing, so calling
        it twice costs two queries and changes no state.
        """
        rows = self.conn.execute(
            "SELECT seq, ts, payload FROM events"
            " WHERE run_id = ? AND type = 'run_started' ORDER BY seq",
            (run_id,),
        )

        out: list[dict] = []
        for row in rows:
            payload = json.loads(row["payload"])
            # Same guard as `ingest`: a bool is an int in Python, and a
            # negative attempt is nonsense, so anything unexpected reads as 0
            # rather than turning into a strange row in the user's table.
            attempt = payload.get("attempt")
            if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
                attempt = 0
            resumed_from = payload.get("resumed_from")
            if not isinstance(resumed_from, str) or not resumed_from:
                resumed_from = None
            out.append(
                {
                    "attempt": attempt,
                    "seq": row["seq"],
                    "ts": row["ts"],
                    "resumed_from": resumed_from,
                    "job_id": None,
                }
            )
        return out

    def max_seq(self, run_id: str) -> int:
        """Highest seq stored for a run, or 0 if we have none."""
        return _scalar(
            self.conn, "SELECT COALESCE(MAX(seq), 0) FROM events WHERE run_id = ?", (run_id,)
        )

    def count_events(self, run_id: str | None = None) -> int:
        if run_id is None:
            return _scalar(self.conn, "SELECT count(*) FROM events")
        return _scalar(self.conn, "SELECT count(*) FROM events WHERE run_id = ?", (run_id,))

    def count_metrics(self, run_id: str | None = None) -> int:
        if run_id is None:
            return _scalar(self.conn, "SELECT count(*) FROM metrics")
        return _scalar(self.conn, "SELECT count(*) FROM metrics WHERE run_id = ?", (run_id,))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metric_names(self, run_id: str) -> list[str]:
        """Every metric name this run has logged, alphabetically.

        The dashboard calls this to decide which charts to draw.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT name FROM metrics WHERE run_id = ? ORDER BY name",
            (run_id,),
        )
        return [row[0] for row in rows]

    def metric_series(
        self,
        run_id: str,
        name: str,
        *,
        since_step: int | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Points for one metric in step order: `[{"step": .., "value": .., "ts": ..}]`.

        `since_step` is exclusive (`step > since_step`) so the browser can send
        the last step it has and receive only new points. It defaults to None
        rather than 0 because step 0 is a real step and must not be silently
        dropped from a first fetch.
        """
        sql = "SELECT step, value, ts FROM metrics WHERE run_id = ? AND name = ?"
        params: list = [run_id, name]
        if since_step is not None:
            sql += " AND step > ?"
            params.append(since_step)
        sql += " ORDER BY step"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return _rows(self.conn.execute(sql, params))

    def latest_metrics(self, run_id: str) -> dict:
        """The most recent value of each metric, for the `relay ls` summary line."""
        rows = self.conn.execute(
            "SELECT name, value FROM metrics m WHERE run_id = ?"
            " AND step = (SELECT MAX(step) FROM metrics WHERE run_id = m.run_id AND name = m.name)",
            (run_id,),
        )
        return {row["name"]: row["value"] for row in rows}

    # ------------------------------------------------------------------
    # Sources (byte offsets)
    # ------------------------------------------------------------------

    def get_offset(self, run_id: str, path: str) -> int:
        """How far we have read this file. 0 when we have never read it.

        Returning 0 rather than None means the daemon's first read of a brand
        new file needs no special case: it just reads from the offset it is
        given, which happens to be the start.
        """
        row = self.conn.execute(
            "SELECT byte_offset FROM sources WHERE run_id = ? AND path = ?",
            (run_id, path),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def list_sources(self, run_id: str) -> list[dict]:
        return _rows(
            self.conn.execute(
                "SELECT run_id, path, byte_offset, updated_ts FROM sources"
                " WHERE run_id = ? ORDER BY path",
                (run_id,),
            )
        )

    # ------------------------------------------------------------------
    # Usage: writing what the scheduler charged us
    # ------------------------------------------------------------------

    def upsert_usage(self, rows: list[dict]) -> int:
        """Write one row per scheduler attempt. Returns how many rows we wrote.

        Each dict has the fields of `backends.base.UsageRow` -- job_id, state,
        partition, start, end, elapsed_s, gpus, cpus -- plus `run_id`, which the
        daemon fills in by mapping the job ID back to a run. Plain dicts rather
        than the dataclass on purpose: the store does not import a backend, and
        a backend does not know the database exists.

        **This is the one exception to relay's `INSERT OR IGNORE` rule.** A
        running attempt's `elapsed_s` grows while we watch it (and its `end` and
        `state` are filled in when it finishes), so ignoring a row we already
        have would freeze every attempt at the first numbers we happened to see.
        It is still idempotent, since re-fetching the same sacct output produces
        the same rows: the second write of an unchanged attempt leaves the table
        byte-for-byte identical. Idempotence is what we actually need here, and
        "never overwrite" was only ever a way of getting it.
        """
        if not rows:
            return 0

        values = [
            (
                str(row["job_id"]),
                str(row["start"]),
                str(row["run_id"]),
                str(row["state"]),
                row.get("partition"),
                row.get("end"),
                int(row.get("elapsed_s") or 0),
                int(row.get("gpus") or 0),
                int(row.get("cpus") or 0),
            )
            for row in rows
        ]

        assignments = ", ".join(f"{column} = excluded.{column}" for column in _USAGE_UPDATED_COLUMNS)
        with self._writing() as conn:
            # One transaction for the whole batch: a usage fetch is a single
            # sacct call and either all of its rows land or none do.
            conn.executemany(
                'INSERT INTO usage (job_id, start, run_id, state, "partition", "end",'
                "                   elapsed_s, gpus, cpus)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                f" ON CONFLICT(job_id, start) DO UPDATE SET {assignments}",
                values,
            )
        return len(values)

    def mark_usage_final(self, run_id: str) -> None:
        """Record that we have fetched this run's usage for the last time.

        A finished job's sacct numbers never change again, so the daemon fetches
        once more after a run goes terminal and then sets this. It is the flag
        that stops relay asking the cluster about a run that ended in March.
        """
        with self._writing() as conn:
            conn.execute(
                "UPDATE runs SET usage_final = 1, updated_ts = ? WHERE run_id = ?",
                (utc_now_iso(), run_id),
            )

    def runs_needing_usage(self) -> list[dict]:
        """The runs worth asking the scheduler about, for one batched call.

        Two groups, and both need a job ID (a run that never made it out of
        `submit` has nothing to look up):

          * non-terminal runs -- their numbers are still moving;
          * terminal runs we have not finalised -- the one last fetch.
        """
        placeholders = ", ".join("?" for _ in TERMINAL)
        sql = (
            "SELECT * FROM runs"
            " WHERE job_id IS NOT NULL AND job_id <> ''"
            f"   AND (status NOT IN ({placeholders}) OR usage_final = 0)"
            " ORDER BY created_ts DESC, run_id DESC"
        )
        return [_decode_run(r) for r in _rows(self.conn.execute(sql, sorted(TERMINAL)))]

    def list_usage(self, run_id: str) -> list[dict]:
        """Every attempt of one run, oldest first. For `relay show`."""
        return _rows(
            self.conn.execute(
                'SELECT job_id, run_id, state, "partition", start, "end", elapsed_s, gpus, cpus'
                " FROM usage WHERE run_id = ? ORDER BY start",
                (run_id,),
            )
        )

    # ------------------------------------------------------------------
    # Usage: the derived numbers
    # ------------------------------------------------------------------
    #
    # None of this is stored. GPU-hours are `elapsed_s * gpus / 3600` and that
    # arithmetic is cheap, while a stored total is a second copy of the truth
    # that can disagree with the rows it came from -- and would have to be
    # recomputed anyway every time an attempt's elapsed_s grew.

    def usage_by(
        self,
        by: str = "run",
        *,
        since: str | None = None,
        group: str | None = None,
    ) -> list[dict]:
        """GPU- and CPU-hours grouped by run, account, or partition.

        `by` is "run", "group" (the run's Slurm account) or "partition".
        `since` keeps only attempts that *started* on or after that date.
        `group` narrows to one account, whatever you are grouping by.

        Rows look like {"key", "gpu_hours", "cpu_hours", "attempts", "runs"},
        biggest first. "attempts" counts scheduler attempts and "runs" counts
        distinct runs, which is how you see that 400 GPU-hours came from three
        runs and nineteen preemptions.
        """
        keys = {
            "run": "u.run_id",
            "group": _ACCOUNT_SQL,
            "partition": "COALESCE(u.\"partition\", '(none)')",
        }
        if by not in keys:
            raise ValueError(f"unknown grouping {by!r}; expected one of {sorted(keys)}")
        key_sql = keys[by]

        where, params = self._usage_filters(since=since, group=group)
        sql = (
            f"SELECT {key_sql} AS key,"
            f"       {_GPU_HOURS_SQL} AS gpu_hours,"
            f"       {_CPU_HOURS_SQL} AS cpu_hours,"
            "        COUNT(*) AS attempts,"
            "        COUNT(DISTINCT u.run_id) AS runs"
            " FROM usage u LEFT JOIN runs r ON r.run_id = u.run_id"
            f"{where}"
            f" GROUP BY {key_sql}"
            " ORDER BY gpu_hours DESC, cpu_hours DESC, key"
        )
        # LEFT JOIN, not JOIN: usage rows for a run whose `runs` row has been
        # deleted are still hours somebody spent, and dropping them would make
        # the per-run report and the total disagree.
        return _rows(self.conn.execute(sql, params))

    def usage_totals(self, *, since: str | None = None, group: str | None = None) -> dict:
        """One overall total, plus the split by partition.

        The partition split is the point of the whole feature on Klone: it is
        how the user sees how much of their work ran for free on preemptible
        `ckpt*` nodes versus how much it charged their group's allocation.
        """
        where, params = self._usage_filters(since=since, group=group)
        row = self.conn.execute(
            f"SELECT {_GPU_HOURS_SQL} AS gpu_hours,"
            f"       {_CPU_HOURS_SQL} AS cpu_hours,"
            "        COUNT(*) AS attempts,"
            "        COUNT(DISTINCT u.run_id) AS runs"
            " FROM usage u LEFT JOIN runs r ON r.run_id = u.run_id"
            f"{where}",
            params,
        ).fetchone()

        totals = dict(row)
        totals["by_partition"] = [
            {
                "partition": part["key"],
                "gpu_hours": part["gpu_hours"],
                "cpu_hours": part["cpu_hours"],
            }
            for part in self.usage_by("partition", since=since, group=group)
        ]
        return totals

    def wasted_hours(self, *, since: str | None = None, group: str | None = None) -> dict:
        """Hours that bought nothing, in the two ways that actually happen.

        `failed`: every attempt of every run whose final status is `failed`.
        All of it is wasted -- a crashed run produces no usable model, and the
        attempts before the crash were spent getting there.

        `lost_to_preemption`: for each interrupted attempt (the scheduler says
        PREEMPTED or REQUEUED), the time between its last `checkpoint` event and
        the moment it died. Everything before that checkpoint survives on disk
        and is resumed from, so only the tail is lost. An attempt that reported
        no checkpoint at all loses the whole thing and is marked
        `no_checkpoints`, which is usually a sign the user's script is not
        printing checkpoints rather than that it never saves one.

        Attempts that are still running are skipped: an attempt with no `end`
        has not lost anything yet.
        """
        where, params = self._usage_filters(since=since, group=group)

        # An aggregate with no GROUP BY always returns exactly one row, even
        # when nothing matched -- zeros and a NULL, which is what we want.
        failed_where = _and(where, "r.status = 'failed'")
        failed = dict(
            self.conn.execute(
                "SELECT " + _GPU_HOURS_SQL + " AS gpu_hours,"
                "       " + _CPU_HOURS_SQL + " AS cpu_hours,"
                "       GROUP_CONCAT(DISTINCT u.run_id) AS run_ids"
                " FROM usage u JOIN runs r ON r.run_id = u.run_id" + failed_where,
                params,
            ).fetchone()
        )
        # GROUP_CONCAT hands back "a,b,c", or NULL when there were no rows.
        run_ids = sorted(failed["run_ids"].split(",")) if failed["run_ids"] else []

        # The time arithmetic is done in SQL with julianday() rather than in
        # Python. julianday parses every shape a timestamp reaches us in -- with
        # a `Z`, without one, date-only -- and turns it into a number of days,
        # so comparing an event's `ts` against an attempt's window is exact
        # instead of a string comparison that would trip over ".000Z" being
        # longer than the same instant written without milliseconds.
        # (`* 86400` converts days back to seconds.)
        interrupted = (
            'u."end" IS NOT NULL'
            " AND UPPER(u.state) IN (" + _placeholders(INTERRUPTED_STATES) + ")"
        )
        attempt_sql = (
            'SELECT u.run_id, u.job_id, u.start, u."end", u.elapsed_s, u.gpus, u.cpus,'
            "       (SELECT MAX(julianday(e.ts)) FROM events e"
            "         WHERE e.run_id = u.run_id AND e.type = 'checkpoint'"
            "           AND julianday(e.ts) >= julianday(u.start)"
            '           AND julianday(e.ts) <= julianday(u."end")) AS last_ckpt_jd'
            " FROM usage u LEFT JOIN runs r ON r.run_id = u.run_id"
            + _and(where, interrupted)
            + " ORDER BY u.start, u.job_id"
        )

        attempts: list[dict] = []
        lost_gpu_s = 0.0
        lost_cpu_s = 0.0
        for row in self.conn.execute(attempt_sql, params + list(INTERRUPTED_STATES)):
            if row["last_ckpt_jd"] is None:
                lost_s = float(row["elapsed_s"])
                no_checkpoints = True
            else:
                end_jd = self.conn.execute("SELECT julianday(?)", (row["end"],)).fetchone()[0]
                lost_s = (end_jd - row["last_ckpt_jd"]) * 86400.0
                # Clamped: a clock skew between the compute node writing events
                # and the scheduler recording the end time must not produce a
                # negative "waste" that quietly cancels out someone else's.
                lost_s = max(0.0, min(lost_s, float(row["elapsed_s"])))
                no_checkpoints = False

            lost_gpu_s += lost_s * row["gpus"]
            lost_cpu_s += lost_s * row["cpus"]
            attempts.append(
                {
                    "run_id": row["run_id"],
                    "job_id": row["job_id"],
                    "start": row["start"],
                    "end": row["end"],
                    "lost_s": round(lost_s),
                    "gpus": row["gpus"],
                    "no_checkpoints": no_checkpoints,
                }
            )

        return {
            "failed": {
                "gpu_hours": failed["gpu_hours"],
                "cpu_hours": failed["cpu_hours"],
                "runs": run_ids,
            },
            "lost_to_preemption": {
                "gpu_hours": lost_gpu_s / 3600.0,
                "cpu_hours": lost_cpu_s / 3600.0,
                "attempts": attempts,
            },
        }

    def _usage_filters(self, *, since: str | None, group: str | None) -> tuple[str, list]:
        """Build the shared WHERE clause for every usage query.

        One helper so "since" and "group" cannot come to mean slightly different
        things in different reports. Returns the clause (possibly empty) and its
        parameters, in order.
        """
        clauses: list[str] = []
        params: list = []
        if since is not None:
            # Compared with julianday so that a plain date ("2026-09-01") and a
            # full timestamp both work and neither has to match the stored
            # format character for character.
            if self.conn.execute("SELECT julianday(?)", (since,)).fetchone()[0] is None:
                raise ValueError(f"could not read {since!r} as a date or timestamp")
            clauses.append("julianday(u.start) >= julianday(?)")
            params.append(since)
        if group is not None:
            clauses.append(f"{_ACCOUNT_SQL} = ?")
            params.append(group)
        if not clauses:
            return "", params
        return " WHERE " + " AND ".join(clauses), params

    # ------------------------------------------------------------------
    # Daemon state
    # ------------------------------------------------------------------

    def set_daemon_state(self, **fields) -> None:
        """Set any subset of the daemon's single state row.

        Only the fields you pass are touched, so recording a transport error
        cannot accidentally blank out the last successful usage fetch.
        """
        unknown = set(fields) - set(_DAEMON_STATE_COLUMNS)
        if unknown:
            raise ValueError(f"unknown daemon_state column(s): {sorted(unknown)}")
        if not fields:
            return

        columns = list(fields)
        assignments = ", ".join(f"{column} = excluded.{column}" for column in columns)
        with self._writing() as conn:
            conn.execute(
                f"INSERT INTO daemon_state (id, {', '.join(columns)})"
                f" VALUES (1, {', '.join('?' for _ in columns)})"
                f" ON CONFLICT(id) DO UPDATE SET {assignments}",
                [fields[column] for column in columns],
            )

    def get_daemon_state(self) -> dict:
        """The daemon's state row, with every field None if it never wrote one.

        Always the same keys, so callers can read `state["last_error_kind"]`
        without first checking whether the daemon has ever run.
        """
        row = self.conn.execute(
            f"SELECT {', '.join(_DAEMON_STATE_COLUMNS)} FROM daemon_state WHERE id = 1"
        ).fetchone()
        if row is None:
            return {column: None for column in _DAEMON_STATE_COLUMNS}
        return dict(row)

    def clear_daemon_error(self) -> None:
        """Forget the last transport error, after a cycle that worked."""
        self.set_daemon_state(
            last_error_kind=None,
            last_error_ts=None,
            last_error_detail=None,
        )

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else str(row[0])

    def meta_set(self, key: str, value: str) -> None:
        """Upsert, not INSERT OR IGNORE: meta values are meant to be overwritten."""
        with self._writing() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def mark_synced(self, ts: str | None = None) -> str:
        """Record that the daemon just finished a cycle; returns the timestamp.

        `relay ls` prints "metrics Ns ago" from this. Showing stale data is
        acceptable; showing stale data as though it were current is not.

        This is the *filesystem* side of the daemon's work -- the event logs it
        tailed. What the scheduler said is a separate question on a separate
        schedule; see `mark_scheduler_synced`.
        """
        stamp = ts or utc_now_iso()
        self.meta_set(META_LAST_SYNC_TS, stamp)
        return stamp

    def last_synced(self) -> str | None:
        return self.meta_get(META_LAST_SYNC_TS)

    def mark_scheduler_synced(self, ts: str | None = None) -> str:
        """Record a successful scheduler poll; returns the timestamp.

        Separate from `mark_synced` because the two run on separate schedules:
        the tail every couple of seconds, the scheduler poll at thirty seconds
        and backing off from there. A user looking at `relay ls` needs to know
        which of the two numbers on screen is the old one.
        """
        stamp = ts or utc_now_iso()
        self.meta_set(META_LAST_SCHEDULER_TS, stamp)
        return stamp

    def last_scheduler_synced(self) -> str | None:
        return self.meta_get(META_LAST_SCHEDULER_TS)

    def poke_scheduler(self, ts: str | None = None) -> str:
        """Tell the daemon to poll the scheduler promptly; returns the timestamp.

        Called by `submit` and `cancel`. Those are the two moments when the
        user has just done something whose result they are about to watch for,
        and they are also exactly the moments when the daemon's backoff is
        likely to be at its most relaxed -- nothing had changed for a while,
        which is why it backed off in the first place.

        A timestamp rather than a boolean flag, so the daemon can tell a poke
        it has already acted on from a new one without having to clear it.
        Clearing would need a write from the daemon on every cycle, and two
        processes writing one flag is a race for no benefit.

        The sophisticated alternative is a real notification -- a socket, a
        pipe, `inotify` on the database file -- which would wake the daemon
        instantly instead of at its next cycle. It is not worth it: the daemon's
        next cycle is at most sixty seconds away even when fully idle, and a
        socket is another thing to bind, clean up and get wrong on a laptop
        that sleeps.
        """
        stamp = ts or utc_now_iso()
        self.meta_set(META_SCHEDULER_POKE_TS, stamp)
        return stamp

    def scheduler_poke(self) -> str | None:
        """The last poke timestamp, or None if nobody has ever poked."""
        return self.meta_get(META_SCHEDULER_POKE_TS)


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    """Run a query that returns exactly one number and hand back that number."""
    return int(conn.execute(sql, params).fetchone()[0])


def _and(where: str, condition: str) -> str:
    """Bolt one more condition onto a WHERE clause that may be empty.

    `_usage_filters` returns either "" or " WHERE ...", so every caller that
    wants to add a condition of its own would otherwise need the same two-line
    `if`. One helper, one place to get it wrong.
    """
    return (where + " AND " + condition) if where else (" WHERE " + condition)


def _placeholders(values: Iterable) -> str:
    """"?, ?, ?" for an IN clause. Values are bound, never interpolated."""
    return ", ".join("?" for _ in values)


def _decode_run(run: dict | None) -> dict | None:
    """Make a run row friendly to callers: JSON spec decoded, stale a real bool."""
    if run is None:
        return None
    if run.get("spec"):
        try:
            run["spec"] = json.loads(run["spec"])
        except (TypeError, ValueError):
            # A spec we cannot parse is not worth crashing `relay ls` over;
            # hand back the raw text and let the caller notice.
            pass
    run["stale"] = bool(run.get("stale"))
    # Stored as 0/1 because SQLite has no boolean type; handed out as a real
    # bool so `--json` output says true/false rather than 1/0.
    run["usage_final"] = bool(run.get("usage_final"))
    run["terminal"] = run.get("status") in TERMINAL
    return run
