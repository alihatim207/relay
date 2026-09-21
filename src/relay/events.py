"""The event log format: relay's only contract between the cluster and the laptop.

A job running on a compute node cannot be reached over the network, so the only
way it can tell us anything is by appending lines to a file on shared storage.
That file is `events.jsonl`: one JSON object per line, never rewritten, only
appended to.

This module owns three things:

  * the shape of an event (`build_event`)
  * how to read one back safely (`parse_event`)
  * how to write them durably (`EventWriter`)

plus the `##relay##` stdout convention that lets a training script report
metrics without importing anything.

Stdlib only. `sidecar.py` is shipped to the cluster as a single self-contained
file and therefore cannot import this module, but keeping the two in agreement
is much easier if neither of them grows dependencies.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Schema constants
# --------------------------------------------------------------------------

# Bump this only when the envelope itself changes shape. Payloads are allowed to
# grow new keys without a version bump; readers ignore what they do not know.
#
# v2 did not change the envelope at all -- it changed what `run_ended.payload`
# is allowed to say (see RUN_ENDED_STATUSES) and added `attempt` to
# `run_started.payload`. Strictly, by the rule above, neither needed a bump.
# It was bumped anyway so that a laptop reading a log can tell "this run was
# written by a relay that knew the difference between `requeued` and
# `interrupted`" from "this run was written by one that called everything
# `requeued`". That distinction is not recoverable from the payload itself.
SCHEMA_VERSION = 2

# Every envelope version this build can read. New writers stamp SCHEMA_VERSION;
# old files stay readable, which matters because `events.jsonl` is append-only
# and a single file can contain both -- a run that started under v1 and was
# requeued after an upgrade has v1 lines followed by v2 lines.
#
# This is deliberately a whitelist rather than `v <= SCHEMA_VERSION`. A version
# we have never heard of could use a different envelope shape entirely, and
# guessing is worse than refusing.
ACCEPTED_VERSIONS = frozenset({1, 2})

# What `run_ended.payload["status"]` may be:
#
#   completed    the child exited 0
#   failed       the child exited nonzero (or never started)
#   requeued     SIGUSR1: the time limit was approaching and the sidecar put
#                the job back in the queue itself
#   interrupted  SIGTERM: preempted or cancelled -- the sidecar cannot tell
#                which, because they are the same signal. The daemon resolves
#                it from the scheduler, under the standing rule that the log is
#                authoritative about the training and the scheduler about the
#                job.
#
# Not enforced by `build_event`: the payload is the type-specific half of the
# envelope and this module validates the envelope. It is here so the sidecar,
# the store and the dashboard have one list to agree with.
RUN_ENDED_STATUSES = ("completed", "failed", "requeued", "interrupted")

# The seven things a run can say. Anything else is a bug or a corrupt line.
EVENT_TYPES = frozenset(
    {
        "run_started",
        "metric",
        "log",
        "heartbeat",
        "checkpoint",
        "signal",
        "run_ended",
    }
)

# The envelope is a fixed set of keys in a fixed order. Order matters only for
# human readability of the file (and for stable test comparisons) -- json.dumps
# preserves dict insertion order, so building the dict in this order is enough.
ENVELOPE_KEYS = ("v", "run", "seq", "ts", "type", "payload")

# The marker a training script prints to report a metric:
#   print('##relay## {"step": 1000, "loss": 0.412}', flush=True)
# The trailing space is part of the marker, so a line that merely starts with
# "##relay##something" is treated as an ordinary log line rather than a
# malformed metric.
METRIC_MARKER = "##relay## "


class EventParseError(ValueError):
    """Raised when a line cannot be read as a valid event.

    The contract for callers: every function in this module that parses an
    *event* raises exactly this (never json's own errors, never KeyError,
    never TypeError). The daemon wraps each line in a try/except for this one
    exception type and drops the bad line, so a single corrupt line -- a
    half-written record, a stray line from a user's `echo` -- can never take
    down ingestion of the whole file.

    `parse_metric_line` is deliberately different: it returns None instead of
    raising, because "this stdout line is not a metric" is the common case, not
    an error.
    """


def _type_name(value: object) -> str:
    """Short name of a value's type, for readable error messages.

    Lives at module level so it can still reach the `type` builtin -- inside
    `build_event` the parameter named `type` shadows it.
    """
    return type(value).__name__


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Current time as ISO-8601 UTC with millisecond precision and a `Z`.

    e.g. "2026-09-16T18:22:09.003Z"

    Python writes UTC offsets as "+00:00"; the schema wants the shorter "Z"
    form, so we swap it. Millisecond precision is plenty for ordering events
    from a training loop, and `seq` -- not `ts` -- is what actually orders them.
    """
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Building and serializing
# --------------------------------------------------------------------------


def build_event(
    run: str,
    seq: int,
    type: str,
    payload: dict | None = None,
    ts: str | None = None,
) -> dict:
    """Build one event dict.

    `run` is relay's own run ID, never Slurm's job ID -- a requeued job gets a
    new Slurm ID but stays the same relay run.

    `seq` is a per-run monotonic counter that continues across requeue attempts.
    Normally you do not pass it yourself; `EventWriter` assigns it.

    `ts` is stamped for you unless you pass one (tests pass one so they can
    compare exact bytes).

    Raises EventParseError for an unknown type or a non-positive seq, so a
    programming mistake fails here rather than writing a bad line to the log
    that some later reader has to cope with.
    """
    if type not in EVENT_TYPES:
        raise EventParseError(f"unknown event type: {type!r}")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        # bool is a subclass of int in Python, so True would otherwise sneak
        # through as seq == 1.
        raise EventParseError(f"seq must be a positive int, got {seq!r}")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise EventParseError(f"payload must be a dict, got {_type_name(payload)}")

    # Built in ENVELOPE_KEYS order on purpose; see the comment on that constant.
    return {
        "v": SCHEMA_VERSION,
        "run": run,
        "seq": seq,
        "ts": ts if ts is not None else utc_now_iso(),
        "type": type,
        "payload": payload,
    }


def serialize(event: dict) -> str:
    """Serialize one event to a single compact JSON line (no trailing newline).

    Compact separators keep the file small; over a long run this is millions of
    lines. `sort_keys` is deliberately NOT used -- we want the envelope order
    from `build_event`, not alphabetical order.
    """
    return json.dumps(event, separators=(",", ":"))


# --------------------------------------------------------------------------
# Parsing events back
# --------------------------------------------------------------------------


def parse_event(line: str) -> dict:
    """Parse one line of `events.jsonl` into an event dict.

    Takes `str`, not `bytes`. The daemon reads raw bytes off the wire and is
    responsible for splitting on newlines and decoding; doing the decode there
    means the "did this chunk end mid-line?" logic and the "is this UTF-8?"
    logic live in one place. Callers should decode with errors="replace" so a
    garbled byte becomes a bad line rather than an exception in the middle of a
    batch.

    Raises EventParseError on anything wrong: bad JSON, not an object, a
    missing envelope key, an unknown type, a non-positive seq, a payload that
    is not an object, or a schema version we do not understand.
    """
    if not isinstance(line, str):
        raise EventParseError(f"expected str, got {_type_name(line)}")

    text = line.strip()
    if not text:
        raise EventParseError("empty line")

    try:
        obj = json.loads(text)
    except ValueError as exc:
        # json.JSONDecodeError is a subclass of ValueError. We re-raise as our
        # own type so callers only have to catch one thing.
        raise EventParseError(f"invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise EventParseError(f"expected a JSON object, got {_type_name(obj)}")

    missing = [key for key in ENVELOPE_KEYS if key not in obj]
    if missing:
        raise EventParseError(f"missing envelope keys: {', '.join(missing)}")

    version = obj["v"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in ACCEPTED_VERSIONS
    ):
        # A version we have never heard of could use a different envelope shape
        # entirely, so guessing would be worse than refusing. The daemon records
        # the bad line and moves on; the fix is to upgrade relay on the laptop.
        #
        # The isinstance guards are not pedantry. `in` on a set compares by
        # equality, and in Python `True == 1` and `2.0 == 2`, so `{"v": true}`
        # and `{"v": 2.0}` would both slip through a bare membership test. A
        # writer that cannot get the type of the version right is not one whose
        # envelope we should trust.
        raise EventParseError(f"unsupported schema version: {version!r}")

    if not isinstance(obj["run"], str) or not obj["run"]:
        raise EventParseError("run must be a non-empty string")

    seq = obj["seq"]
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
        raise EventParseError(f"seq must be a positive int, got {seq!r}")

    if not isinstance(obj["ts"], str) or not obj["ts"]:
        raise EventParseError("ts must be a non-empty string")

    if obj["type"] not in EVENT_TYPES:
        raise EventParseError(f"unknown event type: {obj['type']!r}")

    if not isinstance(obj["payload"], dict):
        raise EventParseError(f"payload must be an object, got {_type_name(obj['payload'])}")

    return obj


# --------------------------------------------------------------------------
# The ##relay## stdout convention
# --------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """True for ints and floats, False for bools.

    `isinstance(True, int)` is True in Python, so a bare isinstance check would
    happily store `"converged": true` as the metric value 1.0. A bool is a flag,
    not a measurement, so it is excluded.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_metric_line(line: str) -> dict | None:
    """Turn a `##relay##` stdout line into a metric payload, or return None.

    Input is one raw line of the training script's stdout, e.g.

        ##relay## {"step": 1000, "loss": 0.412, "sps": 18400}

    Output is the payload shape the schema requires -- numbers nested under
    `values` so the envelope stays a fixed, parseable shape no matter what the
    user logs:

        {"step": 1000, "values": {"loss": 0.412, "sps": 18400}}

    Returns None (never raises) when the line is not a metric line at all, or
    when it is malformed: missing marker, bad JSON, not an object, no `step`, or
    a non-numeric `step`. The caller -- the sidecar -- treats None as "this is
    an ordinary log line" and emits a `log` event instead. A typo in a user's
    print statement should cost them a metric, not crash their training job.

    Non-numeric values other than `step` are skipped rather than rejected, so
    `{"step": 10, "loss": 0.4, "note": "warmup"}` still yields the loss. One bad
    key should not throw away the whole measurement. NaN and
    infinity survive json.loads as floats and are kept; deciding what to do with
    them belongs to the charting layer, not here.
    """
    if not isinstance(line, str):
        return None

    # rstrip only: leading whitespace would mean the marker is not at the start
    # of the line, and we require it to be.
    text = line.rstrip("\r\n")
    if not text.startswith(METRIC_MARKER):
        return None

    remainder = text[len(METRIC_MARKER) :].strip()
    try:
        obj = json.loads(remainder)
    except ValueError:
        return None

    if not isinstance(obj, dict):
        return None

    step = obj.get("step")
    if not _is_number(step):
        return None
    # A step is a count of training iterations, so it is an integer even if the
    # user printed 1000.0.
    step = int(step)

    # Numbers are kept exactly as the user printed them -- an int stays an int,
    # so the spec's example line round-trips as `"sps":18400` rather than
    # `"sps":18400.0`. The store widens everything to REAL later anyway.
    values = {
        name: value
        for name, value in obj.items()
        if name != "step" and _is_number(value)
    }

    return {"step": step, "values": values}


# --------------------------------------------------------------------------
# The writer
# --------------------------------------------------------------------------


def highest_seq(path: str) -> int:
    """Highest `seq` already present in an event file, or 0 if there is none.

    Reads the whole file line by line. That is the simple choice, and it is
    fine: a long run's event log is tens of megabytes and this runs once, at
    process start. The sophisticated alternative is to seek to the end and scan
    backwards for the last newline, but the last line is not guaranteed to hold
    the highest seq if a previous process was SIGKILLed mid-write, and "read it
    all, take the max" has no edge cases.

    Unparseable lines are skipped rather than fatal. A truncated final line is
    exactly what a SIGKILL leaves behind, and refusing to start because of it
    would turn a recoverable job into a dead one.

    Missing file returns 0, since a first attempt has written nothing yet.
    """
    best = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = parse_event(line)
                except EventParseError:
                    continue
                if event["seq"] > best:
                    best = event["seq"]
    except FileNotFoundError:
        return 0
    return best


class EventWriter:
    """Appends events to `events.jsonl`, assigning `seq` as it goes.

    Durability is the whole point of this class, so it deliberately does not use
    Python's buffered file objects:

      * `os.open` with O_APPEND means every write lands at the current end of
        file, atomically with respect to other appenders. Two processes (a
        previous attempt winding down, a requeued attempt starting) cannot
        interleave halfway through each other's line.
      * one `os.write` per complete line, including the newline, means a reader
        never sees half a record unless the kernel truncated the write itself.
      * no buffering at all. An event sitting in a userspace buffer when SIGKILL
        arrives never existed, and SIGKILL is the normal end of a preempted job.

    Use `EventWriter.open(path, run)` rather than the constructor: it continues
    `seq` from whatever is already in the file. A requeued attempt that restarted
    seq at 1 would write records whose (run_id, seq) primary key collides with
    the previous attempt's, and `INSERT OR IGNORE` in the store would silently
    drop them.
    """

    def __init__(self, path: str, run: str, start_seq: int = 0) -> None:
        self.path = path
        self.run = run
        # The seq of the last event written. The next event gets start_seq + 1.
        self.seq = start_seq
        # O_APPEND: always write at end of file. O_CREAT: make it if absent.
        # O_WRONLY: we never read through this descriptor. 0o644 so a collaborator
        # on the shared filesystem can read the log.
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    @classmethod
    def open(cls, path: str, run: str) -> "EventWriter":
        """Open an event file, continuing `seq` from any existing content."""
        return cls(path, run, start_seq=highest_seq(path))

    def emit(self, type: str, payload: dict | None = None) -> dict:
        """Append one event and return it (handy for tests and for logging).

        The returned dict is the exact object that was serialized, so a caller
        can read back the assigned `seq` and `ts` without reopening the file.
        """
        if self._fd is None:
            raise ValueError("writer is closed")
        self.seq += 1
        event = build_event(self.run, self.seq, type, payload)
        line = serialize(event).encode("utf-8") + b"\n"
        # One write call for the whole line. os.write can in principle write
        # fewer bytes than asked, so loop until it is all out -- on a regular
        # file this loop body runs once, but a short write would otherwise
        # corrupt the log silently.
        while line:
            written = os.write(self._fd, line)
            line = line[written:]
        return event

    def close(self) -> None:
        """Close the underlying descriptor. Safe to call more than once."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # Context-manager support, so `with EventWriter.open(...) as w:` works.
    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
