"""relay's sidecar: the only thing that can talk back from a compute node.

A compute node has no inbound network path. Nothing on your laptop can connect
to it. The one channel out is the shared filesystem, so this process wraps the
training command, watches its stdout and stderr, and appends one JSON object
per line to `events.jsonl` in the run directory. The daemon on the laptop tails
that file over SSH. That is the whole architecture.

Invocation (fixed -- the backends depend on this exact shape):

    python3 sidecar.py --run <run_id> --dir <run_dir> --grace <seconds> \
        -- <command and args...>

It is also importable as `relay.sidecar` and runnable as
`python3 -m relay.sidecar`, but on the cluster it is a lone copied file sitting
next to the sbatch script. That is why this module imports **only** the standard
library, and why it does not import `relay.events` even though the two agree on
the schema. A test AST-parses the imports below to keep it that way.

What it does, in order:

  0. read `sidecar.json` once, and -- on a requeued attempt only -- find the
     newest checkpoint the previous attempt left behind and append it to the
     command, so the training script picks up where it stopped
  1. emit `run_started`
  2. spawn the child with stdout and stderr piped, with RELAY_RUN_ID,
     RELAY_RUN_DIR and the run directory on PYTHONPATH in its environment
  3. one reader thread per pipe (a single thread reading one pipe deadlocks
     when the other pipe's buffer fills and the child blocks writing to it)
  4. a heartbeat thread, so the daemon can tell "still alive" from "hung"
  5. wait for the child, emit `run_ended`, exit with the child's code

...unless a signal arrives first, which on a preemptible partition is the
*normal* ending. Two signals, two different situations, and conflating them is
a bug that requeues jobs the user just cancelled:

  SIGUSR1  The time limit is approaching. Slurm sends this because the sbatch
           script asked for `--signal=B:USR1@<grace>`. Nothing else sends it,
           so self-requeue is always the right answer: the child gets
           `grace - 2` seconds to checkpoint, the run ends `requeued`, and the
           sidecar runs `scontrol requeue` itself.

  SIGTERM  Preemption *or* `scancel` -- the same signal, indistinguishable at
           signal time. The child gets `preempt_grace - 2` seconds, the run
           ends `interrupted`, and whether to requeue is decided afterwards
           from two facts that were settled before the signal arrived: a
           `CANCEL_REQUESTED` file that `relay cancel` drops in the run
           directory, and the `requeue_on_term` setting in `sidecar.json`.

`preempt_grace` is not ours to choose. It is the partition's `GraceTime` or the
cluster's `KillWait`, measured at 10 seconds on Klone, and the SIGKILL arrives
whether or not we are finished. So the rule on the SIGTERM path is: emit the
`signal` event, forward SIGTERM to the child, and do nothing else first. No
subprocess calls, no file reads, no scans. `sidecar.json` is read once at
startup, long before any of this, for exactly that reason.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import NamedTuple

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Envelope version. Must match events.py (which also accepts 1, so a log that
# straddles an upgrade still parses).
SCHEMA_VERSION = 2

# The marker a training script prints to report a metric:
#   print('##relay## {"step": 1000, "loss": 0.412}', flush=True)
# The trailing space is part of the marker, so "##relay##nonsense" is just a log
# line rather than a malformed metric.
METRIC_MARKER = "##relay## "

# Seconds between heartbeats. A module constant (and a --heartbeat flag) so the
# tests can shrink it instead of sleeping for half a minute.
HEARTBEAT_INTERVAL = 30.0

# Default seconds of warning before the *time limit*, matching
# `--signal=B:USR1@<grace>` in the sbatch script.
DEFAULT_GRACE = 60.0

# Default seconds between the SIGTERM a preempted or cancelled job gets and the
# SIGKILL that follows. Unlike `grace`, this window is not ours: it belongs to
# the partition (GraceTime) or the cluster (KillWait). 10 is Klone's measured
# value. `sidecar.json` overrides it; see `load_sidecar_config`.
DEFAULT_PREEMPT_GRACE = 10.0

# How much of a signal's window we keep for ourselves. The child gets
# `window - SIGNAL_MARGIN` seconds to checkpoint and exit; the remainder is our
# budget to SIGKILL it and write `run_ended` before the scheduler's SIGKILL
# lands on us too. Two seconds, because what happens in that margin is two
# small unbuffered `os.write` calls and a `waitpid` -- milliseconds of work.
# Taking more would come straight out of the child's checkpoint time, and on
# Klone the whole preemption window is only ten seconds.
SIGNAL_MARGIN = 2.0

# The child never gets less than this, however small the window is. A
# preempt_grace of 1 would otherwise mean "SIGTERM and SIGKILL in the same
# breath", which is strictly worse than being a second late.
MIN_CHILD_WINDOW = 1.0

# Upper bound on the two things we do *after* the child is dead: reaping it
# after a SIGKILL, and the `scontrol requeue` call. Both are supposed to be
# instant; the timeout only exists so neither can wedge us until the scheduler
# kills us mid-write.
POST_KILL_TIMEOUT = 5.0

# How long to wait for the reader threads to drain the pipes after the child
# exits. They normally finish instantly (the pipes hit EOF when the child dies);
# the timeout only exists so a grandchild holding the pipe open cannot wedge us.
READER_JOIN_TIMEOUT = 5.0

EVENTS_FILENAME = "events.jsonl"

# Written into the run directory by the backend at submit time, read by us once
# at startup. See `load_sidecar_config` for the shape and the defaults.
CONFIG_FILENAME = "sidecar.json"

# Dropped into the run directory by `relay cancel` *before* it sends the
# cancel. Its presence is how we tell "the user asked for this to stop" from
# "the scheduler took the node back", which is otherwise unknowable: both
# arrive as a bare SIGTERM.
CANCEL_FILENAME = "CANCEL_REQUESTED"

# What `requeue_on_term` may be by the time it reaches us. "auto" is resolved by
# the backend at submit time (it is the one that can ask Slurm about the
# partition's PreemptMode), so the sidecar never sees it.
REQUEUE_ALWAYS = "always"
REQUEUE_NEVER = "never"

# The placeholder the user's `resume.arg` must contain. It is substituted with
# the shell-quoted checkpoint path. A plain `str.replace` rather than
# `str.format`, so a stray brace elsewhere in the argument (`--opts={a:1}`)
# does not raise KeyError in the middle of starting a training job.
RESUME_PLACEHOLDER = "{path}"

# The `stream` tag on the log events the sidecar writes about itself, as
# opposed to the ones it copies out of the child's stdout and stderr. "relay"
# is not a pipe, and that is the point: it marks the line as ours.
RELAY_STREAM = "relay"


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with milliseconds and a literal Z.

    Example: 2026-09-16T18:22:09.003Z

    `datetime.isoformat()` gives microseconds and writes the zone as "+00:00";
    the schema says milliseconds and "Z", so we slice and substitute. Every
    timestamp in relay is UTC -- the cluster's local time and the laptop's local
    time are nobody's friend.
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------
# Parsing the ##relay## convention
# --------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """True for ints and floats, False for bools.

    `isinstance(True, int)` is True in Python, so a naive check would store
    `"converged": true` as the metric value 1.0. A flag is not a measurement.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_relay_line(line: str) -> tuple[str, dict] | None:
    """Turn a `##relay##` line into `(event_type, payload)`, or return None.

    Two shapes are recognised, both JSON objects after the marker:

        ##relay## {"step": 1000, "loss": 0.412}
            -> ("metric", {"step": 1000, "values": {"loss": 0.412}})

        ##relay## {"checkpoint": "ckpt/step_1000.pt", "step": 1000}
            -> ("checkpoint", {"path": "ckpt/step_1000.pt", "step": 1000})

    Numbers are nested under `values` so the envelope stays a fixed, parseable
    shape no matter what the user logs. Non-numeric values other than `step` are
    skipped rather than rejected, so `{"step": 10, "loss": 0.4, "note": "warm"}`
    still yields the loss.

    Returns None -- never raises -- when the line is not a relay line, or is
    malformed: bad JSON, not an object, or no numeric `step` on a metric. The
    caller emits a `log` event instead. A typo in a print statement should cost
    the user a metric, not their eight-hour training job.

    The sidecar never writes checkpoints; the training script does. All we do
    here is *observe* that one happened, so the daemon can show it and a resumed
    attempt can be checked against it.
    """
    text = line.rstrip("\r\n")
    if not text.startswith(METRIC_MARKER):
        return None

    try:
        obj = json.loads(text[len(METRIC_MARKER) :].strip())
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None

    raw_step = obj.get("step")
    step = int(raw_step) if _is_number(raw_step) else None

    # A checkpoint announcement wins over the metric reading: the line is about
    # the checkpoint, and `step` is there to say which step it belongs to.
    if "checkpoint" in obj:
        return "checkpoint", {"path": str(obj["checkpoint"]), "step": step}

    if step is None:
        # No step means we cannot place the numbers on any axis, so this is not
        # a metric line at all.
        return None

    values = {
        name: float(value)
        for name, value in obj.items()
        if name != "step" and _is_number(value)
    }
    return "metric", {"step": step, "values": values}


# --------------------------------------------------------------------------
# The event writer
# --------------------------------------------------------------------------


def highest_seq(path: str) -> int:
    """Highest `seq` already in an event file, or 0 if the file is new.

    This is what makes requeue safe. A requeued attempt that restarted `seq` at
    1 would write events whose (run, seq) keys collide with the first attempt's,
    and since the store inserts with INSERT OR IGNORE those events would be
    silently dropped. So the second attempt continues where the first stopped.

    Reads the whole file. That is the simple choice and it is fine: this runs
    once, at process start, on a file that is tens of megabytes at worst. The
    sophisticated alternative is to seek to the end and scan backwards for the
    last newline, but the last line is not guaranteed to hold the highest seq
    after a SIGKILL mid-write, and "read it all, take the max" has no edge cases.

    Unparseable lines are skipped, not fatal. A truncated final line is exactly
    what SIGKILL leaves behind, and refusing to start because of one would turn
    a recoverable job into a dead one.
    """
    best = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                seq = event.get("seq") if isinstance(event, dict) else None
                if isinstance(seq, int) and not isinstance(seq, bool) and seq > best:
                    best = seq
    except FileNotFoundError:
        return 0
    return best


class EventWriter:
    """Appends events to `events.jsonl`, assigning `seq` as it goes.

    Durability is the entire point, so this deliberately avoids Python's
    buffered file objects:

      * `os.open` with O_APPEND means every write lands at the current end of
        the file. Two processes (a previous attempt winding down, a requeued
        attempt starting) cannot interleave inside each other's line.
      * one `os.write` per complete line, newline included, so a reader never
        sees half a record unless the kernel truncated the write itself.
      * no buffering. An event sitting in a userspace buffer when SIGKILL
        arrives never existed, and SIGKILL is how a preempted job ends.

    The lock is an RLock, not a Lock, for a specific reason: four things emit
    events -- the main thread, the two reader threads, the heartbeat thread, and
    the signal handler. The signal handler runs *on the main thread*, so if a
    signal arrives while the main thread is inside `emit`, a plain Lock would
    deadlock the process against itself. RLock lets the same thread re-enter.
    """

    def __init__(self, path: str, run: str, start_seq: int = 0) -> None:
        self.path = path
        self.run = run
        self._seq = start_seq
        self._lock = threading.RLock()
        # O_APPEND is the important flag; the kernel does the "seek to end and
        # write" as one step, which is what makes concurrent appends safe.
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self._close_torn_line()

    def _close_torn_line(self) -> None:
        """If the file ends mid-record, terminate that line before appending.

        A SIGKILLed attempt can leave a half-written final line with no newline.
        Appending straight onto it would glue our first event onto that stump
        and produce one line that is neither record -- the daemon would drop it
        as corrupt, and we would lose a real event to someone else's crash.

        Writing a bare newline first quarantines the damage: the stump becomes
        its own (unparseable, safely skipped) line, and every event we write is
        a complete line of its own.
        """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size == 0:
            return
        with open(self.path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                os.write(self._fd, b"\n")

    @classmethod
    def open(cls, path: str, run: str) -> "EventWriter":
        """Open for append, continuing `seq` from whatever is already there."""
        return cls(path, run, start_seq=highest_seq(path))

    def emit(self, type: str, payload: dict | None = None) -> dict:
        """Write one event and return it. Thread-safe and signal-safe."""
        with self._lock:
            self._seq += 1
            event = {
                "v": SCHEMA_VERSION,
                "run": self.run,
                "seq": self._seq,
                "ts": utc_now_iso(),
                "type": type,
                "payload": payload if payload is not None else {},
            }
            # separators= keeps the line compact; there is no reason to ship
            # spaces over a filesystem the daemon reads byte-for-byte.
            line = json.dumps(event, separators=(",", ":")) + "\n"
            os.write(self._fd, line.encode("utf-8"))
            return event

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1


# --------------------------------------------------------------------------
# Startup facts: sidecar.json and the restart count
# --------------------------------------------------------------------------


class ResumeSpec(NamedTuple):
    """How to hand a previous attempt's checkpoint back to the training script.

    `checkpoint_glob` is a shell glob, relative to the job's working directory,
    naming the files the script writes its checkpoints to. `arg` is the
    argument to append to the command, containing `{path}` where the chosen
    file belongs -- `--resume {path}`, or `--resume={path}`, or
    `--ckpt-path={path}`. Whatever the script's own flag happens to be: relay
    does not get to dictate the training script's interface.
    """

    checkpoint_glob: str
    arg: str


class SidecarConfig(NamedTuple):
    """Everything `sidecar.json` tells us, with defaults already applied."""

    requeue_on_term: str
    preempt_grace: float
    # Wall-clock epoch seconds when the *first* attempt of this run started, as
    # the backend saw it. Checkpoints older than this belong to some earlier
    # run that happened to write into the same directory, so they are ignored.
    # 0 means "no filtering", which is what a config without the key gets.
    first_attempt_start: float
    resume: ResumeSpec | None


def load_sidecar_config(run_dir: str) -> SidecarConfig:
    """Read `<run_dir>/sidecar.json` and return it as a `SidecarConfig`.

    The backend writes this file at submit time, because the backend is the one
    that can ask Slurm what the partition's PreemptMode is. We read it exactly
    once, here at startup, and keep the answer in memory. That is not a
    micro-optimisation: the SIGTERM path must forward the signal to the child
    immediately, and "immediately" rules out opening a file on a shared
    filesystem whose metadata server may be having a slow afternoon.

    Shape:

        {"requeue_on_term": "always" | "never",
         "preempt_grace": 10,
         "first_attempt_start": 1758412929.5,
         "resume": {"checkpoint_glob": "ckpt/*.pt", "arg": "--resume {path}"}}

    Every key is optional. A file written by an older relay has only the first
    two, and it must keep working: a missing `first_attempt_start` means no
    mtime filtering, and a missing or null `resume` means the resume feature is
    simply off.

    Every way this can go wrong -- file missing, unreadable, not JSON, not an
    object, nonsense values -- falls back to the defaults rather than raising.
    A missing config file is the normal case when someone runs the sidecar by
    hand, and refusing to start a training job over it would be absurd. The
    default is `never`, the conservative choice: a job that does not come back
    wastes a queue slot, while a job that requeues when it should not can loop
    forever.

    `"auto"` is not a value we accept for `requeue_on_term`. The backend
    resolves it before writing the file, so anything other than "always" here
    means "never".
    """
    defaults = SidecarConfig(
        requeue_on_term=REQUEUE_NEVER,
        preempt_grace=DEFAULT_PREEMPT_GRACE,
        first_attempt_start=0.0,
        resume=None,
    )

    try:
        with open(os.path.join(run_dir, CONFIG_FILENAME), "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return defaults
    if not isinstance(config, dict):
        return defaults

    requeue = defaults.requeue_on_term
    if config.get("requeue_on_term") == REQUEUE_ALWAYS:
        requeue = REQUEUE_ALWAYS

    grace = defaults.preempt_grace
    raw_grace = config.get("preempt_grace")
    if _is_number(raw_grace) and raw_grace > 0:
        grace = float(raw_grace)

    start = defaults.first_attempt_start
    raw_start = config.get("first_attempt_start")
    if _is_number(raw_start) and raw_start > 0:
        start = float(raw_start)

    return SidecarConfig(
        requeue_on_term=requeue,
        preempt_grace=grace,
        first_attempt_start=start,
        resume=_parse_resume(config.get("resume")),
    )


def _parse_resume(raw: object) -> ResumeSpec | None:
    """Validate the `resume` block, or return None to leave the feature off.

    Both fields have to be non-empty strings and `arg` has to contain `{path}`,
    because an `arg` without the placeholder would append a flag with no value
    and the training script would fail in a way that looks like relay broke it.
    Anything short of that is treated as "no resume configured" rather than an
    error: the run is still perfectly runnable from scratch.
    """
    if not isinstance(raw, dict):
        return None
    pattern = raw.get("checkpoint_glob")
    arg = raw.get("arg")
    if not isinstance(pattern, str) or not pattern:
        return None
    if not isinstance(arg, str) or RESUME_PLACEHOLDER not in arg:
        return None
    return ResumeSpec(checkpoint_glob=pattern, arg=arg)


def read_restart_count() -> int:
    """How many times Slurm has already restarted this job: `SLURM_RESTART_COUNT`.

    Slurm sets this on a requeued job; on the first attempt it is absent or
    "0". It goes into `run_started.payload["attempt"]` so the daemon can tell
    one attempt's events from the next inside a single append-only log.

    Anything unparseable counts as 0. A wrong attempt number is a cosmetic
    problem; refusing to start is not.
    """
    raw = os.environ.get("SLURM_RESTART_COUNT")
    if not raw:
        return 0
    try:
        count = int(raw)
    except ValueError:
        return 0
    return count if count >= 0 else 0


# --------------------------------------------------------------------------
# Resuming from the previous attempt's checkpoint
# --------------------------------------------------------------------------
#
# The sidecar does not write checkpoints and does not read them. All it does is
# notice which file the last attempt left behind and put its path on the next
# attempt's command line. That keeps the knowledge of the checkpoint *format*
# entirely inside the training script, where it belongs.
#
# This whole section runs once, at startup, before the child is spawned. None
# of it is anywhere near the signal path: globbing a shared filesystem while a
# SIGTERM is in flight is exactly the kind of unbounded work the ten-second
# preemption window cannot afford.


def find_resume_checkpoint(
    spec: ResumeSpec, since: float = 0.0, root: str | None = None
) -> str | None:
    """Newest file matching `spec.checkpoint_glob` with mtime >= `since`.

    The glob is relative to the job's working directory, which is this
    process's cwd -- the same directory the training script will run in, so the
    pattern the user wrote in their config means what they think it means.
    `**` works, because `recursive=True`.

    `since` is the run's first attempt start time. Filtering on it is what
    stops attempt 1 from resuming off a checkpoint that some *earlier* run left
    in the same directory: silently continuing from a stale checkpoint is worse
    than starting over, because the metrics would look plausible and be wrong.
    A `since` of 0 disables the filter, which is what a config without
    `first_attempt_start` gets.

    Returns an absolute path, or None if nothing qualifies. Never raises: a
    missing directory or an unreadable file is "no checkpoint", not a failed
    job.
    """
    base = root if root is not None else os.getcwd()
    try:
        matches = glob.glob(spec.checkpoint_glob, root_dir=base, recursive=True)
    except OSError:
        return None

    best_path: str | None = None
    best_mtime = 0.0
    for match in matches:
        # os.path.join leaves an absolute match alone, so this handles both a
        # relative pattern (the normal case) and an absolute one.
        path = os.path.abspath(os.path.join(base, match))
        try:
            stat = os.stat(path)
        except OSError:
            # Vanished between the glob and the stat, or unreadable. Skip it.
            continue
        if not os.path.isfile(path):
            continue
        if stat.st_mtime < since:
            continue
        # Ties broken by path so the choice is deterministic; a filesystem with
        # one-second mtime granularity makes ties entirely possible.
        if best_path is None or (stat.st_mtime, path) > (best_mtime, best_path):
            best_path, best_mtime = path, stat.st_mtime
    return best_path


def resolve_resume(
    command: list[str],
    spec: ResumeSpec | None,
    attempt: int,
    since: float = 0.0,
    root: str | None = None,
) -> tuple[list[str], str | None, str | None]:
    """Decide what to append to the command. Returns `(command, path, note)`.

    `path` goes into `run_started.payload["resumed_from"]` and `note` becomes a
    log event written immediately after it. Either may be None.

    On attempt 0 there is nothing to resume from, so nothing happens at all --
    not even the "no checkpoint found" warning, which would be noise on a run
    that has not started yet.

    The argument is built by substituting the shell-quoted path into `arg` and
    then splitting the result with `shlex.split`. Quote-then-split round-trips
    exactly, which is the whole trick: a path with a space in it survives, and
    `--resume {path}` becomes two argv words while `--resume={path}` becomes
    one, without the sidecar having to guess which the user meant.
    """
    if spec is None or attempt == 0:
        return command, None, None

    path = find_resume_checkpoint(spec, since=since, root=root)
    if path is None:
        return (
            command,
            None,
            f"warning: no checkpoint matched {spec.checkpoint_glob} newer than "
            "the run's first start; starting from scratch",
        )

    words = shlex.split(spec.arg.replace(RESUME_PLACEHOLDER, shlex.quote(path)))
    return command + words, path, f"resuming from {path} (matched {spec.checkpoint_glob})"


def child_environment(run: str, run_dir: str) -> dict[str, str]:
    """The environment the training script gets, on every attempt alike.

    Three additions to our own environment:

      RELAY_RUN_ID   so a script can tag its own output with relay's run ID
                     (relay's, never Slurm's job ID -- that one changes on
                     requeue and this one does not).
      RELAY_RUN_DIR  the absolute run directory, which is where a checkpoint
                     helper knows to look.
      PYTHONPATH     with the run directory prepended, so `import relay_ckpt`
                     finds the copy the backend dropped in there. Prepended
                     rather than replacing: the user's own PYTHONPATH is how
                     their project is importable, and taking it away would
                     break the training script we are trying to run.

    Every value here is derived from the run ID and the run directory, both of
    which are fixed for the life of the run. So a requeued attempt gets a
    byte-identical environment by construction, rather than by us remembering
    to keep it the same.
    """
    env = dict(os.environ)
    env["RELAY_RUN_ID"] = run
    env["RELAY_RUN_DIR"] = run_dir

    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        run_dir + os.pathsep + existing if existing else run_dir
    )
    return env


def child_window(total: float) -> float:
    """Seconds of the signal window the child gets: `total - margin`, floored.

    The remainder is ours, to SIGKILL the child and write `run_ended` before
    the scheduler's SIGKILL reaches us as well.
    """
    return max(MIN_CHILD_WINDOW, total - SIGNAL_MARGIN)


# --------------------------------------------------------------------------
# The sidecar itself
# --------------------------------------------------------------------------


class Sidecar:
    """Owns the child process, the threads, and the shutdown decision."""

    def __init__(
        self,
        run: str,
        run_dir: str,
        command: list[str],
        grace: float = DEFAULT_GRACE,
        heartbeat: float = HEARTBEAT_INTERVAL,
    ) -> None:
        self.run = run
        # Absolute from here on. It is handed to the child as RELAY_RUN_DIR and
        # put on its PYTHONPATH, and a relative path would mean something
        # different to anything that chdirs.
        self.run_dir = os.path.abspath(run_dir)
        self.command = command
        self.grace = grace
        self.heartbeat_interval = heartbeat

        # Filled in at the top of `run_until_done`, before the child exists and
        # long before any signal can arrive. Defaults here so the object is
        # always in a usable state, including in tests that never call
        # `run_until_done`.
        self.requeue_on_term = REQUEUE_NEVER
        self.preempt_grace = DEFAULT_PREEMPT_GRACE
        self.attempt = 0
        # The checkpoint this attempt was told to resume from, or None. Goes
        # into run_started so the daemon can show where an attempt picked up.
        self.resumed_from: str | None = None

        self.child: subprocess.Popen | None = None
        self.writer: EventWriter | None = None
        self._readers: list[threading.Thread] = []

        # Shared state touched by several threads, so it gets its own lock.
        self._state_lock = threading.Lock()
        self._last_step: int | None = None

        self._stop_heartbeat = threading.Event()
        # Set once `run_ended` has been written, by whichever path got there
        # first. Guards against writing two endings for one run.
        self._ended = threading.Event()

    # -- small helpers -----------------------------------------------------

    def _note_step(self, step: int | None) -> None:
        if step is None:
            return
        with self._state_lock:
            self._last_step = step

    def _get_last_step(self) -> int | None:
        with self._state_lock:
            return self._last_step

    def _emit(self, type: str, payload: dict | None = None) -> None:
        if self.writer is not None:
            self.writer.emit(type, payload)

    # -- pipe readers ------------------------------------------------------

    def _read_pipe(self, pipe, stream_name: str) -> None:
        """Drain one pipe to EOF, turning each line into an event.

        There is one of these per pipe, and that is not an optimisation. The
        child writes to two pipes; if we read them one at a time, the pipe we
        are not reading fills its ~64KB kernel buffer, the child blocks writing
        to it, and it therefore stops writing to the pipe we *are* reading. Both
        sides wait forever. Two readers, no deadlock.

        Lines are mirrored to our own stdout/stderr as well, so Slurm's
        `slurm-<jobid>.out` still contains the job's output for anyone who goes
        looking at it directly. The event log is relay's copy, not a replacement.
        """
        mirror = sys.stdout if stream_name == "stdout" else sys.stderr
        try:
            # iter(readline, b"") reads line by line until EOF. Binary mode plus
            # an explicit decode means a stray non-UTF-8 byte from the training
            # script degrades one character instead of raising in a thread.
            for raw in iter(pipe.readline, b""):
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")

                try:
                    mirror.write(text + "\n")
                    mirror.flush()
                except (ValueError, OSError):
                    # Our own stdout can be closed out from under us during
                    # shutdown. Losing the mirror must not lose the event.
                    pass

                parsed = parse_relay_line(text)
                if parsed is None:
                    self._emit("log", {"stream": stream_name, "line": text})
                    continue

                event_type, payload = parsed
                self._note_step(payload.get("step"))
                self._emit(event_type, payload)
        except (ValueError, OSError):
            # The pipe was closed while we were reading it, which happens during
            # a hard kill. Nothing to do but stop reading.
            pass
        finally:
            try:
                pipe.close()
            except (ValueError, OSError):
                pass

    # -- heartbeat ---------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Emit `heartbeat` every interval until told to stop.

        `Event.wait(timeout)` rather than `time.sleep` so shutdown is immediate
        instead of waiting out a 30-second nap. It returns True when the event
        was set, which is our signal to stop.
        """
        while not self._stop_heartbeat.wait(self.heartbeat_interval):
            self._emit("heartbeat", {"last_step": self._get_last_step()})

    # -- signals -----------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Two signals, two handlers. The difference is the whole point.

        SIGUSR1 comes from `--signal=B:USR1@<grace>` and means only one thing:
        the time limit is approaching. SIGTERM means preemption *or* `scancel`,
        and the two are indistinguishable at signal time -- which is why the
        old code, which treated every SIGTERM as a preemption and requeued, would
        bring back a job the user had just cancelled.
        """
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGUSR1, self._on_sigusr1)

    # Both handlers run on the main thread -- Python delivers signals there --
    # which is why `EventWriter` uses an RLock (a plain Lock would deadlock the
    # process against itself if a signal arrived mid-`emit`) and why it is safe
    # to do real work here rather than only setting a flag. The main thread is
    # otherwise blocked in `child.wait()` with nothing better to do.
    #
    # Both end in `os._exit(0)` rather than `sys.exit`: raising SystemExit from
    # a handler would unwind through whatever the main thread was doing and
    # could be swallowed by somebody's `try/except`. And 0 is right even though
    # the child was killed -- the job did not fail, and a nonzero exit would
    # stop Slurm from requeueing it.

    def _on_sigusr1(self, signum, frame) -> None:
        """Time limit approaching: checkpoint, end the attempt, requeue.

        Unambiguous, because nothing but Slurm's time-limit warning sends
        USR1. So this path always self-requeues, and `CANCEL_REQUESTED` and
        `requeue_on_term` do not enter into it.
        """
        self._emit("signal", {"signal": "SIGUSR1", "reason": "time_limit"})
        self._stop_child(child_window(self.grace))
        self._end_attempt("requeued")
        self._requeue()
        os._exit(0)

    def _on_sigterm(self, signum, frame) -> None:
        """Preempted or cancelled -- we cannot tell which, so we say neither.

        The first two statements are the ones with a deadline on them: emit the
        `signal` event, then forward SIGTERM to the child so it can start
        writing its checkpoint. Everything that could take unbounded time --
        deciding about requeue, touching the filesystem -- happens *after* the
        child is dead and has had its window, which is why `requeue_on_term`
        was read at startup instead of here.
        """
        self._emit("signal", {"signal": "SIGTERM", "reason": "term"})
        self._stop_child(child_window(self.preempt_grace))

        # `interrupted`, not `cancelled` or `preempted`: the sidecar genuinely
        # does not know which happened. The daemon resolves it from sacct,
        # under the standing rule that the scheduler is authoritative about
        # what the job did.
        self._end_attempt("interrupted")

        # One `os.path.exists`, and only now that the child is dead and there
        # is nothing left to lose by spending a syscall on it. `relay cancel`
        # wrote this file *before* sending the signal, so if it is not here,
        # this was not a relay cancel.
        cancelled = os.path.exists(os.path.join(self.run_dir, CANCEL_FILENAME))
        if not cancelled and self.requeue_on_term == REQUEUE_ALWAYS:
            self._requeue()
        os._exit(0)

    def _stop_child(self, window: float) -> None:
        """Forward SIGTERM, wait up to `window`, then SIGKILL if it is still there.

        SIGTERM even when *we* were woken by SIGUSR1: the convention training
        scripts actually implement is "SIGTERM means save and stop", and a
        script that also handles USR1 is rare enough not to design around.

        The child's own handler may run twice -- once from the scheduler
        signalling the whole job step, once from this forward -- which is why
        the README tells users to make it idempotent.
        """
        child = self.child
        if child is None:
            return
        pid = child.pid

        # os.kill rather than child.send_signal, for the same reason the wait
        # below is a hand-rolled waitpid loop: see `_reap_child`.
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return

        if self._reap_child(pid, window):
            return

        # It had its chance. SIGKILL cannot be caught, so this is the end of
        # the child either way.
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            return
        self._reap_child(pid, POST_KILL_TIMEOUT)

    def _reap_child(self, pid: int, window: float) -> bool:
        """Wait up to `window` seconds for the child to die. True if it did.

        This polls `os.waitpid(pid, WNOHANG)` by hand rather than calling
        `child.wait(timeout=...)`, and that is not a style choice -- the
        obvious version cannot work here.

        We are inside a signal handler, which Python runs on the main thread,
        and the main thread is already inside `child.wait()` at the bottom of
        `run_until_done`. That outer call holds `Popen._waitpid_lock` and is
        suspended for the duration of the handler, so nothing is reaping. A
        nested `child.wait(timeout=...)` can only try the lock, fail, and spin
        until its timeout expires -- even for a child that died instantly.

        The cost of getting this wrong was not cosmetic. Every signal path
        burned its entire window plus another five seconds before writing
        `run_ended`; on Klone, where the whole preemption window is ten
        seconds, SIGKILL would arrive first and the attempt would end with no
        ending in the log at all.

        Reaping with a bare `waitpid` leaves the `Popen` object's bookkeeping
        stale, which does not matter: every caller of this exits via `os._exit`
        a moment later.
        """
        deadline = time.monotonic() + window
        while True:
            try:
                done, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                # Already reaped by somebody else. Gone is gone.
                return True
            except OSError:
                return True
            if done == pid:
                return True
            if time.monotonic() >= deadline:
                return False
            # 50ms: fast enough that a child which checkpoints in half a second
            # does not lose measurable time, cheap enough to be invisible.
            time.sleep(0.05)

    def _end_attempt(self, status: str) -> None:
        """Write `run_ended`, flush everything, and leave nothing in a buffer.

        The reader threads are joined *before* `run_ended` is written. The child
        is dead, so its pipes are at EOF and the join returns as soon as they
        drain -- but a checkpoint line printed in the child's final moments may
        still be sitting in a pipe buffer, and it is the single most valuable
        line in the file. `run_ended` must come after it, and must be the last
        event of this attempt.
        """
        self._stop_heartbeat.set()
        for reader in self._readers:
            reader.join(timeout=READER_JOIN_TIMEOUT)

        if not self._ended.is_set():
            self._ended.set()
            # exit_code is null: the child was signalled, not exited, and
            # inventing 128+15 here would look like the training script's own
            # verdict on itself.
            self._emit("run_ended", {"exit_code": None, "status": status})

        if self.writer is not None:
            self.writer.close()
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (ValueError, OSError):
                pass

    def _requeue(self) -> None:
        """Ask Slurm to put this job back in the queue.

        Guarded on SLURM_JOB_ID so that running the sidecar on a laptop (or in
        the test suite) does not shell out to a `scontrol` that is not there.
        Looked up on PATH rather than by absolute path, which is how Slurm's
        own tools are installed and what lets a test drop a fake `scontrol` in
        front of it.

        Failures are swallowed: we are seconds away from being SIGKILLed and
        there is nobody to report to. The event log already says how the
        attempt ended, and if the requeue did not take, the daemon will see the
        job disappear.
        """
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            return
        try:
            subprocess.run(
                ["scontrol", "requeue", job_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=POST_KILL_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

    # -- the run -----------------------------------------------------------

    def run_until_done(self) -> int:
        """Start the child, watch it, write the ending. Returns an exit code."""
        os.makedirs(self.run_dir, exist_ok=True)
        events_path = os.path.join(self.run_dir, EVENTS_FILENAME)

        # Read here, at startup, and never again. The signal handlers use these
        # values and must not be the ones to go and fetch them; see
        # `load_sidecar_config`.
        config = load_sidecar_config(self.run_dir)
        self.requeue_on_term = config.requeue_on_term
        self.preempt_grace = config.preempt_grace
        self.attempt = read_restart_count()

        # Resolved once, here, before anything is written -- so run_started can
        # carry `resumed_from`, and so the filesystem scan happens now rather
        # than during a preemption window. `self.command` becomes what actually
        # runs, resume argument included, which is what run_started reports.
        self.command, self.resumed_from, resume_note = resolve_resume(
            self.command, config.resume, self.attempt, since=config.first_attempt_start
        )

        # Continues seq across requeue attempts. See highest_seq.
        self.writer = EventWriter.open(events_path, self.run)

        # Handlers go in before the child is spawned. A signal arriving in the
        # gap would otherwise kill us with the default action and leave an
        # orphaned child running on the node.
        self._install_signal_handlers()

        self._emit(
            "run_started",
            {
                "command": self.command,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                # Always present, null off-cluster, so readers can use a plain
                # lookup instead of branching on whether the key exists.
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                # 0 on the first attempt, 1 on the first requeue, and so on.
                # A single append-only log holds every attempt's events, so
                # this is what separates them.
                "attempt": self.attempt,
                # Absolute path of the checkpoint this attempt was handed, or
                # null on attempt 0, with resume unconfigured, or when nothing
                # matched. Always present, so readers need not branch on the
                # key existing.
                "resumed_from": self.resumed_from,
            },
        )

        # Right after run_started, so the story reads in order: this is the
        # attempt, and this is what it picked up from. A warning goes here too
        # -- an attempt that silently restarted from scratch is the failure the
        # user most needs to see in the log.
        if resume_note is not None:
            self._emit("log", {"stream": RELAY_STREAM, "line": resume_note})

        try:
            self.child = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=child_environment(self.run, self.run_dir),
            )
        except OSError as exc:
            # The command does not exist or is not executable. That is a failed
            # run, not a crashed sidecar, so it gets a normal ending.
            self._emit("log", {"stream": "stderr", "line": f"relay: {exc}"})
            self._ended.set()
            self._emit("run_ended", {"exit_code": 127, "status": "failed"})
            self.writer.close()
            return 127

        readers = [
            threading.Thread(
                target=self._read_pipe,
                args=(self.child.stdout, "stdout"),
                name="relay-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=self._read_pipe,
                args=(self.child.stderr, "stderr"),
                name="relay-stderr",
                daemon=True,
            ),
        ]
        # Kept on self so the signal path can drain them too before it writes
        # run_ended -- the checkpoint line printed right before preemption is
        # exactly the one we cannot afford to lose.
        self._readers = readers
        for reader in readers:
            reader.start()

        heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="relay-heartbeat", daemon=True
        )
        heartbeat.start()

        # Wait for the child. A signal arriving here runs one of the handlers
        # on this thread, and that path never comes back (it ends in os._exit).
        returncode = self.child.wait()

        # Drain whatever the child wrote just before exiting, so those log lines
        # land before run_ended rather than after it or not at all.
        self._stop_heartbeat.set()
        for reader in readers:
            reader.join(timeout=READER_JOIN_TIMEOUT)

        exit_code = _normalize_exit_code(returncode)
        if not self._ended.is_set():
            self._ended.set()
            self._emit(
                "run_ended",
                {
                    "exit_code": exit_code,
                    "status": "completed" if exit_code == 0 else "failed",
                },
            )
        self.writer.close()
        return exit_code


def _normalize_exit_code(returncode: int) -> int:
    """Turn Popen's returncode into a number a shell can carry.

    Popen reports a child killed by signal N as -N, but a process exit status is
    a byte. The shell convention for "died from signal N" is 128 + N, so we use
    that; the sidecar exits with the child's code because a wrapper that always
    exits 0 would tell Slurm every job succeeded.
    """
    if returncode < 0:
        return 128 + (-returncode)
    return returncode


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at the first bare `--` into (our flags, the user's command).

    Done by hand instead of with argparse.REMAINDER, which has documented
    misbehaviour when flags appear after the separator -- and the user's command
    is nothing but flags: `-- python train.py --lr 3e-4`.
    """
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own_args, command = _split_command(argv)

    parser = argparse.ArgumentParser(
        prog="sidecar.py",
        description="Wrap a training command and append relay events to a file.",
        epilog="Everything after a bare -- is the command to run.",
    )
    parser.add_argument("--run", required=True, help="relay run ID (not a Slurm job ID)")
    parser.add_argument("--dir", required=True, help="run directory for events.jsonl")
    parser.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE,
        help=(
            "seconds of warning before the job's TIME LIMIT, matching "
            "--signal=B:USR1@<grace> (default: 60). The preemption window is a "
            "different number and comes from sidecar.json, not from a flag: "
            "the backend knows it at submit time and we must not go looking "
            "for it while a SIGTERM is in flight."
        ),
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=HEARTBEAT_INTERVAL,
        help="seconds between heartbeat events (default: 30)",
    )
    args = parser.parse_args(own_args)

    if not command:
        parser.error("no command given; put it after a bare -- separator")

    sidecar = Sidecar(
        run=args.run,
        run_dir=args.dir,
        command=command,
        grace=args.grace,
        heartbeat=args.heartbeat,
    )
    return sidecar.run_until_done()


if __name__ == "__main__":
    sys.exit(main())
