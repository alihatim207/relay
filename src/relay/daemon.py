"""The daemon: the loop that moves bytes from the cluster into local SQLite.

Everything else in relay is a library. This is the only long-running process,
and it does exactly one thing over and over:

    1. ask the store which runs are not finished yet
    2. ask the backend, in ONE call, what the scheduler thinks of all of them
    3. for each run, read the new bytes of `events.jsonl` from the byte offset
       we stored last time, parse them into events, and hand events + the new
       offset to `store.ingest()` -- one transaction
    4. reconcile the scheduler's view of status with the run row
    5. flag runs whose heartbeat has gone quiet
    6. every few minutes, in ONE more call, ask what the scheduler charged us
       for the runs it is still accounting for
    7. sleep for a while, the length of the while depending on how busy things
       are, then do it all again

Four ideas are worth understanding before reading the code.

**`run_once()` is public and separately callable.** That is not an accident of
design, it is the design: a loop you can only start and stop is a loop you can
only test with `sleep()` and hope. Every rule in this module -- torn lines,
offset rewinding, staleness, status precedence, backoff -- is exercised by
calling `run_once()` by hand and looking at the database afterwards.
`run_forever()` adds nothing but a sleep.

**Transport failures are normal, not exceptional.** SSH drops, the login node
is busy, GPFS hiccups. None of that is a crash; it is Tuesday. Every read is
wrapped, every error is counted and logged, and the loop continues with the
next run. The daemon's job is to still be running tomorrow.

**Idempotence makes recovery trivial.** Because `store.ingest()` commits the
events and the byte offset in a single transaction, and because every insert is
`INSERT OR IGNORE` against a composite primary key, a daemon killed at the
worst possible moment can be restarted with no cleanup: at worst it re-reads a
region of the file it has already seen, and re-reading is a no-op. That is why
there is no recovery code here. There is nothing to recover.

**One transport failure needs a human, and the daemon has to know which.** On a
Duo-protected cluster, "ssh failed" splits in two. If the control master has
died and a new connection would need somebody to tap a phone, retrying is not
just useless, it is harmful: every attempt burns a cycle and, without
`BatchMode=yes`, would hang waiting for a prompt on a terminal nobody is
looking at. So the daemon classifies each failure (`relay.ssh.classify_failure`)
and records the verdict in `daemon_state`. On `auth_required` it stops talking
to the cluster entirely and does nothing but glance at its own control socket
once a minute, until the user runs `relay connect`.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from relay import config as cfg
from relay import ssh
from relay import store as store_module
from relay.events import EventParseError, parse_event
from relay.store import Store

log = logging.getLogger("relay.daemon")

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

# The file the sidecar appends to inside each run directory. Spelled out here
# rather than imported from `relay.sidecar`, because the sidecar is a
# standalone file shipped to the cluster and the daemon should not depend on
# being able to import it. The two agree by contract, like everything else that
# crosses the laptop/cluster boundary.
EVENTS_FILENAME = "events.jsonl"

# How long a `running` run may go without a heartbeat before we flag it stale.
# The sidecar heartbeats every 30 seconds, so five minutes is ten missed beats:
# long enough that a slow filesystem or a busy node cannot trigger it, short
# enough that a human notices within a coffee break.
STALE_AFTER_SECONDS = 300.0

# The three polling intervals. See `next_interval` for which applies when.
INTERVAL_ACTIVE = 2.0  # something is running or producing output
INTERVAL_QUEUED = 30.0  # runs exist, but they are all waiting in the queue
INTERVAL_IDLE = 60.0  # nothing to watch at all

# Exponential backoff after whole-cycle failures: 4s, 8s, 16s ... capped.
# The cap matters more than the growth rate. Without it, a laptop left asleep
# for a weekend would come back and wait hours before trying again.
BACKOFF_CAP = 300.0

# How often to ask the scheduler what it charged us. Usage is accounting data,
# not live data: nobody is watching GPU-hours tick up the way they watch a loss
# curve, and `sacct` is the most expensive question we ask the login node. Five
# minutes, plus one final reading the moment a run goes terminal.
USAGE_INTERVAL_SECONDS = 300.0

# How often, while locked out of the cluster, to look at our own control
# socket. `ssh -O check` talks to a local Unix socket and nothing else, so it
# costs nothing and can never trigger Duo -- but it is still a process spawn,
# and spawning one every two seconds for an hour would be silly.
AUTH_CHECK_INTERVAL = 60.0

# How much of a transport error's text we keep in `daemon_state`. Enough for a
# human to recognise the failure in `relay ls`, not enough to turn one row of a
# small table into a log file.
ERROR_DETAIL_CHARS = 200


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class AlreadyRunning(RuntimeError):
    """Another daemon already holds the lock on this machine.

    Its own class, rather than a bare RuntimeError, so the CLI can print a
    plain sentence and exit cleanly instead of showing a traceback.
    """


# --------------------------------------------------------------------------
# Cycle statistics
# --------------------------------------------------------------------------


@dataclass
class CycleStats:
    """What happened during one pass of the loop.

    Returned by `run_once()`. It exists for three audiences: `next_interval`
    decides how long to sleep from it, the log line at the end of a cycle is
    built from it, and the tests assert on it. Counters are cheap; guessing
    what a daemon did from its side effects is not.
    """

    # How many non-terminal runs we looked at this cycle.
    runs_checked: int = 0
    # How many of those the scheduler says are running / queued *after*
    # reconciliation. `next_interval` uses these.
    running: int = 0
    queued: int = 0

    # Ingestion.
    bytes_read: int = 0
    events_parsed: int = 0
    events_inserted: int = 0
    metrics_inserted: int = 0
    # Lines that failed `parse_event`. Counted, logged at debug, and dropped.
    # A single corrupt line must never stop the file behind it from loading.
    bad_lines: int = 0

    # Errors. `read_attempts` counts calls to `read_bytes` that we made;
    # `read_failures` counts the ones that raised something other than
    # FileNotFoundError (which is not a failure -- see `_sync_events`).
    read_attempts: int = 0
    read_failures: int = 0
    status_failed: bool = False
    # Total problems worth reporting this cycle: read failures, ingest
    # failures, and the batched status call if it blew up.
    errors: int = 0

    # Usage accounting. `usage_fetched` means we called `backend.usage()` this
    # cycle and it answered; `usage_rows` is how many attempt rows that
    # produced. Both stay at their defaults on the cycles -- most of them --
    # where no fetch was due.
    usage_rows: int = 0
    usage_fetched: bool = False
    usage_failed: bool = False

    # Transport diagnosis. `error_kind` is "auth_required" or "unreachable"
    # when something failed at the transport level this cycle and we wrote that
    # verdict into `daemon_state`; None when nothing did.
    error_kind: str | None = None
    # True when the cycle deliberately did *not* touch the cluster, because we
    # are waiting for the user to run `relay connect`. Nothing was read, no
    # status was asked for, and that is not a failure -- it is the correct
    # behaviour when we know every request would bounce.
    auth_required: bool = False

    # Reconciliation.
    status_changes: int = 0
    stale_flagged: int = 0
    stale_cleared: int = 0

    # Was the cycle a whole-transport failure? Set at the end of `run_once`.
    failed: bool = False
    # How many cycles in a row have now failed, including this one. Zero after
    # any success. `next_interval` reads this and nothing else to back off.
    consecutive_failures: int = 0

    duration_s: float = 0.0

    @property
    def new_data(self) -> bool:
        """Did any run produce bytes we had not seen before?"""
        return self.bytes_read > 0


def next_interval(stats: CycleStats) -> float:
    """How many seconds to sleep after a cycle that produced `stats`.

    A pure function of the stats so it can be unit-tested without a daemon, a
    store or a clock.

    The policy, in order of precedence:

      * authentication needed -> one minute, flat. This is not a retry
        schedule: the daemon is not talking to the cluster at all, it is
        glancing at a local socket to see whether the user has run
        `relay connect` yet. Backing off would only make it slower to notice
        that they did, and there is nothing to be gentle with -- the check
        never leaves the laptop.
      * consecutive failures -> exponential backoff, capped. If the transport
        is down, hammering it neither fixes it nor helps anyone.
      * something is running, or bytes arrived -> 2 seconds. This is the case
        the user is watching on the dashboard.
      * runs exist but are all queued -> 30 seconds. A job that starts tomorrow
        does not become visible faster because we asked squeue 1800 times an
        hour, and a shared login node has many other users on it.
      * nothing active at all -> 60 seconds. Just enough to notice a new
        submission promptly.
    """
    if stats.auth_required:
        return AUTH_CHECK_INTERVAL
    if stats.consecutive_failures > 0:
        # 2 * 2**n starting at n=1: 4, 8, 16, 32 ... then the cap.
        return min(BACKOFF_CAP, INTERVAL_ACTIVE * (2 ** stats.consecutive_failures))
    if stats.new_data or stats.running > 0:
        return INTERVAL_ACTIVE
    if stats.runs_checked > 0:
        return INTERVAL_QUEUED
    return INTERVAL_IDLE


# --------------------------------------------------------------------------
# The single-instance lock
# --------------------------------------------------------------------------


def acquire_lock(path: str | os.PathLike | None = None) -> IO[str]:
    """Take the daemon's exclusive lock, or raise `AlreadyRunning`.

    Returns the open file object. **The caller must hold on to it for the
    daemon's entire lifetime**: the lock belongs to the open file description,
    so letting the object be garbage collected would close the descriptor and
    silently release the lock while the daemon is still running.

    Why `flock` and not a PID file. A PID file is a claim that outlives the
    process that wrote it: kill -9 the daemon and the file remains, looking
    exactly like a healthy daemon to the next process that reads it. The usual
    patch -- check whether that PID is alive -- is wrong too, because PIDs get
    recycled. A flock is held by the kernel on behalf of an open descriptor,
    and the kernel closes every descriptor a process had when it dies, however
    it dies. Crash, SIGKILL, power loss: the lock is gone, correctly.

    We still write the PID into the file, but only as a courtesy to a human
    running `cat daemon.lock`. It is documentation, not a mechanism.

    Subtlety worth knowing, and what `doctor._check_daemon` relies on: flock
    locks are per *open file description*, not per process. Two separate
    `open()` calls on the same path -- even inside one process -- create two
    descriptions and genuinely conflict. (POSIX record locks, `fcntl.lockf`,
    are the opposite: per process, so a second attempt in the same process
    quietly succeeds and would make this untestable in one interpreter.)
    """
    lock_path = Path(path) if path is not None else cfg.daemon_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # os.open + fdopen rather than plain open(path, "a+"): we want to truncate
    # and rewrite the PID, and a file opened in append mode ignores seeks when
    # writing, so the PIDs of every daemon that ever ran would pile up.
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    handle = os.fdopen(fd, "r+")

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            raise AlreadyRunning(
                f"Another relay daemon is already running: it holds the lock on "
                f"{lock_path}. Stop that one first, or leave it be -- one daemon "
                f"is all you need."
            ) from exc
        raise

    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        # The PID is a comment for humans. Failing to write it is not a reason
        # to refuse to run, and we already hold the lock, which is the part
        # that matters.
        log.debug("could not write pid into %s", lock_path, exc_info=True)

    return handle


def release_lock(handle: IO[str]) -> None:
    """Release the lock and close the file. Safe to call on a closed handle.

    Closing alone would be enough -- the kernel drops the lock with the
    descriptor -- but unlocking explicitly makes the intent obvious.
    """
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError):
        pass
    try:
        handle.close()
    except OSError:
        pass


# --------------------------------------------------------------------------
# The daemon
# --------------------------------------------------------------------------


class Daemon:
    """Tails every active run's event log into the store.

    Construct it with a `Store` and a `Backend`; it does not know or care which
    backend it got. The intervals and the staleness threshold are constructor
    arguments purely so tests can shrink them -- normal callers pass neither.

    The daemon owns no threads and no timers. `run_once()` is synchronous and
    does everything a cycle does; `run_forever()` is that in a `while` with a
    sleep. All the state that survives between cycles lives in the database,
    apart from one integer: how many cycles in a row have failed.
    """

    def __init__(
        self,
        store: Store,
        backend,
        *,
        stale_after_seconds: float = STALE_AFTER_SECONDS,
        interval_active: float = INTERVAL_ACTIVE,
        interval_queued: float = INTERVAL_QUEUED,
        interval_idle: float = INTERVAL_IDLE,
        usage_interval_seconds: float = USAGE_INTERVAL_SECONDS,
        auth_check_interval: float = AUTH_CHECK_INTERVAL,
    ) -> None:
        self.store = store
        self.backend = backend
        self.stale_after_seconds = stale_after_seconds
        self.interval_active = interval_active
        self.interval_queued = interval_queued
        self.interval_idle = interval_idle
        self.usage_interval_seconds = usage_interval_seconds
        self.auth_check_interval = auth_check_interval

        # Consecutive whole-cycle failures. The only cross-cycle state in the
        # process, and losing it on a restart costs nothing: a restarted daemon
        # simply tries immediately, which is what you want anyway.
        self.consecutive_failures = 0
        self.cycles = 0

        # When we last ran `ssh -O check`, on the monotonic clock. Deliberately
        # in memory and not in the database: the only thing it does is stop us
        # spawning a process every two seconds while locked out, and a daemon
        # that has just restarted *should* check immediately. There is nothing
        # here worth surviving a restart.
        self._last_auth_check: float | None = None

    # -- one cycle ---------------------------------------------------------

    def run_once(self) -> CycleStats:
        """Do exactly one pass: status, read, ingest, reconcile, flag stale.

        Public and callable on its own on purpose. Everything interesting the
        daemon does happens inside one cycle, so a test can drive the daemon a
        single step at a time, with a fake backend and a real store, and assert
        on the database afterwards -- no sleeping, no threads, no flakiness.
        `run_forever` is this method plus a sleep and nothing else.
        """
        started = time.monotonic()
        stats = CycleStats()

        # The daemon's own state row, read once here because three separate
        # decisions this cycle depend on it: are we locked out of the cluster,
        # is a usage fetch due, and is there an old error to clear. One read is
        # cheaper than three and, more to the point, they all see the same
        # answer.
        state = self.store.get_daemon_state()

        if not self._may_touch_cluster(state, stats):
            # Waiting for `relay connect`. Nothing was attempted, so there is
            # nothing to count, nothing to sync, and no verdict to reach about
            # the transport.
            self.cycles += 1
            stats.consecutive_failures = self.consecutive_failures
            stats.duration_s = time.monotonic() - started
            return stats

        runs = self.store.list_runs(active_only=True)
        stats.runs_checked = len(runs)

        statuses = self._batched_status(runs, stats)

        # Did any run finish during this cycle? If so we take its final usage
        # reading now rather than whenever the five-minute timer next comes
        # round -- see `_fetch_usage`.
        went_terminal = False

        for run in runs:
            # Order matters here, and it is not the obvious order.
            #
            # The status snapshot was taken BEFORE we read any bytes. So if the
            # scheduler told us a job had finished, everything that job will
            # ever write was already on disk by the time we read. Reading first
            # and applying the status afterwards therefore cannot lose the tail
            # of a run's log -- whereas marking a run terminal first would drop
            # it out of `list_runs(active_only=True)` and we would never come
            # back for those final bytes.
            self._sync_events(run, stats)
            went_terminal |= self._sync_status(run, statuses, stats)
            self._check_stale(run, stats)

            if run["status"] == "running":
                stats.running += 1
            elif run["status"] == "queued":
                stats.queued += 1

        # Usage comes last in the cycle on purpose: it depends on the statuses
        # we have just written (that is how it knows which runs are finished),
        # and it is the one thing here nobody is watching in real time.
        if self._usage_due(state, forced=went_terminal):
            self._fetch_usage(stats)

        stats.failed = self._cycle_failed(stats)
        if stats.failed:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            # "Last synced" only means something if the cycle actually worked.
            # `relay ls` prints it, and stale data presented as current is the
            # one thing we refuse to do.
            self.store.mark_synced()

        self._record_cycle_health(state, stats)
        stats.consecutive_failures = self.consecutive_failures

        self.cycles += 1
        stats.duration_s = time.monotonic() - started

        log.debug(
            "cycle: %d runs, %d bytes, %d events (+%d new), %d errors, %.3fs",
            stats.runs_checked,
            stats.bytes_read,
            stats.events_parsed,
            stats.events_inserted,
            stats.errors,
            stats.duration_s,
        )
        return stats

    # -- the ssh channel: is it usable, and if not, why not? ---------------

    def _may_touch_cluster(self, state: dict, stats: CycleStats) -> bool:
        """May this cycle send commands to the cluster? Usually yes.

        The exception is `auth_required`: ssh has told us there is no live
        master and no way to make a connection without a Duo tap. Every command
        we could send this cycle would fail the same way, and on a relay built
        without `BatchMode=yes` they would not even fail -- they would sit
        waiting for a prompt nobody can see, and the daemon would hang. So we
        send nothing at all and do one cheap thing instead: `ssh -O check`,
        which only speaks to the local control socket, costs nothing, and can
        never trigger Duo.

        The moment the check says a master is up -- the user has run
        `relay connect` -- we clear the error and let *this* cycle carry on
        normally. Making them wait another interval for something we already
        know would be rude.
        """
        if state.get("last_error_kind") != ssh.AUTH_REQUIRED:
            return True

        alias = getattr(self.backend, "ssh_alias", None)
        if alias is None:
            # This backend does not go over ssh at all (the config was switched
            # to local, say), so there is no socket to check and no Duo to
            # answer. An error we have no way to retest must not lock the
            # daemon out forever.
            self.store.clear_daemon_error()
            state["last_error_kind"] = None
            return True

        now = time.monotonic()
        if (
            self._last_auth_check is not None
            and now - self._last_auth_check < self.auth_check_interval
        ):
            stats.auth_required = True
            return False
        self._last_auth_check = now

        if not ssh.master_alive(alias, runner=getattr(self.backend, "runner", None)):
            stats.auth_required = True
            log.debug("no ssh master for %s yet; waiting for `relay connect`", alias)
            return False

        log.info("ssh master for %s is up again; resuming normal polling", alias)
        self.store.clear_daemon_error()
        state["last_error_kind"] = None
        self._last_auth_check = None
        return True

    def _record_transport_failure(self, exc: BaseException, stats: CycleStats) -> None:
        """Work out what kind of transport failure this was and write it down.

        Two kinds, and the difference matters to the human: `auth_required`
        means retrying is pointless until they run `relay connect`, while
        `unreachable` means the network had a bad minute and the ordinary
        backoff will sort it out. `relay ls`, `relay usage` and the dashboard
        read `daemon_state` and say one thing or the other.

        The raw material comes off the exception as attributes rather than out
        of its message: `TransportError` carries `returncode` and `stderr`
        precisely so this function does not have to parse English. Anything
        else -- a bare `ConnectionError` from a backend that is not ssh-shaped
        -- simply has neither, and falls through to `unreachable`.

        We classify at most once per cycle. The second failing read this cycle
        has the same cause as the first, and `ssh -O check` is a process spawn
        we should not repeat once per run.
        """
        if stats.error_kind is not None:
            return

        returncode = getattr(exc, "returncode", None)
        stderr = getattr(exc, "stderr", "") or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        alias = getattr(self.backend, "ssh_alias", None)
        if alias is None:
            # Not an ssh backend: there is no master to check, so there is no
            # such thing as "authentication needed" here. Whatever broke, the
            # only sensible reaction is to try again later.
            kind = ssh.UNREACHABLE
        else:
            kind = ssh.classify_failure(
                returncode if isinstance(returncode, int) else 0,
                stderr,
                ssh.master_alive(alias, runner=getattr(self.backend, "runner", None)),
            )
            if kind == ssh.AUTH_REQUIRED:
                # We have just run the check, so the next cycle's gate should
                # wait a full interval before running it again.
                self._last_auth_check = time.monotonic()

        detail = (str(exc) or stderr.strip()).replace("\n", " ")[:ERROR_DETAIL_CHARS]
        stats.error_kind = kind
        self.store.set_daemon_state(
            last_error_kind=kind,
            last_error_ts=store_module.utc_now_iso(),
            last_error_detail=detail,
        )
        if kind == ssh.AUTH_REQUIRED:
            log.warning("ssh authentication needed; run `relay connect`. (%s)", detail)

    def _record_cycle_health(self, state: dict, stats: CycleStats) -> None:
        """After a cycle with no transport failure at all, say so.

        `last_ok_ts` is "the last time the channel demonstrably worked", which
        is not the same as `meta.last_sync_ts` ("the last time a whole cycle
        succeeded") and is the one the "Authentication needed" banner is turned
        off by. Clearing has to be as prompt as setting: a stale
        `auth_required` row would keep the daemon in check-only mode after the
        problem had gone away.
        """
        if stats.error_kind is not None:
            return
        if state.get("last_error_kind") is not None:
            self.store.clear_daemon_error()
        self.store.set_daemon_state(last_ok_ts=store_module.utc_now_iso())

    # -- step 1: one status call for everything ----------------------------

    def _batched_status(self, runs: list[dict], stats: CycleStats) -> dict[str, str]:
        """Ask the backend about every run's job in a single call.

        One call, not one per run. On Slurm each call is an SSH round trip to a
        Duo-protected login node, so the difference between one `squeue` and
        forty is the difference between a daemon and a nuisance.

        A failure here is recorded and swallowed: we still want to read event
        bytes this cycle, because the event log lives on the filesystem and may
        be reachable even when the scheduler command is not.
        """
        job_ids = [str(run["job_id"]) for run in runs if run.get("job_id")]
        if not job_ids:
            return {}
        try:
            return self.backend.status(job_ids)
        except Exception as exc:  # noqa: BLE001 -- transport errors are normal
            stats.status_failed = True
            stats.errors += 1
            self._record_transport_failure(exc, stats)
            log.warning("status call failed for %d job(s): %s", len(job_ids), exc)
            return {}

    # -- step 2: read and ingest ------------------------------------------

    def _sync_events(self, run: dict, stats: CycleStats) -> None:
        """Read whatever is new in one run's event log and ingest it."""
        run_id = run["run_id"]
        run_dir = run.get("run_dir")
        if not run_dir:
            # No directory recorded means nothing was ever submitted for this
            # row (or a hand-written row). Nothing to tail.
            return

        path = os.path.join(run_dir, EVENTS_FILENAME)
        offset = self.store.get_offset(run_id, path)

        try:
            stats.read_attempts += 1
            data, new_offset = self.backend.read_bytes(path, offset)
        except FileNotFoundError:
            # Not an error. The job is queued and has not created its run
            # directory yet, which is the normal state of a job that starts
            # tomorrow. Counting it as a failure would make the daemon back off
            # from exactly the runs it is waiting for.
            log.debug("no event log yet for %s at %s", run_id, path)
            return
        except Exception as exc:  # noqa: BLE001
            stats.read_failures += 1
            stats.errors += 1
            self._record_transport_failure(exc, stats)
            log.warning("could not read %s for run %s: %s", path, run_id, exc)
            return

        if not data:
            return
        stats.bytes_read += len(data)

        lines, new_offset = _split_complete_lines(data, new_offset)

        events = []
        for raw in lines:
            # errors="replace" rather than letting a UnicodeDecodeError escape:
            # one garbled byte should cost one line, not the whole batch. The
            # replacement character will fail `parse_event` and be counted as a
            # bad line, which is exactly the right outcome.
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                events.append(parse_event(text))
            except EventParseError as exc:
                stats.bad_lines += 1
                log.debug("bad line in %s: %s", path, exc)

        try:
            result = self.store.ingest(run_id, path, events, new_offset)
        except Exception as exc:  # noqa: BLE001 -- a locked database, say
            stats.errors += 1
            log.warning("could not ingest %d event(s) for run %s: %s", len(events), run_id, exc)
            # The transaction rolled back, so the stored offset still points at
            # the start of this chunk. Next cycle re-reads exactly these bytes.
            return

        stats.events_parsed += len(events)
        stats.events_inserted += result["events_inserted"]
        stats.metrics_inserted += result["metrics_inserted"]

    # -- step 3: the scheduler owns status ---------------------------------

    def _sync_status(self, run: dict, statuses: dict[str, str], stats: CycleStats) -> bool:
        """Write the scheduler's opinion of this job onto the run row.

        Returns True when this run *became* terminal in this cycle, which is
        the signal `run_once` uses to take one last usage reading immediately.

        The division of authority, which is the single most important rule in
        the daemon:

          * the **event log** is authoritative about what the *training* did --
            steps, metrics, checkpoints. `store.ingest` already wrote those.
          * the **scheduler** is authoritative about what the *job* did --
            queued, running, preempted, exit code.

        So a run becomes terminal only when the backend says so, never because
        the log contained a `run_ended`. A job SIGKILLed mid-step writes no
        `run_ended` at all, and a job that was preempted writes `run_ended`
        with status `requeued` and then keeps going under a new job ID. Trusting
        the log would strand the first kind at "running" forever and bury the
        second kind while it is still alive.

        `"unknown"` is not a status, it is the absence of one. The backend is
        telling us it could not find out this cycle -- a Slurm job aged out of
        `sacct`, a local PID that vanished without writing an exit code -- and
        overwriting a known status with it would throw away a fact in exchange
        for an admission of ignorance.

        Note what this method does **not** do, because it is the one place
        somebody would expect it to: it never looks at `run_ended`. Schema v2
        added the status `interrupted`, which the sidecar writes when it is
        SIGTERMed -- and it writes that precisely because at signal time it
        cannot tell preemption from `scancel`. Resolving that ambiguity is the
        scheduler's job and nobody else's. Slurm will either put the job back
        in the queue (so `backend.status` reports it queued again, the run stays
        non-terminal, and the same run ID keeps writing to the same event log)
        or report it CANCELLED, which is terminal. Mapping `interrupted` to a
        status here would guess at exactly the question we deliberately
        refused to answer in the sidecar. See "Status conflict resolution" in
        CLAUDE.md.
        """
        job_id = run.get("job_id")
        if not job_id:
            return False
        observed = statuses.get(str(job_id))
        if observed is None or observed == "unknown":
            return False
        if observed not in store_module.STATUSES:
            log.warning("backend reported unknown status %r for job %s", observed, job_id)
            return False
        if observed == run["status"]:
            return False

        self.store.update_run_status(run["run_id"], observed)
        # Keep the in-memory row in step, since `_check_stale` and the
        # running/queued tally below both read it.
        run["status"] = observed
        stats.status_changes += 1
        log.info("run %s: %s", run["run_id"], observed)
        return observed in store_module.TERMINAL

    # -- step 4: staleness is a flag, not a status -------------------------

    def _check_stale(self, run: dict, stats: CycleStats) -> None:
        """Flag a running run whose heartbeat has gone quiet; unflag it when it returns.

        Stale is deliberately a separate column from `status`. A stale run is
        still exactly whatever the scheduler says it is; all we have observed is
        that we stopped hearing from the sidecar. That could be a hung training
        loop, or it could be a filesystem that has not flushed yet. The daemon
        does not diagnose and it certainly does not kill anything -- it raises
        an eyebrow, in one boolean, and lets the human decide.

        A run with no heartbeat at all is *not* stale. We have no evidence
        either way, and the obvious alternative (measure from `created_ts`)
        would flag every job that sat in the queue for longer than the threshold
        the instant it started running, which on a busy cluster is all of them.
        """
        # Re-read the row: `store.ingest` may have just advanced
        # `last_heartbeat_ts` inside its transaction, and the copy we are
        # holding predates that.
        fresh = self.store.get_run(run["run_id"]) or run
        was_stale = bool(fresh.get("stale"))

        is_stale = fresh.get("status") == "running" and _older_than(
            fresh.get("last_heartbeat_ts"), self.stale_after_seconds
        )

        if is_stale == was_stale:
            return
        self.store.set_stale(run["run_id"], is_stale)
        if is_stale:
            stats.stale_flagged += 1
            log.warning(
                "run %s has not sent a heartbeat in over %.0fs; flagging stale",
                run["run_id"],
                self.stale_after_seconds,
            )
        else:
            stats.stale_cleared += 1
            log.info("run %s is heartbeating again; clearing stale", run["run_id"])

    # -- step 5: what the scheduler charged us -----------------------------

    def _usage_due(self, state: dict, *, forced: bool) -> bool:
        """Is it time to ask the scheduler for usage numbers?

        `forced` is set when a run went terminal in this cycle. A finished
        job's numbers never change again, so that is the moment to take the
        last reading -- waiting up to five minutes for the timer would leave
        the run sitting in `relay usage` with whatever partial hours we last
        happened to see.
        """
        if forced:
            return True
        last = state.get("last_usage_fetch_ts")
        if not last:
            return True
        return _older_than(last, self.usage_interval_seconds)

    def _fetch_usage(self, stats: CycleStats) -> None:
        """One batched `usage` call for every run that still needs one.

        The shape is the same as `_batched_status` and for the same reason: one
        `sacct` per fetch, not one per run.

        Two things here are worth spelling out.

        **How a run stops being asked about.** `store.runs_needing_usage()`
        returns non-terminal runs plus terminal runs whose `usage_final` flag
        is not set. So the fetch that happens right after a run goes terminal
        is, by construction, the last one it will ever be part of: we take the
        numbers, set the flag, and the run drops out of the query forever. That
        is the spec's "once more when a run transitions to terminal, then never
        again", implemented as one flag rather than as a schedule.

        **A failure here does not spoil the cycle.** Usage is accounting, not
        live data. If `sacct` is unhappy we record the transport failure like
        any other, leave `last_usage_fetch_ts` alone so the next cycle tries
        again, and the events we ingested a moment ago stay ingested.
        """
        rows_needed = self.store.runs_needing_usage()
        now = store_module.utc_now_iso()

        if not rows_needed:
            # Nothing to ask about. We still stamp the fetch time: without it,
            # an idle daemon would run this query every single cycle forever to
            # rediscover that there is still nothing to do.
            self.store.set_daemon_state(last_usage_fetch_ts=now)
            return

        # Job ID -> run row. The backend reports usage by job ID because that
        # is all the scheduler knows; relay files it by run ID because that is
        # what survives a requeue. This map is the join.
        by_job = {str(run["job_id"]): run for run in rows_needed}

        try:
            reported = self.backend.usage(sorted(by_job))
        except Exception as exc:  # noqa: BLE001 -- transport errors are normal
            stats.usage_failed = True
            stats.errors += 1
            self._record_transport_failure(exc, stats)
            log.warning("usage call failed for %d job(s): %s", len(by_job), exc)
            return

        rows = []
        for item in reported:
            row = asdict(item) if is_dataclass(item) else dict(item)
            run = by_job.get(str(row.get("job_id")))
            if run is None:
                # A job ID we did not ask about. `sacct -j` should not do this,
                # but there is nowhere to file a row whose run we cannot name,
                # and inventing a run_id would corrupt somebody's hours.
                log.debug("dropping usage row for unrequested job %r", row.get("job_id"))
                continue
            row["run_id"] = run["run_id"]
            rows.append(row)

        try:
            stats.usage_rows = self.store.upsert_usage(rows)
        except Exception as exc:  # noqa: BLE001 -- a locked database, say
            stats.usage_failed = True
            stats.errors += 1
            log.warning("could not store %d usage row(s): %s", len(rows), exc)
            return

        for run in rows_needed:
            if run["status"] in store_module.TERMINAL:
                self.store.mark_usage_final(run["run_id"])

        self.store.set_daemon_state(last_usage_fetch_ts=now)
        stats.usage_fetched = True

    # -- failure accounting ------------------------------------------------

    def _cycle_failed(self, stats: CycleStats) -> bool:
        """Was this cycle a transport failure rather than a bad minute?

        Backing off is about the *channel*, not about one unhappy run, so a
        single run whose file could not be read does not slow down polling for
        the other thirty. A cycle counts as failed when either everything we
        tried to read failed, or the status call failed and there was nothing
        else to try. Both describe "the cluster is not answering right now".
        """
        if stats.read_attempts > 0 and stats.read_failures == stats.read_attempts:
            return True
        return stats.status_failed and stats.read_attempts == 0

    # -- the loop ----------------------------------------------------------

    def run_forever(self, stop_event: threading.Event | None = None) -> int:
        """Run cycles until `stop_event` is set. Returns the number of cycles run.

        Shutdown is a flag checked between cycles, never a signal that
        interrupts work in progress. That is what "never interrupt a
        transaction" comes down to in practice: every transaction lives inside
        `store.ingest`, this loop only checks the flag between calls, so there
        is no code path on which a commit is abandoned halfway.

        The sleep is `stop_event.wait(interval)` rather than `time.sleep`, so
        Ctrl-C or a SIGTERM stops the daemon immediately instead of after
        however much of a 60-second nap was left.
        """
        if stop_event is None:
            stop_event = threading.Event()

        while not stop_event.is_set():
            stats = self.run_once()
            interval = self._interval_for(stats)
            if stats.failed:
                log.warning(
                    "cycle failed (%d in a row); retrying in %.0fs",
                    stats.consecutive_failures,
                    interval,
                )
            # Returns True the moment the event is set, False on timeout.
            if stop_event.wait(interval):
                break

        log.info("daemon stopping after %d cycle(s)", self.cycles)
        return self.cycles

    def _interval_for(self, stats: CycleStats) -> float:
        """`next_interval`, but honouring this instance's overrides.

        The module-level function holds the policy (and stays pure, so it is
        trivial to test); this method only substitutes the instance's intervals
        for the defaults, which exists so a test can run a real loop without
        waiting two seconds a cycle.
        """
        base = next_interval(stats)
        if stats.auth_required:
            return self.auth_check_interval
        if stats.consecutive_failures > 0:
            return base
        if base == INTERVAL_ACTIVE:
            return self.interval_active
        if base == INTERVAL_QUEUED:
            return self.interval_queued
        return self.interval_idle


# --------------------------------------------------------------------------
# Line splitting: the torn-line rule
# --------------------------------------------------------------------------


def _split_complete_lines(data: bytes, new_offset: int) -> tuple[list[bytes], int]:
    """Split a chunk into complete lines, rewinding past any partial last line.

    The sidecar writes each event with a single `os.write` of a complete line,
    but "a single write" is not the same as "a single read": we can read at any
    moment, including while the filesystem has only made the first half of a
    line visible. So a chunk that does not end in a newline ends in an
    incomplete record.

    We drop that record and move the offset back by its length, so next cycle
    starts at the beginning of it and reads it whole. Returning the partial
    line to the parser instead would produce a bad line and, worse, a stored
    offset past it -- the event would be lost permanently.

    Returns the complete lines (no newline characters) and the corrected
    offset.
    """
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        # The split leaves a trailing empty element after the final newline.
        lines.pop()
    else:
        partial = lines.pop()
        new_offset -= len(partial)
    return lines, new_offset


def _older_than(ts: str | None, seconds: float) -> bool:
    """Is this ISO-8601 timestamp more than `seconds` in the past?

    Returns False for None and for anything unparseable. "I cannot tell" must
    not become "yes": the only thing this answer is used for is raising an
    alarm, and an alarm you cannot explain is worse than no alarm.
    """
    if not ts:
        return False
    try:
        # 3.11's fromisoformat understands the trailing "Z" that the event
        # schema uses. Earlier versions did not, which is why the project
        # requires 3.11+.
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        log.debug("unparseable timestamp %r", ts)
        return False
    if parsed.tzinfo is None:
        # Everything relay writes is UTC; assume it rather than crash.
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - parsed).total_seconds()
    return age > seconds


# --------------------------------------------------------------------------
# Entry point for `relay daemon`
# --------------------------------------------------------------------------


def build_backend(conf: "cfg.Config", *, runner=None):
    """Construct the backend the config asks for.

    `SlurmBackend` is imported inside the branch, not at module scope. The
    daemon must be importable -- and the local backend usable -- on a machine
    with no cluster and no SSH, and a top-level import would drag in that
    module (and any future dependency of it) just to run a local job.

    `runner` is the seam where `SlurmBackend` would otherwise spawn `ssh`. It
    is threaded through rather than reached for from inside, so an end-to-end
    test can drive a whole daemon against a scripted cluster without a real
    connection. `LocalBackend` has no such seam: it really does start
    processes, which is the point of it.
    """
    if conf.backend == "local":
        from relay.backends.local import LocalBackend

        return LocalBackend(root=conf.local_root)

    if conf.backend == "slurm":
        from relay.backends.slurm import SlurmBackend

        return SlurmBackend.from_config(conf, runner=runner)

    raise ValueError(f"unknown backend {conf.backend!r} in {cfg.config_path()}")


def main(
    store_path: str | os.PathLike | None = None,
    *,
    config_path: str | os.PathLike | None = None,
    lock_path: str | os.PathLike | None = None,
    stop_event: threading.Event | None = None,
    runner=None,
) -> int:
    """Run the daemon until it is asked to stop. Returns a process exit code.

    The CLI calls this for `relay daemon`. Everything it does is arrangement:
    take the lock, build a store and a backend, arrange for signals to set the
    stop flag, loop, then put everything back.

    Signal handlers only *set a flag*. They do not touch the database, print,
    or raise. A handler runs between bytecodes at an arbitrary point -- possibly
    in the middle of a transaction -- so the only safe thing it can do is leave
    a note for the loop to find when it next comes up for air.
    """
    conf = cfg.load(config_path)
    handle = acquire_lock(lock_path)

    store = Store(store_path if store_path is not None else cfg.db_path())
    store.meta_set(store_module.META_DAEMON_STARTED_TS, store_module.utc_now_iso())

    if stop_event is None:
        stop_event = threading.Event()

    def _request_stop(signum, _frame):
        log.info("received %s; finishing the current cycle and stopping", signal.Signals(signum).name)
        stop_event.set()

    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, _request_stop)

    daemon = Daemon(store, build_backend(conf, runner=runner))
    log.info("daemon started (backend=%s, db=%s)", conf.backend, store.path)

    try:
        daemon.run_forever(stop_event)
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)
        store.close()
        release_lock(handle)

    return 0
