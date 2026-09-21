"""End to end: submit -> monitor -> preempt -> resume -> finish, with no cluster.

This file is the one test that drives *all* of relay at once, so it doubles as
living documentation of the architecture. Read it top to bottom and you have
read the whole design.

The story it tells is the one a researcher lives through:

  1. SUBMIT   A training script is handed to `LocalBackend`. The backend copies
              `sidecar.py` next to the job and starts it; the sidecar wraps the
              training script and appends one JSON line per event to
              `events.jsonl` in the run directory. A row goes into SQLite
              saying "this run exists, it is queued, here is its job ID".

  2. MONITOR  `Daemon.run_once()` is called over and over. Each pass asks the
              backend for status in one batched call, reads the new bytes of
              `events.jsonl` from the stored byte offset, parses them and
              commits events + metrics + the new offset in ONE transaction.
              The user sees live metrics because of this loop and nothing else.

  3. PREEMPT  A bare SIGTERM to the job's process group stands in for Slurm
              evicting a preempted job: the sidecar catches it, forwards it so
              the training script can checkpoint, waits out `preempt_grace`,
              and records `signal` + `run_ended{status: "interrupted"}`.

              Sent by hand rather than through `backend.cancel()`, and the
              difference is the point. `cancel` now writes a
              `CANCEL_REQUESTED` file *before* it signals, which is how a
              cancelled run is stopped from requeueing itself. A preemption
              has no such file, because nobody asked for it. The ending is
              `interrupted` either way: SIGTERM is the same signal in both
              cases and the sidecar refuses to guess which one it was.

  4. RESUME   The same run ID is submitted again into the SAME run directory --
              which is exactly what Slurm does to a requeued job. The sidecar
              reads the tail of the existing `events.jsonl` and continues `seq`
              from where the last attempt stopped, instead of restarting at 1
              and colliding with it. The training script reads its checkpoint
              and resumes from the step it saved.

  5. FINISH   The second attempt runs to completion, and the daemon marks the
              run terminal -- because the *backend* said so, never because the
              log contained a `run_ended`.

  6. CODA     A brand new database is pointed at the same files and caught up
              from scratch. It must end up byte-identical in content to the one
              that watched the run live. Same bytes in, same database out. That
              idempotence is what makes "just restart it" a valid recovery
              strategy everywhere else in relay.

Two rules this file obeys throughout, because CI machines are slow and shared:

  * **Every wait is a deadline loop.** `wait_until` / `World.pump` poll until a
    condition actually holds. A `sleep(0.5); assert done` is a test that fails
    on Tuesdays, and a flaky end-to-end test is worse than no end-to-end test
    because people learn to ignore it.

  * **Nothing is mocked.** Real processes, real signals, real files, real
    SQLite. The bugs this test exists to catch -- a lost checkpoint line, a
    truncated log, a colliding `seq` -- only exist between processes.

These tests are an ordered story: they share one class-scoped `World` and each
stage builds on the last. Running them out of order (with `-k`) will fail
loudly rather than silently pass.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from relay.backends.base import JobSpec
from relay.backends.local import EXIT_CODE_FILENAME, LocalBackend
from relay.daemon import EVENTS_FILENAME, Daemon
from relay.store import Store

# Upper bounds on "something has gone wrong", not on how long the test takes.
# Every one of these loops normally finishes in well under a second.
TIMEOUT = 30.0
POLL = 0.05

RUN_ID = "vr_s0_e2e01"

# The training loop's pace. Small enough that the whole file runs in seconds,
# slow enough that the daemon gets several cycles' worth of partial output --
# which is the interesting case, since that is where torn lines come from.
STEP_SECONDS = 0.05
CKPT_EVERY = 5

# The sidecar's own knobs, shrunk for the test.
#   grace 10          -> the time-limit warning window (SIGUSR1), unused here.
#   preempt_grace 10  -> on SIGTERM the child gets 10 - 2 = 8 seconds to
#                        checkpoint before it is SIGKILLed. This script needs
#                        about 0.3 of one.
#   heartbeat 0.5     -> several heartbeats instead of waiting 30s for one.
GRACE_SECONDS = 10
PREEMPT_GRACE_SECONDS = 10
HEARTBEAT_SECONDS = 0.5


# --------------------------------------------------------------------------
# The fake training script
# --------------------------------------------------------------------------
#
# This is deliberately written the way CLAUDE.md says a real user's script
# should be written, with no relay import anywhere in it:
#
#   * metrics are reported by printing `##relay## {...}` with flush=True
#   * checkpoints are written by the *script*, never by the sidecar, and
#     announced with `##relay## {"checkpoint": path, "step": n}`
#   * SIGTERM means "save and stop" -- the contract that makes preemption
#     survivable. Slurm sends it `grace` seconds before SIGKILL; the sidecar
#     forwards it; this handler is what turns that warning into a checkpoint.
#
# `flush=True` is not decoration. The sidecar reads the child's stdout through
# a pipe, and a pipe is not a terminal, so Python would block-buffer 8KB of
# output by default -- the user would see nothing for minutes and then
# everything at once, and a SIGKILLed job would lose the lot.

TRAIN_SCRIPT = r'''
import argparse
import json
import os
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--state", required=True)      # checkpoint file
parser.add_argument("--max-steps", type=int, default=1000000)
parser.add_argument("--ckpt-every", type=int, default=5)
parser.add_argument("--step-seconds", type=float, default=0.05)
args = parser.parse_args()


def load_start_step():
    """Resume from the checkpoint if there is one, else start at 0."""
    try:
        with open(args.state, "r") as handle:
            return int(json.load(handle)["step"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def save(step):
    """Write the checkpoint atomically, then tell relay it happened."""
    tmp = args.state + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({"step": step}, handle)
    os.replace(tmp, args.state)
    print("##relay## " + json.dumps({"checkpoint": args.state, "step": step}),
          flush=True)


start = load_start_step()
current = start
print("training from step %d" % start, flush=True)

_handled = False


def on_term(signum, frame):
    """Preemption warning: checkpoint, then get out of the way."""
    global _handled
    if _handled:
        return
    _handled = True
    save(current)
    # The checkpoint line is now in the pipe, but the sidecar's reader thread
    # has to actually read it before we vanish. A real training script spends
    # this long (and much longer) flushing gigabytes of optimizer state; here
    # it is 0.3s of deliberate politeness, well inside the 8s grace window.
    time.sleep(0.3)
    os._exit(0)


signal.signal(signal.SIGTERM, on_term)

step = start
while step < args.max_steps:
    current = step
    # Loss decreases monotonically so the assertions can check the shape of
    # the series, not just its existence.
    print("##relay## " + json.dumps({"step": step, "loss": round(1.0 / (1.0 + step), 6)}),
          flush=True)
    if step != start and step % args.ckpt_every == 0:
        save(step)
    time.sleep(args.step_seconds)
    step += 1

print("training complete", flush=True)
'''


# --------------------------------------------------------------------------
# Waiting, done properly
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


def kill_group(job_id: str | None) -> None:
    """SIGKILL a job's process group, ignoring every way it can already be gone.

    Only used for cleanup. Nothing in the test proper reaches for SIGKILL --
    that is the whole point of the grace window.
    """
    if not job_id:
        return
    try:
        os.killpg(int(job_id), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, ValueError, OverflowError):
        pass


# --------------------------------------------------------------------------
# The world the story happens in
# --------------------------------------------------------------------------


class World:
    """Everything the stages share: paths, the store, the backend, the daemon.

    It also carries the facts one stage discovers and the next one needs -- the
    checkpoint step, the highest `seq` of the first attempt -- because that is
    precisely what the resume logic has to get right.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_id = RUN_ID
        self.run_dir = root / "runs" / self.run_id
        self.state_path = self.run_dir / "checkpoint.json"
        self.script = root / "train.py"
        self.script.write_text(TRAIN_SCRIPT, encoding="utf-8")

        self.store = Store(root / "relay.db")
        # `root=` keeps the backend's pid -> run_dir index inside tmp_path. No
        # config file is loaded anywhere in this test and nothing touches the
        # real home directory.
        self.backend = LocalBackend(root=root / "local")
        self.daemon = Daemon(self.store, self.backend)

        # The path the daemon tails. Spelled the same way the daemon spells it,
        # because `sources` is keyed on (run_id, path) and a different spelling
        # would silently start a second offset from zero.
        self.events_path = os.path.join(str(self.run_dir), EVENTS_FILENAME)

        # Filled in as the story goes.
        self.job_id_1: str | None = None
        self.job_id_2: str | None = None
        self.checkpoint_step: int | None = None
        self.seq_after_preempt: int | None = None
        self.offsets: list[int] = []

    # -- driving the daemon by hand ------------------------------------

    def cycle(self):
        """One daemon pass, recording the byte offset it left behind.

        `run_once()` being separately callable is the design, not an accident:
        a loop you can only start and stop is a loop you can only test with
        sleeps and hope.
        """
        stats = self.daemon.run_once()
        self.offsets.append(self.store.get_offset(self.run_id, self.events_path))
        return stats

    def pump(self, predicate, timeout: float = TIMEOUT, what: str = "condition"):
        """Run daemon cycles until `predicate` holds, or fail the test."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.cycle()
            value = predicate()
            if value:
                return value
            time.sleep(POLL)
        pytest.fail(f"timed out after {timeout}s waiting for {what}")

    # -- reading the database ------------------------------------------

    def events(self, type: str | None = None) -> list[dict]:
        return self.store.list_events(self.run_id, type=type)

    def seqs(self) -> list[int]:
        return [event["seq"] for event in self.events()]

    def run(self) -> dict:
        row = self.store.get_run(self.run_id)
        assert row is not None, "the run row disappeared"
        return row

    def spec(self, *, max_steps: int) -> JobSpec:
        """A JobSpec for this run. Called twice: once per attempt.

        Note what does NOT change between the two attempts: `run_id` and
        `run_dir`. A requeued Slurm job keeps relay's run ID and lands back in
        the same directory, appending to the same `events.jsonl`. Only the
        scheduler's job ID changes, which is exactly why nothing in relay is
        keyed on the job ID.
        """
        return JobSpec(
            run_id=self.run_id,
            command=[
                sys.executable,
                str(self.script),
                "--state",
                str(self.state_path),
                "--max-steps",
                str(max_steps),
                "--ckpt-every",
                str(CKPT_EVERY),
                "--step-seconds",
                str(STEP_SECONDS),
            ],
            run_dir=str(self.run_dir),
            name="e2e",
            grace_seconds=GRACE_SECONDS,
            preempt_grace=PREEMPT_GRACE_SECONDS,
            heartbeat_seconds=HEARTBEAT_SECONDS,
        )


@pytest.fixture(scope="class")
def world(tmp_path_factory):
    """One world for the whole story. Class-scoped: the stages share it.

    The finalizer SIGKILLs anything still breathing. A test that fails midway
    must not leave a training loop running on the developer's laptop.
    """
    w = World(tmp_path_factory.mktemp("e2e"))
    try:
        yield w
    finally:
        kill_group(w.job_id_1)
        kill_group(w.job_id_2)
        w.store.close()


def stage_result(world: World, attribute: str):
    """Fetch a fact an earlier stage was supposed to establish.

    Fails with an explanation rather than an AttributeError, because the most
    likely cause is running one of these tests on its own.
    """
    value = getattr(world, attribute)
    if value is None:
        pytest.fail(
            f"{attribute} was never set: the stages of this end-to-end test are "
            f"an ordered story and must run together, in file order."
        )
    return value


# --------------------------------------------------------------------------
# The story
# --------------------------------------------------------------------------


class TestSubmitMonitorPreemptResume:
    """submit -> monitor -> preempt -> resume -> finish, in that order."""

    # -- 1. SUBMIT -----------------------------------------------------

    def test_submit_starts_the_job_and_records_the_run(self, world: World):
        """Submitting starts a sidecar and puts a `queued` row in the store.

        The two halves are deliberately separate calls: a backend never touches
        the database (it does not know relay has one) and the store never
        starts a process. The CLI is what joins them, and here the test plays
        that part.
        """
        spec = world.spec(max_steps=1_000_000)  # effectively forever
        world.job_id_1 = world.backend.submit(spec)

        assert world.job_id_1.isdigit(), "a local job ID is a PID"

        # `sidecar.py` is copied next to the job rather than imported, exactly
        # as it will be copied onto shared storage for Slurm -- a compute node
        # may not have relay installed at all.
        assert (world.run_dir / "sidecar.py").is_file()

        row = world.store.create_run(
            world.run_id,
            backend="local",
            name="e2e",
            job_id=world.job_id_1,
            status="queued",
            run_dir=str(world.run_dir),
            spec={"command": spec.command},
        )
        assert row["status"] == "queued"
        assert row["terminal"] is False
        assert row["run_dir"] == str(world.run_dir)

    # -- 2. MONITOR ----------------------------------------------------

    def test_daemon_streams_events_metrics_and_heartbeats(self, world: World):
        """The daemon turns a file on disk into live rows in SQLite.

        Asserted here: the run goes `running`, `run_started` is seq 1, metric
        rows match the `##relay##` lines one for one, checkpoints are recorded,
        heartbeats land, and `seq` is contiguous from 1 with no holes.
        """
        stage_result(world, "job_id_1")

        # Wait for a decent slice of training: several metrics, at least one
        # checkpoint (so step >= CKPT_EVERY) and at least one heartbeat.
        def flowing():
            return (
                world.run()["status"] == "running"
                and len(world.events("metric")) >= CKPT_EVERY + 3
                and len(world.events("checkpoint")) >= 1
                and len(world.events("heartbeat")) >= 1
            )

        world.pump(flowing, what="metrics, a checkpoint and a heartbeat to arrive")

        # -- the scheduler owns `status` --------------------------------
        # "running" here came from LocalBackend (the PID is alive and no
        # exit_code file exists), never from the event log. That division of
        # authority is the daemon's single most important rule.
        assert world.run()["status"] == "running"
        assert world.run()["stale"] is False

        # -- the envelope -----------------------------------------------
        events = world.events()
        assert events[0]["seq"] == 1
        assert events[0]["type"] == "run_started"
        assert events[0]["payload"]["command"][-1].endswith(str(STEP_SECONDS))

        # Contiguous from 1. A gap would mean an event was lost between the
        # sidecar's `os.write` and the store's INSERT; a repeat is impossible
        # by construction, since (run_id, seq) is the primary key.
        seqs = world.seqs()
        assert seqs == list(range(1, len(seqs) + 1))

        # -- metrics: the denormalised read path ------------------------
        # Every `##relay##` metric line became both an event (faithful history)
        # and a row in `metrics` (what the dashboard charts without parsing
        # JSON). The two must agree exactly.
        metric_events = world.events("metric")
        steps_from_events = [event["payload"]["step"] for event in metric_events]
        series = world.store.metric_series(world.run_id, "loss")
        assert [point["step"] for point in series] == steps_from_events
        assert steps_from_events == list(range(len(steps_from_events)))

        # The values survived the round trip through JSON and SQLite's REAL.
        for point in series:
            assert point["value"] == pytest.approx(round(1.0 / (1.0 + point["step"]), 6))
        # ... and the loss is falling, which is the shape a human looks for.
        values = [point["value"] for point in series]
        assert values == sorted(values, reverse=True)
        assert world.store.metric_names(world.run_id) == ["loss"]

        # -- checkpoints: observed, never written by relay ---------------
        checkpoints = world.events("checkpoint")
        assert checkpoints[0]["payload"]["path"] == str(world.state_path)
        assert checkpoints[0]["payload"]["step"] % CKPT_EVERY == 0
        # The sidecar only *observes* that a checkpoint happened. The training
        # script is what wrote the file.
        assert world.state_path.is_file()

        # -- heartbeats: proof of life -----------------------------------
        row = world.run()
        assert row["last_heartbeat_ts"] is not None
        # `last_step` is advanced by the metric events. (Heartbeats carry the
        # step too, but the sidecar spells that payload key `last_step` while
        # store.ingest looks for `step`, so a heartbeat currently contributes
        # only its timestamp. Harmless here -- metrics keep last_step honest --
        # but the two spellings should agree.)
        assert row["last_step"] == steps_from_events[-1]

        # -- the offset only moves forward -------------------------------
        assert world.offsets == sorted(world.offsets)
        assert world.offsets[-1] > 0

    # -- 3. PREEMPT ----------------------------------------------------

    def test_sigterm_preempts_the_job_and_the_sidecar_records_it(self, world: World):
        """SIGTERM to the process group stands in for Slurm preemption.

        Sent by hand, to the whole group -- wrapper, sidecar and training
        script -- which is as close as a laptop gets to being evicted from a
        preemptible partition. Deliberately *not* `backend.cancel()`: that
        writes `CANCEL_REQUESTED` first, which is the cancel case, and a
        preemption leaves no such file because nobody asked for it.

        What follows is the interesting part: the sidecar catches it, forwards
        it, the script checkpoints, and the log gains `signal` +
        `run_ended{status: "interrupted"}`.
        """
        job_id = stage_result(world, "job_id_1")

        os.killpg(int(job_id), signal.SIGTERM)

        # The grace window is real time: the script saves, waits for the reader
        # thread, and exits; then the sidecar writes its ending; then the `sh`
        # wrapper records the exit code. Wait for the last of those, since that
        # is the file `status` consults.
        wait_until(
            lambda: (world.run_dir / EXIT_CODE_FILENAME).is_file(),
            what="the job to die and record its exit code",
        )

        # One more pass to pick up everything written during the shutdown.
        # Note the ordering rule inside `run_once`: the status snapshot is
        # taken BEFORE any bytes are read, so a job that the scheduler already
        # calls finished has, by definition, already written everything it will
        # ever write. Reading first therefore cannot lose the tail of the log.
        world.pump(
            lambda: world.events("run_ended"),
            what="the interrupted ending to be ingested",
        )

        signals = world.events("signal")
        assert signals, "the sidecar should record the signal that preempted it"
        assert signals[-1]["payload"] == {"signal": "SIGTERM", "reason": "term"}

        ended = world.events("run_ended")[-1]
        # `interrupted`, not `cancelled` and not `preempted`: SIGTERM is the
        # same signal in both cases, so the sidecar claims neither and leaves
        # the verdict to the daemon, which asks the scheduler.
        assert ended["payload"]["status"] == "interrupted"
        assert ended["payload"]["exit_code"] is None

        # No `CANCEL_REQUESTED` file was written, because this was a
        # preemption. (There is nothing to requeue into locally either, which
        # is why `requeue_on_term: auto` resolved to `never` at submit time.)
        assert not (world.run_dir / "CANCEL_REQUESTED").exists()

        # The checkpoint written *by the signal handler* is the one a resumed
        # attempt will start from. This is the contract that makes preemption
        # survivable, and the reason the sidecar forwards SIGTERM instead of
        # killing immediately.
        checkpoints = world.events("checkpoint")
        world.checkpoint_step = checkpoints[-1]["payload"]["step"]
        assert json.loads(world.state_path.read_text())["step"] == world.checkpoint_step

        # -- and here is the mismatch that proves the rule ----------------
        # The log says `interrupted`. The backend says `completed`, because the
        # sidecar exits 0 on the requeue path (a nonzero exit would stop Slurm
        # from requeueing the job) and the `sh` wrapper faithfully wrote a 0.
        #
        # The store follows the BACKEND, not the log: the scheduler is
        # authoritative about what the *job* did, the log about what the
        # *training* did. On a real cluster Slurm would say "preempted/pending"
        # here and the run would stay active; locally there is no scheduler to
        # say it, so we get `completed` and the resume stage puts the row back
        # to `queued` by hand -- which is precisely the bookkeeping Slurm does
        # for us in production. Accepting this mismatch is the point: it is why
        # the daemon never derives status from the event log.
        assert world.run()["status"] == "completed"

        world.seq_after_preempt = world.store.max_seq(world.run_id)
        assert world.seq_after_preempt == len(world.seqs())
        assert world.seq_after_preempt >= 10

    # -- 4. RESUME -----------------------------------------------------

    def test_resume_continues_seq_and_restarts_from_the_checkpoint(self, world: World):
        """The second attempt appends to the same log and resumes the same run.

        Two independent promises are checked here, and both are about silent
        data loss rather than crashes:

          * `seq` continues from the first attempt. An attempt that restarted
            at 1 would write keys that collide with the existing ones, and
            because the store inserts with INSERT OR IGNORE, its events would
            be dropped without a single error message.

          * the resumed training starts from the checkpoint step, so the two
            attempts' metrics form one continuous series.
        """
        previous_max_seq = stage_result(world, "seq_after_preempt")
        checkpoint_step = stage_result(world, "checkpoint_step")
        offset_before = world.store.get_offset(world.run_id, world.events_path)

        # Same run ID, same run directory: this models Slurm starting the
        # requeued attempt, not a new experiment.
        spec = world.spec(max_steps=checkpoint_step + 25)
        world.job_id_2 = world.backend.submit(spec)
        assert world.job_id_2 != world.job_id_1, "a requeued job gets a new job ID"

        # Put the row back in the daemon's working set under the new job ID.
        # (`status` is scheduler-owned, and locally we are standing in for the
        # scheduler.) Nothing else about the run changes -- not the run ID, not
        # the run directory, and crucially not the stored byte offset.
        world.store.update_run(world.run_id, status="queued", job_id=world.job_id_2)

        world.pump(
            lambda: world.store.max_seq(world.run_id) > previous_max_seq + 3,
            what="the resumed attempt to start writing events",
        )

        # -- seq continued, nothing collided -----------------------------
        seqs = world.seqs()
        assert len(seqs) == len(set(seqs)), "duplicate seq: the log keys collided"
        assert seqs == list(range(1, len(seqs) + 1)), "a hole in seq means a lost event"
        assert world.store.count_events(world.run_id) == world.store.max_seq(world.run_id)

        resumed = [event for event in world.events() if event["seq"] > previous_max_seq]
        assert resumed[0]["seq"] == previous_max_seq + 1
        assert resumed[0]["type"] == "run_started", (
            "the second attempt's first event should be its own run_started, "
            "numbered one past the first attempt's last event"
        )

        # -- the resumed step matches the checkpoint ----------------------
        resumed_metrics = [event for event in resumed if event["type"] == "metric"]
        assert resumed_metrics, "the resumed attempt should be producing metrics"
        assert resumed_metrics[0]["payload"]["step"] == checkpoint_step

        # Both attempts' points live in one series, in step order, with no
        # duplicates: `metrics` is keyed on (run_id, name, step), so the
        # overlapping step at the resume boundary is de-duplicated for free.
        series = world.store.metric_series(world.run_id, "loss")
        steps = [point["step"] for point in series]
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))
        assert steps[0] == 0, "the first attempt's points are still there"
        assert max(steps) >= checkpoint_step

        # -- appended, never truncated -----------------------------------
        # Slurm needs `--open-mode=append` for this; LocalBackend opens
        # output.log with "ab" for the same reason. If the second attempt had
        # truncated the file, the daemon would be reading from an offset past
        # the end and this would go backwards.
        assert world.store.get_offset(world.run_id, world.events_path) > offset_before
        assert world.offsets == sorted(world.offsets), "a byte offset must only ever grow"

    # -- 5. FINISH -----------------------------------------------------

    def test_second_attempt_finishes_and_the_run_goes_terminal(self, world: World):
        """This time the training runs out of steps and ends normally."""
        stage_result(world, "job_id_2")

        world.pump(
            lambda: world.run()["terminal"],
            what="the second attempt to finish and the run to go terminal",
        )

        row = world.run()
        assert row["status"] == "completed"
        assert row["stale"] is False

        # The log's own ending agrees with the scheduler this time: the sidecar
        # exits with the child's code, and the child exited 0.
        #
        # Heartbeats are filtered out before looking at the tail. The sidecar
        # stops the heartbeat thread before writing `run_ended`, but a beat
        # that was already awake can still land in the microseconds between the
        # two. That is harmless -- `run_ended` is the last thing the run *says*
        # -- and asserting on the raw last line would make this test fail once
        # in a few hundred runs for no reason anyone could act on.
        events = [event for event in world.events() if event["type"] != "heartbeat"]
        assert events[-1]["type"] == "run_ended"
        assert events[-1]["payload"] == {"exit_code": 0, "status": "completed"}

        # A terminal run drops out of the daemon's working set, which is what
        # stops it polling a job that ended last Tuesday forever.
        assert world.run_id not in [
            run["run_id"] for run in world.store.list_runs(active_only=True)
        ]

        # The final shape of the story: two `run_started` events (two
        # attempts, numbered 0 and 1), two `run_ended` events (interrupted,
        # then completed), one continuous seq, one continuous metric series.
        started = world.events("run_started")
        assert len(started) == 2
        assert [event["payload"]["attempt"] for event in started] == [0, 1]
        assert [event["payload"]["status"] for event in world.events("run_ended")] == [
            "interrupted",
            "completed",
        ]
        seqs = world.seqs()
        assert seqs == list(range(1, len(seqs) + 1))

    # -- 6. CODA: idempotence ------------------------------------------

    def test_a_fresh_database_catches_up_to_exactly_the_same_state(
        self, world: World, tmp_path
    ):
        """Same bytes in, same database out.

        A second store, empty, is pointed at the same run directory and caught
        up from offset zero in one go -- the situation of a daemon that crashed
        and was restarted, or a laptop that was asleep for the whole run. It
        must arrive at exactly the state the store that watched it live is in.

        This is the promise that lets every other recovery path in relay be
        "just restart it": composite primary keys plus INSERT OR IGNORE mean
        re-reading bytes is a no-op, so there is no repair code anywhere,
        because there is nothing to repair.
        """
        job_id = stage_result(world, "job_id_2")

        replay = Store(world.root / "replay.db")
        try:
            replay.create_run(
                world.run_id,
                backend="local",
                job_id=job_id,
                status="queued",
                run_dir=str(world.run_dir),
            )
            replay_daemon = Daemon(replay, world.backend)

            expected_events = world.store.count_events(world.run_id)

            def catch_up(what: str) -> None:
                deadline = time.monotonic() + TIMEOUT
                while time.monotonic() < deadline:
                    replay_daemon.run_once()
                    if replay.count_events(world.run_id) >= expected_events:
                        return
                    time.sleep(POLL)
                pytest.fail(f"timed out after {TIMEOUT}s waiting for {what}")

            catch_up("the replay database to catch up")

            assert replay.count_events(world.run_id) == expected_events
            assert replay.max_seq(world.run_id) == world.store.max_seq(world.run_id)
            assert replay.count_metrics(world.run_id) == world.store.count_metrics(
                world.run_id
            )
            assert replay.metric_series(world.run_id, "loss") == world.store.metric_series(
                world.run_id, "loss"
            )

            # Now make it re-read the whole file from the beginning, which is
            # what a daemon killed mid-cycle effectively does with the region
            # it had already consumed. `ingest` with no events and an offset of
            # zero rewinds the bookmark through the store's own API; the next
            # cycle then re-reads and re-inserts every line in the file.
            #
            # Nothing changes. Every insert is INSERT OR IGNORE against a
            # composite primary key, so the second pass over the same bytes
            # produces the same database -- which is exactly why there is no
            # recovery code anywhere in relay.
            offset = replay.get_offset(world.run_id, world.events_path)
            replay.ingest(world.run_id, world.events_path, [], 0)
            replay.update_run(world.run_id, status="queued")

            catch_up("the replay database to re-read the file from the start")

            assert replay.get_offset(world.run_id, world.events_path) == offset
            assert replay.count_events(world.run_id) == expected_events
            assert replay.count_metrics(world.run_id) == world.store.count_metrics(
                world.run_id
            )
        finally:
            replay.close()
