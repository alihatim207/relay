"""End to end: the two ways a relay job picks itself up after preemption.

`test_e2e_local.py` tells the whole story once, with a training script that
happens to take a `--state` flag the test itself passes on both attempts. That
proves the *plumbing* -- seq continues, the log appends, the offset only grows
-- but it quietly does the resuming by hand. Nothing in relay chose the
checkpoint.

This file is about the part relay actually chooses. CLAUDE.md offers two
optional resume paths, and relay must work fully with neither:

  PART A -- config-driven, for a script that already checkpoints.
            `resume.checkpoint_glob` + `resume.arg` go into `sidecar.json`.
            On attempt 0 the sidecar does nothing at all. On attempt > 0 it
            globs, throws away anything older than the run's first start,
            takes the newest survivor, and appends `--resume <that file>` to
            the training command. The script never learns relay exists.

  PART B -- `relay_ckpt`, for a script that does not checkpoint.
            The backend copies `relay_ckpt.py` into the run directory and the
            sidecar puts that directory on the child's PYTHONPATH, so
            `import relay_ckpt` works on a compute node with no relay
            installed. `restore()` reads the newest pickle, `tick()` saves on a
            schedule and -- once SIGTERM has landed -- saves one last time and
            raises `SystemExit(0)` so the script's own `finally` still runs.

Both scenarios follow the same six beats as `test_e2e_local`, and for the same
reasons:

  submit -> pump the daemon until metrics flow -> preempt with a group SIGTERM
  -> wait for the wrapper's `exit_code` -> resubmit the SAME run ID into the
  SAME run directory -> pump -> assert the second attempt picked up where the
  first stopped -> let it finish -> assert the run went terminal.

Three rules inherited from that file, because they are what make an
end-to-end test worth having:

  * **Preemption is a bare `os.killpg(SIGTERM)`, never `backend.cancel()`.**
    `cancel` writes `CANCEL_REQUESTED` into the run directory first, and that
    file is precisely how the sidecar tells "the user asked me to stop" from
    "the scheduler took the node". A preemption leaves no such file because
    nobody asked for it. Calling `cancel` here would test the wrong branch.

  * **Every wait is a deadline loop.** `wait_until` / `World.pump` poll until
    the condition actually holds. `sleep(0.5); assert` is a test that fails on
    a loaded CI box and teaches people to ignore it.

  * **Nothing is mocked.** Real subprocesses, real signals, real pickles, real
    SQLite. The bugs these paths can have -- a checkpoint chosen from the wrong
    run, a `finally` that never ran, a `seq` that restarted at 1 -- only exist
    between processes.

And one rule of its own, which the daemon side of both scenarios asserts:

  * **A resumed run is not a broken run.** After the resubmit the run must come
    back to `running`/`completed` from the *backend*, must never be flagged
    stale, and must never make the daemon count a bad line. A resume that
    leaves the run wedged at "queued forever" or spraying unparseable bytes
    into `events.jsonl` would be a worse failure than not resuming at all, and
    neither would show up in an assertion about checkpoints.

Each scenario is an ordered story sharing one class-scoped `World`. Running the
stages out of order (with `-k`) fails loudly rather than passing silently.
"""

from __future__ import annotations

import contextlib
import glob as globlib
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
# Every loop below normally finishes in well under a second.
TIMEOUT = 30.0
POLL = 0.05

# The training loops' pace: fast enough that the whole file runs in seconds,
# slow enough that the daemon gets several cycles of partial output, which is
# where torn lines come from.
STEP_SECONDS = 0.05

# How many more steps the *second* attempt is given past the checkpoint it
# resumed from. Big enough that the resumed series is obviously a continuation
# and not a coincidence, small enough to finish in under a second.
EXTRA_STEPS = 12

# The sidecar's knobs, shrunk.
#   grace 10         -> the time-limit (SIGUSR1) window. Unused here.
#   preempt_grace 5  -> on SIGTERM the child gets 5 - 2 = 3 seconds to
#                       checkpoint before the sidecar SIGKILLs it. Both scripts
#                       below need a fraction of one, and a short window keeps
#                       the test quick without making the margin theoretical.
#   heartbeat 1      -> several heartbeats instead of waiting 30s for one.
GRACE_SECONDS = 10
PREEMPT_GRACE_SECONDS = 5
HEARTBEAT_SECONDS = 1


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

    Cleanup only. Nothing in the tests proper reaches for SIGKILL -- the grace
    window is the thing under test.
    """
    if not job_id:
        return
    try:
        os.killpg(int(job_id), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, ValueError, OverflowError):
        pass


@contextlib.contextmanager
def working_directory(path: Path):
    """Run the body with the process cwd set to `path`, then put it back.

    This exists for Part A and it is not incidental. `resume.checkpoint_glob`
    is documented as relative to *the job's working directory*, and
    `LocalBackend` deliberately inherits relay's cwd when it spawns the wrapper
    (a user typing `relay submit train.py` means the `train.py` in front of
    them). So the sidecar's cwd -- and therefore the directory the glob
    resolves against, and the directory the training script writes its
    checkpoints into -- is whatever this process's cwd was at submit time.

    On Slurm the same thing is arranged by `#SBATCH --chdir`. Locally the test
    has to arrange it, and doing so explicitly around the `submit` call is
    clearer than a fixture that leaves the whole class chdir'd somewhere odd.
    """
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def stage_result(world, attribute: str):
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
# The world both scenarios happen in
# --------------------------------------------------------------------------


class World:
    """Paths, store, backend, daemon -- plus the facts stages hand each other.

    Shared by both scenarios. The only thing that differs between them is the
    `JobSpec` each builds, which is the point: the two resume paths are two
    configurations of one system, not two systems.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root
        self.run_id = run_id
        self.run_dir = root / "runs" / run_id

        self.store = Store(root / "relay.db")
        # `root=` keeps the backend's pid -> run_dir index inside tmp_path. No
        # config file is loaded anywhere in this file and nothing touches the
        # real home directory.
        self.backend = LocalBackend(root=root / "local")
        self.daemon = Daemon(self.store, self.backend)

        # Spelled the way the daemon spells it: `sources` is keyed on
        # (run_id, path), so a different spelling would silently start a second
        # byte offset at zero.
        self.events_path = os.path.join(str(self.run_dir), EVENTS_FILENAME)

        # Filled in as the story goes.
        self.job_id_1: str | None = None
        self.job_id_2: str | None = None
        self.checkpoint_step: int | None = None
        self.checkpoint_path: str | None = None
        self.seq_after_preempt: int | None = None
        self.offsets: list[int] = []

        # Every cycle's bad-line count, summed. See `cycle`.
        self.bad_lines_total = 0
        self.cycles_run = 0

    # -- driving the daemon by hand ------------------------------------

    def cycle(self):
        """One daemon pass, with the two invariants that hold on every pass.

        `bad_lines` is asserted here rather than at the end of a stage so that
        a failure points at the cycle that produced it. A bad line means the
        daemon read bytes out of `events.jsonl` that `parse_event` rejected,
        and on a resume path the obvious way to produce one is for two attempts
        to interleave writes -- exactly the failure that would otherwise hide
        behind a perfectly healthy-looking checkpoint assertion.
        """
        stats = self.daemon.run_once()
        assert stats.bad_lines == 0, (
            f"cycle {self.cycles_run} parsed {stats.bad_lines} unparseable line(s) "
            f"out of {self.events_path}; the two attempts' writes have collided"
        )
        self.bad_lines_total += stats.bad_lines
        self.cycles_run += 1
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

    def metric_steps(self) -> list[int]:
        return [point["step"] for point in self.store.metric_series(self.run_id, "loss")]

    def relay_log_lines(self) -> list[str]:
        """The log lines the *sidecar* wrote about itself, not the child's.

        `stream` is "relay" rather than "stdout"/"stderr" precisely so these
        are distinguishable: "relay" is not a pipe.
        """
        return [
            event["payload"]["line"]
            for event in self.events("log")
            if event["payload"].get("stream") == "relay"
        ]

    def stdout_lines(self) -> list[str]:
        return [
            event["payload"]["line"]
            for event in self.events("log")
            if event["payload"].get("stream") == "stdout"
        ]

    # -- the two halves the CLI would normally join --------------------

    def register(self, job_id: str, name: str) -> None:
        """Put the `queued` row in the store, as `relay submit` would.

        A backend never touches the database and the store never starts a
        process; the CLI is what joins them, and here the test plays that part.
        """
        self.store.create_run(
            self.run_id,
            backend="local",
            name=name,
            job_id=job_id,
            status="queued",
            run_dir=str(self.run_dir),
        )

    def requeue_row(self, job_id: str) -> None:
        """Put the row back in the daemon's working set under a new job ID.

        Status is scheduler-owned, and locally we are standing in for the
        scheduler: on Slurm a preempted job goes back to PENDING by itself.
        Nothing else about the run changes -- not the run ID, not the run
        directory, and crucially not the stored byte offset.
        """
        self.store.update_run(self.run_id, status="queued", job_id=job_id)

    # -- assertions both scenarios make --------------------------------

    def assert_resume_did_not_look_like_a_failure(self) -> None:
        """The run came back healthy: not stale, not wedged, no bad lines.

        Stale detection compares the last heartbeat against a five-minute
        threshold, and a resumed attempt writes heartbeats from a brand new
        process into the same log. If the daemon tracked heartbeats per
        *attempt* rather than per run -- or if a resubmit reset the timestamp
        -- a healthy run would light up as stale the moment it came back, and
        the user would be told their job is hung while it trains happily.
        """
        row = self.run()
        assert row["stale"] is False, "a resumed run must not be flagged stale"
        assert row["status"] in ("running", "completed"), (
            f"the run is stuck at {row['status']!r}: after a resubmit the "
            f"backend's answer should have moved it on"
        )
        assert self.bad_lines_total == 0


# --------------------------------------------------------------------------
# PART A -- config-driven resume
# --------------------------------------------------------------------------
#
# The script below is written the way CLAUDE.md says a real user's script is:
# no relay import anywhere in it. It reports metrics by printing `##relay##`
# lines, writes its own checkpoints with tmp + os.replace, announces them with
# a `##relay##` checkpoint line, takes a `--resume PATH` of its own invention,
# and treats SIGTERM as "save and stop".
#
# The checkpoint directory is `ckpt/`, *relative*, which is the whole point of
# Part A: the user's glob `ckpt/*.json` and the script's own output directory
# both resolve against the job's working directory, and neither of them has
# ever heard of `run_dir`.
#
# File names are zero-padded (`step-000042.json`) so that lexical order and
# numeric order agree. The sidecar breaks mtime ties by path, and a filesystem
# with one-second mtime granularity makes ties entirely possible; padding turns
# "the tie-break might pick step-9 over step-12" into a non-question.

TRAIN_SCRIPT_CONFIG_DRIVEN = r'''
import argparse
import json
import os
import signal
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-dir", default="ckpt")
parser.add_argument("--max-steps", type=int, default=1000000)
parser.add_argument("--ckpt-every", type=int, default=3)
parser.add_argument("--step-seconds", type=float, default=0.05)
# relay appends this on attempt > 0, built from the user's own `resume.arg`.
# The script has no idea relay exists; as far as it knows, a human typed it.
parser.add_argument("--resume", default=None)
args = parser.parse_args()


def save(step):
    """Write the checkpoint atomically, then tell relay it happened."""
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(args.ckpt_dir, "step-%06d.json" % step))
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({"step": step}, handle)
    # A `.tmp` never matches `ckpt/*.json`, so a half-written checkpoint is
    # invisible to the resume glob by construction rather than by luck.
    os.replace(tmp, path)
    print("##relay## " + json.dumps({"checkpoint": path, "step": step}), flush=True)


start = 0
if args.resume:
    with open(args.resume, "r") as handle:
        start = int(json.load(handle)["step"])
print("script resuming from %s at step %d" % (args.resume, start), flush=True)

current = start
_handled = False


def on_term(signum, frame):
    """Preemption warning: checkpoint, then get out of the way.

    Idempotent, because the README says it has to be: the scheduler signals
    the whole job step and the sidecar forwards SIGTERM as well, so this can
    be entered twice.
    """
    global _handled
    if _handled:
        return
    _handled = True
    save(current)
    # The checkpoint line is in the pipe; the sidecar's reader thread still has
    # to read it. 0.25s of politeness, well inside the 3s child window.
    time.sleep(0.25)
    os._exit(0)


signal.signal(signal.SIGTERM, on_term)

step = start
while step < args.max_steps:
    current = step
    print("##relay## " + json.dumps({"step": step, "loss": round(1.0 / (1.0 + step), 6)}),
          flush=True)
    if step != start and step % args.ckpt_every == 0:
        save(step)
    time.sleep(args.step_seconds)
    step += 1

print("training complete", flush=True)
'''

CKPT_EVERY = 3

# A checkpoint left behind by some *earlier*, unrelated run in the same working
# directory. Its step number is absurdly high and its name sorts last, so if
# the sidecar's mtime filter were missing -- or if every file's mtime tied on a
# coarse filesystem -- this is the file that would win, and the resumed run
# would silently continue from step 999999 with plausible-looking metrics.
# Its mtime is set an hour into the past, which is what makes it lose.
STALE_CKPT_STEP = 999999
STALE_CKPT_AGE_SECONDS = 3600


class ConfigDrivenWorld(World):
    """A `World` whose job runs in its own working directory.

    `job_dir` is deliberately *not* `run_dir`. The run directory belongs to
    relay (events.jsonl, sidecar.py, sidecar.json); the working directory
    belongs to the user's script. Keeping them apart is the only way to notice
    if the glob accidentally started resolving against the wrong one.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root, run_id="vr_s0_resumeA")

        self.script = root / "train_config_driven.py"
        self.script.write_text(TRAIN_SCRIPT_CONFIG_DRIVEN, encoding="utf-8")

        # realpath, not just absolute: on macOS the temp directory is reached
        # through a symlink (/var -> /private/var), and `os.getcwd()` always
        # reports the physical path. The sidecar builds `resumed_from` from
        # `os.getcwd()`, so the test has to compare against the same spelling.
        job_dir = root / "job"
        job_dir.mkdir(parents=True, exist_ok=True)
        self.job_dir = Path(os.path.realpath(job_dir))

        self.ckpt_dir = self.job_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.stale_ckpt = self.ckpt_dir / f"step-{STALE_CKPT_STEP:06d}.json"
        self.stale_ckpt.write_text(json.dumps({"step": STALE_CKPT_STEP}), encoding="utf-8")
        stale_when = time.time() - STALE_CKPT_AGE_SECONDS
        os.utime(self.stale_ckpt, (stale_when, stale_when))

    def spec(self, *, max_steps: int) -> JobSpec:
        """A JobSpec for this run. Called twice: once per attempt.

        What does NOT change between the two: `run_id`, `run_dir`, and the
        resume pair. A requeued Slurm job keeps relay's run ID, lands back in
        the same directory and re-reads the same `sidecar.json`. Only the
        scheduler's job ID changes, which is why nothing in relay is keyed on
        it.

        What is NOT passed: any mention of a checkpoint on the command line.
        `--resume` appears on attempt 1 only, and the sidecar is what puts it
        there.
        """
        return JobSpec(
            run_id=self.run_id,
            command=[
                sys.executable,
                str(self.script),
                "--ckpt-dir",
                "ckpt",  # relative, like the glob: both mean the job's cwd
                "--max-steps",
                str(max_steps),
                "--ckpt-every",
                str(CKPT_EVERY),
                "--step-seconds",
                str(STEP_SECONDS),
            ],
            run_dir=str(self.run_dir),
            name="resume-config",
            grace_seconds=GRACE_SECONDS,
            preempt_grace=PREEMPT_GRACE_SECONDS,
            heartbeat_seconds=HEARTBEAT_SECONDS,
            resume_glob="ckpt/*.json",
            resume_arg="--resume {path}",
        )

    def submit(self, *, max_steps: int) -> str:
        """Submit from inside the job's working directory. See `working_directory`."""
        with working_directory(self.job_dir):
            return self.backend.submit(self.spec(max_steps=max_steps))


@pytest.fixture(scope="class")
def world_a(tmp_path_factory):
    """One world for the whole Part A story. Class-scoped: the stages share it."""
    w = ConfigDrivenWorld(tmp_path_factory.mktemp("resume_config"))
    try:
        yield w
    finally:
        kill_group(w.job_id_1)
        kill_group(w.job_id_2)
        w.store.close()


class TestConfigDrivenResume:
    """`resume.checkpoint_glob` + `resume.arg`, for a script that checkpoints."""

    # -- 1. SUBMIT + MONITOR -------------------------------------------

    def test_attempt_zero_starts_from_scratch_and_says_nothing_about_resume(
        self, world_a: ConfigDrivenWorld
    ):
        """Attempt 0 must behave as if the resume feature did not exist.

        That is the first half of "relay works fully with neither path": a run
        that is never preempted should not be able to tell that `resume` was
        configured. No glob, no argument, no log line, `resumed_from` null --
        even though `sidecar.json` is sitting right there with the pair in it.
        """
        world_a.job_id_1 = world_a.submit(max_steps=1_000_000)  # effectively forever
        assert world_a.job_id_1.isdigit(), "a local job ID is a PID"
        world_a.register(world_a.job_id_1, name="resume-config")

        # The backend wrote the resume pair down where the sidecar will find it.
        config = json.loads((world_a.run_dir / "sidecar.json").read_text())
        assert config["resume"] == {
            "checkpoint_glob": "ckpt/*.json",
            "arg": "--resume {path}",
        }
        assert config["first_attempt_start"] > 0

        def flowing():
            return (
                world_a.run()["status"] == "running"
                and len(world_a.events("metric")) >= CKPT_EVERY + 3
                and len(world_a.events("checkpoint")) >= 1
            )

        world_a.pump(flowing, what="attempt 0's metrics and first checkpoint")

        started = world_a.events("run_started")
        assert len(started) == 1
        assert started[0]["payload"]["attempt"] == 0
        assert started[0]["payload"]["resumed_from"] is None, (
            "attempt 0 has nothing to resume from and must not pretend otherwise"
        )
        assert "--resume" not in started[0]["payload"]["command"], (
            "the sidecar must not touch the command on the first attempt"
        )
        # Not even the "no checkpoint matched" warning: on attempt 0 the
        # sidecar does not look, so there is nothing to warn about.
        assert not any("resuming from" in line for line in world_a.relay_log_lines())
        assert not any("no checkpoint matched" in line for line in world_a.relay_log_lines())

        # The script wrote its checkpoints where the glob will look: relative
        # to the job's working directory, which is not the run directory.
        assert world_a.ckpt_dir.is_dir()
        assert not (world_a.run_dir / "ckpt").exists()
        world_a.assert_resume_did_not_look_like_a_failure()

    # -- 2. PREEMPT ----------------------------------------------------

    def test_preemption_leaves_a_checkpoint_and_an_interrupted_ending(
        self, world_a: ConfigDrivenWorld
    ):
        """A group SIGTERM stands in for Slurm evicting a preempted job.

        Sent by hand to the whole process group -- wrapper, sidecar and
        training script -- which is as close as a laptop gets to being evicted
        from a preemptible partition. Deliberately not `backend.cancel()`: that
        writes `CANCEL_REQUESTED` first, which is the *cancel* branch.
        """
        job_id = stage_result(world_a, "job_id_1")

        os.killpg(int(job_id), signal.SIGTERM)

        wait_until(
            lambda: (world_a.run_dir / EXIT_CODE_FILENAME).is_file(),
            what="the job to die and record its exit code",
        )
        world_a.pump(
            lambda: world_a.events("run_ended"),
            what="the interrupted ending to be ingested",
        )

        signals = world_a.events("signal")
        assert signals[-1]["payload"] == {"signal": "SIGTERM", "reason": "term"}
        ended = world_a.events("run_ended")[-1]
        assert ended["payload"]["status"] == "interrupted"
        assert not (world_a.run_dir / "CANCEL_REQUESTED").exists(), (
            "a preemption leaves no cancel file; nobody asked for it"
        )

        # The checkpoint the *signal handler* wrote is the one attempt 1 should
        # be handed. This is the contract that makes preemption survivable and
        # the reason the sidecar forwards SIGTERM instead of killing outright.
        checkpoints = world_a.events("checkpoint")
        last = checkpoints[-1]["payload"]
        world_a.checkpoint_step = last["step"]
        world_a.checkpoint_path = last["path"]
        assert json.loads(Path(last["path"]).read_text())["step"] == last["step"]
        assert Path(last["path"]).parent == world_a.ckpt_dir

        world_a.seq_after_preempt = world_a.store.max_seq(world_a.run_id)
        assert world_a.seq_after_preempt == len(world_a.seqs())

    # -- 3. RESUME -----------------------------------------------------

    def test_attempt_one_is_handed_the_newest_checkpoint(
        self, world_a: ConfigDrivenWorld
    ):
        """The sidecar globs, picks, appends -- and the script starts there.

        Everything asserted here happened without the test telling anyone which
        file to use. The resubmitted `JobSpec` carries the same `resume_glob`
        it always did and no checkpoint path at all; `SLURM_RESTART_COUNT` goes
        from 0 to 1 because `LocalBackend` bumps its attempt counter; and that
        1 is the entire reason the sidecar looks this time.
        """
        previous_max_seq = stage_result(world_a, "seq_after_preempt")
        checkpoint_step = stage_result(world_a, "checkpoint_step")
        checkpoint_path = stage_result(world_a, "checkpoint_path")
        offset_before = world_a.store.get_offset(world_a.run_id, world_a.events_path)

        world_a.job_id_2 = world_a.submit(max_steps=checkpoint_step + EXTRA_STEPS)
        assert world_a.job_id_2 != world_a.job_id_1, "a requeued job gets a new job ID"
        world_a.requeue_row(world_a.job_id_2)

        # `first_attempt_start` survived the resubmit. If it had been reset to
        # this submit's `time.time()`, every checkpoint attempt 0 wrote would
        # now be "too old" and the run would restart from zero forever.
        config = json.loads((world_a.run_dir / "sidecar.json").read_text())
        assert config["first_attempt_start"] < os.stat(checkpoint_path).st_mtime

        world_a.pump(
            lambda: len(world_a.events("run_started")) == 2
            and [
                event
                for event in world_a.events("metric")
                if event["seq"] > previous_max_seq
            ],
            what="attempt 1 to start and produce a metric",
        )

        started = world_a.events("run_started")[-1]
        assert started["payload"]["attempt"] == 1
        assert started["payload"]["resumed_from"] == checkpoint_path, (
            "the sidecar should have chosen the newest checkpoint of attempt 0"
        )
        # The chosen path really did reach the child's argv.
        assert started["payload"]["command"][-2:] == ["--resume", checkpoint_path]

        # ...and it said so, in a log event tagged as relay's own rather than
        # the child's. A silent resume is unreviewable.
        notes = [line for line in world_a.relay_log_lines() if "resuming from" in line]
        assert notes, "the sidecar must log the checkpoint it resumed from"
        assert checkpoint_path in notes[-1]
        assert "ckpt/*.json" in notes[-1]

        # The script agrees: its first metric of attempt 1 is the checkpoint's
        # step, not zero. This is the assertion that would catch a resume that
        # "worked" on paper but handed the script a file it could not read.
        resumed = [
            event
            for event in world_a.events("metric")
            if event["seq"] > previous_max_seq
        ]
        assert resumed[0]["payload"]["step"] == checkpoint_step

        # seq continued rather than restarting at 1. A restart would collide
        # with the first attempt's keys and, because every insert is INSERT OR
        # IGNORE, the second attempt's events would vanish without an error.
        seqs = world_a.seqs()
        assert seqs == list(range(1, len(seqs) + 1))
        assert world_a.store.get_offset(world_a.run_id, world_a.events_path) > offset_before
        assert world_a.offsets == sorted(world_a.offsets), "a byte offset must only grow"

        world_a.assert_resume_did_not_look_like_a_failure()

    # -- 4. THE STALE CHECKPOINT ---------------------------------------

    def test_a_checkpoint_older_than_the_run_is_never_chosen(
        self, world_a: ConfigDrivenWorld
    ):
        """A file left in `ckpt/` by an earlier run must not resume this one.

        `step-999999.json` has been sitting in the job's working directory
        since before this run was submitted. It matches the glob and it holds
        the highest step, so a resume that picked "the biggest number" -- or
        the first match, or an arbitrary one -- would silently continue
        training from a step that never happened here, with metrics that look
        entirely plausible. That is worse than not resuming at all.

        Its name also sorts last, which matters because the sidecar breaks
        mtime ties on `(mtime, path)`: on a filesystem with one-second
        timestamp granularity, or a run directory that has been copied about,
        this is the file that wins a tie.

        Honest about the limit of what this proves: while attempt 0 does write
        a checkpoint, the `first_attempt_start` filter is belt to the mtime
        comparison's braces, since a file an hour old loses on mtime anyway.
        The filter is the only defence in the case this scenario cannot also
        cover -- an attempt preempted *before* its first save -- so what is
        asserted here is the precondition (the stale file really is older than
        the run) plus the outcome (it was never chosen, and its step never
        reached the series).
        """
        checkpoint_path = stage_result(world_a, "checkpoint_path")

        # The trap is armed: it is genuinely a candidate the glob returns.
        with working_directory(world_a.job_dir):
            matches = globlib.glob("ckpt/*.json")
        assert world_a.stale_ckpt.name in {Path(m).name for m in matches}
        assert world_a.stale_ckpt.is_file()

        # ...and it is genuinely on the wrong side of the filter, while the
        # checkpoint that *was* chosen is on the right side of it.
        config = json.loads((world_a.run_dir / "sidecar.json").read_text())
        first_start = config["first_attempt_start"]
        assert os.stat(world_a.stale_ckpt).st_mtime < first_start
        assert os.stat(checkpoint_path).st_mtime >= first_start

        started = world_a.events("run_started")[-1]
        assert started["payload"]["resumed_from"] != str(world_a.stale_ckpt)
        assert started["payload"]["resumed_from"] == checkpoint_path
        assert all(
            point < STALE_CKPT_STEP for point in world_a.metric_steps()
        ), "a step from the stale checkpoint leaked into the series"

    # -- 5. FINISH -----------------------------------------------------

    def test_the_resumed_run_finishes_and_the_history_shows_two_attempts(
        self, world_a: ConfigDrivenWorld
    ):
        """This time training runs out of steps and the run goes terminal."""
        stage_result(world_a, "job_id_2")
        checkpoint_step = stage_result(world_a, "checkpoint_step")
        checkpoint_path = stage_result(world_a, "checkpoint_path")

        world_a.pump(
            lambda: world_a.run()["terminal"],
            what="attempt 1 to finish and the run to go terminal",
        )

        row = world_a.run()
        assert row["status"] == "completed"
        assert row["stale"] is False
        # Counted from `run_started` events at ingest: two attempts, so two.
        assert row["attempts"] == 2

        # `relay show` reads this: one row per attempt, with the checkpoint
        # each one picked up from.
        history = world_a.store.attempt_history(world_a.run_id)
        assert [entry["attempt"] for entry in history] == [0, 1]
        assert [entry["resumed_from"] for entry in history] == [None, checkpoint_path]
        assert history[0]["seq"] < history[1]["seq"]

        # One continuous series across both attempts. `metrics` is keyed on
        # (run_id, name, step), so the step both attempts emitted at the resume
        # boundary is de-duplicated for free.
        steps = world_a.metric_steps()
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))
        assert steps[0] == 0, "attempt 0's points are still there"
        assert steps == list(range(checkpoint_step + EXTRA_STEPS))

        assert [event["payload"]["status"] for event in world_a.events("run_ended")] == [
            "interrupted",
            "completed",
        ]
        seqs = world_a.seqs()
        assert seqs == list(range(1, len(seqs) + 1))
        assert world_a.bad_lines_total == 0, (
            f"the daemon counted {world_a.bad_lines_total} bad line(s) across "
            f"{world_a.cycles_run} cycles"
        )


# --------------------------------------------------------------------------
# PART B -- the relay_ckpt helper
# --------------------------------------------------------------------------
#
# The script below is the other kind of user: someone who has no checkpointing
# at all and does not want to write any. Three lines buy them preemption
# survival, and none of the three mentions a path, a directory or a run ID.
#
# `import relay_ckpt` works because `LocalBackend` copied `relay_ckpt.py` into
# the run directory at submit and the sidecar prepended that directory to the
# child's PYTHONPATH. Neither relay nor its package is importable from this
# script's point of view -- which is the situation on a compute node.
#
# The `finally` block is the assertion in script form. `tick` raises
# `SystemExit(0)` after its signal-triggered save, and SystemExit is an
# exception: `finally` blocks and context managers still run. If `tick` called
# `os._exit` instead, this marker would never be written, and a real user's
# "close the dataset, flush the logger" cleanup would be silently skipped on
# every preemption.

TRAIN_SCRIPT_RELAY_CKPT = r'''
import json
import os
import sys
import time

import relay_ckpt

total = int(sys.argv[1])
marker = sys.argv[2]

# Printed as an ordinary stdout line, so the test can read back -- from the
# event log, the way a user would -- what the child actually saw. These two
# variables are what make a requeued attempt land in the same checkpoint
# directory as the one before it.
print("relay-env RELAY_RUN_ID=%s RELAY_RUN_DIR=%s"
      % (os.environ.get("RELAY_RUN_ID"), os.environ.get("RELAY_RUN_DIR")), flush=True)

# Step counts, not minutes: a test cannot wait ten minutes for a save. keep=2
# is the default and is also what makes `restore`'s fallback-to-next-oldest
# possible at all.
relay_ckpt.configure(every_steps=3, keep=2)

state = relay_ckpt.restore(lambda: {"acc": 0})
print("restored at step %d" % relay_ckpt.step, flush=True)

try:
    while relay_ckpt.step < total:
        step = relay_ckpt.step
        state["acc"] += 1
        print("##relay## " + json.dumps({"step": step, "loss": round(1.0 / (1.0 + step), 6)}),
              flush=True)
        # tick() counts the iteration, saves if due, and -- if SIGTERM has
        # arrived -- saves and raises SystemExit(0).
        relay_ckpt.tick(state)
        time.sleep(__STEP_SECONDS__)
    print("training complete", flush=True)
finally:
    with open(marker, "a") as handle:
        handle.write("finally ran at step %d\n" % relay_ckpt.step)
        handle.flush()
        os.fsync(handle.fileno())
'''.replace("__STEP_SECONDS__", repr(STEP_SECONDS))

CKPT_EVERY_STEPS = 3
CKPT_KEEP = 2


class RelayCkptWorld(World):
    """A `World` whose job checkpoints through `relay_ckpt`."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, run_id="vr_s0_resumeB")

        self.script = root / "train_relay_ckpt.py"
        self.script.write_text(TRAIN_SCRIPT_RELAY_CKPT, encoding="utf-8")

        # Written by the script's `finally`, one line per attempt.
        self.marker = root / "finally.log"

        # Where relay_ckpt puts its pickles with no `directory=` configured:
        # $RELAY_RUN_DIR/checkpoints. The script never names this path.
        self.ckpt_dir = self.run_dir / "checkpoints"

    def marker_lines(self) -> list[str]:
        try:
            return [
                line for line in self.marker.read_text(encoding="utf-8").splitlines() if line
            ]
        except OSError:
            return []

    def pickles(self) -> list[Path]:
        return sorted(self.ckpt_dir.glob("ckpt_*.pkl"))

    def spec(self, *, total: int) -> JobSpec:
        """A JobSpec with no resume pair at all.

        Part B needs none: the checkpoint directory comes from `RELAY_RUN_DIR`,
        which the sidecar sets identically on every attempt, and `restore()`
        finds the newest pickle by itself. There is nothing for relay to put on
        the command line, which is exactly why this path exists for scripts
        that would rather not grow a `--resume` flag.
        """
        return JobSpec(
            run_id=self.run_id,
            command=[sys.executable, str(self.script), str(total), str(self.marker)],
            run_dir=str(self.run_dir),
            name="resume-ckpt",
            grace_seconds=GRACE_SECONDS,
            preempt_grace=PREEMPT_GRACE_SECONDS,
            heartbeat_seconds=HEARTBEAT_SECONDS,
        )


@pytest.fixture(scope="class")
def world_b(tmp_path_factory):
    """One world for the whole Part B story. Class-scoped: the stages share it."""
    w = RelayCkptWorld(tmp_path_factory.mktemp("resume_ckpt"))
    try:
        yield w
    finally:
        kill_group(w.job_id_1)
        kill_group(w.job_id_2)
        w.store.close()


class TestRelayCkptResume:
    """`relay_ckpt`, for a script that does not checkpoint on its own."""

    # -- 1. SUBMIT + MONITOR -------------------------------------------

    def test_the_helper_is_importable_and_saves_into_the_run_directory(
        self, world_b: RelayCkptWorld
    ):
        """`import relay_ckpt` works, and the pickles land under RELAY_RUN_DIR.

        Two things are being proved at once, and both are about a compute node
        where relay is not installed: the backend copied the helper next to the
        job, and the sidecar put that directory on the child's PYTHONPATH.
        Either one missing and the script dies on line one with an ImportError.
        """
        world_b.job_id_1 = world_b.backend.submit(world_b.spec(total=1_000_000))
        world_b.register(world_b.job_id_1, name="resume-ckpt")

        assert (world_b.run_dir / "relay_ckpt.py").is_file(), (
            "the backend must copy the helper beside the job"
        )

        # Two checkpoints, so the resume has a newest one to prefer and the
        # `keep` bound has something to prune.
        world_b.pump(
            lambda: world_b.run()["status"] == "running"
            and len(world_b.events("checkpoint")) >= 2,
            what="relay_ckpt's first two periodic saves",
        )

        # The environment the child actually saw, read back out of the log.
        env_lines = [line for line in world_b.stdout_lines() if line.startswith("relay-env ")]
        assert env_lines, "the script's environment line never reached the log"
        assert f"RELAY_RUN_ID={world_b.run_id}" in env_lines[0]
        assert f"RELAY_RUN_DIR={world_b.run_dir}" in env_lines[0]

        # Pickles on disk, at the steps the events announced, under the
        # directory relay_ckpt derived from RELAY_RUN_DIR without being told.
        checkpoints = world_b.events("checkpoint")
        assert world_b.ckpt_dir.is_dir()
        for event in checkpoints:
            assert Path(event["payload"]["path"]).parent == world_b.ckpt_dir
            assert event["payload"]["step"] % CKPT_EVERY_STEPS == 0
        assert world_b.pickles(), "relay_ckpt wrote no .pkl files"
        # `.tmp` files are written and replaced; none should survive a
        # successful save, and `restore` would ignore one anyway.
        assert list(world_b.ckpt_dir.glob("*.tmp")) == []

        assert world_b.events("metric"), "no metrics from attempt 0"
        assert world_b.run()["stale"] is False

    # -- 2. PREEMPT ----------------------------------------------------

    def test_sigterm_saves_once_more_and_lets_the_finally_block_run(
        self, world_b: RelayCkptWorld
    ):
        """SystemExit(0), not `os._exit`: the script's cleanup still happens.

        `tick` raises `SystemExit(0)` after its signal-triggered save. That is a
        deliberate choice with a visible consequence -- `finally` blocks and
        context managers run on the way out -- and the marker file is how this
        test sees it. A helper that called `os._exit` here would pass every
        checkpoint assertion in this file and silently skip a real user's
        "flush the logger, close the dataset" cleanup on every preemption.
        """
        job_id = stage_result(world_b, "job_id_1")
        assert world_b.marker_lines() == [], "the script has not exited yet"

        os.killpg(int(job_id), signal.SIGTERM)

        wait_until(
            lambda: (world_b.run_dir / EXIT_CODE_FILENAME).is_file(),
            what="the job to die and record its exit code",
        )
        world_b.pump(
            lambda: world_b.events("run_ended"),
            what="the interrupted ending to be ingested",
        )

        markers = world_b.marker_lines()
        assert len(markers) == 1, (
            "the training script's finally block did not run exactly once; "
            "SystemExit(0) from tick() is what is supposed to make it run"
        )

        ended = world_b.events("run_ended")[-1]
        assert ended["payload"]["status"] == "interrupted"
        assert ended["payload"]["exit_code"] is None
        assert world_b.events("signal")[-1]["payload"]["reason"] == "term"
        assert not (world_b.run_dir / "CANCEL_REQUESTED").exists()

        # The signal-triggered save is the newest checkpoint, and its line made
        # it into the log: the sidecar drains the reader threads *before* it
        # writes `run_ended`, precisely so the last checkpoint line is not lost
        # in a pipe buffer.
        checkpoints = world_b.events("checkpoint")
        world_b.checkpoint_step = max(event["payload"]["step"] for event in checkpoints)
        world_b.checkpoint_path = checkpoints[-1]["payload"]["path"]
        assert world_b.checkpoint_step > 0
        assert Path(world_b.checkpoint_path).is_file()
        assert f"step {world_b.checkpoint_step}" in markers[0]

        world_b.seq_after_preempt = world_b.store.max_seq(world_b.run_id)

    # -- 3. RESUME -----------------------------------------------------

    def test_attempt_one_restores_the_step_it_was_preempted_at(
        self, world_b: RelayCkptWorld
    ):
        """`restore()` finds the newest pickle and the counter comes back with it.

        Nothing on the command line changed. The only reason attempt 1 lands in
        the same checkpoint directory is `RELAY_RUN_DIR`, which the sidecar
        derives from the run directory and is therefore byte-identical across
        attempts -- unlike the Slurm job ID, which is not.
        """
        previous_max_seq = stage_result(world_b, "seq_after_preempt")
        checkpoint_step = stage_result(world_b, "checkpoint_step")

        total = checkpoint_step + EXTRA_STEPS
        world_b.job_id_2 = world_b.backend.submit(world_b.spec(total=total))
        assert world_b.job_id_2 != world_b.job_id_1
        world_b.requeue_row(world_b.job_id_2)

        world_b.pump(
            lambda: len(world_b.events("run_started")) == 2
            and [
                event
                for event in world_b.events("metric")
                if event["seq"] > previous_max_seq
            ],
            what="attempt 1 to start and produce a metric",
        )

        started = world_b.events("run_started")[-1]
        assert started["payload"]["attempt"] == 1
        # No config-driven resume here, so this stays null even on attempt 1.
        # The two paths are independent, and this one leaves no trace on the
        # command line at all.
        assert started["payload"]["resumed_from"] is None
        assert "--resume" not in started["payload"]["command"]

        resumed = [
            event
            for event in world_b.events("metric")
            if event["seq"] > previous_max_seq
        ]
        assert resumed[0]["payload"]["step"] == checkpoint_step, (
            "attempt 1 should continue from the checkpoint, not from zero"
        )
        assert resumed[0]["payload"]["step"] > 0

        # The script said so too, in its own words.
        assert f"restored at step {checkpoint_step}" in world_b.stdout_lines()

        seqs = world_b.seqs()
        assert seqs == list(range(1, len(seqs) + 1))
        world_b.assert_resume_did_not_look_like_a_failure()

    # -- 4. FINISH -----------------------------------------------------

    def test_the_resumed_run_finishes_contiguous_pruned_and_never_stale(
        self, world_b: RelayCkptWorld
    ):
        """One continuous series, `keep` pickles, two attempts, no false alarms."""
        stage_result(world_b, "job_id_2")
        checkpoint_step = stage_result(world_b, "checkpoint_step")
        total = checkpoint_step + EXTRA_STEPS

        world_b.pump(
            lambda: world_b.run()["terminal"],
            what="attempt 1 to finish and the run to go terminal",
        )

        row = world_b.run()
        assert row["status"] == "completed"
        assert row["stale"] is False
        assert row["attempts"] == 2

        # The two attempts' metrics are one series, contiguous from zero, with
        # no holes. Steps at or before the resume point may legitimately be
        # emitted twice -- the second attempt re-does the work that was in
        # flight when the node was taken -- and (run_id, name, step) folds
        # those together, which is why "no duplicates" is a fact about the
        # store rather than about the training script.
        steps = world_b.metric_steps()
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))
        assert steps == list(range(total))

        # Pruning: `keep=2`, so however many saves happened, two survive.
        assert 1 <= len(world_b.pickles()) <= CKPT_KEEP
        assert list(world_b.ckpt_dir.glob("*.tmp")) == []

        # Both `finally` blocks ran: one on the preemption, one on the clean
        # ending.
        assert len(world_b.marker_lines()) == 2

        assert [event["payload"]["status"] for event in world_b.events("run_ended")] == [
            "interrupted",
            "completed",
        ]
        history = world_b.store.attempt_history(world_b.run_id)
        assert [entry["attempt"] for entry in history] == [0, 1]
        assert [entry["resumed_from"] for entry in history] == [None, None]

        assert world_b.bad_lines_total == 0, (
            f"the daemon counted {world_b.bad_lines_total} bad line(s) across "
            f"{world_b.cycles_run} cycles"
        )
        assert world_b.run_id not in [
            run["run_id"] for run in world_b.store.list_runs(active_only=True)
        ]
