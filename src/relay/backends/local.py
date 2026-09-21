"""Run jobs on this machine as plain subprocesses.

`LocalBackend` is how relay works without a cluster, and it is how the entire
test suite runs in CI. It is also a rehearsal for `SlurmBackend`: it does the
same things in the same order -- make a run directory, copy `sidecar.py` into
it, start the sidecar, remember an opaque job ID -- just with `fork`/`exec`
where Slurm would have `sbatch` over SSH. Where a choice here exists only to
mirror the Slurm one, the comment says so.

The job ID is the PID of the process relay started, as a string.

These files in the run directory belong to this backend rather than to the
sidecar:

  `exit_code`         the sidecar's exit status, written by a tiny `sh` wrapper
                      once the sidecar has finished -- or, if one of the spec's
                      `setup` lines failed, that line's status, written before
                      the sidecar was ever started.
  `job.pid`           the wrapper's PID, for a human with a terminal.
  `sidecar.json`      what the sidecar has to know before a signal arrives:
                      whether to requeue itself on SIGTERM, how many seconds
                      the preemption window is, when the run's first attempt
                      started, and the optional resume-after-preemption pair.
  `relay_ckpt.py`     a copy of relay's optional checkpoint helper, so a
                      training script can import it without relay being
                      installed. Only present when the installation has one.
  `attempt`           how many attempts have been started in this directory, so
                      a resubmit can set `SLURM_RESTART_COUNT` the way Slurm
                      would.
  `CANCEL_REQUESTED`  written by `cancel` before it signals, so the sidecar can
                      tell a cancel from a preemption.

`exit_code` exists because of a problem that has no equivalent on Slurm. There,
`sacct` remembers a job's exit code long after the job is gone, so a daemon
that restarts can still ask. Here there is no `sacct`: the only record of a
subprocess's exit status is the `Popen` object held by the process that spawned
it, and that object dies with the process. A daemon restarted five minutes
later has no way to ask the kernel "what did PID 41234 exit with?" -- the answer
was consumed when the process was reaped. So we write it down.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from relay import config
from relay.backends.base import Backend, JobSpec, UsageRow

# Named after the module so a user can silence or raise just this backend's
# noise with `logging.getLogger("relay.backends.local")`.
log = logging.getLogger("relay.backends.local")

# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

# Written by this backend into each run directory.
EXIT_CODE_FILENAME = "exit_code"
PID_FILENAME = "job.pid"
OUTPUT_FILENAME = "output.log"

# The settings the sidecar has to know before a signal arrives, handed over as
# a file because the sidecar is a separate process that we only get to talk to
# once. `SlurmBackend` writes the same file on shared storage.
SIDECAR_CONFIG_FILENAME = "sidecar.json"

# Written by `cancel` before it sends the signal. The sidecar reads it after the
# child is dead to tell "the user asked for this" from "the scheduler took the
# node", which are the same SIGTERM otherwise.
CANCEL_FILENAME = "CANCEL_REQUESTED"

# How many attempts have been started in a run directory. See `_next_attempt`.
ATTEMPT_FILENAME = "attempt"

# The copy of the sidecar that actually runs. Copying rather than importing is
# not an optimisation -- on Slurm the sidecar has to be a standalone file next
# to the sbatch script, because the compute node may not have relay installed
# at all. LocalBackend copies too so that the two backends are the same shape,
# and so that a run directory is a self-contained record of what was run.
SIDECAR_FILENAME = "sidecar.py"

# The optional checkpoint helper that ships beside the sidecar. A training
# script that wants relay's resume path can `sys.path.insert` the run
# directory and import it; a script that writes its own checkpoints never
# touches it. It is copied next to `sidecar.py` for the same reason the
# sidecar is copied at all -- the compute node may have no relay installed --
# and its absence is not an error, because relay runs jobs perfectly well
# without it.
CKPT_HELPER_FILENAME = "relay_ckpt.py"

# The pid -> run_dir map, kept in the backend's root directory. See
# `_load_index` for why this file exists.
INDEX_FILENAME = "local_jobs.json"


# The `sh` wrapper that records the sidecar's exit status.
#
# Read it as: `$0` is the path to write the exit code to, and `$@` (that is,
# `$1` onwards) is the sidecar command. `sh -c SCRIPT arg0 arg1 arg2...` assigns
# arg0 to `$0` and the rest to the positional parameters, which is how we smuggle
# a path in without any quoting games.
#
#   trap ':' TERM
#       `relay cancel` signals the whole process group, which includes this
#       shell. A non-interactive shell with the default disposition would die on
#       SIGTERM and never reach the `printf`, so the job would lose its exit
#       code exactly when we most want it. `:` is the no-op command, so the trap
#       does nothing except keep the shell alive. It must be a real command and
#       not the empty string: a signal the shell *ignores* is inherited through
#       `exec` by every descendant, which would make the training script immune
#       to the SIGTERM that is supposed to make it checkpoint. A signal the
#       shell *catches* is reset to the default in children, which is what we
#       want.
#
#   printf ... > "$0.tmp" && mv "$0.tmp" "$0"
#       write-then-rename, so a reader never sees a half-written file. `mv`
#       within one directory is a rename, which is atomic.
#
#   exit "$code"
#       the wrapper reports the sidecar's status as its own, so anyone holding
#       a `Popen` for it (a `relay submit` that decides to wait, say) sees the
#       real answer rather than a shell's zero.
#
#   {setup}
#       where `spec.setup` goes, when there is any. See `_SETUP_BLOCK`. Empty
#       string when there is none, so a job with no setup runs the exact script
#       this backend has always run.
_WRAPPER_SCRIPT = """\
trap ':' TERM
{setup}"$@"
code=$?
printf %s "$code" > "$0.tmp" && mv "$0.tmp" "$0"
exit "$code"
"""

# The user's `setup` lines, spliced into the wrapper *above* the sidecar
# command and in the same shell, so `export FOO=bar` or `conda activate rl`
# is still in effect when the sidecar -- and therefore the training script --
# starts. Running them in a subshell `( ... )` would be tidier for error
# handling and would defeat the entire purpose: exports do not escape a
# subshell.
#
#   trap '...' EXIT
#       `set -e` makes a failing setup line exit the shell on the spot, which
#       would happen *before* the `printf` at the bottom of the wrapper ever
#       runs -- so the run would leave no `exit_code` file, and `status()`
#       would call it "unknown" (a vanished process) rather than "failed".
#       An EXIT trap closes that hole: however the shell leaves during setup,
#       the trap writes the failing status down first. `trap - EXIT` removes
#       it again once setup is through, so the normal path below keeps its own
#       single write.
#
#   set -e / set +e
#       a failed `conda activate` must fail the job, not run training in the
#       wrong environment. Turned back off before `"$@"`, because the sidecar's
#       own nonzero exit is data we record, not an error we abort on.
#
# Security posture: these lines are shell, and they are executed as shell --
# quoting them as arguments would make `module load cuda` a filename. They come
# out of the user's own relay config, run as the user, on the user's own
# machine. That is the same trust level as their shell rc file, which `sh -c`
# would have sourced anyway. relay does not parse or sanitise them.
_SETUP_BLOCK = """\
trap 'code=$?; printf %s "$code" > "$0.tmp" && mv "$0.tmp" "$0"; exit "$code"' EXIT
set -e
{lines}
set +e
trap - EXIT
"""


def _render_wrapper(setup: list[str] | None = None) -> str:
    """The `sh` script for a job, with `setup` spliced in if there is any.

    A separate pure function so a test can read the script relay is about to
    run without starting a process for it.
    """
    if not setup:
        return _WRAPPER_SCRIPT.format(setup="")
    return _WRAPPER_SCRIPT.format(setup=_SETUP_BLOCK.format(lines="\n".join(setup)))


def default_sidecar_path() -> Path:
    """Path to the installed `relay/sidecar.py`.

    Resolved relative to this file (`relay/backends/local.py` -> `relay/`)
    rather than with `importlib.resources`. The sophisticated alternative,
    `importlib.resources.files("relay") / "sidecar.py"`, is what you would use
    if relay could be imported from a zip or a wheel that is never unpacked,
    because there is no real path inside a zip to copy from. relay is installed
    as ordinary files and the sidecar has to become an ordinary file on the
    cluster anyway, so `__file__` is both simpler and honest about that.
    """
    return Path(__file__).resolve().parent.parent / SIDECAR_FILENAME


def default_ckpt_helper_path() -> Path:
    """Path to the installed `relay/relay_ckpt.py`.

    Resolved exactly the way `default_sidecar_path` resolves the sidecar, so
    there is one answer to "where do relay's shipped-to-the-cluster files
    live". Not a constructor knob on either backend: the sidecar path is
    overridable because tests need to ship a stand-in that does nothing, while
    the helper is inert data that no test needs to fake.

    The file may not be present in every installation, so every caller checks
    `is_file()` before copying it. A missing helper costs a user the
    `relay_ckpt` import, not the job.
    """
    return Path(__file__).resolve().parent.parent / CKPT_HELPER_FILENAME


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _parse_pid(job_id: str) -> int | None:
    """Turn a job ID back into a PID, or None if it is not one.

    Job IDs are strings because the store has one column for them across both
    backends, so every use locally has to convert. IDs come out of a database
    that a human may have edited, so a garbage value is possible and should not
    crash the daemon.
    """
    try:
        pid = int(str(job_id).strip())
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _process_alive(pid: int) -> bool:
    """Does a process with this PID exist?

    `os.kill(pid, 0)` sends no signal; it only performs the "does this process
    exist and may I signal it?" check the kernel would do first.

      * `ProcessLookupError` -- no such process. It is gone.
      * `PermissionError`    -- it exists, but it belongs to someone else.
        That means the PID was recycled and now names a stranger's process, so
        it is certainly not our job. Reporting True would be wrong; we say the
        process is not there, and `status` falls through to "unknown".
      * `OverflowError`      -- the number is too big to be a PID on this
        platform, so no such process can exist. A value that large can only
        have come from a corrupted or hand-edited database row.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OverflowError):
        return False
    return True


def _read_exit_code(path: Path) -> int | None:
    """Read the `exit_code` file, or None if it is absent or unreadable.

    Anything that is not a plain integer is treated as absent rather than as an
    error. The file is written atomically, so a malformed one should be
    impossible -- but "the job's status is undecidable" is a much better
    outcome than "the daemon crashed on a stray byte".
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        # FileNotFoundError (the usual case: the job is still running) and
        # NotADirectoryError are both OSError, as is a permissions problem.
        return None
    except ValueError:
        # UnicodeDecodeError. Not a number, so: absent.
        return None
    try:
        return int(text)
    except ValueError:
        return None


# A timestamp format relay can read back. Seconds precision, explicit UTC, `Z`
# rather than "+00:00" -- the same convention the event log uses, minus the
# milliseconds, because `sacct` reports whole seconds and the two want to be
# comparable.
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _utc_iso(when: float | None = None) -> str:
    """Format an epoch timestamp (default: now) as ISO-8601 UTC."""
    moment = (
        datetime.now(timezone.utc)
        if when is None
        else datetime.fromtimestamp(when, timezone.utc)
    )
    return moment.strftime(_TIME_FORMAT)


def _epoch_from_iso(text: str) -> float | None:
    """Parse `_utc_iso` output back to epoch seconds, or None if it is garbage.

    The index file is plain JSON on disk; a hand-edited or half-written entry
    should cost one usage row, not crash the daemon.
    """
    try:
        return (
            datetime.strptime(text, _TIME_FORMAT)
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except (TypeError, ValueError):
        return None


def _elapsed_seconds(start: str, end: str) -> int:
    """Whole seconds between two `_utc_iso` timestamps, never negative.

    A clock that went backwards (NTP stepping, a laptop waking up) would
    otherwise produce a negative duration, and a negative GPU-hour figure is
    worse than a zero one.
    """
    start_epoch = _epoch_from_iso(start)
    end_epoch = _epoch_from_iso(end)
    if start_epoch is None or end_epoch is None:
        return 0
    return max(0, int(end_epoch - start_epoch))


# --------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------


class LocalBackend(Backend):
    """Runs the sidecar as a subprocess of this machine.

    `root` is where the pid -> run_dir index lives; it defaults to
    `config.default_local_root()` so that every relay process on this machine
    agrees on where to look. `sidecar_path` overrides which `sidecar.py` gets
    copied, which the tests use and nobody else should need.

    Note that `root` does *not* decide where runs go. `JobSpec.run_dir` does.
    The CLI builds run directories under `config.local_root`, and keeping that
    decision in the CLI means a backend never has to invent a path.
    """

    def __init__(
        self,
        root: str | os.PathLike | None = None,
        sidecar_path: str | os.PathLike | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else config.default_local_root()
        self.sidecar_path = (
            Path(sidecar_path) if sidecar_path is not None else default_sidecar_path()
        )

        # pid -> {"run_dir": ..., "run_id": ...}. None means "not read yet";
        # see `_load_index`.
        self._index: dict[str, dict] | None = None

        # Popen objects for jobs *this* instance started. Their only purpose is
        # reaping -- see `_reap`. A fresh backend after a daemon restart has an
        # empty dict and works fine, which is the whole point of the index file.
        self._spawned: dict[str, subprocess.Popen] = {}

    # -- the pid -> run_dir index -----------------------------------------
    #
    # `status(job_ids)` is handed job IDs and nothing else. To decide whether a
    # finished job succeeded, it has to read that job's `exit_code` file, which
    # means it has to know the run directory -- and the interface gives it no
    # way to be told. Slurm does not have this problem because `sacct` is a
    # database the scheduler keeps for us. Locally we keep our own, and it is
    # one small JSON file.
    #
    # It must be on disk rather than in memory because a restarted daemon is a
    # new process with a new `LocalBackend`, and it still has to resolve jobs
    # that the *previous* daemon (or a `relay submit` that exited seconds
    # later) started.
    #
    # This is the dead-simple version: read the whole file, mutate the dict,
    # write the whole file back. Two `relay submit` processes racing could lose
    # one entry. The sophisticated alternative is to put the mapping in the
    # SQLite database, which already serialises writers -- but then the backend
    # would know about the store, and CLAUDE.md says all SQL lives in store.py.
    # Losing an entry costs a job its exit code, not its data, and a local
    # backend is one developer on one laptop.

    def _index_path(self) -> Path:
        return self.root / INDEX_FILENAME

    def _load_index(self) -> dict[str, dict]:
        """Read the index from disk, tolerating every way it can be missing."""
        try:
            raw = self._index_path().read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            # A half-written or hand-edited file. Starting over costs exit
            # codes for old jobs; refusing to start costs the user everything.
            return {}
        return data if isinstance(data, dict) else {}

    def _record(self, job_id: str, run_id: str, run_dir: Path) -> None:
        index = self._load_index()
        index[job_id] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            # When the job started, for `usage`. Locally there is no accounting
            # database to ask afterwards -- `sacct` remembers start times for
            # weeks, the kernel remembers nothing once a process is reaped -- so
            # the only chance to learn this is now, while we are the one
            # starting it.
            "submitted_at": _utc_iso(),
        }

        self.root.mkdir(parents=True, exist_ok=True)
        # Write-then-rename again: a reader either sees the old index or the
        # new one, never a truncated one.
        tmp = self.root / (INDEX_FILENAME + ".tmp")
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._index_path())

        self._index = index

    def _entry_for(self, job_id: str) -> dict | None:
        """Look up a job's index entry, re-reading the index on a miss.

        The re-read is what makes this work across processes: a daemon that
        cached the index at startup would never see a run submitted afterwards
        by a separate `relay submit`. A miss costs one small file read.
        """
        if self._index is None:
            self._index = self._load_index()
        entry = self._index.get(job_id)
        if entry is None:
            self._index = self._load_index()
            entry = self._index.get(job_id)
        return entry if isinstance(entry, dict) else None

    def _run_dir_for(self, job_id: str) -> Path | None:
        entry = self._entry_for(job_id)
        if entry is None:
            return None
        run_dir = entry.get("run_dir")
        return Path(run_dir) if isinstance(run_dir, str) else None

    # -- submit ------------------------------------------------------------

    def submit(self, spec: JobSpec) -> str:
        """Start the sidecar and return the wrapper's PID as the job ID.

        `spec.account`, `spec.partition`, `spec.gres` and `spec.sbatch_extra`
        are ignored: there is no account to charge, no partition to land on and
        no GPU allocator when the cluster is your laptop. They are not an error
        either, so the CLI can build one `JobSpec` and hand it to whichever
        backend is configured.

        `spec.setup` is *not* ignored. Those lines run in the same shell that
        then runs the sidecar, so an `export` in setup reaches the training
        script, and a failing setup line ends the run as `failed` without the
        sidecar ever starting.
        """
        run_dir = Path(spec.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        if spec.sbatch_extra or spec.gres:
            # Debug, not a warning: one JobSpec serves both backends, so a spec
            # carrying Slurm fields to LocalBackend is the normal case, not a
            # mistake. It is still worth being able to see, because "my
            # --mem=64G did nothing" is otherwise a silent surprise.
            log.debug(
                "LocalBackend ignores Slurm-only fields: gres=%r, sbatch_extra=%r",
                spec.gres,
                spec.sbatch_extra,
            )

        # Copy the sidecar next to the job, exactly as SlurmBackend will copy
        # it next to the sbatch script on shared storage. copy2 keeps the mode.
        sidecar_copy = run_dir / SIDECAR_FILENAME
        shutil.copy2(self.sidecar_path, sidecar_copy)

        # The checkpoint helper ships with relay and travels the same way, so
        # a training script can `import relay_ckpt` from the run directory
        # without relay being installed wherever the job runs. Conditional on
        # the file existing: it is optional, and an installation without it
        # must still be able to submit jobs.
        ckpt_helper = default_ckpt_helper_path()
        if ckpt_helper.is_file():
            shutil.copy2(ckpt_helper, run_dir / CKPT_HELPER_FILENAME)

        # A requeued or re-submitted run reuses its directory, and a leftover
        # exit_code from the previous attempt would make `status` announce the
        # new attempt as already finished. Clear it before anything can look.
        exit_code_path = run_dir / EXIT_CODE_FILENAME
        exit_code_path.unlink(missing_ok=True)

        # Same reasoning for the cancel flag: if the previous attempt was
        # cancelled, leaving the file behind would tell the *new* attempt's
        # sidecar that the user had already asked it to stop.
        (run_dir / CANCEL_FILENAME).unlink(missing_ok=True)

        self._write_sidecar_config(run_dir, spec)
        previous_attempts = self._next_attempt(run_dir)

        # sys.executable, not "python3": run the sidecar under the same
        # interpreter running relay, so a virtualenv stays a virtualenv. The
        # Slurm backend uses `remote_python` from the config for the same
        # reason, since over there "the same interpreter" means nothing.
        sidecar_command = [
            sys.executable,
            str(sidecar_copy),
            "--run",
            spec.run_id,
            "--dir",
            str(run_dir),
            "--grace",
            str(spec.grace_seconds),
            "--heartbeat",
            str(spec.heartbeat_seconds),
            "--",
            *spec.command,
        ]
        wrapper = _render_wrapper(spec.setup)
        argv = ["sh", "-c", wrapper, str(exit_code_path), *sidecar_command]

        env = dict(os.environ)
        # Slurm sets this on a requeued job; locally we are the scheduler, so we
        # set it ourselves. The sidecar reads it into `run_started.attempt`.
        env["SLURM_RESTART_COUNT"] = str(previous_attempts)
        # spec.env goes on last, so an explicitly set SLURM_RESTART_COUNT wins
        # over the counter. That precedence is for tests that want to pin an
        # attempt number without submitting the job that many times; in normal
        # use nobody sets it and the counter decides.
        env.update(spec.env)

        # "ab", not "wb". Slurm needs `--open-mode=append` so that a requeued
        # attempt does not truncate the first attempt's output; appending here
        # means a local requeue behaves the same way.
        with open(run_dir / OUTPUT_FILENAME, "ab") as output:
            proc = subprocess.Popen(
                argv,
                stdout=output,
                # The sidecar already separates the two streams into `log`
                # events with a `stream` field. output.log is the human's
                # fallback copy, and interleaving matches what a terminal
                # would have shown.
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                # start_new_session makes the wrapper a session and process
                # group leader. Two things follow. First, `cancel` can signal
                # the whole group with one killpg and reach the sidecar and the
                # training script together. Second, the job does not die when
                # the terminal that launched `relay submit` closes -- a local
                # run should survive the same things a cluster run does.
                start_new_session=True,
            )
        # cwd is deliberately inherited: the user's training command may well
        # be `python train.py`, and it should resolve against the directory
        # they typed it in, not against relay's run directory.

        job_id = str(proc.pid)
        self._spawned[job_id] = proc

        (run_dir / PID_FILENAME).write_text(job_id + "\n", encoding="utf-8")
        self._record(job_id, spec.run_id, run_dir)
        return job_id

    def _write_sidecar_config(self, run_dir: Path, spec: JobSpec) -> None:
        """Write the `sidecar.json` the sidecar reads at startup.

        Everything in it has to be settled *before* the job starts, because
        the sidecar must not go looking anything up while a SIGTERM is in
        flight and its SIGKILL is ten seconds behind.

        `requeue_on_term: "auto"` resolves to `"never"` here, and not by
        accident: "auto" means "let the scheduler requeue preempted jobs if it
        does that", and on a laptop there is no scheduler and no queue to go
        back into. `SlurmBackend` resolves the same value by reading the
        partition's PreemptMode. Either way the sidecar only ever sees a
        decision, never a question.

        `resume` is the optional resume-after-preemption pair, or null. Both
        halves or neither: a glob with no argument has nothing to say about
        the file it found, so a half-filled spec writes null rather than a
        setting the sidecar would have to second-guess.

        `first_attempt_start` is the one value that is *not* rewritten on a
        resubmit -- see `_first_attempt_start`.
        """
        requeue = spec.requeue_on_term if spec.requeue_on_term == "always" else "never"
        if spec.resume_glob is None or spec.resume_arg is None:
            resume = None
        else:
            resume = {"checkpoint_glob": spec.resume_glob, "arg": spec.resume_arg}
        payload = {
            "requeue_on_term": requeue,
            "preempt_grace": int(spec.preempt_grace),
            "first_attempt_start": self._first_attempt_start(run_dir),
            "resume": resume,
        }
        (run_dir / SIDECAR_CONFIG_FILENAME).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _first_attempt_start(self, run_dir: Path) -> float:
        """When the *first* attempt in this run directory was submitted.

        Epoch seconds. Carried forward from an existing `sidecar.json` if
        there is one, and only invented when there is not.

        This is the one key a resubmit must not overwrite, and the reason is
        the resume path. The sidecar globs for checkpoint files and keeps only
        those modified at or after this timestamp, which is what stops it
        resuming from a stale file left behind by an unrelated earlier run in
        the same directory. Attempt 0 writes its checkpoint at some time T.
        If attempt 1's submit reset this value to *its* own `time.time()` --
        which is necessarily later than T -- the filter would throw away
        exactly the file the resume exists to find, and the run would restart
        from step zero every time.

        Slurm has no equivalent problem: it requeues a job in place and
        `sidecar.json` is written once, at submit, and never again.

        A missing, unreadable or nonsense value means "no usable history", and
        we start the clock now. Being wrong here costs a resume, not a run.
        """
        try:
            existing = json.loads(
                (run_dir / SIDECAR_CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict):
            previous = existing.get("first_attempt_start")
            # bool is an int subclass, and `True` is not a timestamp.
            if isinstance(previous, (int, float)) and not isinstance(previous, bool):
                return float(previous)
        return time.time()

    def _next_attempt(self, run_dir: Path) -> int:
        """Count of attempts already started here; bumps the counter for this one.

        Returns 0 the first time a run directory is used, 1 the next time, and
        so on -- which is exactly what Slurm puts in `SLURM_RESTART_COUNT`.

        A one-line counter file that this backend owns, rather than counting
        `run_started` events in `events.jsonl`. That file is right there and
        counting it would work, which is the trap: a backend that parses the
        event log knows the event schema, and the one rule that keeps relay's
        pieces from growing into each other is that it does not. `read_bytes`
        returns bytes; parsing happens in the daemon.

        Unreadable or nonsense contents count as zero previous attempts. A
        wrong attempt number is cosmetic; refusing to start a job is not.
        """
        path = run_dir / ATTEMPT_FILENAME
        try:
            previous = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            previous = 0
        if previous < 0:
            previous = 0
        path.write_text(f"{previous + 1}\n", encoding="utf-8")
        return previous

    # -- status ------------------------------------------------------------

    def _reap(self, job_id: str) -> None:
        """Collect a finished child we started, so it is not left a zombie.

        Only relevant when the process that submitted is still running -- a
        `relay submit` that stays alive, or the tests. A finished child that is
        never `wait`ed stays in the process table as a zombie, and a zombie's
        PID still answers `os.kill(pid, 0)`. `poll()` reaps it, which lets the
        liveness check tell the truth.
        """
        proc = self._spawned.get(job_id)
        if proc is not None:
            proc.poll()

    def status(self, job_ids: list[str]) -> dict[str, str]:
        """Map each job ID to one of `base.STATUSES`.

        The exit code file is consulted *before* the liveness check, which is
        the opposite of the obvious order and is deliberate. PIDs are recycled:
        a finished job's number can be handed to some unrelated process
        minutes later, and a liveness-first check would then report a run that
        ended this morning as still running. A written exit code is a fact
        about our job; a live PID is only a fact about a number.

        There is no "queued" here. A local job starts the instant you submit
        it, so it is never pending.
        """
        result: dict[str, str] = {}
        for job_id in job_ids:
            result[job_id] = self._status_one(str(job_id))
        return result

    def _status_one(self, job_id: str) -> str:
        pid = _parse_pid(job_id)
        if pid is None:
            return "unknown"

        self._reap(job_id)

        run_dir = self._run_dir_for(job_id)
        if run_dir is not None:
            code = _read_exit_code(run_dir / EXIT_CODE_FILENAME)
            if code is not None:
                return "completed" if code == 0 else "failed"

        if _process_alive(pid):
            return "running"

        # The process is gone and left no exit code: killed before the wrapper
        # could write one, or a job this machine has no record of. "unknown" is
        # honest, and the daemon treats it as "no new information" rather than
        # as an ending.
        return "unknown"

    # -- cancel ------------------------------------------------------------

    def cancel(self, job_id: str, run_dir: str | None = None) -> None:
        """SIGTERM the job's whole process group.

        `killpg`, not `kill`. `submit` used `start_new_session=True`, so the
        wrapper shell, the sidecar and the training script are all in one
        process group whose ID is the wrapper's PID. Signalling the group
        reaches all three at once; signalling only the wrapper would leave the
        sidecar running with no one watching it.

        SIGTERM, not SIGKILL, because SIGTERM is the whole point: it is what
        Slurm sends before preempting, it is what the sidecar's handler is
        installed for, and it is what gives the training script its grace
        window to write a checkpoint. This is a request to stop, not a
        guillotine.

        Cancelling a job that has already finished is a no-op. The daemon
        cancels on a snapshot of the world that may be a second or two old, and
        "it beat you to it" is not a failure.

        If `run_dir` is given, a `CANCEL_REQUESTED` file is created there
        **before** the signal goes out. The sidecar cannot tell a cancel's
        SIGTERM from a preemption's SIGTERM, so without this file a cancelled
        job configured with `requeue_on_term: always` would come straight back.
        The ordering is the whole mechanism: the sidecar only looks for the
        file after its child is dead, but it must already be on disk by then,
        and the only moment we can guarantee that is before we signal anything.
        """
        pid = _parse_pid(job_id)
        if pid is None:
            raise ValueError(f"{job_id!r} is not a local job ID; expected a PID.")

        if run_dir is not None:
            try:
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                (Path(run_dir) / CANCEL_FILENAME).touch()
            except OSError:
                # An unwritable run directory must not stop the cancel itself.
                # The worst case is a requeue the user did not want, which they
                # can cancel again; refusing to send the signal would leave the
                # job running with no way to stop it.
                pass

        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OverflowError as exc:
            # Too large to be a PID on this platform, so there is nothing to
            # signal -- but unlike a finished job, this is a bad ID rather than
            # a lost race, and the caller should hear about it.
            raise ValueError(
                f"{job_id!r} is not a local job ID; it is too large to be a PID."
            ) from exc

    # -- usage -------------------------------------------------------------

    def usage(self, job_ids: list[str]) -> list[UsageRow]:
        """Wall-clock accounting for local jobs: one row per job, no attempts.

        `SlurmBackend.usage` asks `sacct`, which is a real accounting database
        and can report every requeue attempt of a job separately. There is no
        such database on a laptop, so this measures the only thing it can see:
        how long the process it started has been alive.

          start     the submit timestamp, recorded in the index because after
                    the process is reaped there is nobody left to ask
          end       the mtime of the `exit_code` file, which the wrapper writes
                    the moment the sidecar finishes -- so it is within
                    milliseconds of the real end -- or None while running
          elapsed_s end - start, or now - start while running, which is why a
                    running row's number grows between calls
          gpus      0. Allocating GPUs is a scheduler's job and there is no
                    scheduler here. Zero is the honest answer, not a placeholder.
          cpus      this machine's CPU count: a local job is not confined to a
                    subset, it just runs
          partition None, for the same reason as gpus

        A job ID the index has never heard of gets no row at all, rather than a
        row of zeroes. "I have no record of this" and "this used nothing" are
        different claims, and the store should not have to guess which one a row
        of zeroes meant.

        One row per job here, never several, because a local resubmit into the
        same run directory gets a *new* PID and therefore a new job ID -- which
        is not how Slurm behaves, where a requeued job keeps its ID and gains an
        attempt.
        """
        cpus = os.cpu_count() or 1
        now = _utc_iso()
        rows: list[UsageRow] = []

        for raw_id in job_ids:
            job_id = str(raw_id)
            entry = self._entry_for(job_id)
            if entry is None:
                continue
            start = entry.get("submitted_at")
            if not isinstance(start, str) or _epoch_from_iso(start) is None:
                # Submitted by an older relay that did not record the time, or
                # a corrupted entry. No start, no row: `UsageRow.start` is not
                # allowed to be None and inventing one would invent hours.
                continue

            state, end = self._usage_state_and_end(job_id, entry)
            elapsed = _elapsed_seconds(start, end if end is not None else now)

            rows.append(
                UsageRow(
                    job_id=job_id,
                    state=state,
                    partition=None,
                    start=start,
                    end=end,
                    elapsed_s=elapsed,
                    gpus=0,
                    cpus=cpus,
                )
            )
        return rows

    def _usage_state_and_end(self, job_id: str, entry: dict) -> tuple[str, str | None]:
        """`(scheduler-style state word, end timestamp or None)` for one job.

        The state words are upper-case and scheduler-shaped ("RUNNING",
        "COMPLETED", "FAILED") because `UsageRow.state` is defined as whatever
        the accounting source calls it, kept raw. Usage accounting needs to
        tell a preempted attempt from a failed one, so it does not reuse the
        deliberately coarser `STATUSES` vocabulary.
        """
        run_dir = entry.get("run_dir")
        exit_code_path = (
            Path(run_dir) / EXIT_CODE_FILENAME if isinstance(run_dir, str) else None
        )

        code = _read_exit_code(exit_code_path) if exit_code_path is not None else None
        if code is not None:
            try:
                end = _utc_iso(exit_code_path.stat().st_mtime)
            except OSError:
                end = _utc_iso()
            return ("COMPLETED" if code == 0 else "FAILED"), end

        pid = _parse_pid(job_id)
        if pid is not None and _process_alive(pid):
            return "RUNNING", None

        # Gone, and it never recorded an exit code: SIGKILLed before the
        # wrapper could write one. It burned real time, so it still gets a row;
        # we just cannot say when it stopped, and the elapsed number is an
        # upper bound rather than a measurement.
        return "UNKNOWN", None

    # -- read_bytes --------------------------------------------------------

    def read_bytes(self, path: str, offset: int) -> tuple[bytes, int]:
        """Read `path` from `offset` to end of file.

        Returns `(data, offset + len(data))`. Reading past the end is not an
        error; it returns no bytes and the same offset, which is the normal
        answer for "nothing new since last time".

        `FileNotFoundError` is allowed to propagate. The daemon reads it as
        "the job has not created its run directory yet, so it is still queued",
        and swallowing it here would turn a queued job into one that looks
        started but silent.
        """
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
        return data, offset + len(data)
