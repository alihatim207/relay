"""The contract between relay's CLI and whatever actually runs a job.

There are exactly two things in this module:

  * `JobSpec` -- everything the CLI knows about a run it wants started. It is
    the *only* thing the CLI hands a backend. If a backend needs something, it
    goes in here; if a backend invents something of its own, that is a smell.

  * `Backend` -- five methods, no more. A backend starts jobs, reports what
    the scheduler thinks of them, cancels them, reads bytes out of files, and
    reports what they cost in GPU- and CPU-hours.

  * `UsageRow` -- one scheduler attempt's resource usage, the shape `usage()`
    returns.

Two deliberate absences are worth naming, because they are what keeps relay's
pieces from growing into each other:

**A backend knows nothing about the event schema.** `read_bytes` returns raw
bytes from a path and that is all. It does not know that the file is JSONL, it
does not know what a `heartbeat` is, and it never parses anything. Parsing
happens in the daemon. That is why a backend can be swapped out (local for
Slurm) without the daemon noticing, and why testing the daemon's parsing does
not require a cluster.

**A backend never touches the database.** It has no idea relay has one. All SQL
lives in `store.py`.

`status` takes a *list* of job IDs and returns a dict, rather than a method you
call once per job. That shape exists for `SlurmBackend`: one `squeue` call
covers every job you have submitted, while a per-job API would mean one SSH
round trip per run, and on a Duo-protected cluster round trips are the
expensive thing. `LocalBackend` does not care, but it honours the same shape so
the daemon has one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Status vocabulary
# --------------------------------------------------------------------------

# The only strings a backend may return from `status()`.
#
# These describe what the *scheduler* (or, locally, the OS) thinks the job is
# doing. They are not derived from the event log -- CLAUDE.md is explicit that
# the log is authoritative about what the training did, while the scheduler is
# authoritative about what the job did. A job SIGKILLed mid-step writes no
# `run_ended` at all, so if backends reported status from the log, such a run
# would sit at "running" forever.
#
#   queued     accepted, not started yet (Slurm PENDING; LocalBackend never
#              returns this, because a local job starts the moment you submit)
#   running    the process/job exists and is executing
#   completed  finished with exit code 0
#   failed     finished with a nonzero exit code, or the scheduler says it
#              failed, timed out, or was preempted out
#   cancelled  a human (or `relay cancel`) stopped it
#   unknown    the backend cannot tell. Not an error: a Slurm job can age out
#              of `sacct`, and a local PID can vanish without leaving a trace.
#              The daemon treats it as "no new information", not as terminal.
STATUSES = ("queued", "running", "completed", "failed", "cancelled", "unknown")

# The subset that means "this job is over, stop polling it". Kept here rather
# than in the daemon so that adding a status cannot leave the two disagreeing.
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


# --------------------------------------------------------------------------
# JobSpec
# --------------------------------------------------------------------------


@dataclass
class JobSpec:
    """Everything a backend needs in order to start one run.

    Built by the CLI, consumed by a backend, never stored. It is a plain
    mutable dataclass rather than a frozen one because `relay submit --seeds
    0-4` builds five of these by copying and tweaking, and copy-and-tweak is
    what `dataclasses.replace` does best either way.
    """

    # relay's own run ID, e.g. "vr_s1_7f3a". Never a Slurm job ID. It goes into
    # every event's envelope, so it has to be decided before the job starts and
    # has to survive a requeue unchanged.
    run_id: str

    # The user's training command, already split into argv. A list, not a
    # string, so nothing ever has to think about shell quoting:
    # ["python", "train.py", "--lr", "3e-4"].
    command: list[str]

    # Absolute path to this run's directory. The sidecar writes `events.jsonl`
    # here and a copy of `sidecar.py` lives here. On Slurm this is under
    # `remote_root` on the shared filesystem -- it is the entire channel from
    # the compute node back to the laptop -- and locally it is under
    # `local_root`. Absolute because a relative path would resolve against
    # whatever directory the job happened to start in.
    run_dir: str

    # Human-readable label for `relay ls`. Optional; the CLI falls back to the
    # run ID when it is None.
    name: str | None = None

    # Seconds of warning the job gets before its *time limit*. On Slurm this is
    # written into `--signal=B:USR1@<grace>`, and the sidecar spends grace - 2
    # of them waiting for the training script to checkpoint before it requeues
    # itself. Passed to the sidecar as --grace so the two numbers cannot drift.
    #
    # This has nothing to do with preemption -- see `preempt_grace` below.
    grace_seconds: int = 60

    # Seconds between the SIGTERM a *preempted or cancelled* job receives and
    # the SIGKILL that follows. Relay cannot control this window; it belongs to
    # the partition (GraceTime) or the cluster (KillWait). The sidecar spends
    # preempt_grace - 2 of them waiting for the child, floored at 1. Klone's
    # measured value is 10, hence the default.
    preempt_grace: int = 10

    # What the sidecar should do after a SIGTERM that was not a cancel:
    #   "always"  run `scontrol requeue` itself
    #   "never"   do nothing; Slurm's PreemptMode=REQUEUE brings the job back
    #   "auto"    the backend resolves this at submit time by reading the
    #             partition's PreemptMode, and writes the result ("always" or
    #             "never") into the run directory's sidecar.json
    # The sidecar itself only ever sees "always" or "never".
    requeue_on_term: str = "auto"

    # Wall-clock limit for the job, in Slurm's `--time` format ("04:00:00",
    # "1-12:00:00", "90"). Required on Slurm: Klone's checkpoint partitions
    # have DefaultTime=01:00:00, so a job without an explicit --time is
    # silently capped at one hour. `SlurmBackend.submit` refuses a spec with
    # no time limit. LocalBackend ignores it.
    time_limit: str | None = None

    # Seconds between heartbeat events. The daemon's staleness check compares
    # against this, so it belongs to the spec rather than being hard-coded in
    # the sidecar's defaults.
    heartbeat_seconds: int = 30

    # Extra environment variables for the job, on top of the inherited
    # environment. Values are strings because that is what the OS stores.
    env: dict[str, str] = field(default_factory=dict)

    # -- Slurm sub-spec ----------------------------------------------------
    #
    # Left as plain optional fields rather than a nested `SlurmJobSpec` object.
    # A nested object would be tidier on paper, but it would mean every caller
    # constructing a local job still has to decide what to put in the nested
    # slot, and every backend has to guard against it being None. Three flat
    # fields with sane defaults are simpler, and this is a two-backend program.
    #
    # **LocalBackend ignores all three.** There is no account to charge and no
    # partition to land on when the "cluster" is your laptop, so LocalBackend
    # neither reads them nor complains about them. Only `SlurmBackend` looks
    # here, and only it decides what is required.

    # Slurm account to charge the job to (`#SBATCH --account=`).
    account: str | None = None

    # Partition to submit to (`#SBATCH --partition=`).
    partition: str | None = None

    # Generic resources, Slurm's spelling for GPUs: "gpu:1", "gpu:a40:2"
    # (`#SBATCH --gres=`). Its own field rather than an `sbatch_extra` entry
    # because it is the one flag nearly every relay job needs, and because
    # usage accounting will later want to compare what was asked for against
    # what `sacct` says was allocated.
    gres: str | None = None

    # Any other #SBATCH directives, each written exactly as it would appear
    # after `#SBATCH `: ["--mem=64G", "--cpus-per-task=8"]. Rendered verbatim,
    # one per line, after the directives relay generates itself. This is the
    # escape hatch for the hundred-odd Slurm options relay does not model.
    #
    # It is a list of whole directives rather than a name -> value dict so the
    # user writes the flag the same way they would in an sbatch script, and so
    # relay never has to guess whether `--gres gpu:1` and `--gres=gpu:1` are
    # the same thing. `validate_sbatch_extra` rejects entries for flags relay
    # owns; the CLI runs it before anything is rendered or sent anywhere.
    sbatch_extra: list[str] = field(default_factory=list)

    # -- Resume after preemption (optional; relay works fully without) -----
    #
    # For scripts that already write their own checkpoints. On any attempt
    # after the first, the sidecar globs `resume_glob` (relative to the job's
    # working directory, `**` allowed), keeps only files modified at or after
    # the run's first attempt started, picks the newest, and appends
    # `resume_arg` with `{path}` replaced by the shell-quoted file to the
    # training command. Both or neither: a glob without an arg has nothing to
    # do with the file it found. Backends write these into sidecar.json.
    resume_glob: str | None = None
    resume_arg: str | None = None

    # Shell lines to run before the training command: `module load cuda`,
    # `conda activate rl`, `export FOO=bar`. Rendered verbatim, in order,
    # between the #SBATCH block and the sidecar command, wrapped in `set -e`
    # / `set +e` so a failed `conda activate` fails the job immediately
    # instead of running training in the wrong environment. Relay does not
    # parse or validate their contents. LocalBackend runs them in the same
    # shell as the sidecar, so an exported variable reaches the child.
    setup: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# sbatch_extra validation
# --------------------------------------------------------------------------

# Flags relay itself writes into every sbatch script, and what breaks if the
# user overrides one. Keyed by flag as it appears on the command line, in both
# long and short forms, so a user cannot slip `-o` past a check that only
# knows `--output`.
OWNED_SBATCH_FLAGS: dict[str, str] = {
    "--output": "log continuity: relay points --output at the run directory with --open-mode=append",
    "-o": "log continuity: relay points --output at the run directory with --open-mode=append",
    "--error": "log continuity: stderr is folded into relay's --output file",
    "-e": "log continuity: stderr is folded into relay's --output file",
    "--open-mode": "log continuity: without append mode a requeued attempt truncates the previous attempt's output",
    "--signal": "time-limit handling: relay relies on --signal=B:USR1@<grace> to checkpoint and requeue",
    "--requeue": "requeue: relay's whole preemption story assumes --requeue",
    "--no-requeue": "requeue: relay's whole preemption story assumes --requeue",
    "--job-name": "run identity: relay names the job after the run so `squeue` output matches `relay ls`",
    "-J": "run identity: relay names the job after the run so `squeue` output matches `relay ls`",
}

# Flags relay sets from its own config fields. Not forbidden because they
# break anything, but because two conflicting values in one script mean the
# last one silently wins, and the user would never know which.
FIELD_SBATCH_FLAGS: dict[str, str] = {
    "--account": "slurm.account",
    "-A": "slurm.account",
    "--partition": "slurm.partition",
    "-p": "slurm.partition",
    "--time": "slurm.time or `relay submit --time`",
    "-t": "slurm.time or `relay submit --time`",
    "--gres": "slurm.gres",
}


class SbatchExtraError(ValueError):
    """An `sbatch_extra` entry relay refuses to render.

    Its own class so the CLI can map it to exit code 2 (bad usage): the user
    typed something relay will not accept, and the message says what and why.
    """


def _flag_of(entry: str) -> str:
    """The flag part of a directive: `--mem=64G` -> `--mem`, `-p ckpt` -> `-p`.

    Long options split on `=`; both forms split on whitespace. `--mem 64G`
    and `--mem=64G` are the same flag to sbatch and must be the same flag to
    us.
    """
    head = entry.strip().split(None, 1)[0]
    if head.startswith("--"):
        head = head.split("=", 1)[0]
    return head


def validate_sbatch_extra(entries: list[str]) -> None:
    """Refuse `sbatch_extra` entries relay must control, before rendering.

    Raises `SbatchExtraError` naming the offending entry and either which
    relay behaviour it would break or which relay field to use instead.
    Silence means every entry is safe to render as `#SBATCH <entry>`.

    Runs in the CLI before anything is rendered or sent over SSH, so a bad
    entry costs nothing but the error message.
    """
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise SbatchExtraError(
                f"sbatch_extra entry {entry!r} is empty. Each entry is one sbatch "
                "directive such as --mem=64G."
            )
        if not entry.strip().startswith("-"):
            raise SbatchExtraError(
                f"sbatch_extra entry {entry!r} does not start with '-'. Each entry "
                "is written as the flag and value, for example --mem=64G or "
                "--cpus-per-task=8."
            )
        flag = _flag_of(entry)
        if flag in OWNED_SBATCH_FLAGS:
            raise SbatchExtraError(
                f"sbatch_extra entry {entry!r} sets {flag}, which relay owns. "
                f"Overriding it would break {OWNED_SBATCH_FLAGS[flag]}. Remove it."
            )
        if flag in FIELD_SBATCH_FLAGS:
            raise SbatchExtraError(
                f"sbatch_extra entry {entry!r} sets {flag}, which relay already "
                f"sets from its own config. Use {FIELD_SBATCH_FLAGS[flag]} instead, "
                "so the script cannot end up with two conflicting values."
            )


# --------------------------------------------------------------------------
# UsageRow
# --------------------------------------------------------------------------


@dataclass
class UsageRow:
    """One scheduler attempt's resource usage, as the scheduler reports it.

    One row per *attempt*, not per job: a Slurm job that was preempted twice
    and requeued has three attempts under the same job ID, and each one burned
    real GPU-hours. `(job_id, start)` is therefore the natural key -- attempts
    share a job ID but never a start time.

    A backend fills this from its own accounting source (`sacct` on Slurm,
    process wall time locally). It does not know the run ID; the daemon maps
    job IDs back to runs before handing rows to the store.
    """

    # The scheduler's job ID, as a string, same as `submit()` returned.
    job_id: str

    # The scheduler's own state word for this attempt, verbatim and
    # upper-cased: "COMPLETED", "PREEMPTED", "REQUEUED", "CANCELLED", "RUNNING"
    # and so on. Kept raw rather than mapped to `STATUSES` because usage
    # accounting needs to tell a preempted attempt from a failed one, and the
    # status vocabulary deliberately folds those together.
    state: str

    # Partition the attempt ran on, or None locally.
    partition: str | None

    # ISO-8601 UTC start time. Never None: an attempt that has not started has
    # used nothing and does not get a row.
    start: str

    # ISO-8601 UTC end time, or None while the attempt is still running.
    end: str | None

    # Wall-clock seconds the attempt has consumed so far. Grows while running.
    elapsed_s: int

    # GPUs allocated to the attempt. Zero when none were requested.
    gpus: int

    # CPUs allocated to the attempt.
    cpus: int


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


class Backend:
    """The interface every backend implements.

    A plain class whose methods raise `NotImplementedError`, rather than an
    `abc.ABC` with `@abstractmethod`. The sophisticated alternative -- ABCs --
    buys you one thing: instantiating an incomplete subclass fails immediately
    at construction rather than later at the first call. With two backends,
    both covered by tests that call all five methods, that is not worth the
    extra machinery. If relay ever grows third-party backends, switch to ABC.

    Subclasses must not raise for conditions that are simply normal:

      * a job that has not started yet is `"queued"`, not an exception
      * cancelling an already-dead job is a no-op, not an error
      * a run directory that does not exist yet is `FileNotFoundError` from
        `read_bytes`, which the daemon reads as "still queued" -- so do not
        catch it and do not translate it into empty bytes
    """

    def submit(self, spec: JobSpec) -> str:
        """Start the job and return its backend-specific job ID.

        The returned ID is opaque to everyone except the backend that made it:
        a Slurm job number for `SlurmBackend`, a PID for `LocalBackend`. It is
        a string in both cases so the store has one column type. It is *not*
        the run ID -- the run ID is relay's, the job ID is the scheduler's, and
        a requeued Slurm job can keep one while changing the other.
        """
        raise NotImplementedError

    def status(self, job_ids: list[str]) -> dict[str, str]:
        """Report what the scheduler thinks of each job.

        Returns a dict mapping every requested ID to one of `STATUSES`. Every
        requested ID gets a key, even if the answer is `"unknown"`; a caller
        should never have to distinguish "missing from the dict" from "the
        backend does not know".

        Takes a list because one `squeue` covers every job at once.
        """
        raise NotImplementedError

    def cancel(self, job_id: str, run_dir: str | None = None) -> None:
        """Ask the scheduler to stop the job.

        If `run_dir` is given, the backend first creates a file named
        `CANCEL_REQUESTED` in it, and only then sends the cancel. The sidecar
        cannot tell a cancel's SIGTERM from a preemption's SIGTERM -- they are
        the same signal -- so this file is how it learns not to requeue a job
        the user just asked to stop. Ordering matters: the file must exist
        before the signal lands.

        Idempotent: cancelling a job that already finished does nothing and
        raises nothing. Returns as soon as the request is made; the job may
        take its full grace period to actually die, and the daemon will notice
        when it does.
        """
        raise NotImplementedError

    def usage(self, job_ids: list[str]) -> list[UsageRow]:
        """Report resource usage for every attempt of every listed job.

        Returns one `UsageRow` per attempt, so a requeued job can yield several
        rows for one ID. Takes a list for the same reason `status` does: one
        `sacct` call covers every job at once.

        Usage is not live data. The daemon calls this every few minutes, not
        every cycle, and once more when a run goes terminal.
        """
        raise NotImplementedError

    def read_bytes(self, path: str, offset: int) -> tuple[bytes, int]:
        """Read `path` from `offset` to end-of-file.

        Returns `(data, new_offset)`, where `new_offset` is `offset +
        len(data)` -- the position to pass in next time. Returning it, rather
        than making the caller add, means the backend can tell the truth about
        how far it actually got if a transport cuts a read short.

        Raises `FileNotFoundError` if the path does not exist. That is
        deliberate and the daemon depends on it: a queued job has no run
        directory yet, and "no file" must stay distinguishable from "a file
        with nothing new in it".
        """
        raise NotImplementedError
