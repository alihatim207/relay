"""Run jobs on a Slurm cluster, reached only through the user's own `ssh`.

This is `LocalBackend` with a network in the middle, and the network is the
interesting part. Three constraints from CLAUDE.md shape every line here.

**Never construct `user@host`.** relay stores an *SSH alias* -- the name of a
`Host` block in the user's `~/.ssh/config` -- and hands that string to the `ssh`
binary untouched. The user's own configuration then supplies the username, the
hostname, the key, any jump host, and the ControlMaster socket. On a
Duo-protected cluster like Hyak a fresh authentication costs tens of seconds and
a phone in someone's pocket, so bypassing their multiplexed connection would
turn every poll into a 2FA prompt.

**Round trips are the expensive thing.** Not bytes, not CPU -- round trips. So
`status` asks about every job in one `squeue` call, `usage` about every job in
one `sacct` call, `read_bytes` is one `tail`, and `submit` spends two -- one
`tar` stream carrying every one of the job's files, then `sbatch` -- plus one
more the first time it has to look a partition up. Anywhere a loop would have
meant one SSH invocation per item, there is a batched command instead, and a
comment saying so.

**Transport failure is routine.** A dropped connection, a login node that is
briefly refusing sessions, a timeout -- these happen on a shared cluster several
times a day and they are not bugs. They raise `TransportError`, which the daemon
catches, records, and retries next cycle with a backoff. A *command* failing
(sbatch rejecting a script, scancel refusing an ID) is a different thing
entirely: that is `SlurmError`, and it means someone has to go read the message.
Conflating the two would have the daemon give up on a healthy cluster, or retry
forever against a script Slurm will never accept.

Everything remote goes through `_ssh()`. If you find yourself calling
`subprocess` anywhere else in this file, that is the bug.

## Testing

`SlurmBackend` takes a `runner` callable -- the one place a real subprocess
would be started. Tests pass a fake and assert on the argv and stdin that would
have gone out, so the whole test suite runs in CI with no cluster and never
opens an SSH connection.
"""

from __future__ import annotations

import io
import json
import re
import shlex
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from relay import ssh as ssh_mod
from relay.backends.base import Backend, JobSpec, UsageRow, validate_sbatch_extra

# --------------------------------------------------------------------------
# Names and constants
# --------------------------------------------------------------------------

# Files this backend writes into a run directory on shared storage.
SBATCH_FILENAME = "job.sbatch"
SIDECAR_FILENAME = "sidecar.py"

# relay's optional checkpoint helper, shipped in the same tar stream as the
# sidecar so a training script can `import relay_ckpt` from the run directory
# on a compute node that has never heard of relay. Optional: an installation
# without the file still submits jobs, it just does not offer the import, so
# every caller checks that the file exists before reading it.
CKPT_HELPER_FILENAME = "relay_ckpt.py"

# The sidecar's own little config file, written beside it at submit time. It
# carries the two decisions the sidecar cannot make for itself on a compute
# node: whether to call `scontrol requeue` after a SIGTERM, and how many
# seconds it has before the SIGKILL arrives. Writing them into a file rather
# than passing them as flags means a human can read (and edit) what the job
# was told, and means the sidecar never has to ask Slurm anything.
SIDECAR_CONFIG_FILENAME = "sidecar.json"

# Written into the run directory by `cancel()` *before* scancel is sent. The
# sidecar cannot tell a cancel's SIGTERM from a preemption's SIGTERM -- they
# are the same signal -- so this file is the whole difference between "the
# user asked me to stop" and "Slurm needs this node back, come again later".
CANCEL_FILENAME = "CANCEL_REQUESTED"

# How long to wait for one ssh invocation. There is deliberately no number
# here: the value lives in `relay.ssh.DEFAULT_SSH_TIMEOUT`, the config key
# `ssh_timeout` overrides it, and this backend carries it as
# `self.ssh_timeout`. Every remote call in this file uses that one value (or
# an explicit multiple of it, each one commented), so no code path can end up
# quietly tighter than the rest -- which is exactly how the first real Klone
# run failed, with a 10 second literal against a login node that needs 4 to 5
# seconds just to start a shell.
SSH_TIMEOUT_DEFAULT = ssh_mod.DEFAULT_SSH_TIMEOUT

# `sbatch` gets twice the budget of an ordinary probe. Submitting is the one
# remote call that waits on the Slurm *controller* rather than just the login
# node: on a busy cluster slurmctld can take seconds to answer while it is
# scheduling, and a submit that times out is worse than a slow one -- relay
# cannot tell whether the job was accepted, so it can neither record a job ID
# nor safely retry.
SBATCH_TIMEOUT_FACTOR = 2

# ssh's own "I could not do my job" exit status. Documented in ssh(1): ssh
# exits with the remote command's status, except that 255 means ssh itself
# failed (could not connect, authentication refused, connection dropped).
#
# A remote command *could* genuinely exit 255 and be misread as a transport
# failure. Nothing relay runs remotely does that, and the cost of the confusion
# is one wasted retry, so this is the right trade.
SSH_FAILURE_RETURNCODE = 255

# The exit status our `read_bytes` shell snippet uses to say "that path does not
# exist". An out-of-the-way number so it cannot be confused with a real tool's
# status (1 = generic error, 2 = usage, 126/127 = not executable / not found).
#
# Using an *exit code* rather than printing a sentinel string is deliberate:
# `read_bytes` returns raw bytes from an event log, and a sentinel inside the
# byte stream could in principle appear in the data itself. An exit status
# travels on a separate channel and cannot corrupt what is being read.
ENOENT_RETURNCODE = 44

# What `sbatch` prints on success. Slurm has printed exactly this sentence for
# many years, but pin the parse to a regex rather than to a column index so a
# changed prefix fails loudly instead of returning the wrong number.
_SUBMITTED_RE = re.compile(r"Submitted batch job (\d+)")

# scancel's complaints that mean "there was nothing to cancel". Cancelling a
# job that already finished is a no-op, per the Backend contract: the daemon
# acts on a snapshot of the world that is a second or two old, and losing that
# race is not a failure.
_HARMLESS_CANCEL_ERRORS = (
    "invalid job id",
    "already completed",
    "already finished",
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class TransportError(Exception):
    """ssh could not deliver the command, or could not come back with an answer.

    Routine, not exceptional. The daemon catches this, records it, and tries
    again next cycle with an exponential backoff. It says nothing about whether
    the remote command would have succeeded -- we never found out.

    It carries `returncode` and `stderr` as attributes, not just inside the
    message, because the daemon has to tell two failures apart and needs the
    raw material to do it: `ssh.classify_failure(rc, stderr, master_alive)`
    decides whether this is `auth_required` (the user must run `relay connect`,
    and retrying will never help) or `unreachable` (back off and try again).
    Re-parsing that out of a human-readable sentence would be silly and
    fragile. `returncode` is None when ssh never ran or never finished -- a
    timeout, or no ssh binary at all.
    """

    def __init__(
        self, message: str, returncode: int | None = None, stderr: str = ""
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class SlurmError(Exception):
    """ssh worked fine; the Slurm command said no.

    A rejected sbatch script, an account the user cannot charge to, a partition
    that does not exist. Retrying will not help, so the message is a plain
    sentence carrying whatever Slurm said, and it goes in front of a human.
    """


class MissingTimeLimit(SlurmError):
    """No `--time` could be found for this job, from any source.

    A subclass rather than a plain `SlurmError` so the CLI can map it to exit
    code 2 ("bad usage") instead of 1 ("runtime failure"): nothing went wrong
    out there, the user simply did not say how long the job may run, and the
    fix is a flag rather than an investigation.

    Why relay refuses to guess: Klone's checkpoint partitions have
    `DefaultTime=01:00:00`, so a job submitted without an explicit `--time` is
    silently capped at one hour. A twelve-hour training run would die at 60
    minutes for no reason the user could see.
    """


# --------------------------------------------------------------------------
# State mapping
# --------------------------------------------------------------------------
#
# Slurm has a large state vocabulary; relay has six words (see base.STATUSES).
# This table is the whole translation, in one place, so that adding a state
# means editing one dict rather than hunting through branches.

_STATE_MAP = {
    # Accepted, waiting for resources.
    "PENDING": "queued",
    "CONFIGURING": "queued",
    "REQUEUED": "queued",
    "REQUEUE_HOLD": "queued",
    "RESV_DEL_HOLD": "queued",
    "SUSPENDED": "queued",
    # PREEMPTED maps to "queued", not "failed", and that is a judgment call
    # worth explaining. relay submits every job with `--requeue`, so a preempted
    # job is not over: Slurm puts it straight back in the pending queue and it
    # will run again, with the same relay run ID and a sidecar that continues
    # `seq` where the last attempt stopped. Surviving preemption is the feature.
    # Calling it "failed" would mark the run terminal, the daemon would stop
    # tailing its event log, and the user would watch a run that is about to
    # resume get filed under "broken".
    "PREEMPTED": "queued",
    # Executing.
    "RUNNING": "running",
    "COMPLETING": "running",  # child processes have exited, cleanup in progress
    # Over, and it worked.
    "COMPLETED": "completed",
    # Over, and it did not.
    "FAILED": "failed",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    # Over, because someone said so.
    "CANCELLED": "cancelled",
}


def map_slurm_state(raw: str) -> str:
    """Translate one Slurm state string into one of `base.STATUSES`.

    Two pieces of Slurm formatting have to be undone first:

      * `sacct` reports a cancelled job as `CANCELLED by 12345` -- the state,
        then the UID of whoever ran scancel. Only the first word is the state.
      * A state that does not fit the output column comes back truncated with a
        `+` appended (`CANCELLED+`, `OUT_OF_ME+`). We strip the `+`; a truly
        truncated name simply will not match and falls through to "unknown".

    Anything unrecognised is "unknown", never an exception. A Slurm upgrade that
    adds a state should make relay uninformative, not broken.
    """
    if not raw:
        return "unknown"
    head = raw.strip().split()[0] if raw.strip() else ""
    head = head.rstrip("+").upper()
    return _STATE_MAP.get(head, "unknown")


# --------------------------------------------------------------------------
# Partitions: what `scontrol show partition` says, and what we do about it
# --------------------------------------------------------------------------


def parse_scontrol_kv(text: str) -> dict[str, str]:
    """Parse one `scontrol show ...` entry into a dict.

    scontrol prints whitespace-separated `Key=Value` tokens, wrapped across
    however many lines it feels like. There is no ordering guarantee and no
    alignment to rely on, so the parse is: split on *any* whitespace, keep
    every token containing `=`, split each on its **first** `=`.

    Splitting on the first `=` matters because some values contain more of
    them -- `TRES=cpu=40,mem=187000M,gres/gpu=8` is one token whose value is
    itself a key=value list.

    Typical tokens relay cares about:

        PreemptMode=REQUEUE   Slurm requeues preempted jobs itself, so the
                              sidecar must not requeue them a second time
        GraceTime=0           seconds between SIGTERM and SIGKILL; 0 means
                              the cluster-wide KillWait applies instead
        MaxTime=UNLIMITED     ceiling on --time
        DefaultTime=01:00:00  what you silently get if you omit --time

    Values are returned as the strings scontrol printed. No type conversion
    here: "UNLIMITED" is not an integer and "01:00:00" is not seconds, and
    each caller knows which shape it expects.
    """
    info: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if not key:
            continue
        info[key] = value
    return info


def split_scontrol_entries(text: str) -> list[str]:
    """Split a multi-entry `scontrol show` dump into one string per entry.

    `scontrol show partition` with no name prints every partition, separated by
    blank lines. Parsing the whole dump as one blob would merge them and give
    you the last partition's PreemptMode under the first one's name.
    """
    entries: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                entries.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        entries.append("\n".join(current))
    return entries


def _select_partition(text: str, name: str | None) -> dict[str, str]:
    """Pick one partition out of a `scontrol show partition` dump.

    With a name, scontrol printed exactly one entry and we parse it. Without
    one, it printed them all and we want the one flagged `Default=YES` -- the
    partition an sbatch script with no `--partition` actually lands on.
    """
    entries = split_scontrol_entries(text)
    if not entries:
        return {}
    if name:
        return parse_scontrol_kv(entries[0])
    for entry in entries:
        info = parse_scontrol_kv(entry)
        if info.get("Default", "").upper() == "YES":
            return info
    return {}


def resolve_requeue_on_term(configured: str, preempt_mode: str | None) -> str:
    """Decide what the sidecar should do after a SIGTERM. Pure function.

    Returns `"always"` or `"never"` -- the sidecar only ever sees those two.
    `"auto"` is resolved here, at submit time, because the compute node must
    not have to ask Slurm anything on the signal path.

    The rule:

      * `always`/`never` are the user's explicit answer and pass straight
        through.
      * `auto` with a partition whose PreemptMode includes REQUEUE resolves to
        `never`: Slurm already puts the job back in the queue, and a second
        `scontrol requeue` on top of that is at best redundant. (This is
        Klone's case -- `PreemptMode=REQUEUE` both on `ckpt-all` and
        cluster-wide.)
      * `auto` with any other PreemptMode resolves to `always`: nobody else is
        going to bring the job back, so the sidecar must.
      * `auto` when we could not read the partition at all (preempt_mode is
        None) also resolves to `always`. That asymmetry is deliberate: a
        double requeue is a nuisance -- one extra attempt that exits
        immediately -- while a job that never comes back is lost work, often
        hours of it. When in doubt, requeue.
    """
    value = (configured or "auto").strip().lower()
    if value == "always":
        return "always"
    if value == "never":
        return "never"
    # "auto", and anything unrecognised. config.py validates the vocabulary;
    # falling through to the auto rule here means a typo degrades to the safe
    # behaviour rather than to a crash inside submit.
    if preempt_mode and "REQUEUE" in preempt_mode.upper():
        return "never"
    return "always"


# --------------------------------------------------------------------------
# Parsing sacct usage rows
# --------------------------------------------------------------------------


def parse_alloc_tres(text: str) -> tuple[int, int]:
    """Pull `(cpus, gpus)` out of one sacct `AllocTRES` field. Pure function.

    The field is a comma-separated list of `key=value` pairs describing what
    Slurm actually allocated:

        billing=80,cpu=40,gres/gpu=2,mem=187000M,node=1
        billing=80,cpu=8,gres/gpu:a40=2,mem=64G,node=1

    GPUs appear either untyped (`gres/gpu=2`) or typed (`gres/gpu:a40=2`), and
    on some clusters *both* are printed for the same allocation. They describe
    the same GPUs, so summing them double-counts -- the spec is explicit here:
    prefer the untyped key, and when only typed keys are present take the
    first one. Adding them together would turn a 2-GPU job into a 4-GPU one
    and quietly double every GPU-hour figure relay reports.

    A job with no GPU key at all has zero GPUs. That is a CPU job, not an
    error.

    Values relay does not need (`billing`, `mem`, `node`) are skipped without
    being parsed, so a unit suffix like `187000M` can never raise here.
    """
    cpus = 0
    untyped_gpus: int | None = None
    first_typed_gpus: int | None = None

    for item in (text or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, _, value = item.partition("=")
        key = key.strip()
        value = value.strip()
        if key == "cpu":
            cpus = _to_int(value)
        elif key == "gres/gpu":
            untyped_gpus = _to_int(value)
        elif key.startswith("gres/gpu:") and first_typed_gpus is None:
            first_typed_gpus = _to_int(value)

    if untyped_gpus is not None:
        gpus = untyped_gpus
    elif first_typed_gpus is not None:
        gpus = first_typed_gpus
    else:
        gpus = 0
    return cpus, gpus


def parse_sacct_usage(text: str) -> list[UsageRow]:
    """Parse `sacct --parsable2` output into `UsageRow`s. Pure function.

    The expected column order is the one `SlurmBackend.usage` asks for:

        JobIDRaw|State|Partition|Start|End|ElapsedRaw|AllocTRES

    `--parsable2` separates fields with `|` and does not pad them. None of the
    requested columns can contain a `|` themselves, so a plain split is safe --
    but the split is capped at six so that anything unexpected lands inside
    AllocTRES rather than shifting every column left.

    Being a pure function is the point: usage accounting is the part of relay
    most likely to be quietly wrong, and "quietly wrong" is exactly what you
    cannot catch without fixture-driven tests. Feed it a string, get rows.

    What gets dropped, and why:

      * comment lines (`#`) -- relay's own fixture files carry a provenance
        header, and real sacct output never starts a line with `#`
      * blank lines
      * step rows (`12345.batch`, `12345.extern`, `12345.0`) -- `-X` should
        already suppress these, but sacct's flags differ between Slurm
        versions and a `.batch` row would otherwise be counted as a separate
        attempt with its own GPU-hours
      * rows with an empty Start -- an attempt that never started consumed
        nothing, and `UsageRow.start` is documented as never None

    What is kept: one row per *attempt*. A job preempted twice has three rows
    under one job ID, which is why `usage` passes `--duplicates` and why the
    store keys on `(job_id, start)`.
    """
    rows: list[UsageRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 6)
        if len(parts) < 7:
            # Not a data row: a stray error message from sacct on stdout, or a
            # header somebody left in. Skip rather than raise -- one odd line
            # must not cost the user every other row in the batch.
            continue

        job_id, raw_state, partition, raw_start, raw_end, raw_elapsed, tres = parts
        job_id = job_id.strip()
        if not job_id or "." in job_id:
            continue

        start = _sacct_time(raw_start)
        if start is None:
            continue

        cpus, gpus = parse_alloc_tres(tres)
        rows.append(
            UsageRow(
                job_id=job_id,
                state=_sacct_state(raw_state),
                partition=partition.strip() or None,
                start=start,
                end=_sacct_time(raw_end),
                elapsed_s=_to_int(raw_elapsed),
                gpus=gpus,
                cpus=cpus,
            )
        )
    return rows


def _sacct_state(raw: str) -> str:
    """The state word, upper-cased, with sacct's decorations removed.

    `CANCELLED by 12345` is one field: the state, then the UID of whoever ran
    scancel. Only the first word is the state. A state too wide for its column
    comes back truncated with a `+` (`CANCELLED+`), so that goes too.

    Kept raw rather than mapped through `map_slurm_state`: usage accounting
    has to tell a PREEMPTED attempt from a FAILED one in order to report
    wasted hours, and the status vocabulary deliberately folds them together.
    """
    head = raw.strip().split()[0] if raw.strip() else ""
    return head.rstrip("+").upper()


def _sacct_time(raw: str) -> str | None:
    """One sacct timestamp as an ISO-8601 string, or None if there isn't one.

    `usage` runs sacct with `SLURM_TIME_FORMAT=%s`, so these normally arrive
    as epoch seconds and come back out as unambiguous UTC:
    `1758045729` -> `"2025-09-16T18:22:09Z"`.

    That env var exists precisely to avoid the default format,
    `YYYY-MM-DDTHH:MM:SS`, which carries no zone at all and is in the
    *cluster's* local time. Storing those unlabelled next to relay's own
    UTC-with-Z event timestamps would produce a database where two columns
    look alike and differ by seven hours.

    The default format is still accepted, because a future relay or a hand-run
    sacct may produce it, and losing the row entirely would be worse. Such a
    value is passed through **unchanged and without a Z**: relay does not know
    the cluster's zone and must not invent one. Its naive shape is the honest
    signal that it is not UTC.

    An attempt still running has no end time. sacct spells that `Unknown`, and
    some versions `None` or an empty field.
    """
    value = (raw or "").strip()
    if not value or value in ("Unknown", "None", "N/A"):
        return None
    if value.lstrip("-").isdigit():
        moment = datetime.fromtimestamp(int(value), tz=timezone.utc)
        return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value


def _to_int(value: str) -> int:
    """An int, or 0 when the field is empty or not a number.

    sacct leaves fields blank for attempts it has no figure for. A blank
    ElapsedRaw means zero seconds used; raising there would lose every other
    row in the same batch.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------
# The default runner
# --------------------------------------------------------------------------


def _subprocess_runner(
    argv: list[str], input_bytes: bytes | None, timeout: float
) -> tuple[int, bytes, bytes]:
    """Run `argv`, feed it `input_bytes`, return (returncode, stdout, stderr).

    Bytes, not text. `read_bytes` pulls raw file contents through here and a
    half-written UTF-8 character at the end of a read would make `text=True`
    raise -- on data that is perfectly fine and will be complete next cycle.

    Exceptions (no ssh binary, timeout, OS refusing to fork) are allowed to
    escape; `_ssh` turns them into `TransportError` so that every caller in this
    module sees one failure type.
    """
    proc = subprocess.run(
        argv,
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def _build_tarball(files: dict[str, bytes]) -> bytes:
    """Pack `{name: contents}` into an uncompressed tar archive in memory.

    Uncompressed on purpose. These are three small text files; gzip would save
    a few kilobytes and add a dependency on the remote `tar` accepting `-z`,
    and bytes were never the expensive thing here -- round trips were.

    Every header field that does not have to vary is pinned: mtime 0, uid/gid
    0, empty owner names, mode 0644. Two effects, both wanted. The archive is
    reproducible, so a test can compare bytes and a human can diff two
    submits. And no ownership travels with the files -- `tar -x` run by a
    normal user ignores the stored uid anyway and creates everything as that
    user, which is exactly right on a shared filesystem with group quotas.

    Names are stored plain ("sidecar.py"), never absolute and never with a
    leading "./", so `tar -x -C <run_dir>` puts them exactly where meant.
    """
    buffer = io.BytesIO()
    # Sorted so the archive's byte layout depends only on its contents.
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in sorted(files):
            payload = files[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _decode(data: bytes) -> str:
    """Bytes from a remote command as text, never raising on bad encoding.

    Used for messages and for parsing `squeue`/`sacct` output, never for file
    contents. Slurm speaks ASCII, but a corrupted stream should produce an ugly
    error message rather than a UnicodeDecodeError inside the daemon's loop.
    """
    return data.decode("utf-8", errors="replace")


def _first_line(text: str) -> str:
    """The first non-blank line of `text`, for putting inside an error sentence."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _last_line(text: str) -> str:
    """The last non-blank line of `text`, for putting inside an error sentence.

    ssh's own failures belong at the *end* of its stderr, not the start. A
    connection that dies during setup prints banners, "debug1" chatter or
    warnings first and the sentence that actually explains the failure last --
    for example `unix_listener: cannot bind to path ...: No such file or
    directory`, which arrives after authentication has already succeeded. Using
    `_first_line` there would quote the least useful line ssh printed.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


# --------------------------------------------------------------------------
# Rendering the sbatch script
# --------------------------------------------------------------------------


def _sanitise_job_name(name: str) -> str:
    """Make a name safe to sit after `--job-name=` on an #SBATCH line.

    Slurm splits directive lines on whitespace, so a name with a space in it
    silently becomes a shorter name plus a stray argument. Replacing whitespace
    with underscores is the least surprising fix.
    """
    return "_".join(name.split()) or "relay"


def resolve_time_limit(spec: JobSpec) -> str:
    """Find this job's `--time` value, or refuse to render the script.

    One source, deliberately: `spec.time_limit`. The CLI fills it in from
    `relay submit --time` or, failing that, from `slurm.time` in the config,
    and that is the whole precedence story. It used to live here too -- the
    backend merged a `slurm.sbatch_defaults` dict on top of the spec -- and
    having two places decide the same question meant a script could end up
    with two `#SBATCH --time=` lines whose winner depended on dict ordering.

    So: global defaults reach the backend only through the JobSpec the CLI
    builds. One source of truth per script, no merging in two places. If you
    are tempted to teach this function about config again, teach the CLI
    instead.

    relay will not invent a number, and it will not let Slurm's silent
    `DefaultTime` stand in for one, because the failure mode of guessing wrong
    is a job killed hours into training.

    Raises `MissingTimeLimit`, which the CLI turns into exit code 2.
    """
    value = spec.time_limit
    if value is not None and str(value).strip():
        return str(value).strip()

    raise MissingTimeLimit(
        f"Run {spec.run_id} has no time limit, and relay will not submit a job "
        f"without one: partitions commonly default to a silent one-hour cap, so "
        f"a longer job would be killed with no explanation. Pass `--time "
        f"12:00:00` to `relay submit`, or set `slurm.time` in your relay config."
    )


def render_sbatch(
    spec: JobSpec,
    remote_python: str = "python3",
    grace_seconds: int = 60,
    account: str | None = None,
    partition: str | None = None,
) -> str:
    """Build the text of the sbatch script for one run.

    A pure function -- no ssh, no filesystem -- so the interesting question
    ("does this script say the right things?") can be answered by a test that
    compares strings, and so a human can print the script before submitting.

    `account` and `partition` are the backend's configured defaults; the
    matching fields on `spec` win when they are set, because a per-run choice
    should beat a config-file one. Everything else -- the time limit, the
    `--gres`, the passthrough `sbatch_extra` directives, the `setup` shell
    lines -- arrives already resolved on the spec, because the CLI is the one
    place that merges config with command-line flags.

    Raises `SbatchExtraError` for an `sbatch_extra` entry relay will not
    render, and `MissingTimeLimit` when the spec carries no time limit.
    """
    # Cheap, pure, and it runs before a single byte reaches shared storage.
    # The CLI already validated these entries; doing it again here means a
    # caller that skipped the CLI (a test, `doctor`, a future API) still
    # cannot write a script that fights relay's own directives.
    validate_sbatch_extra(spec.sbatch_extra or [])

    run_dir = spec.run_dir.rstrip("/") or "/"
    job_name = _sanitise_job_name(spec.name or spec.run_id)
    grace = spec.grace_seconds or grace_seconds

    effective_account = spec.account or account
    effective_partition = spec.partition or partition

    time_limit = resolve_time_limit(spec)

    lines: list[str] = []
    lines.append("#!/bin/bash")
    lines.append("#")
    lines.append(f"# Generated by relay for run {spec.run_id}. Safe to read, edit and")
    lines.append("# resubmit by hand; relay rewrites it on the next submit.")
    lines.append("")
    lines.append(f"#SBATCH --job-name={job_name}")

    # Only emit these when they are set. An empty `#SBATCH --account=` is not
    # the same as no line at all: Slurm reads it as an account named "" and
    # rejects the job.
    if effective_account:
        lines.append(f"#SBATCH --account={effective_account}")
    if effective_partition:
        lines.append(f"#SBATCH --partition={effective_partition}")

    # GPUs. Its own field rather than an `sbatch_extra` entry because nearly
    # every relay job asks for one, and because usage accounting will later
    # want to compare what was requested against what `sacct` says was
    # allocated. Written exactly as the user spelled it ("gpu:1", "gpu:a40:2").
    if spec.gres:
        lines.append(f"#SBATCH --gres={spec.gres}")

    # Start the job in its own run directory, so a relative path the training
    # script writes lands somewhere relay can find.
    lines.append(f"#SBATCH --chdir={run_dir}")

    # %j is Slurm's job number, so a requeued attempt (which gets a new job
    # number) writes a new file and the attempts stay distinguishable. stderr
    # is deliberately not split off with --error: the sidecar already tags every
    # line with its stream in the event log, and one interleaved file is what a
    # human would have seen in a terminal.
    lines.append(f"#SBATCH --output={run_dir}/slurm-%j.out")

    # An explicit time limit, always. Klone's checkpoint partitions have
    # DefaultTime=01:00:00, so an sbatch script with no --time is not
    # "unlimited" -- it is silently one hour, and a twelve-hour run dies at
    # sixty minutes with TIMEOUT and no explanation a user could act on.
    # `resolve_time_limit` has already refused to render at all if nobody said
    # how long the job may run.
    lines.append(f"#SBATCH --time={time_limit}")

    # Three directives that exist entirely to survive preemption and the time
    # limit:
    #
    #   --open-mode=append  without it, a requeued attempt *truncates* the
    #                       previous attempt's output file. The first attempt's
    #                       output would silently disappear at the exact moment
    #                       the user most wants to read it.
    #   --requeue           tells Slurm to put the job back in the queue when it
    #                       is preempted rather than ending it.
    #   --signal=B:USR1@N   send SIGUSR1 N seconds before the *time limit*.
    #                       `B:` sends it to the batch shell (our sidecar)
    #                       rather than only to the job steps, which is what
    #                       makes the sidecar's handler fire.
    #
    # USR1, not TERM, and this is the bug fix that everything else in this
    # round hangs off. `--signal` fires only before the time limit; it has
    # nothing to do with preemption, which sends SIGTERM at the moment of
    # eviction. And `scancel` also sends SIGTERM. So if relay asked Slurm to
    # send TERM here, all three situations -- time limit, preemption, cancel --
    # would arrive as the same signal and the sidecar could not tell them
    # apart. It would requeue a run the user had just cancelled.
    #
    # With USR1 the time-limit case is unambiguous: nothing else on the cluster
    # sends the sidecar USR1, so self-requeue on USR1 is always correct. The
    # sidecar spends N-2 of those seconds waiting for the training script to
    # checkpoint, which is why this number and the `--grace` on the exec line
    # below must be the same one.
    lines.append("#SBATCH --open-mode=append")
    lines.append("#SBATCH --requeue")
    lines.append(f"#SBATCH --signal=B:USR1@{grace}")

    # Passthrough directives, verbatim and in the order the user wrote them.
    # Last, so anything relay generates above is what a reader sees first, and
    # unreformatted, so `--constraint "a40|l40"` reaches sbatch exactly as it
    # was typed. `validate_sbatch_extra` above has already refused anything
    # that would collide with a directive relay owns.
    for entry in spec.sbatch_extra or []:
        lines.append(f"#SBATCH {entry.strip()}")

    if spec.env:
        lines.append("")
        lines.append("# Environment from the job spec.")
        for name in sorted(spec.env):
            # shlex.quote because these values are the user's: a password-like
            # token with a `$` or a path with a space must reach the process
            # unchanged, and this line is parsed by bash on the compute node.
            lines.append(f"export {name}={shlex.quote(spec.env[name])}")

    if spec.setup:
        lines.append("")
        lines.append("# Setup from the job spec, run verbatim before training.")
        # `set -e` so a failed `module load cuda` or `conda activate rl` ends
        # the job right there. Without it bash shrugs off the failure and the
        # run trains in whatever environment happened to be loaded -- which
        # looks like a successful run producing quietly wrong numbers, the
        # worst failure mode there is.
        lines.append("set -e")
        for line in spec.setup:
            lines.append(line)
        # ...and off again before the sidecar starts, so that `set -e` cannot
        # change how a nonzero exit is handled from here on. The sidecar's
        # whole job is to observe the training command's exit code and report
        # it; it must not be killed by the shell first.
        lines.append("set +e")

    lines.append("")
    lines.append("# The sidecar wraps the training command: it appends events to")
    lines.append("# events.jsonl in this directory, and on a signal it gives the")
    lines.append("# training script a chance to checkpoint before the job dies.")
    lines.append("# Its sidecar.json here says whether to requeue after a SIGTERM.")

    # `exec`, so the sidecar *replaces* the batch shell rather than running as
    # its child. Two reasons. Slurm's --signal=B:USR1 goes to the batch shell,
    # and after exec the sidecar *is* the batch shell, so it receives the signal
    # directly instead of hoping bash forwards it. And the job's exit status
    # becomes the sidecar's own, which is how the exit code of the training run
    # reaches sacct.
    sidecar_command = [
        remote_python,
        f"{run_dir}/{SIDECAR_FILENAME}",
        "--run",
        spec.run_id,
        "--dir",
        run_dir,
        "--grace",
        str(grace),
        "--heartbeat",
        str(spec.heartbeat_seconds),
        "--",
        *spec.command,
    ]
    lines.append("exec " + " ".join(shlex.quote(word) for word in sidecar_command))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------


class SlurmBackend(Backend):
    """Submit, watch, cancel and read jobs on a Slurm cluster over SSH."""

    def __init__(
        self,
        ssh_alias: str,
        remote_root: str,
        remote_python: str = "python3",
        account: str | None = None,
        partition: str | None = None,
        grace_seconds: int = 60,
        sidecar_path: str | Path | None = None,
        runner=None,
        ssh_timeout: float = SSH_TIMEOUT_DEFAULT,
    ) -> None:
        if "@" in ssh_alias:
            # Defence in depth: config.py already rejects this, but a backend
            # constructed directly must not be the way a hostname sneaks in.
            raise ValueError(
                f"ssh_alias {ssh_alias!r} contains '@', so it looks like a user@host "
                f"rather than the name of a Host block in your ~/.ssh/config. relay "
                f"hands this string straight to ssh so your own SSH settings supply "
                f"the username, hostname, key and ControlMaster socket."
            )
        self.ssh_alias = ssh_alias
        self.remote_root = remote_root
        self.remote_python = remote_python
        self.account = account
        self.partition = partition
        self.grace_seconds = grace_seconds

        # The one timeout this backend knows about. Named `ssh_timeout` rather
        # than `timeout` because it is the same number the config key, the
        # daemon and doctor use, and a shared name is what stops the four of
        # them drifting apart.
        self.ssh_timeout = ssh_timeout

        # Which local sidecar.py gets shipped. Imported lazily from the local
        # backend so there is exactly one answer to "where is the sidecar?".
        if sidecar_path is None:
            from relay.backends.local import default_sidecar_path

            sidecar_path = default_sidecar_path()
        self.sidecar_path = Path(sidecar_path)

        # The single seam where a real process would be started. Tests pass a
        # fake; nothing else in this module calls subprocess.
        self.runner = runner if runner is not None else _subprocess_runner

        # `scontrol show partition` answers, keyed by the partition name we
        # asked for (None meaning "the cluster default"). Cached for this
        # backend object's whole life, deliberately: partition configuration
        # changes about once a year, and the alternative is one extra SSH round
        # trip per submit on a Duo-protected cluster. One trip per partition
        # per process is the right price. A daemon restart re-reads it, which
        # is a fine invalidation story for something this static.
        self._partition_cache: dict[str | None, dict[str, str]] = {}

    @classmethod
    def from_config(cls, conf, runner=None) -> "SlurmBackend":
        """Build a backend from a loaded `config.Config`.

        `config.load` already refuses a `backend: slurm` config with no
        `ssh_alias` or `remote_root`, but this constructor can also be reached
        from tests and from `doctor`, so it checks rather than trusting.

        Only the cluster-shaped settings come through here: where the cluster
        is (`ssh_alias`, `remote_root`, `remote_python`), what to charge
        (`account`, `partition`), the grace period, and how long relay waits
        for one round trip (`ssh_timeout`). The rest of
        the `slurm:` block -- the time limit, `gres`, `sbatch_extra`, `setup`
        -- reaches the backend on the JobSpec, because the CLI is the one
        place that merges config with what the user typed. Reading config in
        two places is how a script ends up with two conflicting `--time`
        lines and nobody can say which one won.
        """
        if not conf.ssh_alias or not conf.remote_root:
            raise SlurmError(
                "The Slurm backend needs both ssh_alias and remote_root in your "
                "relay config. Run `relay init` and fill them in, or switch "
                "`backend` to local."
            )
        return cls(
            ssh_alias=conf.ssh_alias,
            remote_root=conf.remote_root,
            remote_python=conf.remote_python,
            account=conf.slurm.account,
            partition=conf.slurm.partition,
            grace_seconds=conf.slurm.grace_seconds,
            runner=runner,
            # `getattr` rather than `conf.ssh_timeout`: older configs (and the
            # handful of tests that build a Config by hand) may not carry the
            # key at all, and a backend that refuses to exist because nobody
            # set a timeout would be a silly way to fail. The default is the
            # same one `config.py` writes, so the two agree either way.
            ssh_timeout=getattr(conf, "ssh_timeout", SSH_TIMEOUT_DEFAULT),
        )

    # -- the one place that runs ssh ---------------------------------------

    def _ssh(
        self,
        remote_argv: list[str],
        input_bytes: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run one remote command. Returns (returncode, stdout, stderr).

        Raises `TransportError` when ssh itself could not do its job. Does *not*
        raise on a nonzero remote exit status -- that is a real answer from the
        far end, and each caller decides what it means (`read_bytes` treats 44
        as "no such file", `cancel` treats some stderr as harmless).

        `remote_argv` is the command as a list of words. ssh has no way to pass
        an argv vector: it joins everything after the alias with spaces and
        hands the resulting *string* to a remote shell. So every word is
        `shlex.quote`d here, once, in the one place that can guarantee it. That
        is what keeps a path with a space, or the `|` in `--format=%i|%T` (which
        an unquoted remote shell would read as a pipe), from being reinterpreted
        on the far side.

        A shell *snippet* (one with `;`, `&&` or a redirect in it) is not a
        word list and must not be quoted word by word -- `_ssh_script` takes
        those.

        `timeout` defaults to `self.ssh_timeout`. Callers pass a number only
        to ask for *more* than that, and only as a multiple of it.
        """
        # One join, one quote per word, and the result is a command line a
        # shell will re-split into exactly the words that went in.
        command = " ".join(shlex.quote(word) for word in remote_argv)
        return self._ssh_command(command, input_bytes, timeout)

    def _ssh_script(
        self,
        script: str,
        input_bytes: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run one remote *shell snippet* -- `;`, `&&`, redirects and all.

        `script` is handed to relay's remote shell verbatim, as its `-c`
        argument, so the snippet's own operators keep their meaning. Anything
        inside it that came from outside relay (a path, a job ID) must already
        be `shlex.quote`d by the caller, because this function cannot tell a
        deliberate `;` from one that arrived in a filename.

        There is exactly one shell on the far side, not two. An earlier
        version wrapped snippets in `sh -c` here and then `_ssh` quoted that
        into relay's `bash -c`, which meant every read of an event log forked
        two shells to run one `tail`.
        """
        return self._ssh_command(script, input_bytes, timeout)

    def _ssh_command(
        self,
        command: str,
        input_bytes: bytes | None,
        timeout: float | None,
    ) -> tuple[int, bytes, bytes]:
        """The single place a remote command line becomes a real ssh argv.

        `command` is a line a POSIX shell would execute. It is wrapped in
        relay's own clean shell -- `bash --noprofile --norc -c '<command>'`,
        from `relay.ssh.remote_shell_argv` -- and *that* argv is quoted word
        by word for ssh's own space-joining.

        Why the clean shell: every command in this file is relay's own probe,
        and none of them need the user's environment. Skipping their startup
        files skips whatever `conda init` and friends put in `~/.bashrc`,
        which on Klone is most of the 4 to 5 seconds a round trip costs.

        The honest caveat, copied from `relay.ssh`: this cleans the *inner*
        shell only. sshd still spawns a login shell to run the command relay
        sends, and on a bash built with SSH_SOURCE_BASHRC that outer shell
        sources `~/.bashrc` before it ever reaches our `bash --norc`. relay
        cannot do anything about that from this side; doctor's slow
        round-trip warning is what points at it.

        The training job gets none of this. It runs under Slurm from the
        rendered sbatch script, in the user's own environment, because it
        needs their modules and their conda env -- see `render_sbatch`.

        `build_ssh_argv` is the single place that knows relay's ControlMaster
        options (including `BatchMode=yes`, without which a daemon whose
        credentials have expired blocks forever on a Duo prompt nobody is
        watching -- and a hang is much worse than an error). `relay connect`
        and `doctor` build their argvs from the same module, so relay's three
        ways of touching ssh cannot drift apart and end up using two different
        control sockets.
        """
        remote = ssh_mod.remote_shell_argv(command)
        argv = ssh_mod.build_ssh_argv(
            # The alias goes through bare and unquoted. Never "user@host": see
            # the module docstring.
            self.ssh_alias,
            [shlex.quote(word) for word in remote],
        )
        # One number for every remote call in this file. `None` is the normal
        # case; a caller that passes something passes a multiple of it.
        wait = self.ssh_timeout if timeout is None else timeout
        try:
            returncode, stdout, stderr = self.runner(argv, input_bytes, wait)
        except subprocess.TimeoutExpired as exc:
            raise TransportError(
                f"ssh to {self.ssh_alias} did not answer within "
                f"{wait:.0f} seconds. If this "
                f"cluster needs a two-factor login, run `relay connect` to open the "
                f"shared connection.",
                # No returncode: ssh never finished, so there is nothing to
                # classify. The daemon treats this as `unreachable`.
                returncode=None,
            ) from exc
        except FileNotFoundError as exc:
            raise TransportError(
                "No `ssh` binary was found on your PATH, so relay cannot reach the "
                "cluster.",
                returncode=None,
            ) from exc
        except OSError as exc:
            raise TransportError(
                f"Could not start ssh: {exc}", returncode=None
            ) from exc

        if returncode == SSH_FAILURE_RETURNCODE:
            err = _decode(stderr)
            # ssh almost always says why it failed, and its own explanation is
            # far more useful than any guess relay can make. The first real run
            # on Klone failed with "unix_listener: cannot bind to path ...: No
            # such file or directory" while relay reported "check that plain
            # `ssh klone` works" -- advice that was not just useless but
            # actively wrong, because authentication had already succeeded.
            # So: quote ssh, and only fall back to advice when it said nothing.
            detail = _last_line(err)
            if detail:
                message = (
                    f"Could not reach {self.ssh_alias}: {detail}. "
                    f"Run `relay connect` if the SSH master is down."
                )
            else:
                message = (
                    f"Could not reach {self.ssh_alias}: ssh exited 255 with no "
                    f"message. Check that `ssh {self.ssh_alias}` works from this "
                    f"machine, and run `relay connect` if it needs a Duo prompt."
                )
            raise TransportError(
                message,
                returncode=returncode,
                stderr=err,
            )
        return returncode, stdout, stderr

    # -- partitions --------------------------------------------------------

    def partition_info(self, name: str | None) -> dict[str, str]:
        """What `scontrol show partition` says about one partition.

        Returns the `Key=Value` dict, or an empty dict if Slurm could not tell
        us (an unknown partition name, an old Slurm, a scontrol that is
        unhappy). An empty dict is a real answer here -- callers treat "we do
        not know this partition's PreemptMode" as a case to handle, not as an
        error to raise.

        `name=None` asks bare `scontrol show partition`, which prints every
        partition, and picks the one marked `Default=YES`. That is the
        partition an sbatch script with no `--partition` would land on, so it
        is the one whose PreemptMode applies.

        Cached per name for the life of this backend. One round trip per
        partition per process; see `_partition_cache`.
        """
        if name in self._partition_cache:
            return self._partition_cache[name]

        argv = ["scontrol", "show", "partition"]
        if name:
            argv.append(name)

        returncode, stdout, _stderr = self._ssh(argv)
        if returncode != 0:
            # Not raised on. This lookup is an optimisation -- it refines the
            # `auto` requeue decision -- and an unknown partition name will be
            # rejected by sbatch a moment later with a much better message than
            # anything this method could write. Caching the empty answer stops
            # a misconfigured partition costing a round trip on every submit.
            info: dict[str, str] = {}
        else:
            info = _select_partition(_decode(stdout), name)

        self._partition_cache[name] = info
        return info

    def _resolve_requeue(self, spec: JobSpec) -> str:
        """Turn `spec.requeue_on_term` into the `always`/`never` the sidecar reads.

        Costs one round trip, and only when the configured value is `auto` and
        this partition is not already in the cache. `always` and `never` are
        answers in themselves and never touch the network.
        """
        configured = (spec.requeue_on_term or "auto").strip().lower()
        if configured in ("always", "never"):
            return resolve_requeue_on_term(configured, None)

        effective_partition = spec.partition or self.partition
        preempt_mode = self.partition_info(effective_partition).get("PreemptMode")
        return resolve_requeue_on_term(configured, preempt_mode)

    # -- submit ------------------------------------------------------------

    def submit(self, spec: JobSpec) -> str:
        """Write the job's files to shared storage, sbatch them, return the job ID.

        Two SSH round trips, plus at most one more:

          1. `scontrol show partition` -- **only** when `requeue_on_term` is
             `auto`, and only the first time this process asks about a given
             partition. Cached afterwards.
          2. one `tar -x` carrying all three files into the run directory
          3. `sbatch job.sbatch`

        The files are `job.sbatch`, `sidecar.py`, `sidecar.json`, and --
        when this installation has one -- `relay_ckpt.py`.

        The tar stream replaced two separate `cat >` round trips, one per file.
        That was fine at two files; at three it was not, and the reason is Duo:
        every round trip on this cluster rides a multiplexed connection that a
        sleeping laptop can drop, and the fewer of them a `relay submit
        --seeds 0-4` makes, the fewer chances there are to fail halfway. tar
        also makes the write closer to atomic in the way that matters -- either
        the archive extracts or it does not, rather than leaving a run
        directory holding a script but no sidecar.

        The sbatch script is written as a *real file* on the shared filesystem
        and `sbatch` is pointed at that path, rather than piped into sbatch's
        stdin. CLAUDE.md requires this and the reason is debuggability: when a
        job behaves strangely at 2am, the user can `cat` the exact script that
        was submitted. The file is real even though its *contents* travel over
        ssh's stdin -- stdin is just how the bytes cross the network.
        """
        run_dir = spec.run_dir.rstrip("/") or "/"

        # Render first. `render_sbatch` raises MissingTimeLimit before anything
        # touches the network, so a run with no `--time` costs zero round trips
        # and the user gets the message immediately.
        script = render_sbatch(
            spec,
            remote_python=self.remote_python,
            grace_seconds=self.grace_seconds,
            account=self.account,
            partition=self.partition,
        )

        # Trip 0 (conditional): resolve `auto` against the partition's
        # PreemptMode. Done here, on the laptop, so the sidecar never has to
        # ask Slurm anything from a compute node -- and certainly not on the
        # signal path, where it has a couple of seconds before SIGKILL.
        requeue_on_term = self._resolve_requeue(spec)

        # The sidecar's config file: the decisions the sidecar cannot make for
        # itself. Whether to requeue after a SIGTERM, how long it has before
        # the SIGKILL lands, when this run's first attempt was submitted, and
        # the optional resume pair. `sort_keys` and a trailing newline so two
        # submits of the same run produce identical bytes.
        #
        # `first_attempt_start` is written once, here, and never rewritten:
        # Slurm requeues a job *in place*, reusing the same run directory and
        # the same files, so a later attempt never comes back through
        # `submit`. (LocalBackend has no scheduler to requeue it, so a
        # resubmit does come back through submit, and it has to take care to
        # carry this value forward.) The sidecar filters candidate checkpoint
        # files by mtime against this timestamp, which is how a resume ignores
        # files left in the directory by some earlier, unrelated run.
        #
        # `resume` is both halves or null. A glob with no argument has nothing
        # to say about the file it finds.
        if spec.resume_glob is None or spec.resume_arg is None:
            resume = None
        else:
            resume = {"checkpoint_glob": spec.resume_glob, "arg": spec.resume_arg}
        sidecar_config = json.dumps(
            {
                "requeue_on_term": requeue_on_term,
                "preempt_grace": spec.preempt_grace,
                "first_attempt_start": time.time(),
                "resume": resume,
            },
            sort_keys=True,
            indent=2,
        ) + "\n"

        try:
            sidecar_bytes = self.sidecar_path.read_bytes()
        except OSError as exc:
            raise SlurmError(
                f"Could not read relay's sidecar at {self.sidecar_path}: {exc}. "
                f"Your relay installation looks incomplete; reinstall it."
            ) from exc

        script_path = f"{run_dir}/{SBATCH_FILENAME}"

        # Trip 1: one tar stream, every file the job needs. `mkdir -p` is
        # idempotent, which matters because a resubmitted run reuses its
        # directory, and `tar -x -C` overwrites in place so re-submitting
        # refreshes the sidecar.
        members = {
            SBATCH_FILENAME: script.encode("utf-8"),
            SIDECAR_FILENAME: sidecar_bytes,
            SIDECAR_CONFIG_FILENAME: sidecar_config.encode("utf-8"),
        }
        # The checkpoint helper rides along in the same stream rather than in
        # a round trip of its own -- adding a file to a tar costs bytes, and
        # bytes were never the expensive thing here. Skipped silently when the
        # installation does not have one, because it is optional and a missing
        # helper must not cost the user a submit.
        # Imported lazily from the local backend, the same way `sidecar_path`
        # is, so there is exactly one answer to "where do relay's shipped
        # files live" and no import cycle between the two backends.
        from relay.backends.local import default_ckpt_helper_path

        ckpt_helper = default_ckpt_helper_path()
        if ckpt_helper.is_file():
            members[CKPT_HELPER_FILENAME] = ckpt_helper.read_bytes()
        tarball = _build_tarball(members)
        quoted_dir = shlex.quote(run_dir)
        # One shell snippet, so the `&&` is the remote shell's and not ssh's.
        # The default `ssh_timeout` is the whole budget here: these are three
        # small text files, and bytes were never the expensive thing -- if this
        # call is slow it is the login node or the filesystem, and waiting
        # longer would not help.
        returncode, _stdout, stderr = self._ssh_script(
            f"mkdir -p {quoted_dir} && tar -x -C {quoted_dir}",
            input_bytes=tarball,
        )
        if returncode != 0:
            raise SlurmError(
                f"Could not write run {spec.run_id}'s files into {run_dir} on "
                f"{self.ssh_alias}: "
                f"{_first_line(_decode(stderr)) or f'exit code {returncode}'}. "
                f"Check that remote_root exists and is writable."
            )

        # Trip 2: hand it to Slurm.
        # Twice the usual budget: this is the one call that waits on
        # slurmctld rather than on the login node, and a submit that times out
        # leaves relay unable to say whether the job was accepted. See
        # SBATCH_TIMEOUT_FACTOR.
        returncode, stdout, stderr = self._ssh(
            ["sbatch", script_path],
            timeout=SBATCH_TIMEOUT_FACTOR * self.ssh_timeout,
        )
        out = _decode(stdout)
        err = _decode(stderr)
        if returncode != 0:
            raise SlurmError(
                f"sbatch refused the job for run {spec.run_id}: "
                f"{_first_line(err) or _first_line(out) or f'exit code {returncode}'}. "
                f"The script relay submitted is at {script_path} on {self.ssh_alias}."
            )

        match = _SUBMITTED_RE.search(out)
        if not match:
            raise SlurmError(
                f"sbatch accepted the job but relay could not find a job ID in what "
                f"it printed: {out.strip()!r}. relay expects a line like "
                f"'Submitted batch job 12345'."
            )
        return match.group(1)

    # -- status ------------------------------------------------------------

    def status(self, job_ids: list[str]) -> dict[str, str]:
        """Map every requested job ID to one of `base.STATUSES`.

        One `squeue` call covers every job the user has queued or running. Jobs
        it does not mention get one batched `sacct` call, because `squeue` drops
        a job the instant it finishes -- without the fallback, every completed
        run would read as "unknown" forever.

        Two calls total, no matter how many runs. Zero calls for an empty list.
        """
        wanted = [str(job_id) for job_id in job_ids]
        if not wanted:
            return {}

        # Deduplicate but keep order, so the sacct command line is stable and a
        # test can assert on it.
        unique = list(dict.fromkeys(wanted))

        live = self._squeue()
        result = {job_id: live[job_id] for job_id in unique if job_id in live}

        missing = [job_id for job_id in unique if job_id not in result]
        if missing:
            finished = self._sacct(missing)
            for job_id in missing:
                # Still absent after sacct: Slurm's accounting database ages
                # rows out, so an old job genuinely has no answer. "unknown" is
                # honest and the daemon treats it as "no new information"
                # rather than as an ending.
                result[job_id] = finished.get(job_id, "unknown")

        return result

    def _squeue(self) -> dict[str, str]:
        """Ask about every live job of this user at once: {job_id: relay status}.

        `--me` rather than a `-j` list of IDs. Asking for specific IDs makes
        squeue exit nonzero when one of them has already finished, which is the
        normal case here, and then a routine poll looks like an error.

        `%i|%T` is job ID and state; `|` because it cannot appear in either
        field, while a space appears in plenty of other squeue columns.
        """
        returncode, stdout, stderr = self._ssh(
            ["squeue", "--me", "--noheader", "--format=%i|%T"]
        )
        if returncode != 0:
            raise SlurmError(
                f"squeue failed on {self.ssh_alias}: "
                f"{_first_line(_decode(stderr)) or f'exit code {returncode}'}."
            )
        return self._parse_state_lines(_decode(stdout))

    def _sacct(self, job_ids: list[str]) -> dict[str, str]:
        """Ask the accounting database about finished jobs: {job_id: relay status}.

        `-n` no header, `-P` parsable output with `|` separators and no padding,
        `-X` only the job allocation itself (not its `.batch` and `.extern`
        steps), `-j` a comma-separated list -- one call for every ID we still
        need an answer for.

        A nonzero exit is *not* raised on. sacct is unhappy about jobs it has
        never heard of, and a daemon polling a run whose ID has aged out of the
        database would then see an error every cycle forever. Whatever rows it
        did print are parsed; anything missing becomes "unknown" upstream.
        """
        returncode, stdout, _stderr = self._ssh(
            [
                "sacct",
                "-n",
                "-P",
                "-X",
                "-j",
                ",".join(job_ids),
                "--format=JobID,State",
            ]
        )
        if returncode != 0:
            return {}
        return self._parse_state_lines(_decode(stdout))

    @staticmethod
    def _parse_state_lines(text: str) -> dict[str, str]:
        """Parse `job_id|STATE` lines from squeue or sacct."""
        states: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            job_id, _, raw_state = line.partition("|")
            job_id = job_id.strip()

            # Drop sub-job rows: "12345.batch", "12345.extern", "12345.0".
            # `-X` should already have suppressed these, but sacct's flags vary
            # between Slurm versions and a `.batch` row's state would otherwise
            # overwrite the allocation's. Filtering defensively costs one `if`.
            if "." in job_id:
                continue
            if not job_id:
                continue
            states[job_id] = map_slurm_state(raw_state)
        return states

    # -- cancel ------------------------------------------------------------

    def cancel(self, job_id: str, run_dir: str | None = None) -> None:
        """Ask Slurm to stop the job.

        One round trip, with or without `run_dir`.

        `scancel` sends SIGTERM and then SIGKILL after the grace period, which
        is byte for byte the sequence a *preemption* produces. The sidecar
        cannot tell the two apart from the signal alone, and it must react
        differently: come back after a preemption, stay dead after a cancel. So
        when `run_dir` is given, relay drops a `CANCEL_REQUESTED` file in the
        run directory first and the sidecar looks for it before requeueing.

        Ordering is the whole point -- the file has to exist before the signal
        lands -- and it is preserved here without paying for a second round
        trip, by sequencing both commands inside one remote shell:

            bash --noprofile --norc -c \\
                'touch <run_dir>/CANCEL_REQUESTED; scancel <id>'

        `;` rather than `&&`: if the touch fails (an unwritable directory, a
        filesystem hiccup) the cancel must still go out. A job that keeps
        running because relay could not create a marker file would be a much
        worse outcome than a job that gets requeued once. The shell's exit
        status is the *last* command's, so it is scancel's, and the
        already-finished handling below still reads what it expects.

        Cancelling a job that has already finished is a no-op, not an error --
        the Backend contract requires it, because the daemon acts on a snapshot
        of the world that may be a second or two old.
        """
        if run_dir:
            marker = f"{run_dir.rstrip('/')}/{CANCEL_FILENAME}"
            returncode, _stdout, stderr = self._ssh_script(
                f"touch {shlex.quote(marker)}; scancel {shlex.quote(str(job_id))}"
            )
        else:
            returncode, _stdout, stderr = self._ssh(["scancel", str(job_id)])
        if returncode == 0:
            return

        err = _decode(stderr)
        lowered = err.lower()
        if any(phrase in lowered for phrase in _HARMLESS_CANCEL_ERRORS):
            return

        raise SlurmError(
            f"Could not cancel job {job_id}: "
            f"{_first_line(err) or f'scancel exited {returncode}'}."
        )

    # -- usage -------------------------------------------------------------

    def usage(self, job_ids: list[str]) -> list[UsageRow]:
        """Ask sacct what every attempt of every listed job consumed.

        One round trip for the whole list. Zero for an empty one.

        The command:

            env SLURM_TIME_FORMAT=%s sacct -j <ids> -X --duplicates \\
                --noheader --parsable2 \\
                --format=JobIDRaw,State,Partition,Start,End,ElapsedRaw,AllocTRES

        Each flag is load-bearing:

          `SLURM_TIME_FORMAT=%s`  makes sacct print Start and End as epoch
                seconds. sacct honours this environment variable, and it is
                the only clean way out of the default format
                `YYYY-MM-DDTHH:MM:SS`, which has no timezone on it at all and
                is in the *cluster's* local time. Epoch seconds are
                unambiguous, convert to UTC exactly, and survive the cluster
                and the laptop disagreeing about daylight saving. The `env`
                prefix rather than a shell assignment because `_ssh` sends an
                argv, not a shell line.
          `-X`  allocations only. Without it every job also returns `.batch`
                and `.extern` step rows, which would be counted as extra
                attempts and inflate every GPU-hour figure.
          `--duplicates`  every requeue attempt under the job ID, not just the
                latest. A run preempted three times ran three times and burned
                three attempts' worth of GPU time; without this flag relay
                would report only the last one.
          `--parsable2`  `|`-separated, unpadded, no trailing separator.
          `--noheader`  nothing to skip in the parser.

        Parsing lives in `parse_sacct_usage`, a pure function, so the awkward
        part is testable against fixtures without a cluster.
        """
        wanted = [str(job_id) for job_id in job_ids if str(job_id)]
        if not wanted:
            # No IDs, no questions. The daemon calls this on a timer, and a
            # timer that fires while nothing is running must not cost a round
            # trip on a Duo-protected connection.
            return []

        # Deduplicate but keep order, so the command line is stable and a test
        # can assert on it.
        unique = list(dict.fromkeys(wanted))

        returncode, stdout, _stderr = self._ssh(
            [
                "env",
                "SLURM_TIME_FORMAT=%s",
                "sacct",
                "-j",
                ",".join(unique),
                "-X",
                "--duplicates",
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,Partition,Start,End,ElapsedRaw,AllocTRES",
            ]
        )

        # A nonzero exit is not raised on, for the same reason `_sacct` does
        # not: sacct complains about IDs it has never heard of, and a job that
        # has aged out of the accounting database is a normal thing for a
        # long-lived daemon to ask about. Whatever rows it did print are still
        # good, so parse them either way.
        del returncode
        return parse_sacct_usage(_decode(stdout))

    # -- read_bytes --------------------------------------------------------

    def read_bytes(self, path: str, offset: int) -> tuple[bytes, int]:
        """Read `path` from `offset` to end of file, in one round trip.

        Returns `(data, offset + len(data))`.

        `tail -c +N` prints from byte N *counting from 1*, so the byte at
        zero-based offset N is the (N+1)th byte and the flag gets `offset + 1`.
        Off by one in the other direction re-reads one byte every cycle, which
        would corrupt the daemon's line splitting silently.

        Missing files are reported with exit status 44 rather than a marker in
        the output, so the answer can never be confused with the file's own
        contents. The daemon depends on `FileNotFoundError` staying distinct
        from an empty read: no file means the job has not started yet and is
        still queued, while no bytes means nothing new since last time.
        """
        start = max(0, int(offset))
        quoted = shlex.quote(path)
        script = (
            f"if [ -f {quoted} ]; then tail -c +{start + 1} {quoted}; "
            f"else exit {ENOENT_RETURNCODE}; fi"
        )
        returncode, stdout, stderr = self._ssh_script(script)

        if returncode == ENOENT_RETURNCODE:
            raise FileNotFoundError(f"{path} does not exist on {self.ssh_alias}")
        if returncode != 0:
            # Could be a permissions problem, a filesystem hiccup, or a login
            # node under pressure. All of them are worth retrying next cycle,
            # which is what TransportError means to the daemon.
            err = _decode(stderr)
            raise TransportError(
                f"Could not read {path} from {self.ssh_alias}: "
                f"{_first_line(err) or f'exit code {returncode}'}.",
                returncode=returncode,
                stderr=err,
            )

        return stdout, start + len(stdout)
