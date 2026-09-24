"""Loading, validating and creating relay's config file.

relay keeps exactly one config file, a small YAML document at
`~/.config/relay/config.yaml`, and one data directory, `~/.local/share/relay/`,
holding the SQLite database and the daemon's lock file. Both locations honour
the XDG environment variables when they are set, because that is what the rest
of the Unix world does and because it makes the test suite able to point relay
at a temporary directory instead of the developer's real home.

Two rules shape this module.

The first is that **the config never contains a username or a hostname.** It
stores an *SSH alias*: the name of a `Host` block in the user's own
`~/.ssh/config`. relay hands that string to the `ssh` binary and lets the
user's existing SSH configuration supply the user, the host, the key, the
jump host and — on a Duo-protected cluster like Hyak, where a fresh
authentication takes tens of seconds — the ControlMaster socket. Copying
`you@klone.hyak.uw.edu` into relay's config would duplicate configuration
that already exists, and would quietly bypass it. So `ssh_alias` containing an
`@` is a hard validation error, not a warning.

The second is that **every error out of here is a plain sentence** saying what
happened and what to do next. The CLI catches `ConfigError`, prints
`str(exc)` to stderr and exits. It catches `ConfigNotFound` separately, because
"you have not run `relay init` yet" is exit code 3 (not configured) while
"your YAML has a typo in it" is exit code 1 (runtime failure). That is the only
reason `ConfigNotFound` exists as its own class.

There is deliberately no `save()`. Config is written once by `relay init` and
edited by hand afterwards. A program that rewrites the user's config file also
has to preserve their comments and their key order, and that is a whole
problem (round-trip YAML) we do not need to have.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from relay import ssh

log = logging.getLogger("relay.config")

# --------------------------------------------------------------------------
# SSH timing
# --------------------------------------------------------------------------

# How long one remote round trip may take before relay gives up on it, in
# seconds. The number and the measurement that justifies it live together in
# ssh.py; this reads it from there rather than repeating the literal, so the
# value relay hands out and the comment explaining it cannot drift apart.
# ssh.py keeps it as a float because that is what `subprocess.run(timeout=)`
# wants; the config key is whole seconds, so it is narrowed once, here.
DEFAULT_SSH_TIMEOUT = int(ssh.DEFAULT_SSH_TIMEOUT)

# The lowest ssh_timeout that can possibly work. One round trip through an
# already-open ControlMaster on Klone measured 4 to 5 seconds -- none of it
# network time, all of it the login node starting a shell -- and a loaded
# login node is slower still. Below 5, relay would abandon commands that were
# about to succeed and report a healthy cluster as unreachable.
MIN_SSH_TIMEOUT = 5

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ConfigError(Exception):
    """Something is wrong with the config file, and the user must fix it.

    The message is shown to the user verbatim, so write it as a sentence:
    what happened, then what to do next.
    """


class ConfigNotFound(ConfigError):
    """There is no config file at all.

    A separate class so the CLI can map it to exit code 3 ("not configured")
    instead of lumping it in with parse and validation failures. It is a
    subclass, so callers that do not care can still catch `ConfigError`.
    """


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
#
# These are functions, not module-level constants. A constant would be
# evaluated at import time, which would freeze in whatever HOME and XDG_*
# happened to be set when the first `import relay.config` ran — and the tests
# set those variables *after* importing. Cheap call, no surprises.


def _xdg_dir(env_var: str, default_relative_to_home: str) -> Path:
    """Return an XDG base directory: the env var if set and absolute, else the default.

    The XDG spec says a relative path in one of these variables must be
    ignored, so we ignore it rather than resolving it against the current
    working directory (which would make relay's database location depend on
    where you happened to be standing when you ran it).
    """
    value = os.environ.get(env_var)
    if value:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
    return Path.home() / default_relative_to_home


def config_dir() -> Path:
    """Directory holding `config.yaml`, e.g. `~/.config/relay`."""
    return _xdg_dir("XDG_CONFIG_HOME", ".config") / "relay"


def config_path() -> Path:
    """The config file itself, e.g. `~/.config/relay/config.yaml`."""
    return config_dir() / "config.yaml"


def data_dir() -> Path:
    """Directory holding relay's own state, e.g. `~/.local/share/relay`.

    Config is what the user writes; data is what relay writes. Keeping them
    apart means a user can delete the database to start over without losing
    their cluster settings.
    """
    return _xdg_dir("XDG_DATA_HOME", ".local/share") / "relay"


def db_path() -> Path:
    """The SQLite database the daemon writes and the CLI and dashboard read."""
    return data_dir() / "relay.db"


def daemon_lock_path() -> Path:
    """The file the daemon holds an `flock` on to enforce a single instance."""
    return data_dir() / "daemon.lock"


def default_local_root() -> Path:
    """Where `LocalBackend` puts run directories when the config does not say.

    Derived from `data_dir()` so that setting `XDG_DATA_HOME` moves *all* of
    relay's state together, rather than moving the database but leaving run
    directories behind in the real home directory.
    """
    return data_dir() / "runs"


# --------------------------------------------------------------------------
# The config objects
# --------------------------------------------------------------------------
#
# Frozen dataclasses: a loaded Config is a snapshot of a file on disk, and
# nothing downstream should be mutating it. If the file changes, load it again.


@dataclass(frozen=True)
class SlurmConfig:
    """The `slurm:` section. Only meaningful when `backend` is "slurm"."""

    account: str | None = None
    partition: str | None = None

    # Wall-clock limit for the job, in Slurm's own `--time` format. Optional
    # here because `relay submit --time` can supply it instead; but one of the
    # two must be given, because Klone's checkpoint partitions have
    # DefaultTime=01:00:00 and a job submitted with no --time is silently
    # capped at one hour.
    time: str | None = None

    # How many seconds of warning relay asks Slurm for before the job hits its
    # *time limit*: it becomes `--signal=B:USR1@<grace>` in the sbatch script.
    #
    # This is NOT the preemption window - see `Config.preempt_grace` for that.
    # Slurm's `--signal` only ever fires ahead of the time limit; a preemption
    # arrives with no warning at all. The two are separate because the time
    # limit is predictable and generous (relay picks the number) while
    # preemption is sudden and short (the cluster picks the number).
    grace_seconds: int = 60

    # Generic resources, which is Slurm's spelling for GPUs: "gpu:1" or
    # "gpu:a40:2". Its own key rather than an `sbatch_extra` entry because it
    # is the one flag nearly every relay job needs - and because relay refuses
    # `--gres` inside `sbatch_extra` and points the user here instead, so that
    # a script can never end up with two conflicting --gres lines.
    gres: str | None = None

    # Any other `#SBATCH` directives, each written exactly as it would appear
    # after `#SBATCH `: ("--mem=64G", "--cpus-per-task=8"). This is the escape
    # hatch for the roughly one hundred sbatch flags relay does not model.
    #
    # A tuple of whole directives rather than a name -> value dict, because
    # then the user writes the flag exactly as they would in a real sbatch
    # script and relay never has to guess whether `--gres gpu:1` and
    # `--gres=gpu:1` are the same thing. A tuple, not a list, because Config
    # is frozen and a frozen object holding a mutable list is only half frozen.
    #
    # Shape only is checked here. Whether a particular flag is *allowed* -
    # relay owns --output, --signal, --requeue and friends - is decided by
    # `relay.backends.base.validate_sbatch_extra`, which the CLI calls at
    # submit. Keeping that out of config.py means this module stays a shape
    # validator and there is exactly one list of forbidden flags in the repo.
    sbatch_extra: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostConfig:
    """The `cost:` section. Turns GPU-hours into dollars, if the user says how.

    There is no default rate and there never will be one. A made-up price
    printed next to a real number reads as a real number, and a researcher
    quoting an invented figure in a grant report is a worse outcome than
    showing no figure at all. With `gpu_hour_rate` unset, relay shows hours
    everywhere and dollars nowhere.
    """

    gpu_hour_rate: float | None = None


@dataclass(frozen=True)
class ResumeConfig:
    """The `resume:` section. How a requeued attempt picks up where it left off.

    relay never writes checkpoints - the training script does. What relay can
    do is *find* the checkpoint again after a preemption and tell the script
    where it is. That takes two pieces of information, and neither one is
    useful alone:

      checkpoint_glob  where the script writes its checkpoints, as a glob
                       relative to the job's working directory. On attempts
                       after the first, relay picks the newest matching file
                       that was written since the run began.
      arg              the argument that names that file, containing the
                       literal `{path}` for relay to substitute. It is
                       appended to the training command, so the script sees
                       it exactly as if the user had typed it.

    Setting one without the other is an error rather than a warning: a glob
    with no arg finds a file and does nothing with it, and an arg with no glob
    has no file to name. Both are silent failures that only show up as a run
    quietly restarting from step zero, which is the exact thing this section
    exists to prevent.
    """

    checkpoint_glob: str | None = None
    arg: str | None = None


@dataclass(frozen=True)
class DaemonConfig:
    """The `daemon:` section. How often the syncing loop touches the cluster.

    Two schedules, not one, because the daemon asks two very different things
    of two very different pieces of shared infrastructure:

      the **tail** reads new bytes out of each run's `events.jsonl` on the
      shared filesystem. It is a `tail -c +N` on GPFS. It is cheap, it scales
      with how much the job is printing rather than with how often we ask, and
      nobody else on the cluster notices it. This is the one that can run every
      two seconds, and it is what makes the dashboard feel live.

      the **scheduler poll** asks `squeue` what the jobs are doing. That
      question goes to slurmctld, a single process serving the entire cluster.
      Most HPC sites ask users not to poll it more often than about every
      thirty seconds, and an unattended daemon polling every two seconds for a
      twelve-hour run would make roughly twenty thousand requests nobody asked
      for.

    So the tail keeps an adaptive 2/15/60 second schedule and the
    scheduler poll gets a floor of sixty seconds that no run state can talk it
    out of, plus a backoff that stretches towards `scheduler_interval_max`
    while nothing is changing. Job state is *slow-moving* data: a queued job
    that starts at 14:03 is no less started for being noticed at 14:03:58.

    `usage_interval` is the `sacct` accounting fetch, which is the most
    expensive question relay asks and the least urgent -- nobody watches
    GPU-hours tick up the way they watch a loss curve.
    """

    tail_interval_active: float = 2.0
    tail_interval_queued: float = 15.0
    tail_interval_idle: float = 60.0
    scheduler_interval_min: float = 60.0
    scheduler_interval_max: float = 300.0
    usage_interval: float = 300.0


@dataclass(frozen=True)
class Config:
    """A parsed, validated config file."""

    backend: str = "local"

    # Name of a Host block in ~/.ssh/config. Never "user@host". See the module
    # docstring for why.
    ssh_alias: str | None = None

    # Absolute path on the cluster's shared filesystem, under which relay
    # creates one directory per run. Must be visible from the compute nodes;
    # on Hyak that means somewhere under /mmfs1.
    remote_root: str | None = None

    # Interpreter used to run the sidecar on the cluster. Often a full path
    # into a module or conda environment rather than a bare name.
    remote_python: str = "python3"

    # Seconds relay allows for one remote round trip. Every remote call in
    # relay uses this one value - the Slurm backend, the daemon's polling loop,
    # each of doctor's checks - so that nothing ends up accidentally tighter
    # than the rest and no module carries a timeout literal of its own.
    #
    # It is not a network number. Through a live ControlMaster the connection
    # is already open and there is no handshake left to do; what takes the time
    # is the login node starting a shell to run the command. See
    # relay.ssh.DEFAULT_SSH_TIMEOUT for the Klone measurement behind the 30.
    ssh_timeout: int = DEFAULT_SSH_TIMEOUT

    # Where LocalBackend puts run directories. Stored expanded, so no consumer
    # has to remember to call expanduser().
    local_root: str = ""

    # What the sidecar should do when it is SIGTERMed. A SIGTERM means either
    # "you were preempted" or "somebody ran scancel", and the two are
    # indistinguishable at the moment the signal arrives, so this is a policy
    # the user sets rather than something relay can detect.
    #
    #   never   - never self-requeue on SIGTERM.
    #   always  - always self-requeue (unless relay itself wrote a
    #             CANCEL_REQUESTED file into the run directory first).
    #   auto    - resolved once, at submit time, by reading the partition's
    #             PreemptMode out of `scontrol show partition <p>`: if it
    #             contains REQUEUE then Slurm already requeues preempted jobs
    #             itself and self-requeuing would double up, so auto means
    #             never; otherwise auto means always. The resolved value is
    #             written into the run directory's sidecar.json, so the sidecar
    #             never has to ask Slurm anything on the signal path.
    #
    # On Klone, PreemptMode=REQUEUE cluster-wide, so `auto` resolves to `never`
    # and the sidecar does not call `scontrol requeue` on SIGTERM.
    requeue_on_term: str = "auto"

    # Seconds between the SIGTERM a *preempted* job receives and the SIGKILL
    # that follows. The cluster decides this number (partition GraceTime, or
    # KillWait cluster-wide when GraceTime is 0); relay only needs to know it so
    # the sidecar can budget. The sidecar waits `preempt_grace - 2` seconds for
    # the training script to checkpoint and keeps the remaining 2 to write and
    # flush its own `run_ended` event before SIGKILL reaches it too.
    #
    # Default 10, which is Klone's measured value: GraceTime=0 on ckpt-all and
    # KillWait=10 sec cluster-wide. `relay doctor` reports the real value.
    preempt_grace: int = 10

    # Shell lines run before the training command, in order: `module load
    # cuda`, `conda activate rl`, `export OMP_NUM_THREADS=8`. relay pastes
    # them into the job script verbatim and does not parse, expand or check
    # them - it cannot know what modules your cluster has. A tuple rather than
    # a list because Config is frozen, and a frozen object holding a mutable
    # list is only half frozen.
    #
    # A run's submit config may carry its own `setup`, and when it does it
    # replaces this one entirely rather than being appended to it. That
    # replacement is the CLI's job; config.py only parses what is on disk.
    setup: tuple[str, ...] = ()

    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)

    # How often the daemon polls the filesystem and the scheduler. Its own
    # section because the two schedules inside it answer to different costs;
    # see DaemonConfig.
    daemon: DaemonConfig = field(default_factory=DaemonConfig)

    def __post_init__(self) -> None:
        # The default depends on the environment at load time, which a
        # dataclass default cannot express. Empty string means "not given".
        if not self.local_root:
            # object.__setattr__ is how you assign to a frozen dataclass from
            # inside its own __post_init__; normal assignment would raise.
            object.__setattr__(self, "local_root", str(default_local_root()))


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

BACKENDS = ("local", "slurm")

# The three answers to "SIGTERM arrived, do we requeue ourselves?". See the
# comment on Config.requeue_on_term for what each one means.
REQUEUE_ON_TERM = ("auto", "always", "never")

# The lowest `preempt_grace` that means anything. The sidecar waits
# `preempt_grace - 2` seconds, floored at 1, so at 3 the wait is 1 second and
# at anything below 3 the floor takes over and the configured number stops
# having an effect. A setting that does nothing is worse than an error.
MIN_PREEMPT_GRACE = 3

# The absolute floor on how often relay will ask slurmctld anything, whatever
# the config says. Not a default -- a *limit*. The default is 30 and that is
# the number to leave alone; this is here so that a user who edits the number
# downwards in a hurry cannot accidentally point a 1-second polling loop at
# infrastructure the whole cluster depends on. A configured value below this is
# clamped up to it with a warning rather than rejected, because refusing to
# start the daemon over a polling interval would be a worse outcome than
# quietly being a good citizen.
MIN_SCHEDULER_INTERVAL = 10.0

_TOP_LEVEL_KEYS = frozenset(
    {
        "backend",
        "ssh_alias",
        "remote_root",
        "remote_python",
        "ssh_timeout",
        "local_root",
        "requeue_on_term",
        "preempt_grace",
        "setup",
        "slurm",
        "cost",
        "resume",
        "daemon",
    }
)
_SLURM_KEYS = frozenset(
    {"account", "partition", "time", "grace_seconds", "gres", "sbatch_extra"}
)
_COST_KEYS = frozenset({"gpu_hour_rate"})
_RESUME_KEYS = frozenset({"checkpoint_glob", "arg"})
_DAEMON_KEYS = frozenset(
    {
        "tail_interval_active",
        "tail_interval_queued",
        "tail_interval_idle",
        "scheduler_interval_min",
        "scheduler_interval_max",
        "usage_interval",
    }
)

# What `resume.arg` must contain, so relay has somewhere to put the checkpoint
# it found. A literal, not a format language: relay replaces exactly this
# substring and nothing else, so an arg containing other braces is left alone.
PATH_PLACEHOLDER = "{path}"

# Keys that used to exist and no longer do, mapped to a sentence telling the
# user what to write instead. Without this, upgrading relay turns a working
# config into a bare "unknown key" error and the user has to go and read the
# git history to find out what happened to their settings.
_RENAMED_KEYS: dict[str, str] = {
    "sbatch_defaults": (
        "'sbatch_defaults' has been replaced by 'sbatch_extra', which is a list "
        "of whole sbatch directives instead of a block of flag: value pairs. "
        "Rewrite\n"
        "  sbatch_defaults:\n"
        "    mem: \"32G\"\n"
        "as\n"
        "  sbatch_extra:\n"
        "    - \"--mem=32G\"\n"
        "and move any gres value to the 'gres' key."
    ),
}

# Slurm's `--time` accepts six shapes, and relay accepts exactly those six:
#
#   MM              minutes                       "30"
#   MM:SS           minutes and seconds           "30:00"
#   HH:MM:SS        hours, minutes, seconds       "04:00:00"
#   D-HH            days and hours                "2-12"
#   D-HH:MM         plus minutes                  "2-12:30"
#   D-HH:MM:SS      plus seconds                  "2-12:30:00"
#
# The trailing fields are one or two digits each; the first field can be as
# many as you like, because "1440" minutes and "36" hours are both legal.
# This is a syntax check, not a policy check: whether the cluster will accept
# the duration is between the user and their QOS.
_SLURM_TIME_RE = re.compile(
    r"""^(?:
          \d+                                # MM
        | \d+:\d{1,2}                        # MM:SS
        | \d+:\d{1,2}:\d{1,2}                # HH:MM:SS
        | \d+-\d{1,2}                        # D-HH
        | \d+-\d{1,2}:\d{1,2}                # D-HH:MM
        | \d+-\d{1,2}:\d{1,2}:\d{1,2}        # D-HH:MM:SS
    )$""",
    re.VERBOSE,
)


def _unknown_keys_message(found: dict, allowed: frozenset[str], where: str) -> str | None:
    """Return an error sentence if `found` has keys outside `allowed`, else None.

    Rejecting unknown keys is a choice, and it is the friendly one: a silently
    ignored `remote_pyton:` typo turns into a confusing failure much later,
    when relay tries to run `python3` on a cluster that does not have it.
    """
    unknown = sorted(set(found) - allowed)
    if not unknown:
        return None
    message = (
        f"Unknown {where} {'keys' if len(unknown) > 1 else 'key'} "
        f"{', '.join(repr(k) for k in unknown)} in {config_path()}. "
        f"Valid keys are: {', '.join(sorted(allowed))}. Remove or correct the key."
    )
    # A key that relay used to accept gets a sentence of its own, so a config
    # written against an older version says what to do rather than only what
    # is wrong.
    for key in unknown:
        if key in _RENAMED_KEYS:
            message += "\n\n" + _RENAMED_KEYS[key]
    return message


def _optional_str(raw: dict, key: str, where: str) -> str | None:
    """Read a key that may be absent or null, and must otherwise be a string."""
    if key not in raw or raw[key] is None:
        return None
    value = raw[key]
    # bool is a subclass of int, not str, so `backend: yes` (which YAML reads
    # as True) lands here and is correctly rejected as "not a string".
    if not isinstance(value, str):
        raise ConfigError(
            f"{where} '{key}' must be a string, but it is {type(value).__name__} "
            f"({value!r}) in {config_path()}. Quote the value or correct it."
        )
    stripped = value.strip()
    if not stripped:
        raise ConfigError(
            f"{where} '{key}' is empty in {config_path()}. "
            f"Give it a value or delete the line entirely."
        )
    return stripped


def _string_list(raw: dict, key: str, where: str, example: str) -> tuple[str, ...]:
    """Read a key that must be a YAML list of non-empty strings.

    Absent or null means an empty list, the same way every other optional key
    here treats null as "not given". Returns a tuple, because the config
    objects are frozen and a frozen object holding a mutable list would still
    let a caller change it.

    This checks the *shape* only. What the strings mean - whether a shell line
    will work on your cluster, whether an sbatch directive is one relay allows
    - is not decided here.
    """
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(
            f"{where} '{key}' must be a list, but it is {type(value).__name__} "
            f"({value!r}) in {config_path()}. Write it as a YAML list, one entry "
            f"per line:\n{example}"
        )
    items: list[str] = []
    for item in value:
        # bool before str is not the trap here (a bool is not a str), but an
        # unquoted YAML scalar easily becomes an int or a bool, so say which
        # type actually turned up.
        if not isinstance(item, str):
            raise ConfigError(
                f"{where} '{key}' contains {item!r} in {config_path()}, which is "
                f"{type(item).__name__} and not a string. Every entry is one line "
                f"of text; quote it so YAML keeps it as text."
            )
        if not item.strip():
            raise ConfigError(
                f"{where} '{key}' contains an empty entry in {config_path()}. "
                f"Give it a value or delete the entry."
            )
        # Stored exactly as written. These lines are pasted into a generated
        # script verbatim, so trimming or rewriting them here would mean the
        # user's file and the submitted script disagree.
        items.append(item)
    return tuple(items)


def _validate_ssh_alias(alias: str | None) -> str | None:
    """Reject anything that looks like a user@host instead of an alias."""
    if alias is None:
        return None
    if "@" in alias:
        raise ConfigError(
            f"ssh_alias {alias!r} contains '@', so it looks like a user@host rather "
            f"than an SSH alias. relay wants the name of a Host block in your "
            f"~/.ssh/config (for example 'klone') and hands it straight to ssh, so "
            f"that your own SSH settings supply the username, hostname, key and "
            f"ControlMaster socket. Add a Host block for this cluster and put its "
            f"name in {config_path()}."
        )
    if any(c.isspace() for c in alias):
        raise ConfigError(
            f"ssh_alias {alias!r} contains whitespace in {config_path()}. "
            f"An SSH alias is a single word, like 'klone'."
        )
    return alias


def _validate_slurm_time(value: str | None) -> str | None:
    """Check that a `--time` value is in one of Slurm's six accepted shapes.

    A time limit is not required in the config - `relay submit --time` can
    supply it - but a *wrong* one is worth catching here rather than letting
    sbatch reject the job twenty minutes into a Duo-protected SSH session.
    """
    if value is None:
        return None
    if not _SLURM_TIME_RE.match(value):
        raise ConfigError(
            f"slurm 'time' is {value!r} in {config_path()}, which is not a Slurm "
            f"time limit. Use one of: MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM or "
            f"D-HH:MM:SS - for example \"04:00:00\" for four hours or \"2-00\" for "
            f"two days. Quote the value so YAML keeps it as text."
        )
    return value


def _validate_slurm_section(raw: object) -> SlurmConfig:
    if raw is None:
        return SlurmConfig()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"The 'slurm' section of {config_path()} must be a block of settings, "
            f"but it is {type(raw).__name__}. See the comments in that file for the "
            f"expected shape."
        )

    message = _unknown_keys_message(raw, _SLURM_KEYS, "slurm")
    if message:
        raise ConfigError(message)

    account = _optional_str(raw, "account", "slurm")
    partition = _optional_str(raw, "partition", "slurm")
    time_limit = _validate_slurm_time(_optional_str(raw, "time", "slurm"))

    grace = raw.get("grace_seconds", 60)
    if grace is None:
        grace = 60
    # Again: bool first, because isinstance(True, int) is True and
    # `grace_seconds: yes` should not silently become one second.
    if isinstance(grace, bool) or not isinstance(grace, int):
        raise ConfigError(
            f"slurm 'grace_seconds' must be a whole number of seconds, but it is "
            f"{grace!r} in {config_path()}. Use something like 60."
        )
    if grace < 10:
        # This is the *time limit* warning: --signal=B:USR1@<grace>. The sidecar
        # forwards SIGTERM to the child and waits grace - 2 seconds for it to
        # checkpoint, so a single-digit grace leaves no useful window at all.
        raise ConfigError(
            f"slurm 'grace_seconds' is {grace} in {config_path()}, which is too "
            f"short: it is how much warning relay asks Slurm for before your job "
            f"hits its time limit, and the sidecar spends grace - 2 seconds of it "
            f"waiting for your training script to checkpoint. Use at least 10, and "
            f"60 unless you know your checkpoints are fast."
        )

    gres = _optional_str(raw, "gres", "slurm")

    # Shape only: a list of non-empty strings. Which flags relay refuses -
    # --output, --signal, --requeue, --time, --gres and the rest - is checked
    # by `relay.backends.base.validate_sbatch_extra`, which the CLI calls at
    # submit time. That keeps one list of forbidden flags in the repo instead
    # of two, and it means a config holding a bad directive still *loads*, so
    # `relay ls` keeps working while the user fixes it.
    sbatch_extra = _string_list(
        raw,
        "sbatch_extra",
        "slurm",
        '  sbatch_extra:\n    - "--mem=64G"\n    - "--cpus-per-task=8"',
    )

    return SlurmConfig(
        account=account,
        partition=partition,
        time=time_limit,
        grace_seconds=grace,
        gres=gres,
        sbatch_extra=sbatch_extra,
    )


def _validate_cost_section(raw: object) -> CostConfig:
    """Validate the `cost:` section, which is entirely optional."""
    if raw is None:
        return CostConfig()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"The 'cost' section of {config_path()} must be a block of settings, "
            f"but it is {type(raw).__name__}. It holds one key, gpu_hour_rate."
        )

    message = _unknown_keys_message(raw, _COST_KEYS, "cost")
    if message:
        raise ConfigError(message)

    rate = raw.get("gpu_hour_rate")
    if rate is None:
        return CostConfig()
    # bool before int, same trap as grace_seconds: `gpu_hour_rate: yes` would
    # otherwise become a rate of one dollar an hour.
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ConfigError(
            f"cost 'gpu_hour_rate' must be a number of currency units per "
            f"GPU-hour, but it is {rate!r} in {config_path()}. Write it as a bare "
            f"number, like 1.25, with no currency symbol."
        )
    if rate <= 0:
        raise ConfigError(
            f"cost 'gpu_hour_rate' is {rate} in {config_path()}, but it must be "
            f"greater than zero. To show no dollar figures at all, delete the key "
            f"instead of setting it to zero."
        )
    return CostConfig(gpu_hour_rate=float(rate))


def _validate_resume_section(raw: object) -> ResumeConfig:
    """Validate the `resume:` section, which is optional but all-or-nothing."""
    if raw is None:
        return ResumeConfig()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"The 'resume' section of {config_path()} must be a block of settings, "
            f"but it is {type(raw).__name__}. It holds two keys, checkpoint_glob "
            f"and arg."
        )

    message = _unknown_keys_message(raw, _RESUME_KEYS, "resume")
    if message:
        raise ConfigError(message)

    checkpoint_glob = _optional_str(raw, "checkpoint_glob", "resume")
    arg = _optional_str(raw, "arg", "resume")

    # Neither key set is the normal case: relay simply does not pass a resume
    # argument, and a script that handles its own restart still works.
    if checkpoint_glob is None and arg is None:
        return ResumeConfig()

    if arg is None:
        raise ConfigError(
            f"resume 'checkpoint_glob' is set to {checkpoint_glob!r} in "
            f"{config_path()} but 'arg' is not set. A glob on its own has nothing "
            f"to do with the file it finds: relay needs to know how to hand that "
            f"path to your training script. Add an 'arg' such as "
            f'"--resume {PATH_PLACEHOLDER}", or delete checkpoint_glob as well.'
        )
    if checkpoint_glob is None:
        raise ConfigError(
            f"resume 'arg' is set to {arg!r} in {config_path()} but "
            f"'checkpoint_glob' is not set. An argument on its own has no file to "
            f"name: relay needs a glob such as \"checkpoints/*.pt\" to find the "
            f"checkpoint your script wrote. Add a checkpoint_glob, or delete arg "
            f"as well."
        )

    if PATH_PLACEHOLDER not in arg:
        raise ConfigError(
            f"resume 'arg' is {arg!r} in {config_path()}, which does not contain "
            f"'{PATH_PLACEHOLDER}'. relay replaces {PATH_PLACEHOLDER} with the "
            f"checkpoint it found, so without it your script would be handed an "
            f'argument naming no file. Write it as "--resume {PATH_PLACEHOLDER}" '
            f'or "--resume={PATH_PLACEHOLDER}", matching whatever flag your script '
            f"already accepts."
        )

    return ResumeConfig(checkpoint_glob=checkpoint_glob, arg=arg)


def _positive_seconds(raw: dict, key: str, where: str, default: float) -> float:
    """One interval out of a `daemon:` block, as a positive number of seconds.

    Floats are accepted here, unlike `ssh_timeout` and `preempt_grace`, which
    insist on whole seconds. Those two are budgets measured against a real
    cluster's behaviour, where a fractional value means the user has
    misunderstood what the number is. These are polling intervals: 0.5 is a
    perfectly coherent thing to ask for in a test, and the code that consumes
    them is doing float arithmetic on a monotonic clock anyway.

    `bool` is rejected before `int` for the usual reason -- `isinstance(True,
    int)` is True, so `tail_interval_active: yes` would otherwise become a
    one-second poll.
    """
    value = raw.get(key, default)
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{where} '{key}' must be a number of seconds, but it is {value!r} in "
            f"{config_path()}. The default is {default:g}."
        )
    if value <= 0:
        raise ConfigError(
            f"{where} '{key}' is {value!r} in {config_path()}, but a polling "
            f"interval has to be a positive number of seconds. A zero or "
            f"negative interval would mean a loop with no sleep in it. The "
            f"default is {default:g}."
        )
    return float(value)


def _validate_daemon_section(raw: object) -> DaemonConfig:
    """Validate the `daemon:` section, which is entirely optional.

    The defaults in `DaemonConfig` are the values relay ships with, and most
    users should never write this section at all. It exists for two kinds of
    people: someone on a cluster whose admins have asked for gentler polling
    than a minute, and someone debugging who wants the dashboard to
    update faster than the tail's two.
    """
    if raw is None:
        return DaemonConfig()
    if not isinstance(raw, dict):
        raise ConfigError(
            f"The 'daemon' section of {config_path()} must be a block of settings, "
            f"but it is {type(raw).__name__}. It holds polling intervals in "
            f"seconds; see the comments in that file for the expected shape."
        )

    message = _unknown_keys_message(raw, _DAEMON_KEYS, "daemon")
    if message:
        raise ConfigError(message)

    defaults = DaemonConfig()
    tail_active = _positive_seconds(
        raw, "tail_interval_active", "daemon", defaults.tail_interval_active
    )
    tail_queued = _positive_seconds(
        raw, "tail_interval_queued", "daemon", defaults.tail_interval_queued
    )
    tail_idle = _positive_seconds(
        raw, "tail_interval_idle", "daemon", defaults.tail_interval_idle
    )
    sched_min = _positive_seconds(
        raw, "scheduler_interval_min", "daemon", defaults.scheduler_interval_min
    )
    sched_max = _positive_seconds(
        raw, "scheduler_interval_max", "daemon", defaults.scheduler_interval_max
    )
    usage = _positive_seconds(raw, "usage_interval", "daemon", defaults.usage_interval)

    sched_min = clamp_scheduler_interval(sched_min, source=str(config_path()))

    if sched_max < sched_min:
        raise ConfigError(
            f"daemon 'scheduler_interval_max' is {sched_max:g} in {config_path()}, "
            f"which is below 'scheduler_interval_min' ({sched_min:g}). The maximum "
            f"is the ceiling the poll backs off *towards* while nothing is "
            f"changing, so it cannot be smaller than the interval it starts at. "
            f"Either raise the maximum or lower the minimum."
        )

    return DaemonConfig(
        tail_interval_active=tail_active,
        tail_interval_queued=tail_queued,
        tail_interval_idle=tail_idle,
        scheduler_interval_min=sched_min,
        scheduler_interval_max=sched_max,
        usage_interval=usage,
    )


def clamp_scheduler_interval(value: float, *, source: str) -> float:
    """Hold `value` at or above `MIN_SCHEDULER_INTERVAL`, warning if it was under.

    Its own public function because the same clamp has to apply to two inputs
    that arrive by different routes -- the config file and `relay daemon
    --scheduler-interval-min` -- and a limit enforced in one of two places is
    not a limit. `source` names whichever one it was, so the warning tells the
    user which number to go and edit.

    A warning and a clamp rather than an error, because the cost of being
    wrong in each direction is not symmetric: refusing to start the daemon
    means the user sees nothing at all about their running jobs, while clamping
    means they see everything, slightly later than they asked.
    """
    if value >= MIN_SCHEDULER_INTERVAL:
        return value
    log.warning(
        "scheduler poll interval of %gs from %s is below relay's %gs floor; "
        "using %gs instead. `squeue` asks slurmctld, which every user on the "
        "cluster shares, and most sites ask for no more than one poll every "
        "30 seconds. Relay tails your event logs on its own faster schedule, "
        "so metrics stay live regardless of this number.",
        value,
        source,
        MIN_SCHEDULER_INTERVAL,
        MIN_SCHEDULER_INTERVAL,
    )
    return MIN_SCHEDULER_INTERVAL


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------


def load(path: Path | None = None) -> Config:
    """Read, validate and return the config.

    `path` exists so tests and the daemon can point at a specific file; normal
    callers pass nothing and get `config_path()`.

    Raises `ConfigNotFound` if the file does not exist and `ConfigError` for
    anything else that is wrong with it.
    """
    path = Path(path) if path is not None else config_path()

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigNotFound(
            f"No relay config found at {path}. Run `relay init` to create one, "
            f"then edit it to point at your cluster."
        ) from exc
    except IsADirectoryError as exc:
        raise ConfigError(
            f"{path} is a directory, but relay expects a YAML file there. "
            f"Move it aside and run `relay init`."
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"Could not read {path}: {exc.strerror or exc}. "
            f"Check the file's permissions."
        ) from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # YAML errors already carry line and column information, so pass the
        # original text through rather than replacing it with something vaguer.
        raise ConfigError(
            f"Could not parse {path} as YAML:\n{exc}\n"
            f"Fix the syntax, or delete the file and run `relay init` to start over."
        ) from exc

    # An empty file parses to None. That is a valid config: every field has a
    # default and the default backend is local, so relay works out of the box.
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a block of settings (key: value lines), but it "
            f"parsed as {type(raw).__name__}. Run `relay init --force` to write a "
            f"fresh starter config."
        )

    message = _unknown_keys_message(raw, _TOP_LEVEL_KEYS, "top-level")
    if message:
        raise ConfigError(message)

    backend = _optional_str(raw, "backend", "top-level") or "local"
    if backend not in BACKENDS:
        raise ConfigError(
            f"backend is {backend!r} in {path}, which relay does not know. "
            f"Use one of: {', '.join(BACKENDS)}."
        )

    ssh_alias = _validate_ssh_alias(_optional_str(raw, "ssh_alias", "top-level"))
    remote_root = _optional_str(raw, "remote_root", "top-level")
    remote_python = _optional_str(raw, "remote_python", "top-level") or "python3"

    ssh_timeout = raw.get("ssh_timeout", DEFAULT_SSH_TIMEOUT)
    if ssh_timeout is None:
        ssh_timeout = DEFAULT_SSH_TIMEOUT
    # bool before int, the same trap as preempt_grace: isinstance(True, int) is
    # True, so `ssh_timeout: yes` would otherwise become a one-second budget. A
    # float is rejected rather than rounded, because someone writing 0.5 here
    # thinks this is a network latency, and the error is the place to say that
    # it is not.
    if isinstance(ssh_timeout, bool) or not isinstance(ssh_timeout, int):
        raise ConfigError(
            f"ssh_timeout must be a whole number of seconds, but it is "
            f"{ssh_timeout!r} in {path}. It is how long relay waits for one "
            f"remote command to come back, and every remote call relay makes "
            f"uses it. The default is {DEFAULT_SSH_TIMEOUT}."
        )
    if ssh_timeout < MIN_SSH_TIMEOUT:
        raise ConfigError(
            f"ssh_timeout is {ssh_timeout} in {path}, which is below the "
            f"{MIN_SSH_TIMEOUT} second floor. This is not a network timeout: "
            f"even with relay's ControlMaster already open, one round trip "
            f"costs whatever the login node takes to start a shell, and on "
            f"Klone that measured 4 to 5 seconds on a quiet day. Below "
            f"{MIN_SSH_TIMEOUT} relay would give up on commands that were "
            f"about to succeed and report a healthy cluster as unreachable. "
            f"Use at least {MIN_SSH_TIMEOUT}, or {DEFAULT_SSH_TIMEOUT}."
        )

    local_root_raw = _optional_str(raw, "local_root", "top-level")

    requeue_on_term = _optional_str(raw, "requeue_on_term", "top-level") or "auto"
    if requeue_on_term not in REQUEUE_ON_TERM:
        raise ConfigError(
            f"requeue_on_term is {requeue_on_term!r} in {path}, which relay does "
            f"not know. Use one of: {', '.join(REQUEUE_ON_TERM)}. 'auto' decides at "
            f"submit time from the partition's PreemptMode and is what you want "
            f"unless you have a reason otherwise."
        )

    preempt_grace = raw.get("preempt_grace", 10)
    if preempt_grace is None:
        preempt_grace = 10
    # bool first: isinstance(True, int) is True, so `preempt_grace: yes` would
    # otherwise sail through as a one-second window.
    if isinstance(preempt_grace, bool) or not isinstance(preempt_grace, int):
        raise ConfigError(
            f"preempt_grace must be a whole number of seconds, but it is "
            f"{preempt_grace!r} in {path}. It is how long the cluster waits "
            f"between SIGTERM and SIGKILL when a job is preempted; on Klone that "
            f"is 10."
        )
    if preempt_grace < MIN_PREEMPT_GRACE:
        raise ConfigError(
            f"preempt_grace is {preempt_grace} in {path}, which is too short to "
            f"mean anything: the sidecar waits preempt_grace - 2 seconds for your "
            f"training script to checkpoint, floored at 1, so below "
            f"{MIN_PREEMPT_GRACE} the floor decides and your number is ignored. "
            f"Use at least {MIN_PREEMPT_GRACE}, or 10 on Klone. Run `relay doctor` "
            f"to see your cluster's real value."
        )

    # Shell lines to run before training. relay deliberately does not look
    # inside them: it has no way to know which modules your cluster has or
    # what your conda environments are called, and a check it cannot do
    # properly is worse than no check.
    setup = _string_list(
        raw,
        "setup",
        "top-level",
        "  setup:\n    - module load cuda/12.4\n    - conda activate rl",
    )

    slurm = _validate_slurm_section(raw.get("slurm"))
    cost = _validate_cost_section(raw.get("cost"))
    resume = _validate_resume_section(raw.get("resume"))
    daemon = _validate_daemon_section(raw.get("daemon"))

    if remote_root is not None and not remote_root.startswith(("/", "~")):
        raise ConfigError(
            f"remote_root is {remote_root!r} in {path}, but it must be an absolute "
            f"path on the cluster (for example /mmfs1/gscratch/yourgroup/you/relay). "
            f"A relative path would resolve differently depending on which "
            f"directory a job happens to start in."
        )

    # The slurm backend cannot do anything without a way in and a place to
    # write. Catching this at load time means the user finds out when they run
    # `relay doctor`, not twenty minutes later when a submit fails.
    if backend == "slurm":
        missing = [
            name
            for name, value in (("ssh_alias", ssh_alias), ("remote_root", remote_root))
            if value is None
        ]
        if missing:
            raise ConfigError(
                f"backend is 'slurm' but {' and '.join(missing)} "
                f"{'are' if len(missing) > 1 else 'is'} not set in {path}. "
                f"relay needs an SSH alias from your ~/.ssh/config and a directory "
                f"on the cluster's shared filesystem before it can submit anything. "
                f"Set {'them' if len(missing) > 1 else 'it'} and try again."
            )

    # Expand ~ and $VARS once, here, so that nothing downstream has to.
    if local_root_raw is None:
        local_root = str(default_local_root())
    else:
        local_root = str(Path(os.path.expandvars(local_root_raw)).expanduser())

    return Config(
        backend=backend,
        ssh_alias=ssh_alias,
        remote_root=remote_root,
        remote_python=remote_python,
        ssh_timeout=ssh_timeout,
        local_root=local_root,
        requeue_on_term=requeue_on_term,
        preempt_grace=preempt_grace,
        setup=setup,
        slurm=slurm,
        cost=cost,
        resume=resume,
        daemon=daemon,
    )


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------

# Written verbatim by `relay init`. It is mostly comments on purpose: this file
# is the one piece of relay the user is expected to open in an editor, so it
# doubles as the documentation for itself. Everything except `backend` is
# commented out, which means the file's contents and this module's defaults
# cannot drift apart — an omitted key is a default key.
STARTER_CONFIG = """\
# relay configuration
#
# Created by `relay init`. relay only ever reads this file; edit it by hand.
# Every setting below except `backend` is commented out and showing its
# default, so deleting a line is the same as accepting the default.

# Which backend runs your jobs.
#   local  - run on this machine as a subprocess. Needs no cluster; this is
#            what the test suite uses.
#   slurm  - submit to a cluster over SSH. Requires ssh_alias and remote_root.
backend: local

# Where the local backend creates run directories, one per run.
# local_root: ~/.local/share/relay/runs


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Shell lines run before your training command, in the order written. This is
# where `module load` and `conda activate` go; relay pastes the lines into the
# generated job script verbatim and never tries to parse them.
#
# The lines run under `set -e`, so a line that fails stops the job right there
# instead of letting training start in the wrong environment - a typo in a
# module name costs you a queue wait, not six hours of runs against the wrong
# CUDA.
#
# `relay submit` can pass its own setup list for one run, and when it does it
# replaces this one entirely rather than adding to it.
# setup:
#   - module load cuda/12.4
#   - conda activate rl


# ---------------------------------------------------------------------------
# Preemption
# ---------------------------------------------------------------------------

# What the sidecar does when the job is sent SIGTERM.
#
# A SIGTERM means one of two things and there is no way to tell them apart at
# the moment it arrives: either the job was preempted, or somebody ran
# scancel. So this is a policy you set, not something relay can detect.
#
#   auto    - decided once, at submit time, by reading PreemptMode out of
#             `scontrol show partition <your partition>`. If PreemptMode
#             contains REQUEUE then Slurm requeues preempted jobs by itself,
#             so relay does not (requeuing on top of that would double up) and
#             auto behaves like "never". Otherwise auto behaves like "always".
#   always  - always requeue on SIGTERM.
#   never   - never requeue on SIGTERM.
#
# On Klone PreemptMode=REQUEUE cluster-wide, so auto means never there: Slurm
# brings preempted jobs back on its own.
#
# `relay cancel` writes a CANCEL_REQUESTED file into the run directory before
# calling scancel, so a run you cancelled through relay is never requeued,
# whatever this is set to. A run cancelled with raw `scancel` under "always"
# can come back.
# requeue_on_term: auto

# Seconds the cluster waits between the SIGTERM a *preempted* job receives and
# the SIGKILL that ends it. The cluster sets this number; you are only telling
# relay what it is, so the sidecar can budget. It waits preempt_grace - 2
# seconds for your training script to checkpoint and keeps the last 2 to write
# and flush its own event log before SIGKILL lands.
#
# Klone: GraceTime=0 on the checkpoint partitions and KillWait=10 sec
# cluster-wide, so the whole window is 10 seconds. That is not much: your
# training script must checkpoint periodically on its own. `relay doctor`
# reports your cluster's real value.
# preempt_grace: 10


# ---------------------------------------------------------------------------
# Resume. Optional; set both keys or neither.
# ---------------------------------------------------------------------------

# For training scripts that already write their own checkpoints. On every
# attempt after the first, relay finds the newest file matching
# `checkpoint_glob` that was written since the run began, and appends `arg` to
# your training command with {path} replaced by that file. A script that does
# not checkpoint at all can `import relay_ckpt` instead, which adds saving and
# resuming for you - see the README.
#
# The glob is relative to the job's working directory - on Slurm that is the
# run directory (relay submits with --chdir), locally it is wherever you ran
# `relay submit` - and `**` is allowed ("runs/**/ckpt_*.pkl"). Write
# checkpoints relative to the current directory and one glob works in both
# places. Both keys go together: a glob with no arg has
# nothing to do with the file it finds, and an arg with no glob has no file to
# name, so relay refuses either on its own.
#
# `relay submit --checkpoint-glob PATTERN --resume-arg ARG` overrides both for
# a single run.
# resume:
#   checkpoint_glob: "checkpoints/*.pt"
#   arg: "--resume {path}"


# ---------------------------------------------------------------------------
# Cost. Optional, and there is no default rate.
# ---------------------------------------------------------------------------

# Price of one GPU-hour, as a bare number in whatever currency you think in.
# With this unset, relay shows hours everywhere and dollars nowhere; it will
# never invent a price for you.
# cost:
#   gpu_hour_rate: 1.25


# ---------------------------------------------------------------------------
# Daemon polling. Optional; the defaults are the right answer for most people.
# ---------------------------------------------------------------------------

# The daemon does two different jobs on two different schedules, because they
# cost two different things.
#
# The TAIL reads new bytes out of each run's events.jsonl on the shared
# filesystem. It is cheap, it bothers nobody else on the cluster, and it is
# what makes metrics show up live. It follows the state of your runs: fast
# while something is producing output, slower while everything is queued,
# slowest when there is nothing to watch.
#
# The SCHEDULER POLL asks `squeue` what your jobs are doing. That question
# goes to slurmctld, a single process serving every user on the cluster, so
# relay holds it to at least scheduler_interval_min no matter what your runs
# are doing - and stretches towards scheduler_interval_max, multiplying by 1.5
# each time, while no job changes state. It snaps back to the minimum the
# moment anything moves, and `relay submit` and `relay cancel` reset it too,
# so a job you just submitted is noticed promptly.
#
# Job state is slow-moving data. A job that starts at 14:03:00 is no less
# started for being noticed at 14:03:58, and your metrics are live regardless
# because they come from the tail.
#
# relay will not poll the scheduler more often than every 10 seconds whatever
# you put here; a smaller number is clamped with a warning.
#
# usage_interval is the `sacct` accounting fetch: the most expensive question
# relay asks and the least urgent. Relay also takes one final reading the
# moment a run finishes, whatever this is set to.
#
# `relay daemon` takes the same six values as flags (--tail-interval-active,
# --scheduler-interval-min and so on), and a flag beats this file.
# daemon:
#   tail_interval_active: 2
#   tail_interval_queued: 15
#   tail_interval_idle: 60
#   scheduler_interval_min: 60
#   scheduler_interval_max: 300
#   usage_interval: 300


# ---------------------------------------------------------------------------
# Cluster settings. Ignored while backend is "local".
# ---------------------------------------------------------------------------

# The name of a Host block in your ~/.ssh/config, for example "klone".
#
# Not "user@hostname". relay passes this string straight to the ssh binary so
# that your own SSH config supplies the username, the hostname, the key, any
# jump host, and - on a cluster with two-factor login - the ControlMaster
# socket that lets relay poll without re-authenticating every time.
# ssh_alias: klone

# Absolute path on the cluster's *shared* filesystem where relay writes one
# directory per run. It must be visible from the compute nodes, not just from
# the login node, because that directory is the only channel a running job has
# back to your laptop.
# remote_root: /mmfs1/gscratch/yourgroup/you/relay

# The Python that runs relay's sidecar on the cluster. Often a full path into
# a module or conda environment rather than a bare name.
# remote_python: python3

# Seconds relay allows for one remote command to come back. Every remote call
# uses this one number: submitting a job, the daemon's polling, each of
# `relay doctor`'s checks.
#
# This is not a network number. Through relay's ControlMaster the connection
# is already open, so what you are waiting for is the login node starting a
# shell to run the command - on Klone that measured 4 to 5 seconds with the
# master up, thanks to a conda init in ~/.bashrc, and a busy login node is
# worse. Raise this if `relay doctor`'s ssh check warns that your login node
# is slow and you cannot trim your ~/.bashrc. The minimum is 5.
# ssh_timeout: 30

# slurm:
#   # Slurm account to charge the job to.
#   account: your-account
#
#   # Partition to submit to. relay has no built-in list of partition names;
#   # whatever you put here is what it submits to. On Klone the preemptible
#   # partitions are:
#   #   ckpt      the cluster default; older GPUs (2080 Ti, RTX 6000, A40, A100)
#   #   ckpt-g2   newer GPUs (L40, L40S, H200) and the newer CPU nodes
#   #   ckpt-all  the union of the two, and the usual choice for GPU work
#   # Jobs on these can be preempted at any moment, including seconds after
#   # they start, with a 10 second window before SIGKILL.
#   partition: ckpt-all
#
#   # Wall-clock limit for the job, in Slurm's format. One of:
#   #   MM  MM:SS  HH:MM:SS  D-HH  D-HH:MM  D-HH:MM:SS
#   # Quote it, or YAML may read 4:00:00 as a number.
#   #
#   # You can also pass --time to `relay submit`, but one of the two is
#   # required: submit refuses to run without a time limit. Klone's checkpoint
#   # partitions have DefaultTime=01:00:00, so a job submitted with no --time
#   # is silently capped at one hour and killed there.
#   time: "04:00:00"
#
#   # How much warning relay asks Slurm for before the job hits the *time
#   # limit* above: it becomes --signal=B:USR1@<grace_seconds>. The sidecar
#   # uses that warning to have your script checkpoint, then requeues the job
#   # so it picks up where it left off.
#   #
#   # This is NOT the preemption window - see preempt_grace above. Slurm's
#   # --signal only fires ahead of the time limit; preemption gives no warning.
#   grace_seconds: 60
#
#   # GPUs, in Slurm's --gres spelling: "gpu:1" for any one GPU, or
#   # "gpu:a40:2" to ask for a specific model. Its own key because nearly
#   # every relay job needs it, and because relay refuses --gres inside
#   # sbatch_extra below and sends you here instead.
#   gres: "gpu:1"
#
#   # Any other #SBATCH directives, each written exactly as it would appear
#   # after `#SBATCH ` in a script you wrote yourself. Pasted in verbatim, in
#   # order. This is the escape hatch for every sbatch flag relay does not
#   # model. Quote each entry so YAML keeps it as text.
#   #
#   # relay refuses some flags here and says so at submit time:
#   #   --output, --error, --open-mode, --signal, --requeue, --job-name
#   #     relay writes these itself; overriding one breaks log continuity,
#   #     time-limit handling, requeue, or the link between `squeue` and
#   #     `relay ls`.
#   #   --account, --partition, --time, --gres
#   #     these have their own keys above. Two conflicting values in one
#   #     script mean the last one silently wins, so relay allows only one.
#   sbatch_extra:
#     - "--mem=64G"
#     - "--cpus-per-task=8"
"""


def init(force: bool = False, path: Path | None = None) -> Path:
    """Write a starter config file and return its path.

    Refuses to clobber an existing file unless `force` is true, because that
    file may hold cluster settings the user spent time getting right and there
    is no undo.
    """
    path = Path(path) if path is not None else config_path()

    if path.exists() and not force:
        raise ConfigError(
            f"A relay config already exists at {path}. Edit it directly, or run "
            f"`relay init --force` to overwrite it with a fresh starter config."
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # write_text truncates an existing file, which is what --force means.
        path.write_text(STARTER_CONFIG, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Could not write {path}: {exc.strerror or exc}. "
            f"Check that the directory is writable."
        ) from exc

    return path
