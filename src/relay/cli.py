"""The command line: argument parsing, dispatch, and formatting.

This is the last module in relay and the only one that knows about all the
others. Everything it does falls into three jobs, and keeping them separate is
what keeps this file from turning into a pile of special cases.

**1. Parse arguments.** Plain `argparse` from the standard library. The
sophisticated alternative is `click`, which gives nicer nested commands and
automatic type coercion; argparse gives us subcommands, `--` passthrough and
`SystemExit(2)` on bad usage for free, and it is one less dependency to install
on a laptop.

**2. Run the command.** A command function takes the parsed arguments, does the
work, and returns an `Output`: a plain Python data structure plus the function
that can render it for a human. A command never prints a table and it never
calls `json.dumps`.

**3. Format once, at the boundary.** `_emit` takes an `Output` and either dumps
`output.data` as JSON (when `--json` was given) or hands it to the renderer.
That is the "one code path" rule from CLAUDE.md, and it is not a style
preference: two code paths means `relay ls --json` can silently disagree with
`relay ls`, and the disagreement will be discovered by a script at 3am rather
than by a test. Because `--json` and the human table are both built from the
*same* dict, a test can fake that dict once and check that both renderings
change together.

**Errors never reach the user as a traceback.** `main` wraps the whole dispatch
in one `try`, maps the exception to an exit code, and prints `str(exc)` — which
is why every error message relay raises is written as a sentence saying what
happened and what to do next. A traceback escaping to a user is a bug in this
file, not a bug the user should have to read.

Which commands need a config file:

  * `ls`, `show`, `logs`, `attach` read *only* the local SQLite database. The
    database path comes from the XDG environment, not from the config file, so
    these work on a machine that has never run `relay init`. This is on
    purpose: looking at data you already collected should never require a
    working cluster connection.
  * `usage` reads the database too. It *looks* for a config, but only to find
    `cost.gpu_hour_rate`: without one it prints hours and no dollars, rather
    than refusing to run.
  * `submit`, `cancel`, `daemon` need the config, because they have to build a
    backend. No config means exit code 3. So does `connect`, which needs to
    know which SSH alias to open.
  * `dash` serves the dashboard straight out of the database, so it starts
    without a config; it only loads one if somebody presses Cancel (see
    `_cancel_run`).
  * `doctor` and `init` handle their own missing-config case by design.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import posixpath
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from relay import config as cfg
from relay import daemon as daemon_module
from relay import doctor as doctor_module
from relay import ssh as ssh_module
from relay.backends.base import JobSpec, SbatchExtraError, validate_sbatch_extra
from relay.store import Store

# --------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------
#
# Named constants rather than bare integers, because these are a contract: a
# user's shell script branches on them, and `relay doctor` is meant to work as
# a CI smoke test. Code 2 is produced by argparse itself, which is why nothing
# here raises it directly except `BadUsage`.

EXIT_OK = 0
EXIT_FAILURE = 1  # something went wrong at runtime
EXIT_USAGE = 2  # the command line itself was wrong
EXIT_NOT_CONFIGURED = 3  # no config file; run `relay init`
EXIT_TRANSPORT = 4  # ssh/the cluster could not be reached
EXIT_NOT_FOUND = 5  # no such run

# Ctrl-C. Not in CLAUDE.md's table because it is not a relay outcome: 128 + the
# signal number is the shell convention for "killed by SIGINT", and `relay logs
# --follow` is *meant* to be stopped that way.
EXIT_INTERRUPTED = 130

# How often `--follow` and `attach` re-read the database. Two seconds matches
# the daemon's active polling interval, so following adds no latency worth
# worrying about and costs one cheap SELECT.
FOLLOW_INTERVAL_S = 2.0

# Heartbeat period handed to the sidecar. The config has no setting for it (the
# daemon's staleness threshold is ten times this, so there is nothing to tune),
# so the JobSpec default is the single source of truth.
DEFAULT_HEARTBEAT_SECONDS = JobSpec.heartbeat_seconds

# How many metrics `relay ls` shows per run before giving up on the width.
LS_METRIC_COLUMNS = 3


# --------------------------------------------------------------------------
# Errors owned by the CLI
# --------------------------------------------------------------------------


class NotFound(Exception):
    """The user named a run that is not in the database. Exit code 5.

    Local to the CLI on purpose: the store returns `None` for a missing run
    because "no such row" is not an error at the storage layer, and only the
    command line has an opinion about what to do with it.
    """


class BadUsage(Exception):
    """The arguments parsed, but they do not mean anything. Exit code 2.

    argparse covers unknown flags and missing positionals; this covers the
    cases only relay can judge, such as `--seeds 5-1`.
    """


class ConnectFailed(Exception):
    """`relay connect` could not leave a working SSH master behind. Exit 4.

    Its own class rather than reusing `OSError` (which also maps to exit 4)
    because nothing here is an operating-system error: ssh ran fine and told
    us it failed, or it claimed success and left no socket.
    """


# --------------------------------------------------------------------------
# Output: data plus the way to draw it
# --------------------------------------------------------------------------


@dataclass
class Output:
    """What a command produced.

    `data` is the answer, as plain dicts and lists — the thing `--json` prints
    and the thing the renderer draws. `renderer` turns it into lines of text
    for a human. `exit_code` exists for commands whose *result* decides the
    exit status rather than whether they crashed; `doctor` is the only one.

    A command that streams (`logs --follow`, `attach`) calls `_emit` itself for
    each batch and returns None, which tells the dispatcher there is nothing
    left to print. It still goes through `_emit`, so streaming and one-shot
    output cannot drift apart.
    """

    data: Any = None
    renderer: Callable[[Any], Iterable[str]] | None = None
    exit_code: int = EXIT_OK


def _emit(output: Output, as_json: bool, stream=None) -> None:
    """Render one `Output`: JSON when asked, otherwise the human rendering.

    The single formatting boundary in relay. Every command's output passes
    through here exactly once.
    """
    stream = stream if stream is not None else sys.stdout
    if as_json:
        # default=str so a stray datetime or Path in a payload degrades to a
        # string instead of crashing the command that was trying to report it.
        stream.write(json.dumps(output.data, indent=2, default=str) + "\n")
    elif output.renderer is not None:
        for line in output.renderer(output.data):
            stream.write(line + "\n")
    stream.flush()


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render rows as a left-aligned table with two spaces between columns.

    Hand-rolled rather than pulled from a library: it is fifteen lines, and a
    table library would be a dependency that ships to every user so that relay
    can draw a box it does not want to draw anyway.
    """
    if not rows:
        return []
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    return lines


def _number(value: Any) -> str:
    """Format a metric value compactly: 0.4123, 18400, 1.2e-05."""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _hours(seconds: Any) -> str:
    """Seconds of wall clock as hours, for a table cell."""
    if seconds is None:
        return "-"
    return f"{float(seconds) / 3600.0:.2f}"


def _hours_text(hours: Any) -> str:
    """A number that is *already* in hours, rounded the same way."""
    return f"{float(hours or 0.0):.2f}"


def _money(amount: float) -> str:
    """A cost, shown only when the user configured `cost.gpu_hour_rate`.

    No currency symbol beyond the dollar sign and no locale handling: the rate
    is whatever unit the user typed into their config, and pretending to know
    which currency that is would be inventing information.
    """
    return f"${amount:,.2f}"


def _format_age(seconds: float | None) -> str:
    """Turn an age in seconds into something a human reads at a glance."""
    if seconds is None:
        return "never"
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _age_seconds(iso_ts: str | None) -> float | None:
    """Seconds since an ISO-8601 timestamp, or None if absent or unparseable.

    Unparseable returns None rather than raising: a corrupt timestamp should
    cost the user one "never" in a status line, not their whole `relay ls`.
    """
    if not iso_ts:
        return None
    try:
        parsed = datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds()


def _sync_line(data: dict) -> str:
    """The freshness line at the bottom of `relay ls` and `relay usage`.

    Two ages, not one, because the daemon learns the two halves of what is on
    screen from two different places on two different schedules:

      * **metrics** come from tailing each run's `events.jsonl` on the shared
        filesystem. That is cheap, it bothers nobody else on the cluster, and
        it runs every two seconds while a job is producing output.
      * **job state** comes from `squeue`, which asks slurmctld — one process
        serving every user on the cluster. Relay holds that to a minute
        at the very least and lets it back off towards five minutes while no
        job changes state.

    Collapsing the two into one "last synced" number would mean printing the
    older of them, and that would make a perfectly healthy daemon look stalled:
    a job-state figure four minutes old is the design working, not a symptom.
    Printing both says which half is old, so the user can tell "relay has not
    looked at the scheduler lately, which is normal" from "relay has stopped".

    Stale data the user knows is stale is fine. Stale data presented as current
    is not — so when the daemon has never synced at all we say so, and point at
    the reason.
    """
    if data.get("last_synced") is None:
        return "last synced never — is `relay daemon` running?"
    metrics = _format_age(data.get("last_synced_age_s"))
    if data.get("active_runs") == 0:
        # Nothing to tail and nothing to ask the scheduler about, so neither
        # age means anything: the daemon is cycling locally and touching the
        # cluster not at all. Two fresh-looking ages here read as "relay is
        # polling for nothing", which is exactly what it is not doing.
        return f"no active runs · daemon alive {metrics} ago"
    if data.get("last_scheduler") is None:
        # The daemon is running and tailing, but has not managed a scheduler
        # poll yet: a fresh daemon on its first cycle, or one whose `squeue`
        # calls are all failing. Either way "0s ago" would be a lie.
        return f"metrics {metrics} ago · job state never polled"
    scheduler = _format_age(data.get("last_scheduler_age_s"))
    return f"metrics {metrics} ago · job state {scheduler} ago"


def _sync_footer(store: Store) -> dict:
    """The "how fresh is this?" fields every listing ends with.

    One helper, used by `ls` and `usage`, so the two cannot drift into
    disagreeing about how recent relay's data is. It goes into the *data*, not
    into a rendering: `--json` carries the same keys the table shows, which is
    what lets a script notice a stalled daemon too.

    `last_synced` and `last_scheduler` are the two clocks `_sync_line`
    explains. Both are here rather than one combined number because a script
    watching for a job to start cares about the scheduler clock, while one
    watching a loss curve cares about the tail's.
    """
    last_synced = store.last_synced()
    last_scheduler = store.last_scheduler_synced()
    return {
        # How many runs the daemon is actually watching. Zero changes what the
        # two ages below mean -- see `_sync_line` -- and a script can use it
        # to tell "idle" from "stalled" without parsing the human line.
        "active_runs": len(store.list_runs(active_only=True)),
        "last_synced": last_synced,
        "last_synced_age_s": _age_seconds(last_synced),
        # The scheduler poll's own clock. Legitimately much older than the one
        # above -- see `_sync_line` -- and None until the first `squeue` lands.
        "last_scheduler": last_scheduler,
        "last_scheduler_age_s": _age_seconds(last_scheduler),
        # The daemon's last transport error, if any. The interesting case is
        # `auth_required`: the fix is a command the user can run, so saying
        # "run `relay connect`" is worth more than a generic ssh failure.
        "daemon_state": store.get_daemon_state(),
    }


def _sync_lines(data: dict) -> list[str]:
    """Render `_sync_footer`: the freshness line, plus a reason when stuck."""
    lines = [_sync_line(data)]

    state = data.get("daemon_state") or {}
    kind = state.get("last_error_kind")
    if kind == ssh_module.AUTH_REQUIRED:
        # Deliberately the exact command to type. An ssh failure the user can
        # fix in one line should not be reported as an ssh failure.
        lines.append("Authentication needed. Run `relay connect`.")
    elif kind == ssh_module.UNREACHABLE:
        since = state.get("last_error_ts") or "an unknown time"
        detail = (state.get("last_error_detail") or "no detail recorded").strip()
        # One line: this sits under a table and is a status, not a report.
        lines.append(f"Cluster unreachable since {since}: {detail.splitlines()[0]}")
    return lines


def _open_store() -> Store:
    """Open relay's database at its configured (XDG) location.

    Note what this does *not* need: the config file. The database path comes
    from `XDG_DATA_HOME`, so read commands work before `relay init` has ever
    been run — and, in the tests, point at a temporary directory.
    """
    return Store(cfg.db_path())


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def cmd_init(args) -> Output:
    path = cfg.init(force=args.force)
    return Output(
        data={"config_path": str(path), "created": True},
        renderer=lambda d: [
            f"Wrote a starter config to {d['config_path']}.",
            "Edit it to point at your cluster, then run `relay doctor`.",
        ],
    )


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

# Status word -> the glyph in front of it. Kept next to the renderer because it
# is purely presentation; `doctor.py` owns the meanings.
_DOCTOR_GLYPHS = {
    doctor_module.OK: "ok  ",
    doctor_module.WARN: "warn",
    doctor_module.ERROR: "FAIL",
    doctor_module.SKIP: "skip",
}


def _render_doctor(results: list[dict]) -> list[str]:
    """One aligned line per check, then a one-line tally.

    Takes the *dicts* that `--json` would print, not `CheckResult` objects, so
    that the two renderings cannot drift: if a field is missing from the JSON
    it is missing from the table too, and a test notices.
    """
    lines = []
    width = max((len(r["name"]) for r in results), default=0)
    for result in results:
        glyph = _DOCTOR_GLYPHS.get(result["status"], result["status"])
        lines.append(f"{glyph}  {result['name'].ljust(width)}  {result['detail']}")

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    tally = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    lines.append("")
    lines.append(f"{len(results)} checks: {tally}")
    return lines


def cmd_doctor(args) -> Output:
    results = doctor_module.run_checks()
    return Output(
        data=[r.as_dict() for r in results],
        renderer=_render_doctor,
        # Warnings are not failures: doctor has to stay usable in CI, where
        # there is no cluster and no daemon and those are warnings by design.
        exit_code=doctor_module.worst_status(results),
    )


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


def parse_seeds(text: str) -> list[int]:
    """Parse `--seeds`: "0-4", "0,2,5", or a mix of the two.

    Ranges are inclusive at both ends, because `--seeds 0-4` meaning four runs
    would surprise everyone who has ever written a for loop in a paper.
    """
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            # lstrip above so a negative seed like "-1" is not read as a range.
            lo_text, _, hi_text = part.partition("-")
            try:
                lo, hi = int(lo_text), int(hi_text)
            except ValueError:
                raise BadUsage(
                    f"Could not read {part!r} in --seeds as a range of whole numbers. "
                    f"Write it like 0-4."
                ) from None
            if hi < lo:
                raise BadUsage(
                    f"--seeds range {part!r} counts backwards. Write it as {hi}-{lo}."
                )
            seeds.extend(range(lo, hi + 1))
        else:
            try:
                seeds.append(int(part))
            except ValueError:
                raise BadUsage(
                    f"Could not read {part!r} in --seeds as a whole number. "
                    f"Write seeds like 0-4 or 0,2,5."
                ) from None
    if not seeds:
        raise BadUsage("--seeds was empty. Write it like 0-4 or 0,2,5.")
    # Duplicates removed, order kept: "0-2,1" is a typo, not a request for two
    # identical runs (which would collide on nothing but would waste a GPU).
    return list(dict.fromkeys(seeds))


def _sanitise(text: str) -> str:
    """Reduce a name to characters that are safe in a run ID and a path."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_" for c in text)
    return cleaned.strip("_") or "run"


def make_run_id(base: str, seed: int | None) -> str:
    """relay's own run ID: `<base>[_s<seed>]_<4 hex>`.

    The random suffix is what makes re-submitting the same script twice produce
    two runs instead of one collision. `secrets.token_hex(2)` gives four hex
    characters: 65536 possibilities, which is plenty when the namespace is one
    researcher's experiments and the consequence of a collision is a confusing
    `relay ls`, not data loss (the store's INSERT OR IGNORE would keep the
    first row).
    """
    parts = [_sanitise(base)]
    if seed is not None:
        parts.append(f"s{seed}")
    parts.append(secrets.token_hex(2))
    return "_".join(parts)


def build_command(script: str, python: str, passthrough: list[str], seed: int | None) -> list[str]:
    """Build the argv the sidecar will run.

    A `.py` script is run with an interpreter; anything else is assumed to be
    executable and is run directly, so `relay submit ./train.sh` works.

    The user's `--` arguments go through verbatim and *first*, then relay
    appends `--seed N` when `--seeds` was given. Appending rather than
    prepending means a user who passes their own `--seed` still wins under the
    usual argparse rule of "last one wins"... but relay's copy is the one that
    varies per run, so relay's has to be last.
    """
    command = [python, script] if script.endswith(".py") else [script]
    command.extend(passthrough)
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def _config_for_backend(conf: cfg.Config, override: str | None) -> cfg.Config:
    """Apply a `--backend` override to a loaded config.

    `dataclasses.replace` rather than mutation, because `Config` is frozen: a
    loaded config is a snapshot of a file on disk and nothing should be quietly
    editing it. The copy is local to this one command.
    """
    if override is None or override == conf.backend:
        return conf
    return dataclasses.replace(conf, backend=override)


def _run_dir_for(conf: cfg.Config, run_id: str) -> str:
    """Where this run's directory lives, per backend.

    `posixpath.join` for Slurm, not `os.path.join`: the path is being built on
    the laptop but interpreted on the cluster, and a laptop running Windows
    would otherwise produce backslashes that GPFS has never heard of.
    """
    if conf.backend == "slurm":
        return posixpath.join(conf.remote_root or "", run_id)
    return os.path.join(conf.local_root, run_id)


def _submit_lists(conf: cfg.Config, args) -> tuple[list[str], list[str]]:
    """Decide this submission's `setup` lines and `sbatch_extra` directives.

    Both can come from the config or from the command line, and the rule is
    **replace, never merge**: if `--setup` appears at all, its lines are the
    whole list and the config's are ignored; same for `--sbatch-extra`.

    Merging is the tempting alternative and it is worse. A user who writes
    `--setup "conda activate other-env"` wants *that* environment, not the
    config's `conda activate rl` followed by theirs — and with merging there is
    no way to say "none of the config's lines" at all. Replace has one rule the
    user can hold in their head, and `--setup ""` is not needed to escape it.

    `args.setup` and `args.sbatch_extra` default to None (never given) rather
    than [] so that "flag absent" stays distinguishable from "flag given an
    empty value", which is exactly the distinction the rule above turns on.
    """
    setup = list(args.setup) if args.setup else list(conf.setup)
    extra = list(args.sbatch_extra) if args.sbatch_extra else list(conf.slurm.sbatch_extra)
    return setup, extra


def _resume_pair(conf: cfg.Config, args) -> tuple[str | None, str | None]:
    """Decide this submission's checkpoint glob and its resume argument.

    The same **replace, never merge** rule as `_submit_lists`, except that the
    thing being replaced is a *pair*: `--checkpoint-glob` and `--resume-arg`
    only mean anything together, so giving either one on the command line
    replaces both halves of the config's `resume:` section. Merging one flag
    with one config key would let a user end up resuming from a glob they did
    not write with an argument they did, which is the sort of surprise that
    costs somebody a night of GPU time.

    The two validations here are the ones only the command line can make. The
    same both-or-neither and `{path}` rules apply to the config file, but
    `config.py` checks those when it loads, so a bad config is reported once at
    load time rather than being re-checked here.
    """
    if args.checkpoint_glob is None and args.resume_arg is None:
        return conf.resume.checkpoint_glob, conf.resume.arg

    if args.checkpoint_glob is None or args.resume_arg is None:
        raise BadUsage(
            "both --checkpoint-glob and --resume-arg are needed. A glob on its "
            "own finds a checkpoint with no way to tell your script about it, "
            "and an argument on its own has no file to name."
        )

    if "{path}" not in args.resume_arg:
        raise BadUsage(
            f"--resume-arg {args.resume_arg!r} does not contain {{path}}, so "
            f"relay has nowhere to put the checkpoint it found. Write it like "
            f"--resume-arg='--resume {{path}}'."
        )

    return args.checkpoint_glob, args.resume_arg


def cmd_submit(args) -> Output:
    conf = _config_for_backend(cfg.load(), args.backend)

    setup_lines, sbatch_extra = _submit_lists(conf, args)
    resume_glob, resume_arg = _resume_pair(conf, args)

    # Validate before anything is rendered, before a backend is built, and so
    # before a single byte goes over SSH. A rejected directive should cost the
    # user an error message, not a Duo prompt and a half-written sbatch script
    # on the cluster. Once, not once per seed: every run in a `--seeds` sweep
    # shares this list, so re-checking it would only repeat the same answer.
    try:
        validate_sbatch_extra(sbatch_extra)
    except SbatchExtraError as exc:
        # Re-raised as BadUsage because that is what it is -- the user typed
        # something relay will not accept -- and BadUsage already maps to exit
        # code 2. The original message is kept verbatim: it names the entry and
        # says either what it would break or which config field to use instead.
        raise BadUsage(str(exc)) from None

    backend = daemon_module.build_backend(conf)

    seeds: list[int | None]
    seeds = list(parse_seeds(args.seeds)) if args.seeds else [None]

    base = args.name or Path(args.script).stem
    # The interpreter differs per backend for the same reason LocalBackend uses
    # sys.executable for the sidecar: locally "python3" resolves in the user's
    # own environment, while on the cluster only `remote_python` is meaningful.
    python = conf.remote_python if conf.backend == "slurm" else "python3"

    submitted = []
    with _open_store() as store:
        for seed in seeds:
            run_id = make_run_id(base, seed)
            run_dir = _run_dir_for(conf, run_id)
            spec = JobSpec(
                run_id=run_id,
                command=build_command(args.script, python, args.passthrough, seed),
                run_dir=run_dir,
                name=args.name or base,
                # Copy the config's grace period into the spec. The Slurm
                # backend honours the spec over its own default, and the number
                # has to end up in two places that must agree: the sbatch
                # `--signal=B:TERM@<grace>` and the sidecar's `--grace`.
                grace_seconds=conf.slurm.grace_seconds,
                # The preemption window, which is a different number from the
                # time-limit grace above: the cluster picks this one (partition
                # GraceTime, or KillWait) and the sidecar budgets against it.
                preempt_grace=conf.preempt_grace,
                requeue_on_term=conf.requeue_on_term,
                # `--time` first, then the config's `slurm.time`. Either may be
                # None: the backend is the one place that knows whether a
                # missing time limit is fatal. Pre-rejecting here would mean
                # two rules for the same question, and the CLI's copy would be
                # the one that goes stale. (`--time` cannot arrive through
                # sbatch_extra instead: validation refuses it there, so there
                # is exactly one way to set it.)
                time_limit=args.time or conf.slurm.time,
                heartbeat_seconds=DEFAULT_HEARTBEAT_SECONDS,
                account=conf.slurm.account,
                partition=conf.slurm.partition,
                gres=conf.slurm.gres,
                # A fresh list per spec, not the same object five times: the
                # specs are stored with `dataclasses.asdict` and a shared list
                # would make five runs' stored specs alias one another.
                sbatch_extra=list(sbatch_extra),
                setup=list(setup_lines),
                # Both or neither, guaranteed by `_resume_pair`. The backend
                # writes them into sidecar.json, and the sidecar is the piece
                # that actually globs for a file -- on the compute node, where
                # the checkpoints are, which is the only place that can.
                resume_glob=resume_glob,
                resume_arg=resume_arg,
            )

            job_id = backend.submit(spec)

            # Status "queued" even on the local backend, where the job is
            # already running: status is scheduler-owned, and the daemon will
            # write "running" on its next cycle from what the backend reports.
            # Guessing here would put the CLI in the business of deciding job
            # status, which is exactly the split CLAUDE.md forbids.
            store.create_run(
                run_id,
                backend=conf.backend,
                name=args.name or base,
                job_id=job_id,
                run_dir=run_dir,
                status="queued",
                spec=dataclasses.asdict(spec),
            )
            submitted.append(
                {
                    "run_id": run_id,
                    "job_id": job_id,
                    "name": args.name or base,
                    "backend": conf.backend,
                    "run_dir": run_dir,
                    "seed": seed,
                    "command": spec.command,
                }
            )

        # Once per invocation, not once per seed: the poke is a single "look at
        # the scheduler now" note, and writing it five times for a five-seed
        # sweep would say the same thing five times.
        #
        # The daemon's scheduler poll stretches towards five minutes while no
        # job changes state, and a submit is precisely the moment that backoff
        # is at its most relaxed -- nothing had changed, which is why it backed
        # off. It is also the moment the user starts watching `relay ls` for
        # their job to leave the queue. So we tell the daemon to look now.
        store.poke_scheduler()

    def render(data: dict) -> list[str]:
        lines = [f"{r['run_id']}  job {r['job_id']}" for r in data["runs"]]
        lines.append("")
        lines.append(
            f"Submitted {len(data['runs'])} run(s). "
            f"Watch them with `relay ls` (the daemon does the fetching: "
            f"`relay daemon`)."
        )
        return lines

    return Output(data={"runs": submitted}, renderer=render)


# --------------------------------------------------------------------------
# ls
# --------------------------------------------------------------------------


def _ls_data(store: Store, *, show_all: bool) -> dict:
    """Everything `relay ls` shows, as plain data.

    Separate from the command so that a test can replace it and prove the
    table and the JSON are two renderings of one answer.
    """
    runs = store.list_runs(active_only=not show_all)
    for run in runs:
        run["latest_metrics"] = store.latest_metrics(run["run_id"])
    return {"runs": runs, **_sync_footer(store)}


def _render_ls(data: dict) -> list[str]:
    rows = []
    for run in data["runs"]:
        status = run.get("status", "?")
        if run.get("stale"):
            # Stale is a flag, not a status, so it is shown next to the status
            # rather than instead of it.
            status += " (stale)"
        metrics = run.get("latest_metrics") or {}
        shown = list(metrics.items())[:LS_METRIC_COLUMNS]
        rows.append(
            [
                str(run.get("run_id", "")),
                status,
                str(run.get("backend") or ""),
                str(run.get("job_id") or "-"),
                "-" if run.get("last_step") is None else str(run["last_step"]),
                " ".join(f"{name}={_number(value)}" for name, value in shown),
            ]
        )

    if rows:
        lines = _table(["RUN", "STATUS", "BACKEND", "JOB", "STEP", "METRICS"], rows)
    else:
        lines = ["No runs. Submit one with `relay submit train.py`."]
    lines.append("")
    lines.extend(_sync_lines(data))
    return lines


def cmd_ls(args) -> Output:
    with _open_store() as store:
        data = _ls_data(store, show_all=args.all)
    return Output(data=data, renderer=_render_ls)


# --------------------------------------------------------------------------
# show
# --------------------------------------------------------------------------

# How many of a run's most recent events `relay show` summarises.
SHOW_RECENT_EVENTS = 10


def _require_run(store: Store, run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise NotFound(
            f"No run {run_id!r} in relay's database. "
            f"Run `relay ls --all` to see the runs relay knows about."
        )
    return run


def _show_data(store: Store, run_id: str) -> dict:
    run = _require_run(store, run_id)
    highest = store.max_seq(run_id)
    recent = store.list_events(run_id, since_seq=max(0, highest - SHOW_RECENT_EVENTS))
    return {
        "run": run,
        "metric_names": store.metric_names(run_id),
        "latest_metrics": store.latest_metrics(run_id),
        "counts": {
            "events": store.count_events(run_id),
            "metrics": store.count_metrics(run_id),
            "max_seq": highest,
        },
        "recent_events": [
            {"seq": e["seq"], "ts": e["ts"], "type": e["type"], "payload": e["payload"]}
            for e in recent
        ],
        "sources": store.list_sources(run_id),
        # One row per `run_started` event: when each attempt began and, if it
        # resumed, which checkpoint file it picked up from. The `attempts`
        # count above says a run restarted; this says whether the restart
        # continued the work or quietly began again from step zero.
        "attempts_history": store.attempt_history(run_id),
        # One row per scheduler attempt. A run preempted twice has three, and
        # each one burned real hours — which is exactly the thing a single
        # "elapsed" number would hide.
        "usage": store.list_usage(run_id),
    }


def _render_show(data: dict) -> list[str]:
    run = data["run"]
    lines = [
        f"run       {run['run_id']}",
        f"name      {run.get('name') or '-'}",
        f"status    {run.get('status')}{' (stale)' if run.get('stale') else ''}",
        f"backend   {run.get('backend')}",
        f"job       {run.get('job_id') or '-'}",
        f"dir       {run.get('run_dir') or '-'}",
        f"created   {run.get('created_ts')}",
        f"updated   {run.get('updated_ts')}",
        f"heartbeat {run.get('last_heartbeat_ts') or '-'}",
        f"step      {'-' if run.get('last_step') is None else run['last_step']}",
        # Counted from `run_started` events, so it says how many times the job
        # actually started, not how many times Slurm meant to start it.
        f"attempts  {run.get('attempts') if run.get('attempts') is not None else '-'}",
    ]

    # Indented under the `attempts` line it explains, and only when there is
    # something to say: a run that started once and never came back is the
    # common case, and a one-row table above it would be noise.
    history = data.get("attempts_history") or []
    if history:
        rows = [
            [
                str(row.get("attempt") if row.get("attempt") is not None else "-"),
                str(row.get("ts") or "-"),
                # "-" rather than "fresh start" so that every empty cell in
                # every relay table means the same thing.
                str(row.get("resumed_from") or "-"),
            ]
            for row in history
        ]
        lines.extend(
            "  " + line for line in _table(["ATTEMPT", "STARTED", "RESUMED FROM"], rows)
        )

    lines.append(
        f"events    {data['counts']['events']} ({data['counts']['metrics']} metric points)"
    )

    latest = data.get("latest_metrics") or {}
    if latest:
        lines.append("")
        lines.append("latest metrics")
        for name, value in latest.items():
            lines.append(f"  {name} = {_number(value)}")

    usage = data.get("usage") or []
    if usage:
        lines.append("")
        lines.append("usage by attempt")
        rows = [
            [
                str(row.get("job_id") or "-"),
                str(row.get("state") or "-"),
                str(row.get("partition") or "-"),
                str(row.get("start") or "-"),
                str(row.get("end") or "(running)"),
                _hours(row.get("elapsed_s")),
                str(row.get("gpus") if row.get("gpus") is not None else "-"),
            ]
            for row in usage
        ]
        lines.extend(
            "  " + line
            for line in _table(
                ["JOB", "STATE", "PARTITION", "START", "END", "HOURS", "GPUS"], rows
            )
        )

    recent = data.get("recent_events") or []
    if recent:
        lines.append("")
        lines.append("recent events")
        for event in recent:
            lines.append(f"  {event['seq']:>6}  {event['ts']}  {event['type']}")
    return lines


def cmd_show(args) -> Output:
    with _open_store() as store:
        data = _show_data(store, args.run_id)
    return Output(data=data, renderer=_render_show)


# --------------------------------------------------------------------------
# logs and attach
# --------------------------------------------------------------------------


def _render_logs(data: dict) -> list[str]:
    lines = []
    for event in data["logs"]:
        payload = event.get("payload") or {}
        text = payload.get("line", "")
        lines.append(f"{_stream_prefix(payload)}{text}")
    return lines


def _stream_prefix(payload: dict) -> str:
    """The tag in front of a log line, by which stream it came from.

    stdout gets no tag: it is the common case, and tagging it would double
    the width of every line to carry no information. stderr and relay's own
    lines (the sidecar saying "resuming from ...") are tagged so the user can
    tell the script's output from what relay did to the script.
    """
    stream = payload.get("stream")
    if stream == "stderr":
        return "stderr| "
    if stream == "relay":
        return "relay | "
    return ""


def _render_events(data: dict) -> list[str]:
    """`relay attach`: log lines as themselves, metric events as a summary."""
    lines = []
    for event in data["events"]:
        payload = event.get("payload") or {}
        if event["type"] == "log":
            lines.append(f"{_stream_prefix(payload)}{payload.get('line', '')}")
        elif event["type"] == "metric":
            values = payload.get("values") or {}
            summary = " ".join(f"{k}={_number(v)}" for k, v in values.items())
            lines.append(f"step {payload.get('step')}  {summary}")
        else:
            lines.append(f"[{event['type']}]")
    return lines


def _follow(
    store: Store,
    run_id: str,
    since_seq: int,
    *,
    fetch: Callable[[int], list[dict]],
    key: str,
    renderer: Callable[[Any], Iterable[str]],
    as_json: bool,
) -> None:
    """Poll the *database* until the run is terminal, emitting what is new.

    Worth being explicit about: following reads SQLite and nothing else. It
    never touches the cluster. The daemon is the only process that fetches
    bytes, so `relay logs --follow` with no daemon running will sit there
    politely showing nothing — which is correct, and is why `relay ls` prints
    when it last synced.

    `since_seq` is exclusive, so each poll asks only for what it has not seen.
    """
    while True:
        events = fetch(since_seq)
        if events:
            since_seq = events[-1]["seq"]
            _emit(Output(data={"run_id": run_id, key: events}, renderer=renderer), as_json)

        run = store.get_run(run_id)
        if run is not None and run.get("terminal"):
            # One last look before leaving: the daemon may have committed the
            # final lines of the log in the moment between our fetch and this
            # check, and dropping them would make `--follow` less complete than
            # plain `relay logs`.
            tail = fetch(since_seq)
            if tail:
                _emit(Output(data={"run_id": run_id, key: tail}, renderer=renderer), as_json)
            return

        time.sleep(FOLLOW_INTERVAL_S)


def cmd_logs(args) -> Output | None:
    with _open_store() as store:
        _require_run(store, args.run_id)
        if args.follow:
            # Follow emits each batch through `_emit` itself and returns None,
            # so the dispatcher knows the output is already on the screen.
            _follow(
                store,
                args.run_id,
                args.since_seq,
                fetch=lambda seq: store.list_logs(args.run_id, since_seq=seq),
                key="logs",
                renderer=_render_logs,
                as_json=args.json,
            )
            return None
        logs = store.list_logs(args.run_id, since_seq=args.since_seq)
    return Output(data={"run_id": args.run_id, "logs": logs}, renderer=_render_logs)


def cmd_attach(args) -> None:
    """`logs --follow`, plus metrics as they arrive.

    Deliberately thin: it is the same poll loop with a different fetch and a
    different renderer. Anything more (a live chart in the terminal) is what
    `relay dash` is for.
    """
    with _open_store() as store:
        _require_run(store, args.run_id)
        # Default is to replay from the start, so attaching to a run shows you
        # what you missed; --tail is for when you only want what happens next.
        since = store.max_seq(args.run_id) if args.tail else 0

        def fetch(seq: int) -> list[dict]:
            return [
                event
                for event in store.list_events(args.run_id, since_seq=seq)
                if event["type"] in ("log", "metric")
            ]

        _follow(
            store,
            args.run_id,
            since,
            fetch=fetch,
            key="events",
            renderer=_render_events,
            as_json=args.json,
        )
    return None


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------


def _cancel_run(store: Store, run_id: str) -> dict:
    """Ask the backend to stop a run, then record that we did.

    Shared by `relay cancel` and the dashboard's cancel button, so the two
    cannot behave differently.

    Writing "cancelled" into the database immediately is a UX decision, not a
    claim of authority. Status is scheduler-owned: the daemon would converge to
    this on its next cycle anyway, once `squeue`/`sacct` (or a dead PID) agrees.
    Waiting for that would leave a user staring at "running" for up to thirty
    seconds after they pressed the button, and wondering whether it worked.
    """
    run = _require_run(store, run_id)
    job_id = run.get("job_id")
    if not job_id:
        raise RuntimeError(
            f"Run {run_id} has no job ID recorded, so there is nothing to cancel. "
            f"It may never have been submitted successfully."
        )

    # The run's own backend, not necessarily the one in the config today: a
    # user who switched `backend: local` to `backend: slurm` must still be able
    # to cancel yesterday's local run.
    conf = _config_for_backend(cfg.load(), run.get("backend"))
    backend = daemon_module.build_backend(conf)
    # The run directory is handed over so the backend can drop a
    # CANCEL_REQUESTED file in it *before* the scancel. The sidecar cannot tell
    # a cancel's SIGTERM from a preemption's SIGTERM — same signal — so that
    # file is the only way it learns not to requeue a run the user just
    # stopped. The backend guarantees the ordering; the CLI only supplies the
    # path.
    backend.cancel(str(job_id), run_dir=run.get("run_dir"))
    store.update_run_status(run_id, "cancelled")
    # Same reason as `submit`: the daemon's scheduler poll backs off to five
    # minutes while nothing is changing, and a cancel is exactly when the user
    # is about to watch for the result. The status written above is relay's
    # optimistic one; the scheduler owns the real one, and this asks the daemon
    # to go and get it rather than waiting out the backoff.
    store.poke_scheduler()
    return {
        "run_id": run_id,
        "job_id": str(job_id),
        "run_dir": run.get("run_dir"),
        "status": "cancelled",
    }


def cmd_cancel(args) -> Output:
    with _open_store() as store:
        data = _cancel_run(store, args.run_id)
    return Output(
        data=data,
        renderer=lambda d: [
            f"Asked the backend to stop {d['run_id']} (job {d['job_id']}).",
            "The job gets its grace period to checkpoint before it dies.",
        ],
    )


# --------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------
#
# Everything here reads the `usage` table, which the daemon fills from `sacct`
# (or, locally, from process wall time). Nothing in this section talks to the
# cluster, so `relay usage` works on a plane.

# What `--by` may say, and the column header each grouping gets.
USAGE_GROUPINGS = {"run": "RUN", "group": "ACCOUNT", "partition": "PARTITION"}


def _cost_rate() -> float | None:
    """The configured price of a GPU-hour, or None if the user set none.

    Two rules from CLAUDE.md meet here. "Never invent a default price": with
    no rate configured, relay shows hours and no dollars at all. And "read
    commands need no config": a machine that never ran `relay init` still gets
    its usage report, just without a cost column — so a missing config is None
    rather than exit code 3.

    A config that exists but does not parse is still an error, because a
    silently ignored typo in the rate is the same as inventing a price.
    """
    try:
        conf = cfg.load()
    except cfg.ConfigNotFound:
        return None
    return conf.cost.gpu_hour_rate


def _usage_data(store: Store, *, by: str, since: str | None, group: str | None) -> dict:
    """Everything `relay usage` shows, as plain data.

    Separate from the command for the same reason `_ls_data` is: a test can
    replace it and prove that the table and the JSON are one answer rendered
    twice.
    """
    try:
        rows = store.usage_by(by, since=since, group=group)
        totals = store.usage_totals(since=since, group=group)
        wasted = store.wasted_hours(since=since, group=group)
    except ValueError as exc:
        # The store raises this for a date it cannot read. That is a bad
        # command line, not a runtime failure, so it gets exit code 2.
        raise BadUsage(f"{exc}. Write --since as a date like 2026-09-01.") from None

    data: dict = {"by": by, "rows": rows, "totals": totals, "wasted": wasted}

    rate = _cost_rate()
    if rate is not None:
        # Costs are derived here, not in the store: the store owns hours, which
        # are a fact about the cluster, while a price is a fact about the
        # user's grant. Multiplying at the edge keeps the database free of one
        # user's billing assumptions.
        data["cost"] = {
            "gpu_hour_rate": rate,
            "total": float(totals.get("gpu_hours") or 0.0) * rate,
            "wasted": {
                "failed": float(wasted["failed"]["gpu_hours"] or 0.0) * rate,
                "lost_to_preemption": float(
                    wasted["lost_to_preemption"]["gpu_hours"] or 0.0
                )
                * rate,
            },
        }

    data.update(_sync_footer(store))
    return data


def _render_usage(data: dict) -> list[str]:
    cost = data.get("cost")
    by = data.get("by", "run")
    header = USAGE_GROUPINGS.get(by, by.upper())

    rows = [
        [
            str(row.get("key") or "(none)"),
            _hours_text(row.get("gpu_hours")),
            _hours_text(row.get("cpu_hours")),
            str(row.get("attempts", 0)),
            str(row.get("runs", 0)),
        ]
        for row in data.get("rows") or []
    ]
    if rows:
        lines = _table([header, "GPU-HOURS", "CPU-HOURS", "ATTEMPTS", "RUNS"], rows)
    else:
        lines = [
            "No usage recorded yet. The daemon fetches it from the scheduler "
            "every few minutes."
        ]

    # -- totals, and the partition split ----------------------------------
    totals = data.get("totals") or {}
    lines.append("")
    total = (
        f"total  {_hours_text(totals.get('gpu_hours'))} GPU-hours, "
        f"{_hours_text(totals.get('cpu_hours'))} CPU-hours"
        f"  ({totals.get('attempts', 0)} attempts in {totals.get('runs', 0)} runs)"
    )
    if cost:
        total += f"  {_money(cost['total'])}"
    lines.append(total)

    partitions = totals.get("by_partition") or []
    if partitions:
        # The point of the whole feature on Klone: how much ran for free on
        # preemptible ckpt* nodes versus how much charged the group.
        part_rows = [
            [
                str(part.get("partition") or "(none)"),
                _hours_text(part.get("gpu_hours")),
                _hours_text(part.get("cpu_hours")),
            ]
            for part in partitions
        ]
        lines.extend(
            "  " + line
            for line in _table(["PARTITION", "GPU-HOURS", "CPU-HOURS"], part_rows)
        )

    # -- wasted hours ------------------------------------------------------
    wasted = data.get("wasted") or {}
    failed = wasted.get("failed") or {}
    lost = wasted.get("lost_to_preemption") or {}

    lines.append("")
    lines.append("Wasted")

    failed_line = f"  failed              {_hours_text(failed.get('gpu_hours'))} GPU-hours"
    if cost:
        failed_line += f"  {_money(cost['wasted']['failed'])}"
    lines.append(failed_line)
    failed_runs = failed.get("runs") or []
    if failed_runs:
        lines.append(f"    runs: {', '.join(failed_runs)}")

    lost_line = f"  lost to preemption  {_hours_text(lost.get('gpu_hours'))} GPU-hours"
    if cost:
        lost_line += f"  {_money(cost['wasted']['lost_to_preemption'])}"
    lines.append(lost_line)
    for attempt in lost.get("attempts") or []:
        # Wall-clock hours and the GPU count, kept separate: multiplying them
        # here would print a second GPU-hours number next to the total above
        # and invite the reader to add the two together.
        line = (
            f"    {attempt.get('run_id')}  job {attempt.get('job_id')}  "
            f"{_hours(attempt.get('lost_s'))} h on {attempt.get('gpus', 0)} GPU(s), "
            f"ended {attempt.get('end')}"
        )
        if attempt.get("no_checkpoints"):
            # Worth saying in full every time: the usual cause is not a script
            # that never checkpoints, but one that checkpoints without telling
            # relay, and the user cannot guess that from a number.
            line += " (no checkpoints reported — your script is not printing checkpoint lines)"
        lines.append(line)

    lines.append("")
    lines.extend(_sync_lines(data))
    return lines


def cmd_usage(args) -> Output:
    with _open_store() as store:
        data = _usage_data(store, by=args.by, since=args.since, group=args.group)
    return Output(data=data, renderer=_render_usage)


# --------------------------------------------------------------------------
# connect
# --------------------------------------------------------------------------


def _last_nonblank_line(text: str) -> str:
    """The last non-blank line of `text`, stripped, or "" if there is none.

    ssh prints its banners and warnings first and the sentence that actually
    explains the failure last, so the *last* line is the one worth quoting.

    A four-line copy of `slurm._last_line` rather than an import: `cli.py`
    deliberately imports the Slurm backend lazily, inside the functions that
    need it, so that `relay ls` on a laptop with no cluster does not pull in
    the SSH backend.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _run_interactive(argv: list[str]) -> tuple[int, str]:
    """Run ssh with relay's terminal attached. Returns (exit code, its stderr).

    Two requirements pull in opposite directions here, so this *tees* stderr
    rather than choosing one of them:

      * The user must see ssh in real time. Duo's prompt text and every ssh
        warning go to stderr, and ssh reads the answer from /dev/tty. Capturing
        stderr silently (`capture_output=True`) would turn `relay connect` into
        a command that hangs for no visible reason while a prompt sits unread
        in a buffer.

      * relay needs the text for its own error message. Plain inheritance --
        the old one-liner -- printed everything and kept nothing, so when ssh
        failed all relay could say was "check that plain `ssh klone` works",
        even when ssh had just explained the real reason on the line above.

    So stdin and stdout stay inherited, stderr is piped, and every line read
    off that pipe is written straight through to `sys.stderr` and flushed
    *before* the next read. The user sees each line as ssh prints it, and relay
    still has the whole thing when the process exits.

    Still a module-level function so a test can replace it without going
    anywhere near a real ssh.
    """
    # text=True gives us str lines; stdin and stdout are left alone so ssh keeps
    # the terminal it needs.
    proc = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)
    captured: list[str] = []
    # Iterating the pipe yields one line at a time as ssh writes it, which is
    # what keeps the prompt live rather than arriving all at once at the end.
    assert proc.stderr is not None  # stderr=PIPE guarantees this
    for line in proc.stderr:
        captured.append(line)
        sys.stderr.write(line)
        sys.stderr.flush()
    proc.stderr.close()
    return proc.wait(), "".join(captured)


def _master_alive(alias: str) -> bool:
    """Is a control master listening for this alias? (`ssh -O check`.)

    Wrapped for the same reason as `_run_interactive`: one seam, easy to fake.
    """
    return ssh_module.master_alive(alias)


def cmd_connect(args) -> Output:
    """Open the SSH master the daemon will tunnel every later call through.

    This is the one relay command that is allowed to be interactive. Every
    other ssh relay runs carries `BatchMode=yes` so it fails fast instead of
    waiting forever for a Duo prompt nobody is watching; this one drops
    BatchMode on purpose, so the user can tap their phone once and leave the
    master running for the rest of the working day.
    """
    conf = cfg.load()  # No config is ConfigNotFound, which is exit code 3.

    if conf.backend != "slurm":
        return Output(
            data={"backend": conf.backend, "alias": None, "connected": False},
            renderer=lambda d: [
                "`relay connect` is only for the slurm backend. This config uses "
                "the local backend, which runs jobs on this machine and needs no "
                "SSH connection.",
            ],
        )

    alias = conf.ssh_alias
    if not alias:
        raise cfg.ConfigError(
            f"No ssh_alias is set in {cfg.config_path()}, so relay does not know "
            f"which host to connect to. Add the name of a Host block from your "
            f"~/.ssh/config."
        )

    code, err = _run_interactive(ssh_module.connect_argv(alias))
    if code != 0:
        # ssh has already told the user what went wrong -- it went past on the
        # terminal a moment ago -- so relay repeats that line rather than
        # guessing. The first real run on Klone failed with "unix_listener:
        # cannot bind to path ...: No such file or directory" *after* Duo had
        # succeeded, while relay advised checking that plain `ssh klone` works.
        # That advice sent the user to look at the one thing that was fine.
        detail = _last_nonblank_line(err)
        if detail:
            raise ConnectFailed(
                f"ssh exited {code} while opening a connection to {alias}: "
                f"{detail}. Run `relay connect` again once that is fixed."
            )
        raise ConnectFailed(
            f"ssh exited {code} while opening a connection to {alias}, so relay "
            f"has no control master. Check that plain `ssh {alias}` works, then "
            f"run `relay connect` again."
        )

    # ssh -f backgrounds itself the moment authentication succeeds, so a zero
    # exit code only means "it forked". Whether a master is actually listening
    # is a separate question, and `ssh -O check` is the one that answers it —
    # cheaply, over the local socket, with no chance of a second Duo prompt.
    if not _master_alive(alias):
        raise ConnectFailed(
            f"ssh reported success but no control master is listening for "
            f"{alias}. Check that {ssh_module.relay_dir()} is writable, then run "
            f"`relay connect` again."
        )

    return Output(
        data={
            "backend": conf.backend,
            "alias": alias,
            "connected": True,
            "control_persist": ssh_module.CONTROL_PERSIST,
        },
        renderer=lambda d: [
            f"Connected to {d['alias']}. The connection persists for "
            f"{d['control_persist']} after the last use."
        ],
    )


# --------------------------------------------------------------------------
# dash
# --------------------------------------------------------------------------


def cmd_dash(args) -> None:
    # Imported here, not at module scope: fastapi and uvicorn are heavy, and
    # `relay ls` should not pay for a dashboard it is not starting.
    from relay.web import app as web_app

    db = str(cfg.db_path())

    def cancel_fn(run_id: str) -> None:
        """What the dashboard's POST /cancel calls. Mirrors `relay cancel`.

        A fresh `Store` per call because a `Store` owns one sqlite3 connection
        and is not thread-safe, and this runs on uvicorn's worker thread rather
        than the one that built the app.

        The config is loaded *here* rather than when `dash` starts, so a user
        with no config can still read their dashboard; only pressing Cancel
        needs a backend.
        """
        with Store(db) as store:
            _cancel_run(store, run_id)

    print(f"relay dashboard on http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    # The dashboard never reads the config itself (it is SQLite-only), so the
    # one number it needs from it -- the optional GPU-hour price -- is handed
    # over here. None means "show hours, never dollars", same as `relay usage`.
    web_app.serve(
        store_path=db,
        cancel_fn=cancel_fn,
        port=args.port,
        gpu_hour_rate=_cost_rate(),
    )
    return None


# --------------------------------------------------------------------------
# daemon
# --------------------------------------------------------------------------


def cmd_daemon(args) -> Output:
    """Run the daemon in the foreground until Ctrl-C.

    Foreground on purpose. The sophisticated alternative is real daemonisation
    — double fork, setsid, close the inherited descriptors, reopen them on
    /dev/null, write a pidfile — which buys you a process that survives its
    terminal and loses you every error message it would have printed. The
    simple version is `relay daemon` in a tmux pane or a systemd user unit,
    where the supervisor that already exists does the supervising.
    """
    # Only the flags the user actually typed; argparse leaves the rest None and
    # `daemon_intervals` drops those, so "not given" stays distinguishable from
    # "given the default value" and the config keeps its say. The flag names and
    # the keyword names differ on purpose: the flags are named for what the user
    # is tuning (a tail, a scheduler poll) while the keywords are `Daemon`'s own
    # parameter names, and renaming either to match the other would mean
    # renaming it in the wrong place.
    overrides = {
        "interval_active": args.tail_interval_active,
        "interval_queued": args.tail_interval_queued,
        "interval_idle": args.tail_interval_idle,
        "scheduler_interval_min": args.scheduler_interval_min,
        "scheduler_interval_max": args.scheduler_interval_max,
        "usage_interval_seconds": args.usage_interval,
    }
    _configure_daemon_logging()
    code = daemon_module.main(intervals=overrides)
    return Output(data={"exit_code": code}, renderer=lambda d: [], exit_code=code)


def _configure_daemon_logging(stream=None) -> None:
    """Let the daemon's INFO messages reach the terminal it is running in.

    Every other relay command is quiet unless something is wrong, and that is
    right for a command that prints a table and exits. The daemon is different:
    it is a foreground process that runs for hours, and a foreground process
    that prints nothing for a minute is indistinguishable from one that has
    hung. What it has to say at INFO is exactly what a person leaving it in a
    terminal tab wants to see -- that it started and with which intervals,
    each time a run changes state, when a stale run comes back, when it is
    told to stop -- and nothing per cycle, so a healthy idle daemon says one
    line at startup and then only speaks when something happens.

    Configured on the `relay` logger rather than the root logger, so the
    setting cannot leak into whoever imports this module (a test, the
    dashboard) and so a handler somebody else installed on the root logger is
    left alone. Idempotent: a second call adds no second handler, so nothing
    is ever printed twice.
    """
    logger = logging.getLogger("relay")
    logger.setLevel(logging.INFO)
    if any(getattr(h, "_relay_daemon", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    handler._relay_daemon = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


def _json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the result as JSON instead of a table",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relay",
        description="Local-first experiment orchestrator for Slurm clusters.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -- init -------------------------------------------------------------
    p_init = subparsers.add_parser("init", help="write a starter config file")
    p_init.add_argument(
        "--force", action="store_true", help="overwrite an existing config file"
    )
    p_init.set_defaults(handler=cmd_init)

    # -- doctor -----------------------------------------------------------
    p_doctor = subparsers.add_parser(
        "doctor", help="check the config, the cluster, the database and the daemon"
    )
    _json_flag(p_doctor)
    p_doctor.set_defaults(handler=cmd_doctor)

    # -- submit -----------------------------------------------------------
    p_submit = subparsers.add_parser(
        "submit",
        help="submit a training script as one or more runs",
        epilog=(
            "Everything after `--` is passed to your script verbatim:\n"
            "  relay submit train.py --seeds 0-4 -- --lr 3e-4\n"
            "When --seeds is given, relay also appends `--seed N` to each run's\n"
            "command, so your script receives its seed as an ordinary argument\n"
            "and needs no relay-specific code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_submit.add_argument("script", help="the training script (or executable) to run")
    p_submit.add_argument("--name", help="label for this experiment; defaults to the script name")
    p_submit.add_argument(
        "--seeds",
        help="seeds to run, as a range or a list: 0-4, or 0,2,5. One run per seed.",
    )
    p_submit.add_argument(
        "--backend",
        choices=cfg.BACKENDS,
        help="override the backend from the config for this submission",
    )
    p_submit.add_argument(
        "--time",
        help=(
            "wall-clock limit in Slurm's format (12:00:00, 1-06:00:00, 90). "
            "Overrides slurm.time from the config. Slurm requires one: a job "
            "submitted without it is silently capped at the partition's "
            "default, often one hour."
        ),
    )
    p_submit.add_argument(
        "--setup",
        action="append",
        metavar="LINE",
        help=(
            "a shell line to run before the training command, such as "
            "'module load cuda'. Repeat the flag for more lines; they run in "
            "the order given. Giving it at least once REPLACES the config's "
            "`setup:` list entirely -- the two are never merged."
        ),
    )
    p_submit.add_argument(
        "--sbatch-extra",
        action="append",
        dest="sbatch_extra",
        metavar="ENTRY",
        help=(
            "an extra #SBATCH directive, written exactly as it would appear "
            "after `#SBATCH `, e.g. --sbatch-extra=--mem=64G. Use the `=` form "
            "shown here: a value that starts with a dash would otherwise look "
            "like another relay flag. Repeat for more directives. Giving it at "
            "least once REPLACES the config's `slurm.sbatch_extra` list "
            "entirely -- the two are never merged. Directives for flags relay "
            "sets itself (--output, --signal, --time, --gres and so on) are "
            "refused with exit code 2."
        ),
    )
    p_submit.add_argument(
        "--checkpoint-glob",
        dest="checkpoint_glob",
        metavar="GLOB",
        help=(
            "where your script writes its checkpoints, as a glob relative to "
            "the job's working directory, e.g. --checkpoint-glob='ckpt/*.pt' "
            "(`**` is allowed). On every attempt after the first, relay picks "
            "the newest matching file written since the run started and passes "
            "it to your script with --resume-arg. Needs --resume-arg too, and "
            "giving either one REPLACES the config's `resume:` section "
            "entirely -- the two are never merged. A script that writes no "
            "checkpoints of its own can `import relay_ckpt` instead and let "
            "relay do the saving and restoring."
        ),
    )
    p_submit.add_argument(
        "--resume-arg",
        dest="resume_arg",
        metavar="ARG",
        help=(
            "how to tell your script which checkpoint to load, with `{path}` "
            "standing in for the file: --resume-arg='--resume {path}'. Relay "
            "appends it to the training command, shell-quoted, only on "
            "attempts after the first. Use the `=` form shown here: a value "
            "that starts with a dash and has no space in it would otherwise "
            "look like another relay flag. Needs --checkpoint-glob too, and giving "
            "either one REPLACES the config's `resume:` section entirely. A "
            "script that writes no checkpoints of its own can `import "
            "relay_ckpt` instead and let relay do the saving and restoring."
        ),
    )
    # There is no `passthrough` argument here on purpose; see
    # `_split_passthrough`. argparse's REMAINDER would swallow relay's own
    # flags when they come after the script name.
    p_submit.set_defaults(passthrough=[])
    _json_flag(p_submit)
    p_submit.set_defaults(handler=cmd_submit)

    # -- ls ---------------------------------------------------------------
    p_ls = subparsers.add_parser("ls", help="list runs")
    p_ls.add_argument(
        "--all", action="store_true", help="include finished runs, not just active ones"
    )
    _json_flag(p_ls)
    p_ls.set_defaults(handler=cmd_ls)

    # -- show -------------------------------------------------------------
    p_show = subparsers.add_parser("show", help="show everything known about one run")
    p_show.add_argument("run_id")
    _json_flag(p_show)
    p_show.set_defaults(handler=cmd_show)

    # -- logs -------------------------------------------------------------
    p_logs = subparsers.add_parser("logs", help="print a run's log lines")
    p_logs.add_argument("run_id")
    p_logs.add_argument(
        "--follow", "-f", action="store_true", help="keep printing new lines as they arrive"
    )
    p_logs.add_argument(
        "--since-seq",
        type=int,
        default=0,
        dest="since_seq",
        help="only lines after this event sequence number",
    )
    _json_flag(p_logs)
    p_logs.set_defaults(handler=cmd_logs)

    # -- attach -----------------------------------------------------------
    p_attach = subparsers.add_parser(
        "attach", help="follow a run's logs and metrics as they arrive"
    )
    p_attach.add_argument("run_id")
    p_attach.add_argument(
        "--tail",
        action="store_true",
        help="start from now instead of replaying the run from the beginning",
    )
    _json_flag(p_attach)
    p_attach.set_defaults(handler=cmd_attach)

    # -- cancel -----------------------------------------------------------
    p_cancel = subparsers.add_parser("cancel", help="ask the backend to stop a run")
    p_cancel.add_argument("run_id")
    _json_flag(p_cancel)
    p_cancel.set_defaults(handler=cmd_cancel)

    # -- usage ------------------------------------------------------------
    p_usage = subparsers.add_parser(
        "usage",
        help="report GPU- and CPU-hours, the partition split, and wasted hours",
    )
    p_usage.add_argument(
        "--by",
        choices=sorted(USAGE_GROUPINGS),
        default="run",
        help="group the table by run (default), Slurm account, or partition",
    )
    p_usage.add_argument(
        "--since",
        help="only attempts that started on or after this date, e.g. 2026-09-01",
    )
    p_usage.add_argument("--group", help="only one Slurm account's runs")
    _json_flag(p_usage)
    p_usage.set_defaults(handler=cmd_usage)

    # -- connect ----------------------------------------------------------
    p_connect = subparsers.add_parser(
        "connect",
        help="open the SSH connection the daemon reuses (asks for Duo once)",
    )
    _json_flag(p_connect)
    p_connect.set_defaults(handler=cmd_connect)

    # -- dash -------------------------------------------------------------
    p_dash = subparsers.add_parser("dash", help="serve the dashboard on 127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8765, help="port to listen on")
    p_dash.set_defaults(handler=cmd_dash)

    # -- daemon -----------------------------------------------------------
    p_daemon = subparsers.add_parser(
        "daemon",
        help="run the syncing loop in the foreground until Ctrl-C",
        epilog=(
            "The interval flags below override the `daemon:` section of the\n"
            "config for this one run of the daemon. Every one of them defaults\n"
            "to None, meaning \"not given\", so a flag you leave out keeps\n"
            "whatever the config says."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Every one of these is `type=float, default=None`. The None is what tells
    # `daemon_intervals` the flag was absent; a numeric default here would
    # silently beat the user's config file on every run.
    p_daemon.add_argument(
        "--tail-interval-active",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "how often to read new bytes from the event logs while a run is "
            "producing output (default 2). This is a read on the shared "
            "filesystem, not a question for the scheduler, so it is cheap. "
            "Overrides `daemon.tail_interval_active` in the config."
        ),
    )
    p_daemon.add_argument(
        "--tail-interval-queued",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "how often to read the event logs while every run is still queued "
            "and there is nothing yet to read (default 15). Overrides "
            "`daemon.tail_interval_queued` in the config."
        ),
    )
    p_daemon.add_argument(
        "--tail-interval-idle",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "how often to look when no run is active at all (default 60). "
            "Overrides `daemon.tail_interval_idle` in the config."
        ),
    )
    p_daemon.add_argument(
        "--scheduler-interval-min",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "the shortest gap between `squeue` calls (default 60). That "
            "question goes to slurmctld, which every user on the cluster "
            "shares, so relay will not go below its own 10 second floor "
            "whatever you put here. Overrides `daemon.scheduler_interval_min` "
            "in the config."
        ),
    )
    p_daemon.add_argument(
        "--scheduler-interval-max",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "the longest gap between `squeue` calls (default 300). The poll "
            "backs off towards this while no job changes state, and snaps back "
            "to the minimum when one does. Overrides "
            "`daemon.scheduler_interval_max` in the config."
        ),
    )
    p_daemon.add_argument(
        "--usage-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "how often to fetch GPU- and CPU-hours from `sacct` (default 300). "
            "The most expensive question relay asks and the least urgent; relay "
            "takes one final reading when a run ends whatever this says. "
            "Overrides `daemon.usage_interval` in the config."
        ),
    )
    p_daemon.set_defaults(handler=cmd_daemon)

    return parser


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Cut the command line at the first `--`: relay's args, then the user's.

    Done by hand, before argparse sees anything. The obvious alternative is a
    positional with `nargs=argparse.REMAINDER`, and it is subtly wrong: once
    REMAINDER starts collecting it takes *everything*, so

        relay submit train.py --seeds 0-4 -- --lr 3e-4

    would hand relay's own `--seeds 0-4` to the training script and run one
    seedless job. Splitting first means argparse only ever sees relay's
    vocabulary, and the user's arguments are never interpreted at all — which
    is exactly what "passed through verbatim" should mean.
    """
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


# --------------------------------------------------------------------------
# Exception -> exit code
# --------------------------------------------------------------------------


def _exit_code_for(exc: BaseException) -> int:
    """Map an exception to one of relay's exit codes.

    The Slurm errors are imported lazily, inside the function: `relay ls` on a
    laptop with no cluster should not import the SSH backend just so that this
    mapping can name a class it will never see.
    """
    if isinstance(exc, NotFound):
        return EXIT_NOT_FOUND
    if isinstance(exc, BadUsage):
        return EXIT_USAGE
    # `cmd_submit` re-raises this as BadUsage, so it normally never gets here.
    # The mapping exists anyway because the backends validate their own specs
    # too, and a rejected directive is a bad command line wherever it is
    # noticed -- not a runtime failure the user should go investigating.
    if isinstance(exc, SbatchExtraError):
        return EXIT_USAGE
    if isinstance(exc, ConnectFailed):
        return EXIT_TRANSPORT
    # ConfigNotFound first: it is a subclass of ConfigError, and "you have not
    # run relay init" (3) is a different problem from "your YAML is wrong" (1).
    if isinstance(exc, cfg.ConfigNotFound):
        return EXIT_NOT_CONFIGURED
    if isinstance(exc, cfg.ConfigError):
        return EXIT_FAILURE

    try:
        from relay.backends.slurm import MissingTimeLimit, SlurmError, TransportError
    except Exception:  # pragma: no cover - only if the module itself is broken
        TransportError = ()  # type: ignore[assignment]
        SlurmError = ()  # type: ignore[assignment]
        MissingTimeLimit = ()  # type: ignore[assignment]
    if TransportError and isinstance(exc, TransportError):
        return EXIT_TRANSPORT
    # Before SlurmError, which it subclasses: "you did not say how long the job
    # may run" is a bad command line (2), not something that went wrong out
    # there (1). The fix is a flag, not an investigation.
    if MissingTimeLimit and isinstance(exc, MissingTimeLimit):
        return EXIT_USAGE
    if SlurmError and isinstance(exc, SlurmError):
        # ssh worked and Slurm said no. Retrying will not help, so this is a
        # runtime failure rather than a transport one.
        return EXIT_FAILURE

    if isinstance(exc, OSError):
        # ssh that could not even start, a socket that died, a shared
        # filesystem that went away: "we could not reach the other end".
        return EXIT_TRANSPORT
    return EXIT_FAILURE


def _message_for(exc: BaseException) -> str:
    """The sentence the user sees. Never a traceback.

    Every error relay raises deliberately carries a full sentence, so `str` is
    usually enough. `repr` is the fallback for exceptions that carry no message
    at all (a bare `KeyError`, say), because "relay failed: " followed by
    nothing is worse than an ugly class name.
    """
    text = str(exc).strip()
    return text if text else repr(exc)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse, dispatch, format, and turn any exception into an exit code.

    This is `[project.scripts] relay = "relay.cli:main"`. argparse raises
    `SystemExit(2)` on a malformed command line and we let it through: that is
    already relay's "bad usage" code, and argparse's message is better than
    anything we would write.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    own_args, passthrough = _split_passthrough(raw)

    parser = build_parser()
    args = parser.parse_args(own_args)

    if getattr(args, "handler", None) is None:
        parser.print_help()
        return EXIT_USAGE

    if hasattr(args, "passthrough"):
        args.passthrough = passthrough

    try:
        output = args.handler(args)
        if output is None:
            # The command streamed its own output through `_emit` (follow,
            # attach) or ran something that never returns until Ctrl-C (dash).
            return EXIT_OK
        _emit(output, getattr(args, "json", False))
        return output.exit_code
    except KeyboardInterrupt:
        # Ctrl-C out of `logs --follow` or `dash` is how those are *meant* to
        # end, so it gets a newline and the conventional 128 + SIGINT, not a
        # scary message.
        sys.stderr.write("\n")
        return EXIT_INTERRUPTED
    except BrokenPipeError:
        # `relay logs big_run | head`. The reader went away; that is not an
        # error, and complaining about it into a closed pipe would only raise
        # again on interpreter shutdown.
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - this is the catch-all by design
        # Escape hatch for developing relay itself: RELAY_DEBUG=1 re-raises so
        # a traceback reaches the terminal. It is off by default because a
        # traceback in front of a user is a bug in error handling, not
        # information.
        if os.environ.get("RELAY_DEBUG"):
            raise
        sys.stderr.write(f"relay: {_message_for(exc)}\n")
        return _exit_code_for(exc)


if __name__ == "__main__":  # pragma: no cover
    # `python -m relay.cli ...` for working on relay without installing it.
    sys.exit(main())
