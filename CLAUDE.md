# relay

## What we're building

`relay`, a local-first experiment orchestrator for Slurm clusters. It lets a researcher submit training jobs from their laptop, watch live metrics, and have jobs survive preemption, with no tracking server, no cloud account, and no inbound network connection to the cluster.

The author is an undergraduate learning systems programming through this project. Explain what you're doing and why as you go. Prefer clear code over clever code. When you have a choice between a simple implementation and a sophisticated one, take the simple one and say what the sophisticated alternative would be.

## The constraint that shapes everything

Compute nodes have no inbound network path. Nothing can connect to them. They can write files to a shared filesystem, and that is the entire channel back to the user.

So the architecture is: a sidecar process inside the job appends events to a file on shared storage, and a daemon on the laptop tails that file over SSH and writes into local SQLite. A CLI and a web dashboard read that SQLite. No component calls another component's functions. The only shared contracts are the event log format and the database schema.

## Environment

Cluster: UW Hyak Klone. Shared filesystem is GPFS at `/mmfs1`. User is `ahatim`, group is `mlopt`, remote root is `/mmfs1/gscratch/mlopt/ahatim/relay`. SSH alias is `klone`, defined in the user's `~/.ssh/config`. Relay manages its own ControlMaster; see `relay connect`. Scheduler is Slurm. Authentication involves Duo, so new connections are expensive and multiplexing is mandatory.

Never store a username or hostname in relay's config. Store the SSH alias and hand it to the `ssh` binary, so the user's existing SSH configuration does the work.

### Confirmed Klone facts

Confirmed on Klone (klone-login03, September 2026) with:

```
sinfo -s
scontrol show partition ckpt-all | grep -E "PreemptMode|GraceTime|MaxTime"
scontrol show config | grep -iE "preempt|killwait"
```

- Preemptible partitions: `ckpt` (the cluster default partition), `ckpt-all`, and `ckpt-g2`. `ckpt-all` is the union of the other two. `ckpt` covers the older GPU generations (2080 Ti, RTX 6000, A40, A100). `ckpt-g2` covers L40, L40S, H200, and the newer CPU nodes.
- `PreemptMode=REQUEUE` both on `ckpt-all` and cluster-wide. Slurm requeues preempted jobs itself, so `requeue_on_term: auto` resolves to `never` on Klone and the sidecar does not call `scontrol requeue` on SIGTERM.
- `GraceTime=0` on `ckpt-all`, `KillWait=10 sec` cluster-wide. Effective preemption window: **10 seconds** from SIGTERM to SIGKILL.
- `PreemptType=preempt/qos` and `PreemptExemptTime=00:00:00`. Preemption is decided by QOS priority, and a job can be preempted immediately after it starts.
- `MaxTime=UNLIMITED` at the partition level, `DefaultTime=01:00:00`. A QOS limit may still cap run length; this was not checked.

Do not hardcode partition names in code; they come from config. Suggested config default for Klone GPU work: `partition: ckpt-all`.

## Scope

In scope: `init`, `doctor`, `connect`, `submit`, `ls`, `show`, `logs`, `cancel`, `attach`, `usage`, `dash`, `daemon`. Two backends, local and Slurm. Live metric streaming. SIGTERM and SIGUSR1 handling with checkpoint and requeue. GPU-hour and CPU-hour accounting from `sacct`. Two optional resume paths after preemption: config-driven (`resume.checkpoint_glob` + `resume.arg`) for scripts that already checkpoint, and the `relay_ckpt` helper for scripts that do not. Relay works fully with neither.

Out of scope, do not build: hyperparameter search, early stopping, ASHA, statistical comparison, rliable integration, checkpoint deduplication, content addressing, provenance capture, LaTeX export, MCP server, websockets, authentication, multi-user features.

If a change starts pulling in something from the out-of-scope list, stop and say so.

## Stack

Python 3.11+. Standard library wherever possible. Dependencies limited to roughly: `pyyaml`, `fastapi`, `uvicorn`, and a CLI arg library. `sidecar.py` must import **only** the standard library, enforced by a test. No frontend build step, no npm, no bundler. The dashboard is one HTML file with vanilla JS and a CDN chart library.

## Event schema

One JSON object per line, appended to `events.jsonl` in the run directory. Fixed envelope, type-specific payload.

```json
{"v":2,"run":"vr_s1_7f3a","seq":2,"ts":"2026-09-16T18:22:09.003Z","type":"metric","payload":{"step":1000,"values":{"loss":0.412,"sps":18400}}}
```

Envelope fields: `v` is schema version, currently 2. The daemon must accept both 1 and 2. `run` is relay's own run ID, never Slurm's job ID. `seq` is a per-run monotonic counter, continuing across requeue attempts. `ts` is ISO-8601 UTC with a Z. `type` is one of `run_started`, `metric`, `log`, `heartbeat`, `checkpoint`, `signal`, `run_ended`.

Metric payloads nest their numbers under `values` so the envelope stays a fixed parseable shape regardless of what a user logs.

`run_started.payload` carries `attempt`, the value of `SLURM_RESTART_COUNT` (0 on the first run). `signal.payload` carries `signal` and `reason` (`time_limit` for SIGUSR1, `term` for SIGTERM). `run_ended.payload.status` is one of `completed` (child exit 0), `failed` (child nonzero), `requeued` (time-limit self-requeue), `interrupted` (SIGTERM).

Training scripts report metrics by printing `##relay## {"step": 1000, "loss": 0.412}` with `flush=True`. Any other stdout line becomes a `log` event. No import required on the user's side.

## The sidecar

One stdlib-only file, copied next to the job. It wraps the training command as a child process.

Responsibilities: emit `run_started` with the attempt number. Spawn the child with stdout and stderr piped. Run a reader thread per pipe, since a single thread reading one pipe can deadlock when the other fills. Parse `##relay##` lines into metric events, everything else into log events. Emit a heartbeat every 30 seconds carrying the last seen step. Install handlers for SIGTERM and SIGUSR1. Wait for the child and emit `run_ended` with its exit code. Exit with the child's exit code, not zero.

The sidecar never writes model checkpoints. The training script does. The sidecar provides timing, ordering, and requeue. It only observes that a checkpoint happened.

On startup, if `events.jsonl` already exists, read its tail and continue `seq` from the highest value found. A requeued attempt restarting `seq` at 1 would collide with the previous attempt's events and cause silent data loss.

On startup the sidecar also reads `<run_dir>/sidecar.json`, written by the backend at submit: `{"requeue_on_term": "always"|"never", "preempt_grace": N, "first_attempt_start": <epoch seconds>, "resume": {"checkpoint_glob": ..., "arg": ...} | null}`. Defaults when absent: `never`, 10, 0, null. The sidecar never queries Slurm itself. `LocalBackend` preserves `first_attempt_start` when it resubmits into the same run directory.

The child always gets `RELAY_RUN_ID` and `RELAY_RUN_DIR` in its environment, identical across requeue attempts, and the run directory prepended to `PYTHONPATH` so it can `import relay_ckpt`.

### Resume

Two optional ways for a job to pick up after preemption. Relay must work fully with neither.

**Config-driven, for scripts that already checkpoint.** `resume.checkpoint_glob` (relative to the job's working directory, `**` allowed) and `resume.arg` (a string containing `{path}`), global in config or per run via `relay submit --checkpoint-glob/--resume-arg`. On attempt 0 the sidecar does nothing. On attempt > 0 it globs (stdlib `glob`, `recursive=True`), keeps files whose mtime is at or after `first_attempt_start`, picks the newest by mtime, formats `arg` with `{path}` shell-quoted, and appends the result to the child command. It records the path as `resumed_from` in `run_started` and emits a log event naming it. No match: command unchanged, warning log event, `resumed_from` null. `relay show` lists the resumed-from path per attempt.

**`relay_ckpt`, for scripts that do not checkpoint.** One stdlib-only file, covered by the same AST import test as the sidecar, copied into the run directory at submit by both backends. It must not import jax or torch at module level; it checks `sys.modules` at save time. API, backed by one lazily created default instance: `restore(default)`, `tick(state)`, `step`, `configure(every_minutes=10, every_steps=None, keep=2, directory=None, save_fn=None, load_fn=None)`. Directory: explicit, else `$RELAY_RUN_DIR/checkpoints`, else `./checkpoints`. The first call installs a SIGTERM handler that only sets a flag, chaining any previous callable handler. `tick` increments the counter, then saves if the interval elapsed or the flag is set; if the flag is set it saves and raises `SystemExit(0)` so user `finally` blocks run. The tick counter is saved with the user's state and restored by `restore`, which falls back to the next-newest checkpoint with a stderr warning if the newest fails to load. Save: jax → `device_get`, torch → tensors to CPU; write `ckpt_<step>.pkl.tmp`, flush, fsync, `os.replace`, fsync the directory, prune to the newest `keep`, then print a `##relay##` checkpoint line. `save_fn`/`load_fn` replace pickle but keep the tmp/fsync/replace sequence.

The event writer uses `O_APPEND`, one `os.write` per complete line, and no buffering. Buffered events that never reach disk before SIGKILL never existed.

### Signal semantics

Two different signals for two different situations. `scancel` and preemption both send SIGTERM, so SIGTERM alone cannot mean "requeue"; a user cancelling a run must not get it back.

**SIGUSR1: time limit approaching.** The sbatch directive is `--signal=B:USR1@<grace>`; only Slurm's time-limit warning sends USR1, so self-requeue is always correct here. On SIGUSR1: emit a `signal` event with `"reason": "time_limit"`, forward SIGTERM to the child so it checkpoints, wait up to `grace - 2`, SIGKILL if still alive, emit `run_ended` with status `requeued`, flush, run `scontrol requeue $SLURM_JOB_ID`, exit 0.

**SIGTERM: preemption or cancel, indistinguishable at signal time.** Emit a `signal` event with `"reason": "term"` and forward SIGTERM to the child immediately. Do the minimum possible work between receiving SIGTERM and forwarding it: no SSH, no subprocess calls, no file scans on that path. Then wait up to `preempt_grace - 2` seconds (floored at 1), SIGKILL if still alive, emit `run_ended` with status `interrupted`, flush, and decide on requeue:

- If a file named `CANCEL_REQUESTED` exists in the run directory, do not requeue. Exit.
- Else if `requeue_on_term` is `never`, do not requeue. Exit.
- Else if `requeue_on_term` is `always`, run `scontrol requeue $SLURM_JOB_ID`, then exit.
- `auto` (the config default) behaves like `never` when the partition's PreemptMode includes REQUEUE (Slurm handles it) and like `always` otherwise. The backend resolves it at submit time by reading `scontrol show partition <p>` and writes the result into `sidecar.json`.

`relay cancel` writes `CANCEL_REQUESTED` into the run directory **before** calling `scancel`. Runs cancelled with raw `scancel` under `requeue_on_term: always` can still be requeued; the README documents this limitation.

**Final status is the scheduler's call.** The sidecar writes `interrupted` because it cannot know whether it was preempted or cancelled. The daemon resolves it from `sacct` (`PREEMPTED`, `REQUEUED`, `CANCELLED`) exactly as it already does for job state.

### Grace budget

Config key `preempt_grace` (seconds, default 10, Klone's measured value). The 2 second margin is for the sidecar to write and flush `run_ended` before SIGKILL reaches it too; that is two small `os.write` calls and needs milliseconds. `doctor` reports the real value on other clusters.

On Klone the entire window from SIGTERM to SIGKILL is 10 seconds and preemption can happen at any moment, so the signal-triggered save is best-effort. The README states plainly: the training script must checkpoint periodically; saves must write to a temporary file and `os.replace` it into place; the script's SIGTERM handler must be safe to run twice (Slurm and the sidecar's forward can both deliver it).

## Config

YAML at `~/.config/relay/config.yaml`, written by `relay init` with every key documented and commented out. Store the SSH alias, never `user@host`.

Top level: `backend` (`local` | `slurm`), `ssh_alias`, `remote_root`, `remote_python`, `local_root`, `ssh_timeout` (seconds, default 30, minimum 5), `requeue_on_term` (`auto` | `always` | `never`), `preempt_grace` (seconds, default 10), `setup`, `slurm:`, `cost:`, `resume:`.

`resume:` section: `checkpoint_glob` and `arg`, both or neither; `arg` must contain `{path}`. See Resume under the sidecar section. Per-run: `relay submit --checkpoint-glob ... --resume-arg ...` replaces both.

`slurm:` section: `account`, `partition`, `time` (the job's `--time`, required at submit unless `--time` is passed), `grace_seconds` (the `--signal=B:USR1@<grace>` time-limit warning), `gres` (Slurm's `--gres` value, e.g. `gpu:1`), `sbatch_extra`.

`cost:` section: `gpu_hour_rate`. No default; without it relay shows no dollar figures.

`setup` is a list of shell lines, rendered in order after the `#SBATCH` block and before the sidecar command. Used for things like `module load cuda` and `conda activate rl`. Each line is rendered verbatim; relay does not parse or validate their contents. They run under `set -e` so a failed `conda activate` fails the job immediately instead of running training in the wrong environment, and `set +e` follows them so the sidecar's own exit-code handling is unaffected. `LocalBackend` runs them in the same shell as the sidecar, so an exported variable reaches the child.

`sbatch_extra` is a list of sbatch directives, each written as the flag and value like `--mem=64G` or `--cpus-per-task=8`. Each renders as `#SBATCH <entry>` after the directives relay generates. It is the escape hatch for any flag relay does not model. `LocalBackend` ignores it with one debug log line.

Both `setup` and `sbatch_extra` can appear in the global config and on the `relay submit` command line (`--setup LINE`, `--sbatch-extra ENTRY`, both repeatable). The submit-time list replaces the global list entirely; they are never merged.

`submit` validates `sbatch_extra` before anything is rendered or sent over SSH, exiting 2 with a message naming the entry. Rejected: flags relay owns (`--output`/`-o`, `--error`/`-e`, `--open-mode`, `--signal`, `--requeue`, `--no-requeue`, `--job-name`/`-J`), because overriding them breaks log continuity, time-limit handling, or requeue; flags relay sets from its own fields (`--account`/`-A`, `--partition`/`-p`, `--time`/`-t`, `--gres`), because two values in one script means the last silently wins, so the message names the config field to use instead; and entries that do not start with `-`. The rules live in `backends/base.py` so the CLI and both backends agree.

## Backends

```python
class Backend:
    def submit(self, spec: JobSpec) -> str: ...
    def status(self, job_ids: list[str]) -> dict[str, str]: ...
    def cancel(self, job_id: str, run_dir: str | None = None) -> None: ...
    def read_bytes(self, path: str, offset: int) -> tuple[bytes, int]: ...
    def usage(self, job_ids: list[str]) -> list[UsageRow]: ...
```

`status` takes a list and returns a dict, because one `squeue` call covers every job and per-job calls would mean one SSH round trip per run. `usage` takes a list for the same reason.

Backends know nothing about the event schema. `read_bytes` returns raw bytes from a path. Parsing happens in the daemon.

`SlurmBackend.submit` renders an sbatch script to a real file on shared storage (never piped from stdin, so users can inspect what was submitted), copies `sidecar.py` and `sidecar.json` beside it, runs `sbatch` over SSH, and parses the job ID from `Submitted batch job N`.

The sbatch script is rendered in this order: relay's own directives (`--job-name`, `--account`, `--partition`, `--gres`, `--chdir`, `--output`, `--time`, `--open-mode=append`, `--requeue`, `--signal=B:USR1@<grace>`), then one `#SBATCH <entry>` per `sbatch_extra` entry, then `set -e`, the `setup` lines, `set +e`, then the sidecar `exec` line. The script must include `--requeue`, `--signal=B:USR1@<grace>`, `--open-mode=append`, and an explicit `--time`. Without append mode, a requeued attempt truncates the previous attempt's output file. Without `--time`, Klone's `DefaultTime=01:00:00` silently caps the job at one hour; `submit` requires a time limit from config or `--time` and exits 2 with a clear message if neither is given.

`status` asks `squeue --me --noheader --format` for live jobs, then falls back to `sacct` for any requested ID that `squeue` didn't return, because `squeue` drops jobs the moment they finish.

`cancel` with a `run_dir` creates `CANCEL_REQUESTED` there before sending the cancel.

`usage` makes one call for all IDs:

```
sacct -j <id1,id2,...> -X --duplicates --noheader --parsable2 \
  --format=JobIDRaw,State,Partition,Start,End,ElapsedRaw,AllocTRES
```

`-X` returns allocations only, not steps. `--duplicates` returns every requeue attempt under the same job ID instead of only the latest; without it, a run preempted three times undercounts. GPU count comes from `AllocTRES`, which can say `gres/gpu=2` or typed `gres/gpu:a40=2`; prefer the untyped key when both are present, never sum them. CPU count from `cpu=N`. No GPU key means zero GPUs, not an error. The sacct parser is a pure function tested against fixture strings in `tests/fixtures/sacct_*.txt`; those should be real Klone output.

`LocalBackend` spawns the sidecar as a subprocess with `start_new_session=True`, uses the PID as the job ID, and reads files directly. Its `usage` returns rows from process wall time with zero GPUs. It sets `SLURM_RESTART_COUNT` itself when it restarts a run. This backend is how the full test suite runs in CI without a cluster.

## SSH

Relay manages its own ControlMaster. Every ssh invocation is built by one function in `ssh.py`:

```
ssh -o ControlMaster=auto
    -o ControlPath=~/.relay/cm-%C
    -o ControlPersist=8h
    -o BatchMode=yes
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
    <alias> <command>
```

Command-line `-o` options override `~/.ssh/config`, so this works whether or not the user set anything up. The alias still supplies user, host, keys, and jump hosts.

`ControlPath` lives in `~/.relay/`, not the deeper XDG data directory, because Unix socket paths are limited to about 104 bytes. `%C` is a hash, which keeps the path short and unique per host.

ssh does not create the ControlPath's parent directory; it fails with `unix_listener: cannot bind to path ...: No such file or directory` after Duo has already been answered. So every argv builder in `ssh.py` calls `ensure_control_dir()` first: `makedirs` with mode 0700, then `chmod` to 0700 if it already existed looser. That covers `connect`, the daemon, doctor, and the backend by construction; the daemon and doctor can run before `connect` ever has. `doctor` reports the directory's state as its own check. A fake ssh runner cannot catch this class of bug because the bind happens inside the real ssh binary, which is why `tests/test_ssh_integration.py` runs a real `ssh -M -N -f` against localhost when a local sshd is available and skips otherwise.

When ssh exits 255, relay's error carries the last line of ssh's stderr. The generic "check that `ssh <alias>` works" advice appears only when stderr is empty.

### Timeouts and latency

A round trip through a live ControlMaster on Klone takes 4 to 5 seconds. None of that is network: the login node is slow to start a shell (a `conda init` in `~/.bashrc` on a shared, busy node). A loaded login node is much worse. So the floor to plan for is about 5 seconds per round trip, and latency alone is never evidence that ssh opened a new connection.

One configurable value, `ssh_timeout` (default 30, minimum 5), is the budget for every remote call in `SlurmBackend`, the daemon, and doctor. No remote call anywhere uses a literal timeout of its own or waits less than `ssh_timeout`; a call that genuinely needs longer uses a documented multiple of it.

Relay's own probe commands run with `ssh -T`, stdin closed, and `bash --noprofile --norc -c` so the user's shell startup files are skipped where relay controls the command. This cleans the inner shell only: sshd still spawns the user's login shell to run the command, and some bash builds source `~/.bashrc` on every SSH invocation, which is exactly the cost doctor's warning points at. Never applied to the training job, which runs under Slurm in the user's own environment.

`BatchMode=yes` is essential: without it, a missing master makes ssh wait forever for a Duo prompt nobody will see, hanging the daemon.

`relay connect` runs `ssh -M -N -f` with the same ControlPath and ControlPersist but **without** BatchMode, so the user can complete Duo interactively. On success it prints that the connection is established and how long it persists. Exits 4 on failure.

The real failure mode is not ControlPersist expiring (the daemon's own polling keeps the master busy) but the laptop sleeping or changing networks, which kills the TCP connection regardless of any setting.

## Store

SQLite at `~/.local/share/relay/relay.db`. Tables: `runs`, `events` with primary key `(run_id, seq)`, `metrics` with primary key `(run_id, name, step)`, `sources` with primary key `(run_id, path)` holding `byte_offset`, `usage` with primary key `(job_id, start)`, `daemon_state`, and a small `meta` key-value table.

`events` is the faithful history with payload stored as JSON text. `metrics` is a denormalized read path, one row per metric name per step, written at ingest so charting never parses JSON.

`runs.attempts` is updated at ingest from `run_started` events. `runs.usage_final` marks that the post-terminal usage fetch has happened.

Pragmas: `journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=5000`. WAL is required because three processes share this file and the default rollback journal makes readers and writers block each other.

All inserts use `INSERT OR IGNORE`. Combined with the composite primary keys, re-reading the same bytes produces an identical database. This idempotence is what makes "just restart it" a valid recovery strategy everywhere.

The `usage` table is the one exception: a running attempt's `elapsed_s` grows, so it uses `INSERT ... ON CONFLICT DO UPDATE`. It is still idempotent, since re-fetching the same sacct output produces the same rows. Its primary key is `(job_id, start)` because attempts share a job ID but not a start time.

Critically: **event inserts, metric inserts, and the byte-offset update commit in a single transaction.** If they were separate, a crash between them would leave the offset disagreeing with the data.

All SQL lives in `store.py`. No other module writes SQL.

### Derived usage numbers

Computed in `store.py` queries, never stored:

- GPU-hours and CPU-hours per run, per group (the run's Slurm account), and total over a date range.
- Split by partition, so the user sees how much ran on `ckpt*` versus their group's own allocation.
- **Wasted hours**, in two categories. `failed`: all hours of attempts in runs whose final status is `failed`. `lost_to_preemption`: for each interrupted attempt, time from the last `checkpoint` event to attempt end. If an attempt reported no checkpoint events, count the whole attempt and mark it `no_checkpoints` so the user learns their script is not reporting checkpoints.

Optional config key `cost.gpu_hour_rate`. If absent, show no dollar figures anywhere. Never invent a default price.

## Daemon

A loop. Query non-terminal runs. One batched `status` call for all of them. Then per run, read new bytes from the stored offset, split on newlines, and if the data does not end in a newline, pop the trailing incomplete line and rewind the offset by its length so it gets re-read next cycle. Parse, insert, commit, all in one transaction.

Adaptive interval: 2 seconds when runs are actively emitting, 30 when everything is queued, 60 when nothing is active. A shared login node has many users on it; do not poll aggressively for a job that starts tomorrow.

Status conflict resolution: the event log is authoritative about what the training did (steps, metrics, checkpoints). The scheduler is authoritative about what the job did (queued, running, preempted, exit code). A run is only marked terminal when Slurm says so, because a SIGKILLed job writes no `run_ended` at all.

Stale detection: if a run is `running` and its last heartbeat is older than 5 minutes, flag it stale. Stale is a flag, not a status. Do not change the scheduler-owned status and do not kill anything.

Usage is not live data. Fetch every 5 minutes for non-terminal runs, plus once more when a run transitions to terminal, then never again for that run. One batched call per fetch.

Transport failures are normal, not exceptional. Catch, record, return, try next cycle. Back off exponentially on consecutive failures. `FileNotFoundError` on a run directory is not an error; the job is still queued.

SSH failures are classified into two kinds. `auth_required`: exit 255 with no live master (confirmed with `ssh -O check`) or a permission-denied message. `unreachable`: anything else at the transport level. The last error kind and time live in the `daemon_state` table. `relay ls`, `relay usage`, and the dashboard show "Authentication needed. Run `relay connect`." when the kind is `auth_required`, instead of a generic transport error. Backoff rules are unchanged for `unreachable`. For `auth_required`, check once per minute with `ssh -O check` only, which costs nothing and triggers no Duo.

Single instance enforced with `fcntl.flock` on a lock file, not a bare PID file. A crashed daemon's lock releases automatically; a stale PID file does not.

Shutdown sets a flag that the loop checks each pass. Never interrupt a transaction.

## CLI

Every read command takes `--json`. One code path: commands return data structures, and a formatter at the dispatch boundary renders a table or dumps JSON. Never two code paths.

Exit codes: 0 success, 1 runtime failure, 2 bad usage, 3 not configured, 4 transport failure, 5 not found.

Errors go to stderr as plain sentences saying what happened and what to do next. Catch every exception at the top-level dispatch. A traceback reaching a user is a bug in error handling.

`relay ls` output ends with a "last synced Ns ago" line. Stale data the user knows is stale is acceptable; stale data presented as current is not.

`relay submit train.py --seeds 0-4 -- --lr 3e-4` passes everything after `--` through to the user's script. `--time` supplies the wall-clock limit when config does not.

`relay usage [--group G] [--since DATE] [--by run|group|partition] [--json]` reports GPU-hours, CPU-hours, the partition split, and wasted hours. Output ends with the "last synced" line like `ls`.

`relay connect` establishes the SSH master interactively.

## Dashboard

FastAPI serving one static HTML page plus JSON endpoints. Bind `127.0.0.1` only, never `0.0.0.0`; there is no authentication and this runs on a university network.

Polling every 2 seconds, not websockets. Metric requests take `since=<step>` so the browser only fetches new points. Without that, a run with 200k points re-transfers everything every poll.

Cancel is a `POST`, not a `GET`. A browser prefetcher firing a `GET` would cancel someone's eight-hour run.

A usage panel shows totals, the partition split, and the wasted-hours breakdown. When the daemon's last error is `auth_required`, the page says so and points at `relay connect`.

## doctor

Run every check, never fail fast, report all results. Exit 0 on warnings, nonzero on errors, so it works as a CI smoke test.

Checks: config exists and parses; the ControlPath directory exists with mode 0700; relay's master socket is alive (`ssh -O check` with relay's ControlPath; if not, say "Run `relay connect`"); SSH reaches the host and how long it took; remote Python exists at the configured path; `sbatch` on PATH; Slurm account valid; partition exists and its `PreemptMode`, `GraceTime`, `MaxTime`; `remote_root` exists, is writable, free space; database present and in WAL mode; the `usage` table exists and when usage was last fetched; daemon holds its lock and when it last completed a cycle.

The ssh check never infers a cause from latency alone. It already knows from the master check whether the master is alive. Master alive and the round trip slow: warn that the login node is slow to respond and that remote shell startup (for example `conda init` in `~/.bashrc`) is the usual reason; make no claim about a fresh connection and do not suggest `relay connect`. Master absent: keep the `relay connect` suggestion. The slow threshold is `ssh_timeout`; the check's own subprocess timeout is twice that, so a slow round trip can still complete and warn rather than being cut off as an error. A failed ssh check still skips the remote checks that depend on it; a warning does not.

Partition follow-ups: warn if `PreemptMode` includes REQUEUE and config says `requeue_on_term: always` (double requeue risk). Warn if `PreemptMode` lacks REQUEUE and `requeue_on_term` is `never` (preempted runs will not come back). If `GraceTime` is 0, also read `KillWait` from `scontrol show config` and report that as the effective grace. Warn if the configured `preempt_grace` exceeds the effective grace (the sidecar would plan for time it does not have), and print the value to set. Warn if the configured time limit exceeds `MaxTime`.

## Layout

```
src/relay/{cli,config,store,daemon,doctor,events,sidecar,relay_ckpt,ssh}.py
src/relay/backends/{base,local,slurm}.py
src/relay/web/{app.py,static/index.html}
tests/
tests/fixtures/sacct_*.txt
```

`src/` layout so tests run against the installed package. `[project.scripts] relay = "relay.cli:main"` in `pyproject.toml`.

## Repo layout

relay/
├── pyproject.toml
├── README.md
├── LICENSE                     MIT
├── CHANGELOG.md
├── .github/workflows/ci.yml
├── src/relay/
│   ├── __init__.py
│   ├── cli.py                  argument parsing, dispatch, formatting
│   ├── config.py               load, validate, init
│   ├── store.py                all SQL lives here
│   ├── daemon.py               the loop
│   ├── doctor.py               checks
│   ├── events.py               schema constants, parsing
│   ├── sidecar.py              stdlib only, shipped to the cluster
│   ├── relay_ckpt.py           stdlib only, shipped next to the job; optional checkpoint helper
│   ├── ssh.py                  the one place ssh argv is built
│   ├── backends/
│   │   ├── base.py             the interface
│   │   ├── local.py
│   │   └── slurm.py
│   └── web/
│       ├── app.py
│       └── static/index.html
└── tests/
    ├── fixtures/               sacct output used by the usage parser tests
    ├── test_events.py
    ├── test_store.py
    ├── test_sidecar.py
    ├── test_daemon.py
    └── test_e2e_local.py       submit → monitor → preempt → resume

## Tests

All must pass in CI with no cluster, using `LocalBackend`.

Sidecar is stdlib-only, verified by AST-parsing its imports against `sys.stdlib_module_names`. Daemon killed mid-cycle then restarted yields exactly the right event count with no duplicates. A truncated final line rewinds the offset and is picked up next cycle. `read_bytes` raising leaves the daemon alive and recovering. Stale fires after the threshold when output stops. Preempt and requeue continues `seq` rather than restarting it, and the resumed step matches the checkpoint. Full end-to-end: submit, monitor, preempt, resume.

Signals: SIGUSR1 produces status `requeued` and a requeue call (a fake `scontrol` on PATH). SIGTERM with `CANCEL_REQUESTED` present produces `interrupted` and no requeue call. SIGTERM with `requeue_on_term: never` produces no requeue call; with `always`, one. `run_started` carries the correct `attempt` across a restart.

Usage: the parser handles typed and untyped GPU keys, a missing GPU key, multiple attempts per job, and a still-running attempt with no `End`. Three attempts of one job sum correctly. `lost_to_preemption` uses the last checkpoint before each interruption; the `no_checkpoints` case is labeled. Upsert of a growing running attempt updates rather than duplicates.

SSH: the argv builder produces exactly the options above; `connect` omits BatchMode and adds `-M -N -f`. The failure classifier maps representative stderr and exit codes to the right kind. LocalBackend is unaffected.

## Build order

The original build order was: events, sidecar, store, backends (local), daemon, cli, config, web, doctor, slurm. All ten stages are built. Spec change round 1 (signal semantics, schema v2, usage accounting, relay-managed SSH, doctor additions, Klone facts) is merged into this document.

When adding a feature, the same rule applies: each piece working and tested before the next depends on it, and everything must pass in CI with no cluster.
