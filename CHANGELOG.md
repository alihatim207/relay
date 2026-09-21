# Changelog

## Unreleased

### Spec change round 1

- **Signal semantics fixed.** SIGTERM no longer means "requeue": `scancel` sends the same signal, so a cancelled run could come back. Time-limit warnings now arrive as SIGUSR1 (`--signal=B:USR1@<grace>`) and self-requeue; SIGTERM ends the attempt as `interrupted` and requeues only per `requeue_on_term` and the absence of `CANCEL_REQUESTED`. `relay cancel` writes that file before `scancel`.
- Every rendered sbatch script carries an explicit `--time`; `submit` exits 2 without one.
- `run_started` carries `attempt` from `SLURM_RESTART_COUNT`.
- New config keys: `requeue_on_term` (auto/always/never), `preempt_grace` (default 10), `slurm.time`, `cost.gpu_hour_rate`.
- Event schema v2. The daemon accepts v1 and v2.
- **Usage accounting.** `Backend.usage()` from `sacct --duplicates`; `usage` table with upsert; GPU-hours, CPU-hours, partition split, and wasted hours (`failed`, `lost_to_preemption`) as derived queries; `relay usage` and a dashboard panel.
- **Relay-managed SSH.** All ssh argv built in `ssh.py` with relay's own ControlMaster under `~/.relay/`; `relay connect` establishes it interactively; the daemon classifies failures as `auth_required` vs `unreachable`.
- doctor: relay's master socket check, partition `PreemptMode`/`GraceTime`/`MaxTime` with `KillWait` fallback and `preempt_grace` sanity warnings, usage table check.

### Initial build

- events, sidecar, store, local and Slurm backends, daemon, CLI, config, dashboard, doctor.
