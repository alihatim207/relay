# Changelog

## Unreleased

### Fixed

- `relay connect` failed after Duo with `unix_listener: cannot bind to path ~/.relay/...: No such file or directory`: the ControlPath directory was never created. Every ssh argv builder now ensures `~/.relay` exists with mode 0700 (and corrects a looser mode), so connect, the daemon, doctor, and the backend are all covered. `doctor` gained a `control_dir` check. ssh exit-255 errors now quote ssh's last stderr line instead of always suggesting "check that plain ssh works".

### Added

- Two optional resume paths after preemption: config-driven (`resume.checkpoint_glob` + `resume.arg`) and the `relay_ckpt` helper module.
- `setup` shell lines and `sbatch_extra` directives in config and on `relay submit`; `slurm.gres`. `sbatch_defaults` removed in favour of `sbatch_extra`.

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
