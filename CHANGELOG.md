# Changelog

## Unreleased

### Changed

- **The daemon polls the scheduler on its own schedule, separate from tailing event logs.** One cycle used to do a `squeue` and a `tail` together, as often as every 2 seconds while a job was active. A tail is a filesystem read that bothers nobody; `squeue` asks slurmctld, which every user on the cluster shares and which most sites ask you not to poll more than about every 30 seconds. The tail keeps the 2/30/60 second adaptive schedule, so metrics are as live as they were. The scheduler poll is now held to at least `scheduler_interval_min` (default 60s) whatever the runs are doing, multiplies by 1.5 after every poll in which no job changed state up to `scheduler_interval_max` (default 300s), and snaps back to the minimum on any state change. `relay submit` and `relay cancel` poke the daemon so a job you just submitted or cancelled is noticed promptly. A configured minimum below the 10s floor is clamped up with a warning. **Behaviour change:** job state in `relay ls` may now lag by up to `scheduler_interval_max` while nothing is changing. Metrics are unaffected.
- `relay ls`, `relay usage` and the dashboard end with two freshness ages instead of one: `metrics 2s ago · job state 41s ago`. One "last synced" number had to be the older of the two, which made a healthy daemon look stalled. `--json` keeps `last_synced`/`last_synced_age_s` and adds `last_scheduler`/`last_scheduler_age_s`.

### Fixed

- `relay doctor` blamed a slow round trip on ssh "opening a fresh connection" and, at 15s, failed the check and skipped every remote check. On Klone a round trip through a live master takes 4–5s because the login node is slow to start a shell. The check now branches on whether the master is alive and only points at `~/.bashrc` startup cost when it is; every remote timeout in the backend, daemon, and doctor is the single configurable `ssh_timeout` (default 30); relay's probe commands run with `ssh -T`, closed stdin, and `bash --noprofile --norc -c` (never the training job).
- `relay connect` failed after Duo with `unix_listener: cannot bind to path ~/.relay/...: No such file or directory`: the ControlPath directory was never created. Every ssh argv builder now ensures `~/.relay` exists with mode 0700 (and corrects a looser mode), so connect, the daemon, doctor, and the backend are all covered. `doctor` gained a `control_dir` check. ssh exit-255 errors now quote ssh's last stderr line instead of always suggesting "check that plain ssh works".

### Added

- Optional `daemon:` config section with `tail_interval_active`, `tail_interval_queued`, `tail_interval_idle`, `scheduler_interval_min`, `scheduler_interval_max` and `usage_interval`, all in seconds, and the matching `relay daemon` flags `--tail-interval-active`, `--tail-interval-queued`, `--tail-interval-idle`, `--scheduler-interval-min`, `--scheduler-interval-max` and `--usage-interval`. A flag beats the file.
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
