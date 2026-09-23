# relay

An experiment orchestrator for Slurm clusters. Submit training jobs
from your laptop, watch live metrics, and have jobs survive preemption, with
no tracking server, no cloud account, and no inbound network connection to the
cluster.

## How it works

Compute nodes have no inbound network path. They can only write files to a
shared filesystem, so that is the entire channel:

1. A **sidecar** process wraps your training command inside the job and appends
   events to `events.jsonl` on shared storage.
2. A **daemon** on your laptop tails that file over SSH and writes into local
   SQLite.
3. The **CLI** and a **web dashboard** read that SQLite.

No component calls another component's functions. The only shared contracts
are the event log format and the database schema.

## Install

Python 3.11 or newer on your laptop. Nothing is installed on the cluster:
relay copies its two helper files into each run directory at submit time.

```sh
git clone https://github.com/alihatim207/relay.git
cd relay
pip install -e .
```

Or, without a checkout:

```sh
pip install git+https://github.com/alihatim207/relay.git
```

Then `relay init` writes a commented config to `~/.config/relay/config.yaml`,
and `relay doctor` checks it against the cluster.

## Usage

```sh
relay init                 # write config
relay connect              # open the SSH master (answer Duo once)
relay doctor               # check everything end to end
relay submit train.py --time 04:00:00 --seeds 0-4 -- --lr 3e-4
relay ls                   # list runs
relay show <run>           # run detail, attempts, usage
relay logs <run>           # stream logs
relay usage                # GPU-hours, CPU-hours, wasted hours
relay dash                 # local web dashboard
relay daemon               # start the sync daemon
```

Training scripts report metrics by printing a line like:

```
##relay## {"step": 1000, "loss": 0.412}
```

with `flush=True`. No import required.

## Checkpointing under preemption

On a preemptible partition your job can be evicted at any moment, including in
its first minute. Slurm sends SIGTERM and then SIGKILL a fixed number of
seconds later (10 on UW Klone), and relay cannot extend that window. The
sidecar forwards the SIGTERM to your script immediately and gives it
`preempt_grace - 2` seconds before killing it, but that is a best-effort save,
not a guarantee. Write your training script accordingly:

- **Checkpoint periodically anyway.** The signal handler only makes your last
  checkpoint more recent, and only when the save fits in the window. It is not
  a substitute for saving on a schedule. If a checkpoint takes 30 seconds to
  write and the window is 10, the signal-triggered save will not finish.

- **Save atomically.** Write to a temporary file and `os.replace()` it into
  place:

  ```python
  torch.save(state, path + ".tmp")
  os.replace(path + ".tmp", path)
  ```

  `os.replace` is a rename, which is atomic. A save cut off by SIGKILL then
  leaves the previous checkpoint intact instead of a half-written file that
  will not load.

- **Make the SIGTERM handler safe to run twice.** Your script can receive
  SIGTERM from Slurm directly *and* from the sidecar forwarding it. Guard the
  handler with a flag and return immediately on the second call.

A run whose time limit is approaching is handled separately: Slurm sends
SIGUSR1 first (relay asks for it with `--signal=B:USR1@<grace>`), and on that
signal the sidecar checkpoints, ends the attempt as `requeued`, and puts the
job back in the queue itself.

## Resuming after preemption

Relay works fully without either of these. Both are optional.

**If your script already writes checkpoints**, two config lines tell relay how
to find the newest one and hand it back on the next attempt:

```yaml
resume:
  checkpoint_glob: "checkpoints/*.pt"   # relative to the job's working dir, ** allowed
  arg: "--resume {path}"                 # appended to the command; {path} is shell-quoted
```

On the first attempt nothing happens. On every attempt after a preemption, the
sidecar finds files matching the glob that were written since the run began,
picks the newest, and appends `--resume <that file>` to your command. The glob
is relative to the directory your script runs in: on Slurm that is the run
directory (relay submits with `--chdir`), locally it is wherever you ran
`relay submit`. Write checkpoints relative to the current directory and the
same glob works in both places. `relay
show` lists what each attempt resumed from. Per-run overrides:
`relay submit --checkpoint-glob ... --resume-arg ...`.

**If your script does not checkpoint**, use the helper relay ships next to
your job. It is one file, standard library only, importable because the run
directory is on `PYTHONPATH`:

```python
import relay_ckpt

state = relay_ckpt.restore(lambda: {"params": init_params(), "opt": init_opt()})
for step in range(relay_ckpt.step, total_steps):
    state = train_step(state)
    relay_ckpt.tick(state)
```

`tick` saves every ten minutes by default (`relay_ckpt.configure(every_minutes=...,
every_steps=..., keep=2)` to change it). When Slurm sends SIGTERM, the next
`tick` saves and exits cleanly, so a `finally` block still runs; after the
requeue, `restore` returns the saved state and `relay_ckpt.step` picks up where
the loop left off. Saves go through a temporary file and `os.replace`, JAX
arrays are pulled to host and torch tensors to CPU automatically, and each save
prints a `##relay##` checkpoint line so the dashboard sees it.

**One limitation.** `relay cancel` writes a `CANCEL_REQUESTED` file into the
run directory before it sends the cancel, which is how the sidecar knows not to
requeue a run you asked it to stop. A run cancelled with raw `scancel` has no
such file, so if it is configured with `requeue_on_term: always` it can requeue
itself and come back. Cancel through relay, or use `requeue_on_term: auto`
(the default), which resolves to `never` on any partition whose `PreemptMode`
already includes `REQUEUE`, including Klone's.

## Using relay from Claude Code

relay is a plain CLI with no daemon-side API, which makes it easy for a
coding agent to drive. Every read command takes `--json`, every command exits
with a meaningful code, and errors go to stderr as one sentence saying what
happened and what to do next. Claude Code can submit runs, watch them, read
logs, and report GPU-hours without any integration beyond having `relay` on
`PATH`.

### One-time setup, done by you

Two steps need a human and cannot be done by an agent:

```sh
relay connect   # opens the SSH master; answers Duo once
relay daemon    # leave this running in its own terminal tab
```

`relay connect` waits for a Duo push, so an agent that runs it will sit until
the timeout. The daemon is what moves events from the cluster into the local
database; with it stopped, `relay ls` still works but shows stale data and
says so.

### Tell Claude Code how to use it

Put this in the `CLAUDE.md` of the repo that holds your training scripts,
adjusting the paths:

```markdown
## Submitting jobs with relay

Training runs go to the Slurm cluster through the `relay` CLI. Config is in
~/.config/relay/config.yaml and already sets the account, partition, time
limit, GPU, and the conda environment; do not pass those flags.

- Submit: `relay submit /abs/path/to/train.py -- --script-args here`
  Everything after `--` reaches the script. Use `--seeds 0-4` for a sweep,
  `--time HH:MM:SS` to override the wall clock, `--name` to label the run.
- Watch: `relay ls --json`, `relay show <run_id> --json`,
  `relay logs <run_id>` (never `--follow`, it blocks).
- Metrics: the script must print `##relay## {"step": N, "loss": ...}`
  with `flush=True`; anything else it prints is a log line.
- Cost: `relay usage --json`.
- Stop: `relay cancel <run_id>`. Ask before cancelling anything.
- Never run `relay connect`, `relay daemon`, `relay dash`, or `relay attach`;
  they are interactive or long-running and I run them myself.
- Exit codes: 0 ok, 1 runtime failure, 2 bad usage, 3 not configured,
  4 cannot reach the cluster (tell me to run `relay connect`), 5 no such run.
- Job state in `relay ls` is polled from Slurm no more than once a minute
  and backs off to five minutes while nothing changes. Do not loop on
  `relay ls` waiting for a job to start; check once, report, and move on.
```

### Permissions

In that repo's `.claude/settings.json`, let the read-only commands run without
a prompt and keep the ones that spend cluster time or stop a job behind one:

```json
{
  "permissions": {
    "allow": [
      "Bash(relay ls*)",
      "Bash(relay show*)",
      "Bash(relay logs*)",
      "Bash(relay usage*)",
      "Bash(relay doctor*)"
    ],
    "ask": [
      "Bash(relay submit*)",
      "Bash(relay cancel*)"
    ]
  }
}
```

### What a session looks like

Claude Code edits the training script, runs `relay submit ... -- --lr 3e-4`,
reads back the run ID, and a few minutes later checks `relay show <run_id>
--json` for the latest step and metrics. If a run fails, `relay logs <run_id>`
has the traceback. If the cluster is unreachable, the exit code is 4 and the
message says to run `relay connect`, which is your cue, not the agent's.

## What relay runs on the login node

Relay reaches the cluster from your laptop over a single SSH connection. This
is everything it runs there:

| Command | How often | What it touches |
| --- | --- | --- |
| `tail -c +N <run>/events.jsonl`, one per active run | every 2s while a run is producing output, 30s while everything is queued, 60s when nothing is active | shared filesystem |
| `squeue --me`, one call covering every run | no more often than every 60s, backing off to 5 minutes while nothing changes | slurmctld |
| `sacct -j <ids>`, one call covering every run | every 5 minutes, plus one final reading when a run finishes, then never again for it | Slurm accounting |
| `mkdir` and `tar -x`, then `sbatch` | two round trips, once per `relay submit` | shared filesystem, then slurmctld |
| one idle SSH control socket | held open, `ControlPersist=8h` | sshd |
| `command -v sbatch`, `sacctmgr`, `scontrol show partition`, `df` | only when you run `relay doctor` | shared filesystem and slurmctld |

Every command relay runs there goes through `ssh -T ... bash --noprofile --norc
-c '...'`, so it is one shell per call and your `~/.bashrc` is skipped. None of
it is CPU-heavy: the sidecar and your training script run on the compute node,
not here.

The two schedules are deliberately separate. Tailing an event log is a
filesystem read that scales with how much your job has printed, so it can run
every two seconds without anyone noticing. `squeue` is a question put to
slurmctld, one process serving every user on the cluster, so relay holds it to
at least `scheduler_interval_min` (default 60 seconds) no matter how busy your
runs are. After each poll in which no job changed state it multiplies the
interval by 1.5, up to `scheduler_interval_max` (default 300 seconds); any
state change snaps it straight back to the minimum, and `relay submit` and
`relay cancel` reset it too, so a job you just submitted or cancelled is
noticed promptly. There is a hard floor of 10 seconds: configure anything below
it and relay clamps it up and says so.

Six optional keys in a `daemon:` section of the config tune all of this, all in
seconds: `tail_interval_active`, `tail_interval_queued`, `tail_interval_idle`,
`scheduler_interval_min`, `scheduler_interval_max`, `usage_interval`. `relay
daemon` takes the same six as flags, `--tail-interval-active`,
`--tail-interval-queued`, `--tail-interval-idle`, `--scheduler-interval-min`,
`--scheduler-interval-max`, `--usage-interval`, and a flag beats the file.

Because the two schedules run independently, `relay ls`, `relay usage` and the
dashboard end with two ages rather than one, `metrics 2s ago · job state 41s
ago`, so job state that is minutes old reads as the design working rather than
as a daemon that has stopped.
