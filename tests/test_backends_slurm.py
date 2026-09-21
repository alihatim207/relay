"""Tests for `SlurmBackend`, with no cluster and no SSH connection anywhere.

`SlurmBackend` takes a `runner` -- the single callable that would otherwise
start a real `ssh` process. Every test here passes a `FakeRunner` that records
the argv and stdin it was handed and replays a scripted answer. So these tests
check the two things that actually matter for a backend whose entire job is to
compose command lines correctly:

  * **what goes out** -- the exact argv, the exact bytes on stdin, and how many
    round trips it took. Round trips are the expensive resource on a
    Duo-protected cluster, so their count is part of the contract and is
    asserted on.
  * **what comes back** -- that scripted Slurm output is turned into the right
    status strings, job IDs and exceptions.

`remote_words()` undoes the quoting `_ssh` applies. ssh has no way to pass an
argv vector; it joins everything after the alias with spaces and lets the remote
*shell* re-split the result. So the test rejoins and re-splits the same way the
remote shell would, and asserts on the words that would actually arrive. A test
that looked at the quoted argv directly would pass even if the quoting were
wrong in a way the far end could not undo.

One assertion runs on every single call, in `assert_alias_is_bare`: relay must
never construct `user@host`, and every call must carry relay's own
ControlMaster options. The whole point of storing an SSH alias is that the
user's own `~/.ssh/config` supplies the username, the hostname and the key,
while relay supplies the multiplexed connection that makes polling a 2FA
cluster bearable.
"""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import tarfile
import time
import types
from pathlib import Path

import pytest

from relay import ssh as ssh_mod
from relay.backends import slurm as slurm_mod
from relay.backends.base import STATUSES, JobSpec, SbatchExtraError
from relay.backends.local import default_ckpt_helper_path, default_sidecar_path
from relay.backends.slurm import (
    CANCEL_FILENAME,
    CKPT_HELPER_FILENAME,
    SBATCH_FILENAME,
    SIDECAR_CONFIG_FILENAME,
    SIDECAR_FILENAME,
    MissingTimeLimit,
    SlurmBackend,
    SlurmError,
    TransportError,
    map_slurm_state,
    parse_alloc_tres,
    parse_sacct_usage,
    parse_scontrol_kv,
    render_sbatch,
    resolve_requeue_on_term,
    resolve_time_limit,
)
from relay.config import Config, SlurmConfig

ALIAS = "klone"
REMOTE_ROOT = "/mmfs1/gscratch/mlopt/ahatim/relay"
RUN_DIR = f"{REMOTE_ROOT}/vr_s1_7f3a"

# Hand-written sacct output, one file per shape the parser has to survive.
# Each carries a "# SYNTHETIC" header saying so; they are placeholders until
# real Klone output is captured.
FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Test doubles and helpers
# --------------------------------------------------------------------------


class FakeRunner:
    """Stands in for `subprocess.run`, recording calls and replaying answers.

    `responses` is consumed one per call. Each entry is either a
    `(returncode, stdout, stderr)` tuple or an exception instance, which is
    raised instead -- that is how the "ssh binary is missing" and "ssh timed
    out" paths get exercised without a timeout actually elapsing.

    Running past the end of the script is a test bug, not a pass: it means the
    backend made a round trip nobody planned for, and those are the expensive
    thing. So it fails loudly.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, argv, input_bytes, timeout):
        self.calls.append({"argv": list(argv), "input": input_bytes, "timeout": timeout})
        if not self.responses:
            raise AssertionError(
                f"unscripted ssh call #{len(self.calls)}: {' '.join(argv)}"
            )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    @property
    def count(self) -> int:
        return len(self.calls)


def remote_words(call: dict, alias: str = ALIAS) -> list[str]:
    """The words the *remote* shell would see, with `_ssh`'s quoting undone."""
    argv = call["argv"]
    index = argv.index(alias)
    return shlex.split(" ".join(argv[index + 1 :]))


def assert_alias_is_bare(runner: FakeRunner, alias: str = ALIAS) -> None:
    """Every call must carry relay's ssh options and a bare alias.

    Two separate promises, checked together because they are broken in the
    same place -- the one function that builds an argv:

      * the full ControlMaster option set is present on *every* call, so no
        single code path can sneak out over an unmultiplexed connection and
        trigger a Duo prompt in the middle of a daemon cycle
      * the alias arrives alone, never as `user@host`
    """
    expected_options = ssh_mod.build_ssh_argv(alias, [])[1:-1]
    for call in runner.calls:
        argv = call["argv"]
        assert argv[0] == "ssh"
        assert argv[1 : 1 + len(expected_options)] == expected_options, (
            f"call is missing relay's ssh options: {argv}"
        )
        assert argv[1 + len(expected_options)] == alias, f"alias not bare in {argv}"
        for word in argv:
            assert "@" not in word, f"{word!r} looks like a user@host"


def make_backend(*responses, **kwargs) -> tuple[SlurmBackend, FakeRunner]:
    runner = FakeRunner(*responses)
    backend = SlurmBackend(
        ssh_alias=kwargs.pop("ssh_alias", ALIAS),
        remote_root=kwargs.pop("remote_root", REMOTE_ROOT),
        runner=runner,
        **kwargs,
    )
    return backend, runner


def make_spec(**kwargs) -> JobSpec:
    fields = {
        "run_id": "vr_s1_7f3a",
        "command": ["python", "train.py", "--lr", "3e-4"],
        "run_dir": RUN_DIR,
        # Every rendered script needs a time limit, so the default spec has
        # one. The tests that care about where it comes from set it
        # explicitly, or clear it to check the refusal.
        "time_limit": "04:00:00",
        # Configured rather than "auto" by default, so a test that is not
        # about requeue resolution does not have to script a scontrol call.
        "requeue_on_term": "never",
    }
    fields.update(kwargs)
    return JobSpec(**fields)


def tar_contents(input_bytes: bytes) -> dict[str, bytes]:
    """Unpack the tar stream submit sends on stdin: {name: contents}."""
    with tarfile.open(fileobj=io.BytesIO(input_bytes), mode="r") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }


# --------------------------------------------------------------------------
# render_sbatch
# --------------------------------------------------------------------------


def test_render_includes_the_directives_that_survive_preemption():
    script = render_sbatch(make_spec(grace_seconds=90), remote_python="python3")

    assert script.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=vr_s1_7f3a" in script
    assert "#SBATCH --requeue" in script
    # Without append, a requeued attempt truncates the first attempt's output.
    assert "#SBATCH --open-mode=append" in script
    # USR1, not TERM. --signal only ever fires before the *time limit*, and
    # both preemption and `scancel` send TERM -- so asking for TERM here would
    # make all three situations arrive as one indistinguishable signal and the
    # sidecar would requeue runs the user had just cancelled.
    assert "#SBATCH --signal=B:USR1@90" in script
    assert "B:TERM" not in script
    assert f"#SBATCH --output={RUN_DIR}/slurm-%j.out" in script
    assert f"#SBATCH --chdir={RUN_DIR}" in script


def test_render_uses_the_name_when_given_and_sanitises_whitespace():
    assert "#SBATCH --job-name=sweep_lr3" in render_sbatch(make_spec(name="sweep lr3"))


def test_render_omits_account_and_partition_when_unset():
    script = render_sbatch(make_spec())
    assert "--account" not in script
    assert "--partition" not in script


def test_render_includes_account_and_partition_when_set():
    script = render_sbatch(make_spec(), account="mlopt", partition="gpu-a40")
    assert "#SBATCH --account=mlopt" in script
    assert "#SBATCH --partition=gpu-a40" in script


def test_spec_account_and_partition_beat_backend_defaults():
    """A per-run choice outranks a config-file default."""
    script = render_sbatch(
        make_spec(account="other-acct", partition="ckpt"),
        account="mlopt",
        partition="gpu-a40",
    )
    assert "#SBATCH --account=other-acct" in script
    assert "#SBATCH --partition=ckpt" in script
    # The backend's defaults must not appear at all -- checked as directives,
    # because "mlopt" also occurs inside the run directory path.
    assert "--account=mlopt" not in script
    assert "--partition=gpu-a40" not in script


def test_gres_is_rendered_when_set_and_absent_when_not():
    """The one Slurm flag nearly every relay job needs gets its own field."""
    assert "#SBATCH --gres=gpu:a40:2" in render_sbatch(make_spec(gres="gpu:a40:2"))
    assert "--gres" not in render_sbatch(make_spec(gres=None))


def test_sbatch_extra_entries_are_rendered_verbatim_and_in_order():
    script = render_sbatch(
        make_spec(
            sbatch_extra=[
                "--mem=64G",
                "--cpus-per-task=8",
                # A value with a space and a shell metacharacter: it is the
                # user's own sbatch syntax and must survive untouched.
                '--constraint "a40|l40"',
            ]
        )
    )
    lines = script.splitlines()
    rendered = [line for line in lines if line.startswith("#SBATCH ")]

    assert "#SBATCH --mem=64G" in rendered
    assert "#SBATCH --cpus-per-task=8" in rendered
    assert '#SBATCH --constraint "a40|l40"' in rendered

    # In the order the user wrote them, and after every directive relay
    # generates itself -- so a reader sees relay's own lines first, and so a
    # later entry can never be shadowed by a relay line below it.
    assert rendered[-3:] == [
        "#SBATCH --mem=64G",
        "#SBATCH --cpus-per-task=8",
        '#SBATCH --constraint "a40|l40"',
    ]


@pytest.mark.parametrize("entry", ["--output=x", "--time=1:00"])
def test_render_refuses_an_sbatch_extra_entry_relay_owns(entry):
    """The CLI validates first; this is the backstop for callers that skip it.

    Exhaustive coverage of *which* flags are refused lives in the CLI tests.
    What matters here is that nothing bad can reach a file on shared storage
    even when `render_sbatch` is called directly.
    """
    with pytest.raises(SbatchExtraError):
        render_sbatch(make_spec(sbatch_extra=[entry]))


# --------------------------------------------------------------------------
# setup: the user's shell lines, fenced off with set -e
# --------------------------------------------------------------------------


def test_setup_lines_run_between_the_directives_and_the_sidecar():
    script = render_sbatch(
        make_spec(setup=["module load cuda/12.4", "conda activate rl"])
    )
    lines = script.splitlines()

    last_directive = max(i for i, line in enumerate(lines) if line.startswith("#SBATCH"))
    exec_line = next(i for i, line in enumerate(lines) if line.startswith("exec "))

    # `set -e` so a failed `conda activate` ends the job instead of training in
    # the wrong environment; `set +e` so the sidecar's own exit-code handling
    # is untouched by it.
    block = [
        i
        for i, line in enumerate(lines)
        if line in ("set -e", "set +e", "module load cuda/12.4", "conda activate rl")
    ]
    assert [lines[i] for i in block] == [
        "set -e",
        "module load cuda/12.4",
        "conda activate rl",
        "set +e",
    ]
    assert last_directive < block[0]
    assert block[-1] < exec_line


def test_no_setup_emits_no_set_e_at_all():
    """An empty list is not an empty block -- it is no block."""
    script = render_sbatch(make_spec(setup=[]))
    assert "set -e" not in script
    assert "set +e" not in script


def test_setup_lines_are_not_quoted_or_reformatted():
    """They are shell, not arguments. Relay does not parse them."""
    line = 'export PYTHONPATH="$PWD/src:$PYTHONPATH"'
    assert line in render_sbatch(make_spec(setup=[line])).splitlines()


def test_env_is_exported_and_quoted():
    script = render_sbatch(make_spec(env={"WANDB_MODE": "offline", "NOTE": "a b"}))
    assert "export WANDB_MODE=offline" in script
    # A value with a space must reach the process as one word.
    assert "export NOTE='a b'" in script


def test_exec_line_quotes_arguments_containing_spaces():
    spec = make_spec(command=["python", "train.py", "--tag", "two words"], grace_seconds=45)
    script = render_sbatch(spec, remote_python="/sw/miniconda/bin/python")
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))

    # Re-split it the way bash on the compute node would, and check the sidecar
    # receives exactly the argv we meant.
    words = shlex.split(exec_line)
    assert words[0] == "exec"
    assert words[1] == "/sw/miniconda/bin/python"
    assert words[2] == f"{RUN_DIR}/{SIDECAR_FILENAME}"
    assert words[3:9] == ["--run", "vr_s1_7f3a", "--dir", RUN_DIR, "--grace", "45"]
    assert words[-5:] == ["--", "python", "train.py", "--tag", "two words"]


def test_grace_is_the_same_number_in_the_directive_and_the_sidecar_flag():
    """These two must not drift: Slurm's warning window is the sidecar's budget."""
    script = render_sbatch(make_spec(grace_seconds=120))
    assert "#SBATCH --signal=B:USR1@120" in script
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    words = shlex.split(exec_line)
    assert words[words.index("--grace") + 1] == "120"


def test_preempt_grace_does_not_reach_the_exec_line():
    """--grace is the *time limit* budget; preempt_grace travels in sidecar.json.

    They are different windows with different owners -- Slurm's --signal
    warning versus the partition's GraceTime/KillWait -- so putting the wrong
    one on the command line would have the sidecar plan for time it does not
    have.
    """
    script = render_sbatch(make_spec(grace_seconds=60, preempt_grace=10))
    exec_line = next(line for line in script.splitlines() if line.startswith("exec "))
    words = shlex.split(exec_line)
    assert words[words.index("--grace") + 1] == "60"
    assert "--preempt-grace" not in exec_line


def test_render_is_deterministic():
    """Same inputs, byte-identical script -- so a human can diff two submits."""
    spec = make_spec(
        env={"B": "2", "A": "1"},
        sbatch_extra=["--mem=8G"],
        setup=["module load cuda"],
    )
    assert render_sbatch(spec) == render_sbatch(spec)


# The exact script relay renders for a spec that uses every knob at once.
# `RUN_DIR` is substituted in rather than interpolated, so the literal below
# reads like the file a user would `cat` on the cluster.
SAMPLE_SCRIPT = """\
#!/bin/bash
#
# Generated by relay for run vr_s1_7f3a. Safe to read, edit and
# resubmit by hand; relay rewrites it on the next submit.

#SBATCH --job-name=vr_s1_7f3a
#SBATCH --account=mlopt
#SBATCH --partition=ckpt-all
#SBATCH --gres=gpu:a40:1
#SBATCH --chdir={run_dir}
#SBATCH --output={run_dir}/slurm-%j.out
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint "a40|l40"

# Environment from the job spec.
export WANDB_MODE=offline

# Setup from the job spec, run verbatim before training.
set -e
module load cuda/12.4
source ~/envs/rl/bin/activate
set +e

# The sidecar wraps the training command: it appends events to
# events.jsonl in this directory, and on a signal it gives the
# training script a chance to checkpoint before the job dies.
# Its sidecar.json here says whether to requeue after a SIGTERM.
{exec_line}
"""


def test_the_whole_rendered_script_for_a_realistic_spec():
    """One golden script, so the layout itself is part of the contract.

    Every other render test checks one line. This one checks the order they
    come in: relay's own directives, then the user's passthrough directives,
    then the environment, then the `set -e`-fenced setup lines, then the
    sidecar. A change to any of that shows up here as a diff a human can read.
    """
    spec = make_spec(
        time_limit="12:00:00",
        grace_seconds=90,
        gres="gpu:a40:1",
        sbatch_extra=["--mem=64G", "--cpus-per-task=8", '--constraint "a40|l40"'],
        setup=["module load cuda/12.4", "source ~/envs/rl/bin/activate"],
        env={"WANDB_MODE": "offline"},
    )
    script = render_sbatch(
        spec,
        remote_python="/sw/py/bin/python",
        account="mlopt",
        partition="ckpt-all",
    )
    exec_line = (
        f"exec /sw/py/bin/python {RUN_DIR}/sidecar.py "
        f"--run vr_s1_7f3a --dir {RUN_DIR} --grace 90 --heartbeat 30 "
        f"-- python train.py --lr 3e-4"
    )
    assert script == SAMPLE_SCRIPT.format(run_dir=RUN_DIR, exec_line=exec_line)


# --------------------------------------------------------------------------
# The time limit: always present, never guessed
# --------------------------------------------------------------------------


def test_time_limit_comes_from_the_spec():
    script = render_sbatch(make_spec(time_limit="12:00:00"))
    assert "#SBATCH --time=12:00:00" in script
    assert script.count("#SBATCH --time=") == 1


def test_missing_time_limit_is_refused_with_an_actionable_message():
    """A partition's silent DefaultTime=01:00:00 must never stand in for one."""
    with pytest.raises(MissingTimeLimit) as excinfo:
        render_sbatch(make_spec(time_limit=None))

    message = str(excinfo.value)
    assert "--time" in message
    assert "slurm.time" in message
    # A SlurmError subclass, so existing `except SlurmError` handlers still
    # catch it -- the CLI just gets to give it exit code 2 instead of 1.
    assert isinstance(excinfo.value, SlurmError)


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_blank_time_limit_counts_as_no_time_limit(blank):
    with pytest.raises(MissingTimeLimit):
        resolve_time_limit(make_spec(time_limit=blank))


def test_resolve_time_limit_has_exactly_one_source():
    """`spec.time_limit`, or nothing.

    The backend used to merge a `slurm.sbatch_defaults` dict on top of the
    spec, which meant two places decided the same question and a script could
    render two `#SBATCH --time=` lines whose winner depended on dict order.
    Now the CLI resolves `--time` against `slurm.time` once, and the backend
    renders what it is given. This test is the guard on that: a time limit
    reaches the script through the JobSpec or not at all.
    """
    assert resolve_time_limit(make_spec(time_limit="12:00:00")) == "12:00:00"
    # Surrounding whitespace is trimmed, since it would break the directive.
    assert resolve_time_limit(make_spec(time_limit=" 04:00:00 ")) == "04:00:00"
    with pytest.raises(MissingTimeLimit):
        resolve_time_limit(make_spec(time_limit=None))

    # Nothing else on the spec can supply one -- in particular `sbatch_extra`,
    # which `validate_sbatch_extra` refuses to let carry a --time at all.
    with pytest.raises(MissingTimeLimit):
        resolve_time_limit(make_spec(time_limit=None, sbatch_extra=["--mem=8G"]))


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------

SUBMIT_OK = (0, b"Submitted batch job 4242\n", b"")


TAR_OK = (0, b"", b"")

# What `scontrol show partition ckpt-all` looks like on Klone, trimmed to the
# tokens relay reads. Preemptible, PreemptMode=REQUEUE, GraceTime=0 (so the
# cluster-wide KillWait applies), no MaxTime but a one-hour DefaultTime.
CKPT_ALL_SCONTROL = b"""PartitionName=ckpt-all
   AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL
   AllocNodes=ALL Default=NO QoS=N/A
   DefaultTime=01:00:00 DisableRootJobs=no ExclusiveUser=no GraceTime=0 Hidden=NO
   MaxNodes=UNLIMITED MaxTime=UNLIMITED MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED
   Nodes=g[3001-3120],n[3001-3400]
   PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
   OverTimeLimit=NONE PreemptMode=REQUEUE
   State=UP TotalCPUs=12800 TotalNodes=520 SelectTypeParameters=NONE
   JobDefaults=(null)
   DefMemPerNode=UNLIMITED MaxMemPerNode=UNLIMITED
   TRES=cpu=12800,mem=100000G,node=520,billing=12800
"""


def test_submit_makes_two_round_trips_one_tar_then_sbatch():
    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    spec = make_spec()

    job_id = backend.submit(spec)

    assert job_id == "4242"
    assert runner.count == 2, "submit is one tar stream plus one sbatch"
    assert_alias_is_bare(runner)

    # 1: make the directory and unpack all three files into it, in one
    # command. This replaced two separate `cat >` trips, one per file; the
    # third file (sidecar.json) would have made it three.
    write = remote_words(runner.calls[0])
    assert write[:2] == ["sh", "-c"]
    assert f"mkdir -p {RUN_DIR}" in write[2]
    assert f"tar -x -C {RUN_DIR}" in write[2]

    # 2: sbatch is pointed at the real file on shared storage, never fed the
    # script on stdin -- the user has to be able to read what was submitted.
    assert remote_words(runner.calls[1]) == ["sbatch", f"{RUN_DIR}/{SBATCH_FILENAME}"]
    assert runner.calls[1]["input"] is None


def test_the_tar_stream_carries_exactly_the_files_the_job_needs():
    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    spec = make_spec()
    backend.submit(spec)

    files = tar_contents(runner.calls[0]["input"])
    expected = {SBATCH_FILENAME, SIDECAR_FILENAME, SIDECAR_CONFIG_FILENAME}
    # The checkpoint helper is optional, so it is in the stream exactly when
    # this installation has one. Listing it conditionally rather than always
    # keeps this test honest about what relay actually shipped.
    if default_ckpt_helper_path().is_file():
        expected.add(CKPT_HELPER_FILENAME)
    assert set(files) == expected

    # Names are relative, so `tar -x -C <run_dir>` lands them in the run
    # directory and nowhere else.
    for name in files:
        assert not name.startswith("/")
        assert not name.startswith("./")

    expected = render_sbatch(
        spec,
        remote_python=backend.remote_python,
        grace_seconds=backend.grace_seconds,
        account=backend.account,
        partition=backend.partition,
    )
    assert files[SBATCH_FILENAME] == expected.encode("utf-8")
    assert files[SIDECAR_FILENAME] == default_sidecar_path().read_bytes()


def test_the_tar_stream_is_byte_identical_between_identical_submits(monkeypatch):
    """Pinned mtime and ownership, so a resubmit can be diffed.

    The spec uses every rendering knob -- gres, passthrough directives, setup
    lines, environment -- because those are exactly the places where an
    accidental set or dict would reshuffle the script between two identical
    submits and turn the diff into noise.

    The clock is frozen because `sidecar.json` genuinely carries a timestamp:
    `first_attempt_start`, the moment of submit. Two submits a millisecond
    apart *should* differ there, so the only way to test the rest of the
    stream for reproducibility is to hold the one deliberately-varying value
    still. Replacing the module's `time` reference, rather than patching the
    real `time.time`, keeps the freeze inside this backend.
    """
    monkeypatch.setattr(slurm_mod, "time", types.SimpleNamespace(time=lambda: 1.0))

    spec = make_spec(
        gres="gpu:a40:1",
        sbatch_extra=["--mem=64G", "--cpus-per-task=8"],
        setup=["module load cuda/12.4", "conda activate rl"],
        env={"B": "2", "A": "1"},
    )
    streams = []
    for _ in range(2):
        backend, runner = make_backend(TAR_OK, SUBMIT_OK)
        backend.submit(spec)
        streams.append(runner.calls[0]["input"])
    assert streams[0] == streams[1]


def test_sidecar_json_carries_the_resolved_requeue_and_the_preempt_grace():
    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    before = time.time()
    backend.submit(make_spec(requeue_on_term="always", preempt_grace=10))

    config = json.loads(tar_contents(runner.calls[0]["input"])[SIDECAR_CONFIG_FILENAME])
    # "always"/"never", never "auto": the sidecar must not have to ask Slurm
    # anything from a compute node, least of all on the signal path.
    assert config["requeue_on_term"] == "always"
    assert config["preempt_grace"] == 10

    # Written once, at submit. Slurm requeues in place, so no later attempt
    # comes back through here to move it.
    assert isinstance(config["first_attempt_start"], float)
    assert before <= config["first_attempt_start"] <= time.time()

    assert config["resume"] is None
    assert set(config) == {
        "requeue_on_term",
        "preempt_grace",
        "first_attempt_start",
        "resume",
    }


def test_sidecar_json_passes_never_through_without_asking_slurm():
    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    backend.submit(make_spec(requeue_on_term="never", preempt_grace=30))

    config = json.loads(tar_contents(runner.calls[0]["input"])[SIDECAR_CONFIG_FILENAME])
    assert config["requeue_on_term"] == "never"
    assert config["preempt_grace"] == 30
    assert runner.count == 2, "a configured value needs no scontrol lookup"


def test_sidecar_json_carries_the_resume_pair_when_the_spec_has_one():
    """Both halves travel together, under the names the sidecar reads."""
    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    backend.submit(
        make_spec(
            resume_glob="checkpoints/**/*.pt",
            resume_arg="--resume-from {path}",
        )
    )

    config = json.loads(tar_contents(runner.calls[0]["input"])[SIDECAR_CONFIG_FILENAME])
    assert config["resume"] == {
        "checkpoint_glob": "checkpoints/**/*.pt",
        "arg": "--resume-from {path}",
    }
    assert runner.count == 2, "resume settings cost no extra round trip"


@pytest.mark.parametrize(
    "resume_glob, resume_arg",
    [
        ("checkpoints/*.pt", None),
        (None, "--resume-from {path}"),
    ],
)
def test_half_a_resume_spec_is_no_resume_spec(resume_glob, resume_arg):
    """A glob with no argument has nothing to say about the file it found."""
    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    backend.submit(make_spec(resume_glob=resume_glob, resume_arg=resume_arg))

    config = json.loads(tar_contents(runner.calls[0]["input"])[SIDECAR_CONFIG_FILENAME])
    assert config["resume"] is None


def test_submit_ships_the_checkpoint_helper_in_the_same_tar_stream():
    """`relay_ckpt.py` rides along with the sidecar, costing no extra trip.

    Skipped when the installation has no helper: it is optional, and a relay
    without one must still be able to submit.
    """
    source = default_ckpt_helper_path()
    if not source.is_file():
        pytest.skip(f"{source.name} is not part of this installation yet")

    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    backend.submit(make_spec())

    files = tar_contents(runner.calls[0]["input"])
    assert files[CKPT_HELPER_FILENAME] == source.read_bytes()
    assert runner.count == 2, "a fourth file is still one tar stream, not a fourth trip"


def test_submit_ships_the_real_installed_sidecar():
    backend, runner = make_backend(TAR_OK, SUBMIT_OK)
    backend.submit(make_spec())
    files = tar_contents(runner.calls[0]["input"])
    assert files[SIDECAR_FILENAME] == default_sidecar_path().read_bytes()


def test_submit_passes_backend_config_into_the_script():
    backend, runner = make_backend(
        TAR_OK,
        SUBMIT_OK,
        remote_python="/sw/py/bin/python",
        account="mlopt",
        partition="ckpt",
        grace_seconds=120,
    )
    # grace_seconds=0 means "use the backend default". The time limit has no
    # such fallback: the CLI put it on the spec or there is no job.
    backend.submit(make_spec(grace_seconds=0, time_limit="04:00:00"))

    script = tar_contents(runner.calls[0]["input"])[SBATCH_FILENAME].decode("utf-8")
    assert "#SBATCH --account=mlopt" in script
    assert "#SBATCH --partition=ckpt" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --signal=B:USR1@120" in script
    assert "/sw/py/bin/python" in script


def test_submit_without_a_time_limit_costs_no_round_trips():
    """The refusal happens on the laptop, before anything touches the network."""
    backend, runner = make_backend()
    with pytest.raises(MissingTimeLimit):
        backend.submit(make_spec(time_limit=None))
    assert runner.count == 0


def test_submit_raises_slurm_error_when_the_files_cannot_be_written():
    backend, _runner = make_backend(
        (1, b"", b"tar: /mmfs1/...: Cannot mkdir: Permission denied\n")
    )
    with pytest.raises(SlurmError) as excinfo:
        backend.submit(make_spec())
    assert "Permission denied" in str(excinfo.value)
    assert "remote_root" in str(excinfo.value)


def test_submit_raises_slurm_error_when_sbatch_rejects_the_script():
    backend, runner = make_backend(
        TAR_OK,
        (1, b"", b"sbatch: error: Invalid account or account/partition combination\n"),
    )
    with pytest.raises(SlurmError) as excinfo:
        backend.submit(make_spec())

    message = str(excinfo.value)
    assert "Invalid account" in message
    # The message tells the user where to go and read the script.
    assert f"{RUN_DIR}/{SBATCH_FILENAME}" in message


def test_submit_raises_slurm_error_on_unparseable_sbatch_output():
    backend, _runner = make_backend(TAR_OK, (0, b"ok, sent it\n", b""))
    with pytest.raises(SlurmError) as excinfo:
        backend.submit(make_spec())
    assert "ok, sent it" in str(excinfo.value)


# --------------------------------------------------------------------------
# Resolving requeue_on_term at submit time
# --------------------------------------------------------------------------


def test_auto_costs_one_scontrol_call_then_caches_it():
    """Three trips on the first submit to a partition, two on every one after.

    Partition configuration changes about once a year, so re-asking would be a
    Duo-protected round trip spent on an answer that cannot have changed.
    """
    backend, runner = make_backend(
        (0, CKPT_ALL_SCONTROL, b""),
        TAR_OK,
        SUBMIT_OK,
        TAR_OK,
        SUBMIT_OK,
        partition="ckpt-all",
    )

    backend.submit(make_spec(requeue_on_term="auto"))
    assert runner.count == 3, "first submit: scontrol, tar, sbatch"
    assert remote_words(runner.calls[0]) == [
        "scontrol",
        "show",
        "partition",
        "ckpt-all",
    ]

    backend.submit(make_spec(requeue_on_term="auto"))
    assert runner.count == 5, "second submit reuses the cached partition info"
    assert_alias_is_bare(runner)


def test_auto_resolves_to_never_on_a_requeueing_partition():
    """Klone's case: Slurm requeues preempted jobs itself, so the sidecar must not."""
    backend, runner = make_backend(
        (0, CKPT_ALL_SCONTROL, b""), TAR_OK, SUBMIT_OK, partition="ckpt-all"
    )
    backend.submit(make_spec(requeue_on_term="auto"))

    config = json.loads(tar_contents(runner.calls[1]["input"])[SIDECAR_CONFIG_FILENAME])
    assert config["requeue_on_term"] == "never"


def test_auto_resolves_to_always_when_the_partition_does_not_requeue():
    scontrol = b"PartitionName=gpu-a40\n   PreemptMode=OFF GraceTime=0 State=UP\n"
    backend, runner = make_backend(
        (0, scontrol, b""), TAR_OK, SUBMIT_OK, partition="gpu-a40"
    )
    backend.submit(make_spec(requeue_on_term="auto"))

    config = json.loads(tar_contents(runner.calls[1]["input"])[SIDECAR_CONFIG_FILENAME])
    assert config["requeue_on_term"] == "always"


def test_auto_resolves_to_always_when_scontrol_cannot_answer():
    """A double requeue is a nuisance; a job that never comes back is lost work."""
    backend, runner = make_backend(
        (1, b"", b"scontrol: error: Partition name ckpt-all not found\n"),
        TAR_OK,
        SUBMIT_OK,
        partition="ckpt-all",
    )
    backend.submit(make_spec(requeue_on_term="auto"))

    config = json.loads(tar_contents(runner.calls[1]["input"])[SIDECAR_CONFIG_FILENAME])
    assert config["requeue_on_term"] == "always"


def test_auto_with_no_partition_asks_for_the_default_one():
    """No --partition means the job lands on the cluster default, so that is
    the partition whose PreemptMode applies."""
    dump = b"""PartitionName=compute
   Default=NO PreemptMode=OFF State=UP

PartitionName=ckpt
   Default=YES PreemptMode=REQUEUE GraceTime=0 State=UP
"""
    backend, runner = make_backend((0, dump, b""), TAR_OK, SUBMIT_OK)
    backend.submit(make_spec(requeue_on_term="auto"))

    assert remote_words(runner.calls[0]) == ["scontrol", "show", "partition"]
    config = json.loads(tar_contents(runner.calls[1]["input"])[SIDECAR_CONFIG_FILENAME])
    assert config["requeue_on_term"] == "never"


def test_spec_partition_beats_the_backend_default_when_looking_it_up():
    backend, runner = make_backend(
        (0, CKPT_ALL_SCONTROL, b""), TAR_OK, SUBMIT_OK, partition="gpu-a40"
    )
    backend.submit(make_spec(requeue_on_term="auto", partition="ckpt-all"))
    assert remote_words(runner.calls[0])[-1] == "ckpt-all"


@pytest.mark.parametrize(
    ("configured", "preempt_mode", "expected"),
    [
        # Explicit answers pass straight through, whatever the partition says.
        ("always", "REQUEUE", "always"),
        ("always", "OFF", "always"),
        ("always", None, "always"),
        ("never", "REQUEUE", "never"),
        ("never", "OFF", "never"),
        ("never", None, "never"),
        # auto: Slurm already requeues, so we must not do it twice.
        ("auto", "REQUEUE", "never"),
        ("auto", "requeue", "never"),
        ("auto", "GANG,REQUEUE", "never"),
        # auto: nobody else will bring the job back, so we must.
        ("auto", "OFF", "always"),
        ("auto", "CANCEL", "always"),
        ("auto", "SUSPEND,GANG", "always"),
        # auto with no answer at all. The safe default is to requeue: a double
        # requeue costs one wasted attempt, a job that never returns costs
        # hours of training.
        ("auto", None, "always"),
        ("auto", "", "always"),
        # Unrecognised config values degrade to the auto rule rather than
        # crashing a submit.
        ("", "REQUEUE", "never"),
        ("nonsense", None, "always"),
    ],
)
def test_resolve_requeue_on_term_table(configured, preempt_mode, expected):
    assert resolve_requeue_on_term(configured, preempt_mode) == expected


def test_parse_scontrol_kv_on_a_realistic_partition_blob():
    info = parse_scontrol_kv(CKPT_ALL_SCONTROL.decode())

    assert info["PartitionName"] == "ckpt-all"
    assert info["PreemptMode"] == "REQUEUE"
    assert info["GraceTime"] == "0"
    assert info["MaxTime"] == "UNLIMITED"
    assert info["DefaultTime"] == "01:00:00"
    assert info["State"] == "UP"
    assert info["Default"] == "NO"
    # Values are left as strings: "UNLIMITED" is not an integer and "01:00:00"
    # is not a number of seconds.
    assert all(isinstance(value, str) for value in info.values())
    # Split on the *first* '=' only, so a value that is itself a key=value
    # list survives intact.
    assert info["TRES"] == "cpu=12800,mem=100000G,node=520,billing=12800"


def test_parse_scontrol_kv_ignores_tokens_without_an_equals():
    assert parse_scontrol_kv("PartitionName=ckpt garbage A=1") == {
        "PartitionName": "ckpt",
        "A": "1",
    }


def test_partition_info_is_empty_rather_than_raising_when_slurm_says_no():
    backend, _runner = make_backend((1, b"", b"scontrol: error: no such partition\n"))
    assert backend.partition_info("nope") == {}


def test_submit_raises_transport_error_when_ssh_cannot_connect():
    """ssh exits 255 for its own failures; that is routine, not a bad script."""
    backend, runner = make_backend(
        (255, b"", b"ssh: connect to host klone port 22: Connection refused\n")
    )
    with pytest.raises(TransportError) as excinfo:
        backend.submit(make_spec())

    assert "Connection refused" in str(excinfo.value)
    assert runner.count == 1, "a dead connection should not be retried in-place"


def test_submit_raises_transport_error_when_ssh_is_missing():
    backend, _runner = make_backend(FileNotFoundError("ssh"))
    with pytest.raises(TransportError) as excinfo:
        backend.submit(make_spec())
    assert "ssh" in str(excinfo.value)


def test_submit_raises_transport_error_on_timeout():
    backend, _runner = make_backend(subprocess.TimeoutExpired(cmd="ssh", timeout=60))
    with pytest.raises(TransportError) as excinfo:
        backend.submit(make_spec())
    assert "did not answer" in str(excinfo.value)


def test_constructing_with_a_user_at_host_is_refused():
    with pytest.raises(ValueError) as excinfo:
        SlurmBackend(ssh_alias="ahatim@klone.hyak.uw.edu", remote_root=REMOTE_ROOT)
    assert "~/.ssh/config" in str(excinfo.value)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def test_status_of_nothing_costs_no_round_trips():
    backend, runner = make_backend()
    assert backend.status([]) == {}
    assert runner.count == 0


def test_status_uses_one_squeue_call_for_every_live_job():
    backend, runner = make_backend(
        (0, b"101|RUNNING\n102|PENDING\n103|COMPLETING\n", b"")
    )
    assert backend.status(["101", "102", "103"]) == {
        "101": "running",
        "102": "queued",
        "103": "running",
    }
    assert runner.count == 1, "no sacct call is needed when squeue knows everything"
    assert remote_words(runner.calls[0]) == [
        "squeue",
        "--me",
        "--noheader",
        # The `|` must arrive as part of the format string, not as a pipe.
        "--format=%i|%T",
    ]
    assert_alias_is_bare(runner)


def test_status_falls_back_to_sacct_for_jobs_squeue_has_dropped():
    """squeue forgets a job the moment it finishes; sacct remembers."""
    backend, runner = make_backend(
        (0, b"101|RUNNING\n", b""),
        (0, b"102|COMPLETED\n103|FAILED\n", b""),
    )
    assert backend.status(["101", "102", "103"]) == {
        "101": "running",
        "102": "completed",
        "103": "failed",
    }
    assert runner.count == 2, "one squeue plus one batched sacct, whatever the job count"
    assert remote_words(runner.calls[1]) == [
        "sacct",
        "-n",
        "-P",
        "-X",
        "-j",
        "102,103",  # one call, comma-separated, not one call per job
        "--format=JobID,State",
    ]


def test_status_reports_unknown_for_ids_neither_command_knows():
    backend, _runner = make_backend((0, b"", b""), (0, b"", b""))
    assert backend.status(["999"]) == {"999": "unknown"}


def test_status_covers_every_requested_id_even_when_duplicated():
    backend, runner = make_backend((0, b"101|RUNNING\n", b""))
    assert backend.status(["101", "101"]) == {"101": "running"}
    assert runner.count == 1


def test_status_returns_only_the_agreed_vocabulary():
    backend, _runner = make_backend(
        (0, b"101|RUNNING\n102|PREEMPTED\n", b""),
        (0, b"103|CANCELLED by 12345\n", b""),
    )
    for value in backend.status(["101", "102", "103", "104"]).values():
        assert value in STATUSES


def test_status_ignores_sacct_sub_job_rows():
    """`.batch` and `.extern` rows must not overwrite the allocation's state."""
    backend, _runner = make_backend(
        (0, b"", b""),
        (
            0,
            b"55|COMPLETED\n55.batch|FAILED\n55.extern|COMPLETED\n55.0|CANCELLED\n",
            b"",
        ),
    )
    assert backend.status(["55"]) == {"55": "completed"}


def test_status_survives_sacct_failing():
    """An ID sacct has never heard of makes it exit nonzero. That is not an error."""
    backend, _runner = make_backend(
        (0, b"", b""),
        (1, b"", b"sacct: error: Invalid job id specified\n"),
    )
    assert backend.status(["77"]) == {"77": "unknown"}


def test_status_raises_transport_error_when_squeue_cannot_be_reached():
    backend, _runner = make_backend((255, b"", b"ssh: Connection timed out\n"))
    with pytest.raises(TransportError):
        backend.status(["101"])


def test_status_raises_slurm_error_when_squeue_itself_fails():
    backend, _runner = make_backend((1, b"", b"squeue: error: Unable to contact slurm\n"))
    with pytest.raises(SlurmError) as excinfo:
        backend.status(["101"])
    assert "Unable to contact slurm" in str(excinfo.value)


@pytest.mark.parametrize(
    ("slurm_state", "expected"),
    [
        ("PENDING", "queued"),
        ("CONFIGURING", "queued"),
        ("REQUEUED", "queued"),
        ("RESV_DEL_HOLD", "queued"),
        # With --requeue a preempted job goes back to pending and runs again.
        ("PREEMPTED", "queued"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("COMPLETED", "completed"),
        ("FAILED", "failed"),
        ("TIMEOUT", "failed"),
        ("OUT_OF_MEMORY", "failed"),
        ("NODE_FAIL", "failed"),
        ("BOOT_FAIL", "failed"),
        ("DEADLINE", "failed"),
        ("CANCELLED", "cancelled"),
        ("CANCELLED by 12345", "cancelled"),
        ("CANCELLED+", "cancelled"),
        ("cancelled", "cancelled"),
        ("SOMETHING_NEW", "unknown"),
        ("", "unknown"),
    ],
)
def test_state_mapping_table(slurm_state, expected):
    assert map_slurm_state(slurm_state) == expected


def test_state_mapping_is_applied_end_to_end():
    backend, _runner = make_backend((0, b"", b""), (0, b"9|CANCELLED by 12345\n", b""))
    assert backend.status(["9"]) == {"9": "cancelled"}


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------


def test_cancel_sends_scancel():
    backend, runner = make_backend((0, b"", b""))
    backend.cancel("4242")
    assert runner.count == 1
    assert remote_words(runner.calls[0]) == ["scancel", "4242"]
    assert_alias_is_bare(runner)


def test_cancel_with_a_run_dir_touches_the_marker_before_scancel_in_one_trip():
    """The marker has to exist before the signal lands, and that ordering must
    not cost a second Duo-protected round trip. One remote shell, two commands,
    sequenced."""
    backend, runner = make_backend((0, b"", b""))
    backend.cancel("4242", run_dir=RUN_DIR)

    assert runner.count == 1, "ordering must not cost an extra round trip"
    words = remote_words(runner.calls[0])
    assert words[:2] == ["sh", "-c"]
    script = words[2]

    marker = f"{RUN_DIR}/{CANCEL_FILENAME}"
    assert f"touch {marker}" in script
    assert "scancel 4242" in script
    assert script.index("touch") < script.index("scancel"), (
        "the sidecar must find CANCEL_REQUESTED already there when SIGTERM arrives"
    )
    # ';' not '&&': if the touch fails, the cancel must still go out. A job
    # that keeps running because relay could not write a marker file is a much
    # worse outcome than one that gets requeued once.
    assert "&&" not in script
    assert_alias_is_bare(runner)


def test_cancel_with_a_run_dir_quotes_a_path_with_a_space():
    backend, runner = make_backend((0, b"", b""))
    backend.cancel("4242", run_dir="/mmfs1/my runs/vr_s1")
    assert "'/mmfs1/my runs/vr_s1/CANCEL_REQUESTED'" in remote_words(runner.calls[0])[2]


def test_cancel_with_a_run_dir_still_treats_a_finished_job_as_a_no_op():
    """The shell's exit status is scancel's, so the existing handling still reads."""
    backend, _runner = make_backend(
        (1, b"", b"scancel: error: Kill job error on job id 4242: Job already completed\n")
    )
    backend.cancel("4242", run_dir=RUN_DIR)  # must not raise


@pytest.mark.parametrize(
    "stderr",
    [
        b"scancel: error: Kill job error on job id 4242: Invalid job id specified\n",
        b"scancel: error: Kill job error on job id 4242: Job already completed\n",
    ],
)
def test_cancel_of_a_finished_job_is_a_no_op(stderr):
    """The daemon acts on a snapshot; losing the race is not a failure."""
    backend, _runner = make_backend((1, b"", stderr))
    backend.cancel("4242")  # must not raise


def test_cancel_raises_on_a_real_failure():
    backend, _runner = make_backend(
        (1, b"", b"scancel: error: Kill job error on job id 4242: Access/permission denied\n")
    )
    with pytest.raises(SlurmError) as excinfo:
        backend.cancel("4242")
    assert "permission denied" in str(excinfo.value).lower()


def test_cancel_raises_transport_error_when_ssh_fails():
    backend, _runner = make_backend((255, b"", b"ssh: Connection closed\n"))
    with pytest.raises(TransportError):
        backend.cancel("4242")


# --------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------
#
# The parser is a pure function fed fixture files, because usage accounting is
# the part of relay most likely to be quietly wrong -- a GPU count read twice
# doubles every figure the user sees and nothing crashes.


def test_usage_of_nothing_costs_no_round_trips():
    backend, runner = make_backend()
    assert backend.usage([]) == []
    assert runner.count == 0


def test_usage_asks_sacct_once_for_every_job():
    backend, runner = make_backend((0, fixture("sacct_basic.txt").encode(), b""))

    rows = backend.usage(["12345601", "12345602", "12345603"])

    assert runner.count == 1, "one sacct call covers every job"
    assert remote_words(runner.calls[0]) == [
        "env",
        # sacct honours this, and it is the only clean way out of the default
        # `YYYY-MM-DDTHH:MM:SS` format, which carries no timezone at all.
        "SLURM_TIME_FORMAT=%s",
        "sacct",
        "-j",
        "12345601,12345602,12345603",  # comma-separated, not one call per job
        "-X",  # allocations only; step rows would be counted as extra attempts
        "--duplicates",  # every requeue attempt, not just the latest
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,Partition,Start,End,ElapsedRaw,AllocTRES",
    ]
    assert_alias_is_bare(runner)
    assert [row.job_id for row in rows] == ["12345601", "12345602", "12345603"]


def test_usage_deduplicates_ids_but_keeps_their_order():
    backend, runner = make_backend((0, b"", b""))
    backend.usage(["101", "102", "101"])
    assert remote_words(runner.calls[0])[4] == "101,102"


def test_usage_survives_sacct_failing_on_an_aged_out_id():
    """sacct complains about IDs it has never heard of. That is not an error for
    a daemon that outlives the accounting database's retention."""
    backend, _runner = make_backend((1, b"", b"sacct: error: Invalid job id\n"))
    assert backend.usage(["999"]) == []


# -- parse_sacct_usage against the fixtures --------------------------------


def test_parse_basic_fixture():
    rows = parse_sacct_usage(fixture("sacct_basic.txt"))
    assert len(rows) == 3

    done = rows[0]
    assert done.job_id == "12345601"
    assert done.state == "COMPLETED"
    assert done.partition == "ckpt-all"
    # Epoch seconds in, unambiguous UTC out.
    assert done.start == "2025-09-16T18:02:09Z"
    assert done.end == "2025-09-16T22:02:09Z"
    assert done.elapsed_s == 14400
    assert done.gpus == 2
    assert done.cpus == 8

    cpu_only = rows[1]
    assert cpu_only.gpus == 0, "no gres/gpu key means zero GPUs, not an error"
    assert cpu_only.cpus == 4

    running = rows[2]
    assert running.state == "RUNNING"
    assert running.end is None, "an attempt still running has no end time"
    assert running.elapsed_s == 3600


def test_parse_skips_the_synthetic_provenance_header():
    text = fixture("sacct_basic.txt")
    assert text.lstrip().startswith("# SYNTHETIC"), (
        "each fixture must say it is hand-written and must be replaced with "
        "real Klone output"
    )
    assert all(not row.job_id.startswith("#") for row in parse_sacct_usage(text))


@pytest.mark.parametrize(
    "name",
    ["sacct_basic.txt", "sacct_requeued.txt", "sacct_typed_gpu.txt", "sacct_no_gpu.txt"],
)
def test_every_fixture_carries_the_synthetic_warning(name):
    assert fixture(name).lstrip().startswith("# SYNTHETIC")


def test_parse_real_klone_output():
    """The one fixture that is not synthetic: captured on klone-login03.

    Two things about real Klone output that a hand-written fixture could have
    got wrong: the typed GPU key comes *before* the untyped one
    (`gres/gpu:l40s=1,gres/gpu=1`), and the partition is a GPU partition
    name rather than a ckpt one. The GPU count must be 1, not 2 -- summing
    the two keys is the exact bug the spec warns about.
    """
    text = fixture("sacct_klone_real.txt")
    assert text.lstrip().startswith("# REAL")

    rows = parse_sacct_usage(text)

    assert [row.job_id for row in rows] == ["40268341", "40379788", "40381592"]
    assert all(row.state == "COMPLETED" for row in rows)
    assert all(row.partition == "gpu-l40s" for row in rows)
    assert [row.gpus for row in rows] == [1, 1, 1]
    assert [row.cpus for row in rows] == [8, 8, 2]
    assert [row.elapsed_s for row in rows] == [2387, 264, 32]
    # Epoch 1789691379 is 2026-09-18T00:29:39Z; the `%s` time format is what
    # makes this a UTC instant rather than a zoneless cluster-local string.
    assert rows[0].start == "2026-09-18T00:29:39Z"
    assert rows[0].end == "2026-09-18T01:09:26Z"


def test_parse_requeued_fixture_keeps_every_attempt():
    rows = parse_sacct_usage(fixture("sacct_requeued.txt"))

    # Three attempts under one job ID -- what `--duplicates` is for. Without
    # it sacct returns only the last, and two preemptions' GPU-hours vanish.
    attempts = [row for row in rows if row.job_id == "12345700"]
    assert len(attempts) == 3
    assert [row.state for row in attempts] == ["PREEMPTED", "PREEMPTED", "RUNNING"]

    # Attempts share a job ID but never a start time, which is why the store
    # keys on (job_id, start).
    starts = [row.start for row in attempts]
    assert len(set(starts)) == 3
    assert starts[0] == "2025-09-16T17:50:00Z"

    assert attempts[2].end is None, "the last attempt is still running"
    assert sum(row.elapsed_s for row in attempts) == 5400


def test_parse_drops_step_rows():
    """`.batch` and `.extern` rows would be counted as extra attempts."""
    rows = parse_sacct_usage(fixture("sacct_requeued.txt"))
    assert all("." not in row.job_id for row in rows)


def test_parse_drops_an_attempt_that_never_started():
    """No Start means no resources consumed, and UsageRow.start is never None."""
    rows = parse_sacct_usage(fixture("sacct_requeued.txt"))
    assert all(row.job_id != "12345701" for row in rows)
    assert all(row.start for row in rows)


def test_parse_typed_gpu_fixture_never_sums_typed_and_untyped():
    rows = parse_sacct_usage(fixture("sacct_typed_gpu.txt"))
    assert len(rows) == 3
    # All three allocations are 2 GPUs. Adding the typed and untyped keys
    # together would report 4 for the middle one and double every GPU-hour.
    assert [row.gpus for row in rows] == [2, 2, 2]


def test_parse_no_gpu_fixture():
    rows = parse_sacct_usage(fixture("sacct_no_gpu.txt"))
    assert [row.gpus for row in rows] == [0, 0, 0]
    assert [row.cpus for row in rows] == [4, 2, 1]
    # sacct spells a cancellation "CANCELLED by <uid>" -- the state is the
    # first word, and the uid is not part of it.
    assert rows[1].state == "CANCELLED"
    # A timestamp in sacct's default format has no timezone on it, so relay
    # passes it through unchanged rather than inventing one.
    assert rows[2].start == "2026-09-16T11:22:09"
    assert not rows[2].start.endswith("Z")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CANCELLED by 123", "CANCELLED"),
        ("CANCELLED+", "CANCELLED"),
        ("cancelled by 1", "CANCELLED"),
        ("PREEMPTED", "PREEMPTED"),
        ("OUT_OF_ME+", "OUT_OF_ME"),
    ],
)
def test_state_is_the_first_word_upper_cased_without_its_plus(raw, expected):
    line = f"1|{raw}|ckpt|1758045729|1758049329|3600|cpu=1"
    assert parse_sacct_usage(line)[0].state == expected


@pytest.mark.parametrize("end", ["Unknown", "None", "", "   ", "N/A"])
def test_an_attempt_with_no_end_is_still_running(end):
    line = f"1|RUNNING|ckpt|1758045729|{end}|3600|cpu=1"
    assert parse_sacct_usage(line)[0].end is None


def test_parse_ignores_blank_and_malformed_lines():
    """One odd line must not cost the user every other row in the batch."""
    text = (
        "\n"
        "sacct: error: something went wrong\n"
        "1|COMPLETED|ckpt|1758045729|1758049329|3600|cpu=8,gres/gpu=1\n"
        "\n"
    )
    rows = parse_sacct_usage(text)
    assert len(rows) == 1
    assert rows[0].gpus == 1


def test_a_blank_elapsed_is_zero_not_a_crash():
    line = "1|RUNNING|ckpt|1758045729|Unknown||cpu=1"
    assert parse_sacct_usage(line)[0].elapsed_s == 0


def test_an_empty_partition_field_becomes_none():
    line = "1|COMPLETED||1758045729|1758049329|3600|cpu=1"
    assert parse_sacct_usage(line)[0].partition is None


# -- parse_alloc_tres ------------------------------------------------------


@pytest.mark.parametrize(
    ("tres", "expected"),
    [
        # Untyped.
        ("billing=80,cpu=8,gres/gpu=2,mem=64G,node=1", (8, 2)),
        # Typed only.
        ("billing=80,cpu=8,gres/gpu:a40=2,mem=64G,node=1", (8, 2)),
        # Both, describing the same GPUs: prefer untyped, never sum.
        ("cpu=8,gres/gpu=2,gres/gpu:a40=2,node=1", (8, 2)),
        ("cpu=8,gres/gpu:a40=2,gres/gpu=2,node=1", (8, 2)),
        # Two types on one allocation: take the first, still never sum.
        ("cpu=16,gres/gpu:l40s=2,gres/gpu:h200=1,node=1", (16, 2)),
        # No GPU key at all is a CPU job, not an error.
        ("billing=4,cpu=4,mem=16G,node=1", (4, 0)),
        # Nothing at all.
        ("", (0, 0)),
        ("   ", (0, 0)),
        # Values relay does not need are never parsed, so a unit suffix on
        # mem can never raise here.
        ("mem=187000M,cpu=40,node=1", (40, 0)),
        # Junk between the pairs is skipped rather than fatal.
        ("cpu=2,,nonsense,gres/gpu=1", (2, 1)),
        # A non-numeric value degrades to zero instead of raising.
        ("cpu=lots,gres/gpu=2", (0, 2)),
    ],
)
def test_parse_alloc_tres_table(tres, expected):
    assert parse_alloc_tres(tres) == expected


# --------------------------------------------------------------------------
# TransportError carries what the daemon needs to classify it
# --------------------------------------------------------------------------


def test_transport_error_exposes_the_returncode_and_stderr():
    """The daemon calls ssh.classify_failure(rc, stderr, master_alive) with
    these. Re-parsing them out of the message would be silly and fragile."""
    backend, _runner = make_backend(
        (255, b"", b"klone: Permission denied (publickey,keyboard-interactive).\n")
    )
    with pytest.raises(TransportError) as excinfo:
        backend.status(["101"])

    error = excinfo.value
    assert error.returncode == 255
    assert "Permission denied" in error.stderr
    assert (
        ssh_mod.classify_failure(error.returncode, error.stderr, False)
        == ssh_mod.AUTH_REQUIRED
    )


def test_transport_error_from_a_remote_read_failure_carries_its_status():
    backend, _runner = make_backend((1, b"", b"tail: cannot open: Permission denied\n"))
    with pytest.raises(TransportError) as excinfo:
        backend.read_bytes(f"{RUN_DIR}/events.jsonl", 0)
    assert excinfo.value.returncode == 1
    assert "Permission denied" in excinfo.value.stderr


@pytest.mark.parametrize(
    "boom",
    [FileNotFoundError("ssh"), subprocess.TimeoutExpired(cmd="ssh", timeout=60)],
)
def test_transport_error_has_no_returncode_when_ssh_never_finished(boom):
    """Nothing to classify: the daemon treats these as `unreachable`."""
    backend, _runner = make_backend(boom)
    with pytest.raises(TransportError) as excinfo:
        backend.status(["101"])
    assert excinfo.value.returncode is None
    assert (
        ssh_mod.classify_failure(255, excinfo.value.stderr, True)
        == ssh_mod.UNREACHABLE
    )


# --------------------------------------------------------------------------
# read_bytes
# --------------------------------------------------------------------------

EVENTS = f"{RUN_DIR}/events.jsonl"


def test_read_bytes_tails_from_offset_plus_one():
    """tail -c +N counts from byte 1, so zero-based offset N needs N+1."""
    backend, runner = make_backend((0, b'{"seq":3}\n', b""))

    data, new_offset = backend.read_bytes(EVENTS, 100)

    assert data == b'{"seq":3}\n'
    assert new_offset == 110
    assert runner.count == 1
    words = remote_words(runner.calls[0])
    assert words[:2] == ["sh", "-c"]
    assert f"tail -c +101 {EVENTS}" in words[2]
    assert f"[ -f {EVENTS} ]" in words[2]
    assert_alias_is_bare(runner)


def test_read_bytes_from_the_beginning_asks_for_byte_one():
    backend, runner = make_backend((0, b"hello", b""))
    assert backend.read_bytes(EVENTS, 0) == (b"hello", 5)
    assert "tail -c +1 " in remote_words(runner.calls[0])[2]


def test_read_bytes_with_nothing_new_returns_the_same_offset():
    backend, _runner = make_backend((0, b"", b""))
    assert backend.read_bytes(EVENTS, 4096) == (b"", 4096)


def test_read_bytes_is_binary_safe():
    """Raw bytes, including anything that might look like a sentinel string."""
    payload = b"\xff\xfe not utf-8 \x00 RELAY_ENOENT\n"
    backend, _runner = make_backend((0, payload, b""))
    data, new_offset = backend.read_bytes(EVENTS, 10)
    assert data == payload
    assert new_offset == 10 + len(payload)


def test_missing_file_raises_file_not_found_not_transport_error():
    """The daemon reads FileNotFoundError as "the job has not started yet"."""
    backend, _runner = make_backend((44, b"", b""))
    with pytest.raises(FileNotFoundError):
        backend.read_bytes(EVENTS, 0)


def test_read_bytes_raises_transport_error_when_ssh_fails():
    backend, _runner = make_backend((255, b"", b"ssh: Connection reset by peer\n"))
    with pytest.raises(TransportError):
        backend.read_bytes(EVENTS, 0)


def test_read_bytes_raises_transport_error_on_a_remote_failure():
    backend, _runner = make_backend((1, b"", b"tail: cannot open: Permission denied\n"))
    with pytest.raises(TransportError) as excinfo:
        backend.read_bytes(EVENTS, 0)
    assert "Permission denied" in str(excinfo.value)


def test_read_bytes_quotes_a_path_with_a_space():
    backend, runner = make_backend((0, b"x", b""))
    backend.read_bytes("/mmfs1/my runs/events.jsonl", 0)
    # The remote shell must see the path as one word after re-splitting.
    script = remote_words(runner.calls[0])[2]
    assert "'/mmfs1/my runs/events.jsonl'" in script


# --------------------------------------------------------------------------
# from_config
# --------------------------------------------------------------------------


def test_from_config_builds_a_backend():
    conf = Config(
        backend="slurm",
        ssh_alias=ALIAS,
        remote_root=REMOTE_ROOT,
        remote_python="/sw/py/bin/python",
        slurm=SlurmConfig(
            account="mlopt",
            partition="gpu-a40",
            time="04:00:00",
            grace_seconds=120,
        ),
    )
    runner = FakeRunner()
    backend = SlurmBackend.from_config(conf, runner=runner)

    assert backend.ssh_alias == ALIAS
    assert backend.remote_root == REMOTE_ROOT
    assert backend.remote_python == "/sw/py/bin/python"
    assert backend.account == "mlopt"
    assert backend.partition == "gpu-a40"
    assert backend.grace_seconds == 120
    assert backend.runner is runner
    assert backend.sidecar_path == default_sidecar_path()

    # The backend takes only the four cluster-shaped settings. The time limit
    # reaches it on the JobSpec the CLI builds, never by the backend reading
    # config for itself -- one source of truth per script.
    assert not hasattr(backend, "sbatch_defaults")


def test_from_config_refuses_a_config_missing_its_cluster_settings():
    with pytest.raises(SlurmError) as excinfo:
        SlurmBackend.from_config(Config(backend="slurm"))
    assert "ssh_alias" in str(excinfo.value)


def test_sidecar_path_can_be_overridden(tmp_path: Path):
    fake = tmp_path / "sidecar.py"
    fake.write_bytes(b"# not the real one\n")
    backend, runner = make_backend(
        TAR_OK, SUBMIT_OK, sidecar_path=fake
    )
    backend.submit(make_spec())
    assert tar_contents(runner.calls[0]["input"])[SIDECAR_FILENAME] == (
        b"# not the real one\n"
    )
