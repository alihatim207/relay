"""Tests for relay.cli.

Every test drives `cli.main(argv)` the way a shell would and looks at what came
out of stdout, stderr and the database. Nothing imports a command function and
calls it directly, because the interesting behaviour of a CLI -- exit codes,
error sentences, the `--json` / table split -- only exists at that boundary.

As in `test_config.py`, HOME and both XDG variables point at a temporary
directory. relay's config file *and* its SQLite database are addressed through
those variables, so without this every test would read and write the
developer's real runs.
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
from pathlib import Path

import pytest

from relay import cli
from relay import config as cfg
from relay import daemon as daemon_module
from relay import doctor as doctor_module
from relay.backends import base as backends_base
from relay.store import Store

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point HOME, XDG_CONFIG_HOME and XDG_DATA_HOME at a throwaway directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    # The debug escape hatch re-raises instead of printing a sentence, which
    # would turn every error-handling assertion here into a traceback.
    monkeypatch.delenv("RELAY_DEBUG", raising=False)
    return home


def write_config(text: str) -> Path:
    path = cfg.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def event(seq: int, type_: str, payload: dict, run_id: str = "r") -> dict:
    """One parsed event, in the shape `store.ingest` expects."""
    return {
        "v": 1,
        "run": run_id,
        "seq": seq,
        "ts": f"2026-09-18T12:00:{seq:02d}.000Z",
        "type": type_,
        "payload": payload,
    }


def seed_run(
    run_id: str,
    *,
    status: str = "running",
    stale: bool = False,
    metrics: dict | None = None,
    logs: tuple[tuple[str, str], ...] = (),
    job_id: str | None = "4242",
    backend: str = "local",
    run_dir: str = "/tmp/relay-test",
    spec: dict | None = None,
) -> None:
    """Put a run (and optionally some events) into the database by hand.

    This is what the daemon would have written; the CLI only reads.
    """
    with Store(cfg.db_path()) as store:
        store.create_run(
            run_id,
            backend=backend,
            name=run_id,
            job_id=job_id,
            run_dir=run_dir,
            status=status,
            spec=spec,
        )
        events = []
        seq = 0
        for stream, line in logs:
            seq += 1
            events.append(event(seq, "log", {"stream": stream, "line": line}, run_id))
        if metrics:
            seq += 1
            events.append(event(seq, "metric", {"step": 100, "values": metrics}, run_id))
        if events:
            store.ingest(run_id, f"{run_dir}/events.jsonl", events, 1000)
        if stale:
            store.set_stale(run_id)


def get_run(run_id: str) -> dict | None:
    with Store(cfg.db_path()) as store:
        return store.get_run(run_id)


class FakeBackend:
    """A backend that records what it was asked to do and never starts anything."""

    def __init__(self) -> None:
        self.specs: list = []
        self.cancelled: list[str] = []
        # Every cancel's full call, keyword arguments included, because the
        # run directory is the interesting half now: it is what tells the
        # sidecar a SIGTERM was a cancel rather than a preemption.
        self.cancel_calls: list[dict] = []
        # Set to an exception to make the next submit raise it, so a test can
        # drive the CLI's handling of a backend that refuses a job.
        self.submit_error: Exception | None = None

    def submit(self, spec) -> str:
        # Deliberately creates nothing: a fake that made directories would
        # happily try to mkdir /mmfs1 when the config says slurm.
        self.specs.append(spec)
        if self.submit_error is not None:
            raise self.submit_error
        return f"job{len(self.specs)}"

    def status(self, job_ids):
        return {job_id: "unknown" for job_id in job_ids}

    def cancel(self, job_id: str, run_dir: str | None = None) -> None:
        self.cancelled.append(job_id)
        self.cancel_calls.append({"job_id": job_id, "run_dir": run_dir})

    def usage(self, job_ids):
        return []

    def read_bytes(self, path: str, offset: int):
        raise FileNotFoundError(path)


@pytest.fixture
def fake_backend(monkeypatch):
    """Replace the backend factory the CLI uses, for both submit and cancel."""
    backend = FakeBackend()
    monkeypatch.setattr(daemon_module, "build_backend", lambda conf: backend)
    return backend


@pytest.fixture
def local_config(tmp_path):
    """A minimal `backend: local` config whose run directories are in tmp."""
    return write_config(
        f"""
        backend: local
        local_root: {tmp_path / "runs"}
        """
    )


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def test_init_writes_a_config_that_load_accepts(capsys):
    assert cli.main(["init"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert str(cfg.config_path()) in out
    assert "doctor" in out

    conf = cfg.load()
    assert conf.backend == "local"


def test_init_twice_refuses_then_force_overwrites(capsys):
    assert cli.main(["init"]) == cli.EXIT_OK
    cfg.config_path().write_text("backend: local\n# edited by hand\n", encoding="utf-8")
    capsys.readouterr()

    code = cli.main(["init"])
    err = capsys.readouterr().err
    assert code != cli.EXIT_OK
    assert "already exists" in err
    assert "--force" in err
    # The hand-edited file is untouched.
    assert "edited by hand" in cfg.config_path().read_text(encoding="utf-8")

    assert cli.main(["init", "--force"]) == cli.EXIT_OK
    assert "edited by hand" not in cfg.config_path().read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Which commands need a config
# --------------------------------------------------------------------------


def test_submit_without_config_exits_not_configured(capsys, tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    code = cli.main(["submit", str(script)])
    captured = capsys.readouterr()
    assert code == cli.EXIT_NOT_CONFIGURED
    assert "relay init" in captured.err
    assert captured.out == ""


def test_ls_needs_no_config_at_all(capsys):
    """Read commands go to the database, whose path comes from XDG, not config."""
    assert not cfg.config_path().exists()
    assert cli.main(["ls"]) == cli.EXIT_OK
    assert "last synced never" in capsys.readouterr().out


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


def test_submit_local_creates_a_queued_run(capsys, tmp_path, local_config):
    script = tmp_path / "train.py"
    script.write_text("print('hello from the job')\n", encoding="utf-8")

    code = cli.main(["submit", str(script)])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK

    with Store(cfg.db_path()) as store:
        runs = store.list_runs()
    assert len(runs) == 1
    run = runs[0]

    assert run["run_id"] in out
    assert run["run_id"].startswith("train_")
    # Status is scheduler-owned: the CLI writes "queued" and lets the daemon
    # promote it, even though a local job is already running.
    assert run["status"] == "queued"
    assert run["job_id"]
    assert run["backend"] == "local"
    assert Path(run["run_dir"]).is_dir()
    assert run["spec"]["command"] == ["python3", str(script)]


def test_submit_seeds_makes_one_run_per_seed(capsys, tmp_path, local_config, fake_backend):
    script = tmp_path / "train.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    code = cli.main(
        ["submit", str(script), "--name", "vr", "--seeds", "0-2", "--", "--lr", "3e-4"]
    )
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK

    specs = fake_backend.specs
    assert len(specs) == 3
    assert len({spec.run_id for spec in specs}) == 3

    for seed, spec in zip((0, 1, 2), specs):
        assert f"_s{seed}_" in spec.run_id
        assert spec.run_id in out
        # Passthrough verbatim and first, relay's own --seed appended after it.
        assert spec.command == ["python3", str(script), "--lr", "3e-4", "--seed", str(seed)]
        assert spec.name == "vr"

    with Store(cfg.db_path()) as store:
        rows = store.list_runs()
    assert len(rows) == 3
    assert {row["spec"]["command"][-1] for row in rows} == {"0", "1", "2"}


def test_submit_copies_grace_and_slurm_fields_into_the_spec(tmp_path, fake_backend):
    write_config(
        """
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/you/relay
        remote_python: /sw/python/bin/python3
        slurm:
          account: mlopt
          partition: gpu-a40
          grace_seconds: 90
          time: "04:00:00"
        """
    )

    assert cli.main(["submit", "train.py"]) == cli.EXIT_OK

    spec = fake_backend.specs[0]
    # The backend honours the spec over its own defaults, so the grace period
    # has to be copied in or a per-cluster setting would silently do nothing.
    assert spec.grace_seconds == 90
    assert spec.account == "mlopt"
    assert spec.partition == "gpu-a40"
    assert spec.time_limit == "04:00:00"
    # Nothing was asked for, so nothing is carried: these are lists now, and an
    # empty one renders no extra #SBATCH lines and no setup block.
    assert spec.sbatch_extra == []
    assert spec.setup == []
    # Remote paths are posix, and the remote interpreter comes from the config.
    assert spec.run_dir.startswith("/mmfs1/gscratch/mlopt/you/relay/")
    assert spec.command[0] == "/sw/python/bin/python3"


def test_submit_time_flag_and_signal_policy_reach_the_spec(fake_backend):
    """`--time` wins over the config, and the two signal settings come along."""
    write_config(
        """
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/you/relay
        requeue_on_term: always
        preempt_grace: 20
        slurm:
          account: mlopt
          partition: ckpt-all
          time: "02:00:00"
        """
    )

    assert cli.main(["submit", "train.py", "--time", "12:00:00"]) == cli.EXIT_OK

    spec = fake_backend.specs[0]
    assert spec.time_limit == "12:00:00"
    # Both of these belong to the sidecar's signal handling, and both are
    # top-level config keys rather than slurm ones.
    assert spec.preempt_grace == 20
    assert spec.requeue_on_term == "always"

    # The stored spec is what `relay show` and any later debugging read, so the
    # time limit has to survive into it too.
    with Store(cfg.db_path()) as store:
        stored = store.list_runs()[0]["spec"]
    assert stored["time_limit"] == "12:00:00"
    assert stored["preempt_grace"] == 20


def test_submit_falls_back_to_the_config_time(fake_backend):
    write_config(
        """
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/you/relay
        slurm:
          time: "04:00:00"
        """
    )

    assert cli.main(["submit", "train.py"]) == cli.EXIT_OK
    assert fake_backend.specs[0].time_limit == "04:00:00"
    # Defaults, not silence: the sidecar always gets a policy and a budget.
    assert fake_backend.specs[0].requeue_on_term == "auto"
    assert fake_backend.specs[0].preempt_grace == 10


def test_submit_with_no_time_limit_anywhere_is_a_usage_error(capsys, fake_backend):
    """The backend owns the rule; the CLI only has to map it to exit code 2.

    Relay refuses to guess a time limit because Klone's checkpoint partitions
    default to one hour, so a twelve-hour job would die at sixty minutes with
    no explanation.
    """
    from relay.backends.slurm import MissingTimeLimit

    write_config(
        """
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/you/relay
        """
    )
    fake_backend.submit_error = MissingTimeLimit(
        "Run train_ab12 has no time limit, and relay will not submit a job "
        "without one. Pass `--time 12:00:00` to `relay submit`."
    )

    code = cli.main(["submit", "train.py"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_USAGE
    assert "--time" in captured.err
    assert "Traceback" not in captured.err
    # Nothing is announced as submitted when nothing was.
    assert captured.out == ""
    assert fake_backend.specs[0].time_limit is None


def test_submit_non_python_script_runs_directly(tmp_path, local_config, fake_backend):
    assert cli.main(["submit", "./train.sh"]) == cli.EXIT_OK
    assert fake_backend.specs[0].command == ["./train.sh"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0-4", [0, 1, 2, 3, 4]),
        ("0,2,5", [0, 2, 5]),
        ("0-2,7", [0, 1, 2, 7]),
        ("3", [3]),
    ],
)
def test_parse_seeds(text, expected):
    assert cli.parse_seeds(text) == expected


def test_bad_seeds_is_a_usage_error(capsys, tmp_path, local_config, fake_backend):
    code = cli.main(["submit", "train.py", "--seeds", "5-1"])
    assert code == cli.EXIT_USAGE
    assert "backwards" in capsys.readouterr().err


# --------------------------------------------------------------------------
# submit -- setup lines and extra sbatch directives
# --------------------------------------------------------------------------
#
# Two lists that can come from the config or from the command line, with one
# rule: the flags REPLACE the config's list, they never merge with it. The
# tests below that need the new config keys are marked; everything else drives
# the flags only, so it holds whether or not config.py has landed.


def test_setup_flag_is_repeatable_and_keeps_its_order(local_config, fake_backend):
    code = cli.main(
        [
            "submit",
            "train.py",
            "--setup",
            "module load cuda",
            "--setup",
            "conda activate rl",
        ]
    )

    assert code == cli.EXIT_OK
    # Order matters: `conda activate` after `module load`, not the other way
    # round, and relay renders them exactly as given.
    assert fake_backend.specs[0].setup == ["module load cuda", "conda activate rl"]


def test_sbatch_extra_flag_is_repeatable(local_config, fake_backend):
    """Values starting with a dash need the `=` form, which the help says."""
    code = cli.main(
        [
            "submit",
            "train.py",
            "--sbatch-extra=--mem=64G",
            "--sbatch-extra=--cpus-per-task=8",
        ]
    )

    assert code == cli.EXIT_OK
    assert fake_backend.specs[0].sbatch_extra == ["--mem=64G", "--cpus-per-task=8"]


def test_setup_and_sbatch_extra_survive_into_the_stored_spec(local_config, fake_backend):
    """`relay show` reads the stored spec, so the new fields have to be in it."""
    code = cli.main(
        [
            "submit",
            "train.py",
            "--setup",
            "module load cuda",
            "--sbatch-extra=--mem=64G",
        ]
    )
    assert code == cli.EXIT_OK

    with Store(cfg.db_path()) as store:
        stored = store.list_runs()[0]["spec"]
    assert stored["setup"] == ["module load cuda"]
    assert stored["sbatch_extra"] == ["--mem=64G"]


def test_every_seed_gets_its_own_copy_of_the_lists(local_config, fake_backend):
    """One list object shared by five specs would alias five stored specs."""
    code = cli.main(
        ["submit", "train.py", "--seeds", "0-2", "--sbatch-extra=--mem=64G"]
    )
    assert code == cli.EXIT_OK

    specs = fake_backend.specs
    assert len(specs) == 3
    assert all(spec.sbatch_extra == ["--mem=64G"] for spec in specs)
    assert len({id(spec.sbatch_extra) for spec in specs}) == 3


def test_setup_flag_replaces_the_config_list(tmp_path, fake_backend):
    """Replace, not merge. Depends on config.py exposing `setup`."""
    write_config(
        f"""
        backend: local
        local_root: {tmp_path / "runs"}
        setup:
          - module load cuda
          - conda activate rl
        """
    )

    code = cli.main(["submit", "train.py", "--setup", "conda activate other"])

    assert code == cli.EXIT_OK
    # Not ["module load cuda", "conda activate rl", "conda activate other"]:
    # a user naming an environment means that environment and no other.
    assert fake_backend.specs[0].setup == ["conda activate other"]


def test_config_setup_is_used_when_the_flag_is_absent(tmp_path, fake_backend):
    """Depends on config.py exposing `setup`."""
    write_config(
        f"""
        backend: local
        local_root: {tmp_path / "runs"}
        setup:
          - module load cuda
          - conda activate rl
        """
    )

    assert cli.main(["submit", "train.py"]) == cli.EXIT_OK
    assert fake_backend.specs[0].setup == ["module load cuda", "conda activate rl"]


def test_sbatch_extra_flag_replaces_the_config_list(fake_backend):
    """Replace, not merge. Depends on config.py exposing `slurm.sbatch_extra`."""
    write_config(
        """
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/you/relay
        slurm:
          time: "04:00:00"
          sbatch_extra:
            - --mem=16G
            - --cpus-per-task=2
        """
    )

    code = cli.main(["submit", "train.py", "--sbatch-extra=--mem=64G"])

    assert code == cli.EXIT_OK
    assert fake_backend.specs[0].sbatch_extra == ["--mem=64G"]


def test_config_sbatch_extra_is_used_when_the_flag_is_absent(fake_backend):
    """Depends on config.py exposing `slurm.sbatch_extra`."""
    write_config(
        """
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/you/relay
        slurm:
          time: "04:00:00"
          sbatch_extra:
            - --mem=16G
            - --cpus-per-task=2
        """
    )

    assert cli.main(["submit", "train.py"]) == cli.EXIT_OK
    assert fake_backend.specs[0].sbatch_extra == ["--mem=16G", "--cpus-per-task=2"]


def test_gres_from_config_reaches_the_spec(fake_backend):
    """Depends on config.py exposing `slurm.gres`."""
    write_config(
        """
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/you/relay
        slurm:
          account: mlopt
          partition: ckpt-all
          time: "04:00:00"
          gres: gpu:a40:2
        """
    )

    assert cli.main(["submit", "train.py"]) == cli.EXIT_OK
    assert fake_backend.specs[0].gres == "gpu:a40:2"


# Both dictionaries, iterated rather than copied, so that adding a flag to
# `base.py` automatically adds a case here. A hand-written list would drift the
# first time somebody decides relay owns one more directive.
REJECTED_SBATCH_FLAGS = sorted(
    set(backends_base.OWNED_SBATCH_FLAGS) | set(backends_base.FIELD_SBATCH_FLAGS)
)


@pytest.mark.parametrize("flag", REJECTED_SBATCH_FLAGS)
def test_sbatch_extra_refuses_every_flag_relay_controls(
    flag, capsys, local_config, fake_backend
):
    """Exit 2, a sentence naming the entry, and nothing submitted."""
    # Long flags take `--flag=value`; short ones take `-f value`, which is how
    # a user would write either in an sbatch script.
    entry = f"{flag}=x" if flag.startswith("--") else f"{flag} x"

    code = cli.main(["submit", "train.py", f"--sbatch-extra={entry}"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_USAGE
    assert entry in captured.err
    assert flag in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    # The whole point of validating in the CLI: nothing reached a backend, so
    # on Slurm nothing reached the cluster either.
    assert fake_backend.specs == []


def test_sbatch_extra_without_a_leading_dash_is_refused(capsys, local_config, fake_backend):
    """`mem=64G` is the config-dict spelling; entries are whole directives."""
    code = cli.main(["submit", "train.py", "--sbatch-extra=mem=64G"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_USAGE
    assert "mem=64G" in captured.err
    assert "--mem=64G" in captured.err  # the message shows the right spelling
    assert "Traceback" not in captured.err
    assert fake_backend.specs == []


def test_validation_runs_before_the_backend_is_built(capsys, local_config, monkeypatch):
    """A rejected directive must not cost an SSH connection (or a Duo prompt)."""

    def explode(conf):
        raise AssertionError("build_backend was called for a rejected directive")

    monkeypatch.setattr(daemon_module, "build_backend", explode)

    code = cli.main(["submit", "train.py", "--sbatch-extra=--output=/tmp/x"])
    captured = capsys.readouterr()

    assert code == cli.EXIT_USAGE
    assert "--output" in captured.err
    # If build_backend had run, the AssertionError would have surfaced as exit
    # code 1 with its own message instead.
    assert "build_backend was called" not in captured.err
    assert "Traceback" not in captured.err

    # Nothing was written either: validation happens before the store is opened.
    with Store(cfg.db_path()) as store:
        assert store.list_runs(active_only=False) == []


def test_submit_help_documents_replacement_and_the_equals_form(capsys):
    with pytest.raises(SystemExit):
        cli.main(["submit", "--help"])
    out = capsys.readouterr().out

    # Once each for --setup, --sbatch-extra, --checkpoint-glob and
    # --resume-arg: every flag that replaces a config value rather than adding
    # to it says so in the same word.
    assert out.count("REPLACES") == 4
    assert "--mem=64G" in out


# --------------------------------------------------------------------------
# submit -- resume after preemption
# --------------------------------------------------------------------------
#
# `--checkpoint-glob` and `--resume-arg` are one setting in two flags: the glob
# finds the file, the argument tells the training script about it. They follow
# the same replace-not-merge rule as `--setup`, except that giving *either* one
# replaces *both* halves of the config's `resume:` section.
#
# The tests that need a `resume:` block in the config file are marked; the rest
# drive the flags only and hold whether or not config.py has landed.


def test_resume_flags_reach_the_spec(local_config, fake_backend):
    code = cli.main(
        [
            "submit",
            "train.py",
            "--checkpoint-glob",
            "ckpt/*.pt",
            "--resume-arg",
            "--resume {path}",
        ]
    )

    assert code == cli.EXIT_OK
    spec = fake_backend.specs[0]
    assert spec.resume_glob == "ckpt/*.pt"
    assert spec.resume_arg == "--resume {path}"

    # The stored spec is what `relay show` and any later debugging read.
    with Store(cfg.db_path()) as store:
        stored = store.list_runs()[0]["spec"]
    assert stored["resume_glob"] == "ckpt/*.pt"
    assert stored["resume_arg"] == "--resume {path}"


def test_resume_comes_from_the_config_when_no_flag_is_given(tmp_path, fake_backend):
    """Depends on config.py exposing `resume.checkpoint_glob` and `resume.arg`."""
    write_config(
        f"""
        backend: local
        local_root: {tmp_path / "runs"}
        resume:
          checkpoint_glob: "checkpoints/*.pt"
          arg: "--resume {{path}}"
        """
    )

    assert cli.main(["submit", "train.py"]) == cli.EXIT_OK
    spec = fake_backend.specs[0]
    assert spec.resume_glob == "checkpoints/*.pt"
    assert spec.resume_arg == "--resume {path}"


def test_no_resume_anywhere_leaves_the_spec_empty(local_config, fake_backend):
    """Resume is optional; relay works fully without it."""
    assert cli.main(["submit", "train.py"]) == cli.EXIT_OK
    assert fake_backend.specs[0].resume_glob is None
    assert fake_backend.specs[0].resume_arg is None


def test_resume_flags_replace_the_config_pair(tmp_path, fake_backend):
    """Replace, not merge. Depends on config.py exposing `resume:`."""
    write_config(
        f"""
        backend: local
        local_root: {tmp_path / "runs"}
        resume:
          checkpoint_glob: "checkpoints/*.pt"
          arg: "--resume {{path}}"
        """
    )

    code = cli.main(
        [
            "submit",
            "train.py",
            "--checkpoint-glob",
            "out/latest-*.ckpt",
            "--resume-arg=--load-from={path}",
        ]
    )

    assert code == cli.EXIT_OK
    spec = fake_backend.specs[0]
    # Both halves come from the command line. A glob from the flags paired
    # with an argument from the config would resume the wrong way round.
    assert spec.resume_glob == "out/latest-*.ckpt"
    assert spec.resume_arg == "--load-from={path}"


@pytest.mark.parametrize(
    "argv",
    [
        ["--checkpoint-glob", "ckpt/*.pt"],
        ["--resume-arg", "--resume {path}"],
    ],
)
def test_one_resume_flag_without_the_other_is_a_usage_error(
    argv, capsys, local_config, fake_backend
):
    code = cli.main(["submit", "train.py", *argv])
    captured = capsys.readouterr()

    assert code == cli.EXIT_USAGE
    assert "both --checkpoint-glob and --resume-arg are needed" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    # Checked before a backend is built, so on Slurm nothing reached the
    # cluster and nobody was asked for a Duo push.
    assert fake_backend.specs == []


def test_resume_arg_without_a_path_placeholder_is_a_usage_error(
    capsys, local_config, fake_backend
):
    """Without `{path}` relay has nowhere to put the file it found."""
    code = cli.main(
        [
            "submit",
            "train.py",
            "--checkpoint-glob",
            "ckpt/*.pt",
            "--resume-arg",
            "--resume latest.pt",
        ]
    )
    captured = capsys.readouterr()

    assert code == cli.EXIT_USAGE
    assert "{path}" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert fake_backend.specs == []


# --------------------------------------------------------------------------
# ls -- and the one-code-path rule
# --------------------------------------------------------------------------


FAKE_LS = {
    "runs": [
        {
            "run_id": "vr_s0_7f3a",
            "status": "running",
            "backend": "slurm",
            "job_id": "991",
            "last_step": 1000,
            "stale": True,
            "latest_metrics": {"loss": 0.412, "sps": 18400.0},
        },
        {
            "run_id": "vr_s1_ab12",
            "status": "completed",
            "backend": "local",
            "job_id": "4242",
            "last_step": None,
            "stale": False,
            "latest_metrics": {},
        },
    ],
    "last_synced": "2026-09-18T12:00:00.000Z",
    "last_synced_age_s": 5.0,
    "last_scheduler": "2026-09-18T11:58:00.000Z",
    "last_scheduler_age_s": 125.0,
}


def test_ls_table_and_json_are_two_renderings_of_one_answer(capsys, monkeypatch):
    """The point of the `Output` split: fake the data once, check both outputs."""
    monkeypatch.setattr(cli, "_ls_data", lambda store, **kwargs: FAKE_LS)

    assert cli.main(["ls"]) == cli.EXIT_OK
    table = capsys.readouterr().out
    assert "vr_s0_7f3a" in table
    assert "vr_s1_ab12" in table
    assert "(stale)" in table
    assert "loss=0.412" in table
    # Both clocks, and they disagree on purpose: the tail runs every couple of
    # seconds, the scheduler poll every thirty at the very least.
    assert "metrics 5s ago" in table
    assert "job state 2m ago" in table

    assert cli.main(["ls", "--json"]) == cli.EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == FAKE_LS
    assert [r["run_id"] for r in parsed["runs"]] == ["vr_s0_7f3a", "vr_s1_ab12"]
    assert parsed["runs"][0]["stale"] is True


def test_ls_says_never_when_the_daemon_has_not_synced(capsys, monkeypatch):
    never = dict(FAKE_LS, last_synced=None, last_synced_age_s=None)
    monkeypatch.setattr(cli, "_ls_data", lambda store, **kwargs: never)

    assert cli.main(["ls"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "last synced never" in out
    assert "relay daemon" in out


def test_ls_reads_real_rows_and_hides_finished_runs_without_all(capsys):
    seed_run("live", status="running", metrics={"loss": 0.25})
    seed_run("done", status="completed")

    assert cli.main(["ls", "--json"]) == cli.EXIT_OK
    active = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in active["runs"]] == ["live"]
    assert active["runs"][0]["latest_metrics"] == {"loss": 0.25}

    assert cli.main(["ls", "--all", "--json"]) == cli.EXIT_OK
    every = json.loads(capsys.readouterr().out)
    assert {r["run_id"] for r in every["runs"]} == {"live", "done"}


# --------------------------------------------------------------------------
# The two freshness clocks in the footer
# --------------------------------------------------------------------------
#
# The daemon tails the event logs every couple of seconds and asks `squeue`
# every thirty at the very least, so the two halves of what `ls` shows have
# genuinely different ages. One "last synced" number would have to print the
# older of them and make a healthy daemon look stalled.


def test_sync_line_shows_both_ages_when_both_clocks_have_run():
    line = cli._sync_line(
        {
            "last_synced": "2026-09-18T12:00:00.000Z",
            "last_synced_age_s": 3.0,
            "last_scheduler": "2026-09-18T11:56:00.000Z",
            "last_scheduler_age_s": 240.0,
        }
    )
    assert line == "metrics 3s ago · job state 4m ago"


def test_sync_line_says_never_polled_when_the_scheduler_clock_is_unset():
    """A daemon that is tailing but has not managed a `squeue` yet."""
    line = cli._sync_line(
        {
            "last_synced": "2026-09-18T12:00:00.000Z",
            "last_synced_age_s": 3.0,
            "last_scheduler": None,
            "last_scheduler_age_s": None,
        }
    )
    assert line == "metrics 3s ago · job state never polled"


def test_sync_line_keeps_the_old_sentence_when_nothing_has_synced_at_all():
    """No daemon at all is a different problem, and still says what to run."""
    line = cli._sync_line(
        {"last_synced": None, "last_synced_age_s": None, "last_scheduler": None}
    )
    assert line == "last synced never — is `relay daemon` running?"


def test_sync_line_says_idle_when_there_are_no_active_runs():
    """With nothing to watch, two fresh ages would read as polling for nothing.

    An idle daemon makes no cluster calls at all -- no `squeue`, no `sacct`,
    no event-log reads -- and only cycles locally. But "metrics 3s ago · job
    state 3s ago" under an empty table looks exactly like a daemon hammering
    the cluster on a schedule, which is what a user assumed on seeing it. One
    honest line instead: nothing active, and the daemon is alive.
    """
    line = cli._sync_line(
        {
            "active_runs": 0,
            "last_synced": "2026-09-18T12:00:00.000Z",
            "last_synced_age_s": 3.0,
            # Hours old, and correctly so: nothing has been asked since the
            # last run finished. Not shown, because it would look alarming.
            "last_scheduler": "2026-09-18T09:00:00.000Z",
            "last_scheduler_age_s": 10800.0,
        }
    )
    assert line == "no active runs · daemon alive 3s ago"


def test_sync_line_never_synced_outranks_idle():
    """A daemon that never ran is a different problem from one with no work."""
    line = cli._sync_line({"active_runs": 0, "last_synced": None})
    assert line == "last synced never — is `relay daemon` running?"


def test_sync_footer_counts_active_runs(monkeypatch):
    with Store(cfg.db_path()) as store:
        assert cli._sync_footer(store)["active_runs"] == 0
        store.create_run("vr_a", backend="local", job_id="1", status="running")
        store.create_run("vr_b", backend="local", job_id="2", status="completed")
        assert cli._sync_footer(store)["active_runs"] == 1


def test_sync_footer_carries_both_clocks(monkeypatch):
    with Store(cfg.db_path()) as store:
        store.mark_synced("2026-09-18T12:00:00.000Z")
        store.mark_scheduler_synced("2026-09-18T11:58:00.000Z")
        footer = cli._sync_footer(store)

    assert footer["last_synced"] == "2026-09-18T12:00:00.000Z"
    assert footer["last_scheduler"] == "2026-09-18T11:58:00.000Z"
    # Both ages are real numbers, and the scheduler's is the older one.
    assert footer["last_scheduler_age_s"] > footer["last_synced_age_s"] > 0
    assert "daemon_state" in footer


def test_ls_json_carries_the_scheduler_clock(capsys):
    """Scripts parse `--json`, so both clocks have to be in the data."""
    seed_run("live", status="running")
    with Store(cfg.db_path()) as store:
        store.mark_synced()
        store.mark_scheduler_synced()

    assert cli.main(["ls", "--json"]) == cli.EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    # The original three keys are untouched -- scripts already read them.
    assert parsed["last_synced"] is not None
    assert parsed["last_synced_age_s"] is not None
    assert "daemon_state" in parsed
    assert parsed["last_scheduler"] is not None
    assert parsed["last_scheduler_age_s"] is not None


# --------------------------------------------------------------------------
# show
# --------------------------------------------------------------------------


def test_show_unknown_run_exits_not_found(capsys):
    code = cli.main(["show", "nope"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_NOT_FOUND
    assert "nope" in captured.err
    assert "relay ls" in captured.err
    assert captured.out == ""


def test_show_known_run_in_both_formats(capsys):
    seed_run("vr_s0_7f3a", metrics={"loss": 0.5}, logs=(("stdout", "epoch 1"),))

    assert cli.main(["show", "vr_s0_7f3a"]) == cli.EXIT_OK
    text = capsys.readouterr().out
    assert "vr_s0_7f3a" in text
    assert "loss" in text

    assert cli.main(["show", "vr_s0_7f3a", "--json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["run"]["run_id"] == "vr_s0_7f3a"
    assert data["latest_metrics"] == {"loss": 0.5}
    assert data["metric_names"] == ["loss"]
    assert data["counts"]["events"] == 2
    assert [e["type"] for e in data["recent_events"]] == ["log", "metric"]


def test_show_prints_attempts_and_the_usage_rows_behind_them(capsys):
    """A requeued run is several attempts under one job ID, and each cost hours."""
    seed_run("vr", job_id="991", run_dir="/r/vr")
    with Store(cfg.db_path()) as store:
        # `attempts` is derived at ingest from run_started: SLURM_RESTART_COUNT
        # is 1 on the second start, which makes two attempts so far.
        store.ingest(
            "vr",
            "/r/vr/events.jsonl",
            [
                {
                    "v": 2,
                    "run": "vr",
                    "seq": 10,
                    "ts": "2026-09-18T09:10:00.000Z",
                    "type": "run_started",
                    "payload": {"attempt": 1},
                }
            ],
            100,
        )
        store.upsert_usage(
            [
                usage_row(job_id="991", run_id="vr", state="PREEMPTED", elapsed_s=3600, gpus=1),
                usage_row(
                    job_id="991",
                    run_id="vr",
                    state="RUNNING",
                    start="2026-09-18T12:10:00Z",
                    end=None,
                    elapsed_s=600,
                    gpus=1,
                ),
            ]
        )

    assert cli.main(["show", "vr"]) == cli.EXIT_OK
    text = capsys.readouterr().out
    assert "attempts  2" in text
    assert "usage by attempt" in text
    assert "PREEMPTED" in text
    assert "(running)" in text  # the attempt with no end time yet

    assert cli.main(["show", "vr", "--json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["run"]["attempts"] == 2
    assert [row["state"] for row in data["usage"]] == ["PREEMPTED", "RUNNING"]


def test_show_lists_each_attempt_and_what_it_resumed_from(capsys):
    """Depends on store.attempt_history() landing.

    The `attempts` count says a run restarted. This table says whether the
    restart continued the work or quietly started again from step zero, which
    is the difference between a preemption that cost nothing and one that cost
    the whole attempt.
    """
    seed_run("vr", job_id="991", run_dir="/r/vr")
    with Store(cfg.db_path()) as store:
        store.ingest(
            "vr",
            "/r/vr/events.jsonl",
            [
                {
                    "v": 2,
                    "run": "vr",
                    "seq": 1,
                    "ts": "2026-09-18T09:00:00.000Z",
                    "type": "run_started",
                    # The first attempt has nothing to resume from.
                    "payload": {"attempt": 0},
                },
                {
                    "v": 2,
                    "run": "vr",
                    "seq": 40,
                    "ts": "2026-09-18T10:30:00.000Z",
                    "type": "run_started",
                    "payload": {
                        "attempt": 1,
                        "resumed_from": "/r/vr/ckpt/step-1000.pt",
                    },
                },
            ],
            500,
        )

    assert cli.main(["show", "vr"]) == cli.EXIT_OK
    text = capsys.readouterr().out
    assert "ATTEMPT" in text
    assert "RESUMED FROM" in text
    assert "2026-09-18T09:00:00.000Z" in text
    assert "/r/vr/ckpt/step-1000.pt" in text
    # A fresh start renders as "-", the same empty cell every other relay
    # table uses.
    assert "0        2026-09-18T09:00:00.000Z  -" in text

    assert cli.main(["show", "vr", "--json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    history = data["attempts_history"]
    assert [row["attempt"] for row in history] == [0, 1]
    assert history[0]["resumed_from"] is None
    assert history[1]["resumed_from"] == "/r/vr/ckpt/step-1000.pt"


# --------------------------------------------------------------------------
# logs
# --------------------------------------------------------------------------


def test_logs_prints_lines_and_tags_stderr(capsys):
    seed_run(
        "vr",
        logs=(("stdout", "epoch 1 done"), ("stderr", "warning: lr is high")),
    )

    assert cli.main(["logs", "vr"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "epoch 1 done" in out
    assert "stderr| warning: lr is high" in out

    assert cli.main(["logs", "vr", "--json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert [e["payload"]["line"] for e in data["logs"]] == [
        "epoch 1 done",
        "warning: lr is high",
    ]

    assert cli.main(["logs", "vr", "--since-seq", "1"]) == cli.EXIT_OK
    assert "epoch 1 done" not in capsys.readouterr().out


def test_logs_unknown_run_exits_not_found(capsys):
    assert cli.main(["logs", "nope"]) == cli.EXIT_NOT_FOUND
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------


def test_cancel_asks_the_backend_and_records_it(capsys, local_config, fake_backend):
    seed_run("vr", job_id="4242")

    code = cli.main(["cancel", "vr"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert fake_backend.cancelled == ["4242"]
    assert get_run("vr")["status"] == "cancelled"
    assert "vr" in out


def test_cancel_hands_the_backend_the_run_directory(local_config, fake_backend):
    """Without the run directory the backend cannot write CANCEL_REQUESTED.

    That file is the only way the sidecar can tell a cancel's SIGTERM from a
    preemption's SIGTERM, so a cancelled run would come straight back under
    `requeue_on_term: always`.
    """
    seed_run("vr", job_id="4242", run_dir="/mmfs1/gscratch/mlopt/you/relay/vr")

    assert cli.main(["cancel", "vr"]) == cli.EXIT_OK
    assert fake_backend.cancel_calls == [
        {"job_id": "4242", "run_dir": "/mmfs1/gscratch/mlopt/you/relay/vr"}
    ]


def test_cancel_unknown_run_exits_not_found(capsys, local_config, fake_backend):
    assert cli.main(["cancel", "nope"]) == cli.EXIT_NOT_FOUND
    assert fake_backend.cancelled == []


def test_cancel_without_config_exits_not_configured(capsys):
    seed_run("vr")
    assert cli.main(["cancel", "vr"]) == cli.EXIT_NOT_CONFIGURED
    assert "relay init" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Poking the daemon's scheduler poll
# --------------------------------------------------------------------------
#
# The poll backs off towards five minutes while nothing changes. Submitting or
# cancelling is exactly when the user starts watching for a result, so both
# leave a timestamp the daemon reads as "look now".


def test_submit_pokes_the_scheduler(tmp_path, local_config, fake_backend):
    script = tmp_path / "train.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    with Store(cfg.db_path()) as store:
        assert store.scheduler_poke() is None

    assert cli.main(["submit", str(script)]) == cli.EXIT_OK

    with Store(cfg.db_path()) as store:
        assert store.scheduler_poke() is not None


def test_submit_pokes_once_for_a_whole_sweep(tmp_path, local_config, fake_backend):
    """One note, not one per seed: five seeds still mean one "look now"."""
    script = tmp_path / "train.py"
    script.write_text("print('hi')\n", encoding="utf-8")

    assert cli.main(["submit", str(script), "--seeds", "0-4"]) == cli.EXIT_OK

    with Store(cfg.db_path()) as store:
        assert store.scheduler_poke() is not None
        assert len(store.list_runs()) == 5


def test_cancel_pokes_the_scheduler(local_config, fake_backend):
    seed_run("vr", job_id="4242")
    with Store(cfg.db_path()) as store:
        assert store.scheduler_poke() is None

    assert cli.main(["cancel", "vr"]) == cli.EXIT_OK

    with Store(cfg.db_path()) as store:
        assert store.scheduler_poke() is not None


# --------------------------------------------------------------------------
# connect
# --------------------------------------------------------------------------


@pytest.fixture
def fake_ssh(monkeypatch):
    """Stand in for the two places `relay connect` touches a real ssh.

    `cli._run_interactive` and `cli._master_alive` exist as one-line
    module-level functions precisely so this fixture can replace them; nothing
    here opens a socket, reads ~/.ssh/config or waits for a Duo prompt.
    """
    calls: dict = {
        "argv": None,
        "exit_code": 0,
        "stderr": "",
        "alive": True,
        "checked": [],
    }

    def run(argv):
        calls["argv"] = argv
        # `_run_interactive` tees ssh's stderr: it shows the user every line as
        # ssh prints it *and* hands the text back, so relay can quote ssh's own
        # explanation instead of guessing. Hence the tuple.
        return calls["exit_code"], calls["stderr"]

    def alive(alias):
        calls["checked"].append(alias)
        return calls["alive"]

    monkeypatch.setattr(cli, "_run_interactive", run)
    monkeypatch.setattr(cli, "_master_alive", alive)
    return calls


def cli_ssh_persist() -> str:
    from relay import ssh

    return ssh.CONTROL_PERSIST


SLURM_CONFIG = """
    backend: slurm
    ssh_alias: klone
    remote_root: /mmfs1/gscratch/mlopt/you/relay
    slurm:
      time: "04:00:00"
    """


def test_connect_opens_the_master_and_says_how_long_it_lasts(capsys, fake_ssh):
    write_config(SLURM_CONFIG)

    assert cli.main(["connect"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Connected to klone" in out
    assert cli_ssh_persist() in out

    # The interactive call: master, no remote command, background after auth,
    # and emphatically no BatchMode -- Duo has to be able to prompt.
    argv = fake_ssh["argv"]
    assert argv[0] == "ssh"
    assert {"-M", "-N", "-f"} <= set(argv)
    assert "BatchMode=yes" not in argv
    assert argv[-1] == "klone"
    # A zero exit only means ssh forked, so relay checks the socket after.
    assert fake_ssh["checked"] == ["klone"]


def test_connect_failing_ssh_exits_transport(capsys, fake_ssh):
    write_config(SLURM_CONFIG)
    fake_ssh["exit_code"] = 255

    code = cli.main(["connect"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_TRANSPORT
    assert "klone" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_connect_quotes_sshs_own_reason_instead_of_guessing(capsys, fake_ssh):
    """The failure that started all this: Duo succeeded, then ssh could not
    bind the control socket because ~/.relay did not exist. relay's old message
    told the user to check that plain `ssh klone` works -- the one thing that
    was demonstrably fine.

    Limitation, stated plainly: this fixture never runs ssh, so this test could
    not have caught the missing-directory bug itself. It pins the *shape of the
    message*, nothing more. Another test covers the real `ssh` path.
    """
    write_config(SLURM_CONFIG)
    fake_ssh["exit_code"] = 255
    fake_ssh["stderr"] = (
        "unix_listener: cannot bind to path /x/.relay/cm-abc.123: "
        "No such file or directory\n"
    )

    code = cli.main(["connect"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_TRANSPORT
    assert (
        "unix_listener: cannot bind to path /x/.relay/cm-abc.123: "
        "No such file or directory" in err
    )
    assert "plain" not in err
    assert "Check that plain" not in err


def test_connect_falls_back_to_advice_only_when_ssh_said_nothing(capsys, fake_ssh):
    """No stderr to quote, so the old suggestion is the best relay has.

    Same limitation as above: no real ssh runs here, only the message shape.
    """
    write_config(SLURM_CONFIG)
    fake_ssh["exit_code"] = 255
    fake_ssh["stderr"] = ""

    code = cli.main(["connect"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_TRANSPORT
    assert "Check that plain `ssh klone` works" in err


def test_connect_quotes_the_last_stderr_line_not_the_first(capsys, fake_ssh):
    """ssh prints banners and warnings first; the reason it failed comes last.

    Same limitation as above: no real ssh runs here, only the message shape.
    """
    write_config(SLURM_CONFIG)
    fake_ssh["exit_code"] = 255
    fake_ssh["stderr"] = (
        "Warning: Permanently added 'klone' to the list of known hosts.\n"
        "debug1: Entering interactive session.\n"
        "unix_listener: cannot bind to path /x/.relay/cm-abc.123: "
        "No such file or directory\n"
        "\n"
    )

    assert cli.main(["connect"]) == cli.EXIT_TRANSPORT
    err = capsys.readouterr().err
    # The ConnectFailed sentence quotes the last non-blank line...
    assert "ssh exited 255 while opening a connection to klone: unix_listener" in err
    # ...and not the banner, which is what `_first_line` would have picked.
    assert "Permanently added" not in err.split("ssh exited 255")[-1]


def test_run_interactive_tees_stderr_to_the_terminal_and_back(capfd):
    """The real seam, with a real child process: every stderr line must reach
    the terminal (so a Duo prompt is visible as it is written) *and* come back
    to relay (so relay can quote ssh in its own error).

    capfd rather than capsys because a child process writes to fd 2; relay's
    parent-side rewrite goes through `sys.stderr`, and capfd sees both.
    """
    code, err = cli._run_interactive(
        [
            sys.executable,
            "-c",
            "import sys; print('warn1', file=sys.stderr); "
            "print('warn2', file=sys.stderr); sys.exit(255)",
        ]
    )

    assert code == 255
    # Returned to the caller...
    assert "warn1" in err
    assert "warn2" in err
    # ...and also written out where the user can see it.
    captured = capfd.readouterr().err
    assert "warn1" in captured
    assert "warn2" in captured


def test_connect_without_a_live_master_afterwards_exits_transport(capsys, fake_ssh):
    """ssh -f backgrounds itself, so exit 0 is not proof of a working master."""
    write_config(SLURM_CONFIG)
    fake_ssh["alive"] = False

    code = cli.main(["connect"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_TRANSPORT
    assert "no control master" in err


def test_connect_on_the_local_backend_says_so_and_succeeds(capsys, fake_ssh, local_config):
    assert cli.main(["connect"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "local backend" in out
    # Nothing was run: there is no cluster to connect to.
    assert fake_ssh["argv"] is None


def test_connect_without_config_exits_not_configured(capsys, fake_ssh):
    code = cli.main(["connect"])
    assert code == cli.EXIT_NOT_CONFIGURED
    assert "relay init" in capsys.readouterr().err
    assert fake_ssh["argv"] is None


# --------------------------------------------------------------------------
# usage
# --------------------------------------------------------------------------


def usage_row(**fields) -> dict:
    """One `usage` row in the shape the daemon hands the store."""
    row = {
        "job_id": "1",
        "run_id": "good",
        "state": "COMPLETED",
        "partition": "ckpt-all",
        "start": "2026-09-18T10:00:00Z",
        "end": "2026-09-18T12:00:00Z",
        "elapsed_s": 7200,
        "gpus": 2,
        "cpus": 8,
    }
    row.update(fields)
    return row


@pytest.fixture
def usage_db():
    """Three runs of scheduler accounting, as the daemon would have written it.

    The numbers are chosen so every total is exact:

      good       7200s x 2 GPUs                        = 4.0 GPU-hours
      preempted  3600s x 1 GPU (checkpoint at 08:30)   = 1.0, 0.5 of it lost
      preempted  1800s x 1 GPU (no checkpoint at all)  = 0.5, all of it lost
      crashed    1800s x 1 GPU, run status failed      = 0.5, all of it wasted
                                                  total = 6.0 GPU-hours
    """
    seed_run("good", status="completed", job_id="1", spec={"account": "mlopt"})
    seed_run("preempted", status="running", job_id="2", spec={"account": "mlopt"})
    seed_run("crashed", status="failed", job_id="3", spec={"account": "other"})

    with Store(cfg.db_path()) as store:
        # A checkpoint halfway through the first preempted attempt. Everything
        # after it is what preemption actually cost.
        store.ingest(
            "preempted",
            "/r/preempted/events.jsonl",
            [
                {
                    "v": 2,
                    "run": "preempted",
                    "seq": 1,
                    "ts": "2026-09-18T08:30:00.000Z",
                    "type": "checkpoint",
                    "payload": {"step": 500},
                }
            ],
            100,
        )
        store.upsert_usage(
            [
                usage_row(),
                usage_row(
                    job_id="2",
                    run_id="preempted",
                    state="PREEMPTED",
                    start="2026-09-18T08:00:00Z",
                    end="2026-09-18T09:00:00Z",
                    elapsed_s=3600,
                    gpus=1,
                    cpus=4,
                ),
                usage_row(
                    job_id="2",
                    run_id="preempted",
                    state="PREEMPTED",
                    start="2026-09-18T09:10:00Z",
                    end="2026-09-18T09:40:00Z",
                    elapsed_s=1800,
                    gpus=1,
                    cpus=4,
                ),
                usage_row(
                    job_id="3",
                    run_id="crashed",
                    state="FAILED",
                    partition="gpu-a40",
                    start="2026-09-18T07:00:00Z",
                    end="2026-09-18T07:30:00Z",
                    elapsed_s=1800,
                    gpus=1,
                    cpus=4,
                ),
            ]
        )


FAKE_USAGE = {
    "by": "run",
    "rows": [
        {
            "key": "vr_s0_7f3a",
            "gpu_hours": 4.0,
            "cpu_hours": 16.0,
            "attempts": 2,
            "runs": 1,
        }
    ],
    "totals": {
        "gpu_hours": 4.0,
        "cpu_hours": 16.0,
        "attempts": 2,
        "runs": 1,
        "by_partition": [{"partition": "ckpt-all", "gpu_hours": 4.0, "cpu_hours": 16.0}],
    },
    "wasted": {
        "failed": {"gpu_hours": 0.5, "cpu_hours": 2.0, "runs": ["crashed"]},
        "lost_to_preemption": {
            "gpu_hours": 1.0,
            "cpu_hours": 4.0,
            "attempts": [
                {
                    "run_id": "vr_s0_7f3a",
                    "job_id": "991",
                    "start": "2026-09-18T09:10:00Z",
                    "end": "2026-09-18T09:40:00Z",
                    "lost_s": 1800,
                    "gpus": 1,
                    "no_checkpoints": True,
                }
            ],
        },
    },
    "cost": {
        "gpu_hour_rate": 2.0,
        "total": 8.0,
        "wasted": {"failed": 1.0, "lost_to_preemption": 2.0},
    },
    "last_synced": "2026-09-18T12:00:00.000Z",
    "last_synced_age_s": 5.0,
    "last_scheduler": "2026-09-18T11:58:00.000Z",
    "last_scheduler_age_s": 125.0,
    "daemon_state": {"last_error_kind": None},
}


def test_usage_table_and_json_are_two_renderings_of_one_answer(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_usage_data", lambda store, **kwargs: FAKE_USAGE)

    assert cli.main(["usage"]) == cli.EXIT_OK
    table = capsys.readouterr().out
    assert "vr_s0_7f3a" in table
    assert "4.00" in table  # GPU-hours in the table and in the total
    assert "ckpt-all" in table  # the partition split
    assert "Wasted" in table
    assert "runs: crashed" in table
    assert "no checkpoints reported" in table
    assert "$8.00" in table  # cost, because a rate was configured
    assert "metrics 5s ago · job state 2m ago" in table

    assert cli.main(["usage", "--json"]) == cli.EXIT_OK
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == FAKE_USAGE


def test_usage_reads_real_rows_and_adds_them_up(capsys, usage_db):
    assert cli.main(["usage", "--json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)

    assert data["by"] == "run"
    assert data["totals"]["gpu_hours"] == pytest.approx(6.0)
    assert data["totals"]["cpu_hours"] == pytest.approx(24.0)
    assert {row["key"] for row in data["rows"]} == {"good", "preempted", "crashed"}

    by_partition = {p["partition"]: p["gpu_hours"] for p in data["totals"]["by_partition"]}
    assert by_partition["ckpt-all"] == pytest.approx(5.5)
    assert by_partition["gpu-a40"] == pytest.approx(0.5)

    # The crashed run's whole allocation is wasted; the preempted attempts lose
    # only what came after their last checkpoint, and all of it when there was
    # no checkpoint at all.
    assert data["wasted"]["failed"]["gpu_hours"] == pytest.approx(0.5)
    assert data["wasted"]["failed"]["runs"] == ["crashed"]
    assert data["wasted"]["lost_to_preemption"]["gpu_hours"] == pytest.approx(1.0)
    flags = {a["no_checkpoints"] for a in data["wasted"]["lost_to_preemption"]["attempts"]}
    assert flags == {True, False}

    # No config in this test, so no rate, so no dollars anywhere.
    assert "cost" not in data


def test_usage_human_output_has_the_table_totals_and_wasted_block(capsys, usage_db):
    assert cli.main(["usage"]) == cli.EXIT_OK
    out = capsys.readouterr().out

    assert "RUN" in out
    assert "good" in out
    assert "total" in out
    assert "6.00" in out
    assert "PARTITION" in out
    assert "Wasted" in out
    assert "runs: crashed" in out
    assert "no checkpoints reported — your script is not printing checkpoint lines" in out
    # Nothing in this test ran a daemon cycle, so the footer is the "never"
    # sentence rather than the two ages.
    assert "last synced never" in out
    assert "$" not in out


def test_usage_cost_appears_only_when_a_rate_is_configured(capsys, usage_db):
    write_config(
        """
        backend: local
        cost:
          gpu_hour_rate: 2.0
        """
    )

    assert cli.main(["usage", "--json"]) == cli.EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["cost"]["gpu_hour_rate"] == 2.0
    # 6 GPU-hours at 2.0, 0.5 wasted on the failed run, 1.0 lost to preemption.
    assert data["cost"]["total"] == pytest.approx(12.0)
    assert data["cost"]["wasted"]["failed"] == pytest.approx(1.0)
    assert data["cost"]["wasted"]["lost_to_preemption"] == pytest.approx(2.0)

    assert cli.main(["usage"]) == cli.EXIT_OK
    assert "$12.00" in capsys.readouterr().out


def test_usage_by_partition_and_group_change_the_grouping(capsys, usage_db):
    assert cli.main(["usage", "--by", "partition", "--json"]) == cli.EXIT_OK
    partitions = json.loads(capsys.readouterr().out)
    assert partitions["by"] == "partition"
    assert {row["key"] for row in partitions["rows"]} == {"ckpt-all", "gpu-a40"}

    assert cli.main(["usage", "--by", "group", "--json"]) == cli.EXIT_OK
    groups = json.loads(capsys.readouterr().out)
    # The account comes out of the spec relay stored at submit time.
    assert {row["key"] for row in groups["rows"]} == {"mlopt", "other"}

    # --group narrows to one account whatever the grouping is.
    assert cli.main(["usage", "--group", "mlopt", "--json"]) == cli.EXIT_OK
    mlopt = json.loads(capsys.readouterr().out)
    assert {row["key"] for row in mlopt["rows"]} == {"good", "preempted"}
    assert mlopt["totals"]["gpu_hours"] == pytest.approx(5.5)


def test_usage_since_is_forwarded_to_every_query(capsys, monkeypatch, usage_db):
    seen: dict = {}

    def fake_usage_by(self, by="run", *, since=None, group=None):
        seen["rows"] = (by, since, group)
        return []

    def fake_totals(self, *, since=None, group=None):
        seen["totals"] = (since, group)
        return {"gpu_hours": 0.0, "cpu_hours": 0.0, "attempts": 0, "runs": 0, "by_partition": []}

    def fake_wasted(self, *, since=None, group=None):
        seen["wasted"] = (since, group)
        return {
            "failed": {"gpu_hours": 0.0, "cpu_hours": 0.0, "runs": []},
            "lost_to_preemption": {"gpu_hours": 0.0, "cpu_hours": 0.0, "attempts": []},
        }

    monkeypatch.setattr(Store, "usage_by", fake_usage_by)
    monkeypatch.setattr(Store, "usage_totals", fake_totals)
    monkeypatch.setattr(Store, "wasted_hours", fake_wasted)

    code = cli.main(
        ["usage", "--by", "partition", "--since", "2026-09-01", "--group", "mlopt", "--json"]
    )
    assert code == cli.EXIT_OK
    assert seen["rows"] == ("partition", "2026-09-01", "mlopt")
    assert seen["totals"] == ("2026-09-01", "mlopt")
    assert seen["wasted"] == ("2026-09-01", "mlopt")


def test_usage_rejects_a_date_it_cannot_read(capsys, usage_db):
    code = cli.main(["usage", "--since", "last tuesday"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_USAGE
    assert "2026-09-01" in err  # the message shows the shape relay wants
    assert "Traceback" not in err


def test_usage_on_an_empty_database_still_prints_a_sync_line(capsys):
    assert cli.main(["usage"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "No usage recorded yet" in out
    assert "last synced never" in out


# --------------------------------------------------------------------------
# The daemon's last error, shown under ls and usage
# --------------------------------------------------------------------------


def set_daemon_error(kind: str | None, **fields) -> None:
    with Store(cfg.db_path()) as store:
        store.set_daemon_state(last_error_kind=kind, **fields)


def test_auth_required_tells_ls_and_usage_to_run_connect(capsys):
    seed_run("vr")
    set_daemon_error(
        "auth_required",
        last_error_ts="2026-09-18T12:00:00Z",
        last_error_detail="ssh exited 255 with no live master",
    )

    assert cli.main(["ls"]) == cli.EXIT_OK
    assert "Authentication needed. Run `relay connect`." in capsys.readouterr().out

    assert cli.main(["usage"]) == cli.EXIT_OK
    assert "Authentication needed. Run `relay connect`." in capsys.readouterr().out

    # And a script reading --json sees the same state, not just the sentence.
    assert cli.main(["ls", "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["daemon_state"]["last_error_kind"] == (
        "auth_required"
    )


def test_unreachable_says_what_happened_in_one_line(capsys):
    seed_run("vr")
    set_daemon_error(
        "unreachable",
        last_error_ts="2026-09-18T12:00:00Z",
        last_error_detail="ssh: connect to host klone port 22: Network is unreachable",
    )

    assert cli.main(["ls"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith("Cluster unreachable")]
    assert len(line) == 1
    assert "2026-09-18T12:00:00Z" in line[0]
    assert "Network is unreachable" in line[0]
    assert "relay connect" not in out


def test_no_daemon_error_means_no_extra_line(capsys):
    seed_run("vr")

    assert cli.main(["ls"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Authentication needed" not in out
    assert "Cluster unreachable" not in out


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _fake_checks():
    return [
        doctor_module.CheckResult("config", doctor_module.OK, "Config parses.", 1.0),
        doctor_module.CheckResult("ssh", doctor_module.ERROR, "Could not reach klone.", 2.0),
    ]


def test_doctor_reports_every_check_and_fails_on_an_error(capsys, monkeypatch):
    monkeypatch.setattr(doctor_module, "run_checks", lambda *a, **k: _fake_checks())

    code = cli.main(["doctor"])
    text = capsys.readouterr().out
    # Every check is reported, not just the first failure.
    assert "config" in text
    assert "ssh" in text
    assert "Could not reach klone." in text
    assert code == cli.EXIT_FAILURE

    assert cli.main(["doctor", "--json"]) == cli.EXIT_FAILURE
    data = json.loads(capsys.readouterr().out)
    assert [c["name"] for c in data] == ["config", "ssh"]
    assert data[1]["status"] == doctor_module.ERROR


def test_doctor_exits_zero_when_nothing_errored(capsys, monkeypatch):
    results = [doctor_module.CheckResult("database", doctor_module.WARN, "No database yet.")]
    monkeypatch.setattr(doctor_module, "run_checks", lambda *a, **k: results)
    assert cli.main(["doctor"]) == cli.EXIT_OK


# --------------------------------------------------------------------------
# dash and daemon
# --------------------------------------------------------------------------


def test_dash_serves_with_a_cancel_hook(capsys, monkeypatch, local_config, fake_backend):
    from relay.web import app as web_app

    captured: dict = {}
    monkeypatch.setattr(web_app, "serve", lambda **kwargs: captured.update(kwargs))

    assert cli.main(["dash", "--port", "9999"]) == cli.EXIT_OK
    assert captured["port"] == 9999
    assert captured["store_path"] == str(cfg.db_path())
    assert captured["cancel_fn"] is not None

    # The injected hook is the same path `relay cancel` takes, run directory
    # and all -- the dashboard's Cancel button must not skip CANCEL_REQUESTED.
    seed_run("vr", job_id="77", run_dir="/mmfs1/gscratch/mlopt/you/relay/vr")
    captured["cancel_fn"]("vr")
    assert fake_backend.cancel_calls == [
        {"job_id": "77", "run_dir": "/mmfs1/gscratch/mlopt/you/relay/vr"}
    ]
    assert get_run("vr")["status"] == "cancelled"


def test_daemon_already_running_exits_failure(capsys, monkeypatch):
    lock = str(cfg.daemon_lock_path())

    def boom(*a, **k):
        raise daemon_module.AlreadyRunning(
            f"Another relay daemon is already running: it holds the lock on {lock}."
        )

    monkeypatch.setattr(daemon_module, "main", boom)

    code = cli.main(["daemon"])
    err = capsys.readouterr().err
    assert code == cli.EXIT_FAILURE
    assert lock in err
    assert "Traceback" not in err


# --------------------------------------------------------------------------
# relay daemon's interval flags
# --------------------------------------------------------------------------
#
# The CLI's whole job here is plumbing: turn six flags into the override dict
# `daemon_intervals` merges onto the config. The merge itself is tested in
# test_daemon.py, so these tests stop at the boundary and check that each flag
# arrives under the right key.


@pytest.fixture
def captured_daemon(monkeypatch):
    """Replace the daemon's main with one that records its keyword arguments."""
    calls: list[dict] = []

    def fake_main(*args, **kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(daemon_module, "main", fake_main)
    return calls


@pytest.mark.parametrize(
    "flag,key",
    [
        # The flag names describe what is being tuned; the keys are `Daemon`'s
        # own parameter names. The two deliberately differ, which is exactly
        # why this mapping is worth a test.
        ("--tail-interval-active", "interval_active"),
        ("--tail-interval-queued", "interval_queued"),
        ("--tail-interval-idle", "interval_idle"),
        ("--scheduler-interval-min", "scheduler_interval_min"),
        ("--scheduler-interval-max", "scheduler_interval_max"),
        ("--usage-interval", "usage_interval_seconds"),
    ],
)
def test_each_interval_flag_reaches_the_daemon_under_its_own_key(
    captured_daemon, flag, key
):
    assert cli.main(["daemon", flag, "12.5"]) == cli.EXIT_OK

    intervals = captured_daemon[0]["intervals"]
    assert intervals[key] == 12.5
    # Everything else is None, so the config keeps its say over the five the
    # user did not mention.
    assert all(value is None for name, value in intervals.items() if name != key)


def test_daemon_with_no_flags_overrides_nothing(captured_daemon):
    """Every value None means "the config decides", not "use these numbers"."""
    assert cli.main(["daemon"]) == cli.EXIT_OK

    intervals = captured_daemon[0]["intervals"]
    assert set(intervals) == {
        "interval_active",
        "interval_queued",
        "interval_idle",
        "scheduler_interval_min",
        "scheduler_interval_max",
        "usage_interval_seconds",
    }
    assert all(value is None for value in intervals.values())


def test_daemon_flags_can_be_combined(captured_daemon):
    code = cli.main(
        ["daemon", "--tail-interval-active", "0.5", "--scheduler-interval-max", "600"]
    )
    assert code == cli.EXIT_OK

    intervals = captured_daemon[0]["intervals"]
    assert intervals["interval_active"] == 0.5
    assert intervals["scheduler_interval_max"] == 600.0
    assert intervals["scheduler_interval_min"] is None


def test_daemon_help_says_the_flags_beat_the_config(capsys):
    with pytest.raises(SystemExit):
        cli.main(["daemon", "--help"])
    out = capsys.readouterr().out
    assert "--scheduler-interval-min" in out
    assert "--usage-interval" in out
    assert "config" in out


def test_daemon_rejects_a_non_numeric_interval(capsys):
    """argparse's own type check, which is exit code 2 for free."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["daemon", "--tail-interval-active", "soon"])
    assert exc.value.code == cli.EXIT_USAGE


# --------------------------------------------------------------------------
# Exit codes and usage
# --------------------------------------------------------------------------


def _exception_cases():
    from relay.backends.slurm import SlurmError, TransportError

    return [
        (cli.NotFound("No run 'x'. Run `relay ls`."), cli.EXIT_NOT_FOUND),
        (cli.BadUsage("--seeds was empty."), cli.EXIT_USAGE),
        (cfg.ConfigNotFound("No config. Run `relay init`."), cli.EXIT_NOT_CONFIGURED),
        (cfg.ConfigError("Your YAML has a typo."), cli.EXIT_FAILURE),
        (TransportError("ssh timed out reaching klone."), cli.EXIT_TRANSPORT),
        (SlurmError("sbatch refused the job."), cli.EXIT_FAILURE),
        (OSError("the connection went away"), cli.EXIT_TRANSPORT),
        (RuntimeError("something unexpected"), cli.EXIT_FAILURE),
    ]


@pytest.mark.parametrize("exc,expected", _exception_cases())
def test_every_exception_becomes_an_exit_code_and_a_sentence(capsys, monkeypatch, exc, expected):
    def raiser(args):
        raise exc

    # build_parser() runs inside main(), so it picks up the patched function.
    monkeypatch.setattr(cli, "cmd_ls", raiser)

    code = cli.main(["ls"])
    captured = capsys.readouterr()
    assert code == expected
    assert str(exc) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_relay_debug_re_raises(monkeypatch):
    """The escape hatch for working on relay itself."""

    def raiser(args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "cmd_ls", raiser)
    monkeypatch.setenv("RELAY_DEBUG", "1")
    with pytest.raises(RuntimeError):
        cli.main(["ls"])


@pytest.mark.parametrize("argv", [["nosuchcommand"], ["show"], ["ls", "--nope"]])
def test_bad_usage_exits_two(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == cli.EXIT_USAGE


def test_no_command_prints_help(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert "COMMAND" in capsys.readouterr().out


def test_passthrough_split_keeps_relay_flags_out_of_the_users_argv():
    own, passthrough = cli._split_passthrough(
        ["submit", "train.py", "--seeds", "0-4", "--", "--lr", "3e-4", "--json"]
    )
    assert own == ["submit", "train.py", "--seeds", "0-4"]
    assert passthrough == ["--lr", "3e-4", "--json"]


# --------------------------------------------------------------------------
# relay daemon speaks at INFO
# --------------------------------------------------------------------------


def test_daemon_shows_its_info_messages(monkeypatch):
    """A foreground process that says nothing for an hour looks hung.

    The daemon logs at INFO the things a person leaving it in a terminal tab
    wants to see -- that it started, each run's status change, that it is
    stopping. Python shows only WARNING and above unless somebody configures
    logging, and nothing in relay did, so `relay daemon` sat there silent and
    a user reasonably asked whether it was working. Every other command stays
    quiet: only `cmd_daemon` turns this on.
    """
    import io

    stream = io.StringIO()
    logger = logging.getLogger("relay")
    # Start from a clean logger, whatever an earlier test left behind, and
    # put it back afterwards so this test cannot make the rest of the suite
    # chatty.
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.NOTSET)

    def fake_main(**kwargs):
        logging.getLogger("relay.daemon").info("daemon started (backend=local)")
        return 0

    monkeypatch.setattr(daemon_module, "main", fake_main)
    _real_configure = cli._configure_daemon_logging
    monkeypatch.setattr(cli, "_configure_daemon_logging", lambda: _real_configure(stream))

    assert cli.main(["daemon"]) == 0
    out = stream.getvalue()
    assert "daemon started (backend=local)" in out
    # Timestamped, so a line in a tab left open overnight says *when*.
    assert out[:2].isdigit() and out[2] == ":"

    # Idempotent: a second configure adds no second handler.
    _real_configure(stream)
    _real_configure(stream)
    assert sum(getattr(h, "_relay_daemon", False) for h in logger.handlers) == 1


def test_other_commands_stay_quiet(capsys, monkeypatch):
    """Turning on INFO is the daemon's business alone.

    `relay ls` prints a table and exits; an INFO line from the store or the
    config loader above that table would be noise, and a script parsing the
    output would choke on it.
    """
    logger = logging.getLogger("relay")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    cli.main(["ls"])
    assert not any(getattr(h, "_relay_daemon", False) for h in logger.handlers)
