"""Tests for relay.config.

Every test runs inside a tmp_path with HOME, XDG_CONFIG_HOME and XDG_DATA_HOME
all redirected there. That is not politeness: `config.init()` writes a file and
`config.load()` reads one, and a test suite that touched the developer's real
~/.config/relay/config.yaml could destroy settings that took a Duo login and
some trial and error to get right.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from relay import config as cfg


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point HOME and both XDG variables at a throwaway directory.

    autouse, so no test can forget it.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    return home


def write_config(text: str) -> Path:
    """Write `text` to the config path and return it."""
    path = cfg.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------


def test_paths_follow_xdg_variables(isolated_home):
    home = isolated_home
    assert cfg.config_path() == home / ".config" / "relay" / "config.yaml"
    assert cfg.data_dir() == home / ".local" / "share" / "relay"
    assert cfg.db_path() == home / ".local" / "share" / "relay" / "relay.db"
    assert cfg.daemon_lock_path() == home / ".local" / "share" / "relay" / "daemon.lock"


def test_paths_fall_back_to_home_when_xdg_unset(isolated_home, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.delenv("XDG_DATA_HOME")
    assert cfg.config_path() == isolated_home / ".config" / "relay" / "config.yaml"
    assert cfg.db_path() == isolated_home / ".local" / "share" / "relay" / "relay.db"


def test_relative_xdg_value_is_ignored(isolated_home, monkeypatch):
    # The XDG spec says a relative path must be ignored; otherwise relay's
    # database would move whenever you cd somewhere else.
    monkeypatch.setenv("XDG_DATA_HOME", "some/relative/path")
    assert cfg.data_dir() == isolated_home / ".local" / "share" / "relay"


def test_paths_are_read_at_call_time_not_import_time(isolated_home, monkeypatch, tmp_path):
    other = tmp_path / "elsewhere"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(other))
    assert cfg.config_path() == other / "relay" / "config.yaml"


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def test_init_writes_a_file_that_load_can_parse():
    path = cfg.init()
    assert path == cfg.config_path()
    assert path.exists()

    # It must be valid YAML as well as loadable by relay.
    assert isinstance(yaml.safe_load(path.read_text()), dict)

    conf = cfg.load()
    assert conf.backend == "local"
    assert conf.ssh_alias is None
    assert conf.remote_root is None
    assert conf.remote_python == "python3"
    assert conf.slurm.grace_seconds == 60
    assert conf.slurm.gres is None
    assert conf.slurm.sbatch_extra == ()
    assert conf.setup == ()


def test_init_output_round_trips_with_every_default_intact():
    # Everything in the starter file except `backend` is commented out, so
    # loading it must produce exactly the dataclass defaults. If a comment and
    # a default ever drift apart, this is the test that notices.
    cfg.init()
    conf = cfg.load()

    assert conf == cfg.Config(
        backend="local",
        ssh_alias=None,
        remote_root=None,
        remote_python="python3",
        ssh_timeout=30,
        local_root=str(cfg.data_dir() / "runs"),
        requeue_on_term="auto",
        preempt_grace=10,
        setup=(),
        slurm=cfg.SlurmConfig(),
        cost=cfg.CostConfig(),
        resume=cfg.ResumeConfig(),
    )
    assert conf.slurm.time is None
    assert conf.slurm.gres is None
    assert conf.slurm.sbatch_extra == ()
    assert conf.cost.gpu_hour_rate is None


def test_starter_config_documents_the_new_keys():
    # The starter file is where a user finds out these settings exist at all.
    text = cfg.STARTER_CONFIG
    for key in ("requeue_on_term", "preempt_grace", "gpu_hour_rate", "grace_seconds"):
        assert key in text
    # The Klone facts a user would otherwise have to discover by losing a run.
    assert "ckpt-all" in text
    assert "DefaultTime=01:00:00" in text
    assert "10 seconds" in text
    # grace_seconds is the time-limit warning now, not the preemption window.
    assert "USR1" in text


def test_starter_config_documents_setup_and_sbatch_extra():
    text = cfg.STARTER_CONFIG
    # setup: the example a user will copy, and the one behaviour they cannot
    # guess - a failing line fails the job instead of running training anyway.
    assert "setup:" in text
    assert "module load" in text
    assert "conda activate" in text
    assert "set -e" in text
    # sbatch_extra: the escape hatch, its shape, and which flags are refused.
    assert "sbatch_extra:" in text
    assert '- "--mem=64G"' in text
    assert '- "--cpus-per-task=8"' in text
    for flag in ("--output", "--signal", "--requeue", "--job-name", "--gres"):
        assert flag in text
    assert "gres:" in text
    # The replaced key must not linger in the file it was removed from.
    assert "sbatch_defaults" not in text


def test_init_creates_missing_parent_directories():
    assert not cfg.config_dir().exists()
    cfg.init()
    assert cfg.config_dir().is_dir()


def test_init_refuses_to_overwrite_without_force():
    path = cfg.init()
    path.write_text("backend: slurm\nssh_alias: klone\nremote_root: /mmfs1/x\n")

    with pytest.raises(cfg.ConfigError) as exc:
        cfg.init()
    assert "already exists" in str(exc.value)
    assert "--force" in str(exc.value)

    # The user's file is untouched.
    assert "klone" in path.read_text()


def test_init_force_overwrites():
    path = cfg.init()
    path.write_text("backend: slurm\nssh_alias: klone\nremote_root: /mmfs1/x\n")

    cfg.init(force=True)
    assert path.read_text() == cfg.STARTER_CONFIG
    assert cfg.load().backend == "local"


def test_starter_config_documents_the_alias_rule():
    # The starter file is the only relay documentation most users will read,
    # so it has to explain why there is no user@host field.
    assert "~/.ssh/config" in cfg.STARTER_CONFIG
    assert "ssh_alias" in cfg.STARTER_CONFIG


# --------------------------------------------------------------------------
# load: missing and unparseable
# --------------------------------------------------------------------------


def test_missing_file_raises_config_not_found():
    with pytest.raises(cfg.ConfigNotFound) as exc:
        cfg.load()
    message = str(exc.value)
    assert "relay init" in message
    assert str(cfg.config_path()) in message


def test_config_not_found_is_a_config_error():
    # The CLI catches ConfigNotFound for exit code 3, but any caller that only
    # cares that config failed should be able to catch the base class.
    assert issubclass(cfg.ConfigNotFound, cfg.ConfigError)
    with pytest.raises(cfg.ConfigError):
        cfg.load()


def test_unparseable_yaml_raises_config_error_not_not_found():
    write_config(
        """\
        backend: local
          ssh_alias: klone
        """
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert not isinstance(exc.value, cfg.ConfigNotFound)
    assert "parse" in str(exc.value)


def test_non_mapping_document_raises():
    write_config("- local\n- slurm\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "block of settings" in str(exc.value)


# --------------------------------------------------------------------------
# load: defaults
# --------------------------------------------------------------------------


def test_empty_file_yields_all_defaults():
    write_config("")
    conf = cfg.load()
    assert conf.backend == "local"
    assert conf.ssh_alias is None
    assert conf.remote_root is None
    assert conf.remote_python == "python3"
    assert conf.local_root == str(cfg.data_dir() / "runs")
    assert conf.requeue_on_term == "auto"
    assert conf.preempt_grace == 10
    assert conf.slurm == cfg.SlurmConfig()
    assert conf.slurm.time is None
    assert conf.cost == cfg.CostConfig()
    assert conf.cost.gpu_hour_rate is None


def test_comments_only_file_yields_defaults():
    write_config("# nothing here yet\n")
    assert cfg.load().backend == "local"


def test_local_root_is_expanded(isolated_home):
    write_config("local_root: ~/scratch/relay-runs\n")
    conf = cfg.load()
    assert conf.local_root == str(isolated_home / "scratch" / "relay-runs")
    assert "~" not in conf.local_root


def test_config_is_frozen():
    write_config("")
    conf = cfg.load()
    with pytest.raises(Exception):
        conf.backend = "slurm"  # type: ignore[misc]


# --------------------------------------------------------------------------
# load: unknown keys and wrong types
# --------------------------------------------------------------------------


def test_unknown_top_level_key_raises():
    write_config("remote_pyton: python3.11\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "remote_pyton" in str(exc.value)


def test_unknown_slurm_key_raises():
    write_config(
        """\
        slurm:
          qos: high
        """
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "qos" in str(exc.value)


def test_unknown_backend_value_raises():
    write_config("backend: kubernetes\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "kubernetes" in str(exc.value)


def test_non_string_scalar_raises():
    write_config("remote_python: 3\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "must be a string" in str(exc.value)


def test_list_valued_key_raises():
    write_config("remote_root:\n  - /mmfs1/a\n  - /mmfs1/b\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load()


def test_empty_string_value_raises():
    write_config('ssh_alias: ""\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "empty" in str(exc.value)


def test_slurm_section_must_be_a_mapping():
    write_config("slurm: gpu-a40\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "slurm" in str(exc.value)


def test_null_slurm_section_is_allowed():
    # `slurm:` with nothing under it parses as None; that is an empty section,
    # not an error.
    write_config("slurm:\n")
    assert cfg.load().slurm == cfg.SlurmConfig()


# --------------------------------------------------------------------------
# load: the ssh alias rule
# --------------------------------------------------------------------------


def test_ssh_alias_with_at_sign_raises():
    write_config("ssh_alias: ahatim@klone.hyak.uw.edu\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "@" in message
    assert "~/.ssh/config" in message


def test_ssh_alias_with_whitespace_raises():
    write_config('ssh_alias: "klone login"\n')
    with pytest.raises(cfg.ConfigError):
        cfg.load()


def test_plain_ssh_alias_is_accepted():
    write_config("ssh_alias: klone\n")
    assert cfg.load().ssh_alias == "klone"


# --------------------------------------------------------------------------
# load: the slurm backend's requirements
# --------------------------------------------------------------------------


def test_slurm_backend_without_ssh_alias_raises():
    write_config("backend: slurm\nremote_root: /mmfs1/gscratch/mlopt/ahatim/relay\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "ssh_alias" in str(exc.value)


def test_slurm_backend_without_remote_root_raises():
    write_config("backend: slurm\nssh_alias: klone\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "remote_root" in str(exc.value)


def test_slurm_backend_missing_both_names_both():
    write_config("backend: slurm\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "ssh_alias" in message and "remote_root" in message


def test_local_backend_does_not_require_cluster_settings():
    write_config("backend: local\n")
    assert cfg.load().backend == "local"


def test_relative_remote_root_raises():
    write_config("backend: slurm\nssh_alias: klone\nremote_root: relay/runs\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "absolute" in str(exc.value)


def test_full_slurm_config_loads():
    write_config(
        """\
        backend: slurm
        ssh_alias: klone
        remote_root: /mmfs1/gscratch/mlopt/ahatim/relay
        remote_python: /mmfs1/gscratch/mlopt/ahatim/venv/bin/python
        requeue_on_term: never
        preempt_grace: 10
        setup:
          - module load cuda/12.4
          - conda activate rl
        slurm:
          account: mlopt
          partition: ckpt-all
          time: "04:00:00"
          grace_seconds: 120
          gres: "gpu:a40:2"
          sbatch_extra:
            - "--mem=32G"
            - "--cpus-per-task=8"
        cost:
          gpu_hour_rate: 1.25
        """
    )
    conf = cfg.load()
    assert conf.backend == "slurm"
    assert conf.ssh_alias == "klone"
    assert conf.remote_root == "/mmfs1/gscratch/mlopt/ahatim/relay"
    assert conf.remote_python.endswith("/venv/bin/python")
    assert conf.requeue_on_term == "never"
    assert conf.preempt_grace == 10
    assert conf.slurm.account == "mlopt"
    assert conf.slurm.partition == "ckpt-all"
    assert conf.slurm.time == "04:00:00"
    assert conf.slurm.grace_seconds == 120
    assert conf.cost.gpu_hour_rate == 1.25
    assert conf.slurm.gres == "gpu:a40:2"
    assert conf.slurm.sbatch_extra == ("--mem=32G", "--cpus-per-task=8")
    assert conf.setup == ("module load cuda/12.4", "conda activate rl")


# --------------------------------------------------------------------------
# load: grace_seconds
# --------------------------------------------------------------------------


def test_grace_seconds_must_be_an_integer():
    write_config("slurm:\n  grace_seconds: soon\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "grace_seconds" in str(exc.value)


def test_grace_seconds_too_short_raises():
    # The sidecar waits grace - 5 seconds for a checkpoint, so a 3 second grace
    # is really a zero second grace.
    write_config("slurm:\n  grace_seconds: 3\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "too short" in str(exc.value)


def test_grace_seconds_boolean_raises():
    # isinstance(True, int) is True in Python, so this is a real trap.
    write_config("slurm:\n  grace_seconds: yes\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load()


# --------------------------------------------------------------------------
# load: setup
# --------------------------------------------------------------------------


def test_setup_defaults_to_empty_when_omitted():
    write_config("backend: local\n")
    assert cfg.load().setup == ()


def test_setup_keeps_its_order():
    # Order is the whole point: `conda activate` after `module load`, not
    # before it.
    write_config(
        """\
        setup:
          - module load cuda/12.4
          - conda activate rl
          - export OMP_NUM_THREADS=8
        """
    )
    assert cfg.load().setup == (
        "module load cuda/12.4",
        "conda activate rl",
        "export OMP_NUM_THREADS=8",
    )


def test_setup_null_is_an_empty_list():
    # `setup:` with nothing under it parses as None, which means "not given",
    # the same as every other optional key.
    write_config("setup:\n")
    assert cfg.load().setup == ()


def test_setup_is_a_tuple_so_it_cannot_be_appended_to():
    # Config is frozen; a list here would be a mutable hole in that.
    write_config("setup:\n  - module load cuda\n")
    assert isinstance(cfg.load().setup, tuple)


def test_setup_must_be_a_list():
    # The obvious mistake: writing the whole thing as one shell string.
    write_config("setup: module load cuda && conda activate rl\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "setup" in message
    assert "list" in message


def test_setup_rejects_a_non_string_entry():
    write_config("setup:\n  - module load cuda\n  - 3\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "setup" in message
    assert "int" in message


def test_setup_rejects_an_empty_entry():
    write_config('setup:\n  - "module load cuda"\n  - ""\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "empty" in str(exc.value)


def test_setup_contents_are_not_parsed():
    # relay has no idea which modules your cluster has, so it does not look
    # inside these lines at all. Nonsense loads fine and fails on the cluster,
    # which is the only place that can tell.
    write_config(
        """\
        setup:
          - "source /nonexistent/env.sh && frobnicate --hard"
        """
    )
    assert cfg.load().setup == ("source /nonexistent/env.sh && frobnicate --hard",)


# --------------------------------------------------------------------------
# load: slurm.gres
# --------------------------------------------------------------------------


def test_gres_defaults_to_none():
    write_config("slurm:\n  partition: ckpt-all\n")
    assert cfg.load().slurm.gres is None


@pytest.mark.parametrize("value", ["gpu:1", "gpu:a40:2", "gpu:l40s:4"])
def test_gres_accepts_slurm_spellings(value):
    write_config(f'slurm:\n  gres: "{value}"\n')
    assert cfg.load().slurm.gres == value


def test_gres_must_be_a_string():
    # `gres: 1` is the obvious way to ask for one GPU and the wrong one;
    # Slurm wants "gpu:1".
    write_config("slurm:\n  gres: 1\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "must be a string" in str(exc.value)


def test_gres_empty_string_raises():
    write_config('slurm:\n  gres: ""\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "empty" in str(exc.value)


# --------------------------------------------------------------------------
# load: slurm.sbatch_extra
# --------------------------------------------------------------------------


def test_sbatch_extra_defaults_to_empty_when_omitted():
    write_config("slurm:\n  partition: ckpt-all\n")
    assert cfg.load().slurm.sbatch_extra == ()


def test_sbatch_extra_keeps_its_order():
    write_config(
        """\
        slurm:
          sbatch_extra:
            - "--mem=64G"
            - "--cpus-per-task=8"
            - "--nodes=1"
        """
    )
    assert cfg.load().slurm.sbatch_extra == (
        "--mem=64G",
        "--cpus-per-task=8",
        "--nodes=1",
    )


def test_sbatch_extra_null_is_an_empty_list():
    write_config("slurm:\n  sbatch_extra:\n")
    assert cfg.load().slurm.sbatch_extra == ()


def test_sbatch_extra_is_a_tuple():
    write_config('slurm:\n  sbatch_extra:\n    - "--mem=64G"\n')
    assert isinstance(cfg.load().slurm.sbatch_extra, tuple)


def test_sbatch_extra_must_be_a_list():
    # The old shape was a mapping, so this is exactly what an upgrading user
    # is likely to leave behind under the new name.
    write_config('slurm:\n  sbatch_extra:\n    mem: "64G"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "sbatch_extra" in message
    assert "list" in message


def test_sbatch_extra_rejects_a_non_string_entry():
    write_config('slurm:\n  sbatch_extra:\n    - "--mem=64G"\n    - 8\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "sbatch_extra" in message
    assert "int" in message


def test_sbatch_extra_rejects_an_empty_entry():
    write_config('slurm:\n  sbatch_extra:\n    - "--mem=64G"\n    - "   "\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "empty" in str(exc.value)


def test_sbatch_extra_contents_are_not_validated_here():
    # config.py is a *shape* validator. "--output=x" is a directive relay
    # refuses, but the refusal lives in relay.backends.base.validate_sbatch_extra
    # and the CLI runs it at submit. Keeping it out of load() means one list of
    # forbidden flags in the repo, and a config holding a bad directive still
    # loads, so `relay ls` keeps working while the user fixes it.
    write_config('slurm:\n  sbatch_extra:\n    - "--output=/tmp/x.out"\n')
    assert cfg.load().slurm.sbatch_extra == ("--output=/tmp/x.out",)


def test_sbatch_defaults_is_now_an_unknown_key():
    # The old key. A user upgrading relay must be told what to write instead,
    # not just that their working config has become invalid.
    write_config('slurm:\n  sbatch_defaults:\n    mem: "32G"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "sbatch_defaults" in message
    assert "sbatch_extra" in message
    assert "--mem=32G" in message


# --------------------------------------------------------------------------
# load: requeue_on_term
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["auto", "always", "never"])
def test_requeue_on_term_accepts_each_policy(value):
    write_config(f"requeue_on_term: {value}\n")
    assert cfg.load().requeue_on_term == value


def test_requeue_on_term_rejects_an_unknown_policy():
    write_config("requeue_on_term: sometimes\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "sometimes" in message
    # The message must list the policies that do work.
    assert "auto" in message and "always" in message and "never" in message


def test_requeue_on_term_rejects_a_non_string():
    write_config("requeue_on_term: 3\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "must be a string" in str(exc.value)


def test_requeue_on_term_rejects_a_boolean():
    # `requeue_on_term: no` is YAML for False, which is not one of the three
    # policies even though it reads like one.
    write_config("requeue_on_term: no\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load()


def test_requeue_on_term_null_is_the_default():
    write_config("requeue_on_term:\n")
    assert cfg.load().requeue_on_term == "auto"


# --------------------------------------------------------------------------
# load: preempt_grace
# --------------------------------------------------------------------------


def test_preempt_grace_accepts_an_integer():
    write_config("preempt_grace: 30\n")
    assert cfg.load().preempt_grace == 30


def test_preempt_grace_null_is_the_default():
    write_config("preempt_grace:\n")
    assert cfg.load().preempt_grace == 10


def test_preempt_grace_must_be_an_integer():
    write_config("preempt_grace: ten\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "preempt_grace" in str(exc.value)


def test_preempt_grace_rejects_a_float():
    write_config("preempt_grace: 10.5\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load()


def test_preempt_grace_below_three_raises():
    # The sidecar waits preempt_grace - 2 seconds, floored at 1. At 2 the floor
    # takes over, so the configured number stops meaning anything.
    write_config("preempt_grace: 2\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "too short" in str(exc.value)


def test_preempt_grace_three_is_the_lowest_accepted():
    write_config("preempt_grace: 3\n")
    assert cfg.load().preempt_grace == 3


def test_preempt_grace_boolean_raises():
    # isinstance(True, int) is True in Python, so `preempt_grace: yes` would
    # sail through a naive int check as a one-second window.
    write_config("preempt_grace: yes\n")
    with pytest.raises(cfg.ConfigError):
        cfg.load()


# --------------------------------------------------------------------------
# load: ssh_timeout
# --------------------------------------------------------------------------


def test_ssh_timeout_defaults_to_thirty_when_omitted():
    write_config("backend: local\n")
    assert cfg.load().ssh_timeout == 30


def test_ssh_timeout_null_is_the_default():
    write_config("ssh_timeout:\n")
    assert cfg.load().ssh_timeout == 30


def test_ssh_timeout_default_comes_from_relay_ssh():
    # One number, defined next to the measurement that justifies it. If ssh.py
    # ever changes it, config must follow rather than keep its own copy.
    from relay import ssh

    assert cfg.DEFAULT_SSH_TIMEOUT == int(ssh.DEFAULT_SSH_TIMEOUT)
    assert cfg.Config().ssh_timeout == int(ssh.DEFAULT_SSH_TIMEOUT)


def test_ssh_timeout_accepts_an_integer():
    write_config("ssh_timeout: 45\n")
    assert cfg.load().ssh_timeout == 45


def test_ssh_timeout_below_the_floor_raises():
    # A round trip through a live ControlMaster on Klone measured 4 to 5
    # seconds, all of it the login node starting a shell. At 4 relay would time
    # out commands that were about to succeed and call a healthy cluster dead,
    # so the error has to explain where the floor comes from.
    write_config("ssh_timeout: 4\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "ssh_timeout" in message
    assert "floor" in message
    assert "round trip" in message
    assert "login node" in message


def test_ssh_timeout_five_is_the_lowest_accepted():
    write_config("ssh_timeout: 5\n")
    assert cfg.load().ssh_timeout == 5


def test_ssh_timeout_boolean_raises():
    # isinstance(True, int) is True, so `ssh_timeout: yes` would sail through a
    # naive int check as a one-second budget.
    write_config("ssh_timeout: yes\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "whole number" in str(exc.value)


def test_ssh_timeout_rejects_a_float():
    # 30.5 means the user thinks this is a network latency. It is not, and
    # rounding it silently would leave them believing it.
    write_config("ssh_timeout: 30.5\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "ssh_timeout" in str(exc.value)


def test_ssh_timeout_rejects_a_string():
    write_config('ssh_timeout: "30s"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "whole number" in str(exc.value)


def test_starter_config_documents_ssh_timeout():
    # A user only raises this after doctor complains, so the file has to say
    # what the number actually measures.
    text = cfg.STARTER_CONFIG
    assert "ssh_timeout" in text
    assert "login node" in text


# --------------------------------------------------------------------------
# load: slurm.time
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "30",  # MM
        "1440",  # MM, more than two digits
        "30:00",  # MM:SS
        "04:00:00",  # HH:MM:SS
        "4:30:00",  # HH:MM:SS, single-digit hours
        "2-12",  # D-HH
        "2-12:30",  # D-HH:MM
        "2-12:30:45",  # D-HH:MM:SS
    ],
)
def test_slurm_time_accepts_every_slurm_form(value):
    write_config(f'slurm:\n  time: "{value}"\n')
    assert cfg.load().slurm.time == value


@pytest.mark.parametrize(
    "value",
    [
        "4h",  # not Slurm syntax at all
        "forever",
        "1:2:3:4",  # one field too many
        "2-12-30",  # days separator used twice
        "-30",  # negative
        "04:00:00:00",
        "2-",  # trailing separator, no hours
        ":30",  # leading separator
        "4 hours",
    ],
)
def test_slurm_time_rejects_anything_else(value):
    write_config(f'slurm:\n  time: "{value}"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "time" in message
    # The message has to show the forms that do work, or the user is stuck.
    assert "HH:MM:SS" in message


def test_slurm_time_must_be_a_string():
    # `time: 60` is a YAML integer. Unquoted Slurm durations are a trap in
    # general: YAML reads 4:00:00 as sexagesimal and hands us a number.
    write_config("slurm:\n  time: 60\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "must be a string" in str(exc.value)


def test_slurm_time_is_optional():
    # Not required in config: `relay submit --time` can supply it instead.
    write_config("slurm:\n  partition: ckpt-all\n")
    assert cfg.load().slurm.time is None


# --------------------------------------------------------------------------
# load: the cost section
# --------------------------------------------------------------------------


def test_cost_section_defaults_to_no_rate():
    write_config("")
    assert cfg.load().cost.gpu_hour_rate is None


def test_null_cost_section_is_allowed():
    write_config("cost:\n")
    assert cfg.load().cost == cfg.CostConfig()


def test_gpu_hour_rate_accepts_a_float():
    write_config("cost:\n  gpu_hour_rate: 1.25\n")
    assert cfg.load().cost.gpu_hour_rate == 1.25


def test_gpu_hour_rate_accepts_an_integer_and_stores_a_float():
    write_config("cost:\n  gpu_hour_rate: 2\n")
    rate = cfg.load().cost.gpu_hour_rate
    assert rate == 2.0
    assert isinstance(rate, float)


def test_gpu_hour_rate_null_means_no_rate():
    write_config("cost:\n  gpu_hour_rate:\n")
    assert cfg.load().cost.gpu_hour_rate is None


def test_gpu_hour_rate_zero_raises():
    # Zero is not "free", it is "I did not mean to set this". Deleting the key
    # is how you say you want no dollar figures.
    write_config("cost:\n  gpu_hour_rate: 0\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "greater than zero" in str(exc.value)


def test_gpu_hour_rate_negative_raises():
    write_config("cost:\n  gpu_hour_rate: -1.5\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "greater than zero" in str(exc.value)


def test_gpu_hour_rate_boolean_raises():
    write_config("cost:\n  gpu_hour_rate: yes\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "gpu_hour_rate" in str(exc.value)


def test_gpu_hour_rate_string_raises():
    # "$1.25" is the obvious thing to type and the obvious thing to reject:
    # relay never parses currency symbols.
    write_config('cost:\n  gpu_hour_rate: "$1.25"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "gpu_hour_rate" in str(exc.value)


def test_unknown_cost_key_raises():
    write_config("cost:\n  cpu_hour_rate: 0.05\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "cpu_hour_rate" in str(exc.value)


def test_cost_section_must_be_a_mapping():
    write_config("cost: 1.25\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "cost" in str(exc.value)


# --------------------------------------------------------------------------
# load: the resume section
# --------------------------------------------------------------------------


def test_resume_section_defaults_to_empty():
    write_config("")
    conf = cfg.load()
    assert conf.resume == cfg.ResumeConfig()
    assert conf.resume.checkpoint_glob is None
    assert conf.resume.arg is None


def test_null_resume_section_is_allowed():
    # `resume:` with nothing under it parses as None, which means "not given",
    # exactly like the slurm and cost sections.
    write_config("resume:\n")
    assert cfg.load().resume == cfg.ResumeConfig()


def test_resume_with_both_keys_loads():
    write_config(
        """\
        resume:
          checkpoint_glob: "checkpoints/*.pt"
          arg: "--resume {path}"
        """
    )
    resume = cfg.load().resume
    assert resume.checkpoint_glob == "checkpoints/*.pt"
    assert resume.arg == "--resume {path}"


def test_resume_glob_may_use_double_star():
    # `**` is the whole reason a glob is more useful than a directory name:
    # checkpoints often land one directory down, under the run's own name.
    write_config(
        """\
        resume:
          checkpoint_glob: "runs/**/ckpt_*.pkl"
          arg: "--resume={path}"
        """
    )
    resume = cfg.load().resume
    assert resume.checkpoint_glob == "runs/**/ckpt_*.pkl"
    assert resume.arg == "--resume={path}"


def test_resume_glob_without_arg_raises():
    # A glob with nothing to do with what it finds. The message must name the
    # key that is missing, not just say something is wrong.
    write_config('resume:\n  checkpoint_glob: "checkpoints/*.pt"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "arg" in message
    assert "checkpoints/*.pt" in message


def test_resume_arg_without_glob_raises():
    write_config('resume:\n  arg: "--resume {path}"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "checkpoint_glob" in message


def test_resume_arg_without_the_path_placeholder_raises():
    # "--resume" alone would be appended verbatim and name no file, so the
    # script would either fail to parse it or silently start from scratch.
    write_config(
        'resume:\n  checkpoint_glob: "checkpoints/*.pt"\n  arg: "--resume"\n'
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "{path}" in message
    assert "arg" in message


def test_resume_checkpoint_glob_must_be_a_string():
    write_config('resume:\n  checkpoint_glob: 3\n  arg: "--resume {path}"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "must be a string" in message
    assert "checkpoint_glob" in message


def test_resume_arg_must_be_a_string():
    write_config('resume:\n  checkpoint_glob: "checkpoints/*.pt"\n  arg: 3\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "must be a string" in message
    assert "arg" in message


def test_resume_empty_string_raises():
    write_config('resume:\n  checkpoint_glob: ""\n  arg: "--resume {path}"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "empty" in str(exc.value)


def test_unknown_resume_key_raises():
    write_config(
        'resume:\n  checkpoint_glob: "checkpoints/*.pt"\n'
        '  arg: "--resume {path}"\n  every: 100\n'
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "every" in message
    assert "checkpoint_glob" in message


def test_resume_section_must_be_a_mapping():
    write_config('resume: "checkpoints/*.pt"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "resume" in str(exc.value)


def test_resume_config_is_frozen():
    write_config("")
    resume = cfg.load().resume
    with pytest.raises(Exception):
        resume.arg = "--resume {path}"  # type: ignore[misc]


def test_starter_config_documents_resume():
    # The starter file is where a user learns this section exists, and it has
    # to answer the two questions they will have: what goes in arg, and what
    # to do if their script does not checkpoint at all.
    text = cfg.STARTER_CONFIG
    assert "resume:" in text
    assert "checkpoint_glob" in text
    assert "{path}" in text
    assert "relay_ckpt" in text
    # The per-run overrides, so the user knows they are not stuck with this.
    assert "--checkpoint-glob" in text
    assert "--resume-arg" in text


# --------------------------------------------------------------------------
# load with an explicit path
# --------------------------------------------------------------------------


def test_load_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "somewhere" / "relay.yaml"
    path.parent.mkdir()
    path.write_text("backend: local\nremote_python: python3.11\n")
    assert cfg.load(path).remote_python == "python3.11"


def test_init_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "custom" / "config.yaml"
    assert cfg.init(path=path) == path
    assert cfg.load(path).backend == "local"
