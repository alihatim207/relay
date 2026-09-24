"""Tests for relay.config.

Every test runs inside a tmp_path with HOME, XDG_CONFIG_HOME and XDG_DATA_HOME
all redirected there. That is not politeness: `config.init()` writes a file and
`config.load()` reads one, and a test suite that touched the developer's real
~/.config/relay/config.yaml could destroy settings that took a Duo login and
some trial and error to get right.
"""

from __future__ import annotations

import logging
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
    write_config("ssh_alias: you@klone.hyak.uw.edu\n")
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
    write_config("backend: slurm\nremote_root: /mmfs1/gscratch/yourgroup/you/relay\n")
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
        remote_root: /mmfs1/gscratch/yourgroup/you/relay
        remote_python: /mmfs1/gscratch/yourgroup/you/venv/bin/python
        requeue_on_term: never
        preempt_grace: 10
        setup:
          - module load cuda/12.4
          - conda activate rl
        slurm:
          account: yourgroup
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
    assert conf.remote_root == "/mmfs1/gscratch/yourgroup/you/relay"
    assert conf.remote_python.endswith("/venv/bin/python")
    assert conf.requeue_on_term == "never"
    assert conf.preempt_grace == 10
    assert conf.slurm.account == "yourgroup"
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
# load: the daemon section
# --------------------------------------------------------------------------
#
# Six polling intervals, all seconds, all optional. The interesting one is
# `scheduler_interval_min`, which is the only number in relay's config that is
# silently corrected rather than accepted or rejected: `squeue` asks slurmctld,
# which every user on the cluster shares, so relay holds itself to a floor no
# config file can talk it below.


def test_daemon_section_defaults_when_absent():
    """No `daemon:` section is the normal case, and it must cost nothing.

    Almost nobody should ever write this section. If the defaults here and the
    ones in DaemonConfig ever drift apart, a user who never touched the file
    would get different polling from a user who pasted the commented-out block
    back in unchanged.
    """
    write_config("backend: local\n")
    conf = cfg.load()

    assert conf.daemon == cfg.DaemonConfig()
    assert conf.daemon.tail_interval_active == 2.0
    assert conf.daemon.tail_interval_queued == 30.0
    assert conf.daemon.tail_interval_idle == 60.0
    assert conf.daemon.scheduler_interval_min == 60.0
    assert conf.daemon.scheduler_interval_max == 300.0
    assert conf.daemon.usage_interval == 300.0
    # Floats, not ints: everything downstream does arithmetic on a monotonic
    # clock, and a stray int here would be the one value that behaves
    # differently under division.
    for name in cfg._DAEMON_KEYS:
        assert isinstance(getattr(conf.daemon, name), float)


def test_daemon_section_accepts_every_key():
    """A whole section written out in the obvious way: plain integers.

    YAML hands these over as ints; relay stores floats, so the values have to
    survive the conversion rather than being rejected for having the wrong
    type.
    """
    write_config(
        """\
        daemon:
          tail_interval_active: 1
          tail_interval_queued: 15
          tail_interval_idle: 45
          scheduler_interval_min: 20
          scheduler_interval_max: 200
          usage_interval: 600
        """
    )
    daemon = cfg.load().daemon

    assert daemon == cfg.DaemonConfig(
        tail_interval_active=1.0,
        tail_interval_queued=15.0,
        tail_interval_idle=45.0,
        scheduler_interval_min=20.0,
        scheduler_interval_max=200.0,
        usage_interval=600.0,
    )
    for name in cfg._DAEMON_KEYS:
        assert isinstance(getattr(daemon, name), float)


def test_daemon_accepts_a_fractional_interval():
    """Unlike ssh_timeout and preempt_grace, these take floats.

    Those two are budgets measured against a real cluster, where a fraction
    means the user has misunderstood the number. A polling interval is not:
    half a second is a coherent thing to ask for while watching a run, and the
    tests themselves want sub-second intervals to finish quickly.
    """
    write_config("daemon:\n  tail_interval_active: 0.5\n")
    assert cfg.load().daemon.tail_interval_active == 0.5


def test_daemon_partial_section_keeps_the_other_defaults():
    """Setting one interval must not reset the five you did not mention."""
    write_config("daemon:\n  usage_interval: 900\n")
    daemon = cfg.load().daemon

    assert daemon.usage_interval == 900.0
    assert daemon.tail_interval_active == 2.0
    assert daemon.tail_interval_queued == 30.0
    assert daemon.tail_interval_idle == 60.0
    assert daemon.scheduler_interval_min == 60.0
    assert daemon.scheduler_interval_max == 300.0


def test_unknown_daemon_key_raises():
    """A plausible-sounding key that relay ignores is worse than an error.

    `poll:` reads like it ought to work. Silently dropping it would leave the
    user believing they had slowed relay down when they had not.
    """
    write_config("daemon:\n  poll: 5\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "poll" in message
    assert "daemon" in message
    # And the keys that do exist, or the user has nowhere to go.
    assert "tail_interval_active" in message


def test_daemon_interval_rejects_a_boolean():
    """`tail_interval_active: yes` is YAML for True, and True is an int.

    isinstance(True, int) is True in Python, so a naive number check would turn
    this into a one-second poll - the single most aggressive setting available,
    arrived at by writing something that looks like it means "on".
    """
    write_config("daemon:\n  tail_interval_active: yes\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "tail_interval_active" in message
    assert "number of seconds" in message


def test_daemon_interval_rejects_a_string():
    """Seconds are a bare number here; relay parses no duration suffixes."""
    write_config('daemon:\n  tail_interval_active: "2s"\n')
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    assert "tail_interval_active" in str(exc.value)


def test_daemon_interval_rejects_zero():
    """Zero is a loop with no sleep in it, which is a busy wait, not a setting."""
    write_config("daemon:\n  tail_interval_idle: 0\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "tail_interval_idle" in message
    assert "positive" in message


def test_daemon_interval_rejects_a_negative_number():
    write_config("daemon:\n  usage_interval: -5\n")
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "usage_interval" in message
    assert "positive" in message


def test_scheduler_interval_min_below_the_floor_is_clamped_with_a_warning(caplog):
    """Too-fast scheduler polling is corrected, not refused.

    Every other bad number in this file is an error. This one is not, and the
    asymmetry is deliberate: refusing to start means the user sees nothing at
    all about their running jobs, while clamping means they see everything a
    few seconds later than they asked. The warning has to name the floor and
    the file, because the point is that the user goes and edits the number.
    """
    caplog.set_level(logging.WARNING, logger="relay.config")
    write_config("daemon:\n  scheduler_interval_min: 5\n")

    conf = cfg.load()
    assert conf.daemon.scheduler_interval_min == 10.0
    assert conf.daemon.scheduler_interval_min == cfg.MIN_SCHEDULER_INTERVAL

    records = [r for r in caplog.records if r.name == "relay.config"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert records[0].levelno == logging.WARNING
    assert "10" in message
    assert "floor" in message
    assert str(cfg.config_path()) in message
    # And why there is a floor at all, so the clamp does not read as arbitrary.
    assert "slurmctld" in message


def test_scheduler_interval_min_at_the_floor_does_not_warn(caplog):
    """Exactly at the floor is a legitimate setting, not a near miss."""
    caplog.set_level(logging.WARNING, logger="relay.config")
    write_config("daemon:\n  scheduler_interval_min: 10\n")

    assert cfg.load().daemon.scheduler_interval_min == 10.0
    assert [r for r in caplog.records if r.name == "relay.config"] == []


def test_scheduler_interval_max_below_min_raises():
    """A ceiling below the floor it backs off from is incoherent.

    The max is where the poll stretches *towards* while nothing changes, so a
    max under the min describes a backoff that goes backwards. Unlike the
    floor, this one cannot be silently fixed - relay does not know which of the
    two numbers the user meant.
    """
    write_config(
        """\
        daemon:
          scheduler_interval_min: 30
          scheduler_interval_max: 20
        """
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "scheduler_interval_max" in message
    assert "scheduler_interval_min" in message


def test_scheduler_max_is_compared_against_the_clamped_min(caplog):
    """The clamp happens first, so it can be what pushes the min above the max.

    Written on its own, `min: 5, max: 8` looks consistent: 8 is comfortably
    above 5. But the min is clamped up to 10 before the two are compared, so
    the pair becomes 10 and 8 and the comparison fails. That ordering is worth
    pinning down, because the alternative - compare first, clamp after - would
    accept this file and leave the daemon with a max below its own min.
    """
    caplog.set_level(logging.WARNING, logger="relay.config")
    write_config(
        """\
        daemon:
          scheduler_interval_min: 5
          scheduler_interval_max: 8
        """
    )
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "scheduler_interval_max" in message
    # The message quotes the clamped minimum (10), not the 5 in the file, so
    # the user can see what the comparison was actually made against.
    assert "10" in message


@pytest.mark.parametrize(
    "text",
    [
        "daemon:\n  - 1\n  - 2\n",  # a list
        "daemon: fast\n",  # a bare string
    ],
)
def test_daemon_section_must_be_a_mapping(text):
    """`daemon: fast` is the shape a hurried user reaches for first."""
    write_config(text)
    with pytest.raises(cfg.ConfigError) as exc:
        cfg.load()
    message = str(exc.value)
    assert "daemon" in message
    assert "block of settings" in message


def test_null_daemon_section_is_allowed():
    """`daemon:` with nothing under it parses as None.

    That means "not given", exactly as it does for the slurm, cost and resume
    sections. A user who comments out every line but the heading has not
    written an error.
    """
    write_config("daemon:\n")
    assert cfg.load().daemon == cfg.DaemonConfig()


def test_daemon_is_an_accepted_top_level_key():
    """A file holding nothing but a daemon section is a complete config."""
    write_config("daemon:\n  tail_interval_active: 3\n")
    conf = cfg.load()
    assert conf.backend == "local"
    assert conf.daemon.tail_interval_active == 3.0
    assert "daemon" in cfg._TOP_LEVEL_KEYS


def test_daemon_config_is_frozen():
    write_config("")
    daemon = cfg.load().daemon
    with pytest.raises(Exception):
        daemon.tail_interval_active = 0.1  # type: ignore[misc]


# --------------------------------------------------------------------------
# clamp_scheduler_interval, on its own
# --------------------------------------------------------------------------
#
# Public and called from two places - config.load() and `relay daemon`'s
# flags - because a limit enforced on one of two routes in is not a limit.


def test_clamp_leaves_an_acceptable_interval_alone(caplog):
    caplog.set_level(logging.WARNING, logger="relay.config")
    assert cfg.clamp_scheduler_interval(30.0, source="x") == 30.0
    assert [r for r in caplog.records if r.name == "relay.config"] == []


def test_clamp_names_its_source_in_the_warning(caplog):
    """The warning has to say which number to go and change.

    The same clamp catches a config key and a command-line flag, and telling
    someone who typed `--scheduler-interval-min 1` to go and edit their config
    file would send them to the wrong place.
    """
    caplog.set_level(logging.WARNING, logger="relay.config")
    assert cfg.clamp_scheduler_interval(1, source="--flag") == 10.0

    records = [r for r in caplog.records if r.name == "relay.config"]
    assert len(records) == 1
    assert "--flag" in records[0].getMessage()


def test_clamp_returns_the_floor_itself_unchanged(caplog):
    caplog.set_level(logging.WARNING, logger="relay.config")
    assert cfg.clamp_scheduler_interval(
        cfg.MIN_SCHEDULER_INTERVAL, source="x"
    ) == cfg.MIN_SCHEDULER_INTERVAL
    assert [r for r in caplog.records if r.name == "relay.config"] == []


# --------------------------------------------------------------------------
# init: the daemon block in the starter file
# --------------------------------------------------------------------------


def test_starter_config_documents_the_daemon_section():
    """The starter file is where a user learns this section exists at all.

    Every key commented out, so that the file's contents and this module's
    defaults cannot drift: an omitted key is a default key.
    """
    text = cfg.STARTER_CONFIG
    assert "# daemon:" in text
    for key in sorted(cfg._DAEMON_KEYS):
        assert f"#   {key}:" in text
    # The two things a user cannot guess: that the floor exists, and that
    # metrics stay live regardless because they come from the tail.
    assert "10 seconds" in text
    assert "clamped" in text


def test_starter_config_daemon_block_loads_as_defaults():
    """Init, then load: the commented block must leave every default in place."""
    cfg.init()
    assert cfg.load().daemon == cfg.DaemonConfig()


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
