"""Tests for relay_ckpt, the checkpoint helper training scripts import.

Everything here runs in-process against a `Checkpointer` built with an explicit
directory (and, where timing matters, an injectable clock), so no test depends
on the real clock, the real environment, or a GPU.

torch and jax are not installed in CI and are not needed: `relay_ckpt` decides
what to do by looking in `sys.modules`, so a fake module inserted there
exercises exactly the branch the real one would.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import pickle
import signal
import sys
import time
import types
from pathlib import Path

import pytest

from relay import relay_ckpt

MODULE_PATH = Path(relay_ckpt.__file__)


# --------------------------------------------------------------------------
# Fakes
#
# FakeTensor lives at module level, not inside a test, because pickle stores a
# class by its import path -- a class defined inside a function cannot be
# pickled, and several tests pickle one of these.
# --------------------------------------------------------------------------


class FakeTensor:
    """Stands in for torch.Tensor: the only thing we care about is .cpu()."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def cpu(self) -> "FakeTensor":
        return FakeTensor("cpu")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeTensor) and other.device == self.device

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeTensor({self.device!r})"


def fake_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.Tensor = FakeTensor
    return module


def fake_jax(calls: list) -> types.ModuleType:
    module = types.ModuleType("jax")

    def device_get(state):
        calls.append(state)
        return state

    module.device_get = device_get
    return module


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_module_state():
    """Keep the module's global instance and pytest's SIGTERM handler intact.

    `Checkpointer` installs a SIGTERM handler on first use. Without restoring
    it, one test would leave its handler (and its dead instance) installed for
    the rest of the session.
    """
    previous_handler = signal.getsignal(signal.SIGTERM)
    relay_ckpt._reset_default()
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        relay_ckpt._reset_default()


@pytest.fixture
def ckpt_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


def fake_clock(start: float = 0.0):
    """A clock whose value the test sets by hand."""
    holder = {"now": start}

    def clock() -> float:
        return holder["now"]

    clock.advance = lambda seconds: holder.__setitem__("now", holder["now"] + seconds)
    return clock


def wait_for_flag(checkpointer, timeout: float = 2.0) -> None:
    """Signals are delivered between bytecodes; give the handler a moment."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if checkpointer.term_requested:
            return
        time.sleep(0.01)
    raise AssertionError("SIGTERM handler never set the flag")


def marker_lines(text: str) -> list[dict]:
    return [
        json.loads(line[len("##relay## ") :])
        for line in text.splitlines()
        if line.startswith("##relay## ")
    ]


# --------------------------------------------------------------------------
# The stdlib-only rule
# --------------------------------------------------------------------------


def test_relay_ckpt_imports_only_stdlib():
    """It is copied to the cluster next to sidecar.py, so it has no deps.

    Same AST check the sidecar gets, and for the same reason: there is no pip
    install on a compute node. AST-parsing catches an import hidden inside a
    function, which importing the module and hoping would not.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pytest.fail(
                    "relay_ckpt.py uses a relative import; it must be self-contained"
                )
            if node.module:
                imported.add(node.module.split(".")[0])

    imported.discard("__future__")

    non_stdlib = sorted(imported - sys.stdlib_module_names)
    assert non_stdlib == [], f"relay_ckpt.py imports non-stdlib modules: {non_stdlib}"
    assert "relay" not in imported
    # The expensive ones, called out explicitly: importing torch costs seconds
    # and importing jax can claim a GPU.
    assert "torch" not in imported
    assert "jax" not in imported


def test_import_does_not_pull_in_torch_or_jax():
    """Loading a fresh copy of the module must not import either framework."""
    before = set(sys.modules)
    assert "torch" not in before and "jax" not in before

    spec = importlib.util.spec_from_file_location("relay_ckpt_fresh", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        assert "torch" not in sys.modules
        assert "jax" not in sys.modules
    finally:
        sys.modules.pop("relay_ckpt_fresh", None)


# --------------------------------------------------------------------------
# restore with nothing there
# --------------------------------------------------------------------------


def test_restore_calls_a_callable_default(ckpt_dir: Path):
    calls = []

    def build():
        calls.append(1)
        return {"weights": 0}

    checkpointer = relay_ckpt.Checkpointer(directory=ckpt_dir)
    state = checkpointer.restore(build)

    assert state == {"weights": 0}
    assert calls == [1], "the factory is called exactly once, and only when needed"
    assert checkpointer.step == 0


def test_restore_returns_a_plain_default_as_is(ckpt_dir: Path):
    sentinel = {"weights": 7}
    checkpointer = relay_ckpt.Checkpointer(directory=ckpt_dir)

    assert checkpointer.restore(sentinel) is sentinel
    assert checkpointer.step == 0


# --------------------------------------------------------------------------
# tick, saving, and restoring the counter
# --------------------------------------------------------------------------


def test_tick_counts_and_saves_on_every_steps(ckpt_dir: Path):
    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=2, keep=10
    )
    checkpointer.restore(lambda: {"loss": 1.0})

    checkpointer.tick({"loss": 0.9})
    assert checkpointer.step == 1
    assert list(ckpt_dir.glob("*.pkl")) == [], "no save before the interval"

    checkpointer.tick({"loss": 0.8})
    assert checkpointer.step == 2
    assert [p.name for p in sorted(ckpt_dir.glob("*.pkl"))] == ["ckpt_000000002.pkl"]

    checkpointer.tick({"loss": 0.7})
    checkpointer.tick({"loss": 0.6})
    assert [p.name for p in sorted(ckpt_dir.glob("*.pkl"))] == [
        "ckpt_000000002.pkl",
        "ckpt_000000004.pkl",
    ]


def test_a_fresh_checkpointer_restores_state_and_step(ckpt_dir: Path):
    """This is the preemption path in miniature: new process, same directory."""
    first = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1
    )
    first.restore(lambda: {"loss": 1.0})
    for loss in (0.9, 0.8, 0.7):
        first.tick({"loss": loss})

    second = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1
    )
    state = second.restore(lambda: pytest.fail("should not build a fresh state"))

    assert state == {"loss": 0.7}
    assert second.step == 3, "the counter continues rather than restarting"

    second.tick({"loss": 0.6})
    assert second.step == 4
    assert (ckpt_dir / "ckpt_000000004.pkl").exists()


def test_periodic_save_follows_the_injected_clock(ckpt_dir: Path):
    clock = fake_clock()
    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=10, every_steps=None, clock=clock
    )
    checkpointer.restore(lambda: {"loss": 1.0})

    for _ in range(5):
        checkpointer.tick({"loss": 0.5})
    assert list(ckpt_dir.glob("*.pkl")) == [], "nothing saved before ten minutes"

    clock.advance(9 * 60)
    checkpointer.tick({"loss": 0.4})
    assert list(ckpt_dir.glob("*.pkl")) == [], "nine minutes is still not ten"

    clock.advance(2 * 60)
    checkpointer.tick({"loss": 0.3})
    assert [p.name for p in ckpt_dir.glob("*.pkl")] == ["ckpt_000000007.pkl"]

    # The timer resets on save, so the next tick does not save again.
    checkpointer.tick({"loss": 0.2})
    assert [p.name for p in ckpt_dir.glob("*.pkl")] == ["ckpt_000000007.pkl"]


# --------------------------------------------------------------------------
# A corrupt newest checkpoint
# --------------------------------------------------------------------------


def test_corrupt_newest_checkpoint_falls_back_with_a_warning(
    ckpt_dir: Path, capsys: pytest.CaptureFixture
):
    writer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1, keep=2
    )
    writer.restore(lambda: {"loss": 1.0})
    writer.tick({"loss": 0.9})
    writer.tick({"loss": 0.8})

    newest = ckpt_dir / "ckpt_000000002.pkl"
    newest.write_bytes(b"not a pickle at all")
    capsys.readouterr()

    reader = relay_ckpt.Checkpointer(directory=ckpt_dir)
    state = reader.restore(lambda: pytest.fail("should have fallen back"))

    assert state == {"loss": 0.9}
    assert reader.step == 1

    captured = capsys.readouterr()
    assert "ckpt_000000002.pkl" in captured.err
    assert "next oldest" in captured.err
    assert captured.out == "", "the warning belongs on stderr, not in the metric stream"


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def test_sigterm_saves_then_exits_zero_and_finally_still_runs(ckpt_dir: Path):
    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1000
    )
    state = checkpointer.restore(lambda: {"loss": 1.0})

    os.kill(os.getpid(), signal.SIGTERM)
    wait_for_flag(checkpointer)

    cleaned_up = {"ran": False}
    with pytest.raises(SystemExit) as exit_info:
        try:
            checkpointer.tick(state)
        finally:
            cleaned_up["ran"] = True

    assert exit_info.value.code == 0
    assert cleaned_up["ran"], "SystemExit is an exception, so finally blocks still run"
    assert (ckpt_dir / "ckpt_000000001.pkl").exists(), "it saves before exiting"


def test_previous_sigterm_handler_is_still_called(ckpt_dir: Path):
    seen = []

    def user_handler(signum, frame):
        seen.append(signum)

    signal.signal(signal.SIGTERM, user_handler)

    checkpointer = relay_ckpt.Checkpointer(directory=ckpt_dir)
    checkpointer.restore(lambda: {"loss": 1.0})  # first call installs the handler

    os.kill(os.getpid(), signal.SIGTERM)
    wait_for_flag(checkpointer)

    assert seen == [signal.SIGTERM], "a script's own SIGTERM cleanup must survive"


# --------------------------------------------------------------------------
# Durability
# --------------------------------------------------------------------------


def test_failure_between_write_and_replace_keeps_the_previous_checkpoint(
    ckpt_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1
    )
    checkpointer.restore(lambda: {"loss": 1.0})
    checkpointer.tick({"loss": 0.9})  # step 1 lands normally

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated crash between write and replace")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", flaky_replace)

    with pytest.raises(OSError):
        checkpointer.tick({"loss": 0.8})  # step 2 dies mid-save

    monkeypatch.undo()

    assert not (ckpt_dir / "ckpt_000000002.pkl").exists(), "no half-written checkpoint"
    assert list(ckpt_dir.glob("*.tmp")) == [], "the temporary file is cleaned up"

    reader = relay_ckpt.Checkpointer(directory=ckpt_dir)
    assert reader.restore(None) == {"loss": 0.9}
    assert reader.step == 1


def test_only_keep_checkpoints_remain_and_no_tmp_is_left(ckpt_dir: Path):
    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1, keep=2
    )
    checkpointer.restore(lambda: {"loss": 1.0})
    for index in range(6):
        checkpointer.tick({"loss": index})

    names = sorted(path.name for path in ckpt_dir.iterdir())
    assert names == ["ckpt_000000005.pkl", "ckpt_000000006.pkl"]


# --------------------------------------------------------------------------
# Custom serialization
# --------------------------------------------------------------------------


def test_save_fn_and_load_fn_replace_pickle_but_not_the_sequence(ckpt_dir: Path):
    seen_paths = []

    def json_save(record, path):
        seen_paths.append(path)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle)

    def json_load(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    writer = relay_ckpt.Checkpointer(
        directory=ckpt_dir,
        every_minutes=None,
        every_steps=1,
        save_fn=json_save,
        load_fn=json_load,
    )
    writer.restore(lambda: {"loss": 1.0})
    writer.tick({"loss": 0.5})

    assert len(seen_paths) == 1
    assert seen_paths[0].endswith(".pkl.tmp"), "save_fn writes to the temporary path"
    final = ckpt_dir / "ckpt_000000001.pkl"
    assert final.exists()
    assert list(ckpt_dir.glob("*.tmp")) == []
    # It really is JSON, so pickle was not involved.
    assert json.loads(final.read_text(encoding="utf-8")) == {
        "step": 1,
        "state": {"loss": 0.5},
    }

    reader = relay_ckpt.Checkpointer(
        directory=ckpt_dir, save_fn=json_save, load_fn=json_load
    )
    assert reader.restore(None) == {"loss": 0.5}
    assert reader.step == 1


def test_default_format_is_a_pickled_step_and_state_record(ckpt_dir: Path):
    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1
    )
    checkpointer.restore(lambda: {"loss": 1.0})
    checkpointer.tick({"loss": 0.25})

    with open(ckpt_dir / "ckpt_000000001.pkl", "rb") as handle:
        record = pickle.load(handle)

    assert record == {"step": 1, "state": {"loss": 0.25}}


# --------------------------------------------------------------------------
# The ##relay## line
# --------------------------------------------------------------------------


def test_save_prints_the_relay_checkpoint_line(
    ckpt_dir: Path, capsys: pytest.CaptureFixture
):
    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1
    )
    checkpointer.restore(lambda: {"loss": 1.0})
    capsys.readouterr()

    checkpointer.tick({"loss": 0.5})

    lines = marker_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert lines[0]["step"] == 1
    path = Path(lines[0]["checkpoint"])
    assert path.is_absolute(), "the daemon has no idea what the job's cwd was"
    assert path.exists()
    assert path.name == "ckpt_000000001.pkl"


# --------------------------------------------------------------------------
# Accelerator frameworks, faked through sys.modules
# --------------------------------------------------------------------------


def test_torch_tensors_are_moved_to_cpu_at_save(
    ckpt_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, "torch", fake_torch())

    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1
    )
    state = {
        "model": {"w": FakeTensor("cuda")},
        "history": [FakeTensor("cuda"), 3],
        "pair": (FakeTensor("cuda"),),
        "lr": 3e-4,
    }
    checkpointer.restore(lambda: state)
    checkpointer.tick(state)

    with open(ckpt_dir / "ckpt_000000001.pkl", "rb") as handle:
        saved = pickle.load(handle)["state"]

    assert saved["model"]["w"].device == "cpu"
    assert saved["history"][0].device == "cpu"
    assert saved["history"][1] == 3
    assert saved["pair"][0].device == "cpu"
    assert saved["lr"] == 3e-4
    # The originals are untouched; .cpu() returns a copy.
    assert state["model"]["w"].device == "cuda"


def test_jax_device_get_is_used_at_save(
    ckpt_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list = []
    monkeypatch.setitem(sys.modules, "jax", fake_jax(calls))

    checkpointer = relay_ckpt.Checkpointer(
        directory=ckpt_dir, every_minutes=None, every_steps=1
    )
    state = {"params": [1.0, 2.0]}
    checkpointer.restore(lambda: state)
    checkpointer.tick(state)

    assert calls == [state], "jax.device_get walks the pytree for us"
    with open(ckpt_dir / "ckpt_000000001.pkl", "rb") as handle:
        assert pickle.load(handle)["state"] == state


# --------------------------------------------------------------------------
# Where the checkpoints go
# --------------------------------------------------------------------------


def test_directory_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    work = tmp_path / "work"
    work.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    explicit = tmp_path / "explicit"

    monkeypatch.chdir(work)

    monkeypatch.delenv("RELAY_RUN_DIR", raising=False)
    plain = relay_ckpt.Checkpointer()
    assert plain.directory.resolve() == (work / "checkpoints").resolve()

    monkeypatch.setenv("RELAY_RUN_DIR", str(run_dir))
    from_env = relay_ckpt.Checkpointer()
    assert from_env.directory.resolve() == (run_dir / "checkpoints").resolve()

    chosen = relay_ckpt.Checkpointer(directory=explicit)
    assert chosen.directory.resolve() == explicit.resolve()

    # Nothing is created until something is saved.
    assert not (work / "checkpoints").exists()
    assert not (run_dir / "checkpoints").exists()


# --------------------------------------------------------------------------
# The module-level API
# --------------------------------------------------------------------------


def test_module_level_step_reads_through_to_the_instance(ckpt_dir: Path):
    relay_ckpt.configure(every_minutes=None, every_steps=1, directory=ckpt_dir)

    state = relay_ckpt.restore(lambda: {"loss": 1.0})
    assert relay_ckpt.step == 0

    relay_ckpt.tick(state)
    assert relay_ckpt.step == 1, "a plain module int would still read 0 here"

    relay_ckpt.tick(state)
    assert relay_ckpt.step == 2
    assert (ckpt_dir / "ckpt_000000002.pkl").exists()

    with pytest.raises(AttributeError):
        relay_ckpt.nonexistent_attribute


def test_configure_after_restore_is_refused(ckpt_dir: Path):
    relay_ckpt.configure(directory=ckpt_dir)
    relay_ckpt.restore(None)

    with pytest.raises(RuntimeError):
        relay_ckpt.configure(directory=ckpt_dir / "elsewhere")
