"""relay_ckpt: the checkpoint half of surviving preemption.

The sidecar can give a training script a warning (a forwarded SIGTERM) and a
few seconds of grace, but it never writes model state -- it has no idea what
your state *is*. Writing it is the training script's job, and this module is
the smallest thing that does that job correctly. Two calls:

    import relay_ckpt

    state = relay_ckpt.restore(lambda: make_initial_state())
    while relay_ckpt.step < total_steps:
        state = train_one_step(state)
        relay_ckpt.tick(state)          # may save; may raise SystemExit(0)

`restore` hands back the newest checkpoint it can load, `tick` counts the
iteration and saves when a save is due or when a SIGTERM has arrived, and
`relay_ckpt.step` is how many ticks have happened *including the ones from
before the preemption*. Nothing else is required, and a script written this way
behaves identically on a laptop with no relay anywhere in sight.

Like `sidecar.py`, this file is copied next to the job on shared storage and
put on the child's PYTHONPATH by the sidecar, so it imports **only** the
standard library -- there is no pip install on the compute node, and no relay
package to import from. A test AST-parses the imports below to keep it honest.

It also must not import `torch` or `jax` at module level. Importing torch costs
seconds and a chunk of memory, and importing jax can claim a GPU. A checkpoint
helper has no business doing either to a script that does not use them. Instead
`save` looks in `sys.modules`: if the training script already imported them,
they are there, and if it did not, there is nothing to convert.

Durability follows the same discipline as the sidecar's event writer, for the
same reason -- SIGKILL arrives whether or not we are finished:

    write to `ckpt_<step>.pkl.tmp` -> flush -> fsync -> os.replace() into
    `ckpt_<step>.pkl` -> fsync the directory

`os.replace` is atomic within a filesystem, so a reader (or the next attempt)
sees either the whole old checkpoint or the whole new one, never a half-written
file. Everything before the replace happens under the `.tmp` name, which
nothing looks for. The fsyncs are what make that promise survive the node
losing power rather than just the process dying.

After a successful save the module prints

    ##relay## {"checkpoint": "/abs/path/ckpt_000000123.pkl", "step": 123}

which is the one line of coupling to the rest of relay: the sidecar turns it
into a `checkpoint` event, and the daemon uses those events to work out how
much work preemption actually cost. There is no import in either direction.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

__all__ = ["Checkpointer", "configure", "restore", "save", "tick"]


# Zero-padded so that lexical order and numeric order are the same thing. With
# `ckpt_9.pkl` and `ckpt_10.pkl`, sorted() says 10 comes first, and a helper
# that restores the wrong checkpoint is worse than no helper. Nine digits is
# a billion steps; the parse below does not actually depend on the width.
_STEP_DIGITS = 9
_CHECKPOINT_RE = re.compile(r"^ckpt_(\d+)\.pkl$")
_SUFFIX = ".pkl"
_TMP_SUFFIX = ".pkl.tmp"


# --------------------------------------------------------------------------
# Directory resolution
# --------------------------------------------------------------------------


def _resolve_directory(explicit: str | os.PathLike[str] | None) -> Path:
    """Where checkpoints go: explicit > $RELAY_RUN_DIR/checkpoints > ./checkpoints.

    `RELAY_RUN_DIR` is exported by the sidecar and is identical across requeue
    attempts, which is the whole point: attempt 2 has to look in the same place
    attempt 1 wrote to, and the Slurm job ID is *not* stable enough for that.

    Falling back to `./checkpoints` is what makes the same script work on a
    laptop with no relay in the picture.
    """
    if explicit is not None:
        return Path(explicit).expanduser().absolute()
    run_dir = os.environ.get("RELAY_RUN_DIR")
    if run_dir:
        return Path(run_dir).expanduser().absolute() / "checkpoints"
    return Path("checkpoints").absolute()


# --------------------------------------------------------------------------
# Moving state off accelerators before pickling
# --------------------------------------------------------------------------


def _device_get(state: Any) -> Any:
    """Pull state back to the host if an accelerator framework is loaded.

    Checked at save time via `sys.modules`, never by importing. If the training
    script never imported jax or torch, these branches do not run and this
    module stays a pure-stdlib no-op.
    """
    jax = sys.modules.get("jax")
    if jax is not None:
        device_get = getattr(jax, "device_get", None)
        if device_get is not None:
            # jax.device_get walks pytrees itself, so one call covers a whole
            # nested train-state object.
            state = device_get(state)

    if "torch" in sys.modules:
        state = _torch_to_cpu(state)

    return state


def _torch_to_cpu(obj: Any) -> Any:
    """Recursively move torch tensors to CPU memory.

    Pickling a CUDA tensor records the device in the file, and unpickling it on
    a node whose GPU 0 is busy (or absent) fails or silently allocates. Moving
    to CPU first makes the checkpoint portable, which matters because a
    requeued job is very unlikely to land on the same node.

    torch has no pytree walker in its public API the way jax does, so we walk
    the usual containers ourselves. That is deliberately simple: dicts, lists
    and tuples cover state_dicts and the tuples people nest inside them. A
    custom container class is not handled, and the docstring saying so is
    better than a half-working `__dict__` crawl that mangles someone's object.
    """
    torch = sys.modules.get("torch")
    tensor_type = getattr(torch, "Tensor", None) if torch is not None else None

    if tensor_type is not None and isinstance(obj, tensor_type):
        cpu = getattr(obj, "cpu", None)
        return cpu() if callable(cpu) else obj
    if isinstance(obj, dict):
        return {key: _torch_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_torch_to_cpu(item) for item in obj]
    if isinstance(obj, tuple):
        converted = tuple(_torch_to_cpu(item) for item in obj)
        if hasattr(obj, "_fields"):  # namedtuple: rebuild the same class
            return type(obj)(*converted)
        return converted
    return obj


# --------------------------------------------------------------------------
# The real implementation
# --------------------------------------------------------------------------


class Checkpointer:
    """Periodic, preemption-aware checkpointing for one training loop.

    The module-level functions are thin wrappers over one lazily created
    instance of this class. Tests (and anyone who wants two independent
    checkpoint streams) construct their own.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        every_minutes: float | None = 10,
        every_steps: int | None = None,
        keep: int = 2,
        save_fn: Callable[[Any, str], None] | None = None,
        load_fn: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        install_signal_handler: bool = True,
    ) -> None:
        if keep < 1:
            raise ValueError("keep must be at least 1")
        if every_steps is not None and every_steps < 1:
            raise ValueError("every_steps must be at least 1")
        if every_minutes is not None and every_minutes <= 0:
            raise ValueError("every_minutes must be positive, or None to disable")

        self.directory = _resolve_directory(directory)
        self.every_steps = every_steps
        self.every_seconds = None if every_minutes is None else every_minutes * 60.0
        self.keep = keep
        self.step = 0

        self._save_fn = save_fn
        self._load_fn = load_fn
        # monotonic by default: the wall clock can jump (NTP, suspend) and a
        # backwards jump would postpone checkpointing for as long as the jump.
        self._clock = clock
        self._last_save_at = clock()

        self._want_signal_handler = install_signal_handler
        self._handler_installed = False
        self._previous_handler: Any = None
        self._term_requested = False

    # -- signals ---------------------------------------------------------

    def _ensure_signal_handler(self) -> None:
        """Install a SIGTERM handler that only sets a flag.

        Saving *inside* a signal handler is the obvious implementation and the
        wrong one: the handler interrupts the main thread at an arbitrary
        bytecode, so the state you would pickle may be half-updated, and any
        lock the interrupted code held is still held. Setting a flag and saving
        at the top of the next `tick` costs at most one iteration and gets a
        consistent state every time.

        Whatever handler was already installed is chained, so a script with its
        own SIGTERM cleanup keeps it. SIG_DFL, SIG_IGN and None are not
        callables and are simply not chained -- calling them is not a thing.
        """
        if self._handler_installed or not self._want_signal_handler:
            return
        try:
            self._previous_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle_term)
        except ValueError:
            # signal.signal() only works on the main thread. If a script calls
            # us from a worker thread, periodic saves still work; only the
            # signal-triggered save is lost. Not worth crashing over.
            self._previous_handler = None
        self._handler_installed = True

    def _handle_term(self, signum: int, frame: Any) -> None:
        self._term_requested = True
        previous = self._previous_handler
        if callable(previous):
            previous(signum, frame)

    @property
    def term_requested(self) -> bool:
        return self._term_requested

    # -- restore ---------------------------------------------------------

    def restore(self, default: Any = None) -> Any:
        """Return the newest loadable checkpoint's state, else `default`.

        `default` may be a callable, which is called only when there is nothing
        to restore. That matters when building the initial state is expensive
        (allocating a model) -- you do not want to pay for it and throw it away.

        "Newest" means the highest step parsed out of the filename, not the
        newest mtime. We chose the names, so they are the better source of
        truth: copying a run directory, or a filesystem with coarse timestamps,
        can reorder mtimes, and restoring an older checkpoint silently loses
        training instead of failing.

        If the newest file fails to load -- truncated by a SIGKILL that landed
        between `os.replace` and the fsync of the directory, or corrupted by a
        full filesystem -- we warn and fall back to the next oldest. Keeping
        more than one checkpoint is what makes this possible, which is why the
        default `keep` is 2 and not 1.
        """
        self._ensure_signal_handler()

        for step, path in self._checkpoints():
            try:
                record = self._load(str(path))
            except Exception as exc:  # noqa: BLE001 -- any failure means "try the next"
                print(
                    f"relay_ckpt: could not load {path} ({type(exc).__name__}: {exc}); "
                    "falling back to the next oldest checkpoint",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            self.step, state = _unpack(record, step)
            # Don't let a periodic save fire immediately just because restoring
            # took a while.
            self._last_save_at = self._clock()
            return state

        self.step = 0
        return default() if callable(default) else default

    # -- tick ------------------------------------------------------------

    def tick(self, state: Any) -> None:
        """Count one training iteration, saving `state` if a save is due.

        Called once per loop iteration, *after* the step it is counting, so the
        checkpoint and the counter always describe the same moment: restoring
        `step == 100` means one hundred iterations really happened.

        Raises SystemExit(0) after saving if SIGTERM arrived. SystemExit is an
        exception, so `finally` blocks and context managers in the training
        loop still run, and exiting 0 is correct: being preempted is not the
        training script failing.
        """
        self._ensure_signal_handler()
        self.step += 1

        if self._term_requested:
            self.save(state)
            raise SystemExit(0)

        if self._is_due():
            self.save(state)

    def _is_due(self) -> bool:
        """True when either configured interval has elapsed.

        The two conditions are OR-ed, not exclusive: `every_steps` is a floor
        on how often you save and `every_minutes` is a ceiling on how much
        wall-clock work is at risk. Pass `every_minutes=None` to use step
        counts alone.
        """
        if self.every_steps is not None and self.step % self.every_steps == 0:
            return True
        if self.every_seconds is not None:
            return (self._clock() - self._last_save_at) >= self.every_seconds
        return False

    # -- save ------------------------------------------------------------

    def save(self, state: Any) -> Path:
        """Write a checkpoint for the current step and return its path.

        The sequence is the important part; see the module docstring. Errors
        propagate: a checkpoint that did not get written is something the user
        needs to hear about, and swallowing it would let a job run for hours
        believing it is safe.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        stem = f"ckpt_{self.step:0{_STEP_DIGITS}d}"
        final = self.directory / (stem + _SUFFIX)
        tmp = self.directory / (stem + _TMP_SUFFIX)

        record = {"step": self.step, "state": _device_get(state)}

        try:
            self._write(record, str(tmp))
            # Atomic within a filesystem: readers see the old file or the new
            # one. The previous checkpoint stays loadable until this instant.
            os.replace(tmp, final)
        except BaseException:
            # A stray .tmp would be invisible to `restore` (it only looks at
            # .pkl) but would accumulate forever on shared storage.
            tmp.unlink(missing_ok=True)
            raise

        _fsync_directory(self.directory)
        self._prune()

        # The one line of coupling to relay: the sidecar parses this into a
        # `checkpoint` event. Absolute path, because the daemon reading it has
        # no idea what the job's working directory was.
        print(
            "##relay## "
            + json.dumps({"checkpoint": str(final), "step": self.step}),
            flush=True,
        )

        self._last_save_at = self._clock()
        return final

    def _write(self, record: dict[str, Any], tmp_path: str) -> None:
        """Serialize `record` to `tmp_path` and get it onto the disk.

        A custom `save_fn` replaces pickle but not the durability sequence: it
        writes to the temporary path we hand it, and we do the fsync, the
        replace and the directory fsync afterwards. That way a framework's own
        saver (torch.save, orbax, safetensors) cannot accidentally skip the
        part that makes preemption survivable.
        """
        if self._save_fn is not None:
            self._save_fn(record, tmp_path)
            # We did not hold the handle, so reopen to force it to disk.
            # fsync on a read-only descriptor is allowed and does the job.
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return

        with open(tmp_path, "wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            # flush() only moves bytes from Python's buffer to the kernel's.
            # fsync is what moves them to the device.
            os.fsync(handle.fileno())

    def _load(self, path: str) -> Any:
        if self._load_fn is not None:
            return self._load_fn(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)

    # -- housekeeping ----------------------------------------------------

    def _checkpoints(self) -> list[tuple[int, Path]]:
        """Every complete checkpoint, newest step first.

        `.tmp` files are ignored by construction: the regex requires the name
        to end in `.pkl`, so a half-written file can never be restored from.
        """
        try:
            names = os.listdir(self.directory)
        except FileNotFoundError:
            return []

        found: list[tuple[int, Path]] = []
        for name in names:
            match = _CHECKPOINT_RE.match(name)
            if match:
                found.append((int(match.group(1)), self.directory / name))
        found.sort(key=lambda pair: pair[0], reverse=True)
        return found

    def _prune(self) -> None:
        """Delete all but the newest `keep` checkpoints.

        Model state is large and shared storage has quotas. Pruning after the
        replace, never before, means the old checkpoint is only removed once
        its replacement is fully on disk.
        """
        for _, path in self._checkpoints()[self.keep :]:
            try:
                path.unlink()
            except OSError:
                # Someone else's cleanup got there first, or the filesystem is
                # unhappy. Neither is worth killing a training run over.
                pass


def _unpack(record: Any, step_from_name: int) -> tuple[int, Any]:
    """Split a loaded record into (step, state).

    We always write `{"step": ..., "state": ...}`, so the first branch is the
    normal path. The fallback covers a `load_fn` that returns bare state: the
    filename still carries the step, so restoring the counter keeps working.
    """
    if isinstance(record, dict) and "step" in record and "state" in record:
        return int(record["step"]), record["state"]
    return step_from_name, record


def _fsync_directory(directory: Path) -> None:
    """fsync the directory so the renamed entry itself survives a power loss.

    fsync on the file only promises the file's *contents*. The directory entry
    created by `os.replace` is separate metadata, and on a crash you can end up
    with durable bytes under a filename that no longer exists.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return  # not every platform lets you open a directory
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# Module-level API: one lazily created default Checkpointer
# --------------------------------------------------------------------------

_default: Checkpointer | None = None
_config: dict[str, Any] = {}


def configure(
    every_minutes: float | None = 10,
    every_steps: int | None = None,
    keep: int = 2,
    directory: str | os.PathLike[str] | None = None,
    save_fn: Callable[[Any, str], None] | None = None,
    load_fn: Callable[[str], Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Set options for the default checkpointer. Optional; call before `restore`.

    It raises rather than reconfiguring an instance that already exists,
    because a `directory` change after `restore` would mean reading from one
    place and writing to another -- and the failure would only show up on the
    requeued attempt, hours later.
    """
    global _config
    if _default is not None:
        raise RuntimeError(
            "relay_ckpt.configure() must be called before relay_ckpt.restore()/tick()"
        )
    _config = {
        "every_minutes": every_minutes,
        "every_steps": every_steps,
        "keep": keep,
        "directory": directory,
        "save_fn": save_fn,
        "load_fn": load_fn,
        "clock": clock,
    }


def _instance() -> Checkpointer:
    global _default
    if _default is None:
        _default = Checkpointer(**_config)
    return _default


def _reset_default() -> None:
    """Drop the default instance and its configuration. For tests."""
    global _default, _config
    _default = None
    _config = {}


def restore(default: Any = None) -> Any:
    """Newest loadable checkpoint's state, else `default()` or `default`."""
    return _instance().restore(default)


def tick(state: Any) -> None:
    """Count one iteration on the default checkpointer, saving if due."""
    _instance().tick(state)


def save(state: Any) -> Path:
    """Force a save right now, regardless of the interval."""
    return _instance().save(state)


def __getattr__(name: str) -> Any:
    """Make `relay_ckpt.step` read through to the live instance (PEP 562).

    A plain module-level `step = 0` cannot work. Ints are immutable, so the
    instance's counter and the module global would be two separate objects the
    moment either changed; keeping them in sync would mean every `tick`
    reaching back into its own module's globals, and `from relay_ckpt import
    step` would still capture a number frozen at import time.

    PEP 562 lets a module define `__getattr__`, called only for names *not*
    found in the module dict -- which is why there is deliberately no global
    named `step` here. `relay_ckpt.step` is then an attribute lookup that runs
    this function and returns whatever the instance says right now.

    (`from relay_ckpt import step` still snapshots a number. That is how `from
    ... import` works for any module, and the docstring above tells people to
    write `relay_ckpt.step`.)
    """
    if name == "step":
        return _instance().step
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
