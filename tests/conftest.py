"""Guard: the tests must exercise *this* checkout, not some other install.

This exists because it went wrong once. A `pip install git+https://...` meant
to land in a throwaway venv landed in the main interpreter instead, replacing
the editable install with a frozen snapshot of the package. From then on every
`pytest` run imported that snapshot: edits to `src/` had no effect on what the
tests saw, and the suite kept passing while the code under test drifted away
from the code on disk. Nothing complained. This file complains.

The check is deliberately blunt: `relay.__file__` must live under this
repository's `src/`. CI installs with `pip install -e .`, so it passes there
too. If it fails, the fix is `pip install -e .` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import relay

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"


def pytest_sessionstart(session: pytest.Session) -> None:
    imported_from = Path(relay.__file__).resolve()
    if _REPO_SRC not in imported_from.parents:
        raise pytest.UsageError(
            f"pytest imported relay from {imported_from}, not from {_REPO_SRC}. "
            "The tests would be exercising a stale copy of the package, not the "
            "code in this checkout. Run `pip install -e .` from the repository "
            "root and try again."
        )
