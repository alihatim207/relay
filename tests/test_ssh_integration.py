"""One real ssh connection, against localhost, to prove the socket binds.

This is the only test in the whole suite that makes `ssh` actually run and
actually create a control socket. Everything else -- `tests/test_ssh.py`, the
backend tests, the daemon tests -- hands the argv to a fake runner and asserts
on a list of words. That is fast, hermetic and completely blind to the bug
this file exists for: the first real run on Klone got through Duo and then
died with

    unix_listener: cannot bind to path ~/.relay/cm-<hash>.<random>:
    No such file or directory

The argv was perfect. The directory it named did not exist, and only the real
ssh binary ever finds that out, at the moment it tries to bind.

So this test runs `relay connect`'s own argv against `localhost`, with HOME
redirected into a temporary directory, and checks that a socket file appears
where relay says it will. It needs a local sshd that accepts key
authentication from this account, which many machines (this author's laptop
included) do not have, so it skips with a reason rather than failing. That is
exactly why the unit tests in `tests/test_ssh.py` also exist: they pin the
directory invariant everywhere this one cannot run.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from relay import ssh

# Options added on top of relay's own, for this test only.
#
# `BatchMode=yes` is the important one. `connect_argv` deliberately omits it,
# because on a real cluster that call is the one allowed to prompt for Duo.
# Here there is no human to answer a prompt, so without BatchMode a missing
# key would leave ssh sitting on a password prompt until the test's timeout.
# `StrictHostKeyChecking=accept-new` stops the same thing happening over the
# host key for localhost on a fresh machine.
NON_INTERACTIVE = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=5",
]


def _require_local_ssh() -> None:
    """Skip unless a real `ssh` can reach a real sshd on this machine.

    Two things can be missing, and the skip message says which, because a CI
    log that only says "skipped" teaches nobody anything:

      * no `ssh` binary at all;
      * no sshd listening on localhost, or one that will not accept key
        authentication from this account (macOS ships with Remote Login off).

    The probe itself uses BatchMode so it can never hang on a prompt, and a
    timeout is treated as a skip rather than an error: a probe that did not
    finish has not told us the environment is broken, only that it is not
    usable here.
    """
    if shutil.which("ssh") is None:
        pytest.skip("no `ssh` binary on PATH")

    probe = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "localhost",
        "true",
    ]
    try:
        proc = subprocess.run(probe, capture_output=True, timeout=15, check=False)
    except subprocess.TimeoutExpired:
        pytest.skip("`ssh localhost true` did not finish within 15s")
    except OSError as exc:
        pytest.skip(f"could not run ssh: {exc}")

    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        first = detail[0] if detail else f"exit code {proc.returncode}"
        pytest.skip(
            "no usable sshd on localhost (needs Remote Login enabled and key "
            f"auth to this account): {first}"
        )


def _cm_entries(directory: Path) -> set[str]:
    """Names of the control-socket files in `directory`, if it exists at all."""
    try:
        return {p.name for p in directory.iterdir() if p.name.startswith("cm-")}
    except FileNotFoundError:
        return set()


def test_connect_argv_really_binds_a_control_socket(tmp_path, monkeypatch):
    """Run relay's real connect argv and watch the socket appear.

    The assertions, in order: the directory does not exist to begin with; ssh
    exits 0; the directory now exists with mode 0700; something inside it is a
    Unix socket named `cm-*`; `ssh.master_alive` agrees a master is up. Then
    the master is shut down with `ssh -O exit` and the socket is gone.

    HOME stays redirected for the whole test, including across the
    `subprocess.run` calls, because ssh expands the `~` in relay's
    ControlPath itself, in the child process. A test that redirected HOME only
    for the Python side would bind a socket in the developer's real home
    directory and then assert about an empty temporary one. The last
    assertions check the real home was left alone, to catch exactly that.
    """
    _require_local_ssh()

    real_home = Path(os.environ.get("HOME", str(Path.home())))
    real_relay_dir = real_home / ".relay"
    real_before = _cm_entries(real_relay_dir)

    monkeypatch.setenv("HOME", str(tmp_path))
    control_dir = tmp_path / ".relay"
    assert not control_dir.exists(), "HOME redirect must start from nothing"
    assert ssh.relay_dir() == control_dir, "expanduser is not following HOME"

    argv = ssh.connect_argv("localhost")
    # The alias is the last word; the extra options have to go before it.
    argv = argv[:-1] + NON_INTERACTIVE + [argv[-1]]

    try:
        proc = subprocess.run(argv, capture_output=True, timeout=20, check=False)
        stderr = proc.stderr.decode("utf-8", "replace")
        assert proc.returncode == 0, f"connect failed: {stderr.strip()}"

        # The bug, in one assertion: ssh could not have bound anything if this
        # directory had not been created for it first.
        assert control_dir.is_dir(), f"no {control_dir} after connect: {stderr}"
        assert stat.S_IMODE(control_dir.stat().st_mode) == 0o700

        sockets = [
            entry
            for entry in control_dir.iterdir()
            if entry.name.startswith("cm-") and stat.S_ISSOCK(entry.lstat().st_mode)
        ]
        assert sockets, (
            f"no cm-* socket in {control_dir}; found "
            f"{sorted(p.name for p in control_dir.iterdir())}"
        )

        # And the thing relay actually asks at runtime agrees.
        assert ssh.master_alive("localhost") is True
    finally:
        # Never leave a backgrounded master behind, even if an assertion above
        # failed. `-O exit` is `master_check_argv` with a different command.
        subprocess.run(
            [
                "ssh",
                "-o", f"ControlPath={ssh.CONTROL_PATH_TEMPLATE}",
                "-O", "exit",
                "localhost",
            ],
            capture_output=True,
            timeout=20,
            check=False,
        )

    assert not _cm_entries(control_dir), "the master socket outlived `-O exit`"
    assert ssh.master_alive("localhost") is False

    # Nothing was written to the real home directory: if HOME had leaked, the
    # socket would have been created there instead and this would catch it.
    assert _cm_entries(real_relay_dir) == real_before
