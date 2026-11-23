# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Regression tests for the POSIX-only assumptions that broke Windows runs.

Each case here corresponds to a failure that actually occurred while running
``exgentic preview --benchmark tau2`` on Windows:

1. ``bin/python`` was hardcoded, so ``uv venv`` reported "No virtual
   environment or system Python installation found for path".
2. ``bin/exgentic`` was hardcoded, so launching the service subprocess raised
   ``FileNotFoundError: [WinError 2]``.
3. ``os.getpgid``/``os.killpg`` were called unconditionally, so teardown raised
   ``AttributeError: module 'os' has no attribute 'getpgid'``.
4. ``prepare_subprocess_env`` forwarded only provider credentials, leaving the
   child without ``SystemRoot``; importing asyncio there then failed with
   ``OSError: [WinError 10106]`` because Winsock could not find its providers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from exgentic.environment.venv import _venv_bin_dir, _venv_python, _venv_script


@pytest.fixture
def as_windows(monkeypatch):
    """Pretend to run on Windows without needing an actual Windows host."""
    monkeypatch.setattr(os, "name", "nt")


@pytest.fixture
def as_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")


def test_interpreter_path_on_posix(tmp_path: Path, as_posix):
    """POSIX venvs put the interpreter in bin/python."""
    venv = tmp_path / "venv"

    assert _venv_python(venv) == venv / "bin" / "python"


def test_interpreter_path_on_windows(tmp_path: Path, as_windows):
    """Windows venvs put the interpreter in Scripts/python.exe."""
    venv = tmp_path / "venv"

    assert _venv_python(venv) == venv / "Scripts" / "python.exe"


def test_console_script_path_on_posix(tmp_path: Path, as_posix):
    """Console scripts have no suffix on POSIX."""
    venv = tmp_path / "venv"

    assert _venv_script(venv, "exgentic") == venv / "bin" / "exgentic"


def test_console_script_path_on_windows(tmp_path: Path, as_windows):
    """uv creates console scripts with an .exe suffix on Windows."""  # noqa: D403 - "uv" is a tool name
    venv = tmp_path / "venv"

    assert _venv_script(venv, "exgentic") == venv / "Scripts" / "exgentic.exe"


def test_bin_dir_matches_the_interpreter_location(tmp_path: Path, as_windows):
    """Every path into a venv's executables derives from the same directory."""
    venv = tmp_path / "venv"

    assert _venv_python(venv).parent == _venv_bin_dir(venv)
    assert _venv_script(venv, "exgentic").parent == _venv_bin_dir(venv)


def test_venv_runner_resolves_the_same_paths(tmp_path: Path, as_windows, monkeypatch):
    """VenvRunner must not carry its own copy of the layout rules."""
    from exgentic.adapters.runners.venv import VenvRunner

    venv = tmp_path / "venv"
    runner = VenvRunner.__new__(VenvRunner)
    monkeypatch.setattr(runner, "_get_venv_dir", lambda: venv)

    assert runner._venv_python() == _venv_python(venv)


def test_stop_process_uses_taskkill_on_windows(monkeypatch):
    """Windows has no process groups, so teardown must terminate the tree."""
    from exgentic.adapters.runners import venv as runner_module

    calls: list[list[str]] = []

    class FakeProc:
        pid = 4321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd),
    )

    runner_module._kill_process_tree(FakeProc())

    assert calls, "taskkill was never invoked"
    assert calls[0][:4] == ["taskkill", "/T", "/F", "/PID"]
    assert calls[0][4] == "4321"


def test_stop_process_returns_early_for_a_finished_process(monkeypatch):
    """A process that already exited needs no teardown at all."""
    from exgentic.adapters.runners import venv as runner_module

    calls: list[list[str]] = []

    class DeadProc:
        pid = 1

        def poll(self):
            return 0

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(runner_module.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))

    runner_module._kill_process_tree(DeadProc())

    assert calls == []


def test_windows_system_variables_survive_env_filtering(monkeypatch):
    """A child process needs SystemRoot; credentials alone are not enough."""
    from exgentic.adapters.runners import _utils

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("WINDIR", r"C:\Windows")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("UNRELATED_VAR", "should-not-be-forwarded")

    env = _utils.prepare_subprocess_env()

    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["WINDIR"] == r"C:\Windows"
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert "UNRELATED_VAR" not in env


def test_posix_env_filtering_stays_credential_only(monkeypatch):
    """On POSIX the filter keeps its original, narrower contract."""
    from exgentic.adapters.runners import _utils

    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    env = _utils.prepare_subprocess_env()

    assert env["OPENAI_API_KEY"] == "sk-test"
    assert "SYSTEMROOT" not in env
