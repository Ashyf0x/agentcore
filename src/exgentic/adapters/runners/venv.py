# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""VenvRunner — runs the HTTP service inside a uv virtual environment.

Uses the same HTTPTransport as ServiceRunner and DockerRunner, but the
uvicorn server runs in a subprocess with its own isolated venv instead
of the host Python or a Docker container.

The venv is created and managed by the EnvironmentManager — VenvRunner
only starts the subprocess and optionally installs extra runtime
dependencies.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...environment.venv import _venv_bin_dir, _venv_python, _venv_script

if TYPE_CHECKING:
    from ...core.context import Role

from ._utils import (
    find_free_port,
    inject_exgentic_env,
    make_close,
    prepare_subprocess_env,
    serialize_kwargs,
)
from .service import HTTPTransport, _wait_for_health
from .transport import ObjectProxy

_HEALTH_TIMEOUT = 360.0
_TRANSPORT_TIMEOUT = 600.0


def _uv(*args: str, check: bool = True, **kwargs: Any) -> subprocess.CompletedProcess:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise RuntimeError("uv CLI not found on PATH")
    result = subprocess.run([uv_bin, *args], check=False, **kwargs)
    if check and result.returncode != 0:
        stderr = getattr(result, "stderr", "") or ""
        stdout = getattr(result, "stdout", "") or ""
        raise RuntimeError(f"uv {' '.join(args[:3])} failed (exit {result.returncode}):\n{stderr}\n{stdout}")
    return result


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate *proc* and its descendants on Windows.

    ``os.killpg`` does not exist on Windows and ``start_new_session`` is a
    no-op, so the runner's subprocess is not a group leader. ``taskkill /T``
    walks the parent/child links instead, which is what actually reaps the
    grandchildren (per-session LiteLLM proxies, MCP servers).
    """
    if proc.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
        capture_output=True,
        check=False,
    )
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class VenvRunner:
    """Start an HTTP service in an isolated uv venv and return an ObjectProxy.

    Parameters
    ----------
    target_cls:       Class to instantiate inside the venv subprocess.
    env_name:         Environment name for EnvironmentManager (e.g. "benchmarks/bfcl").
    module_path:      Dotted module path for locating package resources.
    port:             Host port to bind (auto-selected if None).
    dependencies:     Extra pip packages to install in the venv at runtime.
    health_timeout:   Seconds to wait for the health endpoint.
    """

    def __init__(
        self,
        target_cls: type | str,
        *args: Any,
        env_name: str = "",
        module_path: str = "",
        port: int | None = None,
        dependencies: list[str] | None = None,
        health_timeout: float | None = None,
        role: Role | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError(
                "VenvRunner requires keyword-only constructor arguments. "
                "Pass all arguments as kwargs instead of positional args."
            )
        self._target_cls = target_cls
        self._kwargs = kwargs
        self._env_name = env_name
        self._module_path = module_path
        self._port = port or find_free_port()
        self._dependencies = dependencies or []
        self._health_timeout = health_timeout or _HEALTH_TIMEOUT
        self._role = role
        self._process: subprocess.Popen | None = None
        self._log_file: Any = None
        self._log_path: Path | None = None

    # ── venv handling ─────────────────────────────────────────────────

    def _get_venv_dir(self) -> Path:
        """Return the venv directory managed by EnvironmentManager."""
        from ...environment.instance import get_manager

        return get_manager().env_path(self._env_name) / "venv"

    def _venv_python(self) -> Path:
        """Return the path to the Python binary inside the venv."""
        return _venv_python(self._get_venv_dir())

    def _ensure_venv(self) -> Path:
        """Ensure the venv exists via EnvironmentManager."""
        from ...environment import EnvType
        from ...environment.helpers import get_exgentic_install_target
        from ...environment.instance import get_manager

        mgr = get_manager()
        project_root, packages = get_exgentic_install_target()
        mgr.install(
            self._env_name,
            env_type=EnvType.VENV,
            module_path=self._module_path,
            project_root=project_root,
            packages=packages,
        )
        return self._get_venv_dir()

    def _install_deps(self) -> None:
        """Install extra runtime dependencies into the venv."""
        if not self._dependencies:
            return
        python = self._venv_python()
        _uv(
            "pip",
            "install",
            "--python",
            str(python),
            "--no-cache",
            *self._dependencies,
            capture_output=True,
            text=True,
        )

    # ── subprocess lifecycle ──────────────────────────────────────────

    def start(self) -> ObjectProxy:
        venv = self._ensure_venv()
        self._install_deps()

        if isinstance(self._target_cls, str):
            cls_ref = self._target_cls
        else:
            cls_ref = f"{self._target_cls.__module__}:{self._target_cls.__qualname__}"
        kwargs_flag, kwargs_value = serialize_kwargs(self._kwargs)

        # Build a filtered environment (same filtering as DockerRunner).
        env = prepare_subprocess_env()
        env["VIRTUAL_ENV"] = str(venv)
        venv_bin = str(_venv_bin_dir(venv))
        # Prepend venv bin to the *system* PATH so external tools (docker,
        # git, …) remain reachable from within the venv subprocess.
        system_path = os.environ.get("PATH", "")
        env["PATH"] = venv_bin + os.pathsep + system_path
        inject_exgentic_env(env, role=self._role)

        exgentic_bin = _venv_script(venv, "exgentic")
        cmd = [
            str(exgentic_bin),
            "serve",
            "--cls",
            cls_ref,
            kwargs_flag,
            kwargs_value,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
        ]

        # A PIPE that nobody drains deadlocks as soon as the service writes
        # more than the kernel pipe buffer (64 KiB here): the happy path never
        # calls communicate(), so the child blocks forever in write(). Observed
        # with AppWorld, which prints its API docs on startup -- the serve
        # process sat in `anon_pipe_write` with a constant 730 MB RSS. Stream
        # to a per-service log file instead: no back-pressure, and the output
        # stays available for post-mortem debugging.
        log_dir = self._get_venv_dir().parent / "serve-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        role = getattr(self._role, "value", self._role) or "service"
        self._log_path = log_dir / f"{role}-{self._port}.log"
        self._log_file = self._log_path.open("ab", buffering=0)

        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atexit.register(self._stop_process)

        url = f"http://127.0.0.1:{self._port}"
        try:
            _wait_for_health(url, timeout=self._health_timeout)
        except TimeoutError:
            proc = self._process
            if proc is not None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            tail = ""
            if self._log_path is not None:
                try:
                    tail = self._log_path.read_bytes()[-8000:].decode(errors="replace")
                except OSError:
                    pass
            self._stop_process()
            raise TimeoutError(
                f"Venv service did not become healthy within {self._health_timeout}s.\n"
                f"serve log ({self._log_path}):\n{tail}"
            ) from None

        # Liveness callable so RPCs fail fast if the venv subprocess
        # dies mid-session instead of hanging on httpx's transport timeout.
        def _is_alive() -> bool:
            proc = self._process
            return proc is not None and proc.poll() is None

        transport = HTTPTransport(url, timeout=_TRANSPORT_TIMEOUT, is_alive=_is_alive)
        proxy = ObjectProxy(transport)
        object.__setattr__(proxy, "close", make_close(transport, self._stop_process))
        return proxy

    def _stop_process(self) -> None:
        log_file = self._log_file
        self._log_file = None
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass
        if self._process is None:
            return
        proc = self._process
        self._process = None
        # The subprocess was started with start_new_session=True so it leads
        # its own process group. Signal the whole group so grandchildren
        # (per-session LiteLLM proxies, docker helpers, MCP servers) are
        # cleaned up atomically instead of orphaning.
        #
        # Windows has neither process groups in this sense nor killpg, and
        # start_new_session is a no-op there, so fall back to terminating the
        # tree explicitly.
        if os.name == "nt":
            _kill_process_tree(proc)
            return
        try:
            pgid = os.getpgid(proc.pid)
        except (OSError, ProcessLookupError):
            pgid = None
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
