# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Acquire the agentcore runtime: the vendored copy or an external checkout.

By default this uses the copy vendored inside this package, so a run needs
nothing beyond a normal install. Pass ``runtime_path`` to point at an external
checkout instead -- the directory that *contains* the ``agentcore`` package
(normally ``.../agent-runtime``), not the package itself. That is how you
evaluate a working copy that has moved ahead of the vendored snapshot.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

# LiteLLM routing prefixes that are not part of the wire model name. Keep this
# in sync with the agent's default model: an unstripped ``qwen/`` prefix is sent
# verbatim to an OpenAI-compatible endpoint, which answers 404 "Model not exist".
_ROUTING_PREFIXES = (
    "openai/", "hosted_vllm/", "openrouter/", "azure/", "qwen/", "dashscope/",
)

_RUNTIME_CACHE: dict[str, SimpleNamespace] = {}


def load_runtime(runtime_path: str = "") -> SimpleNamespace:
    """Import the agentcore runtime.

    Cached per resolved path so repeated sessions in one process pay the import
    once.
    """
    key = str(Path(runtime_path).expanduser().resolve()) if runtime_path else ""
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        return cached

    if not key:
        module = _load_vendored_runtime()
    else:
        if key not in sys.path:
            sys.path.insert(0, key)
        module = _collect_runtime(package="agentcore")
    _RUNTIME_CACHE[key] = module
    return module


def _load_vendored_runtime() -> SimpleNamespace:
    """Import the copy that ships with this adapter.

    The vendored tree sits beside this module and uses relative imports, so it
    cannot collide with an installed ``agentcore`` or with a checkout on
    ``sys.path``.
    """
    return _collect_runtime(package="exgentic.agents.agentcore")


def _collect_runtime(package: str) -> SimpleNamespace:
    """Import the runtime entry points this adapter needs and bundle them.

    Only these are reached for. Keeping the surface this small is what lets the
    vendored tree skip the upstream ``app/`` CLI and its ``mcp``/``dotenv`` deps.

    ``FatalRunError`` is bundled rather than imported at module scope because
    the vendored copy and an external ``runtime_path`` checkout are *different
    class objects* for the same source. A module-level import would silently
    fail to match the exception the runtime actually raises.
    """
    context = importlib.import_module(f"{package}.context")
    model = importlib.import_module(f"{package}.core.model")
    agent = importlib.import_module(f"{package}.core.agent")
    errors = importlib.import_module(f"{package}.core.errors")
    core_types = importlib.import_module(f"{package}.core.types")
    storage = importlib.import_module(f"{package}.storage")
    tools = importlib.import_module(f"{package}.tools")
    observability = importlib.import_module(f"{package}.observability.report")

    return SimpleNamespace(
        ContextManager=context.ContextManager,
        Agent=agent.Agent,
        AgentConfig=agent.AgentConfig,
        FatalRunError=core_types.FatalRunError,
        ToolFailure=errors.ToolFailure,
        HTTPModelClient=model.HTTPModelClient,
        Store=storage.Store,
        ToolRegistry=tools.ToolRegistry,
        ToolSpec=tools.ToolSpec,
        run_report=observability.run_report,
    )


def wire_model_name(model_id: str, override: str = "") -> str:
    """Strip a LiteLLM routing prefix so the endpoint receives a real model name."""
    if override:
        return override
    lowered = model_id.lower()
    for prefix in _ROUTING_PREFIXES:
        if lowered.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id
