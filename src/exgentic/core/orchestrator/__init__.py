# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""The orchestration layer.

The heavy members are resolved lazily (PEP 562 module ``__getattr__``) rather
than imported at package load. Importing them eagerly created a cycle that made
``observers.handlers.results`` unimportable from a clean interpreter:

    results -> orchestrator/__init__ -> run -> execution -> session
            -> tracker -> results        (still half-built -> ImportError)

``results`` only wants ``Observer`` and the ``termination`` exceptions from
here, neither of which depends on the orchestration loop. Deferring ``run``,
``session`` and ``tracker`` means the package can be entered from any side.
Nothing in the codebase used the eager names anyway -- every caller imports
from the submodule directly -- so this changes no import that already worked.
"""

from .controller import Controller, CoreController, LimitController
from .observer import Observer
from .termination import (
    AgentError,
    AgentTerminationError,
    BenchmarkError,
    BenchmarkTerminationError,
    InvalidActionError,
    InvalidObservationError,
    RunCancelError,
    SessionCancelError,
    SessionLimitReachedError,
)

_LAZY = {
    "core_aggregate": ".run",
    "core_evaluate": ".run",
    "core_execute": ".run",
    "run_session": ".session",
    "Tracker": ".tracker",
}

__all__ = [
    "AgentError",
    "AgentTerminationError",
    "BenchmarkError",
    "BenchmarkTerminationError",
    "Controller",
    "CoreController",
    "LimitController",
    "InvalidActionError",
    "InvalidObservationError",
    "Observer",
    "RunCancelError",
    "Tracker",
    "SessionCancelError",
    "SessionLimitReachedError",
    "SessionTermination",
    "run_session",
    "core_aggregate",
    "core_execute",
    "core_evaluate",
]


def __getattr__(name: str):
    """Resolve the orchestration-loop members on first use."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache so the next access skips this
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
