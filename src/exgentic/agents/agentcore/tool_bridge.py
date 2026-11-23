# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Translate the benchmark's action space into the runtime's tool registry.

The runtime sees exactly the environment tools, in the same shape the bundled
agents advertise. ``message`` is withheld: the runtime speaks through
``RunResult.answer``, and a ``message`` tool would let it reply mid-turn and
again at the end of the turn.

Handlers are declared ``async`` on purpose: ``agentcore``'s executor awaits
coroutine handlers in place but pushes plain functions through
``asyncio.to_thread``, which would land the environment rendezvous (and the
SQLite store) on a pool thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class ToolBridge:
    """Build the runtime's ``ToolRegistry`` from the environment's actions."""

    def __init__(
        self,
        *,
        actions: Iterable[Any],
        bindings: dict[str, Callable],
        read_only: frozenset[str],
        timeout: float,
        runtime: Any,
        logger: Any,
        environment_closed: Callable[[], bool],
    ) -> None:
        # Kept as given: ``build`` validates ``read_only`` before any use of the
        # action space, so the read_only typo surfaces as ValueError even when
        # the instance has not been started yet.
        self.actions = actions
        self.bindings = dict(bindings)
        self.read_only = frozenset(read_only)
        self.timeout = timeout
        self.runtime = runtime
        self.logger = logger
        self._environment_closed = environment_closed

    def build(self):
        unknown_read_only = self.read_only - set(self.bindings)
        if unknown_read_only:
            # A typo here used to be silent: the action stayed writable, so an
            # interrupted read escalated as if it were an unconfirmed write.
            raise ValueError(
                f"read_only_actions not in the action space: {sorted(unknown_read_only)}"
            )
        tools = self.runtime.ToolRegistry()
        for name, parameters, description in self.specifications():
            function = self.bindings.get(name)
            if function is None:
                raise ValueError(f"No function binding for action '{name}'")
            tools.register(
                self.runtime.ToolSpec(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=self._handler(name, function),
                    read_only=name in self.read_only,
                    timeout=self.timeout,
                )
            )
        return tools

    def specifications(self):
        """Tool schemas in the same shape the bundled agents advertise.

        Reusing their registry keeps the schemas byte-identical apart from the
        ``message`` action, which that helper already excludes -- the runtime
        speaks through ``RunResult.answer``, so a ``message`` tool would let it
        reply mid-turn and then again at the end of the turn.
        """
        from ..litellm_tool_calling.utils import ToolsActionsRegistry

        specs = []
        for tool in ToolsActionsRegistry(list(self.actions)).openai_tools():
            function = tool["function"]
            specs.append(
                (function["name"], function["parameters"], function.get("description", ""))
            )
        return specs

    def _handler(self, name: str, function: Callable):
        """Bridge one environment action into the runtime."""

        async def handler(**arguments):
            if self._environment_closed():
                raise self.runtime.ToolFailure(
                    "The environment closed the session", "cancelled"
                )
            try:
                value = function(**arguments)
            except RuntimeError as exc:
                if self._environment_closed():
                    # AgentCoordinator.execute() refuses calls once it is closed.
                    raise self.runtime.ToolFailure(
                        "The environment closed the session", "cancelled"
                    ) from exc
                raise
            if self._environment_closed():
                # execute() returned None: no observation arrived, so there is no
                # result to report. Saying so is the only honest option; handing
                # the model ``None`` would read as "the call returned nothing".
                raise self.runtime.ToolFailure(
                    "The environment closed the session", "cancelled"
                )
            self.logger.debug("Tool %s returned %s", name, type(value).__name__)
            return value

        return handler
