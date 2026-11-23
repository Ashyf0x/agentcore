# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Agent configuration for the agentcore runtime.

This adapter mounts the author's own agent runtime (``agentcore``) as an
Exgentic agent. The runtime owns its tool-call loop, its context budget and its
session store, so this is the ``CodeAgentInstance`` style of adapter: Exgentic
supplies the environment functions and the observation transport, while the
runtime drives the reasoning loop.

Defaults follow the frozen retail configuration in
``agent-runtime/experiments/retail-qwen-tier-order-20261108``:

* ``context_policy="capacity"`` / ``context_compaction="legacy"`` — the B profile
* no argument/protocol corrections — no repair calls the baselines do not make
* memory off — retrieval is keyed by user id, which every task in a run shares,
  so enabling it would leak evidence across independent tasks

One default deliberately departs from that pilot: there is no serialized-byte
request ceiling. The pilot's 28KB ``max_request_bytes`` predates the tau2 domains,
whose policies alone are 7KB (airline), 7KB (retail) and 23KB (telecom); with the
ceiling in place the runtime raised ``ContextBudgetError`` before the agent could
take its first action, which measures the limit rather than the agent.

What bounds a request instead is the model's context window, in tokens:
``context_window_tokens`` is the window size, ``output_reserve_tokens`` the room
kept for the reply, and both must be paired with a trusted ``token_counter``.
Leave ``context_window_tokens=None`` to send the whole context and let the
endpoint decide what fits.

``max_steps`` here counts *model calls inside one user turn*. Exgentic's
``RunConfig.max_steps`` counts environment turns at the run level. They are not
the same quantity and are not meant to match.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from ...core.agent import Agent
from ...core.types import ModelSettings
from .settings import (
    AgentcoreSettings,
    validate_context_window_tokens,
    validate_max_output_tokens,
    validate_max_steps,
    validate_output_reserve_tokens,
    validate_timeout,
    validate_timeout_layering,
)

#: Fields ``AgentcoreSettings`` takes from the agent object by name. ``model_id``
#: is the one exception: the agent calls it ``model``.
_AGENT_FIELDS = frozenset(AgentcoreSettings.__dataclass_fields__) - {"model_id"}


class AgentcoreAgent(Agent):
    """Run the agentcore runtime as an Exgentic agent."""

    display_name: ClassVar[str] = "Agentcore Runtime"
    slug_name: ClassVar[str] = "agentcore"

    model: str = "qwen/qwen3.8-reasoner"

    # Model calls inside one user turn.
    max_steps: int = 24
    model_settings: ModelSettings | None = None

    # Directory that *contains* the ``agentcore`` package (normally
    # ``.../agent-runtime``). Empty means "already importable".
    runtime_path: str = ""

    # Explicit endpoint/model overrides. Empty means: take api_base from
    # litellm_params_extra or OPENAI_BASE_URL, and send the model id stripped of
    # its LiteLLM routing prefix.
    api_base: str = ""
    wire_model: str = ""

    # Protocol branch of the runtime's HTTP client. "compatible" is plain OpenAI
    # Chat Completions and rejects ``thinking``; the others emit the vendor
    # specific reasoning payloads.
    provider: Literal["compatible", "qwen", "deepseek", "bailian_deepseek"] = "compatible"
    thinking: bool | None = None
    preserve_thinking: bool = False

    # Output cap on the wire. ``max_completion_tokens`` replaces it under the
    # field name some providers require.
    max_output_tokens: int = 2048
    max_completion_tokens: int | None = None

    # Model context window, in tokens. This is the only size limit the runtime
    # applies: requests are trimmed until they fit the window. ``None`` disables
    # trimming, so the whole context is sent and the endpoint decides what fits.
    # A configured window needs a trusted ``token_counter`` plus an output reserve.
    context_window_tokens: int | None = None
    output_reserve_tokens: int = 4096

    # Runtime context policy. "capacity" is the frozen B profile; "recent" is A.
    context_policy: Literal["recent", "full", "capacity"] = "capacity"
    context_compaction: Literal["legacy", "task"] = "legacy"

    # Wall-clock limits. ``model_timeout`` wraps one model call (the outer
    # ``asyncio.wait_for``); ``http_timeout`` is the httpx deadline inside it and
    # must stay strictly smaller, or which deadline fires first is a race and the
    # failure is attributed inconsistently. ``tool_timeout`` is advisory, since a
    # blocking coordinator call cannot be pre-empted.
    #
    # 600s matches what the official baseline effectively has: it never passes a
    # deadline, so litellm's ``request_timeout`` of 6000s applies. A 60s deadline
    # is a *throughput* limit masquerading as a liveness check: on this endpoint a
    # reasoning model emits ~18 tokens/s, so 60s caps a reply at roughly 1000
    # tokens and a timed-out call still succeeds on replay. Run limits belong to
    # ``max_steps``; this only has to catch a hung connection.
    model_timeout: float = 600.0
    http_timeout: float = 540.0
    tool_timeout: float = 600.0

    # How many times a retryable model failure is re-sent inside one turn, and
    # how many corrective re-prompts are allowed before the turn is failed.
    # ``None`` for the correction limits means "runtime default": tau2 keeps the
    # frozen zero budget for baseline parity, tool-only benchmarks get headroom.
    model_retries: int = 3
    max_argument_corrections: int | None = None
    max_protocol_corrections: int | None = None

    # `ActionType` has no read/write flag, so every tool defaults to writable and
    # an interrupted dispatch is escalated. List the tools that only read so an
    # interrupted read is handed back to the model instead.
    read_only_actions: list[str] = Field(default_factory=list)

    # Off by default; see the module docstring. Partial under this adapter: only
    # the read path (Memory.retrieve) is wired. Automatic extraction/commit is
    # not connected, and retrieval is keyed by user id (shared across tasks in
    # a run), so enabling it can leak evidence across independent tasks.
    enable_memory: bool = False

    # Directory holding the runtime's SQLite store. Empty means a per-session
    # directory under the Exgentic output tree.
    store_dir: str = ""

    # Namespace for the runtime's own user/session scoping.
    user_id: str = "exgentic"

    litellm_params_extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("max_steps")
    @classmethod
    def _validate_max_steps(cls, value: int) -> int:
        return validate_max_steps(value)

    @field_validator("context_window_tokens")
    @classmethod
    def _validate_context_window_tokens(cls, value: int | None) -> int | None:
        """Accept ``None`` (no window configured) or a positive token count.

        The upper bound is a sanity bound on the configuration, not a model
        constraint: no current endpoint advertises more than a few million tokens.
        """
        return validate_context_window_tokens(value)

    @field_validator("output_reserve_tokens")
    @classmethod
    def _validate_output_reserve_tokens(cls, value: int) -> int:
        return validate_output_reserve_tokens(value)

    @field_validator("max_output_tokens")
    @classmethod
    def _validate_max_output_tokens(cls, value: int) -> int:
        return validate_max_output_tokens(value)

    @field_validator("model_timeout", "http_timeout", "tool_timeout")
    @classmethod
    def _validate_timeouts(cls, value: float) -> float:
        return validate_timeout(value)

    @model_validator(mode="after")
    def _validate_timeout_layering(self):
        """The transport deadline must fire before the outer ``wait_for`` deadline.

        When the two are equal, whichever wins is a scheduling race: sometimes
        the request fails as an httpx timeout, sometimes the coroutine is
        cancelled from outside. The attribution then differs run to run for the
        same endpoint failure.
        """
        validate_timeout_layering(self.http_timeout, self.model_timeout)
        return self

    @classmethod
    def _get_instance_class(cls):
        from .agentcore_instance import AgentcoreAgentInstance

        return AgentcoreAgentInstance

    @classmethod
    def _get_instance_class_ref(cls) -> str:
        return "exgentic.agents.agentcore.agentcore_instance:AgentcoreAgentInstance"

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return str(self.model).split("/")[-1]

    def get_models_names(self) -> list[str]:  # type: ignore[override]
        return [str(self.model)]

    def _get_instance_kwargs(self, session_id: str) -> dict[str, Any]:
        """Build the runner's kwargs from one validated settings object.

        Field-by-field copying used to live here; a setting added in one place
        and forgotten here produced a knob that looked accepted and did nothing.
        The field set is derived from ``AgentcoreSettings``, so forwarding is
        automatic, and ``test_agentcore_agent`` pins the settings against the
        instance signature instead.
        """
        values = {name: getattr(self, name) for name in _AGENT_FIELDS}
        values.update(
            model_id=str(self.model),
            read_only_actions=tuple(self.read_only_actions),
            litellm_params_extra=dict(self.litellm_params_extra),
        )
        settings = AgentcoreSettings(**values)
        return settings.instance_kwargs(session_id)
