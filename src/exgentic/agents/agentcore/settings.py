# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""The adapter's complete configuration surface, validated in one place.

Both entry points -- ``AgentcoreAgent`` (the config object the CLI/API builds)
and ``AgentcoreAgentInstance`` (what the runner constructs) derive from this
dataclass. Instance kwargs are generated from the fields, so a knob added here
cannot be accepted by one side and silently dropped by the other. The shared
``validate_*`` helpers keep the pydantic validators and the runtime-side checks
from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

CONTEXT_POLICIES = frozenset({"recent", "full", "capacity"})
CONTEXT_COMPACTIONS = frozenset({"legacy", "task"})
PROVIDERS = frozenset({"compatible", "qwen", "deepseek", "bailian_deepseek"})


def validate_max_steps(value: int) -> int:
    if value < 1:
        raise ValueError("max_steps must be positive")
    return value


def validate_context_window_tokens(value: int | None) -> int | None:
    """Accept ``None`` (no window configured) or a positive token count.

    The upper bound is a sanity bound on the configuration, not a model
    constraint: no current endpoint advertises more than a few million tokens.
    """
    if value is None:
        return None
    if not 1 <= value <= 10_000_000:
        raise ValueError("context_window_tokens must be None or 1..10000000")
    return value


def validate_output_reserve_tokens(value: int) -> int:
    if value < 1:
        raise ValueError("output_reserve_tokens must be positive")
    return value


def validate_max_output_tokens(value: int) -> int:
    if not 1 <= value <= 32000:
        raise ValueError("max_output_tokens must be 1..32000")
    return value


def validate_timeout(value: float) -> float:
    if not 0 < value <= 86400:
        raise ValueError("timeouts must be in (0, 86400]")
    return value


def validate_timeout_layering(http_timeout: float, model_timeout: float) -> None:
    """The transport deadline must fire before the outer ``wait_for`` deadline.

    When the two are equal, whichever wins is a scheduling race: sometimes the
    request fails as an httpx timeout, sometimes the coroutine is cancelled from
    outside. The attribution then differs run to run for the same endpoint
    failure.
    """
    if http_timeout >= model_timeout:
        raise ValueError(
            "http_timeout must be strictly less than model_timeout "
            f"(got http_timeout={http_timeout}, model_timeout={model_timeout}); "
            "the httpx deadline has to fire before the outer wait_for deadline, "
            "otherwise which one wins is a race and the failure attribution is "
            "not deterministic"
        )


@dataclass(frozen=True)
class AgentcoreSettings:
    """Everything the instance needs, with defaults matching the frozen profile."""

    model_id: str
    #: Model calls inside one user turn (not environment turns at run level).
    max_steps: int = 24
    model_settings: Any = None
    #: Directory that *contains* the ``agentcore`` package. Empty means the
    #: vendored copy.
    runtime_path: str = ""
    #: The model's context window in tokens. ``None`` disables trimming: the
    #: whole context goes out and the endpoint decides what fits.
    context_window_tokens: int | None = None
    output_reserve_tokens: int = 4096
    enable_memory: bool = False
    store_dir: str = ""
    litellm_params_extra: Mapping[str, Any] = field(default_factory=dict)
    context_policy: str = "capacity"
    context_compaction: str = "legacy"
    max_output_tokens: int = 2048
    max_completion_tokens: int | None = None
    model_timeout: float = 600.0
    http_timeout: float = 540.0
    tool_timeout: float = 600.0
    model_retries: int = 3
    max_argument_corrections: int | None = None
    max_protocol_corrections: int | None = None
    provider: str = "compatible"
    thinking: bool | None = None
    preserve_thinking: bool = False
    wire_model: str = ""
    api_base: str = ""
    read_only_actions: tuple[str, ...] = ()
    user_id: str = "exgentic"

    def __post_init__(self) -> None:
        validate_max_steps(self.max_steps)
        validate_context_window_tokens(self.context_window_tokens)
        validate_output_reserve_tokens(self.output_reserve_tokens)
        validate_max_output_tokens(self.max_output_tokens)
        for timeout in (self.model_timeout, self.http_timeout, self.tool_timeout):
            validate_timeout(timeout)
        validate_timeout_layering(self.http_timeout, self.model_timeout)
        if self.context_policy not in CONTEXT_POLICIES:
            raise ValueError(f"context_policy must be one of {sorted(CONTEXT_POLICIES)}")
        if self.context_compaction not in CONTEXT_COMPACTIONS:
            raise ValueError(
                f"context_compaction must be one of {sorted(CONTEXT_COMPACTIONS)}"
            )
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        if self.model_retries < 0:
            raise ValueError("model_retries must be nonnegative")
        for corrections in (
            self.max_argument_corrections,
            self.max_protocol_corrections,
        ):
            if corrections is not None and corrections < 0:
                raise ValueError("Correction limits must be nonnegative")
        if self.max_completion_tokens is not None and self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be a positive integer")

    def instance_kwargs(self, session_id: str) -> dict[str, Any]:
        """Constructor kwargs for ``AgentcoreAgentInstance``.

        Generated from the fields rather than hand-copied, so a field cannot be
        accepted by the settings and dropped on the way to the runner.
        """
        kwargs = {name: getattr(self, name) for name in self.__dataclass_fields__}
        kwargs["read_only_actions"] = list(self.read_only_actions)
        kwargs["litellm_params_extra"] = dict(self.litellm_params_extra)
        kwargs["session_id"] = session_id
        return kwargs
