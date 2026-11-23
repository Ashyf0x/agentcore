# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Execution-policy tests for the vendored agentcore runtime.

``decide`` is the only place that decides whether a failed tool call is handed
back to the model or escalated to the user. The executor performs one dispatch
and never retries a tool behind the model's back.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from exgentic.agents.agentcore.core.errors import Decision, ErrorInfo, ToolFailure, decide
from exgentic.agents.agentcore.tools.executor import ToolRegistry, ToolSpec

PARAMETERS = {"type": "object", "properties": {}}


def _spec(**overrides):
    kwargs = {
        "name": "tool",
        "description": "d",
        "parameters": PARAMETERS,
        "handler": lambda **_: {"ok": True},
        "read_only": False,
    }
    kwargs.update(overrides)
    return ToolSpec(**kwargs)


def test_read_only_verdict_is_required():
    with pytest.raises(TypeError):
        ToolSpec(name="t", description="d", parameters=PARAMETERS, handler=lambda **_: None)


def test_interrupted_write_waits_for_a_human():
    info = ErrorInfo("execution", "transient", True, False, 1, read_only=False)
    assert decide(info) is Decision.WAIT


def test_interrupted_read_is_feedback_so_the_turn_continues():
    info = ErrorInfo("execution", "transient", True, False, 1, read_only=True)
    assert decide(info) is Decision.FEEDBACK


def test_known_transient_failure_returns_feedback_without_retrying():
    info = ErrorInfo("execution", "transient", True, True, 1, read_only=False)
    assert decide(info) is Decision.FEEDBACK


def test_terminal_categories_end_the_run():
    for category in ("internal", "storage", "cancelled"):
        info = ErrorInfo("execution", category, True, False, 1, read_only=True)
        assert decide(info) is Decision.TERMINATE


def test_missing_input_and_authorization_wait():
    for category in ("missing_input", "authorization"):
        info = ErrorInfo("validation", category, False, True, 1)
        assert decide(info) is Decision.WAIT


def test_registry_rejects_non_object_parameters():
    with pytest.raises(ValueError):
        ToolRegistry([_spec(parameters={"type": "string"})])


def test_registry_rejects_nonpositive_timeout():
    with pytest.raises(ValueError):
        ToolRegistry([_spec(timeout=0)])


@dataclass
class Call:
    name: str = "tool"
    arguments: str = "{}"
    id: str = "call-1"


@pytest.mark.asyncio
async def test_timeout_is_dispatched_once_and_never_retried():
    calls = 0

    async def handler(**_):
        nonlocal calls
        calls += 1
        raise TimeoutError

    spec = _spec(handler=handler, read_only=False, timeout=0.01)
    records = []
    result = await ToolRegistry([spec]).execute(Call(), lambda *row: records.append(row))

    assert calls == 1
    assert result.status == "unknown"
    assert result.attempts == 1
    assert result.decision is Decision.WAIT
    assert records[0][0] == 1


@pytest.mark.asyncio
async def test_read_timeout_is_handed_back_to_the_model():
    calls = 0

    async def handler(**_):
        nonlocal calls
        calls += 1
        raise TimeoutError

    spec = _spec(handler=handler, read_only=True, timeout=0.01)
    result = await ToolRegistry([spec]).execute(Call(), lambda *_: None)

    assert calls == 1
    assert result.decision is Decision.FEEDBACK


@pytest.mark.asyncio
async def test_known_transient_failure_is_not_retried():
    calls = 0

    async def handler(**_):
        nonlocal calls
        calls += 1
        raise ToolFailure("temporary", "transient")

    spec = _spec(handler=handler)
    result = await ToolRegistry([spec]).execute(Call(), lambda *_: None)

    assert calls == 1
    assert result.status == "returned"
    assert result.attempts == 1
    assert result.decision is Decision.FEEDBACK