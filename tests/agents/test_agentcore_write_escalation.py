# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""An unknown write outcome must be escalated once, not wedge the session.

``Store.unresolved`` used to report the same rows at the start of *every*
``Agent.run``, and nothing could ever clear them: after a single write timeout
the runtime returned ``waiting_user`` before every model call, forever. The
escalation is now an append-only ledger -- the runtime surfaces the unknown
outcome once, records the delivery, and later turns proceed with the write
visible in ``run_state.unresolved_writes`` so the model can verify it. The
write itself is still never replayed.
"""

from __future__ import annotations

import pytest

from exgentic.agents.agentcore.core.agent import Agent, AgentConfig
from exgentic.agents.agentcore.core.model import ScriptedModel
from exgentic.agents.agentcore.core.types import ModelResponse, ToolCall, Usage
from exgentic.agents.agentcore.storage.sqlite import Store
from exgentic.agents.agentcore.tools.executor import ToolRegistry, ToolSpec

PARAMETERS = {"type": "object", "properties": {}}
USER, SESSION = "u", "s"


def _response(*, text="done", tool=None):
    calls = [] if tool is None else [ToolCall("call-1", tool, {})]
    return ModelResponse(
        content=None if calls else text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(input_tokens=1, output_tokens=1),
        usage_known=True,
    )


def _agent(tmp_path, model, *, handler=None, read_only=False):
    store = Store(tmp_path / "store.sqlite3")
    spec = ToolSpec(
        name="place_order",
        description="Write tool",
        parameters=PARAMETERS,
        handler=handler or (lambda **_: {"ok": True}),
        read_only=read_only,
        timeout=0.01,
    )
    agent = Agent(model, ToolRegistry([spec]), store, config=AgentConfig(max_steps=4))
    return agent, store


@pytest.mark.asyncio
async def test_write_timeout_escalates_once_and_the_next_turn_proceeds(tmp_path):
    def handler(**_):
        raise TimeoutError

    model = ScriptedModel([_response(tool="place_order"), _response(text="resumed")])
    agent, store = _agent(tmp_path, model, handler=handler)

    first = await agent.run("do it", user_id=USER, session_id=SESSION)
    assert first.status == "waiting_user"
    assert first.reason == "transient"

    # The delivery was recorded with the turn that announced it, so the next run
    # is not stopped by the same row again. The runtime replaces provider call
    # ids with its own, so match on the tool name, not on "call-1".
    assert store.unresolved(USER, SESSION) == []
    assert [row["name"] for row in store.unknown_writes(USER, SESSION)] == ["place_order"]

    second = await agent.run("carry on", user_id=USER, session_id=SESSION)
    assert second.status == "completed"

    system = model.requests[1]["messages"][0]["content"]
    assert "unresolved_writes" in system
    assert "place_order" in system


@pytest.mark.asyncio
async def test_crash_leftover_is_escalated_once_then_cleared(tmp_path):
    """A process that died mid-dispatch leaves an attempt row and no answer."""
    model = ScriptedModel([_response(text="checked")])
    agent, store = _agent(tmp_path, model)
    store.attempt(
        "run-x", USER, SESSION, "call-x", "place_order", False, 1, "unknown",
        {"phase": "dispatch_intent"},
    )

    first = await agent.run("one", user_id=USER, session_id=SESSION)
    assert first.status == "waiting_user"
    assert first.reason == "unresolved_write"
    assert model.requests == []  # the model was not called

    second = await agent.run("two", user_id=USER, session_id=SESSION)
    assert second.status == "completed"
    system = model.requests[0]["messages"][0]["content"]
    assert "call-x" in system
    assert "place_order" in system


@pytest.mark.asyncio
async def test_unknown_read_does_not_escalate(tmp_path):
    """An interrupted read has no side effects to confirm; it never blocks."""
    model = ScriptedModel([_response(text="fine")])
    agent, store = _agent(tmp_path, model)
    store.attempt("run-x", USER, SESSION, "call-x", "place_order", True, 1, "unknown", {})

    result = await agent.run("go", user_id=USER, session_id=SESSION)

    assert result.status == "completed"
    assert "unresolved_writes" not in model.requests[0]["messages"][0]["content"]


def test_escalation_ledger_is_idempotent(tmp_path):
    store = Store(tmp_path / "store.sqlite3")
    store.attempt("r", USER, SESSION, "c", "place_order", False, 1, "unknown", {})

    assert [row["call"] for row in store.unresolved(USER, SESSION)] == ["c"]

    store.escalate_writes(USER, SESSION, "r", ["c"])
    store.escalate_writes(USER, SESSION, "r", ["c"])

    assert store.unresolved(USER, SESSION) == []
    # Escalated is not forgotten: the context view keeps reporting it.
    assert [row["call"] for row in store.unknown_writes(USER, SESSION)] == ["c"]
