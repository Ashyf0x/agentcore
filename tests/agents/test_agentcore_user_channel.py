# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A text-only ending on a tool-only benchmark is a stopped run, not a finished one.

Tool-only benchmarks (AppWorld, BFCL, ...) finish through their own tool. When
the model answers in text instead, the runtime used to record ``completed`` and
the harness saw a conversation that ended normally. With no channel to deliver
the text, the honest statement is ``failed / agent / no_user_channel``: it is
still scored, and the adapter log says why.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import BaseModel, create_model

from exgentic.agents.agentcore.agentcore_instance import AgentcoreAgentInstance
from exgentic.agents.agentcore.core.agent import Agent, AgentConfig
from exgentic.agents.agentcore.core.model import ScriptedModel
from exgentic.agents.agentcore.core.types import ModelResponse, Usage
from exgentic.agents.agentcore.storage.sqlite import Store
from exgentic.agents.agentcore.tools.executor import ToolRegistry
from exgentic.core.types import (
    ActionType,
    MessageObservation,
    MessagePayload,
    SingleAction,
)


def _text_response(text="done"):
    return ModelResponse(
        content=text,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(),
        usage_known=True,
    )


def _agent(tmp_path, model, *, user_channel):
    store = Store(tmp_path / "store.sqlite3")
    return Agent(
        model,
        ToolRegistry([]),
        store,
        config=AgentConfig(max_steps=2),
        user_channel=user_channel,
    ), store


@pytest.mark.asyncio
async def test_text_ending_without_a_channel_is_an_agent_failure(tmp_path):
    model = ScriptedModel([_text_response("all done")])
    agent, store = _agent(tmp_path, model, user_channel=False)

    result = await agent.run("go", user_id="u", session_id="s")

    assert result.status == "failed"
    assert result.error_source == "agent"
    assert result.reason == "no_user_channel"
    assert result.answer == "all done"
    # The terminal row says the same thing: no silent completion.
    row = store.db.execute(
        "SELECT status,result FROM runs WHERE id=?", (result.run_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert json.loads(row["result"])["reason"] == "no_user_channel"


@pytest.mark.asyncio
async def test_text_ending_with_a_channel_still_completes(tmp_path):
    model = ScriptedModel([_text_response("all done")])
    agent, _ = _agent(tmp_path, model, user_channel=True)

    result = await agent.run("go", user_id="u", session_id="s")

    assert result.status == "completed"
    assert result.error_source is None


# -- Through the real adapter -------------------------------------------------


class LookupArgs(BaseModel):
    query: str


def _lookup_action() -> ActionType:
    cls = create_model("lookup_Action", __base__=SingleAction, arguments=(LookupArgs, ...))
    return ActionType(name="lookup", description="lookup tool", cls=cls)


def test_adapter_records_no_user_channel_in_the_run_row(tmp_path):
    """The scored ending is written before the turn is released."""
    model = ScriptedModel([_text_response("nothing to do")])
    instance = AgentcoreAgentInstance(
        session_id="s",
        model_id="openai/gpt-4.1",
        store_dir=str(tmp_path / "store"),
        api_base="https://example.invalid/v1",
        model_client=model,
    )
    instance.start(task="t", context={}, actions=[_lookup_action()])
    observation = MessageObservation(
        invoking_actions=[], result=MessagePayload(sender="user", message="go")
    )

    try:
        assert instance.react(observation) is None
    finally:
        instance.close()

    with sqlite3.connect(f"file:{instance._store_path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT status,result FROM runs").fetchone()

    assert row["status"] == "failed"
    assert json.loads(row["result"])["reason"] == "no_user_channel"
