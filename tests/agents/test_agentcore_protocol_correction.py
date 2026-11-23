# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A malformed reply must be correctable even when the provider omits usage.

The correction budget used to be gated on ``exc.usage is not None``, and that
field is only populated when the server reports token usage. A gateway that
omits usage therefore lost the whole correction path: one malformed response
became a fatal run, even though the next attempt would have succeeded.
"""

from __future__ import annotations

import pytest

from exgentic.agents.agentcore.core.agent import Agent, AgentConfig
from exgentic.agents.agentcore.core.model import ScriptedModel
from exgentic.agents.agentcore.core.types import FatalRunError, ModelResponse, Usage
from exgentic.agents.agentcore.storage.sqlite import Store
from exgentic.agents.agentcore.tools.executor import ToolRegistry


def _response(content="done", *, usage_known=False):
    return ModelResponse(
        content=content,
        tool_calls=[],
        finish_reason="stop",
        usage=Usage(),
        usage_known=usage_known,
    )


def _agent(tmp_path, model, **config):
    store = Store(tmp_path / "store.sqlite3")
    return Agent(model, ToolRegistry([]), store, config=AgentConfig(max_steps=3, **config)), store


@pytest.mark.asyncio
async def test_protocol_correction_needs_no_usage_field(tmp_path):
    # The first reply is empty, which is a protocol error; the provider reports
    # no usage. The correction re-prompts and the second reply completes.
    model = ScriptedModel([_response(""), _response("done")])
    agent, store = _agent(tmp_path, model)

    result = await agent.run("hi", user_id="u", session_id="s")

    assert result.status == "completed"
    kinds = [event["kind"] for event in store.events(result.run_id)]
    assert "protocol_correction" in kinds
    assert len(model.requests) == 2


@pytest.mark.asyncio
async def test_without_a_correction_budget_the_malformed_reply_is_fatal(tmp_path):
    model = ScriptedModel([_response("")])
    agent, _ = _agent(tmp_path, model, max_protocol_corrections=0)

    with pytest.raises(FatalRunError):
        await agent.run("hi", user_id="u", session_id="s")
