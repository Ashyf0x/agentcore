# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""The transport deadline must fire strictly before the outer wall-clock deadline.

When ``http_timeout`` and ``model_timeout`` were equal (the old default pair was
600/600), whichever fired first was a scheduling race: the same endpoint hang
could surface as an httpx timeout on one run and as an outside cancellation of
the coroutine on the next, with different attribution. Layering the deadlines
makes the failure deterministic, and the connect/write phases get a short cap
so a dead endpoint fails fast instead of eating the whole budget.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exgentic.agents.agentcore.agentcore_agent import AgentcoreAgent
from exgentic.agents.agentcore.agentcore_instance import AgentcoreAgentInstance
from exgentic.agents.agentcore.core.model import ScriptedModel, phase_timeout


def test_defaults_lay_the_transport_deadline_below_the_model_deadline():
    agent = AgentcoreAgent()

    assert agent.http_timeout == pytest.approx(540.0)
    assert agent.http_timeout < agent.model_timeout


def test_equal_deadlines_are_rejected_at_the_config_surface():
    with pytest.raises(ValidationError, match="strictly less"):
        AgentcoreAgent(model_timeout=600.0, http_timeout=600.0)


def test_instance_rejects_deadlines_that_do_not_lay_out(tmp_path):
    with pytest.raises(ValueError, match="strictly less"):
        AgentcoreAgentInstance(
            session_id="s",
            model_id="openai/gpt-4.1",
            store_dir=str(tmp_path / "store"),
            api_base="https://example.invalid/v1",
            model_client=ScriptedModel([]),
            model_timeout=600.0,
            http_timeout=600.0,
        )


def test_httpx_phases_cap_connect_and_keep_the_read_budget():
    timeout = phase_timeout(600.0)

    assert timeout.read == 600.0
    assert timeout.connect == 60.0
    assert timeout.write == 60.0
    assert timeout.pool == 60.0
    # A budget shorter than the phase cap is not stretched by it.
    assert phase_timeout(10.0).connect == 10.0
    assert phase_timeout(10.0).read == 10.0
