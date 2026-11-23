# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Attribution of ``ContextBudgetError`` in ``Agent.run``.

The runtime raises ``ContextBudgetError`` when it cannot fit its *own* request
into the model's context window: the tool schemas, the accumulated evidence and
the current turn together exceed what the model accepts. Every one of those
conditions is a property of the request the runtime built, not of the endpoint's
health, so a retry reproduces the failure exactly.

``run`` used to catch it together with ``ModelError``/``TimeoutError`` and tag
the result ``infra``. That label told the operator to retry an unfixable task,
and it also leaked downstream: ``FatalRunError.error_source`` mirrors the
result, so the harness recorded a context-management fault as an endpoint
outage. Two sessions in the 2026-11-14 appworld run (``007c3743``,
``fe04972a``) were mislabelled this way before the split.

What is under test:

* ``ContextBudgetError`` -> ``error_source="agent"``;
* ``ModelError`` and ``TimeoutError`` keep ``"infra"`` -- the split must not
  have swept them along;
* the label survives the trip out as ``FatalRunError``, which is what the
  harness actually reads.
"""

from __future__ import annotations

import asyncio

import pytest

from exgentic.agents.agentcore.core.agent import Agent, AgentConfig
from exgentic.agents.agentcore.core.types import (
    ContextBudgetError,
    FatalRunError,
    ModelError,
)
from exgentic.agents.agentcore.storage.sqlite import Store
from exgentic.agents.agentcore.tools import ToolRegistry


class _BoomContext:
    """Stands in for ``ContextManager`` and fails on the first ``prepare``.

    ``prepare`` is where every ``ContextBudgetError`` in the real manager is
    raised, so raising from here reaches ``run`` through the same call site the
    production failure uses.
    """

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.calls = 0

    def prepare(self, *args, **kwargs):
        self.calls += 1
        raise self._error


class _UnusedModel:
    """A model that must never be called: the run fails before sending."""

    async def complete(self, request):  # pragma: no cover - asserted via calls
        raise AssertionError("the model must not be reached")


def _run(context, tmp_path, *, max_steps=4):
    """Drive ``Agent.run`` once and return (result, raised FatalRunError|None)."""
    store = Store(tmp_path / "store.sqlite3")
    agent = Agent(
        _UnusedModel(),
        ToolRegistry({}),
        store,
        config=AgentConfig(max_steps=max_steps, model_retries=0),
        context=context,
    )
    raised = None
    try:
        result = asyncio.run(agent.run("do the thing", user_id="u", session_id="s"))
    except FatalRunError as exc:
        raised = exc
        result = exc.result
    finally:
        store.close()
    return result, raised


class TestContextBudgetIsAnAgentFault:
    def test_result_is_labelled_agent(self, tmp_path):
        result, _ = _run(_BoomContext(ContextBudgetError("too big")), tmp_path)

        assert result.status == "failed"
        assert result.error_source == "agent"

    def test_reason_keeps_the_exception_type(self, tmp_path):
        result, _ = _run(_BoomContext(ContextBudgetError("too big")), tmp_path)

        assert result.reason == "ContextBudgetError"

    def test_the_failure_reaches_the_caller_as_fatal(self, tmp_path):
        """A failed run must not look like a finished conversation."""
        _, raised = _run(_BoomContext(ContextBudgetError("too big")), tmp_path)

        assert raised is not None

    def test_the_label_survives_the_trip_out(self, tmp_path):
        """``FatalRunError.error_source`` mirrors the result; the harness reads it."""
        _, raised = _run(_BoomContext(ContextBudgetError("too big")), tmp_path)

        assert raised.error_source == "agent"


class TestInfraFaultsStayInfra:
    """The split must not have carried the endpoint faults into ``agent``."""

    def test_model_error_stays_infra(self, tmp_path):
        error = ModelError("model HTTP request rejected", kind="capability")
        result, raised = _run(_BoomContext(error), tmp_path)

        assert result.error_source == "infra"
        assert raised.error_source == "infra"

    def test_timeout_error_stays_infra(self, tmp_path):
        result, raised = _run(_BoomContext(TimeoutError("no answer")), tmp_path)

        assert result.error_source == "infra"
        assert raised.error_source == "infra"


class TestTheBranchIsReachedBeforeTheModel:
    def test_prepare_failure_prevents_any_model_call(self, tmp_path):
        context = _BoomContext(ContextBudgetError("too big"))
        _run(context, tmp_path)

        assert context.calls == 1


@pytest.mark.parametrize(
    "message",
    [
        "Current request, tool evidence and schemas exceed the model's context window",
        "Output token reserve is smaller than the serialized output cap",
        "Request body is empty",
    ],
)
def test_every_context_budget_message_is_an_agent_fault(message, tmp_path):
    """All nine raise sites share one semantics: the request cannot be sent."""
    result, _ = _run(_BoomContext(ContextBudgetError(message)), tmp_path)

    assert result.error_source == "agent"
