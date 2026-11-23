# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""An agent that knows a fault was infrastructural must not be blamed for it.

agentcore distinguishes ``infra`` (endpoint down, timeout, storage dead) from
``agent`` (the runtime worked, the agent just did not solve the task). The
harness only sees an exception, so ``on_session_error`` re-derives the cause
from the exception type -- which always says "agent". These tests pin the
narrow exception to that rule: an ``infra`` attribution carried by the root
error is adopted, everything else is left alone.

The scoring outcome is deliberately *not* part of this: both labels resolve to
``SessionOutcomeStatus.ERROR`` and neither is scored. This only affects which
cause is displayed and logged.
"""

from exgentic.core.orchestrator.termination import AgentError
from exgentic.core.types import SessionOutcomeStatus, SessionScore
from exgentic.observers.handlers.results import ResultsObserver


class _InfraError(RuntimeError):
    """Stands in for ``FatalRunError`` without importing agentcore."""

    error_source = "infra"


class _AgentError(RuntimeError):
    error_source = "agent"


class _NoSourceError(RuntimeError):
    pass


def _run(session_error: Exception) -> SessionScore:
    observer = ResultsObserver()
    captured: list[SessionScore] = []
    observer._record_session = lambda session, score, agent=None: captured.append(score)
    observer.on_session_error(session=None, error=session_error)
    assert len(captured) == 1
    return captured[0]


class TestInfraAttribution:
    def test_infra_source_is_adopted(self):
        score = _run(AgentError(_InfraError("endpoint timed out")))
        assert score.session_metadata["error_source"] == "infra"

    def test_agent_source_stays_agent(self):
        score = _run(AgentError(_AgentError("step limit")))
        assert score.session_metadata["error_source"] == "agent"

    def test_missing_source_falls_back_to_agent(self):
        score = _run(AgentError(_NoSourceError("boom")))
        assert score.session_metadata["error_source"] == "agent"

    def test_bare_exception_is_not_upgraded(self):
        # No AgentError wrapper: there is no root error to read a source from.
        score = _run(_InfraError("loose"))
        assert "error_source" not in score.session_metadata

    def test_error_message_is_preserved(self):
        score = _run(AgentError(_InfraError("endpoint timed out")))
        assert "endpoint timed out" in score.session_metadata["error"]


class TestInfraStatusResolution:
    def test_infra_resolves_to_error(self):
        score = SessionScore(
            score=0,
            success=False,
            is_finished=None,
            session_metadata={"error": "x", "error_source": "infra"},
        )
        assert ResultsObserver()._resolve_session_status(score) is SessionOutcomeStatus.ERROR

    def test_infra_is_not_scored(self):
        # ``success=False`` + ``is_finished=None`` must never become
        # ``unsuccessful``: that status counts in the denominator.
        score = SessionScore(
            score=0,
            success=False,
            is_finished=None,
            session_metadata={"error": "x", "error_source": "infra"},
        )
        assert ResultsObserver()._resolve_session_status(score) is not SessionOutcomeStatus.UNSUCCESSFUL
