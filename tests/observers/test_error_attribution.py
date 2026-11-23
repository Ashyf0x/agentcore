# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""What ``error_source`` says about a dead RPC, and why "agent" was wrong.

Three facts about the 2026-11-14 runs drove this file:

1. 24 sessions failed with an error message of exactly ``"timed out"``. Every
   one of them had run longer than the 600 s transport timeout, so the only
   thing that could have timed out was the RPC read -- the agent was waiting,
   not deciding.
2. All 24 were recorded as ``error_source="agent"``, because the label is
   derived from *which side raised* (``AgentError`` around the RPC), not from
   what actually failed.
3. ``compare_metrics.py`` classified the same 24 as ``infra`` by matching text.
   So the analysis tool and the runtime disagreed about the same sessions.

``on_session_error`` now runs the orchestrator's own transport predicate over
the wrapped exception, so a builtin ``TimeoutError`` is labelled ``infra`` on
both sides. The tests below pin the new rule, the wrapper case that already
worked, and the logic errors that must *not* be excused as infrastructure.
"""

from __future__ import annotations

import pytest

from exgentic.core.context import run_scope
from exgentic.core.orchestrator.failures import is_infra_failure
from exgentic.core.orchestrator.termination import AgentError, BenchmarkError


@pytest.fixture()
def obs(tmp_path):
    from exgentic.observers.handlers.results import ResultsObserver

    with run_scope(run_id="error-attribution", output_dir=str(tmp_path), cache_dir=str(tmp_path)):
        yield ResultsObserver(run_id="error-attribution")


@pytest.fixture()
def session():
    class _Session:
        session_id = "sess-err"
        task_id = "task-err"

    return _Session()


def _record(obs, session, error):
    """Run one error through the observer and return the metadata it filed.

    Reads ``_session_results`` rather than ``_sessions``: the latter holds
    per-session counters (``_SessionData``), while the filed record -- the one
    that carries ``details`` -- is the ``SessionResults`` appended there.

    ``error_source`` lives at ``details["session_metadata"]["error_source"]``,
    because ``details`` is ``score.model_dump()`` and the score keeps its
    metadata in a nested dict. A flatter read returns ``KeyError`` and would
    have made every assertion below fail for the wrong reason.
    """
    obs.on_session_start(session, None, None)
    obs.on_session_error(session, error)
    assert obs._session_results, "the observer filed nothing"
    return obs._session_results[-1].details["session_metadata"]


class TestBareTimeoutIsInfrastructure:
    def test_builtin_timeout_becomes_infra(self, obs, session):
        """The rpc read timeout: no ``error_source`` attribute to inherit.

        This is the case that produced 24 mislabelled sessions. ``TimeoutError``
        is a builtin, so the old "adopt the wrapper's attributedinfra" rule had
        nothing to read and fell through to the type-based default.
        """
        rec = _record(obs, session, AgentError(TimeoutError("timed out")))
        assert rec["error_source"] == "infra", (
            "a transport timeout must not be reported as an agent failure; "
            "the agent was waiting on a socket, it did not make a bad decision"
        )

    def test_the_message_is_preserved(self, obs, session):
        """Attribution changed; the text did not. Keep the evidence readable."""
        rec = _record(obs, session, AgentError(TimeoutError("timed out")))
        assert rec["error"] == "timed out"

    def test_connection_reset_becomes_infra(self, obs, session):
        rec = _record(obs, session, AgentError(ConnectionResetError("peer reset")))
        assert rec["error_source"] == "infra"

    def test_wrapped_timeout_is_seen_through_the_chain(self, obs, session):
        """A timeout re-raised as RuntimeError still counts.

        ``raise ... from`` is how most of this codebase surfaces a lower-level
        failure, so the predicate has to walk the chain rather than look once.
        """
        try:
            try:
                raise TimeoutError("timed out")
            except TimeoutError as inner:
                raise RuntimeError("rpc failed") from inner
        except RuntimeError as wrapped:
            rec = _record(obs, session, AgentError(wrapped))
        assert rec["error_source"] == "infra"


class TestLogicErrorsStayAgentFaults:
    @pytest.mark.parametrize(
        "exc",
        [ValueError("bad value"), KeyError("missing"), AttributeError("no such attr")],
        ids=["value-error", "key-error", "attribute-error"],
    )
    def test_logic_errors_are_not_excused(self, obs, session, exc):
        """The new rule must not launder agent bugs into infrastructure.

        Excusing everything would be worse than the bug being fixed: a
        genuine ``AttributeError`` in the agent would be retried forever.
        """
        rec = _record(obs, session, AgentError(exc))
        assert rec["error_source"] == "agent"

    def test_benchmark_side_logic_error_is_not_infra(self, obs, session):
        rec = _record(obs, session, BenchmarkError(ValueError("bad transcript")))
        assert rec["error_source"] == "benchmark"


class TestThePredicateItself:
    """The shared predicate is used by two call sites; pin its edges directly."""

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (TimeoutError("timed out"), True),
            (ConnectionResetError("reset"), True),
            (BrokenPipeError("pipe"), True),
            (ValueError("bad"), False),
            (RuntimeError("plain"), False),
        ],
    )
    def test_classification(self, exc, expected):
        assert is_infra_failure(exc) is expected

    def test_cycle_in_the_cause_chain_terminates(self):
        """A self-referential chain must not hang the recorder.

        Exceptions can be chained by hand, and a cycle would spin the walk
        forever -- inside an error handler, which is the worst place to hang.
        """
        a = ValueError("a")
        b = ValueError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert is_infra_failure(a) is False

    def test_the_same_predicate_serves_both_call_sites(self):
        """One definition, not two that can drift apart.

        ``session.py`` keeps the private alias for its own call site; both names
        must resolve to the same function object, or the two places that decide
        attribution could disagree again.
        """
        from exgentic.core.orchestrator import session as session_module

        assert session_module._is_infra_failure is is_infra_failure
