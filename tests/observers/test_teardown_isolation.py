# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""A session whose teardown fails must not take the rest of the run with it.

Two production crashes motivated these tests, both from the
``v2-full`` reruns (2026-11-14):

* **airline** aborted after 8 of 50 tasks.  The serve process died, and the
  next observer callback read ``session.session_id`` -- an HTTP RPC through
  ``ObjectProxy`` -- which raised ``httpx.RemoteProtocolError``.  That is
  neither a ``RuntimeError`` nor an ``AgentError``, so it passed every handler
  in ``run_session`` and reached the executor's ``except Exception``, which
  cancels every pending session.

* **telecom** aborted after 37 of 114.  ``score()`` re-raised a tau2 runner
  error as a bare ``RuntimeError`` from *inside* an ``except`` block, where the
  sibling ``except RuntimeError`` cannot see it.

The invariant both violated: a failure while *filing* a session's result says
nothing about the other N-1 sessions, so it must stay inside that session.
"""

from __future__ import annotations

import httpx
import pytest
from exgentic.core.orchestrator.session import _is_infra_failure
from exgentic.core.orchestrator.termination import AgentError, BenchmarkError
from exgentic.core.types import SessionOutcomeStatus
from exgentic.observers.handlers.results import ResultsObserver


# ---------------------------------------------------------------------------
# Attribution: which failures are infrastructure, not benchmark logic
# ---------------------------------------------------------------------------


class TestInfraFailureClassification:
    def test_httpx_transport_error_is_infra(self):
        assert _is_infra_failure(httpx.RemoteProtocolError("peer gone")) is True

    def test_connect_error_is_infra(self):
        assert _is_infra_failure(httpx.ConnectError("refused")) is True

    def test_builtin_connection_error_is_infra(self):
        assert _is_infra_failure(ConnectionError("reset")) is True

    def test_timeout_is_infra(self):
        assert _is_infra_failure(TimeoutError("slow")) is True

    def test_wrapped_cause_is_followed(self):
        """tau2 buries the real cause one level down via ``raise ... from``."""
        cause = httpx.RemoteProtocolError("peer gone")
        wrapper = RuntimeError("TAU2 runner thread failed")
        wrapper.__cause__ = cause
        assert _is_infra_failure(wrapper) is True

    def test_pure_logic_error_is_not_infra(self):
        assert _is_infra_failure(ValueError("AssistantMessage must have content")) is False

    def test_self_referential_cause_terminates(self):
        """A cyclic cause chain must not hang the classifier."""
        exc = RuntimeError("loop")
        exc.__cause__ = exc
        assert _is_infra_failure(exc) is False


# ---------------------------------------------------------------------------
# Adoption: a BenchmarkError carrying ``infra`` must be believed
# ---------------------------------------------------------------------------


class _InfraRuntime(RuntimeError):
    error_source = "infra"


def _score_for(error: Exception):
    observer = ResultsObserver()
    captured = []
    observer._record_session = lambda session, score, agent=None: captured.append(score)
    observer.on_session_error(session=None, error=error)
    assert len(captured) == 1
    return captured[0]


class TestBenchmarkSideInfraAdoption:
    """``_score_or_error`` wraps teardown failures in ``BenchmarkError``.

    Before this rule, only ``AgentError`` could upgrade to ``infra`` -- so an
    endpoint dropout during scoring was filed as the benchmark's fault and the
    run looked like a benchmark bug instead of a retryable infrastructure one.
    """

    def test_benchmark_error_can_carry_infra(self):
        score = _score_for(BenchmarkError(_InfraRuntime("serve died")))
        assert score.session_metadata["error_source"] == "infra"

    def test_benchmark_error_without_source_stays_benchmark(self):
        score = _score_for(BenchmarkError(RuntimeError("unscorable transcript")))
        assert score.session_metadata["error_source"] == "benchmark"

    def test_infra_still_resolves_to_error_status(self):
        score = _score_for(BenchmarkError(_InfraRuntime("serve died")))
        assert ResultsObserver()._resolve_session_status(score) is SessionOutcomeStatus.ERROR

    def test_agent_error_still_adopts_infra(self):
        # Regression guard: widening the rule must not break the original case.
        score = _score_for(AgentError(_InfraRuntime("endpoint timed out")))
        assert score.session_metadata["error_source"] == "infra"


# ---------------------------------------------------------------------------
# Observer must never RPC for the ids
# ---------------------------------------------------------------------------


class _DeadProxySession:
    """Stands in for an ``ObjectProxy`` whose peer is gone.

    Any attribute read that was not pinned locally raises, exactly as
    ``httpx.RemoteProtocolError`` does once the serve process exits.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_exgentic_session_id", "sess-dead")
        object.__setattr__(self, "_exgentic_task_id", "task-dead")

    def __getattr__(self, name):  # only called for missing attributes
        raise httpx.RemoteProtocolError(f"peer died before {name!r} could be read")


class TestObserverSurvivesDeadPeer:
    def test_pinned_ids_are_used_without_rpc(self):
        """The crash path: filing a failure for a session whose peer just died."""
        observer = ResultsObserver()
        captured = []
        observer._record_session = lambda session, score, agent=None: captured.append(score)
        session = _DeadProxySession()

        observer.on_session_error(session=session, error=BenchmarkError(RuntimeError("boom")))

        assert len(captured) == 1
        assert captured[0].session_metadata["error"] == "boom"

    def test_dead_peer_without_pinned_ids_still_fails_loudly(self):
        """Guard against silently inventing an id: no pin, no write."""

        class _Unpinned(_DeadProxySession):
            pass

        session = _Unpinned()
        object.__delattr__(session, "_exgentic_session_id")

        observer = ResultsObserver()
        with pytest.raises(httpx.RemoteProtocolError):
            observer.on_session_error(session=session, error=BenchmarkError(RuntimeError("boom")))
