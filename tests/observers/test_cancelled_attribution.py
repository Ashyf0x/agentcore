# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""The cancellation rule, and the metric it quietly moves.

``on_session_error`` sets ``error_source`` from the exception type, then
overrides it to ``"cancelled"`` for ``SessionCancelError`` / ``RunCancelError``
/ ``KeyboardInterrupt``. A ``SessionScore`` carrying that label resolves to
``SessionOutcomeStatus.CANCELLED`` rather than ``ERROR``.

That rule had no test. It is worth one because ``error_source`` decides which
of two published metrics a session lands in, and the mapping is not obvious:

* ``status`` is a *label* for display.
* The published buckets are computed from ``is_finished``, not from ``status``:
  ``percent_error`` counts ``is_finished is None`` and ``percent_unfinished``
  counts ``is_finished is False``.
* ``on_session_error`` builds every score -- cancelled included -- with
  ``is_finished=None``.

So a cancelled session renders as ``cancelled`` and still counts toward
``percent_error``. That is a real (if debatable) decision, not a bug to fix
here: ``is_finished`` answers "did this task complete", and a cancelled task
did not. The tests below pin the behaviour so any future change to it is a
deliberate one, and record the consequence in the assertion message rather
than leaving it to be rediscovered from a confusing run summary.
"""

from __future__ import annotations

import pytest

from exgentic.core.context import run_scope
from exgentic.core.orchestrator.termination import (
    AgentError,
    BenchmarkError,
    RunCancelError,
    SessionCancelError,
)
from exgentic.core.types import SessionOutcomeStatus, SessionScore


@pytest.fixture()
def obs(tmp_path):
    from exgentic.observers.handlers.results import ResultsObserver

    # ``_record_session`` writes results.json and resolves paths through the
    # active context, so the observer needs a run scope even when the test only
    # cares about the in-memory outcome.
    with run_scope(run_id="cancel-attribution", output_dir=str(tmp_path), cache_dir=str(tmp_path)):
        yield ResultsObserver(run_id="cancel-attribution")


@pytest.fixture()
def session():
    """A minimal session: the callbacks only need an id and a reason slot."""

    class _Session:
        session_id = "sess-cancel"
        task_id = "task-cancel"

    return _Session()


def _score(metadata):
    return SessionScore(score=0, success=False, is_finished=None, session_metadata=metadata)


def _start(obs, session):
    """Register the session the way the orchestrator does before any callback.

    ``_set_reason`` only writes into an entry ``on_session_start`` created, so
    skipping this makes every reason assertion silently pass on an empty dict --
    which is exactly how the reason text went untested in the first place.
    """
    obs.on_session_start(session, None, None)
    return session


class TestCancellationIsLabelled:
    @pytest.mark.parametrize(
        "error",
        [SessionCancelError(), RunCancelError(), KeyboardInterrupt()],
        ids=["session-cancel", "run-cancel", "keyboard-interrupt"],
    )
    def test_cancel_exceptions_all_become_cancelled(self, obs, error):
        assert obs._resolve_session_status(_score({"error_source": "cancelled"})) == (
            SessionOutcomeStatus.CANCELLED
        )

    def test_cancelled_wins_over_other_labels(self, obs):
        """A cancelled session is never reported as an ordinary error.

        ``on_session_error`` computes the type-based label first and overrides
        it for cancellation, because "we stopped it" and "it broke" call for
        different responses even though both are failures.
        """
        metadata = {"error": "boom", "error_source": "cancelled", "limit_reached": True}
        assert obs._resolve_session_status(_score(metadata)) == SessionOutcomeStatus.CANCELLED

    def test_cancelled_is_not_success_even_when_score_says_finished(self, obs):
        score = SessionScore(
            score=1.0, success=True, is_finished=True,
            session_metadata={"error_source": "cancelled"},
        )
        assert obs._resolve_session_status(score) == SessionOutcomeStatus.CANCELLED


class TestTheOtherErrorSourcesStillResolveToError:
    """Guard the neighbours: adopting ``cancelled`` must not blur the rest."""

    @pytest.mark.parametrize("source", ["agent", "benchmark", "infra"])
    def test_error_sources_map_to_error(self, obs, source):
        assert obs._resolve_session_status(_score({"error_source": source})) == (
            SessionOutcomeStatus.ERROR
        )


class TestCancelledStillCountsAsError:
    """The consequence nobody expects: a label, not a bucket.

    These assertions are the point of the file. If a future change gives
    cancellation its own bucket, this class fails and forces the decision to be
    made on purpose -- including whatever it does to ``percent_error``.
    """

    def test_cancelled_session_has_is_finished_none(self, obs, session):
        """The callback itself is what puts it in the error bucket."""
        _start(obs, session)
        obs.on_session_error(session, SessionCancelError())
        recorded = obs.session_results()
        assert len(recorded) == 1
        assert recorded[0].status == SessionOutcomeStatus.CANCELLED
        assert recorded[0].is_finished is None, (
            "percent_error counts (is_finished is None); a cancelled session "
            "has not finished, so it lands there too"
        )

    def test_non_cancel_error_also_has_is_finished_none(self, obs, session):
        """Both branches agree on is_finished; only the label differs."""
        _start(obs, session)
        obs.on_session_error(session, AgentError(RuntimeError("boom")))
        recorded = obs.session_results()
        assert recorded[0].status == SessionOutcomeStatus.ERROR
        assert recorded[0].is_finished is None

    def test_benchmark_error_is_also_none(self, obs, session):
        _start(obs, session)
        obs.on_session_error(session, BenchmarkError(RuntimeError("boom")))
        assert obs.session_results()[0].is_finished is None

    def test_cancelled_and_error_are_indistinguishable_to_the_buckets(self, obs, session):
        """Same ``is_finished``, so the two published metrics cannot tell them apart.

        This is the assertion that would have prevented the confusion: the
        ``status`` field distinguishes them, the percentages do not.
        """

        class _Other:
            session_id = "sess-error"
            task_id = "task-error"

        _start(obs, session)
        obs.on_session_error(session, SessionCancelError())
        _start(obs, _Other())
        obs.on_session_error(_Other(), AgentError(RuntimeError("boom")))
        by_status = {r.status: r for r in obs.session_results()}

        cancelled = by_status[SessionOutcomeStatus.CANCELLED]
        errored = by_status[SessionOutcomeStatus.ERROR]
        assert cancelled.is_finished == errored.is_finished is None


class TestReasonTextMatchesTheLabel:
    """``_set_reason`` builds the human-readable cause; it must agree with the label.

    The reason is what ends up in the console recap and in ``details``, so a
    cancelled session described as "unexpected exception" is misleading even
    though the status is right.
    """

    def test_cancel_reason_says_cancelled(self, obs, session):
        _start(obs, session)
        obs.on_session_error(session, SessionCancelError())
        assert "cancel" in (obs.session_results()[0].details.get("reason") or "").lower()

    def test_infra_reason_says_infrastructural(self, obs, session):
        class InfraError(RuntimeError):
            error_source = "infra"

        _start(obs, session)
        obs.on_session_error(session, AgentError(InfraError("connection reset")))
        recorded = obs.session_results()[0]
        assert recorded.status == SessionOutcomeStatus.ERROR
        assert "infrastructural" in (recorded.details.get("reason") or "").lower()
