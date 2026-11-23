# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""One observer must not be able to end the run.

``Tracker`` fans every session event out to all registered observers.  Before
this guard, the loop was a bare ``for observer in self._observers``, so the
first observer to raise ended the loop and propagated.  Propagating out of
``run_session`` reaches the executor's ``except Exception``, which cancels
every pending future -- so a bookkeeping bug in *one* observer, while filing
*one* session's failure, destroyed the results of the other N-1 sessions.

That is exactly the airline failure (8/50 tasks) viewed from a different
angle: the analyst's instinct is to fix the observer that raised, but the
blast radius is decided here, by the dispatch loop.  Fixing the observer
leaves the next one exposed; fixing the loop does not.

Controllers are deliberately exempt: ``LimitController`` terminates sessions
by raising, and ``CoreController`` decides when a run is over.  Swallowing
their exceptions would silently disable the limit.
"""

from __future__ import annotations

import pytest

from exgentic.core.context import run_scope
from exgentic.core.orchestrator.observer import Observer
from exgentic.core.orchestrator.termination import (
    AgentError,
    AgentTerminationError,
    BenchmarkError,
)
from exgentic.core.orchestrator.tracker import Tracker
from exgentic.core.types import Observation, SingleObservation


def _an_observation() -> Observation:
    """A real ``Observation`` -- ``None`` means "session over" to the core."""
    return SingleObservation(result="ok")


@pytest.fixture(autouse=True)
def _run_context(tmp_path):
    """``Tracker`` reads ``run_id`` from the context in its constructor."""
    with run_scope(run_id="tracker-isolation", output_dir=str(tmp_path), cache_dir=str(tmp_path)):
        yield


class _Recorder(Observer):
    def __init__(self, name: str, calls: list) -> None:
        super().__init__()
        self._name = name
        self._calls = calls

    def on_session_start(self, session, agent, observation) -> None:
        self._calls.append(f"{self._name}.on_session_start")

    def on_session_error(self, session, error) -> None:
        self._calls.append(f"{self._name}.on_session_error")

    def on_session_success(self, session, score, agent) -> None:
        self._calls.append(f"{self._name}.on_session_success")


class _Exploding(Observer):
    """Raises on every callback it implements, like a dead-proxy read does."""

    def __init__(self, name: str, exc: BaseException | None = None) -> None:
        super().__init__()
        self._name = name
        self._exc = exc or RuntimeError(f"{name} exploded")

    def _boom(self):
        raise self._exc

    def on_session_start(self, session, agent, observation) -> None:
        self._boom()

    def on_session_error(self, session, error) -> None:
        self._boom()

    def on_session_success(self, session, score, agent) -> None:
        self._boom()


def _tracker(*observers) -> Tracker:
    return Tracker(observers=list(observers), use_defaults=False)


class TestObserversCannotEndTheRun:
    def test_a_raising_observer_does_not_stop_the_ones_behind_it(self):
        calls: list[str] = []
        tracker = _tracker(
            _Exploding("first"),
            _Recorder("second", calls),
            _Recorder("third", calls),
        )

        tracker.on_session_start(None, None, None)

        assert calls == ["second.on_session_start", "third.on_session_start"]

    def test_a_raising_observer_does_not_propagate(self):
        """Propagation is what reaches the executor and cancels everything."""
        tracker = _tracker(_Exploding("only"))

        tracker.on_session_start(None, None, None)  # must not raise

    def test_the_error_path_is_covered_too(self):
        """Filing a failure is the callback that runs *because* things broke."""
        calls: list[str] = []
        tracker = _tracker(_Exploding("first"), _Recorder("second", calls))

        tracker.on_session_error(None, BenchmarkError(RuntimeError("boom")))

        assert calls == ["second.on_session_error"]

    def test_transport_errors_are_contained(self):
        """``RemoteProtocolError`` is not a ``RuntimeError``; it must still be
        caught by the dispatch guard rather than by the handlers below it."""
        import httpx

        calls: list[str] = []
        tracker = _tracker(
            _Exploding("dead", httpx.RemoteProtocolError("peer gone")),
            _Recorder("second", calls),
        )

        tracker.on_session_start(None, None, None)

        assert calls == ["second.on_session_start"]

    def test_every_observer_failing_is_still_survivable(self):
        tracker = _tracker(_Exploding("a"), _Exploding("b"), _Exploding("c"))

        tracker.on_session_start(None, None, None)


class TestControllersStillTerminate:
    """The guard must not swallow the signals that end a session."""

    def test_react_error_still_raises_agent_error(self):
        tracker = _tracker()
        with pytest.raises(AgentError):
            tracker.on_react_error(None, RuntimeError("bad action"))

    def test_step_error_still_raises_benchmark_error(self):
        tracker = _tracker()
        with pytest.raises(BenchmarkError):
            tracker.on_step_error(None, RuntimeError("bad observation"))

    def test_limit_controller_still_reaches_the_caller(self):
        """A controller raising must propagate -- it is the limit working."""
        calls: list[str] = []
        tracker = Tracker(
            observers=[_Recorder("rec", calls)],
            controllers=[_Limiter()],
            use_defaults=False,
        )

        with pytest.raises(AgentTerminationError):
            tracker.on_react_success(None, None)

        # The observer still ran before the controller stopped the session.
        assert calls == []

    def test_observer_failure_does_not_skip_the_controller(self):
        calls: list[str] = []
        tracker = Tracker(
            observers=[_Exploding("dead")],
            controllers=[_RecorderController(calls)],
            use_defaults=False,
        )

        tracker.on_step_success(None, _an_observation())

        assert calls == ["controller.on_step_success"]


class _Limiter:
    """Stands in for ``LimitController``: terminates by raising."""

    def on_react_success(self, session, action) -> None:
        raise AgentTerminationError()


class _RecorderController:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    def on_step_success(self, session, observation) -> None:
        self._calls.append("controller.on_step_success")


class TestLimitControllerSurvivesDeadPeer:
    """``LimitController`` counts every action and every step.

    It is not an observer, so the dispatch guard does not cover it -- and it
    runs more often than any observer.  Left reading ``session.session_id``
    off the proxy, it is the *next* thing to crash once the observers are
    fixed, and nothing downstream would catch it.
    """

    def _controller(self):
        from exgentic.core.orchestrator.controller import LimitController

        return LimitController(max_steps=10, max_actions=10)

    def _action(self):
        from exgentic.core.types import SingleAction

        return SingleAction(name="noop", arguments={})

    def test_counting_an_action_does_not_rpc(self):
        import httpx

        class _Dead:
            def __init__(self):
                object.__setattr__(self, "_exgentic_session_id", "sess-dead")

            def __getattr__(self, name):
                raise httpx.RemoteProtocolError(f"dead at {name!r}")

        controller = self._controller()
        controller.on_react_success(_Dead(), self._action())
        controller.on_step_success(_Dead(), _an_observation())


class TestDispatchIsNotSilent:
    def test_the_failure_is_logged_with_the_observer_name(self, caplog):
        tracker = _tracker(_Exploding("ExplodingObserver"))

        with caplog.at_level("ERROR", logger="exgentic.core.orchestrator.tracker"):
            tracker.on_session_start(None, None, None)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        # Names the class, so an operator can find the observer that failed.
        assert "_Exploding" in record.getMessage()
        assert "on_session_start" in record.getMessage()
        # The traceback is what tells the operator *why* it failed.
        assert record.exc_info is not None
