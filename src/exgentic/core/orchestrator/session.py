# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from ..types import SessionConfig
from .controller import Controller
from .failures import is_infra_failure as _is_infra_failure
from .observer import Observer
from .termination import (
    AgentError,
    AgentTerminationError,
    BenchmarkError,
    BenchmarkTerminationError,
    RunCancelError,
    SessionCancelError,
    SessionLimitReachedError,
)
from .tracker import Tracker


def _close_session_agent(session, agent_instance) -> None:
    session.close()
    agent_instance.close()


def _score_or_error(session, tracker: Tracker):
    """Score the session, turning a scoring failure into a session error.

    ``score()`` talks to the benchmark after the run has ended, so it can fail
    for reasons that have nothing to do with the agent: a benchmark runner
    thread that died mid-turn (tau2 re-raises it here as ``RuntimeError``), a
    message the benchmark's own validator rejects, a serve process that dropped
    the connection while writing its checkpoint.

    Left unguarded, such a failure propagates out of the ``except`` block that
    called it, escapes ``run_session`` entirely, and is caught by the executor
    loop -- which cancels every pending session and aborts the whole run. One
    bad transcript then destroys the other N-1 tasks' results.

    Returns ``None`` after recording the failure, so the caller closes up and
    lets the run continue.
    """
    try:
        return session.score()
    except Exception as exc:  # noqa: BLE001 -- see docstring
        # A scoring failure is a teardown failure -- the transcript is gone
        # either way, and one unscorable transcript must not void the other
        # N-1 results.  What still matters is attribution: tag the cause so
        # the observers can record ``infra`` instead of blaming the benchmark
        # for an endpoint that dropped.
        source = "infra" if _is_infra_failure(exc) else "benchmark"
        try:
            exc.error_source = source
        except Exception:  # noqa: BLE001 -- builtins with __slots__ refuse
            try:
                object.__setattr__(exc, "error_source", source)
            except Exception:  # noqa: BLE001
                pass
        tracker.on_session_error(session, BenchmarkError(exc))
        return None


def run_session(
    session_config: SessionConfig,
    session,
    agent,
    observers: list[Observer] | None = None,
    controllers: list[Controller] | None = None,
    *,
    tracker: Tracker | None = None,
) -> None:
    """Process a single session.

    The *agent_instance* is created via ``agent.get_instance()`` for
    runner isolation.
    """
    if tracker is None:
        tracker = Tracker(observers=observers, controllers=controllers)

    # session_scope is entered by the caller (run_session_config); role
    # is baked into each service at spawn time via per-service runtime.json.
    tracker.on_session_creation(session)

    try:
        agent_instance = agent.get_instance(session_id=session.session_id)

        observation = session.start()

        agent_instance.start(
            task=session.task,
            context=session.context,
            actions=session.actions,
        )

        tracker.on_session_start(session, agent_instance, observation)
        while not session.done() and observation is not None:
            try:
                action = agent_instance.react(observation)
            except Exception as exc:
                tracker.on_react_error(session, exc)

            tracker.on_react_success(session, action)

            try:
                observation = session.step(action)
            except Exception as exc:
                tracker.on_step_error(session, exc)

            tracker.on_step_success(session, observation)

        if session.done():
            raise BenchmarkTerminationError()
        raise AgentTerminationError()

    except KeyboardInterrupt:
        tracker.on_session_error(session, RunCancelError())
        _close_session_agent(session, agent_instance)
        raise
    except AgentError as exc:
        tracker.on_session_error(session, exc)
        _close_session_agent(session, agent_instance)
    except SessionLimitReachedError as exc:
        tracker.on_session_scoring(session)
        score = _score_or_error(session, tracker)
        if score is None:
            _close_session_agent(session, agent_instance)
            return
        score.is_finished = False
        score.session_metadata = {
            **(score.session_metadata or {}),
            "limit_reached": True,
            "limit_reason": exc.reason,
            "max_steps": exc.max_steps,
            "max_actions": exc.max_actions,
            "steps": exc.steps,
            "actions": exc.actions,
        }
        tracker.on_session_success(session, score, agent_instance)
        _close_session_agent(session, agent_instance)
    except (AgentTerminationError, BenchmarkTerminationError):
        tracker.on_session_scoring(session)
        score = _score_or_error(session, tracker)
        if score is None:
            _close_session_agent(session, agent_instance)
            return
        if score.is_finished is None:
            score.is_finished = True
        tracker.on_session_success(session, score, agent_instance)
        _close_session_agent(session, agent_instance)
    except BenchmarkError as exc:
        tracker.on_session_error(session, exc)
        # Close both sides.  The old code only closed the agent, leaving the
        # benchmark serve process orphaned after a failed session.
        try:
            session.close()
        finally:
            agent_instance.close()
    except RuntimeError as exc:
        from ...integrations.litellm.health import HealthCheckError

        if isinstance(exc, HealthCheckError):
            tracker.on_session_error(session, AgentError(exc))
        else:
            raise
    except SessionCancelError as exc:
        tracker.on_session_error(session, exc)
        _close_session_agent(session, agent_instance)
    except RunCancelError as exc:
        tracker.on_session_error(session, exc)
        _close_session_agent(session, agent_instance)
        raise
