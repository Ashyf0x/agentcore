# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import contextvars
import threading
import time
import traceback
from abc import ABC, abstractmethod
from typing import List, Optional

from ...core.agent_instance import AgentInstance
from ...core.types import (
    Action,
    MultiObservation,
    Observation,
    ParallelAction,
    SingleAction,
)


class _AgentTurnError(RuntimeError):
    """The agent thread failed, re-raised on the harness thread.

    A plain ``RuntimeError`` here would drop the *cause* an agent attached to
    its own exceptions -- notably ``error_source``, which agents use to say
    "this was infrastructure, not me". The harness reads that attribute to
    label the session, so it has to survive the thread hand-off. Subclassing
    ``RuntimeError`` keeps every existing ``except RuntimeError`` working.
    """

    def __init__(self, message: str, original: BaseException) -> None:
        super().__init__(message)
        self.original = original
        # ``None`` when the agent does not classify its own failures, which
        # leaves the harness's exception-type-based guess in charge.
        self.error_source = getattr(original, "error_source", None)

    def __reduce__(self):
        """Survive the runner's pickle hop between the agent process and the
        harness process.

        Default exception pickling replays ``__init__(*self.args)``, but here
        ``args`` is only the message -- reconstruction would fail on the missing
        ``original``, the transport would fall back to rebuilding a builtin from
        the type name, and ``error_source`` would be lost.
        """
        return (self.__class__, (self.args[0], self.original))


class CoordinatedAgent(ABC):
    """Internal agent that runs inside an AgentCoordinator.

    The agent:
    - receives observations via get_observation()
    - sends actions via execute()
    - signals termination by execute(None)
    """

    @abstractmethod
    def run(self, adapter) -> None:
        pass


class AgentCoordinator(AgentInstance):
    """Coordinates turn-based communication between threads.

    - an environment thread (react)
    - an internal agent thread (run / execute)
    """

    def __init__(
        self,
        session_id,
        internal_agent: CoordinatedAgent,
        accumulate_window_seconds: float | None = None,
    ):
        super().__init__(session_id)
        self.internal_agent = internal_agent
        self._accumulate_window_seconds = accumulate_window_seconds

        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        self._agent_error: Optional[BaseException] = None
        self._agent_traceback: str | None = None

        self._turn = 0
        self._current_observation: Observation | None = None
        self._agent_seen_turn = -1

        self._pending_actions: List[Action | None] = []
        self._last_actions: List[SingleAction] = []

    def start(self, task, context, actions) -> None:
        """Receive work payload and start the internal agent thread (once)."""
        super().start(task, context, actions)
        with self._condition:
            if self._started:
                raise RuntimeError("AgentCoordinator already started")
            self._started = True
            ctx = contextvars.copy_context()
            self._thread = threading.Thread(
                target=ctx.run,
                args=(self._run_internal_agent,),
                name=f"AgentCoordinator[{self.session_id}]",
                daemon=False,
            )
            self._thread.start()

    def _run_internal_agent(self) -> None:
        """Entry point for the internal agent thread."""
        try:
            self.internal_agent.run(self)
        except BaseException as exc:
            with self._condition:
                # Keep the first failure. A fatal recorded by ``record_fatal``
                # is the one that explains the turn; the ``RuntimeError`` that
                # ``execute(None)`` raises afterwards is only its echo.
                if self._agent_error is None:
                    self._agent_error = exc
                    self._agent_traceback = traceback.format_exc()
                self._closed = True
                self._turn += 1
                self._current_observation = None
                self._pending_actions.clear()
                self._condition.notify_all()
            self.logger.exception("Internal agent crashed")
        finally:
            self.close()

    def record_fatal(self, error: BaseException) -> None:
        """Publish a fatal agent-side failure *before* the turn is released.

        A code agent ends every turn with ``execute(None)`` in a ``finally``, and
        that call marks the coordinator closed and wakes the environment thread.
        If the failure were only recorded afterwards -- which is what happens
        when it simply propagates to ``_run_internal_agent`` -- the environment
        thread could wake up, see ``_closed`` and no error, and read the turn as
        "the agent finished". For a benchmark that is the difference between an
        infrastructure error and a scored wrong answer.

        Recording the error first, in the same critical section that marks the
        coordinator closed, closes that window: any later ``react()``/``execute()``
        raises instead of returning ``None``.
        """
        with self._condition:
            if self._agent_error is None:
                self._agent_error = error
                self._agent_traceback = traceback.format_exc()
            self._closed = True
            self._turn += 1
            self._current_observation = None
            self._pending_actions.clear()
            self._condition.notify_all()

    def _raise_if_agent_failed(self) -> None:
        """Re-raise the agent thread's failure on this (the harness) thread.

        The original exception is chained as ``__cause__`` so the harness can
        classify it: ``AgentError`` marks a session error and is what makes
        ``results.json`` say ``error`` instead of an ``unsuccessful`` score.
        """
        if self._agent_error is not None:
            if isinstance(self._agent_error, Exception):
                tb = self._agent_traceback or ""
                raise _AgentTurnError(
                    f"{self._agent_error}\n\n{tb}", self._agent_error
                ) from self._agent_error
            raise RuntimeError("Internal agent failed") from self._agent_error

    def _flush_actions(self) -> Action | None:
        """Combine pending actions into a single Action or ParallelAction."""
        if not self._pending_actions:
            return None
        actions = self._pending_actions
        self._pending_actions = []

        if any(a is None for a in actions):
            self._closed = True
            return None

        return actions[0] if len(actions) == 1 else ParallelAction(actions=actions)

    def _remember_actions(self, action: Action | None) -> None:
        if isinstance(action, Action):
            actions = list(action.to_action_list())
            if all(isinstance(act, SingleAction) for act in actions):
                self._last_actions = actions
                return
        else:
            self._last_actions = []
            return
        self._last_actions = []

    def _rewire_observation(self, observation: Observation | None) -> Observation | None:
        if observation is None or not self._last_actions:
            return observation
        if not isinstance(observation, Observation):
            return observation

        obs_list = observation.to_observation_list()
        if not obs_list:
            return observation

        if len(obs_list) == 1 and len(self._last_actions) > 1:
            obs = obs_list[0]
            if not obs.invoking_actions:
                obs.invoking_actions = list(self._last_actions)
                return observation

        used_ids = {act.id for obs in obs_list for act in obs.invoking_actions if isinstance(act, SingleAction)}
        remaining = [act for act in self._last_actions if act.id not in used_ids]
        for obs in obs_list:
            if obs.invoking_actions:
                continue
            if not remaining:
                break
            obs.invoking_actions = [remaining.pop(0)]

        if remaining:
            self.logger.warning(
                "Unassigned actions after rewiring observations (actions=%s, observations=%s)",
                len(self._last_actions),
                len(obs_list),
            )
        return observation

    def _select_observation_for_action(
        self, action: Action | None, observation: Observation | None
    ) -> Observation | None:
        if observation is None:
            return observation
        if not isinstance(observation, Observation):
            return observation
        if not isinstance(action, SingleAction):
            return observation

        obs_list = observation.to_observation_list()
        if not obs_list:
            return observation

        matched = [
            obs
            for obs in obs_list
            if any(isinstance(inv, SingleAction) and inv.id == action.id for inv in obs.invoking_actions)
        ]
        if matched:
            if len(matched) == 1:
                return matched[0]
            return MultiObservation(observations=matched)

        self.logger.warning(
            "No matching observation for action id=%s (observations=%s)",
            action.id,
            len(obs_list),
        )
        return observation

    def _publish_observation(self, observation: Observation | None) -> None:
        self.logger.info("Publishing observation (turn=%s): %s", self._turn + 1, observation)
        observation = self._rewire_observation(observation)
        self._last_actions = []
        self._current_observation = observation
        self._turn += 1
        self._condition.notify_all()

    def _wait_for_pending_actions(self) -> bool:
        while not self._closed and not self._pending_actions:
            self._condition.wait()
        self._raise_if_agent_failed()
        return not self._closed

    def _accumulate_pending_actions(self) -> None:
        if not self._accumulate_window_seconds:
            return
        if any(a is None for a in self._pending_actions):
            return
        deadline = time.monotonic() + self._accumulate_window_seconds
        while not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if any(a is None for a in self._pending_actions):
                return
            self._condition.wait(timeout=remaining)

    def get_observation(self) -> Observation | None:
        """Block until a new observation is available or the coordinator closes."""
        with self._condition:
            self._raise_if_agent_failed()
            if self._closed:
                return None

            # Wait until a new observation is published by the environment thread.
            # We also block if _current_observation is still None (initial call).
            while not self._closed and (self._turn <= self._agent_seen_turn or self._current_observation is None):
                self._condition.wait()

            self._raise_if_agent_failed()
            if self._closed:
                return None

            self._agent_seen_turn = self._turn
            self.logger.info(
                "Delivered observation (turn=%s): %s",
                self._turn,
                self._current_observation,
            )
            return self._current_observation

    def execute(self, action: Action | None) -> Observation | None:
        """Publish an action for the current turn.

        If action is None, signals agent termination.
        """
        with self._condition:
            if self._closed and action is None:
                # The agent's ``finally: execute(None)`` still runs after a
                # recorded fatal. Treat it as the no-op it is; raising here
                # would replace the real failure with an echo of it. ``react()``
                # is what surfaces the failure to the harness.
                return None
            self._raise_if_agent_failed()
            if self._closed:
                raise RuntimeError("execute() called after close()")

            self.logger.info("Received action (turn=%s): %s", self._turn, action)
            self._pending_actions.append(action)
            self._condition.notify_all()

            if action is None:
                self._closed = True
                return None

            my_turn = self._turn
            while not self._closed and self._turn == my_turn:
                self._condition.wait()

            self._raise_if_agent_failed()
            if self._closed:
                return None

            self._agent_seen_turn = self._turn
            self.logger.info(
                "Delivered observation after action (turn=%s): %s",
                self._turn,
                self._current_observation,
            )
            return self._select_observation_for_action(action, self._current_observation)

    def react(self, observation: Observation | None) -> Action | None:
        """Publish an observation and wait for the agent's action.

        If observation is None, signals environment termination.
        """
        with self._condition:
            self._raise_if_agent_failed()
            if self._closed:
                if observation is not None:
                    raise RuntimeError("react() called after close()")
                return None

            self._publish_observation(observation)

            if observation is None:
                self._closed = True
                return None

            if not self._wait_for_pending_actions():
                return None

            self._accumulate_pending_actions()

            action = self._flush_actions()
            self._remember_actions(action)
            return action

    def close(self) -> None:
        """Close the coordinator and wait for the agent thread to exit."""
        with self._condition:
            if not self._closed:
                self._closed = True
                self._turn += 1
                self._current_observation = None
                self._pending_actions.clear()
                self._condition.notify_all()
            t = self._thread

        if t and t.is_alive() and t is not threading.current_thread():
            t.join()
