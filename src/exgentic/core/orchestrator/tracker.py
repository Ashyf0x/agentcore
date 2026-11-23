# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterable

from ...observers.handlers.configs import ConfigsObserver
from ...observers.handlers.file_logger import FileLoggerObserver
from ...observers.handlers.logger import ConsoleLoggerObserver
from ...observers.handlers.recap import RunRecapObserver
from ...observers.handlers.results import ResultsObserver
from ...observers.handlers.warnings import WarningsObserver
from ...utils.settings import get_settings
from ..context import get_context
from .controller import CleanupController, Controller, CoreController, LimitController
from .observer import Observer
from .termination import AgentError, BenchmarkError

# Module-level, not per-instance: an observer failure should be visible from
# whatever context dispatched it.  No handler is attached here on purpose --
# ``logging.lastResort`` prints ERROR records to stderr, so the traceback shows
# up without requiring the caller to configure logging first.
_log = logging.getLogger(__name__)


class Tracker(Observer, Controller):
    def __init__(
        self,
        *,
        observers: Iterable[Observer] | None = None,
        controllers: Iterable[Controller] | None = None,
        use_defaults: bool = True,
        max_steps: int = 100,
        max_actions: int = 100,
    ) -> None:
        self._run_id = get_context().run_id
        self._observers: list[Observer] = []
        self._controllers: list[Controller] = []
        self._results: ResultsObserver | None = None
        self.events = None
        self._run_lock = threading.Lock()
        self._run_state: str | None = None
        if use_defaults:
            self._register_observer(ResultsObserver())
            self._register_observer(ConfigsObserver())
            self._register_observer(WarningsObserver())
            self._register_observer(FileLoggerObserver())
            self._register_observer(ConsoleLoggerObserver())
            self._register_observer(RunRecapObserver(console=False))
            if get_settings().otel_enabled:
                from ...observers.handlers.otel import OtelTracingObserver

                self._register_observer(OtelTracingObserver())
            self._register_controller(LimitController(max_steps=max_steps, max_actions=max_actions))
            self._register_controller(CleanupController())
        if observers:
            for observer in observers:
                self._register_observer(observer)
        if controllers:
            for controller in controllers:
                self._register_controller(controller)
        self._ensure_core_controller()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def observers(self) -> tuple[Observer, ...]:
        return tuple(self._observers)

    @property
    def session_results(self):
        if self._results is not None:
            return self._results.session_results()
        return []

    def results(self):
        if self._results is not None:
            return self._results.results()
        raise RuntimeError("No results observer configured to provide results.")

    def _register_observer(self, observer: Observer | None) -> None:
        if observer is None:
            return
        if observer in self._observers:
            return
        if isinstance(observer, ResultsObserver):
            self._results = observer
        if observer.events is not None:
            self.events = observer.events
        self._observers.append(observer)

    def _register_controller(self, controller: Controller | None) -> None:
        if controller is None:
            return
        if controller in self._controllers:
            return
        self._controllers.append(controller)

    # -- dispatch ------------------------------------------------------------
    #
    # Observers are bookkeeping, and bookkeeping must never decide whether a
    # run survives.  Before this guard, one observer raising while *filing* a
    # session's failure propagated out of ``run_session`` into the executor's
    # ``except Exception``, which cancels every pending future -- so a single
    # unscorable task voided the other N-1.  That is what happened to airline
    # (8/50) when a serve process died and the next callback read
    # ``session.session_id`` off the dead proxy.
    #
    # Fixing each observer is necessary but not sufficient: the next observer
    # added, or the next RPC introduced into an existing one, re-opens the same
    # hole.  The invariant belongs here — a callback that blows up is logged
    # against the observer that raised it and then skipped, and dispatch
    # continues to the remaining observers.

    def _dispatch_observers(self, event: str, call: Callable[[Observer], None]) -> None:
        for observer in self._observers:
            try:
                call(observer)
            except Exception:  # noqa: BLE001 -- see the block comment above
                _log.exception(
                    "Observer %s failed during %s; continuing with the rest of the run",
                    type(observer).__name__,
                    event,
                )

    def _dispatch_controllers(self, event: str, call: Callable[[Controller], None]) -> None:
        # Controllers decide run outcome -- LimitController raises
        # AgentTerminationError to stop a session, CoreController decides when
        # the run is over -- so their exceptions must propagate untouched.
        # They are in-process objects with no transport, so they are not the
        # failure mode the observer guard exists for.
        for controller in self._controllers:
            call(controller)

    def on_run_start(self, run_config) -> None:
        self._dispatch_observers("on_run_start", lambda o: o.on_run_start(run_config))

    def on_run_success(self, results, run_config) -> None:
        if not self._mark_run_final("success"):
            return
        self._dispatch_observers("on_run_success", lambda o: o.on_run_success(results, run_config))
        self._dispatch_controllers("on_run_success", lambda c: c.on_run_success(results, run_config))

    def on_run_error(self, error) -> None:
        if not self._mark_run_final("error"):
            return
        self._dispatch_observers("on_run_error", lambda o: o.on_run_error(error))
        self._dispatch_controllers("on_run_error", lambda c: c.on_run_error(error))

    def on_session_enter(self, session_id: str, task_id: str | None) -> None:
        self._dispatch_observers("on_session_enter", lambda o: o.on_session_enter(session_id, task_id))

    def on_session_creation(self, session) -> None:
        self._dispatch_observers("on_session_creation", lambda o: o.on_session_creation(session))

    def on_session_start(self, session, agent, observation) -> None:
        self._dispatch_observers("on_session_start", lambda o: o.on_session_start(session, agent, observation))

    def on_react_success(self, session, action) -> None:
        self._dispatch_observers("on_react_success", lambda o: o.on_react_success(session, action))
        self._dispatch_controllers("on_react_success", lambda c: c.on_react_success(session, action))

    def on_step_success(self, session, observation) -> None:
        self._dispatch_observers("on_step_success", lambda o: o.on_step_success(session, observation))
        self._dispatch_controllers("on_step_success", lambda c: c.on_step_success(session, observation))

    def on_react_error(self, session, error) -> None:
        self._dispatch_observers("on_react_error", lambda o: o.on_react_error(session, error))
        self._dispatch_controllers("on_react_error", lambda c: c.on_react_error(session, error))
        raise AgentError(error)

    def on_step_error(self, session, error) -> None:
        self._dispatch_observers("on_step_error", lambda o: o.on_step_error(session, error))
        self._dispatch_controllers("on_step_error", lambda c: c.on_step_error(session, error))
        raise BenchmarkError(error)

    def on_session_error(self, session, error) -> None:
        self._dispatch_observers("on_session_error", lambda o: o.on_session_error(session, error))

    def on_session_success(self, session, score, agent) -> None:
        self._dispatch_observers("on_session_success", lambda o: o.on_session_success(session, score, agent))

    def on_session_scoring(self, session) -> None:
        self._dispatch_observers("on_session_scoring", lambda o: o.on_session_scoring(session))

    def on_session_reuse(self, session_results) -> None:
        self._dispatch_observers("on_session_reuse", lambda o: o.on_session_reuse(session_results))

    def _ensure_core_controller(self) -> None:
        cores = [item for item in self._controllers if isinstance(item, CoreController)]
        non_cores = [item for item in self._controllers if not isinstance(item, CoreController)]
        if not cores:
            self._controllers = [*non_cores, CoreController()]
            return
        self._controllers = [*non_cores, cores[-1]]

    def _mark_run_final(self, state: str) -> bool:
        with self._run_lock:
            if self._run_state is not None:
                return False
            self._run_state = state
            return True
