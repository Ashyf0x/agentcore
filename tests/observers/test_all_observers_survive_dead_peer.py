# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""No observer callback may touch a session whose peer is gone.

Two production crashes motivated this, both from the ``v2-full`` reruns
(2026-11-14): airline aborted after 8 of 50 tasks, telecom after 37 of 114.
In both cases the serve process died, and the next observer callback read an
attribute off the session -- an HTTP RPC through ``ObjectProxy`` -- which
raised ``httpx.RemoteProtocolError``.  That is neither a ``RuntimeError`` nor
an ``AgentError``, so it passed every handler in ``run_session``, reached the
executor's ``except Exception``, and cancelled every pending session.

Fixing the reads one at a time was not enough, and finding them by reading
code was not reliable either: the first sweep found 13, a mechanical audit of
every callback against a dead proxy found 19, and the difference included a
``session.get_cost()`` inside the completion path of the observer that had
already been "fixed".

So these tests are a *matrix*: every observer, every callback, one dead peer.
A new observer -- or a new RPC added to an existing one -- fails here instead
of in a 50-task run.

They drive the callbacks directly rather than through ``run_session`` so the
failure names the exact callback that broke.
"""

from __future__ import annotations

import httpx
import pytest

from exgentic.core.context import run_scope
from exgentic.core.orchestrator.termination import AgentError, BenchmarkError
from exgentic.core.types import SessionScore


class _DeadProxySession:
    """An ``ObjectProxy`` whose peer is gone.

    Any attribute read that was not pinned locally raises, exactly as
    ``httpx.RemoteProtocolError`` does once the serve process exits.  The
    pinned ids are the ones ``execution._pin_session_ids`` writes before any
    callback fires.
    """

    def __init__(self, pinned: bool = True) -> None:
        if pinned:
            object.__setattr__(self, "_exgentic_session_id", "sess-dead")
            object.__setattr__(self, "_exgentic_task_id", "task-dead")

    def __getattr__(self, name):  # only for attributes not found normally
        raise httpx.RemoteProtocolError(f"peer died before {name!r} could be read")


class _SilentLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _observers():
    """One instance of each observer that ships by default."""
    from exgentic.observers.handlers.dashboard_events import DashboardEventsObserver
    from exgentic.observers.handlers.file_logger import FileLoggerObserver
    from exgentic.observers.handlers.logger import ConsoleLoggerObserver
    from exgentic.observers.handlers.recap import RunRecapObserver
    from exgentic.observers.handlers.results import ResultsObserver

    results = ResultsObserver(run_id="dead-peer")
    results.on_session_enter("sess-dead", "task-dead")

    file_logger = FileLoggerObserver(run_id="dead-peer")
    file_logger._ensure_logger = lambda: None
    file_logger._logger = _SilentLogger()

    dashboard = DashboardEventsObserver()
    dashboard._emit = lambda *args, **kwargs: None

    console = ConsoleLoggerObserver()
    console._enabled = lambda level: False

    return [
        ("ResultsObserver", results),
        ("FileLoggerObserver", file_logger),
        ("DashboardEventsObserver", dashboard),
        ("ConsoleLoggerObserver", console),
        ("RunRecapObserver", RunRecapObserver(console=False)),
    ]


_SCORE = SessionScore(score=1.0, success=True, is_finished=True)

_CALLBACKS = {
    # name: (session, agent, observation) -> None
    "on_session_start": lambda o: o.on_session_start(_DeadProxySession(), None, None),
    "on_session_scoring": lambda o: o.on_session_scoring(_DeadProxySession()),
    "on_session_success": lambda o: o.on_session_success(_DeadProxySession(), _SCORE, None),
    "on_react_success_with_none": lambda o: o.on_react_success(_DeadProxySession(), None),
    "on_step_success_with_none": lambda o: o.on_step_success(_DeadProxySession(), None),
    "on_react_error": lambda o: o.on_react_error(_DeadProxySession(), RuntimeError("boom")),
    "on_step_error": lambda o: o.on_step_error(_DeadProxySession(), RuntimeError("boom")),
    "on_session_error_agent": lambda o: o.on_session_error(_DeadProxySession(), AgentError(RuntimeError("boom"))),
    "on_session_error_benchmark": lambda o: o.on_session_error(
        _DeadProxySession(), BenchmarkError(RuntimeError("boom"))
    ),
}


@pytest.fixture(autouse=True)
def _run_context(tmp_path):
    """Give every observer the run context it needs to resolve paths."""
    with run_scope(run_id="dead-peer", output_dir=str(tmp_path), cache_dir=str(tmp_path)):
        yield


@pytest.mark.parametrize("observer_name,observer", _observers(), ids=lambda v: getattr(v, "__class__", v))
@pytest.mark.parametrize("callback", sorted(_CALLBACKS))
def test_callback_survives_a_dead_peer(observer_name, observer, callback):
    """The whole matrix must hold, not just the paths that already crashed."""
    _CALLBACKS[callback](observer)


# ---------------------------------------------------------------------------
# No silent invention
# ---------------------------------------------------------------------------


class TestNoIdIsInvented:
    """A placeholder id would hide the fault being filed, so none is used."""

    def test_unpinned_dead_peer_still_raises(self):
        from exgentic.observers.handlers.session_refs import local_session_id

        with pytest.raises(httpx.RemoteProtocolError):
            local_session_id(_DeadProxySession(pinned=False))

    def test_pinned_ids_bypass_the_transport(self):
        from exgentic.observers.handlers.session_refs import local_session_id, local_task_id

        session = _DeadProxySession()
        assert local_session_id(session) == "sess-dead"
        assert local_task_id(session) == "task-dead"

    def test_plain_session_uses_the_real_attribute(self):
        """In-process benchmarks hand over a real object, not a proxy."""

        class _Plain:
            session_id = "sess-plain"
            task_id = "task-plain"

        from exgentic.observers.handlers.session_refs import local_session_id, local_task_id

        assert local_session_id(_Plain()) == "sess-plain"
        assert local_task_id(_Plain()) == "task-plain"
