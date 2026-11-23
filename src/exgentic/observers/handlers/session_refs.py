# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Reading a session's identity without touching the transport.

An observer runs on the orchestrator side, but the session it is handed may be
an ``ObjectProxy`` whose real object lives in another process: every attribute
read is then an HTTP round-trip, and raises ``httpx.RemoteProtocolError`` once
that process is gone.

Observers are exactly the code that must keep working in that situation,
because filing the failure is their job.  ``RemoteProtocolError`` is neither a
``RuntimeError`` nor an ``AgentError``, so it slips past every handler in
``run_session`` and reaches the executor's ``except Exception`` -- which
cancels every pending session.  One unscorable session becomes one cancelled
run.

The ids therefore travel out of band: ``execution._pin_session_ids`` writes
them onto the proxy with ``object.__setattr__`` before any callback fires, so
resolving an id is a dict lookup on a healthy peer and a dict lookup on a dead
one.  The attribute fallback keeps direct ``run_session`` callers (tests,
in-process benchmarks) working, and deliberately still raises for an unpinned
dead proxy: inventing a placeholder id would hide the very fault being filed.
"""

from __future__ import annotations

SESSION_ID_ATTR = "_exgentic_session_id"
TASK_ID_ATTR = "_exgentic_task_id"


def local_id(session, attr: str, fallback: str):
    """Return ``session.<attr>`` if pinned, else ``session.<fallback>``."""
    pinned = getattr(session, attr, None)
    if pinned is not None:
        return pinned
    return getattr(session, fallback)


def local_session_id(session) -> str:
    return local_id(session, SESSION_ID_ATTR, "session_id")


def local_task_id(session):
    return local_id(session, TASK_ID_ATTR, "task_id")
