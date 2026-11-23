# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Classify an exception as infrastructure or logic, without importing either side.

This module exists to break a cycle. ``_is_infra_failure`` is needed in two
places that cannot import each other:

* ``core/orchestrator/session.py`` -- the scoring guard, deciding whether an
  unscorable transcript was the endpoint's fault.
* ``observers/handlers/results.py`` -- the error recorder, deciding what to
  write in ``error_source``.

``session`` imports ``tracker``, ``tracker`` imports ``results``, so having
``results`` import the predicate from ``session`` closes the loop and every
entry point fails with ``partially initialized module``. The predicate has no
dependencies of its own -- it only inspects exception types and class
namespaces -- so it lives here instead, where both sides can reach it.
"""

from __future__ import annotations

_INFRA_MODULES = ("httpx", "httpcore", "http", "socket", "urllib3")
_INFRA_TYPES = (
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    BrokenPipeError,
    TimeoutError,
)


def is_infra_failure(exc: BaseException) -> bool:
    """True when *exc* came from the transport layer, not from benchmark logic.

    The distinction drives ``error_source``: ``infra`` means "the endpoint or
    the serve process dropped, retry this task"; ``benchmark`` means "this
    transcript is genuinely unscorable, move on".  ``score()`` runs after the
    run has ended, so a transport error there is infrastructure by definition
    -- the benchmark itself already produced its verdict.

    Walks the ``__cause__``/``__context__`` chain because the interesting
    exception is usually wrapped: tau2 re-raises a ``ValueError`` from its
    runner thread as a bare ``RuntimeError``, and ``raise ... from`` puts the
    real cause one level down.

    Matches by module name rather than ``isinstance`` so this module does not
    take a hard dependency on httpx/httpcore.

    Why a bare ``TimeoutError`` counts as infrastructure: it is what the
    ``httpx`` read timeout surfaces as when the agent's RPC never returns. It
    is a builtin, so it carries no ``error_source`` attribute of its own and
    cannot be recognised by the wrapper-based rule in ``results.py`` -- which
    is exactly how 24 sessions in the 2026-11-14 runs got labelled "agent"
    after spending 600 s waiting on a dead socket.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _INFRA_TYPES):
            return True
        module = type(current).__module__.split(".")[0]
        if module in _INFRA_MODULES:
            return True
        nxt = current.__cause__ or current.__context__
        current = nxt if isinstance(nxt, BaseException) else None
    return False
