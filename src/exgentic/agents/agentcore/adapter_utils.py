# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Stateless helpers shared by the agentcore adapter.

These live outside the ``CodeAgentInstance`` subclass because they take no
instance state: the benchmark context-block renderer, the first-non-empty
selector behind the credential lookups, and the explicit "memory off" stub.
"""

from __future__ import annotations

import json
from typing import Any


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def render_context(context: dict[str, Any] | None) -> str:
    """Render a benchmark context block the way the bundled agents do."""
    parts: list[str] = []
    for key, value in (context or {}).items():
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        parts.append(f"<{key}>\n{text}\n</{key}>")
    return "\n\n".join(parts)


class NullMemory:
    """Explicitly disabled memory.

    ``Agent.__init__`` falls back to ``Memory(store)`` when ``memory`` is falsy,
    so "no memory" has to pass a truthy object. Retrieval returns nothing and
    writing is refused rather than silently dropped.
    """

    def retrieve(self, user: str, query: str, *, limit: int = 4) -> list:
        return []

    def commit(self, user: str, candidates: list) -> None:
        raise RuntimeError("Memory is disabled for this run")
