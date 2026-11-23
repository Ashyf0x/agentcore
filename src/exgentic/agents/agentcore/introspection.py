# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Introspection the Exgentic harness reads back over its RPC channel.

``AgentcoreAgentInstance`` reaches the endpoint over ``httpx`` and keeps its
evidence in the runtime SQLite store, so the three channels the harness asks
about -- cost, per-call latency and the run report -- cannot be read the way
LiteLLM-backed agents expose them. These compute each from the usage-tracking
wrapper and, for the report, a *separate read-only* store handle, so the harness
thread never touches the connection the agent thread owns.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Any

from ...utils.cost import CostReport, LiteLLMCostReport


def cost_report(model_id: str, usage: Any) -> CostReport:
    """A LiteLLM cost report for the model calls made this run.

    ``usage`` is the ``UsageTrackingModel`` wrapper (or ``None`` before the first
    turn). With no calls, or no published price for the endpoint, report the
    token counts with zero cost rather than a guess.
    """
    if usage is None or usage.calls == 0:
        return LiteLLMCostReport.initialize_empty(model_name=model_id)
    try:
        return LiteLLMCostReport.from_token_counts(model_id, usage.input_tokens, usage.output_tokens)
    except Exception:
        return LiteLLMCostReport(
            model_name=model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            input_cost=0.0,
            output_cost=0.0,
        )


def llm_timings(usage: Any) -> list[float]:
    """One wall-clock float per *physical* model call, retries included.

    Every attempt spends wall time, so a retried call contributes more than one
    float. The element type stays ``float`` because the external observer does
    ``[float(d) for d in reported]``; call-level correlation lives in the report.
    """
    if usage is None:
        return []
    return list(usage.durations)


def run_report_from_store(runtime: Any, store_path: str, run_id: str) -> dict[str, Any]:
    """The runtime run report, read through a separate read-only handle.

    The store owns its connection on the agent thread; opening a second
    read-only handle here means an RPC from the harness thread never touches it.
    ``run_report()`` only needs ``.events()``. An unreadable store is reported
    as a gap, not raised across the RPC boundary -- diagnostics must never
    affect the run.
    """
    try:
        with closing(sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)) as ro:
            ro.row_factory = sqlite3.Row

            class _ReadOnlyEvents:
                def events(self, run):
                    return [
                        {"kind": row["kind"], **json.loads(row["body"])}
                        for row in ro.execute(
                            "SELECT kind,body FROM events WHERE run=? ORDER BY seq", (run,)
                        )
                    ]

            return runtime.run_report(_ReadOnlyEvents(), run_id)
    except Exception:
        return {"run_id": run_id, "error": "store unreadable"}
