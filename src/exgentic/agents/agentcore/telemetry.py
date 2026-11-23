# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Exgentic-side telemetry for the runtime's model calls.

``agentcore`` talks to the endpoint over ``httpx`` directly and never goes
through LiteLLM, so the ``trace_logger`` callback that feeds
``SessionLLMTimings`` never fires, and Exgentic cannot read the runtime's
SQLite usage table. This wrapper is the only place that knows where one call
begins and ends, so cost and latency have to come from here.
"""

from __future__ import annotations

import time
from typing import Any


class UsageTrackingModel:
    """Delegating model client that counts tokens and latency for Exgentic.

    ``agentcore`` records usage per call in SQLite, but Exgentic asks the agent
    instance for a ``CostReport`` and cannot read that store, so the counters
    are kept here as the calls go by.

    Two counts with different meanings are kept separate on purpose:
    ``logical_requests`` increments once per prepared request (one model step);
    ``calls`` increments once per physical HTTP attempt, including retries.
    run_report().model_calls counts the former, get_llm_timings() measures the
    latter (every attempt spends wall time). Conflating them made the two RPCs
    disagree on how many model calls a run made.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.input_tokens = 0
        self.output_tokens = 0
        self.logical_requests = 0
        self.calls = 0
        self.durations: list[float] = []

    def prepare(self, messages, tools):
        # ContextManager.prepare() calls this once per step, before any retry.
        self.logical_requests += 1
        return self._inner.prepare(messages, tools)

    async def complete(self, request):
        started = time.perf_counter()
        try:
            response = await self._inner.complete(request)
        finally:
            # Recorded even for a failed call. A call that burned 300 s and
            # then raised still consumed 300 s of the session's wall time, and
            # reporting only the successful half would understate the cost of
            # a flaky endpoint -- the thing this number exists to expose.
            self.durations.append(time.perf_counter() - started)
        self.calls += 1
        if getattr(response, "usage_known", False):
            usage = response.usage
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        return response
