# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""CostReport must survive being written to results.json and read back.

Every assertion here failed at some point in a real run, and each failure was
silent: the run completed, the JSON parsed, and the numbers were wrong. The
symptom in production was a v2 run reporting ``agent_cost: 0.0`` across 500
sessions while agentcore's store held 530 model calls and 4.3M input tokens.

Three independent defects produce that symptom, so there are three groups:

1. **Subclass fields are dropped on write.** ``cost_reports`` is declared
   ``dict[str, CostReport]``, so pydantic serialises every value using the
   *base* schema. ``LiteLLMCostReport.input_tokens`` never reaches the file.
2. **The total is dropped on read.** ``total_cost`` is a ``computed_field``
   derived from the private ``_total_cost``, and a private attribute is not a
   pydantic field, so ``model_validate`` has nothing to assign it from. A
   re-loaded report reports zero.
3. **The two losses are indistinguishable from "nothing happened."** With 1
   and 2 in place, "no model calls" and "4.3M tokens at an unpriced endpoint"
   both serialise to ``{"model_name": ..., "total_cost": 0.0}``. That is what
   made the bug survive so long: the file could not show it was wrong.

The tests avoid ``litellm``'s price table on purpose -- pricing drifts and is
covered by ``test_cost.py``. These are about shape, not money.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("src"))

from exgentic.core.types.session import SessionResults  # noqa: E402
from exgentic.utils.cost import (  # noqa: E402
    CostReport,
    LiteLLMCostReport,
    UpdatableCostReport,
    accumulate_reports,
)


def _litellm_report(inp=1234, out=567, icost=0.002, ocost=0.003):
    """A report with costs supplied, so no price lookup is involved."""
    return LiteLLMCostReport(
        model_name="openai/qwen3.8-reasoner",
        input_tokens=inp,
        output_tokens=out,
        input_cost=icost,
        output_cost=ocost,
    )


def _session(reports: dict) -> SessionResults:
    return SessionResults(
        session_id="s1",
        success=True,
        score=1.0,
        is_finished=True,
        steps=1,
        agent_cost=sum(r.total_cost for r in reports.values()),
        benchmark_cost=0.0,
        execution_time=1.0,
        cost_reports=reports,
    )


class TestUnpricedEndpointIsDistinguishable:
    """Defect 3: an unpriced endpoint must not look like an idle agent.

    This is the highest-value test in the file. It is the only one that pins
    *why* the bug was invisible rather than merely that it existed.
    """

    def test_empty_and_unpriced_do_not_serialise_identically(self):
        """'no calls' and 'calls but no price' must produce different JSON.

        Both have total_cost 0.0, so the token counts are the only evidence
        that work happened. If they serialise identically, a reader of
        results.json cannot tell a broken run from an idle one.
        """
        idle = LiteLLMCostReport.initialize_empty(model_name="m")
        unpriced = LiteLLMCostReport(
            model_name="m", input_tokens=36433, output_tokens=1855,
            input_cost=0.0, output_cost=0.0,
        )
        assert idle.total_cost == unpriced.total_cost == 0.0

        idle_json = json.loads(_session({"agent": idle}).model_dump_json())["cost_reports"]["agent"]
        unpriced_json = json.loads(_session({"agent": unpriced}).model_dump_json())["cost_reports"]["agent"]

        assert idle_json != unpriced_json, (
            "an unpriced endpoint is indistinguishable from an idle agent in "
            "results.json; the token counts that prove otherwise are dropped"
        )
        assert unpriced_json["input_tokens"] == 36433
        assert unpriced_json["output_tokens"] == 1855

    def test_zero_cost_with_tokens_survives_the_round_trip(self):
        """Re-loading must not turn 'unpriced' into 'idle' either."""
        unpriced = LiteLLMCostReport(
            model_name="m", input_tokens=10, output_tokens=5,
            input_cost=0.0, output_cost=0.0,
        )
        back = SessionResults.model_validate_json(_session({"agent": unpriced}).model_dump_json())
        report = back.cost_reports["agent"]
        assert report.input_tokens == 10
        assert report.output_tokens == 5


class TestSubclassFieldsSurviveSerialization:
    """Defect 1: the declared type must not erase the runtime type's fields."""

    @pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "input_cost", "output_cost"])
    def test_litellm_report_fields_reach_the_json(self, field):
        payload = json.loads(_session({"agent": _litellm_report()}).model_dump_json())
        assert field in payload["cost_reports"]["agent"], (
            f"{field} was dropped because cost_reports is typed dict[str, CostReport]"
        )

    def test_tokens_are_preserved_exactly(self):
        payload = json.loads(_session({"agent": _litellm_report(11, 22)}).model_dump_json())
        assert payload["cost_reports"]["agent"]["input_tokens"] == 11
        assert payload["cost_reports"]["agent"]["output_tokens"] == 22

    def test_updatable_report_total_reaches_the_json(self):
        """tau2 reports via UpdatableCostReport; its cost lives in _total_cost."""
        r = UpdatableCostReport.initialize_empty("m")
        r.add_cost(0.0012202)
        payload = json.loads(_session({"benchmark": r}).model_dump_json())
        assert payload["cost_reports"]["benchmark"]["total_cost"] == pytest.approx(0.0012202)


class TestTotalsSurviveDeserialization:
    """Defect 2: a re-loaded report must still know what it cost."""

    def test_reloaded_litellm_total_is_not_zero(self):
        payload = _session({"agent": _litellm_report()}).model_dump_json()
        back = SessionResults.model_validate_json(payload)
        assert back.cost_reports["agent"].total_cost == pytest.approx(0.005)

    def test_reloaded_updatable_total_is_not_zero(self):
        r = UpdatableCostReport.initialize_empty("m")
        r.add_cost(0.25)
        payload = _session({"benchmark": r}).model_dump_json()
        back = SessionResults.model_validate_json(payload)
        assert back.cost_reports["benchmark"].total_cost == pytest.approx(0.25)

    def test_reloaded_tokens_match_the_file(self):
        payload = _session({"agent": _litellm_report(70, 30)}).model_dump_json()
        back = SessionResults.model_validate_json(payload)
        assert back.cost_reports["agent"].input_tokens == 70
        assert back.cost_reports["agent"].output_tokens == 30

    def test_reload_is_idempotent(self):
        """Writing a re-loaded report must reproduce the same numbers."""
        once = _session({"agent": _litellm_report()})
        twice = SessionResults.model_validate_json(once.model_dump_json())
        a = json.loads(once.model_dump_json())["cost_reports"]["agent"]
        b = json.loads(twice.model_dump_json())["cost_reports"]["agent"]
        assert a == b


class TestAccumulationAfterReload:
    """The run-level total is summed from re-loaded sessions.

    A resume reuses finished sessions by reading them off disk, so this path is
    not hypothetical: in the observed run exactly the 8 reused sessions
    contributed 0 to ``accumulated_benchmark_report`` while their
    ``benchmark_cost`` float was intact. The two sums disagreed by exactly the
    amount those 8 sessions were worth.
    """

    def test_reloaded_reports_still_sum_to_the_float_total(self):
        reports = [_litellm_report(10, 5, 0.1, 0.1) for _ in range(3)]
        sessions = [_session({"agent": r}) for r in reports]
        reloaded = [SessionResults.model_validate_json(s.model_dump_json()) for s in sessions]

        float_total = sum(s.agent_cost for s in reloaded)
        report_sum = sum(s.cost_reports["agent"].total_cost for s in reloaded)
        assert report_sum == pytest.approx(float_total)
        assert report_sum == pytest.approx(0.6)

    def test_accumulate_reports_over_reloaded_agents(self):
        sessions = [SessionResults.model_validate_json(_session({"agent": _litellm_report()}).model_dump_json())]
        acc = accumulate_reports([s.cost_reports["agent"] for s in sessions])
        assert acc.total_cost == pytest.approx(0.005)

    def test_mixed_subclasses_fall_back_without_losing_money(self):
        """accumulate_reports rejects mixed types; the caller then folds manually.

        The manual fold uses the public total_cost, so it is correct as long as
        the individual reports are. This pins that contract, because the
        fallback in results.py depends on it.
        """
        a = UpdatableCostReport.initialize_empty("m")
        a.add_cost(0.5)
        b = _litellm_report(icost=0.3, ocost=0.2)
        with pytest.raises(ValueError):
            accumulate_reports([a, b])

        acc = CostReport.initialize_empty()
        for r in (a, b):
            acc.accumulate_from(r)
        assert acc.total_cost == pytest.approx(1.0)
