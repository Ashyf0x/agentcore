"""Pin the comparison rules, because a wrong denominator reverses the conclusion.

The first version of ``scripts/compare_metrics.py`` divided by *planned* tasks
instead of executed ones and reported all four benchmarks as regressions
(-25%, -29%, -12%, -11%).  The real numbers were all positive.  Nothing in the
code failed; the definition was just wrong.

These tests lock down four decisions that are easy to get wrong and impossible
to notice once wrong:

  1. the denominator is executed sessions, not planned tasks
  2. ``percent_error`` counts ``is_finished is None``, not ``status == "error"``
  3. a paired comparison must ignore tasks only one side ran
  4. an infrastructure failure is not the agent's failure
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "compare_metrics.py"


@pytest.fixture(scope="module")
def cm():
    """Import scripts/compare_metrics.py, which is not on the package path."""
    spec = importlib.util.spec_from_file_location("_cm_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["_cm_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_session(root: Path, run_id: str, sid: str, *, task_id: str,
                   status: str, details: dict | None = None,
                   error_log: str | None = None) -> None:
    """Create one session dir shaped the way the real runs write them."""
    d = root / run_id / run_id / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    flags = {
        "success": status == "success",
        "is_finished": {"success": True, "unsuccessful": True,
                        "unfinished": False, "error": None,
                        "cancelled": None, "limit_reached": False}[status],
        "status": status,
        "task_id": task_id,
        "details": details or {},
    }
    (d / "results.json").write_text(json.dumps(flags), encoding="utf-8")
    (d / "config.json").write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
    if error_log is not None:
        (d / "error.log").write_text(error_log, encoding="utf-8")


class TestDenominatorIsExecutedNotPlanned:
    def test_planned_tasks_without_results_do_not_enter_the_denominator(self, cm, tmp_path, monkeypatch):
        root = tmp_path
        monkeypatch.setattr(cm, "OUTPUTS", root)

        # 4 planned tasks, only 2 executed, both successful
        run = root / "r" / "r"
        run.mkdir(parents=True)
        (run / "run").mkdir()
        (run / "run" / "config.json").write_text(
            json.dumps({"task_ids": ["a", "b", "c", "d"]}), encoding="utf-8")

        _write_session(root, "r", "s1", task_id="a", status="success")
        _write_session(root, "r", "s2", task_id="b", status="success")

        s = cm.summarize("x", "r")
        assert s["planned"] == 4
        assert s["total"] == 2, "denominator must be executed sessions"
        assert s["pct_successful"] == 100.0, (
            "dividing by planned would report 50% and invent a regression")
        assert s["no_results"] == 0

    def test_unstarted_tasks_are_counted_as_missing_not_as_zero(self, cm, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        run.mkdir(parents=True)
        (run / "run").mkdir()
        (run / "run" / "config.json").write_text(
            json.dumps({"task_ids": [str(i) for i in range(10)]}), encoding="utf-8")
        _write_session(tmp_path, "r", "s1", task_id="0", status="success")

        s = cm.summarize("x", "r")
        assert s["planned"] == 10
        assert s["total"] == 1
        assert s["pct_successful"] == 100.0


class TestErrorIsIsFinishedNone:
    def test_error_is_derived_from_is_finished_not_status(self, cm, tmp_path):
        p = tmp_path / "results.json"
        p.write_text(json.dumps({"status": "error", "is_finished": None, "success": False}),
                     encoding="utf-8")
        assert cm.load_flags(p)["is_finished"] is None

    def test_unfinished_is_not_an_error(self, cm, tmp_path):
        p = tmp_path / "results.json"
        p.write_text(json.dumps({"status": "unfinished", "is_finished": False,
                                 "success": False}), encoding="utf-8")
        f = cm.load_flags(p)
        assert f["is_finished"] is False, "unfinished is the agent stopping early"

    def test_limit_reached_is_not_finished(self, cm, tmp_path):
        p = tmp_path / "results.json"
        p.write_text(json.dumps({"status": "limit_reached", "is_finished": False,
                                 "success": False}), encoding="utf-8")
        assert cm.load_flags(p)["is_finished"] is False

    def test_legacy_record_without_flags_is_reconstructed(self, cm, tmp_path):
        """Old results.json files carry only status; the flags must be derived."""
        p = tmp_path / "results.json"
        p.write_text(json.dumps({"status": "success"}), encoding="utf-8")
        f = cm.load_flags(p)
        assert f["is_finished"] is True and f["success"] is True

    def test_unreadable_results_is_not_counted_as_executed(self, cm, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        run.mkdir(parents=True)
        d = run / "sessions" / "s1"
        d.mkdir(parents=True)
        (d / "results.json").write_text("{not json", encoding="utf-8")
        s = cm.summarize("x", "r")
        assert s["total"] == 0
        assert s["no_results"] == 1

    def test_aggregate_puts_unfinished_outside_the_error_bucket(self, cm, tmp_path, monkeypatch):
        """The three buckets must not bleed into each other.

        This is the assertion that pins the *aggregate* rule, not just
        ``load_flags``: a run with one of each outcome must report
        error=1, unfinished=1, successful=1.  Counting unfinished (or
        limit_reached) as error would silently move sessions between two
        published metrics while every per-record test still passed.
        """
        monkeypatch.setattr(cm, "OUTPUTS", tmp_path)
        _write_session(tmp_path, "r", "s1", task_id="a", status="success")
        _write_session(tmp_path, "r", "s2", task_id="b", status="unfinished")
        _write_session(tmp_path, "r", "s3", task_id="c", status="error")
        _write_session(tmp_path, "r", "s4", task_id="d", status="limit_reached")

        s = cm.summarize("x", "r")
        assert s["total"] == 4
        assert s["n_success"] == 1
        assert s["n_unfinished"] == 2, "unfinished and limit_reached are both is_finished=False"
        assert s["n_error"] == 1, "only is_finished=None is an error"
        assert s["pct_error"] == 25.0
        assert s["pct_unfinished"] == 50.0
        assert s["pct_successful"] == 25.0
        # the three buckets partition the executed set
        assert s["n_success"] + s["n_unfinished"] + s["n_error"] == s["total"]


class TestPairedComparison:
    def test_tasks_only_one_side_ran_are_excluded(self, cm, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "OUTPUTS", tmp_path)
        # v2 ran a,b,c; legacy ran b,c,d.  Only b,c are comparable.
        _write_session(tmp_path, "v2", "s1", task_id="a", status="success")
        _write_session(tmp_path, "v2", "s2", task_id="b", status="success")
        _write_session(tmp_path, "v2", "s3", task_id="c", status="unsuccessful")
        _write_session(tmp_path, "lg", "s4", task_id="b", status="unsuccessful")
        _write_session(tmp_path, "lg", "s5", task_id="c", status="unsuccessful")
        _write_session(tmp_path, "lg", "s6", task_id="d", status="success")

        v2, _ = cm.load_by_task("v2")
        lg, _ = cm.load_by_task("lg")
        assert sorted(set(v2) & set(lg)) == ["b", "c"]

    def test_discordant_pairs_are_fixed_and_regressed(self, cm, tmp_path, monkeypatch):
        monkeypatch.setattr(cm, "OUTPUTS", tmp_path)
        for i, (vs, ls) in enumerate([("success", "unsuccessful"),
                                      ("unsuccessful", "success"),
                                      ("success", "success"),
                                      ("unsuccessful", "unsuccessful")]):
            _write_session(tmp_path, "v2", f"a{i}", task_id=str(i), status=vs)
            _write_session(tmp_path, "lg", f"b{i}", task_id=str(i), status=ls)

        v2, _ = cm.load_by_task("v2")
        lg, _ = cm.load_by_task("lg")
        keys = sorted(set(v2) & set(lg))
        fixed = [k for k in keys if v2[k]["success"] and not lg[k]["success"]]
        broke = [k for k in keys if lg[k]["success"] and not v2[k]["success"]]
        assert fixed == ["0"], "only task 0 is a genuine fix"
        assert broke == ["1"], "only task 1 is a genuine regression"


class TestMcNemar:
    def test_no_discordant_pairs_is_not_significant(self, cm):
        assert cm.mcnemar_exact(0, 0) == 1.0

    def test_symmetric_counts_are_not_significant(self, cm):
        assert cm.mcnemar_exact(5, 5) == 1.0

    def test_lopsided_counts_are_significant(self, cm):
        assert cm.mcnemar_exact(20, 2) < 0.01

    def test_one_sided_small_sample_is_not_significant(self, cm):
        """1 vs 0 is a coin flip: with one discordant pair p cannot be below 1."""
        assert cm.mcnemar_exact(1, 0) == 1.0

    def test_moderate_counts_are_suggestive_but_not_significant(self, cm):
        # 8 vs 2 is the appworld result: p ~= 0.109
        p = cm.mcnemar_exact(8, 2)
        assert 0.05 < p < 0.20


class TestInfraAttribution:
    def test_timeout_is_infrastructure(self, cm, tmp_path):
        _write_session(tmp_path, "r", "s1", task_id="a", status="error",
                       details={"session_metadata": {"error": "timed out",
                                                    "error_source": "agent"}})
        flags = cm.load_flags(tmp_path / "r" / "r" / "sessions" / "s1" / "results.json")
        assert cm.error_class(tmp_path / "r" / "r" / "sessions" / "s1", flags) == "infra"

    def test_model_http_rejection_is_infrastructure(self, cm, tmp_path):
        d = tmp_path / "s"
        d.mkdir()
        _write_session(tmp_path, "r", "s1", task_id="a", status="error",
                       details={"session_metadata": {
                           "error": "agentcore run failed: http (ModelError: "
                                    "Model HTTP request rejected)"}})
        flags = cm.load_flags(tmp_path / "r" / "r" / "sessions" / "s1" / "results.json")
        assert cm.error_class(tmp_path / "r" / "r" / "sessions" / "s1", flags) == "infra"

    def test_success_is_never_infrastructure(self, cm, tmp_path):
        _write_session(tmp_path, "r", "s1", task_id="a", status="success")
        flags = cm.load_flags(tmp_path / "r" / "r" / "sessions" / "s1" / "results.json")
        assert cm.error_class(tmp_path / "r" / "r" / "sessions" / "s1", flags) == "ok"

    def test_unfinished_is_agent_behaviour_not_infrastructure(self, cm, tmp_path):
        """The agent stopping early is the agent's outcome, not the pipeline's."""
        _write_session(tmp_path, "r", "s1", task_id="a", status="unfinished")
        flags = cm.load_flags(tmp_path / "r" / "r" / "sessions" / "s1" / "results.json")
        assert cm.error_class(tmp_path / "r" / "r" / "sessions" / "s1", flags) == "ok"


class TestRunRootResolution:
    def test_nested_layout_is_found(self, cm, tmp_path):
        (tmp_path / "r" / "r" / "sessions" / "s1").mkdir(parents=True)
        assert cm.run_root(tmp_path / "r", "r") == tmp_path / "r" / "r"

    def test_flat_layout_is_found(self, cm, tmp_path):
        (tmp_path / "r" / "sessions" / "s1").mkdir(parents=True)
        assert cm.run_root(tmp_path / "r", "r") == tmp_path / "r"

    def test_missing_run_returns_none(self, cm, tmp_path):
        (tmp_path / "r").mkdir()
        assert cm.run_root(tmp_path / "r", "r") is None
