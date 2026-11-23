"""Pin the three ways a baseline can be invalid.

``audit_baseline.py`` exists because a directory tree that *looks* like a run
can be assembled from several different invocations.  The audit must catch that,
and equally must not cry wolf when the only difference is unstarted tasks --
otherwise the check gets ignored, which is worse than not having it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "audit_baseline.py"


@pytest.fixture(scope="module")
def ab():
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location("_ab_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["_ab_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


BASE_KWARGS = {
    "context_window_tokens": 131072,
    "output_reserve_tokens": 8192,
    "max_output_tokens": 4096,
    "model_timeout": 600,
    "http_timeout": 600,
}

BENCH_KWARGS = {
    "subset": "dev",
    "user_simulator_model": "openai/qwen3.8-reasoner",
}


def _session(run_dir: Path, sid: str, task_id: str, *, agent_kwargs: dict | None = None,
             bench_kwargs: dict | None = None, with_results: bool = True) -> None:
    d = run_dir / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    cfg = {
        "benchmark": "appworld",
        "agent": "agentcore",
        "task_id": task_id,
        "subset": "dev",
        "model": "openai/qwen3.8-reasoner",
        "run_id": run_dir.name,
        "output_dir": str(run_dir.parent),
        "agent_kwargs": agent_kwargs if agent_kwargs is not None else dict(BASE_KWARGS),
        "benchmark_kwargs": bench_kwargs if bench_kwargs is not None else dict(BENCH_KWARGS),
    }
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if with_results:
        (d / "results.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")


def _run_cfg(run_dir: Path, task_ids: list[str], agent_kwargs: dict | None = None) -> None:
    (run_dir / "run").mkdir(parents=True, exist_ok=True)
    cfg = {
        "benchmark": "appworld",
        "agent": "agentcore",
        "subset": "dev",
        "model": "openai/qwen3.8-reasoner",
        "run_id": run_dir.name,
        "output_dir": str(run_dir.parent),
        "task_ids": task_ids,
        "agent_kwargs": agent_kwargs if agent_kwargs is not None else dict(BASE_KWARGS),
        "benchmark_kwargs": dict(BENCH_KWARGS),
    }
    (run_dir / "run" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


class TestIdRecomputation:
    def test_id_matches_name_when_config_is_honest(self, ab, tmp_path):
        d = tmp_path / "s"
        d.mkdir()
        cfg = {
            "benchmark": "appworld", "agent": "agentcore", "task_id": "abc_1",
            "subset": "dev", "model": "openai/qwen3.8-reasoner",
            "run_id": "r", "output_dir": str(tmp_path),
            "agent_kwargs": dict(BASE_KWARGS), "benchmark_kwargs": dict(BENCH_KWARGS),
        }
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        assert ab.session_id_from(cfg) is not None
        # deterministic: same payload, same id
        assert ab.session_id_from(cfg) == ab.session_id_from(dict(cfg))

    def test_missing_kwargs_changes_the_id(self, ab, tmp_path):
        full = {
            "benchmark": "appworld", "agent": "agentcore", "task_id": "abc_1",
            "subset": "dev", "model": "openai/qwen3.8-reasoner",
            "run_id": "r", "output_dir": str(tmp_path),
            "agent_kwargs": dict(BASE_KWARGS), "benchmark_kwargs": dict(BENCH_KWARGS),
        }
        partial = dict(full)
        partial["agent_kwargs"] = {"model": full["model"]}
        assert ab.session_id_from(full) != ab.session_id_from(partial), (
            "dropping agent kwargs must change the id; this is the resume bug")

    def test_bad_payload_returns_none_instead_of_raising(self, ab):
        assert ab.session_id_from({}) is None


class TestShapeGrouping:
    def test_one_shape_when_all_dirs_agree(self, ab, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        _run_cfg(run, ["t1", "t2"])
        cfg = json.loads((run / "run" / "config.json").read_text())
        for tid in ("t1", "t2"):
            _session(run, ab.session_id_from({**cfg, "task_id": tid}), tid)

        ab.audit("r")
        out = capsys.readouterr().out
        assert "config shapes: 1" in out
        assert "VERDICT   clean" in out

    def test_mixed_config_is_detected(self, ab, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        _run_cfg(run, ["t1", "t2"], agent_kwargs={**BASE_KWARGS, "model_timeout": 900})
        cfg = json.loads((run / "run" / "config.json").read_text())
        # one dir matches the run config, one recorded a different timeout
        _session(run, ab.session_id_from({**cfg, "task_id": "t1"}), "t1",
                 agent_kwargs={**BASE_KWARGS, "model_timeout": 900})
        _session(run, ab.session_id_from({**cfg, "task_id": "t2"}), "t2",
                 agent_kwargs={**BASE_KWARGS, "model_timeout": 600})

        ab.audit("r")
        out = capsys.readouterr().out
        assert "MIXED CONFIG" in out, "a dir with different kwargs must be flagged"
        assert "NOT A CLEAN BASELINE" in out

    def test_benchmark_kwargs_also_count_as_a_shape(self, ab, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        _run_cfg(run, ["t1", "t2"])
        cfg = json.loads((run / "run" / "config.json").read_text())
        _session(run, ab.session_id_from({**cfg, "task_id": "t1"}), "t1")
        _session(run, ab.session_id_from({**cfg, "task_id": "t2"}), "t2",
                 bench_kwargs={**BENCH_KWARGS, "user_simulator_model": "other/model"})

        ab.audit("r")
        out = capsys.readouterr().out
        assert "MIXED CONFIG" in out, "tau2 sessions differ only in benchmark kwargs"


class TestOrphans:
    def test_orphan_is_reported(self, ab, tmp_path, monkeypatch, capsys):
        """A dir whose id is not in the planned set does not belong to this run."""
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        _run_cfg(run, ["t1"])
        cfg = json.loads((run / "run" / "config.json").read_text())
        real = ab.session_id_from({**cfg, "task_id": "t1"})
        _session(run, real, "t1")
        # unrelated dirs with made-up names
        for i in range(5):
            _session(run, f"bad{i:05d}", f"x{i}")

        ab.audit("r")
        out = capsys.readouterr().out
        assert "orphan" in out
        assert "NOT A CLEAN BASELINE" in out

    def test_unstarted_tasks_alone_are_not_a_problem(self, ab, tmp_path, monkeypatch, capsys):
        """Missing future tasks is normal and must NOT be called unclean."""
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        task_ids = [f"t{i}" for i in range(10)]
        _run_cfg(run, task_ids)
        cfg = json.loads((run / "run" / "config.json").read_text())

        # build sid -> task_id so the dir name is right for its own config
        pairs = [(ab.session_id_from({**cfg, "task_id": t}), t) for t in task_ids]
        for sid, tid in pairs[:2]:
            _session(run, sid, tid)

        ab.audit("r")
        out = capsys.readouterr().out
        assert "ids disagree" not in out
        assert "clean" in out.lower()
        assert "NOT A CLEAN BASELINE" not in out

    def test_dir_without_results_is_counted_but_not_flagged(self, ab, tmp_path, monkeypatch, capsys):
        """Incomplete is benign: the session started and did not finish."""
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        run = tmp_path / "r" / "r"
        _run_cfg(run, ["t1"])
        cfg = json.loads((run / "run" / "config.json").read_text())
        sid = ab.session_id_from({**cfg, "task_id": "t1"})
        _session(run, sid, "t1", with_results=False)

        ab.audit("r")
        out = capsys.readouterr().out
        assert "executed  0" in out
        assert "NOT A CLEAN BASELINE" not in out


class TestMissingInputs:
    def test_missing_run_is_reported(self, ab, tmp_path, monkeypatch):
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        assert ab.audit("nope") == 1

    def test_no_run_config_is_reported(self, ab, tmp_path, monkeypatch):
        monkeypatch.setattr(ab, "OUTPUTS", tmp_path)
        (tmp_path / "r" / "r" / "sessions").mkdir(parents=True)
        assert ab.audit("r") == 1
