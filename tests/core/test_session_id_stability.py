"""Session identity must be a function of the full config, not of defaults.

``SessionConfig.get_session_id()`` hashes ``agent_kwargs`` and
``benchmark_kwargs`` verbatim.  A resume command that omits a kwarg therefore
produces a different id, finds no finished results, and reruns everything --
silently.  That happened twice in one day; these tests pin the behaviour so a
future change to the resume path cannot reintroduce it unnoticed.

They also cover ``scripts/check_session_ids.py``, the guard that turns the
silent failure into a refusal to start.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from exgentic.core.types.session import SessionConfig  # noqa: E402
from exgentic.utils.paths import RunPaths  # noqa: E402

import check_session_ids  # noqa: E402

# The six agent kwargs the finished appworld sessions recorded.
APPWORLD_AGENT_KWARGS = {
    "context_window_tokens": 131072,
    "http_timeout": 600,
    "max_output_tokens": 4096,
    "model": "openai/qwen3.8-reasoner",
    "model_timeout": 900,
    "output_reserve_tokens": 8192,
}

# tau2 additionally records the user simulator and its two cost rates.
TAU2_BENCH_KWARGS = {
    "llm_user_input_cost_per_token": 2e-07,
    "llm_user_output_cost_per_token": 8e-07,
    "subset": "airline",
    "user_simulator_model": "openai/qwen3.8-reasoner",
}


def make_config(**overrides) -> SessionConfig:
    """A SessionConfig built the way ``RunConfig.to_session_config`` builds it."""
    fields = dict(
        benchmark="appworld",
        agent="agentcore",
        task_id="68ee2c9_1",
        subset="dev",
        output_dir="./outputs/v2-full-appworld-dev",
        cache_dir="/home/ashy/.exgentic",
        run_id="v2-full-appworld-dev",
        model="openai/qwen3.8-reasoner",
        overwrite_sessions=False,
        benchmark_kwargs={"subset": "dev"},
        agent_kwargs=dict(APPWORLD_AGENT_KWARGS),
    )
    fields.update(overrides)
    # model_construct on purpose: this is what the real code path does, and it
    # bypasses the validators that would otherwise normalise the kwargs.
    return SessionConfig.model_construct(**fields)


class TestAgentKwargsArePartOfIdentity:
    def test_full_kwargs_reproduce_the_recorded_id(self):
        # 007c3743 is the directory the real run created for this task.
        assert make_config().get_session_id() == "007c3743"

    def test_model_alone_produces_a_different_id(self):
        """The exact mistake: reduce agent_kwargs to {"model": ...}."""
        broken = make_config(agent_kwargs={"model": "openai/qwen3.8-reasoner"})
        assert broken.get_session_id() != "007c3743"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("model_timeout", 600),          # tau2's value, wrong for appworld
            ("context_window_tokens", None),  # materialised default when omitted
            ("max_output_tokens", 2048),      # materialised default when omitted
        ],
    )
    def test_changing_one_value_changes_the_id(self, key, value):
        kwargs = dict(APPWORLD_AGENT_KWARGS, **{key: value})
        assert make_config(agent_kwargs=kwargs).get_session_id() != "007c3743"

    def test_dropping_one_key_changes_the_id(self):
        kwargs = dict(APPWORLD_AGENT_KWARGS)
        kwargs.pop("http_timeout")
        assert make_config(agent_kwargs=kwargs).get_session_id() != "007c3743"


class TestBenchmarkKwargsArePartOfIdentity:
    def test_tau2_bench_kwargs_change_the_id(self):
        with_bench = make_config(
            benchmark="tau2",
            subset="airline",
            task_id="0",
            benchmark_kwargs=dict(TAU2_BENCH_KWARGS),
        )
        without_bench = make_config(
            benchmark="tau2",
            subset="airline",
            task_id="0",
            benchmark_kwargs={"subset": "airline"},
        )
        assert with_bench.get_session_id() != without_bench.get_session_id()


class TestOutputDirDoesNotAffectIdentity:
    def test_output_dir_is_not_hashed(self):
        """It decides *where* results are found, so it must be checked separately."""
        a = make_config(output_dir="./outputs/v2-full-appworld-dev")
        b = make_config(output_dir="/home/ashy/projects/exgentic/outputs")
        assert a.get_session_id() == b.get_session_id()


class TestRunPathsLayout:
    """``root`` is ``output_dir / run_id``; both spells are valid, one holds data."""

    def test_documented_nesting(self):
        p = RunPaths(output_dir="./outputs/run-a", run_id="run-a")
        assert p.root == Path("outputs/run-a/run-a")

    def test_flat_layout_matches_a_different_root(self):
        p = RunPaths(output_dir="./outputs", run_id="run-a")
        assert p.root == Path("outputs/run-a")


class TestFindRunRoot:
    """The guard must look at both legal layouts, not just one."""

    def test_prefers_the_tree_that_has_sessions(self, tmp_path):
        outer = tmp_path / "outputs" / "run-a"
        nested = outer / "run-a"
        (nested / "sessions" / "abc12345").mkdir(parents=True)
        (outer / "run").mkdir(parents=True)
        (outer / "run" / "config.json").write_text("{}", encoding="utf-8")

        assert check_session_ids.find_run_root(outer, "run-a") == nested

    def test_falls_back_to_the_flat_tree_when_that_is_all_there_is(self, tmp_path):
        outer = tmp_path / "outputs" / "run-a"
        (outer / "sessions" / "abc12345").mkdir(parents=True)
        (outer / "run").mkdir(parents=True)
        (outer / "run" / "config.json").write_text("{}", encoding="utf-8")

        assert check_session_ids.find_run_root(outer, "run-a") == outer

    def test_returns_none_when_neither_layout_has_a_config(self, tmp_path):
        outer = tmp_path / "outputs" / "run-a"
        outer.mkdir(parents=True)
        assert check_session_ids.find_run_root(outer, "run-a") is None


class TestCheckSessionIdsScript:
    """The guard that refuses to start a resume which would rerun everything."""

    @staticmethod
    def _write_run(root: Path, *, agent_kwargs: dict, bench_kwargs: dict, task_ids: list[str]) -> None:
        (root / "run").mkdir(parents=True, exist_ok=True)
        (root / "run" / "config.json").write_text(
            json.dumps(
                {
                    "benchmark": "appworld",
                    "agent": "agentcore",
                    "subset": "dev",
                    "output_dir": str(root.parent),
                    "cache_dir": "/tmp/cache",
                    "run_id": root.name,
                    "model": "openai/qwen3.8-reasoner",
                    "benchmark_kwargs": bench_kwargs,
                    "agent_kwargs": agent_kwargs,
                    "task_ids": task_ids,
                    "overwrite_sessions": False,
                }
            ),
            encoding="utf-8",
        )

    @pytest.fixture
    def run_tree(self, tmp_path):
        """A run whose single session dir matches what the config computes."""
        root = tmp_path / "outputs" / "run-a" / "run-a"
        self._write_run(
            root,
            agent_kwargs=dict(APPWORLD_AGENT_KWARGS),
            bench_kwargs={"subset": "dev"},
            task_ids=["68ee2c9_1"],
        )
        sid = make_config().get_session_id()
        (root / "sessions" / sid).mkdir(parents=True)
        (root / "sessions" / sid / "config.json").write_text(
            json.dumps(
                {
                    "agent_kwargs": APPWORLD_AGENT_KWARGS,
                    "benchmark_kwargs": {"subset": "dev"},
                }
            ),
            encoding="utf-8",
        )
        return tmp_path / "outputs", root, sid

    def test_matching_config_is_accepted(self, run_tree):
        outputs, _root, _sid = run_tree
        assert check_session_ids.check(outputs, "run-a") is True

    def test_reduced_agent_kwargs_are_rejected(self, run_tree):
        outputs, root, _sid = run_tree
        self._write_run(
            root,
            agent_kwargs={"model": "openai/qwen3.8-reasoner"},
            bench_kwargs={"subset": "dev"},
            task_ids=["68ee2c9_1"],
        )
        assert check_session_ids.check(outputs, "run-a") is False

    def test_changed_value_is_rejected(self, run_tree):
        outputs, root, _sid = run_tree
        self._write_run(
            root,
            agent_kwargs=dict(APPWORLD_AGENT_KWARGS, model_timeout=600),
            bench_kwargs={"subset": "dev"},
            task_ids=["68ee2c9_1"],
        )
        assert check_session_ids.check(outputs, "run-a") is False

    def test_finds_the_nested_tree_when_both_exist(self, run_tree):
        """The guard must locate the tree that holds sessions, not an empty one.

        Two layouts are legal (``outputs/<run>`` and ``outputs/<run>/<run>``),
        and a stale config in the flat one must not shadow real data in the
        nested one.  This is what makes the lookup load-bearing: if
        ``find_run_root`` returned the flat tree, ``check`` would find zero
        session dirs there and report "cannot be verified" instead of OK.
        """
        outputs, root, sid = run_tree
        flat = outputs / "run-a"
        (flat / "run").mkdir(parents=True, exist_ok=True)
        (flat / "run" / "config.json").write_text(
            (root / "run" / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
        )

        # the nested tree holds the session, the flat one holds only a config
        assert (root / "sessions" / sid).is_dir()
        assert not (flat / "sessions").exists()
        assert check_session_ids.find_run_root(flat, "run-a") == root

        # check() only reaches the id comparison by using the nested tree
        assert check_session_ids.check(outputs, "run-a") is True

    def test_missing_sessions_tree_does_not_report_success(self, run_tree, capsys):
        """No sessions anywhere means the ids cannot be verified, not "fine"."""
        outputs, root, _sid = run_tree
        import shutil as _shutil

        _shutil.rmtree(root / "sessions")

        # check() returns True here by design (nothing to protect yet), but it
        # must say so rather than printing the "OK" verdict.
        assert check_session_ids.check(outputs, "run-a") is True
        out = capsys.readouterr().out
        assert "cannot be verified" in out
        assert "-> OK" not in out
