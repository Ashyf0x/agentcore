"""Baseline coherence audit: is a run's on-disk state actually one run?

Three separate problems showed up in this project, and from the outside they all
look the same -- a session directory that does not agree with the run config
sitting next to it:

1. **Orphans.**  A directory whose recomputed ``session_id`` is not in the set
   the run config implies.  Caused by writing into ``outputs/<run>`` while the
   config recorded ``outputs/<run>/<run>``, so the ids never matched and the
   directories accumulated from several invocations.

2. **Mixed configs.**  Every directory is self-consistent, but the directories
   disagree with *each other* -- some recorded ``model_timeout=600``, others
   ``900``.  ``run/config.json`` only preserves whichever call wrote last, so
   the file on disk is not a description of the data on disk.

3. **Incomplete.**  Directory present, no ``results.json``.  This one is benign:
   the session was started and did not finish.

Only (1) and (2) invalidate a baseline.  (3) is normal and simply means the
denominator is executed sessions, not directories.

Usage:
    python scripts/audit_baseline.py                # all known runs
    python scripts/audit_baseline.py appworld-dev   # one run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from exgentic.core.types.session import SessionConfig  # noqa: E402

from compare_metrics import OUTPUTS, run_root  # noqa: E402

DEFAULT_RUNS = [
    "v2-full-appworld-dev", "v2-full-airline", "v2-full-retail", "v2-full-telecom",
    "appworld-dev", "full-airline", "full-retail", "full-telecom",
]

# the agent kwargs that actually change behaviour, i.e. the ones worth grouping by
SHAPE_KEYS = ("model_timeout", "http_timeout", "max_output_tokens",
              "context_window_tokens", "output_reserve_tokens", "max_steps")


def config_of(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def session_id_from(cfg: dict) -> str | None:
    """Recompute a session id from the payload a config.json carries."""
    try:
        sc = SessionConfig.model_construct(
            benchmark=cfg["benchmark"],
            agent=cfg["agent"],
            task_id=str(cfg["task_id"]),
            subset=cfg.get("subset"),
            output_dir=cfg.get("output_dir"),
            cache_dir=cfg.get("cache_dir"),
            run_id=cfg.get("run_id"),
            model=cfg.get("model"),
            overwrite_sessions=cfg.get("overwrite_sessions", False),
            benchmark_kwargs=dict(cfg.get("benchmark_kwargs") or {}),
            agent_kwargs=dict(cfg.get("agent_kwargs") or {}),
        )
        return sc.get_session_id()
    except Exception:
        return None


def planned_ids(run_cfg: dict) -> set[str]:
    out = set()
    for tid in run_cfg.get("task_ids") or []:
        sc = SessionConfig.model_construct(
            benchmark=run_cfg["benchmark"],
            agent=run_cfg["agent"],
            task_id=str(tid),
            subset=run_cfg.get("subset"),
            output_dir=run_cfg.get("output_dir"),
            cache_dir=run_cfg.get("cache_dir"),
            run_id=run_cfg.get("run_id"),
            model=run_cfg.get("model"),
            overwrite_sessions=run_cfg.get("overwrite_sessions", False),
            benchmark_kwargs=dict(run_cfg.get("benchmark_kwargs") or {}),
            agent_kwargs=dict(run_cfg.get("agent_kwargs") or {}),
        )
        out.add(sc.get_session_id())
    return out


def audit(run_id: str) -> int:
    outer = OUTPUTS / run_id
    if not outer.exists():
        print(f"{run_id}: no such run")
        return 1
    root = run_root(outer, run_id)
    if root is None:
        print(f"{run_id}: cannot locate the directory that holds sessions")
        return 1

    run_cfg_path = root / "run" / "config.json"
    run_cfg = config_of(run_cfg_path)
    sessions = root / "sessions"
    dirs = sorted(p for p in sessions.iterdir() if p.is_dir()) if sessions.is_dir() else []

    print("=" * 78)
    print(f"{run_id}")
    print(f"  root      {root}")
    print(f"  config    {run_cfg_path if run_cfg else '<missing>'}")
    print(f"  dirs      {len(dirs)}")

    if not run_cfg:
        print("  cannot audit without a run config")
        return 1

    # --- group directories by the config they recorded for themselves ---
    groups: dict[tuple, list[str]] = defaultdict(list)
    unreadable = []
    for d in dirs:
        c = config_of(d / "config.json")
        if c is None:
            unreadable.append(d.name)
            continue
        ak = c.get("agent_kwargs") or {}
        bk = c.get("benchmark_kwargs") or {}
        key = (tuple((k, ak.get(k)) for k in SHAPE_KEYS),
               tuple(sorted((k, str(v)) for k, v in bk.items())))
        groups[key].append(d.name)

    run_shape = (
        tuple((k, (run_cfg.get("agent_kwargs") or {}).get(k)) for k in SHAPE_KEYS),
        tuple(sorted((k, str(v)) for k, v in (run_cfg.get("benchmark_kwargs") or {}).items())),
    )
    matching = groups.get(run_shape, [])
    differing = {k: v for k, v in groups.items() if k != run_shape}

    if unreadable:
        print(f"  !! {len(unreadable)} dir(s) without a readable config.json: {unreadable[:5]}")
    if differing:
        print(f"  !! MIXED CONFIG: {len(differing)} shape(s) differ from run/config.json")
        for key, names in sorted(differing.items(), key=lambda kv: -len(kv[1])):
            ak = dict(key[0])
            desc = ", ".join(f"{k}={v}" for k, v in ak.items() if v is not None)
            print(f"       {len(names):3d} dir(s) with {desc}")
    else:
        print("  config shapes: 1 (every dir agrees with run/config.json)")

    # --- id self-consistency per group ---
    bad = 0
    for key, names in groups.items():
        for name in names:
            c = config_of(sessions / name / "config.json")
            got = session_id_from(c) if c else None
            if got != name:
                bad += 1
                if bad <= 3:
                    print(f"  !! {name} recomputes to {got}")
    if bad:
        print(f"  !! {bad} dir(s) whose name does not match their own config")

    # --- planned coverage ---
    exp = planned_ids(run_cfg)
    disk = {d.name for d in dirs}
    orphan = disk - exp
    missing = exp - disk
    have_results = sum(1 for d in dirs if (d / "results.json").exists())

    print(f"  planned   {len(exp)}")
    print(f"  matched   {len(exp & disk)}")
    print(f"  orphan    {len(orphan)}" + ("  <-- not part of this run's params" if orphan else ""))
    print(f"  missing   {len(missing)}  (never started)")
    print(f"  executed  {have_results}  (dirs with results.json -- the real denominator)")

    # --- verdict ---
    problems = []
    if differing:
        problems.append("mixed configs")
    if bad:
        problems.append("ids disagree with their own config")
    if orphan and len(orphan) > max(3, len(disk) // 10):
        problems.append(f"{len(orphan)} orphans")
    if problems:
        print(f"  VERDICT   NOT A CLEAN BASELINE: {'; '.join(problems)}")
    else:
        print("  VERDICT   clean (differences from planned are just unstarted tasks)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", default=None,
                    help="run ids to audit (default: all known runs)")
    args = ap.parse_args()
    runs = args.runs or DEFAULT_RUNS
    for r in runs:
        audit(r)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
