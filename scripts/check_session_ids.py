"""Refuse a resume that would silently rerun work already on disk.

Two independent ways a resume can miss the finished sessions:

1. ``agent_kwargs`` / ``benchmark_kwargs`` drift.  ``get_session_id()`` hashes
   them verbatim, so omitting an agent kwarg -- or passing a different value
   for one -- produces a completely different set of ids.  The orchestrator
   then finds nothing and starts from task one.

2. ``--output-dir`` drift.  ``RunPaths.root`` is ``output_dir / run_id``, so
   ``--output-dir outputs`` and ``--output-dir outputs/<run-id>`` point at two
   different trees.  Both are valid; only one holds the results.

Neither failure raises.  The progress bar just restarts, which is how a full
day of appworld work was lost twice.

This script asks the real code the real question: for each run under
``outputs/``, find the tree that actually holds sessions, compute the ids the
*recorded* config implies, and compare.  Exit status 1 means a mismatch.

Usage:
    python scripts/check_session_ids.py               # check every v2-full run
    python scripts/check_session_ids.py --run <id>    # check one run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from exgentic.core.types.session import SessionConfig  # noqa: E402

DEFAULT_RUNS = ("v2-full-appworld-dev", "v2-full-airline", "v2-full-telecom")


def find_run_root(outer: Path, run_id: str) -> Path | None:
    """Locate the directory that actually holds this run's sessions.

    Two layouts are in play: ``outputs/<run>/<run>`` (created with
    ``--output-dir outputs/<run>``) and ``outputs/<run>`` (created with
    ``--output-dir outputs``).  Both are legal.  Only one usually has sessions,
    and that is the one to trust -- a stale config in the other must not win.
    """
    nested = outer / run_id
    flat = outer

    nested_sessions = (nested / "sessions").is_dir() and any(
        p.is_dir() for p in (nested / "sessions").iterdir()
    )
    flat_sessions = (flat / "sessions").is_dir() and any(
        p.is_dir() for p in (flat / "sessions").iterdir()
    )

    if nested_sessions:
        return nested
    if flat_sessions:
        return flat

    # No sessions anywhere: fall back to whichever layout has a config, so the
    # caller can still report the planned task count and say "cannot verify".
    for candidate in (nested, flat):
        if (candidate / "run" / "config.json").exists():
            return candidate
    return None


def load_run_config(root: Path) -> dict:
    path = root / "run" / "config.json"
    if not path.exists():
        raise SystemExit(f"missing run config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def session_configs(cfg: dict) -> list[SessionConfig]:
    """Build session configs the same way ``RunConfig.to_session_config`` does."""
    out = []
    for task_id in cfg.get("task_ids") or []:
        out.append(
            SessionConfig.model_construct(
                benchmark=cfg["benchmark"],
                agent=cfg["agent"],
                task_id=str(task_id),
                subset=cfg.get("subset"),
                output_dir=cfg.get("output_dir"),
                cache_dir=cfg.get("cache_dir"),
                run_id=cfg.get("run_id"),
                model=cfg.get("model"),
                overwrite_sessions=cfg.get("overwrite_sessions", False),
                benchmark_kwargs=dict(cfg.get("benchmark_kwargs") or {}),
                agent_kwargs=dict(cfg.get("agent_kwargs") or {}),
            )
        )
    return out


def check(outputs: Path, run_id: str) -> bool:
    outer = outputs / run_id
    if not outer.exists():
        print(f"{run_id}: run directory absent, nothing to protect -- skipped")
        return True

    root = find_run_root(outer, run_id)
    if root is None:
        print(f"{run_id}: no run config found under {outer} -- skipped")
        return True

    sessions_dir = root / "sessions"
    cfg = load_run_config(root)
    configs = session_configs(cfg)
    computed = {sc.get_session_id(): sc.task_id for sc in configs}

    on_disk = sorted(p.name for p in sessions_dir.iterdir() if p.is_dir()) if sessions_dir.exists() else []
    disk = set(on_disk)
    hit = set(computed) & disk

    rel = root.relative_to(outputs.parent)
    print(f"{run_id}")
    print(f"  run root           : {rel}")
    print(f"  recorded output_dir: {cfg.get('output_dir')}")
    print(f"  planned tasks      : {len(configs)}")
    print(f"  computed ids       : {len(computed)}")
    print(f"  session dirs       : {len(disk)}")
    print(f"  computed & on disk : {len(hit)}")

    # The resume will recompute root as output_dir/run_id; if the config's own
    # output_dir disagrees with where the data is, --output-dir is wrong.
    recorded_out = cfg.get("output_dir")
    if recorded_out:
        implied = (Path(recorded_out).expanduser().resolve() / run_id)
        if implied != root.resolve():
            print(f"  !! MISMATCH: config output_dir implies {implied}")
            print(f"     but the data is at {root.resolve()}")
            print("     the resume command must pass --output-dir " + str(root.parent))

    if not disk:
        print("  -> no finished sessions yet, ids cannot be verified")
        return True

    if not hit:
        print("  -> MISMATCH: none of the computed ids exist on disk")
        print("     a resume would rerun every task from scratch")
        for sid in on_disk[:3]:
            sess_cfg = sessions_dir / sid / "config.json"
            if sess_cfg.exists():
                recorded = json.loads(sess_cfg.read_text(encoding="utf-8"))
                print(f"     e.g. {sid} recorded agent_kwargs = "
                      f"{json.dumps(recorded.get('agent_kwargs'), sort_keys=True)}")
                print(f"          {sid} recorded benchmark_kwargs = "
                      f"{json.dumps(recorded.get('benchmark_kwargs'), sort_keys=True)}")
        print(f"     computed from run config = "
              f"{json.dumps(cfg.get('agent_kwargs'), sort_keys=True)}")
        return False

    # Partial reuse is the subtle failure: a few ids hit by luck while the rest
    # silently rerun.  Point at the sessions that exist but were not matched.
    orphan = disk - set(computed)
    if orphan:
        print(f"  note: {len(orphan)} session dirs are not in the planned set "
              f"(stale ids or a changed config): {sorted(orphan)[:3]}")
        sample = next((s for s in sorted(orphan)), None)
        if sample:
            sess_cfg = sessions_dir / sample / "config.json"
            if sess_cfg.exists():
                recorded = json.loads(sess_cfg.read_text(encoding="utf-8"))
                print(f"     {sample} recorded agent_kwargs = "
                      f"{json.dumps(recorded.get('agent_kwargs'), sort_keys=True)}")
                print(f"     {sample} recorded benchmark_kwargs = "
                      f"{json.dumps(recorded.get('benchmark_kwargs'), sort_keys=True)}")
                print(f"     run config agent_kwargs = "
                      f"{json.dumps(cfg.get('agent_kwargs'), sort_keys=True)}")
                print(f"     run config benchmark_kwargs = "
                      f"{json.dumps(cfg.get('benchmark_kwargs'), sort_keys=True)}")

    print("  -> OK")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(REPO / "outputs"))
    ap.add_argument("--run", action="append", dest="runs", help="run id to check (repeatable)")
    args = ap.parse_args()

    outputs = Path(args.output_dir)
    runs = tuple(args.runs) if args.runs else DEFAULT_RUNS
    ok = True
    for run_id in runs:
        ok = check(outputs, run_id) and ok
        print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
