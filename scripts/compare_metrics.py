"""Aggregate metrics straight from session results, bypassing results.json.

``results.json`` at the run root is a derived snapshot: it goes stale when a
run is resumed, and it was silently overwritten once during this session.  The
session directories are the source of truth, so this script recomputes the
headline numbers from them.

The definitions below mirror ``ResultsObserver._record_session`` exactly,
because a comparison that uses a different denominator is not a comparison:

    total_sessions      = len(results_snapshot)   # sessions that produced results
    percent_successful  = success              / total
    percent_finished_*  = is_finished is True  / total
    percent_unfinished  = is_finished is False / total
    percent_error       = is_finished is None  / total
    percent_finished    = (finished ok + finished not ok) / total

Note the denominator is *executed* sessions, not planned ones.  Dividing by
planned inflates every gap and makes a half-done run look like a regression;
``percent_finished`` is the field that tells you how much of the run is real.

Two further corrections are applied before any delta is quoted, because the raw
aggregate comparison is misleading in ways that are easy to miss:

1. **Paired comparison.**  The two sides execute different subsets of the task
   list.  Aggregate rates therefore mix "the agent got better" with "one side
   happened to run easy tasks".  Restricting to tasks both sides executed, and
   counting discordant pairs (fixed vs regressed), removes that.

2. **Error attribution.**  Every `status == "error"` in these runs traces to an
   infrastructure failure (model endpoint timeout, HTTP rejection, context
   budget), not to agent behaviour.  Those sessions were never really attempted
   to completion, so including them as agent failures understates whoever had
   more of them.  `--no-infra` recomputes without them and the conclusion must
   hold under both.

Usage:
    python scripts/compare_metrics.py --detail
    python scripts/compare_metrics.py --paired
    python scripts/compare_metrics.py --paired --no-infra
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

# (display name, v2 run id, legacy run id)
PAIRS = [
    ("airline", "v2-full-airline", "full-airline"),
    ("retail", "v2-full-retail", "full-retail"),
    ("telecom", "v2-full-telecom", "full-telecom"),
    ("appworld", "v2-full-appworld-dev", "appworld-dev"),
]


def run_root(outer: Path, run_id: str) -> Path | None:
    """Both layouts exist in this repo: outputs/<run> and outputs/<run>/<run>."""
    for candidate in (outer / run_id, outer):
        if (candidate / "sessions").is_dir():
            return candidate
    for candidate in (outer / run_id, outer):
        if (candidate / "run").is_dir():
            return candidate
    return None


def load_flags(results_path: Path) -> dict | None:
    """Extract the three fields the official aggregation keys on."""
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    status = data.get("status")
    is_finished = data.get("is_finished")
    success = data.get("success")

    # Older records carry only status; derive the flags from it.
    if is_finished is None and status is not None:
        if status in ("success", "unsuccessful"):
            is_finished = True
            success = status == "success"
        elif status == "unfinished":
            is_finished = False
        elif status in ("error", "cancelled", "limit_reached"):
            is_finished = None

    return {"status": status, "is_finished": is_finished, "success": bool(success)}


def summarize(name: str, run_id: str) -> dict | None:
    outer = OUTPUTS / run_id
    if not outer.exists():
        return None
    root = run_root(outer, run_id)
    if root is None:
        return None

    planned = None
    cfg_path = root / "run" / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tids = cfg.get("task_ids") or []
        planned = len(tids) or None

    sessions = root / "sessions"
    records = []
    n_dirs = 0
    no_results = 0
    if sessions.is_dir():
        for d in sorted(p for p in sessions.iterdir() if p.is_dir()):
            n_dirs += 1
            rp = d / "results.json"
            if not rp.exists():
                no_results += 1
                continue
            flags = load_flags(rp)
            if flags is None:
                no_results += 1
                continue
            records.append(flags)

    total = len(records)  # executed only, matching ResultsObserver
    if total == 0:
        return {
            "name": name, "run_id": run_id, "planned": planned, "dirs": n_dirs,
            "total": 0, "no_results": no_results,
            "pct_successful": None, "pct_error": None,
            "pct_unfinished": None, "pct_finished": None,
            "n_success": 0, "n_error": 0, "n_unfinished": 0,
        }

    n_success = sum(1 for r in records if r["is_finished"] is True and r["success"])
    n_fin_unsucc = sum(1 for r in records if r["is_finished"] is True and not r["success"])
    n_unfinished = sum(1 for r in records if r["is_finished"] is False)
    n_error = sum(1 for r in records if r["is_finished"] is None)

    return {
        "name": name,
        "run_id": run_id,
        "planned": planned,
        "dirs": n_dirs,
        "total": total,
        "no_results": no_results,
        "n_success": n_success,
        "n_error": n_error,
        "n_unfinished": n_unfinished,
        "n_finished_unsuccessful": n_fin_unsucc,
        "pct_successful": round(n_success / total * 100, 2),
        "pct_error": round(n_error / total * 100, 2),
        "pct_unfinished": round(n_unfinished / total * 100, 2),
        "pct_finished": round((n_success + n_fin_unsucc) / total * 100, 2),
    }


INFRA_MARKERS = (
    "timed out",
    "ModelError",
    "Model HTTP request rejected",
    "ContextBudgetError",
    "protocol_error",
    "TransportFailure",
    "ConnectionError",
)


def error_class(session_dir: Path, flags: dict) -> str:
    """Classify one session: 'ok', 'infra', or 'agent'.

    'error' is a single bucket in results.json that mixes two unrelated things.
    The distinguishing evidence is the error text recorded in details, which
    names the failing layer.  Anything that names the model endpoint, the HTTP
    transport, or the context budget is infrastructure -- the task was never
    carried to a real conclusion, so scoring it against the agent is wrong.
    """
    if flags.get("status") != "error":
        return "ok"
    rp = session_dir / "results.json"
    text = ""
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
        det = data.get("details")
        if isinstance(det, dict):
            meta = det.get("session_metadata")
            if isinstance(meta, dict):
                text = str(meta.get("error", "")) + " " + str(meta.get("error_source", ""))
            text += " " + str(det.get("error", ""))
        el = session_dir / "error.log"
        if el.exists():
            text += " " + el.read_text(errors="replace")[-2000:]
    except Exception:
        pass
    if any(m in text for m in INFRA_MARKERS):
        return "infra"
    return "agent"


def load_by_task(run_id: str) -> tuple[dict, dict]:
    """Map task_id -> {success, status, cls}, plus a count of what was skipped."""
    outer = OUTPUTS / run_id
    out: dict[str, dict] = {}
    skipped = {"missing_run": 0, "no_results": 0, "no_config": 0}
    if not outer.exists():
        skipped["missing_run"] = 1
        return out, skipped
    root = run_root(outer, run_id)
    if root is None:
        skipped["missing_run"] = 1
        return out, skipped

    sessions = root / "sessions"
    if not sessions.is_dir():
        return out, skipped
    for d in sorted(p for p in sessions.iterdir() if p.is_dir()):
        rp, cp = d / "results.json", d / "config.json"
        if not rp.exists():
            skipped["no_results"] += 1
            continue
        if not cp.exists():
            skipped["no_config"] += 1
            continue
        try:
            r = json.loads(rp.read_text(encoding="utf-8"))
            tid = str(json.loads(cp.read_text(encoding="utf-8")).get("task_id"))
        except Exception:
            skipped["no_results"] += 1
            continue
        flags = load_flags(rp) or {}
        flags = dict(flags)
        flags["status"] = r.get("status")
        out[tid] = {
            "success": r.get("status") == "success",
            "status": r.get("status"),
            "cls": error_class(d, flags),
        }
    return out, skipped


def mcnemar_exact(fixed: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    Under the null (the change does nothing) each discordant task is a coin
    flip, so the count of 'fixed' among 'fixed + regressed' is Binomial(n, 0.5).
    Exact rather than chi-squared because the counts here are small.
    """
    n = fixed + regressed
    if n == 0:
        return 1.0
    k = min(fixed, regressed)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_table(no_infra: bool) -> int:
    """Compare only tasks both sides executed, counting discordant flips."""
    mode = "excluding infra errors" if no_infra else "all executed tasks"
    print(f"Paired comparison ({mode})")
    print("  Only tasks both sides executed are counted.  'fixed'/'regressed' are")
    print("  the discordant pairs; identical outcomes carry no information.")
    print()
    print(f"  {'benchmark':10s} {'N':>5} {'v2':>8} {'legacy':>8} {'delta':>8} "
          f"{'fixed':>6} {'broke':>6} {'p':>8}")
    print("  " + "-" * 72)

    tot_fix = tot_brk = tot_n = 0
    for name, v2_id, legacy_id in PAIRS:
        v2, _ = load_by_task(v2_id)
        legacy, _ = load_by_task(legacy_id)
        keys = sorted(set(v2) & set(legacy))
        if no_infra:
            keys = [k for k in keys if v2[k]["cls"] != "infra" and legacy[k]["cls"] != "infra"]
        if not keys:
            print(f"  {name:10s} {'-':>5}  no overlap")
            continue

        n = len(keys)
        a = sum(1 for k in keys if v2[k]["success"])
        b = sum(1 for k in keys if legacy[k]["success"])
        fixed = sum(1 for k in keys if v2[k]["success"] and not legacy[k]["success"])
        broke = sum(1 for k in keys if legacy[k]["success"] and not v2[k]["success"])
        p = mcnemar_exact(fixed, broke)
        tot_fix += fixed
        tot_brk += broke
        tot_n += n
        print(f"  {name:10s} {n:>5} {100.0 * a / n:>7.2f}% {100.0 * b / n:>7.2f}% "
              f"{100.0 * (a - b) / n:>+7.2f} {fixed:>6} {broke:>6} {p:>8.4f}")

    if tot_n:
        print("  " + "-" * 72)
        print(f"  {'TOTAL':10s} {tot_n:>5} {'':>8} {'':>8} {'':>8} {tot_fix:>6} {tot_brk:>6} "
              f"{mcnemar_exact(tot_fix, tot_brk):>8.4f}")
        print()
        print(f"  Net effect: {tot_fix} task(s) fixed, {tot_brk} broken, "
              f"net {tot_fix - tot_brk:+d} out of {tot_n}.")
        print("  A p-value above 0.05 means the sample cannot separate the two agents;")
        print("  quote the interval, not the point estimate.")
    return 0


def sensitivity_report() -> int:
    """Show whether the delta's *sign* depends on how infra errors are treated.

    This exists because one benchmark (airline) flipped sign between the two
    treatments in this project.  A single number cannot be defended when that
    happens, so the range has to be printed.

    It also answers the fair challenge to `--no-infra`: if a side's infra
    failures sit mostly on tasks the *other* side solved, then dropping them
    mechanically favours that side, and the correction looks self-serving even
    when it is technically right.  Better to surface that than to be accused of
    it later.
    """
    print("Sensitivity: does the delta's sign depend on the infra treatment?")
    print()
    print(f"  {'benchmark':10s} {'mode':10s} {'N':>4} {'v2':>8} {'legacy':>8} "
          f"{'delta':>8} {'fixed':>6} {'broke':>6} {'p':>7}")
    print("  " + "-" * 80)

    signs: dict[str, list[str]] = {}
    for name, v2_id, legacy_id in PAIRS:
        v2, _ = load_by_task(v2_id)
        legacy, _ = load_by_task(legacy_id)
        for mode in ("all", "no-infra"):
            keys = sorted(set(v2) & set(legacy))
            if mode == "no-infra":
                keys = [k for k in keys if v2[k]["cls"] != "infra" and legacy[k]["cls"] != "infra"]
            if not keys:
                continue
            n = len(keys)
            a = sum(1 for k in keys if v2[k]["success"])
            b = sum(1 for k in keys if legacy[k]["success"])
            fixed = sum(1 for k in keys if v2[k]["success"] and not legacy[k]["success"])
            broke = sum(1 for k in keys if legacy[k]["success"] and not v2[k]["success"])
            d = 100.0 * (a - b) / n
            signs.setdefault(name, []).append("+" if d > 0.01 else "-" if d < -0.01 else "0")
            print(f"  {name:10s} {mode:10s} {n:>4} {100.0 * a / n:>7.2f}% "
                  f"{100.0 * b / n:>7.2f}% {d:>+7.2f} {fixed:>6} {broke:>6} "
                  f"{mcnemar_exact(fixed, broke):>7.4f}")

    flipping = [b for b, s in signs.items() if len(set(s)) > 1]
    print()
    if flipping:
        print(f"  Sign flips with the treatment on: {', '.join(flipping)}")
        print("  -> no single number is defensible there; report the range.")
    else:
        print("  Sign is stable under both treatments on every benchmark.")

    print()
    print("  Are the infra failures loaded toward tasks the other side solved?")
    print()
    print(f"  {'benchmark':10s} {'side':7s} {'infra':>6} {'other solved':>13} {'other failed':>13}")
    print("  " + "-" * 58)
    for name, v2_id, legacy_id in PAIRS:
        v2, _ = load_by_task(v2_id)
        legacy, _ = load_by_task(legacy_id)
        for label, me, other in (("v2", v2, legacy), ("legacy", legacy, v2)):
            infra = [k for k in set(me) & set(other) if me[k]["cls"] == "infra"]
            if not infra:
                continue
            solved = sum(1 for k in infra if other[k]["success"])
            print(f"  {name:10s} {label:7s} {len(infra):>6} {solved:>13} {len(infra) - solved:>13}")
            if solved and solved == len(infra):
                print(f"  {'':10s} {'':7s} all of them are tasks {label} would have won by not failing")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail", action="store_true", help="print per-status counts")
    ap.add_argument("--paired", action="store_true",
                    help="per-task comparison (the only delta worth quoting)")
    ap.add_argument("--no-infra", action="store_true",
                    help="with --paired: drop sessions that died of infrastructure")
    ap.add_argument("--sensitivity", action="store_true",
                    help="show whether the delta's sign depends on infra treatment")
    args = ap.parse_args()

    if args.sensitivity:
        return sensitivity_report()
    if args.paired:
        return paired_table(no_infra=args.no_infra)

    print("Recomputed from session results.json, using ResultsObserver's definitions")
    print("(denominator = executed sessions, not planned)")
    print()

    print(f"  {'benchmark':10s} {'v2 exec':>8} {'legacy':>7} {'succ%':>7} {'legacy':>7} "
          f"{'delta':>7} {'fin%':>7} {'legacy':>7}")
    print("  " + "-" * 74)

    rows = []
    for name, v2_id, legacy_id in PAIRS:
        v2 = summarize(name, v2_id)
        legacy = summarize(name, legacy_id)
        rows.append((name, v2, legacy))

        def g(d, key, width=7):
            if not d or d.get(key) is None:
                return f"{'-':>{width}}"
            return f"{d[key]:>{width}.2f}"

        delta = "-"
        if v2 and legacy and v2.get("pct_successful") is not None and legacy.get("pct_successful") is not None:
            delta = f"{v2['pct_successful'] - legacy['pct_successful']:+.2f}"

        print(
            f"  {name:10s} {str(v2['total']) if v2 else '-':>8} "
            f"{str(legacy['total']) if legacy else '-':>7} "
            f"{g(v2, 'pct_successful')} {g(legacy, 'pct_successful')} "
            f"{delta:>7} {g(v2, 'pct_finished')} {g(legacy, 'pct_finished')}"
        )

    print()
    for name, v2, legacy in rows:
        print(f"=== {name} ===")
        for label, s in (("v2", v2), ("legacy", legacy)):
            if not s:
                print(f"  {label:8s} (missing)")
                continue
            if s["total"] == 0:
                print(f"  {label:8s} planned={s['planned']} but 0 executed")
                continue
            print(
                f"  {label:8s} planned={s['planned']}  executed={s['total']}  "
                f"success={s['n_success']}  unfinished={s['n_unfinished']}  "
                f"error={s['n_error']}  no-results={s['no_results']}"
            )
            print(
                f"  {'':8s} succ={s['pct_successful']:.2f}%  fin={s['pct_finished']:.2f}%  "
                f"unfin={s['pct_unfinished']:.2f}%  err={s['pct_error']:.2f}%"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
