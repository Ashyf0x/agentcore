#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.
"""Export agentcore run evidence into post-training datasets.

Per run the adapter records the exact wire request (messages + tools) and the
model reply (content + tool_calls + provider metadata, which carries
``reasoning_content`` for thinking models); the benchmark side records the
per-session reward in ``sessions/<id>/results.json``. This joins them and emits:

* SFT: one line per model call from sessions with reward 1.0 --
  ``messages`` is the verbatim wire prefix plus that call's assistant reply
  (tool calls in OpenAI shape, reasoning kept as ``reasoning_content``).
* DPO: for each task, the first model-call sample of a rewarded run paired
  with the first model-call sample of a failed run of the same task (their
  prompts are identical: same system prompt, same kick-off).

Runs that ended without a terminal model response (infra failure,
interruption) are dropped.

Usage:
    python scripts/export_trajectories.py <outputs_root> --sft out/sft.jsonl --dpo out/dpo.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path


def iter_session_dirs(outputs_root: Path):
    # os.walk follows directory symlinks (Path.glob does not), so caller-supplied
    # per-run roots can be symlinks. Exact shape: .../sessions/<sid>/results.json.
    for dirpath, _dirnames, filenames in os.walk(str(outputs_root), followlinks=True):
        if "results.json" in filenames and Path(dirpath).parent.name == "sessions":
            yield Path(dirpath)


def load_reward(session_dir: Path) -> tuple:
    """(reward, task_id); reward None when unscored/unfinished."""
    try:
        data = json.loads((session_dir / "results.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not data.get("is_finished"):
        return None, data.get("task_id")
    score = data.get("score")
    reward = float(score) if isinstance(score, (int, float)) else None
    return reward, data.get("task_id")


def assistant_message(response: dict) -> dict:
    """Rebuild the assistant turn in OpenAI shape, keeping reasoning."""
    message = {"role": "assistant", "content": response.get("content") or ""}
    tool_calls = []
    for call in response.get("tool_calls") or []:
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        tool_calls.append(
            {"id": call.get("id", ""), "type": "function",
             "function": {"name": call.get("name", ""), "arguments": args or {}}}
        )
    if tool_calls:
        message["tool_calls"] = tool_calls
    reasoning = (response.get("provider_metadata") or {}).get("reasoning_content")
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def samples_from_store(store_path: Path, run_id: str):
    """One sample per model call: verbatim wire prefix + that reply."""
    with closing(sqlite3.connect("file:" + str(store_path) + "?mode=ro", uri=True)) as db:
        pending = {}
        pairs = []
        for kind, body in db.execute(
            "SELECT kind, body FROM events WHERE run=? ORDER BY seq", (run_id,)
        ):
            data = json.loads(body)
            if kind == "model_request":
                pending[data.get("model_call_id")] = data.get("body")
            elif kind == "model_response" and data.get("model_call_id") in pending:
                request = pending.pop(data["model_call_id"])
                if request:
                    pairs.append((request, data))
    samples = []
    for index, (request, response) in enumerate(pairs):
        if response.get("finish_reason") not in {"stop", "tool_calls", "length"}:
            continue
        samples.append({
            "step": index,
            "messages": list(request.get("messages") or []) + [assistant_message(response)],
            "tools": request.get("tools"),
        })
    return samples


def load_meta(session_dir: Path) -> dict:
    try:
        data = json.loads((session_dir / "config.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {"benchmark": data.get("benchmark"), "model": data.get("model")}


def session_samples(session_dir: Path):
    reward, task_id = load_reward(session_dir)
    if reward is None:
        return None
    resolve = os.environ.get("TRAJECTORY_EXPORTER_RESOLVE") == "True"
    if resolve:
        stores = sorted(str(store) for store in
                        session_dir.glob("agent/agentcore/*.sqlite3"))
    else:
        stores = sorted(str(store) for store in
                        session_dir.glob("agent/**/*.sqlite3"))
    if not stores:
        return None
    with closing(sqlite3.connect("file:" + str(stores[0]) + "?mode=ro", uri=True)) as db:
        run_ids = [row[0] for row in db.execute("SELECT id FROM runs ORDER BY created")]
    meta = load_meta(session_dir)
    samples = []
    for run_id in run_ids:
        for sample in samples_from_store(stores[0], run_id):
            sample.update({"task_id": task_id, "run_id": run_id,
                           "session_id": session_dir.name, "reward": reward,
                           "benchmark": meta.get("benchmark"),
                           "model": meta.get("model")})
            samples.append(sample)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Export agentcore runs to SFT/DPO jsonl")
    parser.add_argument("outputs_root", type=Path, nargs="+", metavar="ROOT",
                        help="one or more outputs/run roots")
    parser.add_argument("--sft", type=Path, default=None)
    parser.add_argument("--dpo", type=Path, default=None)
    parser.add_argument("--benchmark", default=None,
                        help="only sessions from this benchmark (e.g. tau2, appworld)")
    parser.add_argument("--model", default=None,
                        help="only sessions whose recorded model string contains this")
    args = parser.parse_args()

    sft_rows = []
    by_task = {}
    stats = {"sessions": 0, "scored_sessions": 0, "sft_samples": 0, "dpo_pairs": 0}

    for root in args.outputs_root:
        for session_dir in iter_session_dirs(root):
            stats["sessions"] += 1

            if args.benchmark or args.model:
                meta = load_meta(session_dir)
                if args.benchmark and meta.get("benchmark") != args.benchmark:
                    continue
                if args.model and args.model not in str(meta.get("model") or ""):
                    continue
            samples = session_samples(session_dir)
            if samples is None:
                continue
            stats["scored_sessions"] += 1
            for sample in samples:
                if sample["reward"] >= 1.0:
                    sft_rows.append(sample)
                by_task.setdefault(sample["task_id"], []).append(sample)

    dpo_rows = []
    for task_id, items in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        wins = [s for s in items if s["reward"] >= 1.0 and s["step"] == 0]
        losses = [s for s in items if s["reward"] < 1.0 and s["step"] == 0]
        for win in wins:
            for loss in losses:
                dpo_rows.append({
                    "task_id": task_id,
                    "prompt": win["messages"][:-1],
                    "chosen": win["messages"][-1],
                    "rejected": loss["messages"][-1],
                    "tools": win["tools"],
                })

    stats["sft_samples"] = len(sft_rows)
    stats["dpo_pairs"] = len(dpo_rows)
    if args.sft:
        args.sft.parent.mkdir(parents=True, exist_ok=True)
        with args.sft.open("w", encoding="utf-8") as out:
            for row in sft_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.dpo:
        args.dpo.parent.mkdir(parents=True, exist_ok=True)
        with args.dpo.open("w", encoding="utf-8") as out:
            for row in dpo_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
