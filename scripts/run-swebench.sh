#!/usr/bin/env bash
# SWE-bench Verified batch runner (resumable, retries failed sessions).
#
#   RUN_ID=swebench-verified-3 WORKERS=2 TOTAL=3 bash scripts/run-swebench.sh
#
# Every task needs its own Docker image (swebench/sweb.eval.x86_64.<instance>).
# Pull them first; a run with a missing image fails at session start.
set -uo pipefail

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True

REPO=${REPO:-/home/ashy/projects/exgentic}
OUT=$REPO/outputs
RUN_ID=${RUN_ID:-swebench-verified}
APP_RUN=$OUT/$RUN_ID/$RUN_ID
LOG=$OUT/$RUN_ID.orchestrator.log
WORKERS=${WORKERS:-2}
MODEL=${MODEL:-openai/qwen3.8-reasoner}
MAX_PASSES=${MAX_PASSES:-2}
TOTAL=${TOTAL:-3}
MAX_STEPS=${MAX_STEPS:-60}
TASKS=${TASKS:-}

TASK_ARGS=()
if [ -n "$TASKS" ]; then
  read -r -a _task_list <<< "$TASKS"
  for _task in "${_task_list[@]}"; do TASK_ARGS+=(--task "$_task"); done
  TOTAL=${TOTAL:-${#_task_list[@]}}
fi

cd "$REPO" || { echo "repository not found: $REPO" >&2; exit 1; }
mkdir -p "$OUT"
log() { echo "[$(date "+%F %T")] $*" | tee -a "$LOG"; }

exec 9>"$OUT/$RUN_ID.lock"
if ! flock -n 9; then
  log "another $RUN_ID instance is already running, exiting"
  exit 1
fi

set -a; . ./.env; set +a
if ! curl -sf -m 20 -o /dev/null "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"; then
  log "model endpoint unreachable: $OPENAI_BASE_URL"
  exit 2
fi

pending() {
  python3 - "$APP_RUN" "$TOTAL" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
total = int(sys.argv[2])
sessions = root / "sessions"
results = list(sessions.glob("*/results.json")) if sessions.is_dir() else []
bad = 0
for rp in results:
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        bad += 1
        continue
    if str(data.get("status")) in ("error", "agent error"):
        bad += 1
print(bad + max(total - len(results), 0))
PY
}

for pass in $(seq 1 "$MAX_PASSES"); do
  if [ "$pass" -gt 1 ]; then
    left=$(pending)
    log "before pass $pass: pending $left"
    [ "$left" -eq 0 ] && { log "nothing to retry, done"; break; }
  fi

  log "===== SWE-bench pass$pass (workers=$WORKERS, tasks=$TOTAL, model=$MODEL) ====="
  printf "%s running pass%s %s\n" "$RUN_ID" "$pass" "$(date "+%F %T")" > "$OUT/$RUN_ID.state"
  .venv/bin/exgentic evaluate execute \
    --benchmark swebench --agent agentcore \
    --subset princeton-nlp/SWE-bench_Verified \
    --num-tasks "$TOTAL" \
    "${TASK_ARGS[@]}" \
    --model "$MODEL" \
    --set agent.context_window_tokens=131072 \
    --set agent.output_reserve_tokens=20480 \
    --set agent.max_output_tokens=16384 \
    --set agent.http_timeout=600 \
    --set "agent.model=$MODEL" \
    --set agent.model_retries=12 \
    --set agent.model_timeout=900 \
    --set agent.max_protocol_corrections=2 \
    --set agent.max_steps="$MAX_STEPS" \
    --max-workers "$WORKERS" \
    --output-dir "./outputs/$RUN_ID" \
    --run-id "$RUN_ID" \
    --log-level INFO >>"$OUT/$RUN_ID.pass$pass.log" 2>&1
  rc=$?
  printf "%s pass%s rc=%s %s\n" "$RUN_ID" "$pass" "$rc" "$(date "+%F %T")" > "$OUT/$RUN_ID.state"
  log "pass$pass finished rc=$rc, pending=$(pending)"
done

log "===== SWE-bench done ====="