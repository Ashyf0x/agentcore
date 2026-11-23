#!/usr/bin/env bash
# AppWorld on_demand 工具发现模式整轮跑批（独立 run-id，不覆盖 full 模式结果）。
# on_demand：只暴露 api_docs__* 四个发现工具 + call_api 通用执行器，模型按需读 schema。
set -uo pipefail

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

REPO=/home/ashy/projects/exgentic
OUT=$REPO/outputs
RUN_ID=${RUN_ID:-agentcore-full-v3-appworld-ondemand}
APP_RUN=$OUT/$RUN_ID/$RUN_ID
LOG=$OUT/$RUN_ID.orchestrator.log
WORKERS=${WORKERS:-4}
MODEL=openai/qwen3.8-reasoner
MAX_PASSES=${MAX_PASSES:-2}
TOTAL=${TOTAL:-168}

cd "$REPO" || { echo "找不到 $REPO" >&2; exit 1; }
mkdir -p "$OUT"
log() { echo "[$(date "+%F %T")] $*" | tee -a "$LOG"; }

exec 9>"$OUT/$RUN_ID.lock"
if ! flock -n 9; then
  log "另一个 $RUN_ID 实例正在运行，退出"
  exit 1
fi

set -a; . ./.env; set +a
if ! curl -sf -m 20 -o /dev/null "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"; then
  log "模型端点不可达：$OPENAI_BASE_URL"
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
error = 0
for rp in results:
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        error += 1
        continue
    if str(data.get("status")) == "error":
        error += 1
print(error + max(total - len(results), 0))
PY
}

log "===== AppWorld on_demand 开始（workers=$WORKERS, model=$MODEL, total=$TOTAL）====="

for pass in $(seq 1 "$MAX_PASSES"); do
  if [ "$pass" -gt 1 ]; then
    left=$(pending)
    log "第 $pass 轮前：待补 $left"
    [ "$left" -eq 0 ] && { log "没有 error / 缺失结果，结束"; break; }
  fi

  printf "%s running pass%s %s\n" "$RUN_ID" "$pass" "$(date "+%F %T")" > "$OUT/$RUN_ID.state"
  .venv/bin/exgentic evaluate execute \
    --benchmark appworld --agent agentcore --subset test_normal \
    --model "$MODEL" \
    --set benchmark.tool_schema_mode=on_demand \
    --set agent.context_window_tokens=131072 \
    --set agent.output_reserve_tokens=20480 \
    --set agent.max_output_tokens=16384 \
    --set agent.http_timeout=600 \
    --set "agent.model=$MODEL" \
    --set agent.model_retries=12 \
    --set agent.model_timeout=900 \
    --set agent.max_protocol_corrections=2 \
    --max-workers "$WORKERS" \
    --output-dir "./outputs/$RUN_ID" \
    --run-id "$RUN_ID" \
    --log-level INFO >>"$OUT/$RUN_ID.pass$pass.log" 2>&1
  rc=$?
  printf "%s pass%s rc=%s %s\n" "$RUN_ID" "$pass" "$rc" "$(date "+%F %T")" > "$OUT/$RUN_ID.state"
  log "第 $pass 轮结束 rc=$rc，剩余待补 $(pending)"
done

log "===== AppWorld on_demand 结束 ====="