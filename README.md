# exgentic-tau

Fork of [Exgentic/exgentic](https://github.com/Exgentic/exgentic) — a working copy for building, evaluating, and post-training our own agent runtime (**`agentcore`**) on Exgentic's benchmark harness.

## What this fork adds

- **`agentcore` adapter** (`src/exgentic/agents/agentcore/`) — mounts a self-contained agent runtime (model loop, token-budget context management, SQLite execution evidence, tool dispatch) as an Exgentic `CodeAgentInstance`, so it can be scored side-by-side with the bundled agents on tau2, AppWorld, SWE-bench and more. The runtime is vendored in-tree and import-isolated from any installed `agentcore`.
- **Reliability fixes in the runtime** — resolved-write escalation (a failed write is surfaced once, never replayed, and never deadlocks the session), a context budget that falls back to a byte-length upper bound when the token encodings are unavailable, and strictly layered HTTP/model timeouts so a hung endpoint fails fast and attribution stays deterministic.
- **Observability hardening** (`src/exgentic/observers/`, `src/exgentic/core/orchestrator/`) — failure attribution that separates agent errors from infrastructure errors (timeouts, endpoint outages, cancelled runs) instead of scoring them as wrong answers.
- **Post-training pipeline** (`docs/post-training-plan.md`, `scripts/export_trajectories.py`) — turns successful benchmark trajectories into SFT datasets (Qwen-3.8-reasoner → Qwen3-8B LoRA distillation) and supports the on-policy tau2 re-run loop for DPO/GRPO.

## Results

End-to-end runs of `agentcore` (model: `openai/qwen3.8-reasoner`, runs `agentcore-full-v3-*`, 2026-11-16…18):

| Benchmark | Subset | Tasks | Success | Rate |
|---|---|---:|---:|---:|
| tau2 | airline | 50 | 34 | 68.0% |
| tau2 | retail | 114 | 92 | 80.7% |
| tau2 | telecom | 114 | 99 | 86.8% |
| AppWorld | test_normal (on-demand) | 168 | 147 | 87.5% |
| **Total** | | **446** | **372** | **83.4%** |

Caveats:

- tau2 counts one attempt per task (the last attempt when a task was retried); infrastructure failures are attributed separately and not scored as wrong answers.
- On the same 168-task AppWorld `test_normal` set, the non on-demand configuration scored 55/168 (32.7%) — the on-demand figure is a same-task-set configuration comparison, not a generalization claim.
- Per-run details: [`docs/v2-results/v2-summary.csv`](docs/v2-results/v2-summary.csv) and [`docs/v2-results/v2-data.json`](docs/v2-results/v2-data.json).
