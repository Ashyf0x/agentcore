# agentcore 后训练（Post-training）拓展方案

> 基于 2026-11-22 对 WSL `~/projects/exgentic` 与 `~/projects/Agent-Project/agent-runtime` 的实地核查。
> 所有数字均来自磁盘上的真实产物，未使用任何历史聊天结论。

---

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| 现有数据能不能做 SFT？ | **能，数据量充足**（8161 条成功步 / 约 120M 上下文 token 上界） |
| 现在能不能开始训？ | **不能，缺三样东西**：可训权重、GPU、vLLM |
| 最该先做的是什么？ | **不是训练**，是 Phase 0 的数据隔离 + Phase 1 的导出器 |
| 最大的风险 | `ondemand-v2` 的 87.5% 是**同任务集上的配置对比**，不是泛化能力 |

---

## 1. 实地核查结果（与 Windows 侧文档的差异）

### 1.1 代码：两侧完全一致，且已修好已知 bug

10 个适配器模块 SHA256 在 WSL 与 Windows 两侧逐一相同：

```
agentcore_agent.py     8e4542ab...   agentcore_instance.py  4cf163b8...
settings.py            c2dad44d...   core/agent.py          5988f5bd...
```

WSL 侧 `pytest tests/agents/test_agentcore_integration.py` → **14 passed, 1 skipped**。

### 1.2 数据：比文档记录的多 5 倍

文档（`docs/v2-results/`）只覆盖 4 个 run / 335 会话。磁盘上实际有 **1796 个会话 / 38 个 run / 约 2.17 万条模型步**：

| run | 会话 | 成功 | 成功步数 | 带 reasoning |
|---|---:|---:|---:|---:|
| agentcore-full-v3-appworld | 168 | 55 | 2495 | 2480 |
| agentcore-full-v3-appworld-ondemand-v2 | 168 | **147** | 2345 | 2342 |
| agentcore-full-v3-tau2-retail | 173 | 113 | 1756 | 1722 |
| agentcore-full-v3-tau2-telecom | 212 | 107 | 2015 | 1929 |
| agentcore-full-v3-tau2-airline | 100 | 53 | 943 | 920 |
| v2-full-*（4个） | 335 | 229 | 4101 | 4089 |
| 其余 27 个 smoke/pilot run | — | — | — | — |
| **合计** | **1796** | **982** | **21693** | **21481** |

**关键**：每个 session 的 runtime store 都在
`outputs/<run>/<run>/sessions/<sid>/agent/agentcore/<sid>-<agentid>.sqlite3`，
且 `model_request` 事件里存的是**实际发出的完整 wire body**（不是重建的 prompt），
`model_response` 里存了 `reasoning_content`。这是理想的 SFT 素材。

### 1.3 `ondemand-v2` 的 87.5% 是真的，但不能当泛化证据

我逐项排除过：

| 检查 | 结果 |
|---|---|
| 是否靠 `no_op`（什么都不做）拿分 | **否**。147 个成功全部至少通过一条 `no_op_fail` 断言 |
| `scores.json` 与 `results.json` 是否一致 | **一致**，0 mismatch |
| 是否 world 状态跨 run 污染 | **否**。`experiment_name = run_id__session_id`，逐 session 隔离 |
| `answer: null` 是否等于漏交答案 | **否**。action-only 任务（如建歌单+发短信）本该为 null |
| 是否读文档后瞎试 | **否**。168/168 都读了 `api_docs`，1517 次文档读取，平均每题 19 次 `call_api` |

但它是**同一批 168 个 test_normal 任务、同一模型**上与 full 模式的配对结果：

```
full      32.7%  (55/168)
ondemand  87.5%  (147/168)
```

唯一配置差异是 `benchmark_kwargs.tool_schema_mode=on_demand`（外加
`max_workers 4→6`、`max_output_tokens 4096→16384`、`model_retries 默认→12`）。

**这意味着：+55 个百分点来自"工具 schema 怎么给"，不是"模型变强了"。**
用它训练后必须在**另一个任务集**上评测，否则是把配置收益错记成模型收益。

### 1.4 训练侧现状：零

```
nvidia-smi        不存在
torch             未安装
vllm              未安装
CPU 8 核 / 内存 7G / 磁盘剩余 943G
```

而当前模型 `openai/qwen3.8-reasoner` 走 USTC 端点，**权重不可训**。

---

## 2. 数据可用性判定

### 2.1 可作 SFT 正向样本

`success=true` 的 8161 条模型步，约 **120M token 上下文上界**。
注意这是"整段上下文"口径：同一 session 的连续步共享前缀，去重后实际语料远低于此，
所以它是**上界**。7B LoRA 需要 1–10M tokens，这批数据完全够；30B+ 建议先扩采样。

### 2.2 必须处理的四个数据问题

1. **重复采样**：同一 task 有 3–5 次尝试（appworld test_normal 92 个任务有 3 次、72 个有 4 次）。
   直接随机切 train/val 会**泄漏**：69–91 个任务在不同 run 里都有"成功"记录。
   必须按 **task family** 切分（168 个任务全部分属 56 个 family，且每个 family 都有 2–3 个变体）。
2. **`limit_reached` / `unfinished`**：这类会话的末尾步是"预算耗尽"，不是策略选择，
   作为正向样本会教模型"说到一半停下"。
3. **cost/token 字段**：`agent_cost=0.0`（实测确认），因为 agentcore 直连 httpx 绕过 LiteLLM。
   训练时如需做成本加权，必须用 store 里的 `usage` 重算。
4. **vendored 与 upstream 分叉**：仓库内 `src/exgentic/agents/agentcore/` 比
   `Agent-Project/agent-runtime` **新**（多了 `write_escalations`、`FatalRunError`、`_goal_status`）。
   现在跑 `scripts/sync_agentcore.sh` 会**用旧代码覆盖新代码**。

### 2.3 一个必须承认的既有缺陷

`status=error` 的会话里，`error_source` 字段不可信（文档已记录，修复只对新跑生效）。
作为标签时，**error 一律丢弃**，不要按 `error_source` 分类。

---

## 3. 分阶段方案

### Phase 0：先让结果可归因（0.5 天，无算力）

这一步不做，后面所有数字都对不上代码版本。

- [ ] `git add` 掉 `src/exgentic/agents/agentcore/`（当前 untracked）和 10 个测试文件
- [ ] 决定 vendored 与 upstream 谁是真源；给 `sync_agentcore.sh` 加 `--check`，禁止降级覆盖
- [ ] 补回 0 字节的 `scripts/rerun-agentcore-failed.sh`（三个脚本引用它）
- [ ] 加 `.gitattributes` 正常化换行（当前 CRLF 造成 5600+ 行假 diff）
- [ ] 在 WSL 记录 `git rev-parse HEAD` + 上述 10 个 hash，写入每个 run 目录的 provenance

**验收**：任意 session 能反查到产生它的代码 hash。

### Phase 1：轨迹 → 数据集导出器（2–3 天，无算力）

新增，纯离线：

```
src/exgentic/agents/agentcore/observability/dataset.py
scripts/export_sft_dataset.py
tests/agents/test_agentcore_dataset.py
```

设计约束（都能在现有代码上直接落地）：

- **只读**打开 store，复用 `introspection.py` 已用的 `file:...?mode=ro` 模式
- **一条样本 = 一个模型步**，`input = model_request.body`，`output = model_response`
- **标签三态**：`positive`（success 且 is_finished）/ `negative`（失败但有终态）/ `unusable`（error、无终态、store 不可读）
- **切分按 task family**，绝不按单 task
- **保留失败轨迹**：4713 条失败步是 DPO/GRPO 的负例，不要扔

**验收**：输出 `stats.json`，含每 benchmark 正负比、token 分布、丢弃原因计数。

### Phase 2：选基座 + 搭推理（1 天，需算力）

```bash
# 目标链路
agentcore ──httpx──> vLLM (OpenAI 兼容) ──> 你的 LoRA
```

`HTTPModelClient` 是纯 httpx + Chat Completions，**不需要改代码**：

```
--set agent.api_base=http://127.0.0.1:8000/v1
--set agent.wire_model=Qwen3-8B
--set agent.provider=qwen          # Qwen 系
--set agent.thinking=true
```

**选型建议**：

| 基座 | 显存 | 说明 |
|---|---|---|
| Qwen3-8B | 24G（LoRA 训练）/ 16G（推理） | 起步首选，迭代快 |
| Qwen3-32B | 80G | 数据够且想冲成绩时再用 |

### Phase 3：SFT（LoRA）

- 只学 tool-calling 格式与"何时调 `finish`"
- **不学 AppWorld 的 on_demand 路由**——那是 schema 模式给的，不是模型学的
- 训练后第一件事：`pytest tests/agents/test_agentcore_response_validation.py`
  然后看 `run_report` 里 `protocol_corrections` / `argument_corrections` 是否暴涨

### Phase 4：评测纪律（贯穿）

- **在 `train` 子集训（90 题）、`dev` 选（57 题）、`test_normal` 只报一次（168 题）**
- 训练集与报告集必须是**不同 task family**
- 报告的固定列：`score / is_finished / steps / protocol_corrections / argument_corrections`
- 成本列必须从 store 的 `usage` 重算，不能用 `results.json` 里的 0

---

## 4. 三个必须现在回答的问题

1. **训练谁的权重？** 云租 GPU（推荐 A100/H100 按小时）还是本地无卡只做数据？
2. **训 8B 还是 32B？** 取决于能租到多大的卡，以及是否愿意等更久。
3. **报告在哪个集合上？** 如果只在 `test_normal` 报，而训练也用了 `test_normal`，
   那这个项目就没有可信的成绩——必须用 `train(90)` 训、`dev(57)` 选。

---

## 5. 附：本次核查用到的只读脚本

已写入 `A:\Github\exgentic\.tmp\`（可从 WSL 通过 `/mnt/a/Github/exgentic/.tmp/` 调用）：

| 脚本 | 用途 |
|---|---|
| `survey_runs.py` | 全量 run 清点：会话数、成功数、步数、reasoning 数 |
| `audit_run.py` | 单 run 配置 + 结局分布 + `scores.json` 一致性 |
| `audit_tools.py` | 实际调用的工具分布，判断是否真在做事 |
| `classify_labels2.py` | 按 `no_op_pass` / `no_op_fail` 区分真假成功 |
| `verify_by_requirement.py` | 按断言文本还原"是否需要提交答案" |
| `raw_finish.py` | 单会话完整动作序列（含 finish 参数） |
| `coverage.py` / `attempts.py` | 任务覆盖、重复采样、按 run 的成功率对照 |
| `token_budget.py` | 成功步的 token 量上界估算 |
