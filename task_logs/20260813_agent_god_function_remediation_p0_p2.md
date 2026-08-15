# Research Agent 巨型函数 P0–P2 整改交接（2026-08-13）

## 结论

本轮按“先刻画、再提取 owner、最后钉住边界”的顺序完成了当前审阅范围内的 P0–P2。它是行为保持型结构整改，不改变 Canonical9 问题、shared prompt、paper rubric、Provider 配置或科学判定。

- **P0（执行热区）**：`_execute_one_step` 的候选执行/修复循环已迁入 `execution/candidate_loop.py`，以 frozen host/attempt frame、mutable state 与显式 action 驱动；原本约 2,424 行的隐式 while 状态机被拆为 8 个有名 transition 和 1 个 118 行 orchestrator。
- **P1（修复与门禁 owner）**：step attempt control plane、repair reservation、concept repair、provenance fail-closed analyzer、runner repair dispatch 已各自拥有小型公开边界和负回归，不再继续堆入 execute phase。
- **P2（计划与写作阶段）**：manuscript write stages 与 plan generation/validation 已从大函数拆出；`run_write_phase` 为 258 行，`_run_plan_phase` 为 783 行。

## 提交

1. `2ad63f0` — extract step attempt control plane
2. `64ecd31` — extract concept repair lifecycle
3. `4a05f11` — split provenance fail-closed analyzer
4. `071a3b2` — split preflight and runner repair owners
5. `60b815b` — pin extracted repair boundaries
6. `0bd572d` — split manuscript write stages
7. `f64769e` — split plan generation and validation
8. `1e5182a` — split candidate execution state machine

## 结构测量

冻结架构基线到 `1e5182a`：

| 指标 | 基线 | 当前 | 变化 |
|---|---:|---:|---:|
| `_execute_one_step` | 6,743 | 2,918 | -3,825 |
| `run_execute_phase` | 8,890 | 4,987 | -3,903 |
| `execution/phase.py` LOC | 12,433 | 7,513 | -4,920 |
| step own bound names | 361 | 196 | -165 |
| step callable closure captures | 33 | 23 | -10 |
| `pipeline.py` LOC | 11,367 | 10,713 | -654 |

候选状态机各 transition 为 61–516 行，orchestrator 为 118 行；结构回归要求 transition 不超过 550 行、orchestrator 不超过 150 行、`_execute_one_step` 不重新长回 3,000 行以上。

架构 diff 仍报告 `authority/evidence_store.py` 相对旧基线 `+29 LOC`。该漂移早于且不属于本轮改动；本轮没有刷新或放松架构基线。

## 验证

使用规范环境 `.venv`，只跑与改动直接相关的测试：

- 95 passed：candidate-loop decomposition、execution phase contracts、worker state、gate evaluator contracts。
- 7 passed：真实 pipeline repair/fallback 路径，包括生成代码修复、跨步分母漂移、固定队列漂移、contract/runtime repair 预算隔离与 deterministic fallback。
- 6 passed：package dependency directions。
- Ruff format/check 与 `git diff --check` 通过。

依照开发阶段测试策略，没有把本轮小步重构当 release checkpoint，也没有启动或等待 full exact-head CI。E1 11/11、冻结/合并/正式实验前仍需一次完整 exact-head CI。

## 边界与下一步

“P0–P2 已完成”只指本轮审阅列出的职责和验收，不表示全仓所有大文件都已拆完。`audits/validators.py`、`pipeline.py`、`gates/preflight.py` 等仍是后续结构债，但当前不应继续无边界大拆。下一步回到 Web E1 科学内容验收；只有新的具体 failure 指向这些 owner 时，才按独立 characterized refactor 继续收缩。

