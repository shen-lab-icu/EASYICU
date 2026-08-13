# 2026-08-13 分批提交 checkpoint（E1 继续前）

## 目的

把多人共享工作树中的大批未提交改动按 owner/职责边界封存，先恢复可审阅的 Git 历史，再继续 Web E1。此次不启动 Canonical9 Provider batch，也不运行 full exact-head CI。

## 提交序列

| SHA | 提交 | 边界 |
|---|---|---|
| `20e1b30` | `fix(data): bind source identity and selection policy` | 数据源内容身份、数据库检测、概念选择与缓存 |
| `dc810f3` | `feat(agent): add governed Skill and MCP extensions` | 用户 Skill/MCP 注册、冻结、授权与安全边界 |
| `dbd5521` | `fix(agent): separate runtime authority and timing semantics` | 运行时失败分类、依赖方向与时间/表示语义 |
| `a8c0ddd` | `fix(data): preserve fixed-horizon outcome semantics` | 固定 horizon outcome 与结构性不可用语义 |
| `5939844` | `feat(agent): enforce publication-grade scientific review` | 文献到 Plan 精确绑定、调整/敏感性 authority、科学预审、图件/稿件成熟度 |
| `d00d990` | `feat(web): unify evidence-bound research workflow` | Idea/文献、StudyContext、数据包审阅、Pi 工具、Agent run 与前端动态投影 |

## 聚焦验证

- 数据身份批次：clean detached worktree `113 passed`。
- Skill/MCP + 运行时语义组合：clean detached worktree `98 passed, 3 skipped`。
- Agent 科学质量批次：Ruff 通过；`147 passed`（含 pipeline synthetic-cohort 邻接 smoke）。
- Web/Copilot 批次：Ruff、`git diff --check`、全部修改 JS/MJS `node --check` 通过；后端直接合同 `381 passed`，3 条因新数据身份/固定 horizon/更精确 artifact code 而过期的断言定点修正后 `3 passed`；前端/静态/JS owner 合同 `186 passed`。

## 当前边界

- 工作树在 `d00d990` 后恢复干净；这些提交尚未被描述为 release/frozen checkpoint。
- E1 仍在开发验收阶段；按顶层协议只运行直接相关测试，E1 11/11 且准备冻结/合并/正式实验前才运行一次 full exact-head CI。
- 不启动 E3，不启动 Canonical9 正式 Provider batch，不把工程 canary 改写成论文证据。

