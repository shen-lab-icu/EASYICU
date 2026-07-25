# Characterization golden：语义别名稳定化

## 范围

本批只修改 characterization 测试与 golden fixture，不修改
`src/easyicu/research_agent/**`。目标是修复跨环境不稳定的测试预言机，而不是改变当前
EvidenceStore、current evidence 或 alias 权威行为。

## 根因

旧版 golden 将 EvidenceStore 的所有 current alias 一起哈希。除产品/语义 alias 外，
其中还包含每条 evidence 的兼容性 self-alias（`alias == evidence_id`）。evidence id
由记录内容派生；例如允许变化的 analyzer 文本可改变 id，即使 semantic owner、current
evidence 选择和研究结果均未改变。因此旧哈希混入了不属于规范化等价契约的环境敏感值。

## 修正

- `current_aliases` 只哈希 `alias != evidence_id` 的产品/语义 alias 映射；
- `current_evidence` 继续锁定所有 current 记录的规范化内容；
- `current_self_aliases.count` 单独锁定每条 current evidence 均发布兼容性 self-alias；
- fixture schema 升为 `easyicu.freeze_char_golden/2`。

## 验证

- 当前 `6a440b0`：五个 characterization 文件连续两轮全绿，分别使用
  `PYTHONHASHSEED=0` 与 `42`，均为 `48 passed`；
- 当前 golden 单测在 `PYTHONHASHSEED=0/42` 下均通过；
- 归档的冻结前 `7fd8cbd` 生产源码配当前测试，在 `PYTHONHASHSEED=7` 下通过，证明
  预言机不依赖本轮生产代码；
- Ruff、Black check、`git diff --check` 全绿。

## 边界

这不是 EvidenceStore schema 或 alias 权威迁移。自别名仍存在且数量被锁；产品/语义
alias 的目标 kind、owner step、artifact path 与 numeric claim authority 仍由 golden
完整验证。
