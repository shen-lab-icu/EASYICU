# 2026-07-26 跨层复审修复与测试

## 范围

- 分支：`fix/external-review-20260724-p0-p1`
- 基线：`98dfd51`
- 活跃模块 / task：数据底座 `DATA-FIX1`、Web
  `WEBAPP-FASTAPI-NATIVE-QA`、Agent `FIG2-CANONICAL9-GATE`
- 未调用 Provider、Docker 或患者级分析流程；未 push、未 commit。
- `/Volumes/外置硬盘/databases` 仅用于只读核对本地 item dictionary，
  未修改外置盘文件。

## 已修复

### 数据层

1. `load_ts()` 不再吞掉调用方显式传入的 `time_vars`；numeric eICU offset
   可按声明的 `time_unit` 转为 `Timedelta`。
2. `load_difftime()` 在测量行找不到声明的 ICU admission origin 时失败关闭；
   错误只报告行数、distinct 数和标识符集合 SHA-256，不回显患者 ID。
3. `cbind_tbl()` 即使第一个输入是裸 `DataFrame`，也会校验所有输入的长度和
   index；后续 typed tables 仍须逐行 key 一致，并且只保留首个 typed anchor
   的一份 key。
4. 只读核对：
   - MIMIC-III `d_items.parquet`：12,487 行，`228866`、`229254` 均不存在。
   - MIMIC-IV `d_items.parquet`：4,095 行，两项均存在。
   因此只从 `mimic` / `mimic_demo` 的 `mech_circ_support` source 删除两项，
   `miiv` 保留；`concept-dict.json` 与 `sofa2-dict.json` 同步。

### Web 层

`provider_readiness()` 现在对显式配置的 base URL 执行和实际凭据加载相同的
安全校验。私网、metadata、非法 scheme、带凭据 URL、不可解析 host 等不再
被误报为 ready；返回值只含 rejection reason，不含 key 或原始 URL。

### Agent 层

计划截断 finding 新增逐 step 的 `{step_id, planned_analysis_role,
expected_outputs}` 合约。最终 readiness 只有在同一 step identity、同一分析
角色和全部 expected outputs 都恢复时才解除 paper-authority block；同名产物
出现在另一步、复用 step ID 但改变角色、或 finding 未声明产物均失败关闭。
`plan_truncation_recorded` 保留完整审计历史。

## 验证

- 数据 / Web / public API / dictionary 聚焦组：`73 passed`
- 数据 / time 邻接回归：`85 passed`
- Provider 聚焦组：`15 passed, 139 deselected`
- Concept catalog 聚焦组：`31 passed`
- Agent step-budget / DAG / primary 聚焦组：`44 passed`
- Agent completion / replan / reporting / execution / envelope：`181 passed`
- Agent pipeline readiness：`15 passed`
- Agent step-budget 最终组：`21 passed`
- Ruff（全部修改的 Python 文件）：通过
- `git diff --check`：通过
- 两份 JSON parse：通过
- 非 Agent 全套：`1942 passed, 54 skipped, 107 failed`
  - 105 项为同一个 Figure 2 v3 scorer-tree digest mismatch 的级联；
    当前工作树改动 scorer core 会按设计触发它，且只用 `HEAD` 文件重算所得
    digest 也已与冻结 manifest 不同。本轮不擅自重签论文评分权威。
  - 另外 2 项为既有 architecture / resource frozen baseline drift；
    `arch_measure` 当前报告 9 个 lower-is-better regression，本轮不以刷新
    baseline 洗绿。
- `tests/research_agent` 单进程全套（Python 3.13.5，
  `-p no:randomly`）：`7093 passed, 7 skipped, 0 failed,
  638 warnings in 2281.98s (0:38:01)`。

## 诚实边界

- 这次修复没有重新生成或覆盖 `full6_20260717`。
- 字典修复验证的是本地 prepared item dictionary 与代码 catalog 的一致性；
  不宣称重跑了六库提取。
- Figure 2 scorer authority 和 architecture/resource baseline 需要在整批 Agent
  改动定稿后单独审查、冻结，不能作为普通单测快照自动更新。
