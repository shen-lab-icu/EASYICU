# Web E1 Plan 审核恢复配置修复

日期：2026-08-14  
分支：`fix/pi-workspace-review-20260809`  
修复提交：`6980819`

## 真实故障

- 普通 Web E1 项目：`study_adf40bd3133d3490`
- Web job：`215ca0bcfd1c`
- Research Agent run：`run_20260814T051544_080a82`
- Planner 前三版被合同拒绝，第四版通过科学合同，生成 16 步 Plan。
- 在向用户展示人工计划审核门之前，Web 以稳定错误码 `research_pipeline_review_config_not_recoverable` 终止。
- 本次没有批准 Plan，也没有执行分析；失败运行及其 provider hard-stop ledger 保留为不可变历史。

## 根因

Web 错把用于 provenance 的 `PipelineConfig.canonical_payload()` 当作可逆恢复载荷。该表示会对嵌套的 `key` 字段做脱敏散列，导致已绑定文献 citation key 在重建时再次散列，恢复后的 PipelineConfig digest 与暂停 checkpoint 不一致。

## 最小修复

- 增加与 provenance 表示分离的 lossless `PipelineConfig.recovery_payload()`。
- 增加 exact-digest 校验的 `PipelineConfig.from_recovery_payload()`。
- 恢复载荷拒绝 API key、opaque runner kwargs、credential-shaped 字段和不可序列化值。
- Web recovery record 升级为 v2，分别绑定恢复载荷 SHA 与 PipelineConfig digest。
- 历史 v1 record 继续 fail closed，不伪装成可恢复运行。

## 验证

- secret/recovery/durable human-review checkpoint：12 passed。
- Web recovery/routing 定点测试：4 passed。
- 真实 E1 当前配置、文献 authority、extension、capability 与 hard-stop 参数的 recovery round-trip 与失败 checkpoint digest 精确一致。
- Ruff、diff check、architecture ratchet 通过。
- 按开发策略未运行全套 CI。

## 下一步

从相同普通 E1 对话发起 fresh replan。旧 run 只作为修订来源，不复用旧批准或旧预算任务；新 Plan 必须重新停在人工计划审核门，审阅后才可决定是否执行。
