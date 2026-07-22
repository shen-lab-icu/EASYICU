# Framework v2 Phase 1 独立审阅修复

日期：2026-07-22
审阅基线：`26ca3ae`
修复发布 commit：`8c6ea461c25fea7ddc823bd1430b3e15f31781c8`

## 审阅结论修正

`26ca3ae` 是核心 runtime 与安全底座的 Phase 1，不是完整的 Action/Software/Data、reviewed memory 与新方法生产链。生产已接入的是 Planner ProtocolCard；其余三类资源只有 schema、选择器、store/approval 模型和单测。

## 已修复

- `667b6dc`：HITL request/decision 的 `review_id` 必须唯一；同 ID 的 reject+approve 不再被 dict 覆盖，恢复前 fail-close。
- `667b6dc`：旧 RunMemory 镜像 permissioned quarantine 失败改为非致命 warning，不会推翻已完成 run。
- `452495d`：Protocol Scheduler 默认只接受 `clinical_reviewed`；`curated_mvp` 仅由明确 development 路径放行。
- `8c6ea46`：release report schema v2 绑定 exact Git commit、dirty 状态和 status SHA；dirty tree 不能通过。删除伪装为观测量的常量 provider/patient counts，改为诚实记录 static command allowlist 与 `runtime_monitoring=not_instrumented`。

## 验证

- 审阅修复专项：33 项通过；两条真实 pipeline 回归另行确认 `2 passed in 55.77s`。
- Phase 1 release：resource/context、architecture、module graph、framework tests 4/4 通过；framework tests `89 passed in 39.91s`。
- 报告：`task_logs/20260722_framework_v2_phase1_release.json`，绑定 clean commit `8c6ea46`。

## 未完成的 Phase 2

- Action/Software/Data 的真实 Coder step 选择与 receipt 接线。
- reviewed/promoted memory 的 profile-bound 生产 context 接线。
- capability gap → request → HITL → approved resource 的运行闭环。

这些必须以生产调用点测试验收；在完成前不得再写“Framework v2 完整资源链已完成”。
