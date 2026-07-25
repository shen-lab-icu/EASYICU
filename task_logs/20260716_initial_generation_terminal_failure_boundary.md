# Initial-generation 普通失败与 authority-integrity 边界

日期：2026-07-16
分支：`refactor/agent-control-plane`
起点：`e8ad2b9`

## 问题

Capsule 集成后的 initial-generation 路径会先预留/支付 provider receipt，再持久化候选
并封存 capsule。两个普通失败场景被错误升级为 control-plane integrity exception：

1. provider 返回 `{}`、prose 或其他非 Python payload；
2. provider 本身 outage/普通调用异常。

旧路径会让 receipt 进入 failed 后抛 `StepAuthorityRuntimeError`，炸穿整个 pipeline；部分
测试中的后续 figure/audit step 因收到 `{}` 也会在 capsule seal 才失败。正确边界是：
普通 provider/candidate failure 写 terminal `coder_failed`，但 receipt/capsule digest、
storage 或 tamper 错误仍 hard raise，绝不能用 prior/untracked code fallback。

## 修复

- `CoderAgent.run` 在 `persist_candidate` 与 receipt completed 之前使用
  `looks_like_executable_python` 校验去 fence 后的 response。literal/prose/patch JSON 不会
  获得 candidate authority；已支付 initial call 的 transport 原子进入 `failed`。
- `pipeline_execute` 对“coordinates 存在且 initial transport 已 terminal failed”的普通
  exception 直接写 `coder_failed` finding/step record/partial manifest 并返回，不再抛成
  `StepAuthorityRuntimeError`，也不读取 prior code 或走 deterministic untracked fallback。
- `ProviderCallBudgetReceiptError`、`StepAuthorityRuntimeError`、
  `StepAuthorityCapsuleError` 仍在专门分支原样 hard raise。若 receipt 已 completed 后
  `on_initial_candidate` 报 capsule digest mismatch，receipt 保持 completed 且异常继续传播，
  不会被伪装成普通 coder failure。
- reservation/candidate checkpoint 的其他 storage/I/O exception 在 callback authority
  boundary 包装为 `StepAuthorityRuntimeError`；`_checkpoint_capsule` 写盘失败会先恢复旧
  `current_capsule_ref` 再传播，不能把 prior/fallback code 洗成新 initial capsule 的
  deterministic child。

## 回归

- 新增低层反例：`{}` 在 candidate persistence 前拒绝，receipt 为 failed、provider call
  恰一次、零 candidate bytes。
- 新增 integrity 反例：有效脚本 receipt completed 后 callback 报 capsule digest mismatch，
  异常 hard raise，receipt 不回退为 failed。
- 原三条 pipeline 回归恢复：cross-step denominator repair、fixed-cohort drift repair、
  generic association figure coder outage 均通过。
- 聚焦 coder/provider/capsule/meta/pipeline 最终：`140 passed`。
- 完整 `test_pipeline.py` 诊断跑在 5m16s/32 tests 时主动停止，另暴露 3 条合法
  `raise`/`print` 初稿被旧 marker detector 误拒；改为 compile-valid AST 语义后，普通
  executable statement/Call 可进入 runtime/output gate，而 literal、lone name、inert
  expression 和 invalid syntax 仍拒绝。6 条 pipeline 生成/repair/fallback 路径加 2 条
  authority 边界及 code-patch 反例最终 `12/12`；完整 265 条留给里程碑分片，不冒充已跑完。
- Ruff、Black、`py_compile`、`git diff --check` 全绿。
- 对抗复核发现并关闭 completed-receipt 后 checkpoint 裸 `OSError` 的 authority-laundering
  窗口；pending/candidate 两路 I/O 反例分别验证 0/1 次 coder call 后 hard raise，未改文件。

## 边界

本批不修改 repair prompt、科学方法、deterministic runner、evidence seal 或 resume selection；
不调用 API、不运行 E3。它只纠正普通 provider failure 的 terminal presentation，未放松
authority integrity。
