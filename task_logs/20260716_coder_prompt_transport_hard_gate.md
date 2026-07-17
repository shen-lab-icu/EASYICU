# Coder prompt/context 与 repair transport 硬门收口

日期：2026-07-16
分支：`refactor/agent-control-plane`
起点：`7251c15`

## 目标

把 Coder 从“单步携带大段全局教程、局部错误反复改写整份脚本”改成按当前
`AnalysisStep` 的精确 method/product contract 装载上下文；repair 默认只接收相关
代码块并输出 exact patch，只有 patch 无法安全应用时才允许一次显式 full rewrite。
科学选择仍由 Planner/Coder 所有，不加入 E3、KDIGO、MIMIC 或九题特定规则，也不
放松 provenance、receipt、capsule、concept audit 或 fail-close 门。

## 落地内容

1. `coder_context.py` 按 exact normalized method head、typed inputs/outputs、model
   requirements 和共享 contract predicate 选择 guide；不读 step intent，不用裸 token
   substring 路由。scoped context 只保留当前 step 所需变量及完整 `source_concept`
   companion family，不再用无关列补满 36 列。
2. initial / minimal patch / full rewrite 分别有 42,000 / 30,000 / 65,000 byte 的
   fail-before-provider 硬门。超限时不截断 typed binding 或科学坐标，也不产生 provider
   receipt 消费。
3. minimal patch 只接受 `easyicu.code_patch/1` exact unique replacement。patch 通道若
   返回完整可执行脚本，也不得直接替换未展示的代码；必须进入显式 full-rewrite
   transport，且另记 provider category。
4. patch 使用 compact authority context；full rewrite 始终携带完整 scoped science
   JSON（cohort inclusion/exclusion、temporal constraints、user preferences/covariates、
   aggregation、missingness、pitfalls、clinical caveats）和完整旧脚本。full rewrite 的
   guide 只去掉已由 typed repair contract 与完整旧脚本重复承载的通用 runtime、
   serialization、hygiene 教程，保留 method/product 的 source/table/clinical 等合同。
5. repair authority 由 host-owned typed side channel 提供。LLM/Critic/VLM prose 和 raw
   stdout/stderr 只能作为 JSON-encoded diagnostic data，不能伪造 route、line/helper
   coordinate 或 host authority。unknown validator deny by default；raw runtime 只能自动
   授权严格 syntactic repair，structural/method substitution 不能由日志触发。
6. 当前 finding 与历史 monotonic regression constraint 分成两个 host 参数：历史约束
   仍防回归，但不会把当前 mechanical patch 膨胀成 semantic repair。二者都进入 repair
   prompt binding digest；reservation 与实际 current authority 不一致时在 0 provider
   call 处 fail closed。旧的可伪造 `constraint_role=prior_regression` 不再属于 schema。
7. assignment model roster、resolved inputs、Planner step spec 和 host notes 继续走独立
   system authority，不再混进 user/run notes。相同 code/context/binding digest 继续复用
  已有 audit/capsule authority。

## 真实 E3 离线 transport 复核

使用既有归档 run
`research_output/_diagnostic_e3_8317_fresh_ceb00f2_20260716T072600Z/E3_kdigo_gradient/aware/run_20260716T072721_7fd5c5`
的真实 ResearchContext、plan、Step 02 脚本、当前 deterministic findings 与历史 monotonic
constraints，仅用 capture stub 构造消息；没有调用 API、没有执行实验、没有改归档产物。

| transport | exact payload bytes | hard gate | 结果 |
|---|---:|---:|---|
| initial generation | 39,343 | 42,000 | pass |
| minimal patch | 28,397 | 30,000 | pass |
| full rewrite fallback | 63,896 | 65,000 | pass |

这证明真实失败步的常规 repair 已从完整脚本反复重写切到约 28 KB 的 exact patch；
63.9 KB full rewrite 只在 patch 失败时出现。这里是离线 byte gate，不冒充真实 provider
token 或 wall-time 结果；后者只在架构冻结后的同一 E3 Step 02 resume 中验收。

## 验证

- prompt / repair focused：`79 passed`。
- coder/context/method/capability/meta/anti-pipeline：`560 passed`。
- provider/resume/capsule/characterization：最终分片见提交交接；中途唯一旧测试仍让
  patch prompt 返回完整脚本，已改为标准 patch JSON，未恢复不安全兼容路径。
- 对抗审阅最终 ACCEPT：完整脚本 patch 不能越权、full rewrite science 完整、
  current-vs-combined receipt binding 在 0 call 处 fail closed；审阅者未改文件。
- Ruff、Black、`py_compile`、`git diff --check` 作为提交前门。
- `test_meta_benchmark_spec.py` 保持全绿；production diff 没有新增 benchmark/case literal。

## 已知独立基线 discrepancy

三个 `test_pipeline.py` 用例在当前 dirty tree 与起点 `7251c15` 均失败，而在 capsule
接线前 `823169f` 为 3/3 通过。根因属于 `4b57162` 的既存 initial-generation capsule
错误传播：普通 provider 非 Python 响应或 outage 会以 capsule/runtime integrity 异常
炸穿，而不是写结构化 `coder_failed`。本批不吞 receipt/capsule tamper 错误来掩盖它；
下一独立小批应在 receipt completed 前验证 executable response，并只把普通 provider
失败收口为 terminal `coder_failed`，真正的 digest/receipt/capsule integrity 错误仍 hard
raise。

## 下一步

1. 单独修复上述 initial-generation 普通失败终态回归，恢复三条 pipeline 测试；不要与
   prompt transport commit 混成一个不可审阅大补丁。
2. 完成剩余分片回归与性能硬指标后，才对同一 E3 run 只 resume Step 02；Step 01 不重跑。
3. E3 真实 provider usage 用 receipt/perf harness 记录 calls、input tokens、active wall；
   离线 payload bytes 只作为 transport 前置门。
