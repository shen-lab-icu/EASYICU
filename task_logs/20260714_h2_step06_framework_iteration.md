# H2 Step 06 通用框架迭代与收口（2026-07-14）

## 结论

H2 development run 的 `06_confounder_set_and_assignment_model` 已在原 run 原地 resume 并达到 `status=ok`。00–05 和已有正文图均未重跑。修复均为 case-neutral 的 repair/provenance/assignment-product 合同能力，没有加入 H2、具体变量或九题规则，也没有放松 provenance、concept、contract 或 meta gate。

## 运行证据

- run：`research_output/_diagnostic_h2_8317_dagfix_20260714/H2_vasopressor_causal/aware/run_20260714T090014_75dd3c`
- Step 06 script SHA-256：`c9604a35b3637012a1ddf281bf093bc5059bcabd29ae5fe1eb665942d40f2f94`
- Step 06 summary SHA-256：`24c2f20405f577099d34207ebe61a3d9203454cfa9c40fded70b7cb85871e6e6`
- latest record：`status=ok`、`generation_mode=repaired`、concept/contract/stat/clinical/guard findings 均为空、quarantine 已退休、Critic `pass`。
- Planner-owned 13 个混杂因素全部保留。source-aware assignment model：`n=65,666`，暴露/对照 `3,189/62,477`；complete-case：`n=17,361`，暴露/对照 `1,611/15,750`；两者 `fit_status=fitted`。
- 10 组 `*_measured ↔ *_n` provenance companion 共按 74,528 行检查，全部 `invalid_pair_n=0`、`discordant_n=0`。
- `map_first` 34 个、`hr_first` 4 个域外值被明确计数并从模型资格中处理，没有静默覆盖验证结果。

## 暴露出的通用框架问题与提交

- `aec55e5`：count-only provenance 扫描也必须进入 bidirectional companion repair。
- `438ff68`：`candidate_first` 的时间伴随列统一为 `candidate_first_time`，避免机械生成 `*_first_first_time`。
- `83e888e`：quarantine 保存所有单调概念约束，后发现的问题不再覆盖已修复约束。
- `cac2331`：assignment-model 合同把总体暴露分布、模型集暴露分布、fit error 和 eligibility 后单类塌缩写入结构化 repair detail。
- `213657f`：单调概念约束同时持久化到未完成 step record；concept 通过但 contract 未通过时仍跨 resume 保留，只有整步 `ok` 才退休。

## 验证

- 完整 `tests/research_agent/` 首跑：`3526 passed, 3 skipped, 1 failed`；唯一失败为 submission-profile 默认版本测试漂移，修正后定点全绿。
- post-repair/quarantine + resume + meta：`49 passed`。
- declared-product + coder-output-scope + meta：`195 passed`。
- count/provenance/first-time focused + meta：分别 `122 passed`、`125 passed`。
- Ruff、Black、`git diff --check` 全绿；`test_meta_benchmark_spec.py` 未放松。

## 下一步

从同一 H2 run 的 Step 07 精确 resume；不重跑 00–06。结构拆分计划继续冻结，直到 E2/E3/H2/H3 development runs 全部收口并冻结 canonical protocol。
