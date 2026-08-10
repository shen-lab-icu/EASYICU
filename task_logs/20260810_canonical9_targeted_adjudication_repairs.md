# Canonical9 E2/H2/H3 定点裁决与最小修复

日期：2026-08-10
分支：`fix/pi-workspace-review-20260809`
起始冻结基线：`964cf47cb4ebd994551e6217c753a89abc834b9f`
任务：`FIG2-CANONICAL9-SCIENCE-AUTHORITY-V4`

## 范围与状态

本轮只关闭 adversarial review 已裁决的 G-1、E2、H2-current 与 H3 blocker/major finding。没有修改 Canonical9 问题文本、shared prompt、frozen paper rubric，也没有调用 Provider 或启动正式 Canonical9 batch。

旧人工签署包 `项目进度/benchmark实验/Canonical9_E2_H2_H3_人工双签包_964cf47.md` 已标记 `OBSOLETE / SUPERSEDED`，不得签署，也不得生成 formal authority。

## 定点 adjudication 结论

| Finding | 裁决 | 严重度 | 是否阻止旧包签署 | 最小处置 |
|---|---|---:|---:|---|
| G-1 签署协议未唯一治理实际 runtime scientific guardrail / deterministic execution | Confirmed | blocker | 是 | typed case protocol 确定性编译 `RuntimeScientificProjection` 与 case-neutral execution contract；attestation、authority、JSONL、launcher、plan validator 与 representation/candidate/stability executor 共同绑定 protocol/projection/execution digest |
| E2 exposure opportunity / primary estimand 不唯一 | Confirmed | major | 是 | 冻结 24 h landmark primary；人群为 24 h 时仍存活并在观察中且有完整 0–24 h 合法乳酸，随访从 landmark 后开始；原 variable-opportunity 分析降为 descriptive sensitivity |
| E2 primary model 不唯一 | Confirmed | major | 是 | 唯一 headline 为 logistic restricted cubic spline，knots=10/50/90 percentile、median reference、95% CI；linear per 1 mmol/L 仅 sensitivity |
| H2 当前 source 无法确认 verified non-use/initiator status | Confirmed | blocker for effect estimate, none for fail-closed benchmark | 是 | 保留 `H2_VERIFIED_NON_USE_UNAVAILABLE`；不造 control、不做 PSM/IPTW/effect；仅冻结 source-specific 解除条件 |
| H2 future target-trial 细节继续阻塞当前 27-run | Not confirmed | none | 否 | future design 标为 non-authorizing / not executable；不新增 universal ESS cutoff |
| H3-1 SOFA-2 total 与 components 双重计权 | Confirmed | major | 是 | clustering 只用 six components + lactate；SOFA-2 total 仅描述/审计 |
| H3-2 missingness 声明与 owner semantics 不一致 | Confirmed | blocker | 是 | 保留 direct observed / owner LOCF available / unavailable receipts；排除 owner-unavailable synthetic value；clustering 不再额外插补 |
| H3-3 scaling 仅声明、未冻结执行/证据 | Confirmed | major | 是 | deterministic observed-value coordinate z-score（ddof=0）；输出 center/scale/mask policy 与 digest-bound scaling manifest |
| H3-4 k=6 上界可被误称为充分最优 | Confirmed | major | 是 | H3 固定 2–6；上界 minimum BIC fail closed，reason=`H3_NO_INTERIOR_BIC_OPTIMUM` |
| H3-5 refit engine failure 与低 ARI 混为一类 | Confirmed | major | 是 | 稳定性低与数值/refit failure 分别给出稳定 failure class/reason code |

## 实现证据

- 单一科学真源：`benchmarks/figure2_canonical9/case_scientific_protocol.py`
- runtime/launch 双摘要门：`benchmarks/figure2_canonical9/scientific_protocol_authority.py`、`benchmarks/figure2_canonical9/realrun_authority.py`
- materialization 与 Agent-visible projection：`benchmarks/figure2_canonical9/materialization_plan.py`、`benchmarks/figure2_canonical9/protocol_prompt.py`、`tools/materialize_canonical9_miiv.py`
- H3 native-export owner receipt：`src/easyicu/api/extraction.py`、`src/easyicu/concept/export_metadata.py`、`src/easyicu/research_agent/cohort/materializer.py`；真实 native-v2 producer 保留 `*_observed` / `*_available`，duplicate-grain consolidation 只用 owner-available value 聚合，缺一即 fail closed
- E2/H2 current-run authority：`src/easyicu/research_agent/authority/current_case_scientific_runtime.py`；E2 唯一 24 h landmark/RCS primary 与 H2 唯一 current-source feasibility-only plan 绑定 execution digest
- E2/H2 deterministic executors：`src/easyicu/research_agent/execution/runners/landmark_spline_executor.py`、`src/easyicu/research_agent/execution/runners/source_feasibility_executor.py`
- H3 digest-bound execution contract：`src/easyicu/research_agent/trajectory/scientific_runtime_authority.py`
- H3 exact seven-coordinate representation：`src/easyicu/research_agent/execution/runners/trajectory_scientific_representation_executor.py`
- H3 pre-BIC scaling / candidate k / BIC selection：`src/easyicu/research_agent/execution/runners/trajectory_scientific_candidate_executor.py`
- H3 stability replay and digest verification：`src/easyicu/research_agent/execution/runners/trajectory_stability_executor.py`
- pipeline/runner binding：`src/easyicu/research_agent/pipeline.py`、`src/easyicu/research_agent/execution/phase.py`、`src/easyicu/research_agent/execution/runners/selection.py`、`tools/run_research_agent_bench.py`；caller 对 execution contract 或 projection digest 的覆盖在 Provider 前 fail closed

当前 normalized digests：

| Case | Protocol content SHA-256 | Runtime projection SHA-256 | Execution contract SHA-256 |
|---|---|---|---|
| E2 | `a3213192320226ddfd7c767885686551fca63e1df0ecfb863279233e52286a86` | `1a267cd6e03452c2bbf5cf71d7281837f7d9529ba91f15901bbf9f6b39fcdd23` | `ea104a6245672106b75eaa568bf1d10db2c97774df9de9152061f8215f947f90` |
| H2 | `8242d0f783e894eb45d578dc1630beedeb38c3f537a3a0cf3c2962f6f223956c` | `ac990d62490906436fd234eb6fc780def048bf8eb60aba8e7edb7599519752a2` | `28967edb8e689ff734aa8115471e75001bd12f18315849f3a6c6d39c370c7a0d` |
| H3 | `6c3d46d0db23770826b494845670ae5319d862e5e8ffbbc664bc130b507f3fc7` | `2b8e1926097aa74128b9b6c9c12abf8739ad46da66e6fc9efcec58a01c30ff85` | `304f74d726eaa7523e2b8ab5b643ffe045b04d0228d3bb0878b46255593f2130` |

## 负向回归

- human attestation 之后只篡改 Agent-visible runtime guardrail，即使同时重算 JSONL 自声明摘要、协议版本和协议内容不变，real-run launcher 仍在 Provider 前 `PRODUCTION_INPUT_AUTHORITY_INVALID`。
- owner-unavailable SOFA-2 synthetic zero 不进入 H3 trajectory；provenance 仍计入 `unavailable` 分母。
- native ExportPackage 未授权或缺失任一 SOFA-2 owner receipt 时 materializer fail closed；不会回退为 `direct_observed`。
- native ExportPackage 同一 stay/time 同时存在 unavailable synthetic zero 与 owner-observed value 时，pandas 与 bounded DuckDB 两条 producer 路径均只聚合 owner-available value，不会把 receipt 与错误数值拼接。
- H3 Agent 将 feature/k/stability 声明改成另一组时，plan validator 或 deterministic executor 在科学结果前 fail closed；签署 projection 不再只是提示词/provenance。
- E2 Agent 改写 landmark primary method/intent/input/output/digest 时 plan validator fail closed；真实 execution router 只能选择签署的 deterministic RCS owner。
- H2 plan 只能包含一个签署的 feasibility-only owner；增加 PSM/IPTW/control/effect step 即 fail closed，deterministic output 的 effect estimate 恒为 null。
- launcher 收到 caller-supplied scientific authority 或 projection digest 时，只接受与签署 projection 完全一致的值；任何覆盖尝试均在 pipeline/Provider 前拒绝。
- seven-coordinate matrix 只由 six SOFA-2 components + lactate 的 0–72 h / 12 h maximum grid 构建；SOFA-2 total 不进入 clustering coordinates。
- pooled observed-value z-score 在 candidate BIC 前只计算一次；candidate-selection 与 stability 必须共享同一个 scaling manifest/digest。
- H3 minimum BIC 落在 k=6 时输出 `H3_NO_INTERIOR_BIC_OPTIMUM`，不冻结 phenotype、不扩大候选范围。
- stability refit 不足为 `TRAJECTORY_REFIT_ENGINE_FAILURE`；已完成但 ARI<0.70 为 `TRAJECTORY_STABILITY_BELOW_THRESHOLD`。

## 本地验证

- 最终定点与相邻回归：`191 passed`（protocol/authority/launcher/materialization/runner/config/native-v2 producer/materializer/review packet）。
- native-v2 SOFA-2 producer 正/负/duplicate pairing 两条 backend：`5 passed`。
- E2/H2/H3 authority → real execution router：`8 passed`（包含 plan drift、deterministic result 与 config pairing）。
- 完整 research-agent CI 首轮在 `9,789 passed` 后只暴露一条冻结 golden 漂移：新增 H3 scaling manifest 合法增加一个 current evidence/alias，但旧 golden 仍把后端浮点摘要派生的 evidence id 当成跨 Python 稳定身份。golden 现改为验证 claim 的稳定 evidence role/owner join，并保留数值、source field 与 owner 绑定；更新后的 golden 文件本地 `5 passed`。
- Ruff、JSON 解析、`git diff --check`：通过。
- import-linter：7 个 contracts 全部 kept；module-graph diff 通过；deptry 无问题。

注意：上面的本地扩大回归不是 exact-head GitHub full CI 的替代品。新 SHA 推送后仍须等待主 CI、research-agent CI、Pi security、runner trust 与 portability 全绿；之后才在仓库外生成新的 `UNSIGNED / NON-AUTHORIZING` 真人双签包，并只对上述 finding 做一次独立 AI 定点复核。
