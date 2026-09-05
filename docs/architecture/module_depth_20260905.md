# 三项架构深化的实现与验证

日期：2026-09-05。检查起点为 `e2a0d10e729acc0a24239e2c333556842712c596`；三项功能实现汇总为 `a7b0ded38dd8ac4c276311fbc1f1ef259b9d4dd4`，位于 `codex/authority-closure-20260904`。后继仅同步结构测试、ratchet、证据和文档。

用户已要求修复 HTML 架构审阅中的全部三个候选。本轮沿用现有 module，收紧 interface、seam 和 adapter 的职责；没有增加运行时模块或另建 authority。此前 supplementary 绑定、Discovery 来源成员关系、绘图适用性、视觉建议与 presentation 参数的五项修复仍以 [上一轮报告](authority_followup_20260904.md) 为证据。

## 完整执行器判定：`f7e7fc9`

原来的计划声明检查只需要查询 owner，却调用会生成代码的 selector；候选报告又丢弃拒绝原因和缺失声明，并自行运行 missingness 分类器。两项负向回归在修改前均失败，见 [negative-executor.log](../evidence/module_depth_20260905/negative-executor.log)。

`StepExecutorRegistry.resolve()` 现在返回完整的 `StepExecutorDecision`。计划门与测量工具查询该结果，执行阶段明确调用 `render_selection()`，报告直接投影相同判定。所有 owner 先完成 claim，歧义仍拒绝；receipt-required、无 owner 和 declaration refusal 的规则保留。Missingness 细分诊断回到该 runner adapter，拒绝原因和缺失字段进入候选报告 `/3`。

删除独立 `selection_report.py` 后，判定与诊断的 locality 集中到现有 module；同一 interface 同时服务计划、执行与测量，获得实际 leverage。测试覆盖唯一 owner、注册顺序、拒绝条件、查询不 render、投影不重新分类及报告字段完整性。该判定是短期查询结果，不是新的计划批准或执行许可。

## 当前图件合同读取：`2fcd930`

三处报告检查原来分别打开 JSON，读取异常后静默跳过。修改前同一损坏合同在三处均缺少可追溯的读取错误，见 [negative-figure-reader.log](../evidence/module_depth_20260905/negative-figure-reader.log)。

现有 `figures/contracts.py` 增加 `FigureContractInventory`：一次读取记录内容 SHA256、深层冻结的合同和明确错误；在一个 readiness pass 内共享给 article、display-suite 和 figure-strategy。复用必须匹配 run 与当前合同选择；可变输出是独立副本。损坏、不可读、非法结构或越出 run 的合同都会保留原因并阻断相应检查。

三个调用方的重复 reader 被删除，主图/步骤图分类与读取错误在同一 seam 上处理。当前成功步骤、显式空 `contract_files`、legacy 记录兼容，以及 primary-lineage 排除 publication 目录的规则保留。该 inventory 只是本次读取观察，不是 EvidenceStore、run ledger 或长期缓存。严格图件计划绑定、数值/来源、claim-boundary 校验继续由原 owner 独立执行；共享读取不意味着整个验证过程只访问一次文件。

## 完成状态与论文权限：`a7b0ded`

原 `_compute_readiness_gates()` 在内容就绪时先返回 `paper_authorized=true`，随后 artifact writer 才按执行身份、计划摘要和降级状态重写。使用已有合成证据 fixture，可在修改前复现该中间状态，见 [negative-completion.log](../evidence/module_depth_20260905/negative-completion.log)。**这不证明原最终写入门能被绕过**；问题是正确性依赖后续调用顺序。

`RunCompletionFacts` 明确接收现有 validator 的布尔结论；`RunCompletionDecision` 在首次投影之前统一组合内容门、scientific maturity、plan truncation、diagnostic/replan 降级、execution eligibility 和 verified plan SHA256。缺少身份时默认无法授权论文；非布尔结论拒绝进入该 interface。

Writer 直接消费最终 `completion_status` 和权限，不再修正临时 `paper_authorized`。报告使用同一个状态规则，并拒绝矛盾的状态投影。原有数值、来源、科学要求、审批与独立 human sign-off 的职责均保留。`publication_ready` 表示内容就绪；`submission_ready` 继续采用内容加行政元数据的历史语义，两者均不替代论文权限或正式实验合同。

删除测试中“缺失键自动返回 True”的 `_Gates`，改为通过公开 typed interface 验证每项阻断条件；另用真实 artifact writer 的合成 fixture 检查无身份、缺计划、合法摘要、非法摘要、强制降级五组情况，核对返回值、JSON 和文本报告一致。

## 验证与结构变化

可复现环境、命令、文件清单和日志 SHA256 见 [validation.json](../evidence/module_depth_20260905/validation.json)。

| 范围 | 结果 |
|---|---|
| 三项改动的 combined 聚焦/相邻检查 | 381 passed、1 skipped、23 warnings |
| Readiness/display 集成检查 | 22 passed、261 deselected |
| 五项廉价架构门 | 全部通过；结构/预算 142 passed，import contracts 7 kept |
| 遗留跳过项和更新后的结构检查 | 2 passed、1 skipped；替代的实际文件权限测试通过 |
| 第二位 Agent 只读源码审查 | 未发现新增 actionable correctness/authority regression |

唯一跳过项是旧的 debug-dump 源码字符串断言，仓库早已注明由实际文件权限测试替代；本轮补跑替代测试通过。架构门首次失败有两个明确原因：命令 PATH 未包含现有 import-linter，以及结构测试仍查找旧 selector 赋值字符串。修正后重跑五门通过；保留首次和最终日志，没有使用 xfail 隐藏新失败。

| Module / 指标 | 本轮起点 | 本轮实现 |
|---|---:|---:|
| `execution/phase.py` LOC | 5,679 | 5,636 |
| `reporting/readiness.py` LOC | 3,044 | 3,009 |
| `planning/figure_strategy.py` LOC | 936 | 891 |
| `reporting/article_contract.py` LOC | 1,313 | 1,280 |
| 生产模块数 | 642 | 641 |
| 静态 import edges | 2,642 | 2,642 |
| 循环模块 / SCC | 0 / 0 | 0 / 0 |

复杂度下沉到现有深 module：executor registry 为 331 LOC、图件合同为 592 LOC、completion 为 311 LOC。Pipeline 保持 7,062 LOC。Ratchet 只下调 execution/readiness 的 LOC 上限，并记录删除的 report module；旧 readiness ratchet 尚为 3,067，现收紧为 3,009，没有放宽其他大文件或循环约束，历史准入说明保留。

## 交接范围

这次完成的是三项工程修复及 combined/架构验证，**不是 full exact-head CI、正式实验、研究结果或论文授权**。没有启动 Provider、研究运行或新的图件导出验收；此前绘图修复的合成导出证据仍属于上一轮。

产品目录继续为 `main@89cb7ed00dd0`，保留 84 个已跟踪修改和 3 个未跟踪路径，8765 listener 仍为 PID 32526。正式实验候选仍为干净的 `802bcf5fc3e1`。本轮未整合或部署到 main；下一步应在并行工作稳定后进行定向整合，再按最终汇总候选执行所需 CI。
