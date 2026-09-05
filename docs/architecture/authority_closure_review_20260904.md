# EasyICU 架构复审与权限边界修补（2026-09-04）

结论：受治理的临床研究 runtime 是符合现有代码的定位；“governed research compiler”可以作为演进目标。当前不能仅凭 frozen 标志、SHA256 字段、模块图或测试数，就宣布整个科研架构已经封口。本轮按实际调用链修补可定位的缺陷，并保留正式实验、论文权威和后续结构演进的边界，不给没有统一评分量表的 8.4/10 再加一个主观分数。

## 复核坐标

- 产品源码基线：`main@89cb7ed00dd08545bf1a455f2e63ed86ad87fff8`。
- 隔离修复：`codex/authority-closure-20260904`，工作目录 `EASYICU-authority-closure-20260904`。
- 主 checkout 有 84 个已跟踪修改、3 个未跟踪路径和活动 Web 服务。本轮不合并到该工作区，不重启它的服务。分支选择来自实时 Git/进程核查，不沿用旧文档“只有两个 clean worktree”的描述。
- 正式 Figure 2 候选仍为 `802bcf5fc3e1bf44304a6d6e626b74d41852d557`。其 [CI run 33859935779](https://github.com/shen-lab-icu/EASYICU/actions/runs/33859935779) 的三版 Python、安装包及三平台任务均为 success，实时读取结果保存在 [prior_candidate_ci.json](../evidence/authority_closure_20260904/prior_candidate_ci.json)。这不能验证本分支，也不能授权 Provider/Qualification/核心实验/论文。
- 复核范围：executor selection、discovery/Web handoff、plan review payload、capability catalog、figure ownership、终态 receipt、Web RunRecord 及相邻恢复/签核调用链。未运行新的患者数据分析、付费 Provider、正式 benchmark 或浏览器端完整研究 UAT。

## 对原审阅需要补充和校正的判断

| 原判断 | 当前源码支持的精确说法 |
|---|---|
| Executor 注册顺序可以决定 owner | 确认。重复 key 门不能防两个不同 key 同时认领。必须先完成全部 claim，再允许唯一 owner render。 |
| 0 owner 都应 UnsupportedCapability | 需要修正。某些科学能力明确采用受限 Coder，0 个确定性 owner 不能直接等同于不支持；是否允许 Coder 仍由上层 capability、binding、receipt 和审批控制。 |
| Discovery 没有 authority | 过度概括。Web 已有 canonical 文件摘要、readiness/prior-art 绑定和接受事务；缺的是 core packet 的真实冻结、来源内容 seal 和跨入口一致性。 |
| Frozen ResearchContext 已完全不可变 | 需要限定。外层 frozen 不会递归冻结 cohort/dict/list。当前深层可变 ResearchContext 本身不等于完整 execution authority；RunInputCapsule 和审批 checkpoint 的 digest/revalidation 才是强边界。 |
| Human approval cryptographically binds = 签名 | 内容/执行身份的摘要绑定已经存在，但 SHA256 不是签名者认证。普通本地确认、formal signer 和 publication signoff 不能混用术语。 |
| Capability IR 尚不存在 | 已有稳定 ID、输入/产物/diagnostic/claim ceiling 注册表，应加强这一 owner，避免再造第二个 catalog。字符串声明仍不能替代 typed executable contract。 |
| 当前还有 29 个基线失败 | 这是旧提交的历史证据，不能作为当前状态。较新的正式候选全量 CI 已绿；本轮遇到的问题逐项诊断，没有添加 xfail、放宽 validator 或删除失败测试。 |
| 有 receipt 就能证明删除的 run | receipt 留下内容承诺和当时的状态摘要；它不保存原始产物，也不能单凭自摘要认证来源或重建已删除的实验。 |

## 已实施的修补

### 1. Executor 的语义唯一 owner

[step_executor_registry.py](../../src/easyicu/research_agent/execution/step_executor_registry.py) 将查询拆成 `claim` 和 `render_selection`。先评估所有适用 executor；多个科学 owner 时抛出带排序 owner keys 和 step ID 的 `AmbiguousExecutorOwnership`，任何 renderer 都不会启动。缺少 plausibility receipt 的认领者仍计入 ownership，不能借它的拒绝把权威让给另一条路由。零 owner 和唯一 receipt refusal 保留上层已存在的受治理降级判断。

验证包含：不同注册顺序的语义冲突、全部 claim 先于 render、后续 predicate 异常、inapplicable 路由、receipt refusal、重复 trace，以及完整 built-in registry 在两个保存的真实计划 corpus 上正序/逆序决策一致。组合回归还覆盖现有 selector/runner 合同。该 corpus 回归不声称穷举所有未来科学计划；运行时的全量 claim 检查负责拒绝新出现的重叠。

### 2. Discovery 与审批载荷的深层冻结

新增只负责 JSON 容器的 [frozen_payload.py](../../src/easyicu/research_agent/contracts/frozen_payload.py)，将嵌套字典/列表复制为不可变映射/元组，并显式产生不共享引用的 JSON 投影。

- `DiscoveryHandoffPacket` 升到 v4：真实 frozen，canonicalization 在字段校验中完成，来源 SHA256、candidate SHA256、handoff SHA256 必填。使用/写入时复核 seal 和源文件；未经校验的 `model_copy(update=...)` 不能沿用旧 seal。
- `LongitudinalAnalysisTaskPack` 升到 v2：父包、子任务及集合冻结，绑定 manifest、候选和整包摘要；保持 `hold/awaiting_human_confirmation`，不产生科学结果/论文权威。
- Web 按 `handoffs/<handoff_sha256>.json` 保存新版本，确认或内容变更生成新 digest；旧文件不可覆盖。文件字节摘要与语义 packet 摘要分别验证。
- `PlanReviewAuthority` 的完整 plan payload 与 evidence map 也深层冻结，Web/Workflow 使用显式 wire projection，避免把不可变内部对象误交给 JSON 边界。

v3 Discovery 和 v1 longitudinal 旧包不会被静默补摘要并当成已批准新包。需从仍可验证的来源重建新 proposal，重新绑定该版本的确认。Discovery 的确认是研究设置的输入，仍不能替代科学计划审查和 execution approval；这些摘要也不自动证明文献支持性或 signer 身份。

### 3. 在已有 capability owner 上收紧准入

[capability_registry.py](../../src/easyicu/research_agent/planning/capability_registry.py) 在导出 vocabulary 前拒绝缺失/重复 ID、缺输入/结果/diagnostic、非法执行策略或 claim ceiling；reportable 能力必须声明 scientific validator，确定性能力必须声明 runner 或拥有组合运行时的 scientific owner。各 family 的默认能力改为显式 capability ID，不再依赖 catalog 顺序。

这使新增注册必须补齐基本合同；它不是“所有旧 route 已迁成完整 Capability IR”的声明。科学输入绑定、适用研究设计、必需验证器和结果权限仍由现有 typed owner 在执行时检查。

### 4. Figure implementation 退出 pipeline

association、sensitivity 的 prior-output renderer 及其 parent contract 解析分别移入 `figures/association_prior_outputs.py`、`figures/sensitivity_prior_outputs.py`、`figures/prior_output_contracts.py`。Figure registry 自己有 lazy owner loader，无需先导入 pipeline 才能找到实现；pipeline 只保留调用和兼容导出。

[architecture_delta.json](../evidence/authority_closure_20260904/architecture_delta.json) 记录精确差异：pipeline **8,444 → 7,062 LOC**；其内部 import edges **124 → 127**，并没有声称所有依赖指标下降。包内模块 **635 → 640**，边 **2,613 → 2,632**，循环模块和 SCC 均为 0。增加的五个 Research Agent owner 在 graph baseline 中逐项记名，另有一个包级共享 state-path owner。

只把 pipeline 的 LOC 上限从旧 ratchet 8,458 收紧至 7,062，其他大文件/函数门槛原样保留；图门禁的 cycle/top-level 限制没有放宽。这不是以“刷新整个基线”掩盖旧代码增长。

### 5. 自动保存终态 receipt

将 CLI 的 receipt 算法下沉为 [authority/run_receipt.py](../../src/easyicu/research_agent/authority/run_receipt.py)，pipeline 在终态和最终 checkpoint 写入后自动调用，保存在 run scratch 之外的应用状态目录，文件名为 receipt 内容摘要。Discovery launcher 写完后续 package assessment 后再保存一个版本。

采用先后两次 inventory 检查可观察的读期间漂移；损坏的 optional authority JSON 不再当作缺失；verify 除摘要外还对照源文件重建 facts，防止改写 status 后重算自摘要冒充原 run。写入不可覆盖已有不同内容，也不能把 receipt 写进自身被扫描的 run tree。

完整保留范围、CLI 命令、失效原因和限制见 [run receipts](../evidence/run_receipts/README.md)。未补造旧 run 的 receipt；Web wrapper/local signoff、崩溃前未形成终态的运行和独立 formal harness 各自仍有自己的保留边界，不能把这一个 hook 描述成整机/全流程备份。

### 6. Web 投影的内容一致性

`RunRecord` 的 gate/checks/artifact/signoff payload 深层冻结，HTTP projection 返回独立 JSON 对象。JSON 内容与首次 inventory 的 digest 来自同一批实际读取的字节；再次观察发生漂移时返回 `run_record_changed_during_read`，run_context 与 ledger 身份冲突时返回 `run_record_identity_conflict`。

保留已有 domain/evidence owner，不新增一套负责重新裁决科学状态的 Web RunLedger。该检查能拒绝观测到的混合状态，不是跨多个文件的事务性快照证明。额外验证了 nested gate 签核序列化与 pending plan 投影，防止深层冻结破坏已有 Web wire 接口。

### 7. 组合验证暴露出的两个额外缺陷

- **状态根不一致**：ExtensionRegistry 忽略 `EASYICU_HOME`，使已有隔离测试尝试锁定真实用户扩展目录。将状态路径 owner 从 Web 下沉至 `easyicu.state_paths`，Web 保留兼容导入；扩展和 receipt 复用它。显式 `EASYICU_EXTENSION_HOME`/构造参数优先级保留。修复既防测试污染，也让隔离服务状态一致。
- **文献 wire/type 不匹配**：Web literature authority 返回 dict，但执行适配直接访问 `LiteratureBundle.screening_decisions`。在消费前按原 typed model 验证；合法已绑定文献不重复检索，非法 seed 返回 `bound_literature_schema_invalid`，在构造执行 pipeline 前停止。测试保留原 positive 路径，增加 negative 输入，没有修改科学 screening 标准。

## 验证与可支持的结论

最终代码候选为 `f9340065a0434236a663f1363f93526325742d47`：**1,310 tests passed**，五项架构门全绿（含 **142** 项结构/预算检查、**7** 项 import contracts），进度文档六模块通过且零 warning。后续提交仅登记审阅报告、证据与已验证的 ratchet 文件。

可重复运行的命令、文件列表、源码摘要和最终输出见 [validation.json](../evidence/authority_closure_20260904/validation.json)。测试都是工程合成/保存 fixture，未把科研失败从正式分母中移除。

首次宽组合中的 46 个失败都先被真实状态目录权限错误挡住；共享 state owner 修复后剩下一项文献 dict/type 真缺陷，再经修复重跑。早期 selector 子进程的相对 PYTHONPATH 错误已通过绝对 checkout 路径纠正。所有中间失败保留在验证说明中，不把它们叫作可永久接受的 baseline。

本轮证据分级为 **combined owner/adjacent tests + architecture gates**。没有启动新的 full exact-head CI：产品 main 仍有并行未提交变更，最终合并队列未结清。旧候选的全量绿灯不移植给本分支，今后汇总时仍需按项目政策验证新的精确 aggregate SHA。

## 尚不能宣布完成的内容

1. pipeline 仍有 7,062 行，execution phase、preflight、repair/audit 仍有明确结构债。下阶段应按 typed service 的职责逐个迁出 implementation；不能为了 LOC 新建第二套审批、repair 或 scientific routing。
2. Capability IR 的所有旧 route 迁移、必需 validator 的统一可执行 manifest，以及跨 family 合同编译仍是渐进工程。此次完成 admission 和唯一 executor resolution，不冒充整个 compiler 已实现。
3. ResearchContext/AnalysisPlan 的编辑期模型仍不是全面深层不可变。执行权威必须继续由 capsule、review checkpoint、typed binding 和使用时 revalidation 保证；本轮强化的是已批准 snapshot 及 Discovery/Web 边界。
4. 正式 source tree、独立 signer、tag/image/input/network/budget 的外部锁定和备份保留需要正式流程；本次没有新正式 run，无法补齐已删除的历史实证。receipt 自动化解决未来终态 snapshot 的默认保留，不自动授予软件论文或临床科研结论。
5. 本分支待并行 main 工作稳定后审阅整合。整合时特别检查双方都修改的 Web runner、capability 和投影 owner；不得直接覆盖主目录的未提交改动。
