# Data-first Idea Mining route（2026-07-21）

## 结论

EasyICU Idea Mining 已新增 provider-free 的 data-first 正式入口：它从既有 prepared cohort 和谐和概念可用性生成候选，再把候选送入原有且唯一的 source snapshot → concept mapping → real-data feasibility → PubMed prior-art → registry → discovery ledger → human-confirmation 流程。它不另建第二套科学账本，也不调用 LLM 猜选题。

在既有 MIIV 94,458-stay 宽队列上，系统筛选 22 个 derived predictors，得到 10 个可执行 predictor→hospital mortality rows；随后在既有 full6 prepared exports 上完成 provider-free source-status 审计。第一轮形成两个可用于 Figure 5 过程栏的可审计 vignette：

1. **DSI → mortality：fail-closed rejection。** 精确检索 19 条，冻结命中 PMID 32296976 已在两个 populations 中评估 DSI 与死亡；因此不再把它列为“未外部验证”的候选。
2. **Albumin-corrected calcium：data-quality reframe。** 死亡关联精确检索 102 条、已有 eICU-CRD 直接研究（PMID 36619536），故拒绝重复关联研究；provider-free full6 审计证明它可作为跨库 measurement/source-status audit：5 库达到预设的每库 ≥500 个有效 stay，合法同时间派生覆盖从 HiRID 1.50% 到 AUMCdb 80.17%，跨度 78.67 个百分点；SICdb 为 0，明确排除。该终态为 `answerable_requires_human_confirmation`，仍不授权科学分析或论文 claim。

两条都不是新科学结果；权威人工复核记录为 `research_output/experiments/FIG5-DISC-010/review/shortlist_review.json`。

## FIG5-DISC-011：第二个独立候选（2026-07-22）

系统用同一 provider-free 标准账本扩展到 `los_icu`、`persistent_critical_illness` 与 `aki`，自动把 **modified shock index → AKI** 排为 external-validation review 首位；不是人工指定题目。泛 PubMed query 最初被 `modified`/`shock` 单词污染而返回 1,921 条，`eb1c2e8` 删除无意义 singleton expansion 后降为 15 条邻近记录，同时保留完整术语 `modified shock index`/`shock index`；exact query 仍为 0。该信号只触发人工复核，不构成 novelty claim。

`d3a16df` 新增通用 predictor/outcome pair answerability：必须在预声明数量的数据库同时满足 predictor 与 outcome stay-level overlap，且永远不推断时间窗、结局 onset、估计目标或因果方向。只读既有 `/Volumes/外置硬盘/easyicu_data/full6_20260717` 后，六库联合可用率均高：AUMC 99.6%、MIMIC-III 85.2%、eICU 93.0%、MIMIC-IV 99.5%、SICdb 93.0%、HiRID 96.3%；数据门终态为 `answerable_requires_temporal_protocol`，analysis/paper 均为 false。

随后的人工作为科学 gate 复核了近邻文献：PMID 32004778 已在 82 家 ICU、18,197 名患者中报告 cumulative abnormal SI exposure 与 AKI；PMID 38920191 报告术前 SI 与术后 AKI；PMID 36586312 也把 SI 列为创伤后 AKI predictor。故“exact MSI→AKI 为 0”只是术语空白，不是足够的科学 gap。候选最终 `reject_as_insufficiently_differentiated`；`7ea1383` 将该经验写成通用选择规则：存在同结局近邻构念 prior art 且无具体 differentiator 时，不得进入 external-validation shortlist。

派生语义也被显式区分：导出的 materialized MSI 是在原始同步 HR/MAP 上计算后再进入宽表；在聚合后的 HR/MAP 上重算非线性比值不应被误当作同一权威。系统因此以 materialized MSI 为候选权威，同时保留重算差异为描述性 provenance，而不把预期的 aggregation non-equivalence 报成数据错误。

`16122d6` 将 measurement/source-status audit 按 predictor/audit identity 跨 outcome 去重：同一 corrected-calcium audit 在三个结局上下文中只检索和入 shortlist 一次，同时保存全部 origin IDs/topics/outcome contexts，不再靠重复题目凑实例。

权威复核：`research_output/experiments/FIG5-DISC-011/review/shortlist_review.json`；所有 5 个被引用的 triage/manifest/shortlist/source-status 文件 SHA 均已重算一致。当前累计 **0 个 paper-authorized 科学发现**；FIG5-DISC-010/011 是两个 agent-produced、可审计的过程实例，后者证明“数据充足 + exact term gap”仍可被科学 gate 拒绝。

## 数据边界

- 未重新抽取六库。
- 复用 prepared cohort：`research_output/experiments/FIG5-DISC-001/triage/miiv_wide_idea.parquet`
- 该 cohort 来源于既有 `/Volumes/外置硬盘/easyicu_data/full6_20260717` 工作流。
- 候选生成只看 prepared-data 可测性和字典覆盖；结局结果不参与候选排序。
- PubMed 是唯一网络依赖；candidate generation 的 LLM 为 `_ProviderCallForbidden`，任何 provider 调用都会测试失败。
- source-status profiler 只读取 `/Volumes/外置硬盘/easyicu_data/full6_20260717` 下既有 `demographics/chemistry/blood_gas.parquet`，没有启动 convert/extract，也没有改动 full6 字节。

## FIG5-DISC-012/013：从“多列组合”转向“文献缺口 × 数据可回答”（2026-07-22）

`FIG5-DISC-012` 将 data-first review surface 从单一 top-1 扩为最多 3 个 predictor/outcome 均尽量不同的候选，避免第一个泛化组合占满人工审阅。但真实复核仍拒绝 MSI→LOS、eGFR→prolonged ICU stay、BUN/Cr→LOS：三者数据完整，却都缺少来源文献给出的具体人群、time zero、测量政策或机制差异化。该运行进一步证明“枚举更多组合”不能替代科学问题来源。

`FIG5-DISC-013` 因此首次运行真实 literature-first 路线：30 篇来源中 26 篇使用 Discussion/Limitations，4 篇为 abstract fallback；EasyICU 经 5 次 provider extraction calls 生成 35 条 literature ideas。运行没有重抽六库，仍复用 94,458-stay prepared cohort。其优点是能提出 vasopressin timing、protein dosing/UCR、sepsis coding validity 等来源可追溯的 gap；同时暴露三类通用缺口：

1. 共享词可造成假映射（`pulsatile blood flow`→`pH of blood`）；
2. 治疗动作可错误绑定测量值（`protein dosing`→`total_protein`、fluid dosing→net fluid balance、care bundle→KDIGO urine-output stage）；
3. 旧顺序先给全部 35 条做 prior-art，再发现多数不可执行或数据不可回答，约 66 个 cached queries 中大量没有决策价值。

本批据此做了可泛化而非题目特异的修复：

- partial concept match 必须高于 0.5 specificity；`blood` 等载体词不能单独提供概念身份；
- concept catalog 暴露 host-owned semantic category，dose/initiation/therapy/bundle/protocol/strategy 等治疗语义不能绑定到非 intervention/medication 概念；
- 自动 data-first predictor pool 不再把 AKI、persistent critical illness 等派生 endpoint 静默改作 exposure；
- 新叶子模块 `idea_mining_selection.py` 在 prior-art 前按 host concept mapping、joint data feasibility、具体 differentiator、exposure contrast 和 completeness 形成有界 review surface；真实工具默认最多筛 12 条；未检索候选不获得 novelty verdict，也不能进入 proposed choice set；底层 API 默认仍保持历史 unbounded 行为以兼容已有调用者；
- real discovery tool 的无效默认模型别名 `gpt5.4` 改为代理实际支持的 `gpt-5.6-luna`。

权威复核：`research_output/experiments/FIG5-DISC-012/review/shortlist_review.json`、`research_output/experiments/FIG5-DISC-013/review/shortlist_review.json`。两轮累计仍为 **0 个 paper-authorized 发现**；价值是系统从“产生很多看似能跑的题”转为先识别语义真实、数据可答、值得查文献的题。

## FIG5-DISC-014/015/016：批次恢复、answerability-first 收敛与论文过程例（2026-07-22）

`FIG5-DISC-014` 暴露了新的工程浪费：30 篇来源分 5 批提取时，第 5 个 HTTP 200 response 是 malformed JSON，旧实现因此丢弃前 4 批已经合法解析的结果。`5a463d0` 新增 content-bound extraction batch receipt：精确 request、raw response、parse status 与 SHA 写入独立 receipt；后续只复用验证通过的 parsed receipt，失败批可单独重跑，tamper 必须 fail closed。该机制不修补模型 JSON，也不把 malformed output 收入科学账本。

第一次恢复后得到 34 条 provenance-valid ideas；旧选择器仍把 5 个不可用题送进 prior art，包括 pediatric sepsis 与 adult prepared cohort 冲突、bicarbonate joint n=0、mannitol 暴露无 contrast、vasopressin 仅约 6% observed 且没有 absence-as-zero 权威，以及缺少差异化的 baseline SOFA→mortality。`c3443a4` 将 prior-art search budget 进一步绑定到 host-owned answerability：已知 joint n=0 或 contrast=0 直接退出；显式人群冲突退出；治疗/因果候选在极低观测率且无 absence 语义时不消耗检索预算。这个门只决定“是否值得查文献”，不代替科学可行性或 novelty 审核。

`FIG5-DISC-015` 复用 014 的 5 个 verified extraction receipts（0 个新 extraction provider calls），再次从 30 篇来源得到 34 条有效 ideas，但只有 1 条进入 prior art。唯一候选 baseline SOFA→30/90-day mortality 有 94,449/94,458 joint-complete rows，却有 1,252 broad PubMed hits，且没有 time zero、transportability、measurement-policy、subgroup、mechanism 或 model-role differentiator，因此被 `reject_as_undifferentiated_crowded_association`。prior-art cache artefact 从早期 `FIG5-DISC-013` 的 70 个降至 2 个（97.1% operational file-count reduction）；这只是调度效率证据，不是科学 endpoint。

`FIG5-DISC-016` 将两个互补的 agent-produced 过程例做成带 SHA 的 manuscript index：① `FIG5-DISC-010` 的 corrected-calcium 跨库 measurement/source-status audit，已验证可回答但仍须临床+方法双审；② `FIG5-DISC-015` 对高完整度但泛化 SOFA-mortality association 的正确拒绝。该 synthesis 明确 `paper_authorized_scientific_discoveries=0`，没有把过程例改写成阳性发现，也没有重抽六库或运行新分析。

权威入口：

- `research_output/experiments/FIG5-DISC-015/review/shortlist_review.json`，SHA-256 `7610850c4861a29e636bc47c77fc0a97e4770079f67d7484255a18965f3bd2cf`
- `research_output/experiments/FIG5-DISC-016/manuscript_examples/examples.json`，SHA-256 `83366ec3749caea42b453c37a3ba08295a54ef2c9ad807e0663e5865a0049768`
- `research_output/experiments/FIG5-DISC-016/manuscript_examples/examples.md`，SHA-256 `2a2bf0b7d5b1c58bc62686ec405f76970afcc786fa62c4ae4e0eef0757a24ee8`

## 本批架构修复

- `run_idea_mining_dry_run` 接受经过严格 snapshot/citation/verbatim 校验的 `precomputed_literature_ideas`，让 deterministic/data-first 路线复用同一标准 ledger。
- 新增 `idea_mining_data_first_route.py`，输出冻结 prepared-data SHA、route manifest 和 bounded human-review shortlist。
- 文献精确查询可带 host-curated 同义词，避免内部 concept key / 缩写造成伪 gap。
- cross-database prior art 从 title + abstract + rationale 识别，并覆盖 `external validation`、`multicenter`、`two/both populations`、多医院等常见表述。
- prior-art client 返回 search error/空结构时 fail-closed，不能把网络失败读成文献空白。
- association 文献饱和不再阻断独立的 measurement/source-status audit 路线；两者被明确拆成不同研究问题。
- measurement/source-status reframe 获得独立稳定 `reviewidea_*` 身份，不复用原死亡关联 candidate ID；并运行自己的 ICU measurement-availability/missingness PubMed query。本次 corrected-calcium audit query 为 0 hits，但只作为人工复核触发器，不作为 novelty claim。
- 新增通用、formula-agnostic `idea_mining_source_status.py`：绑定每个输入 parquet 的 byte SHA/schema SHA，区分 structural no-source、source-present unmeasured、out-of-range/contradictory 与 valid observed；同时核对物化派生列、host 重算和 measured alternative。
- 新增预声明 `MeasurementAuditCriteria`：候选必须在足够多数据库达到最小有效 stay 数，并有足够的跨库 coverage contrast；字典 `6/6 resolvable` 不再被当成真实数据 `6/6 answerable`。
- 新增 `PairAnswerabilityCriteria`：跨库关联候选必须有足够的 predictor/outcome stay-level overlap；通过只允许进入 temporal-protocol 设计，不授权分析。
- prior-art broad query 不再把复合术语拆成无意义 singleton；measurement audit 跨 outcome 去重并保持稳定 review identity。

## full6 source-status 结果

| 数据库 | denominator stays | 合法同时间 corrected calcium | 覆盖率 | ionized calcium 覆盖率 |
|---|---:|---:|---:|---:|
| AUMCdb | 23,106 | 18,523 | 80.2% | 79.3% |
| MIMIC-III | 61,532 | 27,880 | 45.3% | 47.3% |
| eICU-CRD | 200,859 | 138,236 | 68.8% | 18.2% |
| MIMIC-IV | 94,458 | 45,661 | 48.3% | 46.3% |
| SICdb | 27,386 | 0 | 0.0% | 97.1% |
| HiRID | 33,905 | 508 | 1.5% | 86.9% |

物化 `corrected_calcium` 与同一 prepared row 的 `ca + 0.8*(4-alb)` 在 AUMC/MIMIC/eICU/MIIV 有少量 >0.1 mg/dL 差异（分别 142/49/897/136 rows）。这不是被静默抹平的浮点噪声；在任何 value-level 跨库比较前必须由临床/方法审核解释其时间对齐或物化来源语义。

## 权威产物

- Triage：`research_output/experiments/FIG5-DISC-010/triage/candidate_triage_report.json`
  - SHA-256 `d50d6f6faa75c98c3cce538ea676080aaa820e0dd70df85066ca44d646a5467a`
- Route manifest：`research_output/experiments/FIG5-DISC-010/triage/data_first_route_manifest.json`
  - SHA-256 `b756f504d1cbb33f414e40fe5c4bcb063fb34c488ea677ca922728fa82fe8022`
- Review shortlist：`research_output/experiments/FIG5-DISC-010/triage/data_first_review_shortlist.json`
  - SHA-256 `27aa55a72917c5d921a761845d60022115b3b180da493f251026cec796738b77`
- Human review：`research_output/experiments/FIG5-DISC-010/review/shortlist_review.json`
- Source-status JSON：`research_output/experiments/FIG5-DISC-010/source_status/corrected_calcium_source_status.json`
  - SHA-256 `fec949f5271bb56d7afcfc041423556d0cdc02589106450b7f91b0769a5cd553`
- Source-status readable report：`research_output/experiments/FIG5-DISC-010/source_status/corrected_calcium_source_status.md`
  - SHA-256 `d74789dc4ed1a30cd55d239848b6c4281080a7915b9b45ae5c4030362ee484fb`

## 验证

- `PYTHONPATH=src .venv/bin/python -m pytest tests/research_agent/test_idea_mining*.py -q` → **178 passed**（selection boundary 增补后 focused 78 项另行通过）
- concept catalog + route focused → **113 passed**
- Ruff → clean
- architecture lower-is-better gate → 0 regression；research-agent module graph → no new cycle
- 实际离线运行 → 10 ideas / 10 executable / 10 discovery rows；shortlist 只剩 corrected-calcium measurement audit；DSI 已因跨人群 prior art 自动退出。
- 实际 literature-first 运行 → 30 articles / 35 ideas / 5 extraction calls；旧版本暴露的 3 类错误映射均有确定性回归锁。

## 下一步

1. 临床+方法双审 corrected-calcium/ionized-calcium 各库来源、单位、时间对齐与物化差异；审核完成前不做 value-level 跨库比较。
2. 把通过 data-answerability 的 audit protocol 经人工确认后接入 `run_discovery_to_manuscript.py`；当前仍为 `analysis_authorized=false`。
3. MSI→AKI 已因近邻 prior art 且差异化不足而拒绝，不运行关联模型；下一候选必须带可审计的具体 differentiator，不能只替换相近指数。
4. 使用修正后的 literature-first 顺序继续搜索：gap excerpt → semantic mapping → prepared-data answerability → bounded prior art；不再为不可执行的 35 条候选全部查文献。
5. 只有通过 route-specific prior art、data answerability、temporal protocol 和人工确认的候选，才进入正式 analysis。
