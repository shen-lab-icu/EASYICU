# EasyICU 生产代码全量审阅报告（2026-09-20）

- 日期：2026-09-20（UTC+8 会话）
- 分支：`codex/dev9-expansion-20260917`
- HEAD：**`e5e7e4059`**（Wave 3 复核时的实时坐标）。本报告初稿与 Wave 1/2 的坐标基线为 `2835cbbd1`；期间同机并行任务把它的 73 个脏文件整体提交为 `e5e7e4059`，工作树脏项由 **65→73→0**。**本报告自身（283 行）也被那次提交带入 git 历史**，非本次审阅主动提交。
- 逐文件复核：`selection_policy.py`、`study_context_update.py`、`turn_authority.py`、`concept/callbacks.py`、`research_launch_scientific.py`、`tool_catalog.json`、`reporting_checklist.py`、`writer_evidence.py`、`node_app/src/main.mjs` 在两 HEAD 间**零差异**；`webserver/pi_copilot/service.py`（+86）与 `projections.py`（+35）有改动 ⇒ 其中引用行号已在 Wave 3 重新定位（`service.py:2084/2091-2092`、`projections.py:285` 现值均已逐字复读）。

- 范围：仅 `src/easyicu/` 生产代码（用户选定的风险分层口径）。**未含** `tests/`、`tools/`、`scripts/`、`docs/`、`webserver/static` 前端全量（`main.mjs`、`next-actions.js` 等为追 P0 链路顺带读取）。
- 方法：83 个只读审阅批次（48 深读批 + 35 扫描批）逐文件读到末行，无抽样；每条上报 P0/P1 由独立 `gap-claim-verifier` 代理以证伪优先复验；两条 P0 由主审亲跑探针复核。
- 结论口径：**P0** = 静默破坏科学结论/证据等级，或可达性已确认的可利用授权缺陷；**P1** = 可达的真实缺陷；**P2** = 边界/潜在/一致性；**P3/NIT** = 微小。"机制确证" ≠ "影响已量化"，下表分母列显式区分。
- 证据等级：全部为 **focused**（局部探针 + 逐消费端穷举）。未跑 full CI，未真跑任何 research run；下文"盲区"列说明尚未取证的部分。

## 与 20260917 报告的关系

`docs/EASYICU_CODE_REVIEW_20260917.md` 覆盖整仓库但采用"高风险约 40 文件精读 + 其余 AST/模式扫描"的混合口径。本报告的 12 个关键坐标（`_expand_patient_ids`、`rrt_source_complete`、`selection_policy`、`reporting_checklist`、`distribute_amount`、`fill_gaps`、培养判据、`coerce_numeric_fail_closed` 等）在 0917 报告中命中数为 **0**，即本轮发现均为**新增**而非旧问题回归。两份报告口径互补：0917 广度优先，本报告在 `src/easyicu` 内深度优先并强制逐条证伪。

## 严重度统计

| 项 | 数值 |
|---|---|
| 审阅批次 | 83（全部回报，无遗漏） |
| 覆盖文件 | `src/easyicu/**/*.py` **990/990**，616,797 行 |
| 上报 P0 / P1 | 7 / 147 |
| 复验组 | 39 |
| 逐条判定 | 74 VERIFIED / 12 REFUTED / 2 UNDETERMINED |
| 最终定档 | ~~**P0 = 3**、**P1 = 29**（Wave 2 后）~~ 以 **Wave 4 表（最新）→ Wave 3 定档表**为准（Wave 2 后原 45，16 条降档/改判零实例、C59 升 P0、038 回升 P1） |
| Wave 2 降档清单（16） | C10、C4、C37、C12、C13、C23、C24、C30、C16、C67、C21、C51、C45、C47、C48、C49 |
| Wave 2 二次对抗复验 | 14 组（G1–G14 + G4R/G5a/G5b）；结论：2 条 P0 维持并加重、C59 升 P0、038 回升 P1、C44 量化由主审亲自复算确认（26/30 = 87%）、webserver 安全类 REFUTED 判 UPHELD |
| 子代理 token 量级 | 约 1.9×10⁹（Wave 1）+ 约 1.6×10⁸（Wave 2）+ 约 1.9×10⁸（Wave 3） |

### Wave 4 针对性扫描（2026-09-21，未取证条目 + 新线索）

Wave 3 只取证了 65 条被点名条目中的 44 条。Wave 4 补扫剩余未取证条目与本轮新挖线索，回执在 `/tmp/easyicu_review/w4/*.md`（T1/T2 已回收，T3/T4 见本节末"仍在跑"）。

| 条目 | Wave 4 判定 | 关键证据 / 对本报告净剩影响 |
|---|---|---|
| **C38** `cohort_review.py::unique_entity_intersection` 名为交集实为 `min()` 夹取 | **WRONG-DESC + 降 P2** | 登记册 `:127` 与 `:292` 并不互斥，差在分母层：复现的 8 行夹取里 **7 行在 10-stay 历史 scratch 目录 `~/easyicu_export`**，真实发表语料 `full6_20260717/miiv`（94,458 stay）19 行**零夹取**。`:127` 的"12 行"无 provenance，建议删。假 ok 确会落盘，但 133 行快照里仅 1 行、属 demo 源 |
| **新增（自 C38 挖出）** 同形夹取的**放行门** | **新列 P1 候选** | 真实缺陷本体在 `_covered_entities:917/928`，而**同形夹取**另见于 `extraction_filters.py:265`（`min_coverage_pct` 作为**准入放行门**使用）与 `patient_drilldown/__init__.py:2440`。三处须同修，且放行门那一处的后果是"覆盖率不足的数据被放行进入分析"，比 C38 的披露失真更重 |
| **infection_icd 整列 NaN**（我在 Wave 3 当作新证据提出） | **REFUTED（非缺陷）** | 我那句"11 份带该列、11/11 全 NaN"口径错了：列在 **13/13** 份导出里都存在，只有 eICU 那份有部分值（25.8%，约 52,566 个），其余整列空是该概念的**设计边界**；`api/extraction.py:2924-2930` 明文承认 native-v2 会物化 all-null 结构列，并说明该校验故意不拒 `infection_icd` |
| **新增（自 infection_icd 挖出）：Angus 归属误述仍在对外目录发表** | **新列 P1** | owner 早在 **2026-07-17** 就在 `data/concept-dict.json` 的 `description` + `_comment` 里更正："NOT the Angus 2001 ICD-9 code list"、"must not be cited as Angus"，实为对 `eicu.diagnosis.diagnosisstring` 的 30 关键词正则（392,374 / 2,710,672 行命中，已知假阳性含 ulcerative colitis/IBD 约 709 行、非感染性心包炎/膀胱炎）。但**用户可见目录仍在教 Angus**：`concept/catalog.py:204`、`:416` 与前端 `webserver/static/js/data-catalog.js:226`、`:501` 四处全部写 "Angus 2001 (explicit infection codes)"，`tests/` 内 **0 处**钉住该标签。后果：研究者按目录写方法学会把关键词正则声称为 ICD-9 码表定义 |
| **C12** `io/ts_utils.py::fill_gaps` | **STILL-CONFIRMED，改述** | 真实暴露 **135,308 行 / 0.0201%**（headline 805,967 中 83.2% 是 stay-level 设计性空时间戳）；仅 `limits=` 向量化路径、且需整组等距；episode 隔离门不拦。`ts_utils.py` 本轮被并发任务改过，但 hunk 全在 `change_interval`，本条仍未修 |
| **C13** 公共 `pafi`/`safi` | **机制确证，不升档（P2）** | 真实 pafi 最大 800、19 份文件 `share≥10000 = 0.00%`；"docstring 示例本身即触发"**REFUTED**（该仓无 doctest 配置） |
| **WRITER_PANEL**（我 Wave 3 自挖：`writer_evidence.py:1684-1686` 两标签共用一个列表） | **WRONG-LEVEL → P2**（我的"恒假字段"过头） | 恒等性 100% 成立（3,840 配置），但 **55.9% 恰好等于真实收敛数**、偏离时只偏低从不偏高；且合并行为有 gate 测试钉死（护身符已找到）。我说的"8 条不等"错，全语料 **71 条**，且不等件全 ≤09-01、相等件全 ≥09-09，与 `3fd5dfe16` 零重叠 ⇒ "出自旧代码"这个猜测被证实。另我漏了发布导出根 244 件（那里 0 条相等） |
| **C76 残留**（谓词为何不生效） | **CONFIRMED-P1，真因钉死** | 见上表 infection/`:223` 的修订：`envelope_consumers.py:541/586` 重建丢 9 个标识字段。建议补一条穿透 `authoritative_writer_records` 的整合测试 |
| **PROMOTION-OVERWRITE**（`publication_bundles.py:478-491` 自盖 `rendering_only`） | **REFUTED** | 我提的"可伪造溯源声明"不成立 |
| **C10-W2 残留** `datasource.py:1236-1239` dedup 吞 9,372 条 | **REFUTED，数字撤回** | labevents/micro/services 的删除量与 join 放大量**逐位相等**，被吞真实事件 **0**；prescriptions 自带 `icustay_id`，根本进不了该分支（引用其 35,422 会是新的口径错误）。本报告"取证完成前不得开工"清单里这一项现可关闭 |

| **C41** 图件绕过 claims 层按声明水平重算 | **WRONG-DESC + WRONG-LEVEL（P1→P2）** | 被审文件不在 `reporting/`（该路径不存在），在 **`authority/descriptive_scientific_claims.py:148-185`**（行号对）。机制半边确证：`figures/` 27 文件对该函数引用数 0、无等效重算，3 个 claims 编译器只有 1 个有保护（另补出登记册漏计的第 4 条内联分支 `scientific_claims.py:416-459`）。**但真实语料 0 载体**：53 份带 CI 标签的产物标签恒为 95%、非 95% 标签 **0 份**，可按 claims 语义重算的 24 行区间 **0 失配**（执行器 `exposure_outcome_distribution_render.py:467-486` 落盘前已重算）。"99% CI" 只存在于构造探针 ⇒ 降 P2。真实缺口改述为：**6,099 行区间所在文件根本没有 `confidence_level` 列，而标签是字面量** |
| **C63 + C33** resume 冻结字节 / "digest-bound" | **登记册"共同根因"分组作废，须拆三条** | C63 现场在 `scientific_maturity.py:812-816`（`.is_file()` 冒充 digest-bound）+ `:1764` 措辞，**不是** first-write-wins。真实函数重放（`/private/tmp` 副本）：三个 sha 改全 0 后 maturity 仍判 present，而 `agent_pipeline_runs.py:2582-2587` 当场抛错 ⇒ "从不比对"作为**全称** REFUTED。两根 93 份 receipt、活体 85 份摘要 **100% 匹配、0 失配、0 全 0/畸形**，多代 run **0**。⇒ **建议 C46 与 C33 撤回、C63 独立留 P2**，唯一代码修复项是把 `scientific_maturity.py:812-816` 的存在性判据改为读 receipt 并复核绑定字节摘要 |
| **C61** `scores/` 批次失败静默截断 | **WRONG-DESC（容器）+ 机制 STILL-CONFIRMED，维持 P2** | 点名 `memory_manager.subprocess_batch_load:962-1115`，截断在 `:1074-1078`；构造 40 患者 4 批崩 2 批 ⇒ 静默返回 20 行、无异常无失败字段。"正式重提取不经此函数"**成立但理由错了**：真正原因是 `extraction.py:2491` 把内层 `batch_size` 设成本批大小，不是"有自有记账环" |

### P0-4 `modules` 里的实验性队列定义既不对话端受检、也不在发射端受检（Wave 4 新增，主审亲自复现）

- 坐标：`src/easyicu/webserver/research_launch_scientific.py:405-428`（发射门）与 `:477-478 _configured_modules` → `:694`、`:785/:787`（真正的执行选择器）；`src/easyicu/concept/selection_policy.py:51-80`（`_POLICIES` 只有 `sep3_sofa2` 一个键）。
- 机制（两跳缺一不可，均已实测）：
  1. 发射门只吃 **executable primary exposure**：`:408 if not primary_exposure: return`。真实 store 行 `study_c3610aa6fab6d35c` 的 `execution_concepts.primary_exposure` 为 `None` ⇒ 校验器**静默 PASS**，而它的 `modules` 里就装着 `sepsis3_sofa2`、`confirmations={}`、question 不含任何 sofa 字样。
  2. `modules` **不是展示字段**：`_configured_modules` 经 `ScientificConfiguration.inspect(study).modules()` 取值，并被用来在 `:785/:787` 抛 `research_pipeline_target_outside_configured_modules` / `..._outcome_outside_...`，即它真实约束哪个概念被计算；把 `sepsis3_sofa2` 换成 `sepsis3_sofa1`，筛出的概念集合随之改变（280 概念目录 → 该模块）。
- **修法陷阱（照字面修会修出一个空门）**：`_POLICIES` 的键是 **concept id `sep3_sofa2`**，而 module 拼写是 **`sepsis3_sofa2`**。主审实测：
  ```
  policy table keys: ['sep3_sofa2']
    lookup 'sep3_sofa2'    -> allowed=False mode=explicit_only reason=concept_explicit_selection_required
    lookup 'sepsis3_sofa2' -> allowed=True  mode=ordinary      reason=concept_selection_ordinary
  ```
  ⇒ 若只按"发射端也查 `modules`"字面实现，每个 module 都会命中 `concept_selection_ordinary` 全放行，门形同虚设。**正确修法**是先把 module 名归一化成 concept id 再查策略表，并且策略表要覆盖到 module 维度。
- 与 P0-1 的关系：P0-1 是"对话端可被自授权写进配置"，本条是"**即便没有被自授权，配置里的 `modules` 通道也全程无人校验**"。两者独立，须分别修。
- 触发频率：真实 store 1,094 个 context 中该形态（`modules` 含 sofa2 且无 confirmation 键）**1 条**；未观测到由此产生的真实 run。定 P0 的依据是"队列定义可被换掉且两道门都不看它"，不是已发生。

### Wave 4 追加的主审自纠（T4 抓出，两处我已亲验）

1. **`w3/C42.md` 与本表 P0-1a 里我写的 "`primary_exposure=None`" 不准**：该行的**顶层 `primary_exposure` 字段是字符串 `"Sepsis-3"`**，`None` 出现在 `execution_concepts.primary_exposure`（也就是喂给校验器的那个实参）。我把两个容器混成了一个。（已亲验）
2. **"该行的写入者是 `POST /api/study-contexts`"是我的推断，不是证据**：store 里没有任何写入者字段，该归属**不可核验**。Wave 3 我把它当事实写进了回执。（已亲验：store 无 provenance 字段）
3. **`clinical_definition_*` 与 C42 "同族等重"不成立 → 降 P3**：自撰确认键确属同一无枚举容器，但 `pi_copilot/` 全域 `clinical_definition` **0 命中**，它只产一条软项，实测 `planning_prerequisites_missing=()` 不阻断提交与发射；该族在 1,094 个真实 context 上触发 **0 次**。
4. **`POST /api/study-contexts` 判为不可被模型间接到达**：45/45 工具、sidecar、前端全穷举无路径；且真正的写容器是**同进程直调** `upsert_context`（`study_context_update.py:1563`），该路由并不比工具多给授权。⇒ 钉为"不构成授权旁路"。
5. **`tools.py:190-206` 空 `user_text` fail-open 确认不可达**（我 Wave 3 的猜测成立）：`ToolExecutionContext(` 生产构造点仅 1 处，且前置 `pi_message_required`；但那层复核在恒真空转。
6. **C51 不进 `ConceptAuditSeal`**（`extra="forbid"` + 两处 `findings=state.usage_findings`）→ 维持 **P2**，"是否回 P1"这个开放项关闭。



### Wave 4 追加的主审自纠（T3 抓出，我已逐处复核）

1. **`CONFIRMED.md:105`（C46 行）** 我写"权威门 `readiness.py:697-830` 确有逐字节复核" —— 该区间 `sha256|hexdigest|read_bytes` **命中 0**（那段是 `_publication_figure_bundle_ready`）。真实比对在 **`readiness.py:383`、`:391`、`:419`**（`sha256_of_file`）。⇒ 结论（存在逐字节复核门）不塌，塌的是我引的坐标；我用一个错坐标给一条降级理由"背书"，这比没引更糟。
2. **C33 行** 我引 `research_evidence_preview.py:455-497` 为"只自校验摘要" —— 该区间摘要比对命中 **0**（实为 figure-id/唯一性抛错），真实 sha 在 **`:798`**（`registered_sha = record["sha256"]`）。
3. **"13 份留存日志"** → 实测 **20 份**（0 例批次失败的结论不变）。
4. **`CONFIRMED.md:144`** "C33/C46 共同根因 = C63" 不成立，须拆成三条独立登记。

仍在跑：T4（`clinical_definition_*` 族量化、`modules`-only 在发射端无人校验的后果面、`POST /api/study-contexts` 能否被模型间接到达、`tools.py:204-205` 空文本 fail-open、C51 是否回 P1）。




### Wave 3 定档（取证件制，2026-09-21）

> **本节两处已被上面的 Wave 4 表改判**：①C76 的归因——Wave 3 我写的"否证投影丢 `axis`"是查错容器（拿磁盘 JSON 否证内存对象），真因确在 `envelope_consumers.py:541/586` 的内存重建；②`WRITER_PANEL` 不是"恒假字段"，降为 P2（合并行为有 gate 测试钉死）。

Wave 3 不再产出"结论的转述"：每条必须留五段式取证件（CODE 逐字行 / REPRO 脚本+完整 stdout / POPULATION 枚举过的根+provenance 标签 / GUARD 找过的全部保护层含未找到处 / VERDICT）。**下表是当前唯一有效的定档**；下方 P1 分族正文里 Wave 1/2 的行号与级别若与本表冲突，以本表为准（正文的完整逐条重写待这批改判落地后统一做）。

| 类别 | 条数 | 条目 |
|---|---|---|
| **P0 维持并经实物取证** | **3**（Wave 4 再加 1 条 → **共 4**，见 P0-4） | C42（跨轮种植在真实 owner+真实 store 复现；11/16 反例放行）、C43（四环链条逐环实物；纯疑问句亦铸权）、C59（三态塌陷 + 最新导出 32.99% 阴性为缺证据） |
| **P1 经取证维持** | **24** | C1、C5、C14、C15、C20、C25、C26（人群半边）、C29、C35、C40、C44、C52、C53、C54、C64、C65、C68、C69、C71、C72、C75、C76、C77、CGUARD |
| **Wave 3 由 P1 下调** | **19** | C2、C3、C16、C17、C18、C19、C28、C39、C46、C55、C56、C57、C58、C62、C66、C70、C73、C74、CGUARD 的 `:2557` 半边（→P2/P3；理由见各 `w3/*.md`。C50/C60/C78/C79 另列"否证/撤下"，不重复计入） |
| **否证 / 移出修复序列** | 4 | C50（未知键硬抛 + 评审门覆盖三角色）、C60（原理成立但语料分母为 0 → P3）、C78（NOT-ATTESTED）、C79（NEWS `o2sat` 与本仓声明的参考实现 R ricu `news_score` 全域 81 点 0 分歧） |
| **P1 维持但表述须改写** | 5 | C1（"四出口"→**三处可达 + 一处死码**：③`:1137` 在出货字典面不可达）、C75（WAVE2.md:71 "根因不同"判 WRONG-DESC，出局点就是 C1 的 `:1105`；真实在体测量 `patient_ids={'row_id':458}` → weight 表 **52,570 行 / 43,256 stay = 全库 61,532 的 70.3%**）、CGUARD（`:2391` 返回全表 CONFIRMED；`:2557` 只放大 IO、`:1613-1615` 把数据收回 0 行 ⇒ 原"并列同罪"over-claim）、C76（后果成立、**我的归因被否证**，见下）、C44（数值三项经独立复算吻合，但 `bin_level` 坐标与对外效力收窄） |


覆盖以 manifest 与真实文件集的双向差集为准（`real - covered = 0`）；各批自报 `files_read` 合计 989，与批次表解析口径有关，非漏读。

## P0（**4 条**，必须修；第 4 条由 Wave 4 新增）

### P0-1 实验性临床定义可被自授权（患者队列的科学定义被换掉）

- 坐标：`src/easyicu/concept/selection_policy.py:121`；`src/easyicu/webserver/pi_copilot/study_context_update.py:157,167,541-571,760-802`；`src/easyicu/webserver/pi_copilot/tool_catalog.json:34`；消费端 `src/easyicu/webserver/research_launch_scientific.py:419-427`
- 机制（两半缺一不可）：
  1. **裸子串匹配**：`explicitly_named = any(term.casefold() in text for term in policy.explicit_terms)`，其中 `sofa 2` / `sofa2` 会撞上 `SOFA 28天死亡率`、`sofa 28-day mortality`、`SOFA 2.5 分界`。
  2. **无来源绑定**：`confirmations` 是 `easyicu_update_study_context` 的 **model 可写参数**（catalog `"host": []`），全仓**无键枚举、无来源位**，`_merge_nested_study_patch` 递归合并模型提交的 mapping，且在 sepsis 判断之前就已合并（`:760`）。
- 主审亲验：8 条普通提问中 **6 条 ALLOWED**，含 `"sofa 28-day mortality in sepsis"`、`"评估乳酸清除率与 SOFA 2.5 分界的关系"`、`"SOFA 28天死亡率与脓毒症预后的关联"`；`tests/core/test_concept_selection_policy.py` 只 pin 了泛化脓毒症/正例/否定例，**未覆盖数字碰撞**。
- 端到端复验：真实 `execute_tool` + 真实 store 读回，两轮种植成功（R1 只写 `concept_selection_sep3_sofa2_authorized=True`，R2 只提 `modules:["sepsis3_sofa2"]`），`execution_concepts.primary_exposure` 同路径亦通，负对照正确拒绝且不落盘；launch 端放行。10 条候选保护层中 **9 条不覆盖**，唯一残余是 human plan review，而它拦不住被伪造的那一项 explicit-only 临床同意。
- 关键形状：**同轮自授权被拒、跨轮种植成功**——防护只做在轮内，而 `study_context_update.py:1387-1389` 的注释断言强于代码实际保证。`projections.py:285` 还把全量 confirmation 键名回显给模型。
- 最小修复：给 `confirmations` 加闭合键枚举 + 来源位（同仓已有正确范式：`study_contexts.py:1318-1322,1808-1841` 的 `cohort_eligibility_authority`），`explicit_terms` 改词边界匹配，并补"跨轮种植必拒"的 pin 测试。
- 盲区：未真跑 run；`study_scientific_configuration.py:445-476` 的 `clinical_definition_*` 属同缺陷族，未量化。

### P0-1a Wave 3 取证与自纠（`w3/C42.md`）

- **种植链在实时 HEAD 上端到端复现**：真实 owner 函数 `study_context_update.update_study_context` + 真实 `study_contexts` store（重定向到 `/tmp`，未写 `~/.easyicu`），同句中性的 `user_message`（"帮我把这个研究配置改成只做院内死亡终点…"）——R1 只写 `confirmations:{concept_selection_sep3_sofa2_authorized: true}` 即落盘（`status: ok`），R2 再提 `modules:["sepsis3_sofa2"]` 即被放行（`stored modules = ['demographics','sepsis3_sofa2']`）；R5 同轮两键一起提交仍被挡（防护只做在轮内）；R3 任意键名同样被接受（`made_up_authority_key: true` 落盘）。前置条件仅为该轮带非 privileged 的 `configure` grant（`full` 模式每条消息都发）。
- **碰撞半扩样后更强**：16 条刻意 SOFA-2-negative 的表述里 **11 条 ALLOWED**，含 `"SOFA 2016 定义下的脓毒症诊断准确性"`（语义正好相反）与 `"评估乳酸清除率与 SOFA 2.5 分界的关系"`。真实 store（1,094 个 study context）里已 **2 条**用户自撰 question 命中碰撞：`"…胆红素和 SOFA-2 肝脏分项与院内死亡的关系…"`（`sofa-2` 指肝脏子分项，不是实验表型）。旧稿"8 条提问 6 条 ALLOWED"作废，改用带逐命中项的 16 条样本。
- **新增一条更硬的事实**：发射端 `_validate_primary_concept_selection(study, primary_exposure)`（`research_launch_scientific.py:405-428`）**只校验 primary_exposure，不校验 `modules`** ⇒ 只把实验变体放进 `modules` 的配置在 launch 端无人校验；真实 store 里就有这样一条（`study_c3610aa6fab6d35c`：`modules` 含 `sepsis3_sofa2`、`confirmations={}`、question 不含 SOFA-2、`primary_exposure=None`）。该行的写入者是 `POST /api/study-contexts`（`routes/study_contexts.py:100-112`，body 直进 `upsert_context` 且无 policy 校验）——那是**人驱动面**，不算模型自授权证据，但证明 `modules` 侧确有第二入口且发射端不查。
- **自我否证两处**：①"新建 context 因 `if current` 而绕门"是我自己的推断，探针返回 `study_context_not_found` ⇒ **否证**，copilot 工具不创建 context；②"execution_concepts.primary_exposure 同路径亦通"**有条件**——实测被 `study_execution_source_required` 挡住，需先绑定已验证数据源。
- 唯一策略覆盖：`_POLICIES` 实测**只有 `sep3_sofa2` 一条**；`tests/core/test_concept_selection_policy.py` 全文 4 个 test，无一条数字碰撞样本，也无一条"跨轮种植必拒"。

### P0-2 模型自撰文本可铸 privileged one-shot 授权

- 坐标：`src/easyicu/webserver/pi_copilot/turn_authority.py:179-207`（`_DIRECT_EXTRACTION_AUTHORIZATION_MARKERS` 直取分支）；`service.py:2084-2091`；`node_app/src/main.mjs:713`
- 机制：`infer_explicit_turn_actions` 的 docstring 明文断言 **"The model cannot supply this text"**，而产品自己的 prompt 要求模型逐字撰写 `选择方向 1：` 前缀的**可点击选项文本**"so the host can formalize it"。该直取短语分支**没有** report / provider_run 两条分支都带的疑问-否定守卫，而 `PiMessageRequest` 无来源标记 → 人点击模型撰写的选项与真人输入在服务端不可区分。
- **Wave 3 取证（`w3/C43.md`）**：链条四环全部取到实物 —— ①`main.mjs:713`/`:588` 命令模型自撰 `选择方向 N：` 与 `- ` 选项；②`static/js/screens-guided-pi-next-actions.js:11-13`（`NEXT_STEP`/`CHOICE` 正则）+ `:110`（`if (!subject) return { label: choice, message: choice }`）+ `:373`（`data-gpi-next-choice="${esc(action.message)}"`）把**模型 bullet 逐字**变成按钮出栈文本，`events.js:414/422` 原样 `sendText(message)`；③`message_input.py:53` 不含本地路径的消息逐字成为 `provider_message`；④`service.py:2084/2089-2092` 铸权。前端用真实 `project()`/`render()` 在 node 垫片下跑通（`probes/c43_frontend_roundtrip.mjs`），后端用真实函数复现（`probes/c43_backend_grant.py`）。第二层 defense-in-depth（`tools.py:190-206`）调用**同一谓词、同一文本**，不构成额外保护。
- **必须区分"结构可达"与"历史已发生"**：137 份真实转写（`~/.easyicu/pi-agent/sessions/*.jsonl`，索引 100 份在库）全量重算得 433 条用户轮、**165 次 privileged 铸权**（151 provider_run / 10 extract / 4 report_revision），但**click 形状 0 次、用户轮含该直取短语 0 次、模型 bullet 含该短语 0 次**；模型自撰该短语确有 2 次（都在解释这道门的散文里，非选项）。⇒ 维持 P0 的依据是"纯疑问句即铸真实提取权 + 模型自选 `database`"（`tool_catalog.json` 里 `easyicu_start_extraction` 为 `host_mutating: true`、`model: [database, source_mode]`），**不是**"已经发生过"。
- **撤回**：Wave 2 由代理上报、我未复算即转述的"18 mint grants / 151 host button rows"。165 与 151 的差异说明对方口径串位；本轮数字以 `probes/c43_population.py` + `_stdout.txt` 为准。
- 主审亲验（`.venv/bin/python`）：
  - `"请解释什么是一次性 extraction 授权？"` → `{'extract'}`
  - `"选择方向 1：请解释一次性 extraction 授权"` → `{'extract'}`
  - `"选择方向1：what is a one-time extraction authorization"` → `{'extract'}`
  - `"选择方向 2：乳酸轨迹与新发 AKI"`（正常选项）→ 空集
  即直取短语仅 2 条字面量（`一次性 extraction 授权` / `one-time extraction authorization`），**纯疑问句即可铸权**。
- 对照：cohort 侧的注册源确认走 HMAC 绑定，grant 侧没有（同批 verify_AL 逐条核过）。
- 最小修复：privilege grant 需要来源位或 HMAC 回执；至少把 `_CONFIRMATION_MARKERS` 的疑问/否定守卫补到直取分支，并把 `provider_text` 的来源（真人 typed vs 模型撰写回送）纳入判定。

### P0-3 已发表产物里"没做培养"与"培养阴性"不可分（Wave 2 升档，主审亲自复核）

- 坐标（**Wave 3 勘误：文件归属先前写错**）：真实归属是 **`src/easyicu/scores/microbiology.py`** —— `:101-113` hadm 级 any-row 判定、`:116` `merge(how="left")`、**`:117-118 fillna(False)`**、`:162-163 astype("boolean")`（在 fillna **之后**，第三态被主动擦除）；eicu 分支 `:142-157` 同构。原稿写的 `concept/callbacks.py:101-113/:116/:117-118/:163` 是错的：该文件该区间是 `_callback_int`/`_callback_bool`，全文件仅 `:6705` 一处提到 `culture_positive`（在一条 ValueError 文案里）。行号巧合接近、文件不同，属主审引用错误。
- **升级依据是本报告自己的条款**：登记册原第 4 条"条件性升级"写明 C59"若发现真实产物命中即升 P0"。Wave 2 找到了命中件。
- 主审独立复核（我自己跑的读数，非转述）：产物 `/Volumes/外置硬盘/easyicu_data/full6_20260717/miiv/sepsis_shared.parquet`（1.9 MB，568,101 行）含 7 列 `[stay_id, charttime, susp_inf, infection_icd, samp, bld_culture_positive, culture_positive]`；按 `stay_id` 聚合后 **94,458 stay，True 31,985 / False 62,473**，其中 **18,401 个（19.4806%）是零培养记录被写成 False**。全量源核对：`microbiologyevents` 3,988,224 行，`hadm_id` 缺失 = 0，故该分母是精确值而非近似。
- loader **没有第三态**：输出仅 `[stay_id, culture_positive, bld_culture_positive]`，"缺证据"与"证据阴性"共用同一个 `False`，无 `partial`/`NA`/`covered` 列。
- 该列是**被推荐的发表结局端点**（`callbacks.py:6700-6708` 注释明文"阳性分层请用 culture_positive 端点"），并被密封与审计契约点名：`scripts/releases/EX-A01_seal_full6_release.py:164-169`、`scripts/figures/QC-A02_*.py:79-88`（`classification=admission_level_static_flag`）——但两处**只披露计时/窗口口径，不披露缺证据混同**。
- **双重不可恢复**：同一产物内采集佐证列 `samp` 的 568,101 行**无一为真**（唯一值 `0.0`/`NaN`，`>0` 行数实测 0），而修 `samp` 的提交 `af21256ab "fix(sepsis): harmonize specimen sampling events"` 日期是 **2026-08-03**，晚于 07-17 的产物。故在现行已发表物料里，"做过培养/未做培养"与"阴性/缺证据"两层都无法反推。当前 HEAD 重新导出会带真 `samp`，但 `culture_positive` 的语义混同仍未修、也无伴随披露列。
- Wave 1 的 `verify_AK` 结论"本地留存导出中尚无 `culture_positive` 实体产物"**被推翻**（其只查了 `feature_definitions` 单一口径）。
- 最小修复：`culture_positive` 拆分或缺省第三态（`NA` = 无培养记录），并新增 `culture_source_complete`/`cultures_obtained` 披露列；已发表物料需附勘误或重导。

## P1（45 条，按根因族）

### 族 1 临床假阴性与假安心（真实数据已量化，处置首位）

| # | 坐标 | 一句话 | 分母/证据 |
|---|---|---|---|
| ~~C59~~ | `scores/microbiology.py`（批次 077） | **"培养未做"被报成"培养阴性"** | **Wave 2 升 P0、Wave 3 已量化**：13 份导出全普查，11 份带该列；最新一份（`full6_native_v6_dev9_2e0c0441_20260915`，90,088 stay）发表面阴性 59,885 例中 **19,757 例（32.99%，占全队列 21.93%）根本没有任何送检行**；阳性率随口径可差 9.4pp。另 `infection_icd` 在 11/11 份导出里整列 NaN。详见 `w3/C59.md` |
| C40 | `scores/kdigo_aki.py:1722`、`scores/aki_profiles.py:466`、`callbacks.py:7754` | `rrt_source_complete=True` 硬编码：同一数据仅改标志位即由 `partial/aki=NA` 升为 `negative_complete/aki=False` | 配合子概念吞异常 `concept/__init__.py:2131-2134` + 空帧不抛错 `callbacks.py:7782-7789`；verify_W VERIFIED、verify_AK 补齐 3 个赋值点。**若发现真实产物走过该路径即升 P0** |
| C60 | SICdb `ICUOffset` 空值补 0 | 死亡时钟原点偏移，探针复现伪造"存活"假阴性 | 现语料 27,386 行 **0 空值** → 未触发 |
| C62 | SIC 仅取 `ICD10Main` | 属"不披露窗口"而非"漏用数据" | 实测源仅此一列；0.33% 缺码报 0 分 |
| — | `mews_score` RR≤9 / temp≤35 边界；`news`/`mews` 缺 SpO2 补 100、缺意识补 A | 边界与缺测填补使评分虚增/假安心 | 真实语料 0.61% 小时行 +2 分；≥2.85% 行受影响（verify_AK，P2→P1） |

### 族 2 患者身份展开 fail-open（读整表）

- **C1** `concept/__init__.py:1105,1108,1137,1177` `_expand_patient_ids` 四个出口未物化目标 ID 键即返回 → 调用方省略患者过滤、退化成整表读。函数自身注释（`:1187-1189`）写明不变量，`:1110-1114` 记载该失效**已发生过**（首批 5,000 stay 的 partial cache → 第二批转换得空值 → 读入近似全库）。
- 同族剩余出口（verify_AQ 穷举）：族 A = C1 四出口 + **A5–A12 共 8 个新出口**；族 B = `load_concepts` 5 处；族 C = `visualization` 6 处（`visualization/utils.py:187` 的 owner 谓词反而有 `subject_id`，名册缺项是 Q2 根因，匿名导出"只删列不删行"再标 Selected patient）；族 D = `prepared_frames` 3 处（`webserver/dataio.py:3310-3330` 是反着做的正确模板）；族 E = 缓存键折叠。
- **Wave 2 更正（重要，原先这三条修法/约束都写错了）**：
  1. **不要一律物化空键**。`datasource.py:2391` 与 `:2557` 的 `if value_list:` 在空列表时**整条 WHERE 不追加**（实测读全文件），`:1061-1063` 在身份列缺失时静默 `pass` —— 所以把 `[]` 物化进字典在这两条路径上**不阻止扩大读取**。正解是**分出口区别对待**：②`if not source_values` 该物化空键；①④（跨库不可转换 / 映射表加载异常）必须 **raise**；③eICU 提前返回是**设计**（`icustays` 对 eICU 不存在，见 `dataio.py:48-55`），不算缺陷。
  2. **"C1 必须与 `compute_patient_ids_hash` 折叠同批修"——撤回**。折叠等式本身为真（`H({'stay_id':[100],'subject_id':[101]}) == H({'stay_id':[100,101]})`），但**因果不成立**：`_expand_patient_ids` 在 `:1082` 复制字典、只写副本，结果绑定到 `expanded_patient_ids`，而 5 个缓存点全部吃**入参** `patient_ids`；全仓 15 处构造点一律单键，`api/extraction.py:2112,2728` 更以 `len(...) != 1 → raise` 强制单键。E1/E2 降为**独立 P2**，其真实残留是缓存键缺 `database`/`source_identity` 成分（磁盘键 `:8363-8382` 齐全且 fail-closed），与 C1 无因果。
  3. **"`webserver/dataio.py:3310-3330` 是可直接照抄的正确模板"——撤回**。同一函数的 xlsx 分支 `:3355-3358` 与被指控的 D3 是**同一行形态**（缺列即跳过过滤、整帧返回）；它只做单键 `stay_id`、无跨库身份映射，与族 A 问题域不交；把它的"缺列⇒空表"搬到 `prepared_frames` 会把 **WIDEN 反转成 EMPTY**（另一种错）。只保留三条纪律可参照。
- **引用可落行率与新增出口**：族 A–E 的 22 条坐标中仅 1 条 CITATION-INVALID（E2 的 `_raw_cache_key` 真实在 `:875-887`）、3 处 ±1~3 行漂移、1 处分母误数（称 10 个消费端，穷举为 8）。**Wave 1 漏掉了第 5 个出口 `concept/__init__.py:1050-1051`**（`patient_ids={}` → 一个过滤器都不追加）。可达面收敛：A1/A3 在当前字典面 0 命中应移出 P1，真正可达仅 7 处（labevents 挂 79 概念），且下游 `:3841/:3958` 会内存内收窄回正确队列 —— 故 **A2 是唯一"错行"出口**，A4 属可用性/静默吞异常。D2/D3 降为"缺防御而非现行缺陷"（`prepared_frames.py:150-154`、`patient_drilldown:751-752`、`cohort_review:480` 已把身份列放进 `selected`）。
- **修复次序**：先修 `datasource.py` 那三处空值/缺列 fail-open，再改 `_expand_patient_ids` 的出口语义，A10 须同批。

### 族 3 分母、单位与覆盖率口径

| # | 坐标 | 一句话 | 分母 |
|---|---|---|---|
| C44 | `research_agent/reporting/reporting_checklist.py:867-908`（`:896-897` 判定、`:835-843` 整词匹配） | 纯关键词条目在**整篇 bound manuscript** 上全文匹配、无邻近性/长度要求，且 `evidence_ids` 为空也照样判 addressed，coverage 虚高并进对外核心列 | **主审在真实产物上亲自复算（Wave 2 更正分母）**：生产 run 根 `~/.easyicu/pi-agent/workspace/projects/…` 全量 **30** 个（登记册原写 31 = 30 生产 + 1 个 `e1-binding-replay-20260908` replay 副本，属"QA 语料当生产语料"）；**26/30 = 87%** 的 run 里 `12b` 与 `22` 两条是"仅关键词命中 + `evidence_ids=[]`"仍判 addressed；coverage 虚高**均值 +0.0788、最大 +0.0909**；按阈值 `0.85/0.55/0.25`（`evaluation_scorecard.py:84-86`）重算，**18/30 会掉一个 `bin_level`**。消费端 `evaluation_scorecard.py:747-752` `subscore=float(coverage)` 原样取用、`:763` 定 bin；`plot_canonical9_scorecard.py:59-65` 第 5 个核心维度即 `reporting_completeness` 且 audit decision = "include"。**铁证**：`run_20260901T052202_d61f48` 的稿件第 71-73 行写着 "**No** sensitivity or subgroup variants were available for reporting"，即**否定句本身提供了关键词 `subgroup`** → 12b 判 addressed；另 "requires author verification" 占位语出现在 **298** 份稿件里，22（Funding）靠标题词 `funding` 判 addressed。语料侧 `subgroup` 在 414 份 md 中出现，该词在有该小节的稿件里几乎不可避免 |
| — | STROBE 模板整缺 17–21 却以 22 行伪装全量 | 与 C44 同后果，并条处置 | verify_AE U2（P2→P1） |
| ~~C4~~ | `utils/callback_utils.py:466`（向量化支）与 **`:580`（datetime 支，实测活跃）** | ~~区间两端各多写 1 个 bin（100/2h → 3×50=150），并错标 `'units/hr'`~~ **Wave 2 降 P2**：off-by-one bin 属实（`n_points = end_floor - start_floor + 1`；`same_mask` 令 end==start 变 duration=1），但**该列语义是 rate 不是剂量**，"剂量翻倍"只在假设按小时求和时成立 | **无消费端做该求和**（G15）：语义四处一致声明 rate（`concept-dict.json:6884`、`catalog.py:143`、`callback_utils.py:393-403`、交付 `column_metadata` 带 `canonical_unit: units/hr` 且 lineage 写明 `distribute_amount`，故"错标单位"一并撤回）；防线 `concept/__init__.py:7265-7266` 数值列窗口聚合**默认 median**（`ins` 不在 overrides）、`:5423` 列入 `callbacks_that_expand` 使膨胀不叠加；3,119 parquet 中仅 15 份含 `ins`（均 medications 导出）、交付研究队列 0 份。**G12 的两数也被否证**：stay 级 Σins/Σamount 中位 1.948、**仅 49.8% 落 2.0×**（非 98.7%）、pooled 1.28。**另立新条勿并入**：`concept-dict.json:6925` 的 `"callback": "aggregate_fun('sum', 'units')"`（`callback_apply.py:1644` 处理）是**真剂量求和**路径，若求和发生在展开之后另案取证 |
| C37 | `concept/callback_apply.py` `mimic_kg_rate` | 体重合并致行数放大，真正入口是 `id_cols[0]` 取到 `row_id` | **真实 MIMIC-III 300 → 432 行**，每人最多 3 个体重；可与 `:2456/:3007` 合并前塌缩一次改完，与 C10 **非同根因** |
| ~~C10~~ | `datasource.py:1117-1361` | ~~hadm→stay 补全退化为多对多，行数放大且仅 warning~~ **Wave 2 已整句替换，降 P2** | G14 全量：多对多放大**只在 `services` 成立**（12,805→28,891 行、2.256×；hadm 28755811 实测 10→50），其余 5 表被 `merge_asof` 走成行中性；须删"仅 warning、下游被跳过"（`verbose=False` 时该处零日志）。**且我漏看了一道专用后置门**：`concept/__init__.py:231-268` `_quarantine_rows_outside_icu_episode`（2026-09-18 v6 增设，注释直接点名"rolling join 使一个 stay 会吸跨年跨入院的事故"并给出 stay 31206864/31672975 的 -52,588..+52,711 h 实例），按行裁掉episode 窗外并 `logger.warning`，native-v2 发布端独立再执行同一窗口。**残留两项仍活**：`services` 的 2.256× 行放大；`:1236-1239` 对多 stay hadm 先删 `target_id_col` 再 `drop_duplicates()`，G14 称可吞掉 **9,372 条**真实事件（[AGENT]，分母未给，暂不据此开工） |
| C38 | `webserver/cohort_review.py` `unique_entity_intersection` | 交集实为 `min()` 夹取 | 探针 100%/ok vs 真值 50%/warn；真实导出 12 行未偏移，但**模块层面 94,458 行夹取确已触发**；语料 0 命中故不升 P0 |
| C3 | `io/data_converter.py:1412-1436` | warnings 计坏行在默认 `ThreadPoolExecutor(workers=4)` 下错配，真含坏行的文件被写 `clean/ready_for_analysis=True` | 探针复现（workers=8 时 8/15 错配，208≠255）；入口 `webserver/dataio.py:474` 不覆盖默认并发。门控跨文件求和 → 40 次并行 run 级 0 次变绿；真实 dump 144 文件坏行 0。原 P0 下调 |
| C12 | `io/ts_utils.py` `fill_gaps` | NaN 时间戳行的值被覆盖进首个 bin | 探针 99 毁 1.0；上游仅排序不 dropna，null-charttime 带值是明文保留类别 |
| C13 | `callbacks.py` 公共 `pafi`/`safi` | 忽略 `match_win` 与 FiO2 单位，静默产出 26667 量级假值 | docstring 示例本身即触发；仅公共导出面受影响（提取管线走 owner） |
| C74 | `concept/` 键冲突 + 静默丢概念 | 键冲突机制复现（3×3→9），但行放大半边被 charttime 归一化**证伪** | 实扫 full6 六库 30 模块 29 个键**零重复**（批次 049"本机无 prepared 数据"系误述） |
| Q4→一类修 | `endswith("id")` 同形谓词另见 `utils/callback_utils.py:412` | 名册误剔 | `uric_acid` 1/324=0.31%，本身降 P2 |

### 族 4 门控失明（结构性缺口，建议作为一类统一修）

- **C28（根因条目）** `gates/preflight.py` 40 余个检测器**全部是"某写法存在才报"的在场检测**：语句被删除在原理上不可见。探针实测修复前后 findings 完全相同；其声明的 `row_set_unchanged`/`n_unchanged` 不变量在代码改写层恒记 UNVERIFIED、失败仅 warning。含义：任何"删除安全关键语句"的修复都结构上不可审计。
- **C20** `gates/preflight.py:91` `_STRUCTURAL_ACCOUNTING_PRODUCTS`：7 个精确名匹配 agent 自撰的 table 产物名，等价改名即整项静默跳过。分母 4741 步 / 1385 脚本；13 份已执行脚本按声明名报 0 finding、仅改名即报。
- **C21** `gates/preflight.py:6315` `_single_imputation_findings`：只认 `fillna` 位置参数直写形态。语料反证——被识别的那一形态出现 **0 次**，别名/字典式 6 次全部漏掉（本组最硬一条）。
- **C51** `audits/statistical.py`：四道门在真实语料**从未被求值一次**，而 `README:917`、`pipeline.py:19` 称其为每步 error 门（verify_AG 新增 AG-1）。
- **C35** STRICT 删句可被补引绕开：把句子挂到无关队列步即令 Methods 区 removed 由 1→0。
- 并入本族（verify_AS）：062 `failure_classification` 双签（被识别签名全库 0 命中，漏掉的 `TableOneContractError` 形态在真实 run **命中 9 文件**并被误路由烧修复预算）；069 action↔method 门（417/44/0 全量独立复算逐数吻合，"0 判别"精确化为"44 步在域内无违规、**373 步结构性失明**"）。
- 修向：不变量改为**在产物上实际求值**（执行或等价性检查），UNVERIFIED 视为失败；键名改为声明式 + 缺失/为空即 error 回执（参照 `plausibility_receipt.py:369-390`）。

### 族 5 审计与对账可空转

| # | 坐标 | 一句话 | 分母 |
|---|---|---|---|
| C2 | `audits/cross_step.py:3674-3678` | `except Exception: return []` 使跨步骤溯源审计判 ok，并连带跳过 range-flag 与 percentage 两项检查 | 4 种畸形父表探针复现；全仓无 required-auditor 回执兜底；带表头空父表仍 fail-closed；本机 0 份真实产物落进该分支 |
| C29 | `audits/cross_step.py:3090` | `audit()` 只收 step/step_summary/out_dir，**既不绑定 run 目录也不绑定证据** → 把对账表指向 run 之外自写 CSV，审计产出 0 findings | 与 C2 应并成一次"父表来源绑定 + fail-closed"修复 |
| C66 | 054 `verified_tool` 生产入口 | 只采信手写 decision 块，裁证据探针复现；`decide_tool_promotion` 生产调用 0 | verify_AS；与 C5 同族，独立记账 |
| C68 | `discovery/idea_mining_priorart.py:354` | **比上报更重**：corroboration 客户端生产 0 接线、0 命中时 `"no_hits_to_screen"` 绕过 NOT-screened、失败记录被缓存固化 | 探针实测 `recommend` 可达；最小修 = 补 `exact.search_ok` 对称校验（照搬 `idea_mining_data_first_route.py:301`） |
| C69 | `discovery/concept_proposal.py:395-444` Gate 5 | 标为 "joint coverage" 实为各 itemid 边际覆盖率之和 | 探针：两个 0.006 相加即逃 error（单项则 rejected）；跨表只 probe 首项表；13 个 test 无一覆盖该语义 |
| C23 | `research_agent/literature.py` | 引用静默丢失且全仓无聚合收据（"静默"仅对未被引用的键成立） | 只有 record 级 decision 收据 + 人审投影可见；latex/reader fail-closed |
| C24 | PRISMA 与引用计数 | 数字由本文件自算并直抄种子，无独立对拍 | verify_AQ 扩写：`:116` 那道"独立对拍"是**自指**（`prior_art_result_count` 由同一模块截断后算出）；需纠两处（独立计数实有 3 个、`literature_authority.py:310` 确有一道真实比对）；给出 11 个计数点清单 + 一个跨源门 |
| C67 | `know_how/registry.py:842` | 中位卡摘要校验失败**连带丢弃其后所有卡**，必需 stop_condition/requires_confirmation 消失，而持久化回执仍记 selected | 4 卡探针丢 3、15 条强制约束消失；`withheld_card_count` 全仓 0 消费方；触发需带外改包内 JSON、`enable_know_how` 默认 False → 不升 P0 |
| C25 | `discovery/idea_mining.py` 候选分母 | 预注册分母被低估确证；丢弃可追溯 | verify_U 维持 P1；非 C20 名门控所致 |
| C26 | `discovery/idea_mining.py` 口径矛盾 | 人群/形状矛盾进入人审物料，未达成稿 | verify_U |

### 族 6 发表物料（图件/成稿/复现）

- **C41（根因条目）** 图件/表格路径**绕过 claims 层的按声明水平重算**：`_require_interval_arithmetic`（`descriptive_scientific_claims.py:148-185`）在 `figures/` 内**调用数为 0**。探针：同一数值输入下 claims 抛 ValueError，图件照画并贴 "99% CI"。该保护在 3 个 claims 编译器中只有 1 个有；图件侧已有同类 owner（`exposure_outcome_distribution_render.py:436-483`）但只覆盖一个闭合产品。**单一修复点**：run 级 `PublicationFigureSkill` 改为消费已 seal 的 envelope，并复用该重算规则 + `effect_scale` owner。
- **C47** causal `_load_effect` **跨行拼接**主估计与 CI（est 取 row0、ci_low 取 row1）——非"取首行"，verify_AD 探针复现。**本组首位。**
- **C48** causal 硬写口径：字面量全仓唯一、无对账门、无 pin 测试。
- **C49** survival 猜测式 KM 识别（坐标纠为 `:136-142`），且面板 A/C `evidence_ids=[]` 却声称 registered；同批另查出 `missingness_publication.py:810` 与宿主 `missingness_source.py:31` **共用同一 sealed id**。
- **C17/C18/C19** `figures/skill.py`：null 被画在 1.0 且 `xlim(0,1.46)` 令全部数据点出画而审计零 finding；CI 合成为 `2.10 (2.10-2.10)` 且比对源是自身副本；主估计按 row 0 标定，`core_claim` 实测输出 `"for Age"`。
- **C45** `reporting/scientific_maturity.py:1769-1799`：simulated 自评只要非 major/reject 即清掉 `INDEPENDENT_SCIENTIFIC_REVIEW` blocker 并给 clinical_review 满分（探针 35→100、score 36→43）；`review_mode`/`claim_boundary` 全仓 **0 个 src 读者**（仅 3 处测试）；`readiness.py:2299` 不传参故裸文件生效；54/54 真实 run 恒 `analysis_only` → 不上 P0。
- **C46 + C63 + C33（共同根因）** `reporting/` resume 后冻结字节：C46 图件/manuscript 维度按目录位置读、receipt 内 sha 从不比对却自称 "digest-bound"（改全 0 仍判 present）；C33 兜底**不覆盖全部消费端**（`tools/agent_experiment_package.py:281-329`、`research_evidence_preview.py:455-497` 只自校验，剥前缀后上一轮 tex/pdf 会混进人审包）；根因是 **`readiness.py:2689` / `write_phase.py:2619-2671` 的 first-write-wins 注册守卫**——verify_AM 核过 43 个分歧根文件**全部是本轮合法改写**（0 例陈旧根副本，封存侧才旧），074-2 `128/128` 图行无摘要；074-3 降 P2。
- **C73** 图件轴界只认上边界 fail-closed，内部 K+占比不足形态必抛（语料 447 步 0 例故不升）。
- **C50** `contracts/` 契约出口：三道保护逐消费端穷举后**全不覆盖**（评审门 4 处 primary-only；`plan_scope` 只绑名单；`plan_payload.py:1660-1740` 原样收 agent 的 `model_requirements`，无宿主独占）。primary 空理由同样被契约 ACCEPT，出口比断言更宽。

### 族 7 修复器自身

| # | 坐标 | 一句话 | 证据 |
|---|---|---|---|
| C56 | `repairs/availability_fraction.py` | 修复器直接改**已报告的 availability 分母**且从不验恒等式（1.0→0.75/0.25，可引入 NaN），并把脚本自写的 "not imposed" 披露覆盖为 "reconstructed exactly" | verify_AJ 维持 P1，**本族最高、近 P0**；宿主 prompt 自宣告的 positive-finite 前置从未实现；pin 测试 exec 断言 1.0→0.75/0.25 |
| C57 | `repairs/` 授权来源 | 多条修复的授权取自 **agent 自撰 step_summary 散文**而非 runner stderr：`status:"ok"`+note 即可换取删除校验（`nullable_validation.py` 删非有限校验、`figure_distribution.py` 放宽角色白名单）；`clinical_bin` 宿主闭集不存在 | verify_AJ F5 上调 P2→P1；"删除不可见"半条并入 C28 |
| C55 | `repairs/name_alias.py` | 相似度门只比"首个下划线后后缀"，把**反义键**改写：`adjusted_↔unadjusted_` **端到端可复现**（取到语义相反的值）；`excluded_ids_set→included_ids_set` 需候选名实际绑定**模块级 dict 字面量**（`:69-79` 的 `ast.Dict` 门），集合字面量不触发 ⇒ 原例 WRONG-DESC | **Wave 3 勘误**："requires_disclosure 全仓 0 消费者"**是错的**：`repair_registry.py:100/201-202/747/802` 有执行期真求值。准确表述是"在 `repair_registry.py` 之外无读者（人审物料/发布列不读它）"。别名改写半条降 **P2**（真实规模 0/52，按取证制不得支撑 P1 量化） |
| C58 | `replication/paper.py:764-793` | 句末数值（`OR 1.23.`）致未捕获 `ValueError`，CLI/pipeline 三入口裸调崩 | 句末句点是主流形态，端到端探针确证 |
| C16 | `research_agent/repairs/` 台账 | 同路径重开时把磁盘台账真写成 `{"repairs": []}`，历史 repair 记录丢失 | verify_K VERIFIED；071 并入（mock 台账 2→0 探针复现，mock 三处保护层与本场景正交） |
| — | 修复前后审计不可见 | `profile_roles.py` 注入的 `coerce_numeric_fail_closed` 确证 0 定义/0 注入（引入提交 `0c5a5dca0` 未附 helper），但 `gates/preflight.py:6402-6403` 的 `undefined_helper_call`/`unresolvable_name`（error 级）实测点名该符号且每轮执行前运行 → **响亮、不出产物，降 P2** | verify_AJ |

### 族 8 授权与溯源（P0 之外的可伪造面）

- **C5** `webserver/agent_pipeline_runs.py:236-239` + `routes/agent.py:505`：人审门缺来源绑定——模型可用 `decision=approved` 走到与真人点击**同一函数**，落库身份是常量 `easyicu_local_web_operator`，论文侧 reviewer-auth 由该常量直接满足。校验链本身完整（坐标/decision/opt-in/ai_enabled/pending/study 匹配/`resumable_here`/canary/`plan_approval_allowed`），故**不能自造 grant**、不能批准不存在的暂停计划；会话级 replay 留有 `host_action` vs `tool_start(easyicu_resume)` 通道差异 → 维持 P1。
- **C14（扩写）** `pi_copilot/service.py` `idea_source`：前端可控值自称宿主 receipt、ingest 丢弃字节致事后无可比对台账。verify_AQ 补齐真正落点：`discovery/idea_mining.py:2045-2100 _source_record` 的 origin/label/**file_sha256 全部由请求体提供**、无闭合枚举、不受连接器门控 → 一句话应改为"**标注与摘要均可自拟**"。
- **C15** `pi_copilot/tools.py`：一次性授权在校验之前被 consume（无效入参烧掉用户额度并误报 already consumed），**共 13 处**（原发现 4 处），全仓无退款路径。并修清单见 verify_AL。
- **C65** 081 注册源确认：新形态 `"先别使用本地完整数据"` → True，且会**重绑 `record.binding`**；授不出新源。
- **C39** `execution/candidate_loop.py` 等 **7 处**（原报 4 处）吞 `ProviderHardStopError`：全局终止不保证、归因由 `execution_raised`+止损改写为 `repair_failed`，终止标签落到下一个不相干步骤。坐标需纠 2 处（`coordination.py` 实在 `repairs/`；重抛点是 `step_candidate_recovery.py:256-260`）。不上 P0：pre-transport 付费上限与耐久台账 `budget_exhausted` 仍生效。**对照** C36（`phase_support.py` replanner 同类指控因标签未被改写而降 P2）。
- **C27** `webserver/patient_drilldown/` 时间轴：时间值被再除 60（上游已归一为小时，`datasource.py:3494-3501`、`concept/__init__.py:6000-6009`，且有 pin 测试 `tests/.../test_cross_layer_safety_regressions.py:54-89`，前端并未乘回）。与 `_as_percent` 同根因家族：**值已归一、消费端按列名或量级推断后再换算** → 建议一次修"显式尺度声明 + 禁止推断"。
- **C64** 083 中文否定正则误判致 fail-open 绕过结局守卫；**最大漏报词是「性别」**（守卫实际输入语料 0/204）。
- **C30** `api/concepts.py:529-560` `keep_cache`：在按 config 共享的 loader（`_loader_cache`）上翻一个无引用计数的布尔位、退出即拆缓存；`max_running=8` 且强制 in-process，探针定序复现"A 退出把 B 的位翻 False 并清其缓存"。后果限于重复重建（查-取在 `_cache_lock` 内、键含 patient-hash）。
- **C9** `webserver/ideas/mining.py` `_idea_title`：标题写死 "in adult ICU patients" 与 `population` 字段矛盾；权威口径是 `population`。副产新发现：标题回灌 topic 可生成"仅成人 NOT 儿童"检索式。
- **C52/C53/C54** 统计内核三条 P1：`methods/rmst.py:83-88` NaN 事件误分类（出厂两条 RMST 产物路线闭锁，Coder 路线无保护但 **0/969** 脚本调用过）；`methods/rcs_dose_response.py:706` `startswith("s")` 认样条项（同设计仅改列名 `sex→gender`，质检 p 由 5.5e-05 翻成 **0.46**）；`step_contract.py:251-399` 数值契约对字符串/布尔/**NaN/inf** 全判通过、仅显式 null 报错，经 `contract.py:744` 入早/终两道门并顺带解除 `manuscript_claims` AUROC 幻觉审计（`True`→登记 1.0）。

### 族 9 摄入完整性与响亮失败

| # | 坐标 | 一句话 | 边界 |
|---|---|---|---|
| C70 | `io/download.py:90-99,208-218,251` | 摄入层无完整性校验：SHA-256 分支在真实配置不可达（166 表零 hash、无 override）、`num_rows` 110 处声明零消费、截断文件"存在即完成"被接收 | 不升 P0：webserver 自有 demo 路径有 size+sha256 强校验、`raw_source_authority` 哈希钉住两表、CLI 断流多数响亮报错 |
| C71 | `io/src_utils.py:120-217` | `src_data_avail`/`src_tbl_avail`/`is_data_avail`/`is_tbl_avail` 四个公开 API 每次必抛 `ModuleNotFoundError`（import 写在 `try` 外且 `easyicu.io.utils` 不存在），且**比上报更坏**：修好 import 后 `get_default()` 仍不存在 | verify_AR |
| C72 | `io/export.py:60-65` `write_psv` | 文档化的 datetime index_col→hours 转换在 pandas 3.0.5 必 `TypeError`，函数自身 docstring 示例即失败 | 数值路径可用故停 P1 |
| C61 | `scores/` subprocess 批次失败静默截断 | 机制确证 | 正式重提取走 extraction 自有记账环、不经此函数；13 份留存日志 0 例失败 |
| P2→P1 | import dtype 契约整体失效；`_read_table` rglob 任意取表；foundation provenance 吞错仍 `blocked=False` | verify_AR 晋升三条 | 见 verify_AR.md |

### 族 6 发表物料 —— Wave 2 裁决（本节取代上文 C41/C45–C49 的原始表述）

**结论：这一族的"修复点"选错了，而且多数条目零实例。**

- **C41 的"单一修复点"撤回**。`figures/skill.py` 确实默认开启、`write_phase.py:752-759` 每轮调用（所以不是纯死代码），**但数值上改它是 no-op**：383 份真实契约里 **279 份（72.8%）出自 16 个宿主 runners**，skill 自渲染只有 **5 份（1.3%）**；31 份 run 级 bundle 中 **29 份是 `shutil.copy2` 字节原样晋升**。改 skill 影响不到 98.7% 的交付数值。
- **C47/C48/C49 零实例**：`easyicu_*_publication_figure` 在 49,547 个 json 中 **0 命中**，causal / survival / prediction / phenotype family 的产物一份都没有 ⇒ 三条降级为"代码缺陷真实、生产零实例"，从处置清单移出。
- **归属不能挪到宿主侧**：`_significance` / `_estimate_ci` 全仓不存在；宿主同类实现（`robustness_figure_executor.py:1030-1045`、`association_publication_figure_renderer.py:118-129`）**已是修正后的形态**，即"宿主侧有正确实现、skill 侧没有"，与 C41 原表述方向相反。
- 我原先写的"runner 629 命中 vs skill 0 命中"也不干净：实为 **580（生产）+ 49（QA replay 容器）**，step_summary 口径是 **38 份**，既不是 629 也不是 87/31。
- **C45 降 P2**（推翻我上一轮的判断）。score **确实对外**——进浏览器投影 `scientific_readiness.json`（`agent_pipeline_runs.py:3247/3279`、`screens-guided-pi-run-outcome.js:150/288`）与 Copilot 载荷（`pi_copilot/tools.py:2282-2286`），所以"自读自用"不成立；但**路由只看 blocker**（`scientific_maturity.py:1858-1863` → `completion.py:236`），53/53 份真实审计在删掉 review blocker 后仍是 `analysis_only`（novelty blocker 恒在），accept 形态出现 **0 次**。真缺陷收窄为一句话：**读了裸的 `reviewer_report.json`，却没有复用同文件 `readiness.py:2018-2037` 已有的 producer/digest 绑定**。
- 附带更正一个我交给复验代理的假前提：`tests/test_publication_figure_skill_contract.py:219` 与 `mock_execute.call_count == 0` **不存在**（该路径无此文件，真实文件是 `tests/research_agent/figures/test_publication_figures.py`；`mock_execute` 全仓 0 命中），`panel_role` 实测 **1,295 次 / 502 个文件**而非"196,037 全在 outputs/"。这两句是我自己的推断被我用"G13 称"的措辞转述，`w2_G13.md` 里 0 命中。

**本族净变化**：C45/C47/C48/C49 降 P2；C17 收窄（关联图已按 `null_value` 尺度化，只有稳健面板仍硬写 1.0）；C18/C19 维持 P1（真实 164 行 CI 0 例；仍按 row 0 标定、26/979 上下文 `primary_exposure` 为空）；C41 的修复点改为"先修宿主 runners 与 skill 的**共同 seal 消费口**，不是 skill 单点"。

## 系统性结论（Wave 2 复验后重写；原版有三处被推翻）

缺陷**集中在把关层与"交付端口径"，不在评分判据**。经 14 组对抗复验后仍成立的三条：

1. **授权与溯源可铸**（最硬，三条 P0 全在此）：临床定义可跨轮自授权（C42）、privileged grant 可从模型自撰文本铸出（C43）、缺证据可被报成完整采集且阴性（C59/P0-3）。这一类不被任何兜底挡住，因为缺的正是来源位本身。
2. **交付端口径与判据端口径不一致**：判据端普遍有门（SOFA-2 用 `uo_*_covered_h`、KDIGO 用实际 `total_h`、`bind_primary_output` 覆写 `primary_or`、`_quarantine_rows_outside_icu_episode` 裁窗外事件、窗口聚合默认 median），**但交付出去的列常常绕过这些门**——`uo_24h` 89.77% 行是稀释值而 `covered_h` 不在 renal 模块列清单里、`culture_positive` 把"没做"与"阴性"合成同一个 False、checklist coverage 把模板标题算成已回答、`writer_evidence` 的 `n_independent` 虚高（**Wave 4 定谳，并撤回 Wave 3 我写的那句"勘误"**：真因**就是**投影丢 `axis`，但丢在内存侧——`audits/envelope_consumers.py:541` 以 `rebuild_observed_scalar_tree(loaded.envelope.observed_scalars)` 重建 `step_summary`，`:586` 整体替换，行对象 24 键→15 键、`axis` 等 9 个标识字段消失 ⇒ `row.get("axis") != "primary"` 恒真。Wave 3 那次"否证"**查错了容器**：拿磁盘 JSON 去否证内存对象，磁盘上 132/132 行 `axis` 当然健在。全语料 19 个可配对 run 中该谓词**生效 0 次**。）。**共同形状：判据正确、交付物口径失真**，所以缺陷不在科学计算里而在证据物料里。
3. **门控的失效方式是"入口窄"而非"没有门"**：`gates/preflight.py` 46 个检测器里 **13 个已在做缺失/义务判定**，仓内也早有义务门模板（`plausibility_obligation`/`plausibility_receipt`）；真正瞎的是**无悬空引用的删除**（守卫、校验调用、产物写出、self-rebind 过滤），而删 import/def/赋值有 96%/94%/72% 会被 error 级 `unresolvable_name` 抓到。因此第三批的落点从"造缺失检测"改为"**复用已有义务门模板**"。

被 Wave 2 **推翻或大幅收窄**的三条原结论：`figures/` 一族多数零实例且"单一修复点"选在只占 1.3% 产出的路径上；概念/IO 层的分母与单位类缺陷（C37/C12/C13/C4/C10）经全库重算后成规模降档；"图件不接 claims 重算"虽成立，但宿主 runners 才是 72.8% 契约的来源。

第二条贯穿性根因是**"值已归一，消费端按列名或量级推断后再换算"**（C27、C4 的 `units/hr`、C13、Q1 的 unit_conversion）；第三条是**"溯源标注可自拟"**（C5/C14/C42/C43/C65/C66）——本轮确证：**没有任何可伪造的审批或签名回执**，可伪造的是**身份常量、来源标注、以及从文本推断出的 privileged grant**。

## 条件性升级（需要产物证据才能定档）

1. **C52**：一旦发现某份真实产物走过 Coder→rmst 路线 → 升 P0。
2. **073-A4**（or/hr/rr 共用 25% 相对带宽、跨尺度判 aligned）：现为 P2，因 comparator 唯一实例化点需显式 `--paper`+LLM opt-in 且本机真实复现产物 0 份；**出现任一真实 `comparison.csv` 即按 P1 处理**。
3. **065 P5**（`learning/memory.py:756` 重建 StrategyCard 漏传 confidence/validation 字段致计数归零、已退役卡静默复活）：现为 P2，因 `validate_card`/`retire_card`/`record_retrieval` 生产 0 调用方、全仓 0 张策略卡；**一旦接线立即回 P1**。
4. **C40 / C52 / C38** 均写明：若发现真实产物命中即升 P0。**C59 该条款已在 Wave 2 触发并已升 P0（见 P0-3）**。
5. 另需在 Wave 2 后重判的对称条款：`pipeline.py enable_cache`（开闸即 P1）、`065 P5`（接线即 P1）——两项的前提都是"生产 0 调用方"，而并行任务正在改 `webserver/` 与 `research_agent/`，**这类"零读者"结论必须随树重验**，不是一次性事实。

## 已下调 / 已证伪（不要重复上报）

- **`uo_6h` 稀释越过 SOFA/KDIGO 阈值**：机制真实（分母固定名义窗，探针 50% 稀释），但 `scores/sofa2.py:1296-1298` 用 `uo_*_covered_h >= window` 门控、`scores/kdigo_aki.py:812-872` 用实际 `total_h` 自行重算，`scores/urine_windows.py` docstring 明文宣告"判据证据与描述性 UO 分离"；唯一未门控的 `callbacks.py:510 _legacy_sofa2_renal` 全仓无调用方。**结论不成立**。
- **053 统计审计 fail-open 两条 → P2**：`outcome_rate` 在 4109+3772 份摘要与 6909 个计划步 `expected_outputs` 中 **0 命中**；533 份 manifest、3170 条 `stat_findings` 记录非空 **0**。primary-OR 三重字面量门命中率 **0/425**，且宿主 `output_files.py:237-277` `bind_primary_output`（经 `phase_support.py:3241`→`phase.py:3624`，**在门评估之前**）按真实表头覆写 `primary_or`，408/425 已绑定、17 份未绑定者状态全非 ok ⇒ 3993 条 ok 步中未核验 `primary_or` **0 例**。
- **055 "resume 字典漂移门恒空转" REFUTED**：`authority/runtime_artifacts.py:346-421` 会选中带指纹的 `manifest.json`，573 个真实 run 中 **525 个门确实在比对**（当场全抛 `ConceptDictDriftError`），139 次实际 resume 均有 manifest.json，端到端 pin 测试实跑通过；仅 46 个未 finalize 的 `/1` partial 走空转支 → P2。055 D2 也不升 P0：definition 为 None 仍触发 `cohort_contract` error 并卡住 `analysis_validated`，384 个未过滤轮全是显式 `all_input_rows` 且有 info 回执，该翻译路径真实产物调用 0 次。
- **050 其余五条 → P2**：B2（`force=True` 与 detach+attach 均可改指、44 行有日志、webserver 不用全局环境）；B3（传 `path=` 的全在明文废弃的 `ConceptLoader`，`setup_data` 0 调用方、无文档旅程）；B4（恒 False 已证但 `DataEnv` 全仓 0 构造方）；B5（真实产物 0/53 命中）；B6（NaN 起点→整窗暴露机制确证，但全量语料 **2.05e7 行** win_tbl 起点缺失 **0 行**）。
- **081 A → P2**（返回 `registered_source=None`，不铸绑定/授权/receipt；`file://` 确被拦）；**081 D MCP REFUTED**（`mcp_client.py:294-302` 冻结逐工具 allowlist、每消息 1 次、结果带 untrusted 标记）。
- **066 fairness → P2**：`finalize.py:155-169+572` 已把 outcome 绑主模型并要求 `outcome_type=="binary"`；多位数整型使信息阵奇异→返回 None，算不出假 OR。
- **058 三条**：①空 ledger→Full **P2**（`completion.py:254-259` 链不含 scorecard，`scoring_inputs.py:1517` 硬拒空 ledger + `scoring.py:353-401` 覆写两层独立保护）；②去重先于过滤、顺序敏感 **维持 P1**（触发面比原文窄：derivation 状态不同即不撞键）；③`six_database` 硬编码 **P2**（属 C9 同类，非来源不可区分类）。
- **078 Q1 unit_conversion → P1 但不改变真实提取值**：26 个 substance 键中 18 个被通用键顶替（lactate 0.111 应 0.222，本模块自述示例即错），`can_convert` 还为错系数背书；仓内 0 处带 substance 调用、批导走 `callback_utils`+DuckDB 内联另一套实现。**Q3 CLI DEFAULT_GROUPS → P2**（全量差集恰 2/98，真实导出走 `CONCEPT_GROUPS_INTERNAL` 19 模块 324 概念，COMPAT 拷贝 0 调用方）。
- **079 Q5 封装机密汇总恒判 blocked → P1 但改定性**：只对 value 角色成立（event_status 暴露可放行）；同一次调用里 `execution_readiness:25` 已读过概念帧，"不扫全模块"的理由不覆盖 ≤3 个执行概念 → 属**投稿旅程阻塞**非授权缺陷；另确证"旧注册行走回退可放行、新注册恒 blocked"双态。
- **抽样假象教训（两条）**：批次 048 的"9/25 零行真实产物"是 `limit=25` 按 glob 排序截断的取样产物，全量复算为 14/59 parquet（23.7%）且全部集中在 demo study 的 `feature/outcome_concepts=[]` 算术结果、0 个轨迹消费步骤，唯一行级读者对空表硬失败 → P2；批次 036 的"planning 时间学判据永假"REFUTED → P2。**恒真/恒假判据的语料分母必须全量数，不接受抽样。**
- 其他：`audits/figures.py:5121` 主图标 auxiliary 使 ≥2 面板审计失效 **REFUTED/P3**；`execution/phase.py:1595` probe_summary 无条件重置 → P2（handoff 字段全仓 0 读取方）；`concept/callbacks.py:3255` sofa=0 → P2；`:5172`/`:6461` 属明文宣告的刻意对齐 R/ricu → P2；`api/extraction.py:6039` → P2（5940 行 manifest 拒绝守卫使其仅 flat/崩溃目录可达）；`pipeline.py:4830-4855` → P2（src/CLI/MCP/spec 无任何 `enable_cache=True`，开闸即升 P1）；`webserver/dataio.py:3748/3753` REFUTED 原口径；`gates/preflight.py:373` **REFUTED→P3**（同函数"Load 必须在 test 内"约束使被过滤临时量无法逃逸）；`:294`/`:578` → P2；`webserver/pi_copilot` 一批断言因 `app.py` loopback/Host/跨源写中间件 + `host_security` + `provider_url_security` 使鉴权/SSRF/路径遍历后果不可达。

## 正面项（复核确认有效，勿"顺手加固"）

- `authority/`、`contracts/` 核心的 fail-closed 设计：逐字节摘要重验、copy+rollback 台账写、`Literal` 锁死 `claim_ceiling`、`allow_mock=False`、`mock_exempt` 被 `_paper_eligible` 硬拒。
- `bind_primary_output`（`execution/output_files.py:237-277`）在门评估之前按真实表头覆写 `primary_or` —— 这是本轮多个"字面量门过窄"指控被证伪的真正原因。
- 完成态门 `_has_external_preregistration` 恒 False 与 completion 合取，使 `article_grade`/paper authority 在当前代码下不可达；54/54 真实 run 均 `analysis_only`、`paper_authorized=False`。
- webserver 中间件栈（loopback + AllowedHosts + proxy-headers + 跨源写拦截）经 AST 全量枚举 **143 条路由**确认全局作用域、无未覆盖前缀、0 重复、0 websocket。
- 持久化兜底：`agent_pipeline_runs` 的双路（checkpoint payload + 摘要绑定重算，resume 再判）；531 个真实 wrapper 中翻 True 态 0 例。
- `dataio.py:3310-3330` 的患者身份处理是本仓已有正确模板，可直接照抄修 C1 族。

## 建议修复顺序（Wave 2 重排；原七批版已被复验推翻部分排序）

**排序依据已改**：只把 `[产物计数]` / `[全库重算]` 级证据排在前面；`[探针推导]` / `[构造样本]` 级的量化一律不给首批资格（本轮 16 条 P1 就是这样掉下去的）。

1. **P0 三条**：C42（`confirmations` 闭合键枚举 + 来源位 + `explicit_terms` 词边界 + 跨轮种植必拒 pin 测试；顺带补 `primary_exposure`/`outcome` 的文本门缺失——那半条比子串碰撞更重）、C43（grant 来源位/HMAC + 疑问否定守卫）、**C59/P0-3**（`culture_positive` 拆第三态 + `cultures_obtained` 披露列；**已发表物料需勘误或重导**，且 `samp` 恒假使"是否采过培养"在旧物料里不可反推）。
2. **交付端口径三条（均产物级证据）**：**038 / C76**（`writer_evidence` 的 `n_independent` 虚高：19/19 份 digest 无一按 `axis` 排除，其中 11 份由**已含该谓词**的代码于 9/09–9/16 生成 ⇒ 非陈旧产物；同时 `:1684-1686` 两个标签共用一个列表，真实 run 上"5 应为 3"，+2）、**C44**（30 份 checklist 中 26 份的 `12b`/`22` 靠关键词转绿、18/30 掉 bin、对外五列之一）、**uo_24h 交付侧**（把 `covered_h`/`assessment_rate` 纳入 renal 模块列清单与导出 schema，否则通用统计与人审入口无从门控）。
3. **患者身份 fail-open**：先修 `datasource.py:2391`、`:2557`（空 `value_list` 整条 WHERE 不追加 → 读全文件）与 `:1061-1063`（身份列缺失静默 `pass`），**再**改 `_expand_patient_ids` 的出口语义（②物化空键、①④raise、③eICU 属设计、A10 同批），并把 `row_id` 当患者过滤被静默忽略那处（整表 52,570 行）一并收。**注意**：一律物化空键不是解法。
4. **RRT/AKI 完整性旗标**：C40 三处赋值点 + `rrt_evidence_reason` 进发布列 + 旗标不得自证 `coverage=complete`（`concept/__init__.py:2131` 是保护不是共因，勿当共因修）。
5. **门控一类修（落点已换）**：复用仓内义务门模板，覆盖面是"无悬空引用的删除"那五类；配 `statistical.py` 的取键改为宿主 canonical（C51 属契约漂移，按漂移修）。
6. **审计父表绑定**：`cross_step.audit()` 加 `run_dir` 并走 `parent_artifact`/`verified_run_evidence_path`，同批收 C2（异常不得返回空 findings）。
7. **发表物料**：先修宿主 runners 与 skill 的**共同 seal 消费口**（改 skill 只影响 1.3% 产出），再收 C18/C19、C46/C63/C33（first-write-wins 注册守卫）、C50。
8. **修复器**：C56（改已报告 availability 且不验恒等式、覆盖披露）、C57（授权取自 agent 自撰散文）、C55（反义键改写）、C58（句末数值崩溃）。
9. **残余**：C14 扩写项、C15 的 13 处 consume、C65/C66/C64、摄入完整性（C70/C71/C72）、`services` 行放大与 `aggregate_fun('sum','units')` 两条待取证项。

**取证完成前不得开工的条目**：C10-W2 的 `:1236-1239` dedup 吞 9,372 条（[AGENT]，无分母）、C51 是否回 P1（取决于 `stat_findings` 是否进 `ConceptAuditSeal`）、C44 的"模板标题 vs 实质正文"细分（机制与 87% 已确证，但"空壳"判据还可更严）。

## 主审自纠（Wave 2 期间的六次误述，逐条留痕）

本报告第一版的量化数字有相当一部分**不可复现**，且我在转述复验结果时六次把"自己的推断"说成"某代理的结论"。两类错误都记在这里，供后续引用者打折。

**误述（均为我的推断/转述，非代理所言，已 grep 证伪）**
1. 我把一批"培养被默认挡死、MEWS 实为 0.0206%、NEWS 是 2026-08-06 明文设计"的说法当作 G3 的产出转述——这些内容在任何 Wave 2 报告中 **0 命中**，且 G3 实际结论相反。
2. 我指控某复验代理"编造证据"，引用它"声称 `kdigo_aki.py:1722` 已改为 `_resolve_rrt_source_completeness`、`datasource.py:2167-2186` 新增守卫"——这两个符号**全 workspace 0 命中**，且从不出现在该代理报告里（`w2_G4.md`/`w2_G4R.md` grep 0）；该代理的真实结论恰是"本组零 STALE"。
3. 我说 G1 否认我的 6/8 探针——`w2_G1.md:58` 原文是"主审那 6/8 ALLOWED 探针定性不变"。
4. 我说 C10 的 HEAD 行为是"取第一个 stay / 已改为精确校验并抛错"——`:1195-1240` 既无 first-stay 也无 raise。
5–6. 我给 G5b 的提示词里以"G13 称"转述了 `tests/test_publication_figure_skill_contract.py:219` 断言 `mock_execute.call_count == 0`（**该文件不存在**，真实是 `tests/research_agent/figures/test_publication_figures.py`；`mock_execute` 全仓 0 命中）与"`panel_role` 出现 196,037 次"（实测 **1,295 次 / 502 文件**）。

**量化错误（第一版 → 复算值）**
- C59 "本机无实体产物" → 产物在，18,401/94,458 stay 已发表为 False，升 P0。
- C44 "31 份中 27 份（87%）" → **30 份生产 run 中 26 份（87%）**；+0.079 → **+0.0788（最大 +0.0909）**；"18/31" → **18/30**。原 31 混入了 1 份 replay 副本。
- C37 "真实 MIMIC-III 300→432 行、每人最多 3 个体重" → 全库放大 **+0.29%**（1 个双体重 stay），且被同链吸收；那句删。
- C38 "94,458 行夹取已触发" → 94,458 是**单个模块行**夹取前 nunique；真实现是 157 行中 8 行夹取、**7 行真值 0% 报 100%/ok**。
- C8 "58/531 终态 wrapper" → 另有口径给 5/53，我自己数到 `~/.easyicu` 下 547 个同名文件；**三口径未统一，数字先撤**。
- C4 "98.7% stay 剂量 2.0×"（G12 给） → 仅 **49.8%** 落 2.0×、pooled 1.28，且无消费端做该求和。
- "runner 629 命中 vs skill 0" → **580 生产 + 49 replay**；step_summary 口径 38 份。
- 066 fairness "94,458 可达" → 系 agent cohort 工件，非注册 drilldown 源。
- 基线数字：脏度记 65/41，全程实测从 71 漂到 **73**；路由数 143 → **145**；`reporting/reporting_checklist.py` 缺 `research_agent/` 前缀；C4 坐标 `:466`（向量化支）→ 活跃的是 **`:580`**。

**由此固化的两条纪律**（已写入复验契约，也适用于以后引用本报告）：
① 任何"某兜底不存在 / 某形态语料 0 次 / 本机无产物"的**否定式全称**，必须列出搜索过的根目录与形态集合，否则改写成"在 X 范围内未找到 Y"；
② 每个数字标 provenance：`[产物计数]` / `[全库重算]` / `[探针推导]` / `[构造样本]`，后两类**不具备 P1 量化资格**。本报告第一版的教训是：16 条 P1 里绝大多数掉档，正是因为我用后两类证据支撑了前两类才该有的结论强度。

## Wave 3 新增自纠（2026-09-21，取证件制下暴露的问题）

1. **文件级引用错误**：P0-3 的机制我写在 `concept/callbacks.py`，真实归属 `scores/microbiology.py`。这类错误 `grep <列名>` 一步即可发现，Wave 1/2 都没做。
2. **转述即复用**：P0-2 的"18 次铸权 / 151 按钮行"来自代理、我未复算就写进结论；全量重算为 165 次（151 provider_run / 10 extract / 4 report_revision），且 **click 形状 0 次**。以后凡引用他方数字，必须我自己有脚本。
3. **"结构可达"与"历史已发生"被混写**：P0-2 在 137 份真实转写里 0 次触发。维持 P0 的理由是"纯疑问句即铸真实提取权 + 模型可自选 `database`"，不是"已发生"。
4. **把自己的推断当结论写**：本轮"新建 context 绕过 explicit-only 门"是我自己的假设，探针返回 `study_context_not_found` ⇒ 当场否证；同一条款下 C76 的"envelope 投影丢 `axis`"归因也被独立复算否证（落盘行 `axis` 健在）。
5. **代理的两处过度断言我抓出来后不再转述**：`deterministic_publication_bundle_promotion`"全库 0 命中"实为**仓库源码 1 命中**（`publication_bundles.py:491`）、产物侧 0 命中；C65 的"8/8 否定句误判"我实测为 **4/6**。两处均已改为可复现口径。
6. **计数须排除打包残品**：`build/lib/easyicu/**` 有 988 个 .py，任何"全仓 N 处"若不点名是否含 `build/lib`，数字就不可比（上一轮"969"疑即此因）。
7. **本报告的实时坐标**：初稿 HEAD `2835cbbd1`，现 `e5e7e4059`；本报告自身已被并发任务的提交带入 git 历史。

## 验证状态与未决事项

- 本报告是**工程审阅证据**，`focused` 级；不授予任何研究启动、正式实验或论文权限，也不把 `analysis_only` 升格为发表权威。
- 未做：full CI、真跑 run、`tests/`+`tools/`+前端全量审阅。
- 探针残留待用户处置：`~/.easyicu/idea_mining_runs/idea_20260920_185037_589480000_1d187f30` 及 `~/.easyicu/webserver_idea_mining_runs.json` 中可能的悬空条目（该目录有运行中的锁文件，未经确认不删）。
- 明细台账与全部探针输出：`/tmp/easyicu_review/`（`CONFIRMED.md` 为逐条索引，`verify_*.md` 为 39 份复验全文，`finding_*.md` 为 83 份批次原文）。**Wave 3 取证件在 `/tmp/easyicu_review/w3/*.md`（49 份，五段式：CODE/REPRO/POPULATION/GUARD/VERDICT），认定矩阵在 `WAVE3.md`，探针脚本与其完整 stdout 在 `probes/`。** 本文档是该台账的整理版；三者若有冲突，以当前代码与取证件为准。
