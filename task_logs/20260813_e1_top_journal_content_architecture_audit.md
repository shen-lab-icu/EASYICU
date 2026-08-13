# E1 顶级期刊内容质量与 Agent 架构审阅（2026-08-13）

## 结论

当前 E1 **未达到顶级期刊投稿水准，也不允许进入分析审批**。审阅对象是现有真实 Web E1 运行：

- Study：`study_d7b6b3174469e59c`
- Web run：`run_8720f3d4a5cc`
- Research Agent run：`run_20260813T011213_84b0fa`
- 只读产物：`/Users/haibo/easyicu/projects/study_d7b6b3174469e59c/run_8720f3d4a5cc/pipeline/run_20260813T011213_84b0fa`
- 代码基线：`fix/pi-workspace-review-20260809@45deba325f58` 加当前未提交的共享工作树

旧运行持久化的审阅是 58/100。使用当前代码从原 `run_input_capsule.json`、原 cohort、原 Plan、原 LiteratureBundle 和原 figure strategy **离线重建 ResearchContext 后重审**，得到 **51/100，`changes_required`，`approval_allowed=false`，`top_journal_candidate=false`**。旧证据没有被覆盖；51/100 是新版 evaluator 对同一旧 Plan 的只读 adjudication。

## 多维评分

| 维度 | 分数 | 裁决 |
|---|---:|---|
| 文献检索 | 70 | 有真实 PubMed 检索，但没有建立 direct comparator；候选集合混有儿科、治疗试验和泛综述。 |
| 创新性 | 70 | 没有 direct-comparator 对照，不能区分真正创新与“换数据库复做”。 |
| 文献→Plan | 40 | Plan 将文献绑定到其 method card 不支持的设计元素，并漏掉适用的方法学层。 |
| ICU 临床设计 | 0 | Sepsis-3 时间锚已对齐，但 24 h 后基线暴露、重复 ICU stay、临床独立审阅和数据库算法一致性均未闭合。 |
| 统计设计 | 40 | 年龄/性别只是候选而非用户确认的精确调整集；年龄线性假设未执行检查。 |
| 稳健性 | 15 | 用户要求的 timing/readmission sensitivity 仍是协议文字；可执行的独立稳健性轴不足。 |
| 图件 | 70 | 仅表示 Plan 中图件角色覆盖，缺 data-quality 角色；**不表示图片视觉合格**。 |
| 内容完整度 | 100 | 仅表示 Plan 的文章角色存在；**不表示结果丰富或稿件达到投稿质量**。 |

总分是八维均值；维度分数有 blocker/major penalty 下限，不能当作临床量表。

## 真实 E1 证据与主要缺口

### 1. 文献并未可靠对应研究路线

旧 LiteratureBundle 有 16 条记录，但没有一条通过 direct-comparator screening。Plan 主要依赖 Sepsis-3 定义、STROBE、RECORD、immortal-time、landmark、splines 和 missing-data 方法文献；这些可以分别支持定义或方法，不能代替同人群/暴露/结局的直接既往研究。

外部 PubMed 定点核验发现旧检索漏掉了至少以下高相关候选：

- Raith et al., JAMA 2017，184,875 名疑似感染成人 ICU 患者，直接评估 SOFA≥2 对院内死亡的预后准确性：PMID 28114553。
- Fullerton et al. 2017，成人 ICU 数据中比较新 sepsis/SOFA 定义的患病比例与院内死亡：PMID 28215126。
- 2020 ICU cohort 对 `SIRS≥2 / SOFA≥2 / SOFA_change≥2` 的频率和院内死亡关联进行比较：PMID 32898161。
- 2026 多数据库 SOFA-1/SOFA-2 Sepsis-3 诊断、时机和死亡结局比较：PMID 42092945。

这些是必须审阅的候选，并不自动等于 exact comparator 或创新性成立。

原 Plan 在当前 exact-binding validator 下直接失败：

```text
05_primary_landmark_adjusted_association:
  anderson_landmark_1983 = unsupported outcome
  strobe_2007 = unsupported adjustment
08_robustness_replay:
  anderson_landmark_1983 = unsupported robustness
  suissa_immortal_time_2008 = unsupported robustness
```

### 2. ICU 临床语义进入了上下文，但权威边界此前不完整

离线重建后，主暴露 `sep3_sofa1_max` 的 typed clinical definition 为：

- contract/version：`sepsis3_2016` / `2016`
- source：`PMID:26903338`
- clinical-definition anchor：`suspected_infection_onset`
- physical observation window：`icu_admission[0,24]h`
- observation-window role：`outer_observation_window`
- MIMIC-IV conformance：`mapping_only`
- clinical review：`automated_golden; independent_clinical_review_pending`

这关闭了“把 ICU admission 的导出窗口误当作 Sepsis-3 定义 time zero”的语义错误，但没有把映射级验证冒充临床算法验证。当前 E1 因而继续产生临床独立审阅和数据库 conformance major finding。

### 3. 现有 E1 没有可审阅的实际图片或稿件

Web wrapper 的真实状态为：

- `figure_gallery.json`: `status=no_figures`, `embedded_count=0`
- `manuscript_draft.json`: `status=locked_pending_human_review`, `claims=[]`, `markdown_preview=""`
- `scientific_readiness.json`: `status=blocked`, `publication_ready=false`, `paper_authorized=false`

因此不能回答“现有图是否达到顶刊水准”；正确结论是 **N/A，尚无 rendered figure**。Plan-level 70 分只说明缺少 data-quality figure 角色。实际执行后仍需 source-data lineage、SVG/PNG、轴/单位/不确定性、绝对风险、调整标签和视觉 QA 才能放行。

## 本轮修复的是 Agent 架构，不是 E1 答案

1. **多维计划前审阅 owner**
   - 新增 digest-bound `PlanScientificReview`，在人工批准前从 Literature、Plan、ResearchContext、FigureStrategy 评估八个维度。
   - finding 分流为 `agent_plan_revision`、`study_authority_change`、`external_evidence`、`independent_review`；只有第一类可交给新的 Planner 自动修 Plan。
   - blocker 不再显示“批准继续”。

2. **文献检索与 direct-comparator recall**
   - 固定五个检索层：broad ICU、全年 direct observational comparator、近五年 comparator、review/guideline、critical-care database。
   - 结果按层 round-robin 保留，避免 broad query 吞掉候选预算。
   - 修复自然语言推断路径：PubMed 使用 owner concept identity（如 `Sepsis-3` / `mortality`），不再把 UI 展示名 `Sepsis-3 (SOFA-1 based)` 当精确短语。
   - 全年层与近期层分开：既不漏 foundational cohort，也不放弃当前文献。

3. **文献→Plan 的 exact design binding**
   - 每个 scientific step 必须绑定 exact citation key 和明确 design element/application。
   - host 将 binding 与 frozen method card 逐项比对；一篇文献不能因“被引用”而自动支持 timing、adjustment、dependence、robustness 等所有方法。
   - primary Plan 必须绑定经过 screening 的 direct comparator；topic paper 不能冒充 method authority。

4. **ICU 临床定义合同**
   - `ClinicalDefinitionReference` 现在携带 definition/version/source、definition time anchor、validation status、ascertainment limitations 和 per-database conformance。
   - physical observation window 与 phenotype time zero 分开进入 Agent outbound context。
   - `mapping_only != algorithm_golden` 和 `automated_golden != independent clinician review` 均产生独立审阅 finding，Planner 无权自动抹掉。

5. **统计/稳健性与权限边界**
   - 候选 covariates 不再静默成为精确调整集；精确 roster、临床理由和 baseline timing 要由 StudyContext authority 确认。
   - post-baseline exposure、readmission dependence、required sensitivity、连续变量 functional form 和 robustness axes 分别检查，协议文字不能冒充可执行分析。

6. **图件与内容质量不再靠数量冒充**
   - FigureStrategy 要求 descriptive absolute-risk/prevalence、primary estimand、robustness、data quality 四种读者角色；缺角色即 major。
   - data-quality figure 必须消费真实 missingness/measurement table，禁止用说明文字或假数组补图。
   - post-run maturity 才检查 rendered visual、source-data digest、publication export、labels 和 figure/manuscript claim lineage。
   - plan-level `figures` 与 `content_completeness` 明确标注 assessment scope，Web 不再把二者显示成已完成图片/稿件质量。

7. **创新性与稿件权威**
   - 新颖性使用六个预先规定维度，与 direct comparator 逐项比较；Agent 只能生成 unsigned packet，必须由外部 reviewer owner 签署。
   - Writer 复用 run-bound literature，精确 citation audit；论文 PDF/LaTeX、Nature writing/figure receipt 和 draft watermark 进入 post-run gate。

## 仍需真人/用户/数据解决的 E1 科学问题

架构已经不会替用户编造这些答案，因此它们仍然正确阻断：

1. 24 h 暴露分类究竟采用哪一种可执行 temporal estimand；
2. 是否补 patient identity，并采用 first-stay 或 clustered/mixed dependence route；
3. timing 和 non-readmission sensitivity 的确切可执行定义；
4. 年龄/性别是否为最终调整集，以及各自 baseline availability 和临床理由；
5. `sepsis3_2016@miiv` 的独立 ICU 临床审阅与 algorithm-level conformance；
6. fresh search 后对 direct comparator 和六维 novelty 的独立判断。

这些不能由“自动 Plan 修复”代签；否则会从修 Agent 架构退化成给 E1 写答案。

## 验证

- 相关 Python/合同/Web 投影测试：`265 passed, 3 warnings in 29.19s`
- 新增自然语言 PubMed identity 回归及检索分层：`4 passed`
- clinical definition/outbound/time anchor 焦点：`9 passed`
- Ruff（相关 Python 文件）：通过
- `git diff --check`（相关文件）：通过
- 真实旧 Plan exact literature-binding replay：按上述 4 个 unsupported binding fail closed
- 没有运行 full CI；遵守 E1/Web 开发迭代只跑焦点/邻接测试的顶层规则
- 没有启动 Provider、没有修改 Canonical9 文本/shared prompt/frozen paper rubric、没有手写 E1 研究答案

