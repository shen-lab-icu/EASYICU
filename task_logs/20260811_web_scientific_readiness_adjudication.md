# Web 科学就绪度定点裁决与 fail-closed 修复

- 日期：2026-08-11
- 分支：`fix/pi-workspace-review-20260809`
- 实现提交：`3982609`
- 范围：Web/Pi 产品演示、真实 Research Agent Web 投影、Idea Mining prior-art receipt、文献→Plan 映射
- 明确不在范围：Canonical9 题面、shared prompt、paper rubric、正式 Provider batch

## 结论

历史 E1 运行能证明工程链路可执行，但不能证明该 Idea 可靠、文献检索充分、数据提取科学充分、分析达到投稿要求或稿件可发表。旧演示把这些层次混在一起，尤其把 9 条人工种子描述成已完成检索、把 `manuscript_ready` 显示得近似 publication-ready。当前实现保留真实工程结果，同时让五个科学域独立 fail-closed。

| 域 | 历史证据 | 裁决 | Web/架构处理 |
|---|---|---|---|
| Idea | 历史运行没有 live prior-art search；当前已有 2025–2026 年同主题 SOFA-2/MIMIC-IV 工作 | 只能作工程验证案例，不能作新颖投稿 Idea | 新增 `IDEA_PRIOR_ART_AUTHORITY_NOT_ESTABLISHED`；演示改为 4/8，不再显示 7/8 |
| 文献 | `search_conducted=false`、`curated_seed_count=9`、`sources_enabled=[]`、无 PRISMA | 相关基础文献存在，但时效性、同题研究与逐步骤适配未闭合 | 新增检索时间戳、year range、科学步骤引用映射和日期门禁；current audit 与 historical run 分开显示 |
| 数据 | 准备子集 140 stays / 15 deaths，文件 provenance 存在；`cohort_definition=null` | 准备后字段完整，不等于完整来源人群、纳排路径和代表性闭合 | 新增 `COHORT_SOURCE_SCOPE_NOT_EXPLICIT`，数据卡明确“140 complete rows ≠ 科学充分提取” |
| 分析 | 11 步中 6 个分析/审计、5 个 render-only；完整病例复跑与主分析相同；reviewer=`major_revision` | 可执行，但不全面、不足以支撑投稿 | 新增 reviewer/display-suite findings；计划卡显示 15 events、信息充分性和敏感性分析缺口 |
| 稿件 | evidence-bound draft 已生成；`publication_ready=false`、`paper_authorized=false` | 不是可投稿稿件 | `manuscript_ready` 改名为 “evidence-bound draft generated”；新增 publication/paper authority checks |

## 当前文献复核

本轮只把当前文献用于独立产品裁决，没有追溯性写回历史 Agent run：

- SOFA-2 development/validation，JAMA 2025：<https://jamanetwork.com/journals/jama/fullarticle/2840822>
- SOFA-2 vs SOFA-1 in MIMIC-IV，2026，PMID 41877184：<https://pubmed.ncbi.nlm.nih.gov/41877184/>
- SOFA-2 in pneumonia-associated sepsis，Critical Care 2026：<https://doi.org/10.1186/s13054-026-06027-4>
- TARMOS missing-data framework，2021：<https://pubmed.ncbi.nlm.nih.gov/33539930/>
- Surviving Sepsis Campaign，2021：<https://pubmed.ncbi.nlm.nih.gov/34605781/>
- ICMJE manuscript preparation：<https://www.icmje.org/recommendations/browse/manuscript-preparation/preparing-for-submission.html>

这些证据表明：140-stay 的实验性 SOFA-2–死亡关联不适合作为新颖论文 Idea；它可以继续作为工程链路验证案例。

## 架构修复

### 1. Typed scientific-readiness owner

新增 `src/easyicu/webserver/scientific_readiness_projection.py`，只读取 owner artifacts，不重复计算科学结果，也不提升 authority。固定投影五个域：Idea、Literature、Data、Analysis、Manuscript；每条 finding 带稳定 code、severity、evidence refs 和 remediation。

真实 Web pipeline 新增 `scientific_readiness.json`，并加入白名单 public review payload。`source_run_manifest.json` 记录科学就绪度，但 Research Agent 的原始分析/论文 authority 不被覆盖。

### 2. 文献时效性与逐步骤适配

- Idea Mining 的 PubMed prior-art receipt 新增 timezone-aware `searched_at`。
- 该时间戳与 receipt digest 一起进入 accepted handoff，并在 Agent 读取时再次比对；旧的无时间戳检索不能绑定为当前 prior-art authority。
- Web literature projection 升为 `easyicu.web-literature-evidence/2`，公开 `citation_year_range`。
- 只要求 `primary` / `secondary` / `sensitivity`（以及缺失 role 的旧计划）绑定 exact citation key；明确的 `auxiliary`/render step 不误报科学文献缺口。

### 3. Demo 真值与可审阅性

- 删除虚构的 identified/deduplicated 检索数字和“检索在 Plan 前完成”表述。
- 历史 9 条人工种子与独立 2026-08-11 current audit 分栏；独立文献不会被伪装成历史 Agent 输入。
- Idea、数据、Plan、结果、科学闸门和稿件均在对话内标出 authority ceiling。
- 右侧 preview 可逐项打开 15 条文献、数据质量表、11-step Plan、结果表、Agent 图件、结果解读、科学就绪度和 locked manuscript。

## 验证

### 聚焦测试

按 E1/Web 开发策略只运行直接相关与必要邻接测试，未启动 full exact-head CI：

```text
109 passed, 1 warning in 10.89s
```

覆盖：

- `tests/test_webserver_idea_sources.py`
- `tests/test_idea_prior_art_receipt.py`
- `tests/test_scientific_readiness_projection.py`
- `tests/test_pi_copilot_static.py`
- `tests/test_pi_copilot_research_workflow.py`
- `tests/research_agent/test_method_literature_pack.py`

另通过：`ruff check`、Python compile、JS `node --check`、`git diff --check`。

### 浏览器逐项 QA

桌面 URL：`http://127.0.0.1:8765/?ui=20260811-science-readiness2#guided`

逐项打开并截图：

- `output/playwright/scientific-readiness-demo/01-scientific-readiness.png`
- `output/playwright/scientific-readiness-demo/02-literature-audit.png`
- `output/playwright/scientific-readiness-demo/03-data-quality.png`
- `output/playwright/scientific-readiness-demo/04-analysis-plan.png`
- `output/playwright/scientific-readiness-demo/05-manuscript-gate.png`
- `output/playwright/scientific-readiness-demo/06-idea-adjudication.png`
- `output/playwright/scientific-readiness-demo/07-result-tables.png`
- `output/playwright/scientific-readiness-demo/08-figure-gallery.png`
- `output/playwright/scientific-readiness-demo/09-result-interpretation.png`

JAMA 当前来源链接从右侧审计卡实际打开到新标签。桌面三栏无横向溢出或裁切；console 仅有浏览器 autocomplete 提示和 EasyICU hydration info，无运行错误。

## 当前边界与下一步

- 这次修复的是“系统不再把不合格案例显示成合格”，不是把历史 140-stay canary 变成论文。
- 当前演示正确停在 4/8；历史结果继续是 `analysis_only`。
- 真正投稿路线应重新运行 live Idea Mining + current prior-art adjudication，选择有明确差异化的问题，再用完整来源人群合同重建 extraction/Plan/analysis。
- 未运行 Canonical9 正式 Provider batch；未改变 frozen benchmark authority。
