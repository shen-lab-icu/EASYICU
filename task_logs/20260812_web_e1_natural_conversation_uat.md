# Web E1 natural-conversation UAT — 2026-08-12

## Scope

- Treat E1 as one ordinary user research project and Pi conversation.
- Do not expose Canonical9 or any nine-question navigation in the product UI.
- Exercise the same `research_agent_pipeline` used by code-driven runs.
- Verify result interpretation and governed artifacts through the chat surface.

## Ordinary user question

> 在 MIMIC-IV 中，Sepsis-3 的患病率是多少？它与院内死亡有什么关联？请给出透明、可复现的队列定义和明确分母。

Project: `Sepsis-3 患病率与院内死亡`  
StudyContext: `study_1b6646faf82d4cef`  
Run: `run_20260812T161404_c164f9`

## Verified real-run outcome

- Pipeline: 10/10 planned execution steps succeeded.
- Analysis unit/denominator: 94,458 ICU stays.
- Sepsis-3 positive: 31,596 / 94,458 = 33.45% (95% CI 33.15–33.75%).
- Hospital death: 13.63% in positive stays vs 8.21% in negative stays.
- Unadjusted OR: 1.764 (95% CI 1.689–1.841).
- Gate remained truthful: `analysis_only`, `reportable=false`, `draft_unlocked=false`.

## Product findings closed

1. Guided/Copilot product sources contain no Canonical9 or nine-question entry. A negative regression locks this boundary.
2. Result interpretation now receives a bounded, aggregate-only projection of validated result tables; Pi does not recalculate values.
3. Governed resources returned by tools are rendered directly beneath the corresponding assistant answer, not only inside the folded activity trace.
4. `result_tables.json`, `figure_gallery.json`, and `manuscript_draft.json` were each clicked in the real browser and opened in the right governed preview.
5. The project rail now projects scientific lifecycle (`结果待审阅`, `计划待审阅`, `分析中`, `研究已配置`) instead of showing the metadata storage implementation as `仅配置`.

## Browser evidence

- Screenshot: `task_logs/screenshots/20260812_web_e1_natural_conversation_artifacts.png`
- URL used: `http://127.0.0.1:8765/?ui=20260812-natural-chat7#guided`
- Page contained no `Canonical9` or `九题独立对话` text.
- The result-table preview displayed the real 94,458 denominator, exposure distribution, group mortality rates, and OR evidence tables.

## Focused verification

```text
10 passed
node --check screens-guided-pi.js
node --check screens-guided-projects.js
git diff --check
```

The focused set covered asset wiring, benchmark-navigation absence, governed resources beside answers, frontend parse, Guided wiring, project rail ownership, metadata-only storage preservation, internal-evaluation hiding, and lifecycle projection.

No full exact-head CI and no formal Canonical9 provider batch were started during this E1 development iteration.

## Remaining boundary

- This run's manuscript PDF receipt was not created because the historical run predates the corrected LaTeX figure-path handling. The immutable run was not post-hoc modified. A fresh ordinary conversation/run is required to prove the PDF document entry end to end.
- Scientific interpretation and manuscript stages remain human-review-required; this UAT does not claim publication or paper authorization.
