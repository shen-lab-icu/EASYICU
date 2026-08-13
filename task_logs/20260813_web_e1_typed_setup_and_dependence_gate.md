# Web E1 typed setup and repeated-stay dependence gate

## Scope

- Task: `WEB-E1-TOP-JOURNAL-GATE`
- Branch/head observed before this increment: `fix/pi-workspace-review-20260809@45deba3`
- This is a development/UAT increment in the shared dirty worktree, not a release checkpoint or formal Canonical9 result.
- No Canonical9 question text, frozen paper rubric, benchmark-specific UI, or case literal was added to a shared prompt.
- No formal Provider batch or full exact-head CI matrix was started.

## Problem exposed by the natural Web E1 conversation

The ordinary Pi conversation could previously reach plan/run readiness with two scientific setup gaps:

1. a label-only time window could silently become a 24-hour execution window; and
2. an ICU-stay analysis retaining readmissions could use ordinary model-based variance without addressing within-patient dependence.

The selected full6 MIMIC-IV export contains an owner-issued `icu_readmission` concept, so a first-stay restriction is executable. Its published schemas do not expose a verified patient grouping coordinate, so patient-clustered inference cannot currently be authorized.

## Architecture repair

### Typed setup and execution alignment

- Web workflow setup now requires an exact executable exposure/outcome binding, typed `analysis_design`, and a numeric time-window duration plus anchor.
- `StudySetupReceipt` projects the path-free typed scientific authority used downstream.
- The real pipeline rejects exposure-outcome launches without typed analysis design and rejects label-only or unanchored execution windows instead of guessing 24 hours.
- The Pi system prompt asks for one highest-impact independently answerable scientific decision at a time.

### Repeated-stay dependence owner

- `easyicu.webserver.study_contexts.analysis_dependence_finding` is the single owner of the cohort/inference compatibility rule.
- If ICU stays are the analysis unit, readmissions are explicitly retained, and the design does not specify patient-clustered variance, StudyContext persistence fails closed with `study_repeated_stay_dependence_unaddressed`.
- The workflow projects this missing decision, and the real pipeline independently rejects a bypass with `research_pipeline_repeated_stay_dependence_unaddressed`.
- A user-authorized first-stay restriction requires the owner concept `icu_readmission`; absence of that concept fails closed with `research_pipeline_readmission_indicator_unavailable`.
- The Pi tool boundary now has a negative regression proving an attempted repeat-stay + model-based update is rejected without changing the persisted revision.

## Natural Web/API UAT evidence

Project/session under development UAT:

- project: `e1-natural-uat2-20260813`
- Pi session: `pi_0d1577b6f8b3a7a070fb`
- StudyContext: `study_c1b02f6f465680e8`

After inspecting the bound workflow, data package, and source capability, Pi produced one bounded decision rather than silently changing the study:

> 当前已验证数据包支持 94,458 个 ICU stays，但未证明可用患者级分组坐标；因此在保留重复 stays 时，不能安全实施患者聚类推断。唯一可执行的修订选择是：改为每位患者仅保留首次 ICU stay（排除 ICU 再入院），并继续以 ICU stay 为分析单位；是否采用这一修订？

This is the correct fail-closed state. The system has not answered that question on the user's behalf and E1 is not accepted yet.

## Focused verification

- `tests/test_pi_copilot_research_workflow.py` + `tests/test_webserver_study_contexts.py`: **107 passed**.
- focused Pi gateway/static prompt and workflow checks: **7 passed**.
- Pi tool-boundary repeated-stay negative regression: **1 passed**.
- Ruff and scoped `git diff --check`: passed.

The full suite was intentionally not run: current policy requires focused tests during E1/Web iteration and reserves full exact-head CI for the 11/11 stable checkpoint, merge, release, or formal experiment preparation.

## Remaining blocker and next transition

E1 cannot continue until the user makes one scientific choice:

- approve restricting the analysis to each patient's first eligible ICU stay, which the current source can execute through `icu_readmission`; or
- retain repeated ICU stays, which requires a new export with a verified patient grouping coordinate and a cluster-capable deterministic executor.

Only after E1 completes ordinary-conversation engineering acceptance should the next structurally different Canonical9 item be opened as a separate normal Web conversation. The intended next generalization probe is E3 (KDIGO stage gradient), with no dedicated nine-question UI and no E1-specific shared prompt repair.
