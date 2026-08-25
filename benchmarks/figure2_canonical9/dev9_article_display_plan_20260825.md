# Dev9 article-level display plan

Status: development planning input only (`analysis_only`; paper authorization 0/9).

## Comparator calibration

The accessible full-text pack contains 14 published comparators. One JAMA page
could not be counted reliably from the local extractor and is excluded from the
numeric summary. Across the other 13 papers, the main article contains 2–10
figures (median 3) and 1–4 tables (median 2). Eleven of the 14 comparators expose
supplementary material; complex phenotyping papers can contain dozens of
supplementary figures and tables. Therefore one composite is one figure, not a
complete article package.

Planning target, not a rigid gate: most Dev9 questions should reserve 2–4 main
figures, 2–3 main tables, and a question-specific supplement. Counts may be
smaller for a fail-closed feasibility report or larger for validated prediction
and phenotyping studies.

## Per-question package

| Task | Main figures | Main tables | Supplementary figures/tables | Missingness placement |
|---|---|---|---|---|
| E1 Sepsis-3 | Fig 1 cohort/Sepsis-3 denominator and absolute prevalence/mortality; Fig 2 adjusted estimates; Fig 3 definition/denominator sensitivity | Table 1 cohort; Table 2 prevalence and mortality; Table 3 adjusted/sensitivity estimates | Variable definitions, infection/SOFA timing, component availability, full models and centre/database checks | Supplementary by default; promote only if ascertainment changes the headline denominator |
| E2 Lactate | Fig 1 exposure distribution and absolute mortality; Fig 2 continuous/non-linear dose-response; Fig 3 adjusted and sensitivity estimates | Table 1 cohort; Table 2 lactate windows and outcomes; Table 3 model estimates | Detailed measurement timing, missing-data handling, alternative windows and specifications | Conditional main: measurement-by-indication may be main only when it materially changes interpretation; otherwise supplementary |
| E3 KDIGO | Fig 1 mortality and ICU length of stay across ordered stages; Fig 2 adjusted mortality association and scientific sensitivity | Table 1 cohort; Table 2 stage-specific mortality/LOS; Table 3 adjusted and sensitivity estimates | Component availability, creatinine/urine-output definitions, trend-test details and missingness | Supplementary; routine missingness is not a primary E3 result |
| M1 Hepatobiliary | Fig 1 measured versus unmeasured/source-state outcomes; Fig 2 adjusted estimate and sensitivity; Fig 3 measurement-process structure | Table 1 cohort; Table 2 bilirubin/source states and outcomes; Table 3 models | Full missingness table, timing, component definitions and alternate source-state rules | Main eligible because whether absence means normal or unmeasured is the scientific question |
| M2 Prediction | Fig 1 calibration, discrimination and repeated-split uncertainty; Fig 2 external/temporal validation when available | Table 1 cohort; Table 2 model performance with uncertainty; Table 3 calibration/utility | Decision curve unless thresholds are authorized, preprocessing, leakage, feature availability, full coefficients/hyperparameters | Supplementary; feature missingness is validity evidence, not the headline model result |
| M3 Static phenotype | Fig 1 structure/separability; Fig 2 clinical profiles; Fig 3 stability/replication; Fig 4 downstream outcomes only after stable clusters | Table 1 cohort; Table 2 phenotype characteristics; Table 3 stability/replication | Feature provenance/scaling/missingness, alternate K/algorithms, assignment details and full outcomes | Supplementary unless missingness itself determines that clustering is infeasible |
| H1 Ventilation survival | Fig 1 survival/absolute risk with risk table; Fig 2 adjusted time-varying contrast; Fig 3 assumptions/censoring sensitivity when material | Table 1 cohort; Table 2 risk sets/events; Table 3 effect and sensitivity estimates | Detailed censoring, follow-up, missingness and proportional-hazards diagnostics | Supplementary unless informative censoring changes the estimand or conclusion |
| H2 Vasopressor causal | No effect figure until exposure strategies, time zero, positivity and a real comparison group pass; a protocol/feasibility diagram may be shown | Target-trial protocol table and feasibility/positivity table only | Full eligibility, strategy, balance/positivity and failure receipts | No cosmetic missingness panel can substitute for a valid comparison group; remain fail closed |
| H3 Dynamic trajectory | Fig 1 trajectory structure only after K is authorized; Fig 2 profiles; Fig 3 stability/alternative algorithms; Fig 4 external replication | Table 1 cohort; Table 2 trajectory phenotype profiles; Table 3 replication | Availability by time, imputation, alternate windows/distances/K and assignments | Supplementary in a successful article; current availability/BIC figure remains a development diagnostic because no cluster result is authorized |

## Placement rule

Routine missingness, feature availability and measurement audits remain required
evidence, but are supplementary by default. They can enter a main figure only
when missingness/measurement is the research question, changes the estimand or
denominator, or materially changes the interpretation of the primary result.
They must never displace the primary outcome, effect, calibration, survival, or
stability evidence merely to fill a four-panel grid.

## Evidence boundary

This plan borrows article structure and reporting expectations only. It does not
copy published numerical results, infer unobserved analyses, or convert Dev9
development outputs into paper-ready evidence.
