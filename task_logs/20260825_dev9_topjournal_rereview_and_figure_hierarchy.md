# Dev9 top-journal shadow review and figure-hierarchy repair

Date: 2026-08-25  
Code baseline: `40619dbc7e06bf52494cc08e1d028a2a6af23a6c`  
Review authority: development / `analysis_only`; this review does not authorize manuscript claims or paper release.

## Executive finding

The nine runs pass the prose-oriented quality audit, but that is not equivalent to top-journal readiness. Re-review against 14 published anchors (11 supplements visually inspected) found that seven tasks can support a development-grade article display after the present renderer/contract repairs. H2 and H3 remain scientifically fail-closed: H2 has no identified comparator and H3 has no defensible selected trajectory solution. All nine original reviewer reports still aggregate to `major_revision`; all nine top-journal supplement inventories remain incomplete; paper authority remains 0/9.

## External anchor pack

Frozen source: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_84f31fd_refresh_20260825/anchor_source_pack.json`  
SHA-256: `91324b6725f55eaeb0171b28b771709f8c516d5f426dd116476fe4b2ff0f9866`

| Task | Published anchors | Main-display / supplement signal used for review |
|---|---|---|
| E1 | PMID 38905261; PMID 34259454 | 7 figures + 3 tables + 3 supplements; 3 figures + 2 tables + 1 supplement |
| E2 | PMID 20181242 | 3 figures + 2 tables; nonlinear exposure and timing/measurement context |
| E3 | PMID 35674748 | 4 figures + 4 tables + 1 supplement; stage gradient and kidney-definition detail |
| M1 | PMID 35353902; PMID 34238904 | 7/3 figures, 2/2 tables, one supplement each; missingness is central only when it changes the construct |
| M2 | PMID 31209213; PMID 32383124 | 10 figures + 3 tables; external-validation paper 2 figures + 3 tables + 2 supplements |
| M3 | PMID 31104070; PMID 42223936 | Derivation/validation plus multicentre reproducibility; the PMC count for the first anchor is unavailable and was not treated as zero content |
| H1 | PMID 32735841 | 3 figures + 3 tables + 1 supplement; time-varying exposure and informative-censoring handling |
| H2 | PMID 32059682; PMID 37073334 | 2/6 figures, 2 tables each, one supplement each; explicitly identified exposure timing groups |
| H3 | PMID 35786445 | 6 figures + 1 table + 1 supplement; alternative clustering and multi-cohort validation |

Published effect sizes were not used as answer keys. Only design, reporting, figure hierarchy, and supplement coverage were compared.

## Three independent reviewer lenses

### Reviewer 1 — statistician

Recommendation: **major revision**.

- E2 now gives the continuous adjusted curve the visual priority it needs and retains absolute-risk/measurement context plus prespecified sensitivity. It still lacks external replication.
- E3 now shows the stage distribution/outcome gradient, adjusted association, and robustness without spending a main panel on routine missingness. Baseline kidney-function operationalisation remains a manuscript-level limitation.
- M2 now shows calibration, ROC, precision-recall, Brier/calibration slope, zero patient overlap, and repeated patient-level split variability. The display clearly says internal validation only; it cannot establish transportability.
- H1 is adequate for development display, but the published anchor's time-varying exposure and informative-censoring model remains methodologically stronger.
- H2 must not estimate a causal contrast without an identified comparator. H3 must not select or name clusters when the selection/stability contract fails.

### Reviewer 2 — ICU clinician

Recommendation: **major revision**.

- E1's Sepsis-3 denominator and operationalisation are transparent, but chart review and cross-centre reproduction are absent.
- E3 needs the baseline-creatinine/KDIGO ascertainment boundary to remain explicit; a clean gradient plot does not validate the clinical definition.
- M1 appropriately keeps measurement completeness visible because missing bilirubin measurements directly affect the SOFA-component construct; this is different from routine missingness in E2/E3.
- H1 should not imply a causal ventilation effect. H2's absence of a valid unexposed/delayed comparator is a correct negative result, not an analysis failure to be cosmetically repaired.
- H3's refusal to force a phenotype is clinically safer than assigning unstable trajectory labels.

### Reviewer 3 — journal and figure editor

Recommendation: **major revision**.

- Published comparators commonly distribute evidence across several figures, tables, and supplements. A single overloaded 2×2 composite is not an adequate article package by itself.
- Routine missingness/measurement-process panels belong in the supplement unless missingness is the scientific question or changes interpretation. E2 and E3 were repaired accordingly; M1 keeps it because it is central.
- The repaired E2 figure is a three-panel asymmetric layout; E3 is a two-panel result row plus a shallow robustness strip; M2 makes calibration the hero panel with compact discrimination and validation panels, while decision-curve analysis is supplementary.
- All repaired numeric figures have PNG, SVG, PDF, TIFF, source-data CSVs, and a figure contract. The image-generated storyboard is composition-only and contains no scientific values.

## Cross-review synthesis

| Task | Current development display | What the evidence supports | Remaining top-journal gap |
|---|---|---|---|
| E1 | pass | Transparent Sepsis-3 prevalence/mortality development analysis | chart review and external reproducibility |
| E2 | pass after renderer replay | Nonlinear lactate association with absolute-risk context and sensitivity | external reproducibility |
| E3 | pass after renderer replay | KDIGO stage gradient, adjusted association, and robustness | external reproducibility; clinical baseline-kidney definition boundary |
| M1 | pass | Hepatobiliary component result with construct-relevant measurement audit | external reproducibility |
| M2 | pass after current-code replay | Patient-separated **internal** validation with discrimination, calibration, Brier, PR, and repeated splits | external validation; no transportability claim |
| M3 | pass as a negative/stability-bounded analysis | Observed internal instability and algorithm agreement boundary | external reproducibility; no strong phenotype naming |
| H1 | pass with limitation | Time-to-event/RMST development analysis with risk accounting and diagnostics | external reproducibility; time-varying/informative-censoring method gap versus anchor |
| H2 | fail-closed (correct) | The requested causal comparison is not identified from the available groups | identified comparator and positivity evidence |
| H3 | fail-closed (correct) | No defensible selected trajectory solution was established | baseline table, alternative algorithm, external reproducibility, stable selection |

Counts after this repair review:

- Prose-oriented manuscript quality audit: 9/9 pass (not a scientific-readiness claim).
- Development supplement completeness: 9/9.
- Article-display strategy: 7/9 usable for development; H2/H3 remain fail-closed.
- Top-journal supplement completeness: 0/9.
- Original three-role reviewer recommendation: 9/9 `major_revision`.
- Paper-authorized / publication-ready: 0/9.

## Repairs made

1. Added a typed `main` versus `supplementary` panel placement owned by the final article figure strategy.
2. Projected that placement into deterministic renderer execution before the plan digest is sealed.
3. Kept supplementary panels out of the main-figure binding requirement while preserving their dedicated supporting contracts.
4. Added precise, one-way chart aliases so legacy deterministic renderers are not falsely rejected; no selection curve can masquerade as a phenotype/stability plot.
5. Re-rendered E2 as a three-panel main figure and moved routine measurement availability to supporting material.
6. Re-rendered E3 as a compact three-panel result figure and moved routine missingness to supporting material.
7. Re-rendered M2 with calibration as the hero, explicit internal-validation identity, ROC/PR, repeated patient-level split variability, and supplementary decision-curve analysis.
8. Accepted a metric dot-interval as a generic validation display while preserving the external-validation requirement in the top-journal supplement gate.

No E2, E3, Sepsis, KDIGO, lactate, or benchmark-specific condition was added to a shared prompt or renderer decision. Changes are owner-scoped and case-neutral.

## Verified artifacts

- Layout-only storyboard: `output/dev9_topjournal_rereview/layout_storyboard.png`
- E2: `output/dev9_topjournal_rereview/e2_main_figure/e2_main_results.{png,svg,pdf,tiff}`
- E3: `output/dev9_topjournal_rereview/e3_main_figure/e3_main_results.{png,svg,pdf,tiff}`
- M2: `output/dev9_topjournal_rereview/m2_main_figure/m2_main_results.{png,svg,pdf,tiff}`
- M2 supplement: `output/dev9_topjournal_rereview/m2_main_figure/m2_main_results_supplementary_decision_curve.{png,svg,pdf,tiff}`
- Strategy replay audits: `output/dev9_topjournal_rereview/strategy_audit_staging/{e2,e3,m2}/article_figure_strategy_audit.json` — all complete with empty error lists.

These are renderer-only replays of exact agent-produced source tables. They do not refit models and remain `analysis_only`.
