# Dev9 gold-free comparator shadow review

Date: 2026-08-24 EDT  
Protocol: `easyicu.dev9_comparator_shadow_review/20260824-v1`  
Review scope: evaluator-only method and presentation comparison; no comparator
effect estimate, direction, or numerical result was used as a gold answer.

## Decision

Dev9 has closed the main deterministic execution contracts, but the nine cases
are not yet one homogeneous set of article-grade outputs. Five cases have a
complete analysis-to-manuscript chain (E1, E2, E3, M1, M2), H1 has a complete
article chain with a correctly retained proportional-hazards block, M3 is
currently validated through analysis only, and H2/H3 are scientifically correct
failed-closed non-solutions.

The next efficient gate is not nine fresh Provider runs. Close the remaining
generic evaluation and design gaps, replay only the affected deterministic
suffixes, then freeze one exact HEAD/image for full CI.

State key: `M` meets the anchor for this development scope; `G` is an actionable
gap; `F` is an appropriate fail-closed boundary; `NA` is not applicable.

| Case | Population | Time zero/window | Variable definition | Missing/censoring | Model/sensitivity | Tables/figures | Conclusion boundary |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | M | G | M | M | G | M | M |
| E2 | M | M | M | M | G | M | M |
| E3 | M | G | M | M | G | M | M |
| M1 | M | G | M | M | G | M | M |
| M2 | M | M | M | M | G | M | M |
| M3 | M | M | M | M | G | G | M |
| H1 | M | M | M | G | G | M | F |
| H2 | M | M | F | F | F | NA | F |
| H3 | M | M | M | M | F | M | F |

## Case findings and ownership

- **E1**: Sepsis-3 construction, denominators, missingness audit, sensitivity
  reporting, and figure provenance are strong. A user-signed timing/adjustment
  specification and at least one genuinely distinct sensitivity axis remain.
  Owner: study-design authority; no E1-specific executor patch.
- **E2**: Continuous lactate presentation and nonlinear/absolute-risk outputs
  are present. The robustness grid remains narrower than the external method
  anchor. Owner: robustness specification authority.
- **E3**: KDIGO staging gradient and LOS/outcome presentation are present.
  Post-baseline timing and adjustment-set authorization remain study-design
  decisions. Owner: temporal and adjustment authority.
- **M1**: Missing bilirubin is not converted to normal and the measurement
  process is visible. Timing, adjustment authorization, and a non-duplicate
  sensitivity axis remain. Owner: temporal/adjustment/robustness authority.
- **M2**: Patient-level split, leakage controls, calibration, Brier score,
  held-out validation, and decision-curve output are present. External
  transport validation is absent and cannot be synthesized from the same
  database. Owner: validation design/data availability, not model code.
- **M3**: The internally stable candidate is analysis-validated and is not
  presented as a universal biological subtype. External replication remains
  absent; the current exact scientific run also stops before article/display
  completion. Owner: phenotyping validation plus reporting suffix.
- **H1**: Landmark timing and risk-set accounting are explicit. The mixed
  unadjusted-KM/adjusted-Cox figure was correct, but the maturity evaluator did
  not follow upstream runtime-receipt lineage. Generic fix `a602388` restores
  adjustment authority without changing the result. The global PH violation
  remains a real block and requires a prespecified non-PH/time-varying strategy,
  not relabeling.
- **H2**: The source cannot establish a real non-use comparator and therefore
  cannot establish positivity. The correct result is
  `H2_VERIFIED_NON_USE_UNAVAILABLE`; no causal effect is reportable. Owner:
  source feasibility/data acquisition.
- **H3**: The selected K is at the prespecified BIC boundary, so stability and
  outcome binding were not authorized. The correct result is
  `H3_NO_INTERIOR_BIC_OPTIMUM`; no phenotype labels are reportable. Owner:
  signed trajectory authority.

## Cross-cutting gaps

1. The runs used curated seed literature, not a dated, reproducible top-journal
   search and source-bound novelty matrix. The comparator protocol is a valid
   evaluator shadow review, but it does not retroactively turn those runs into
   literature-complete manuscripts.
2. Independent scientific review and final PDF/render QA are not closed.
3. Several sensitivity and adjustment choices require author authorization;
   the agent must not silently invent them to raise a maturity score.
4. Analysis-only and fail-closed results remain valid development evidence, but
   are not publication-ready artifacts.

## Run evidence

- E1: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_final_c8efd12_20260824/e1_r2/e1_sepsis3_prevalence_mortality/aware/run_20260824T101506_789e1d`
- E2: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_final_17e7449_20260824/e2/e2_lactate_mortality/aware/run_20260824T112713_57761f`
- E3: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_final_7cafed3_20260824/e3/e3_kdigo_gradient/aware/run_20260824T102752_9b7751`
- M1: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_final_7cafed3_20260824/m1/m1_hepatobiliary_missingness/aware/run_20260824T103304_f4d713`
- M2: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_final_7cafed3_20260824/m2/m2_mortality_prediction/aware/run_20260824T103808_8138bf`
- M3: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_m3_da6d93f_analysis_replay_20260824/m3_sepsis_subphenotype/aware/run_20260824T121442_fece49`
- H1: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_final_9f37050_20260824/h1/h1_ventilation_survival/aware/run_20260824T084257_06f487`
- H2: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h2_3fe26c0_analysis_replay_r3_20260824/h2_vasopressor_causal/aware/run_20260824T123019_ea10b6`
- H3: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h3_5451192_analysis_replay_20260824/h3_trajectory_clustering/aware/run_20260824T124822_ae50b3`

## Acceptance consequence

Do not start Held-out27 yet. First close the generic literature/novelty preflight,
obtain the small set of author design decisions, finish M3's deterministic
article/display suffix if the validated analysis remains unchanged, and choose
a prespecified H1 non-PH handling path. H2/H3 need no fabricated completion.
After those items, build one exact-head image, run one full exact-head CI, and
then proceed through the sealed qualification/formal protocol.
