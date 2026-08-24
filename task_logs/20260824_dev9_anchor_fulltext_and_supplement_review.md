# Dev9 anchor full-text and supplementary-material review

Date: 2026-08-24

Review HEAD: `e271409c1cca151ced49acce37efe974a25f6ffb`

Branch: `codex/dev9-quality-remediation`
Provider calls: `0`

## Scope

Re-audited the nine current Dev9 runs against the 14 retained published anchors. Unlike the earlier seven-dimension shadow review, this pass inspected available main texts and supplementary materials and compared study design, time zero, operational definitions, missingness/censoring, model family, sensitivities/validation, main figures/tables, supplement content, manuscript consistency, and claim boundaries. Published numerical values or effect directions were not used as expected answers.

## Source evidence

The review acquired and inspected supplements for E1, E3, M1, M2, M3, H1, H2, and H3. E2 had no separate file in the Europe PMC supplementary package; its complete main text supplied the comparator. Notable inventories included:

- M3 Seymour: 96 pages, 39 eFigures, 32 eTables.
- H3: 74 pages, 19 tables, 20 figures.
- H1: 22 pages with joint-model alternatives and missingness tables.
- M2 external validation: calibration intercept/slope, Brier/scaled Brier, decision curves, recalibration, and FINNAKI external validation.
- E1: definition sensitivities, Cox models, risk-adjusted trends, and predictive-validity metrics.

Converted-source hashes and the complete per-task matrix are in:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_fulltext_supplement_review_e271409_20260824/`

## Findings

- `0/9` current runs meet a complete paper-level analysis-package bar.
- H2 is accepted only as a correct source-feasibility fail-closed non-solution.
- H3 correctly refuses a boundary K. A later exact replay adds AIC as a
  diagnostic criterion and confirms agreement with BIC, while alternative
  algorithms, window/missingness variants, and external reproduction remain
  absent.
- M2 is nearest to Dev9 acceptance; it still has a PR-axis labeling defect, incomplete calibration reporting, no decision curve in the main figure, and no repeated internal-split uncertainty.
- H1 is the highest-priority scientific blocker: proportional hazards are rejected, yet a single Cox point estimate remains prominent in the manuscript and figure.
- E1/E3/M1 require generic post-baseline time alignment and independent,
  nonduplicate sensitivity axes. Later receipt review confirms E2 already uses
  a real 24-hour landmark; its unresolved issue is measurement selection, not
  time-zero construction.
- M3 has low resampling stability and must not name biological subtypes or bind outcome claims without stronger selection/stability evidence.
- Main figures often use engineering-audit panels where a mature paper would use scientific result or sensitivity panels.
- Current manuscript section completeness does not establish manuscript quality; method-result-figure consistency is still insufficient.

## Governance correction

`benchmarks/figure2_canonical9/dev9_scientific_decision_packet_20260824.{json,md}` was marked `withdrawn_pending_fulltext_and_supplement_review`. It must not be signed or executed. Progress CURRENT pages now state the review-in-progress truth and keep Qualification12/Held-out27 at zero.

## Next implementation order

1. P0 generic owners: non-PH survival authority; post-baseline time alignment; prediction-figure QA/calibration; clustering stability/naming gates.
2. P1 generic owners: distinct sensitivity axes, study-family figure suites, and a supplement assembler.
3. Focused owner tests and affected deterministic replays only.
4. Repeat the nine-task full-text/supplement review, then one exact-HEAD image and one full exact-HEAD CI.

No Planner/Coder replay, new experiment, Qualification12, Held-out27 run, or Provider call was made during this audit.

## Generic remediation landed after the audit

The first P0/P1 owner-scoped remediation was implemented on the same isolated
branch without changing a Planner prompt or adding a case-specific executor:

- survival owner: PH rejection now withholds the constant Cox headline and the
  signed landmark suite emits a source-backed 27-day post-landmark RMST
  difference with 95% CI for the PH-free figure panel;
- prediction figure owner: calibration intercept, slope and Brier are visible,
  the nonstandard `1-Brier` panel was replaced, and the renderer must consume
  the registered `table:clinical_utility` decision-curve product rather than
  recomputing an unregistered look-alike;
- phenotyping figure owner: unstable solutions remain candidate clusters;
  phenotype naming and outcome claims are explicitly unauthorized;
- reporting owner: Writer receives a hard instruction not to place a rejected
  constant Cox effect in headline manuscript locations;
- supplement owner: every finalized run receives a family-aware inventory of
  present and missing cohort, missingness, robustness, diagnostic, validation,
  figure-source and reproducibility sections. The inventory records presence
  only and never certifies publication readiness.

Focused verification after these changes: `48 passed, 2 deselected` across the
survival, prediction, phenotyping, figure-shaping, Writer and supplement
contracts. Full CI remains intentionally deferred until the nine regenerated
development packages have been re-reviewed.

## Nine-task paper-package acceptance matrix

The comparator is the retained paper's design and reporting completeness,
including its supplementary methods, tables and figures. The comparator is
not its numerical estimate, effect direction, selected cluster count, or model
performance value.

| Task | Current paper-package disposition | Missing or insufficient relative to the anchor package |
|---|---|---|
| E1 Sepsis-3 | not accepted | Post-baseline timing and repeated-stay handling; explicit operational-definition sensitivity axes; a main scientific prevalence/mortality figure rather than a thin or mislabeled audit display; fuller epidemiology supplement tables. |
| E2 lactate | not accepted | The primary runtime receipt proves a 24-hour alive-and-observed landmark and the figure includes nonlinear association and absolute-risk displays. About 46% remain lactate-unmeasured, so measurement-by-indication and an independent missingness/measurement sensitivity remain unresolved; engineering audit panels also displace scientific sensitivity content. |
| E3 KDIGO | not accepted | Temporal KDIGO construction and baseline renal-function authority; stage-definition sensitivity; an appropriate LOS model; bootstrap uncertainty; scientific result panels. |
| M1 hepatic component | not accepted | Exact bilirubin timing and adjustment authority; first-versus-maximum measurement sensitivity; linear-versus-spline functional-form sensitivity; nonduplicate missingness/measurement analyses. |
| M2 prediction | near acceptance, not accepted | Repeated patient-level splits or bootstrap uncertainty; externally transported validation remains absent. The main figure now reports AUROC, average precision, calibration intercept/slope, Brier score and registered decision-curve net benefit. |
| M3 static phenotyping | not accepted | Mean resampling ARI is low; no alternative algorithm, alternate feature/window analysis, or external reproduction. Cluster naming and outcome claims must remain unauthorized. |
| H1 survival | not accepted | The rejected PH assumption is now handled with a promoted RMST difference, but measurement/missingness and a second independent sensitivity axis remain absent; literature and manuscript binding are incomplete. |
| H2 causal | accepted only as correct fail-closed | No authoritative non-exposed comparator or positivity support exists. No causal contrast, effect estimate, PSM/IPTW figure, or manuscript claim may be generated. |
| H3 trajectories | not accepted | Refusal of the upper-boundary K is correct, and AIC now agrees with the prespecified BIC decision. Alternate algorithm/model limits, alternate trajectory window/missingness analysis, and external reproduction remain absent. |

Result: `0/9` complete paper packages. H2 is a correct non-solution, not a
completed causal paper. M2 is the nearest development-quality package. No task
is authorized for Qualification12, Held-out27, or paper-result promotion.

## Exact replay evidence after the first remediation

### H1 non-PH survival

- Exact replay commit/image: `3132cd50fd3aacf323632d6dfd6ecd11bbd5a722` / `sha256:6691666bde80593e8e1f187ec863849f8106401cb4a1dbba4a638900808df07b`.
- Required/completed: `3/3`; missing/failed steps: `[]/[]`.
- One Writer call: 25,737 prompt + 247 completion = 25,984 tokens; `$0.26478`.
- The primary figure now promotes the unadjusted 27-day post-landmark RMST
  difference and retains Cox only as a PH diagnostic. The prior adjustment
  authority conflict is absent.
- Supplement inventory remains incomplete because the
  `missingness_measurement` section is absent. The run remains
  `analysis_only`, not a paper package.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_3132cd5_quality_replay_20260824/h1/h1_ventilation_survival/aware/run_20260824T164130_12a3f6`.

### M2 prediction

- Exact replay commit/image: `6aa12110ee3265bdfbb0b614c8352f921733284e` / `sha256:1a7586ec754b7b82071b78df79b64395b695e422c4f461803772af8bd16fec30`.
- Required/completed: `11/11`; missing/failed steps: `[]/[]`.
- One Writer call: 27,411 prompt + 228 completion = 27,639 tokens; `$0.28095`.
- Clinical-utility and calibration source CSVs bind their producing table,
  step, and exact source row. The figure displays AUROC, average precision,
  calibration intercept/slope, Brier score and a registered DCA curve.
- Supplement inventory correctly reports missing `resampling_validation` and
  `external_validation`. Writer hard-stop and exact literature binding also
  prevent a complete manuscript. The run remains `analysis_only`.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_6aa1211_quality_replay_20260824/m2/m2_mortality_prediction/aware/run_20260824T164537_3f6006`.

### M3 static phenotyping

- Exact replay commit/image: `6aa12110ee3265bdfbb0b614c8352f921733284e` / `sha256:1a7586ec754b7b82071b78df79b64395b695e422c4f461803772af8bd16fec30`.
- Required/completed: `10/10`; missing/failed steps: `[]/[]`.
- One Writer call: 28,356 prompt + 279 completion = 28,635 tokens; `$0.29193`.
- Selected candidate `K=2`; five resampling ARIs are 0.431, 0.117, 0.184,
  0.443 and 0.270 (mean 0.289). This is inadequate for stable biological
  phenotype naming.
- Supplement inventory correctly reports missing `alternative_algorithm` and
  `external_reproducibility`; it no longer mistakes an external LLM provider
  for external scientific validation. The figure and report explicitly limit
  the result to candidate clusters and authorize neither phenotype naming nor
  outcome claims.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_6aa1211_quality_replay_20260824/m3/m3_sepsis_subphenotype/aware/run_20260824T164811_7c5744`.

These replays validate the repaired generic owners and figure lineage only.
They do not change the overall `0/9` paper-package disposition.

## Follow-up exact replays: resampling and algorithm agreement

### M2 repeated patient-level split validation

- Exact replay commit/image: `d2f47d57711e833c1375e8fc9d84d28304271e31` / `sha256:638aef26268679ea5ee71d29b8983afe6aa453025996accc506f76b5b896375d`.
- Required/completed: `11/11`; missing/failed steps: `[]/[]`.
- One Writer call: 28,410 prompt + 206 completion = 28,616 tokens; `$0.29028`.
- The locked input plan was reused. The host-shaped final plan is byte-identical
  to the prior M2 replay (`ce2b6c606719340caba92b7f94c736dd7f91030565012f6ed71a9593ab873c06`).
- Ten deterministic patient-group splits independently refit preprocessing and
  the model, with zero patient overlap in every split. AUROC mean/SD was
  0.76689/0.00420, average precision mean/SD 0.41115/0.01497, and Brier
  mean/SD 0.07362/0.00121.
- This closes the internal resampling gap only. External validation remains
  absent, Writer/literature binding still prevents a complete manuscript, and
  the run remains `analysis_only`.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_d2f47d5_quality_replay_20260824/m2_retry3/m2_mortality_prediction/aware/run_20260824T170120_66a5dc`.

### M3 alternative-algorithm agreement

- Exact replay commit/image: `a5925cb98ed91595767ebb5abda39f560a3f7c82` / `sha256:c738d45f53a3cdfdbd9e429a7357160022e036ac85926f0014436eacc69f4215`.
- Required/completed: `10/10`; missing/failed steps: `[]/[]`.
- One Writer call: 28,323 prompt + 175 completion = 28,498 tokens; `$0.28848`.
- The locked and final plan SHA stayed unchanged
  (`b7b9ebdbc15dbc67c498e77a02930daccc41f57927ada2a8efb651a9e2e77c0a`).
- The deterministic diagonal-covariance Gaussian mixture converged at the same
  candidate `K=2`, but agreement with MiniBatchKMeans was only ARI 0.09216;
  mean resampling ARI remained 0.28891. This materially strengthens the reason
  to reject stable phenotype naming and outcome claims.
- The revised main figure was visually checked at original resolution: the
  agreement line, axes, legend and panel labels are present without clipping.
  External reproduction remains absent and the run remains `analysis_only`.
- Run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_a5925cb_quality_replay_20260824/m3/m3_sepsis_subphenotype/aware/run_20260824T170718_e82e67`.

These additions close one internal-validation inventory item in M2 and one
alternative-algorithm inventory item in M3. They do not change the overall
paper-package result: `0/9` accepted; no Dev9 result is authorized for formal
promotion, Qualification12, or Held-out27.

## Receipt correction and primary-figure visual audit

The full-text matrix originally grouped E2 with post-baseline time-zero gaps.
That was too broad. The exact E2 primary receipt records a 24-hour landmark,
requires patients to be alive and under observation at that landmark, and
reports 44,111 eligible patients (44,095 complete cases; 5,480 events). The
time-zero item is therefore closed. The remaining E2 threat is measurement by
indication: roughly 46% of the cohort has no landmark-window lactate, and the
current measured-versus-unmeasured display is not an independent correction
for that selection process.

Original-resolution review of the current association-family main figures
found the following paper-package gaps:

- E1 displays absolute prevalence/outcome risk and the adjusted odds ratio,
  but omits the already executed landmark, non-readmission, and flexible-model
  sensitivity results from the main scientific figure.
- E2 displays the landmark nonlinear curve and measured/unmeasured absolute
  risk. Its remaining panels are engineering convergence/measurement audits,
  rather than an independent scientific missingness or sampling sensitivity.
- E3 uses a primary-versus-missing robustness panel that is effectively
  duplicated and leaves substantial unused space; no stage-definition, LOS,
  or bootstrap-uncertainty panel is present.
- M1 similarly duplicates a primary-versus-missing estimate with large unused
  space; it omits first-versus-maximum bilirubin and linear-versus-spline
  sensitivities expected from the anchor package.

These are figure-suite/content deficiencies, not evidence that the displayed
numbers are wrong. Published effects were not used as expected values.

## H3 multi-criterion exact replay and readiness-axis correction

- Exact code/image: `ac6b281ebd51085851d04c8ac3af88bdbd7f375c` /
  `sha256:99068d410f9f201445a7e81acfc824a0f3963d560a41c5627166efb0f20ad4ad`.
- Run:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_h3_ac6b281_readiness_fresh_20260824/h3_trajectory_clustering/aware/run_20260824T174017_647004`.
- Locked/final plan SHA-256 remained
  `486a12676f24f65cb87b10889e33ddcca6f06403339dfb030bcdb59134384ef7`.
- Required/completed: `4/4`; missing/failed: `[]/[]`; Provider calls/tokens/cost:
  `0/0/$0.00`.
- BIC and diagnostic AIC both reach their minimum at the prespecified upper
  boundary `K=6`; `H3_NO_INTERIOR_BIC_OPTIMUM` remains in force, stability
  authorization remains false, and no outcome binding or phenotype naming is
  permitted.
- The readiness owner now keeps manuscript-stage failures on the manuscript
  axis. The exact replay is `analysis_validated=true` with empty
  `analysis_errors`, while Writer/literature/critic failures still make
  `evidence_complete=false`, `manuscript_ready=false`, and
  `paper_authorized=false`.
- Original-resolution inspection found no clipping or overflow. The BIC and
  AIC curves nearly overlap, as expected from their agreement; the boundary
  point is explicitly labelled as unauthorized.

This closes H3's multi-selection-metric inventory item only. It does not turn
the boundary non-solution into a phenotype result and does not alter the
overall `0/9` paper-package disposition.
