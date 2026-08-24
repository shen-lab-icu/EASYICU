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
- H3 correctly refuses a boundary K, but the current search lacks multi-metric and alternative-algorithm diagnostics.
- M2 is nearest to Dev9 acceptance; it still has a PR-axis labeling defect, incomplete calibration reporting, no decision curve in the main figure, and no repeated internal-split uncertainty.
- H1 is the highest-priority scientific blocker: proportional hazards are rejected, yet a single Cox point estimate remains prominent in the manuscript and figure.
- E1/E2/E3/M1 require generic post-baseline time alignment and independent, nonduplicate sensitivity axes.
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
