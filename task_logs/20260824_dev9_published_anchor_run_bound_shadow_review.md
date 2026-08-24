# Dev9 published-anchor current-quality run-bound shadow review

Date: 2026-08-24

Branch: `codex/dev9-quality-remediation`

Evaluator owner commit: `b8b2aa5`

Evidence correction commit: this file's containing commit

Provider calls: `0`

Authority: `ai_development_review` / `analysis_only`; no human or publication authorization

## Correction of run authority

The first typed pass bound the anchors to the earlier execution-closure runs. A later tracked evidence log, `task_logs/20260824_dev9_gold_free_shadow_review.md`, identifies the newer quality-remediation run for every task. The earlier typed artifacts remain immutable historical comparison, but they no longer represent the current Dev9 quality baseline.

This corrected review binds the same exact 14 published anchors to the nine newer runs. It does not delete, overwrite, or relabel the earlier run evidence.

## Exact anchor authority

- Protocol: `benchmarks/figure2_canonical9/dev9_comparator_shadow_review_v1.json`
- Protocol canonical SHA-256: `3769bf7987bfe0b950912c44dd204baeda441df49c00565ccb818132bc605a39`
- Source pack: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/anchor_source_pack.json`
- Source-pack canonical SHA-256: `9932ff2272e8520d980f7c019110700f4262da059018151326f880a8af9b9b8c`
- Source-pack file SHA-256: `4acab125a5438df339dad52c876cdc7cc1d1fdba3aa46c54f000248c91b702c9`
- Exact anchors hydrated: `14/14`; PubMed abstract coverage `14/14`; PMC full-text coverage `10/14`.

The fixed papers remain evaluator-only. They were not exposed to Planner, and no published number, effect direction, result similarity, definition, or figure template was treated as a gold answer.

## Current run coordinates

| Task | Exact HEAD | Exact image digest prefix | Completion / authority |
|---|---|---|---|
| E1 | `c8efd12` | `sha256:8985d7e5` | 12/12, analysis/numeric/artifact/manuscript validated; analysis_only |
| E2 | `17e7449` | `sha256:f41d7899` | 11/11, analysis/numeric/artifact/manuscript validated; analysis_only |
| E3 | `7cafed3` | `sha256:043b9e9f` | 12/12, analysis/numeric/artifact/manuscript validated; analysis_only |
| M1 | `7cafed3` | `sha256:043b9e9f` | 11/11, analysis/numeric/artifact/manuscript validated; analysis_only |
| M2 | `7cafed3` | `sha256:043b9e9f` | 11/11, analysis/numeric/artifact/manuscript validated; analysis_only |
| M3 | `da6d93f` | `sha256:4c837463` | 10/10, analysis validated; article/display suffix incomplete |
| H1 | `9f37050` | `sha256:dd5f31df` | 3/3, analysis/numeric/artifact/manuscript validated; PH block retained |
| H2 | `3fe26c0` | `sha256:7208b274` | 1/1, signed source-feasibility fail closed |
| H3 | `5451192` | `sha256:bc4a1df8` | 4/4, analysis validated; no interior BIC solution |

## Current typed result

- Per-task reviews: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/run_bound_reviews_current_quality/*.json`
- Machine summary: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/current_quality_run_bound_shadow_review_summary.json`
- Reader summary: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/current_quality_run_bound_shadow_review_summary.md`
- Machine-summary SHA-256: `579da6a4622ac5087769039886f44e1c336f3f9985a5bde9ce086f31f34636e3`
- Reader-summary SHA-256: `52edad9732e89fb6c2cf584563ae95124335bc1776b663fa7cb94d7dae4d8e4a`
- Validation: `9/9` typed reviews; `63/63` frozen dimensions; missing evidence paths `0`.

| Task | Status | Actionable gaps | Meets | Correct fail-closed | N/A |
|---|---|---:|---:|---:|---:|
| E1 | changes_required | 2 | 5 | 0 | 0 |
| E2 | changes_required | 1 | 6 | 0 | 0 |
| E3 | changes_required | 2 | 5 | 0 | 0 |
| M1 | changes_required | 2 | 5 | 0 | 0 |
| M2 | changes_required | 1 | 6 | 0 | 0 |
| M3 | changes_required | 2 | 5 | 0 | 0 |
| H1 | changes_required | 2 | 4 | 1 | 0 |
| H2 | accepted | 0 | 2 | 4 | 1 |
| H3 | accepted | 0 | 5 | 2 | 0 |

Across 63 dimensions: `12 actionable_gap`, `43 meets_anchor`, `7 fail_closed_appropriate`, and `1 not_applicable`.

## Scientific interpretation

- E1/E3/M1 still need a signed post-baseline timing/adjustment decision; E1/M1 also need a genuinely nonduplicate sensitivity axis.
- E2 needs a broader robustness grid, not a new result-driven exposure definition.
- M2 already has patient-level split, no overlap, leakage controls, calibration, Brier, held-out internal validation, and decision curves. Its remaining gap is external/temporal transport validation, which cannot be synthesized from the same database.
- M3's candidate is analysis-validated and not overnamed. External replication and its deterministic article/display suffix remain.
- H1 has explicit landmark timing and risk-set accounting. The global PH violation remains a real block requiring a prespecified non-PH strategy; it must not be relabeled away.
- H2 `accepted` means only that `H2_VERIFIED_NON_USE_UNAVAILABLE` correctly prevents a fabricated control arm, positivity claim, or causal estimate.
- H3 `accepted` means only that `H3_NO_INTERIOR_BIC_OPTIMUM` correctly prevents K selection, stability/outcome binding, and phenotype labels.

## Historical comparison retained

The earlier typed review is retained at `run_bound_reviews/*.json` with `run_bound_shadow_review_summary.json` (30 gaps). It is useful for showing improvement from execution-closure to quality-remediation runs, but it must not be quoted as the current state.

## Next efficient action

Do not rerun all nine tasks. Close the small author decision set, implement a generic non-PH route for H1, finish M3's deterministic reporting suffix, and use a validation-design boundary for M2. H2/H3 require no fabricated completion. Then replay only affected owners, freeze one exact HEAD/image, and run one full exact-head CI before Qualification12/Held-out27.
