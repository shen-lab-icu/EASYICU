# Dev9 published-anchor run-bound shadow review

Date: 2026-08-24

Branch: `codex/dev9-quality-remediation`

Evaluator owner commit: `b8b2aa5`

Provider calls: `0`
Authority: `ai_development_review` / `analysis_only`; no human or publication authorization

## Question closed

The 14 user-supplied published papers are now used as evaluator-only methodological and presentation anchors for the nine Dev9 runs. They are not numeric gold answers, expected effect directions, result-similarity targets, or templates. The fixed anchors are not exposed to Planner before execution.

## Exact source authority

- Protocol: `benchmarks/figure2_canonical9/dev9_comparator_shadow_review_v1.json`
- Protocol canonical SHA-256: `3769bf7987bfe0b950912c44dd204baeda441df49c00565ccb818132bc605a39`
- Source pack: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/anchor_source_pack.json`
- Source-pack canonical SHA-256: `9932ff2272e8520d980f7c019110700f4262da059018151326f880a8af9b9b8c`
- Source-pack file SHA-256: `4acab125a5438df339dad52c876cdc7cc1d1fdba3aa46c54f000248c91b702c9`
- Exact anchors hydrated: `14/14`; PubMed abstract coverage `14/14`; PMC full-text coverage `10/14`.
- The source pack persists metadata, coverage, section/figure/table counts, and content digests only; it does not persist article text.

The owner fixes two live NCBI lineage hazards: PMC EFetch receives numeric identifiers at the transport boundary, and PubMed parsing reads only the current record's top-level `ArticleIdList`, never cited-reference identifiers.

## Run-bound review result

Every task is bound to its historical execution HEAD, exact image digest, run path, exact task anchors, and seven frozen dimensions:

1. study population;
2. time zero and windows;
3. variable operationalization;
4. missingness and censoring;
5. primary model and sensitivities;
6. table and figure completeness;
7. conclusion boundaries.

Artifacts:

- Machine summary: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/run_bound_shadow_review_summary.json`
- Reader summary: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/run_bound_shadow_review_summary.md`
- Per-task typed reviews: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_b8b2aa5_20260824/run_bound_reviews/*.json`
- Summary JSON SHA-256: `2b2d0f5a4b477b821656a3445b09e673a4407f39457a9a18f1236f98432d13bb`
- Summary Markdown SHA-256: `883f4e7c358ec2e85c21a622c2a2dec1eeb3d703f49de9024d6a993c1e9a0e51`

Validation: `9/9` typed reviews pass the protocol/source-pack validator; `63/63` review rows have existing run evidence paths; missing evidence paths `0`.

| Task | Status | Actionable gaps | Correct fail-closed | Meets/stronger |
|---|---|---:|---:|---:|
| E1 | changes_required | 5 | 0 | 2 |
| E2 | changes_required | 5 | 0 | 2 |
| E3 | changes_required | 4 | 0 | 3 |
| M1 | changes_required | 3 | 0 | 4 |
| M2 | changes_required | 2 | 0 | 5 |
| M3 | changes_required | 4 | 0 | 3 |
| H1 | changes_required | 5 | 0 | 2 |
| H2 | accepted | 0 | 5 | 0 |
| H3 | changes_required | 2 | 2 | 3 |

Across 63 dimensions: `30 actionable_gap`, `23 meets_anchor`, `1 stronger_than_anchor`, `7 fail_closed_appropriate`, and `2 not_applicable`.

## Important interpretation

- H2 `accepted` means only that the runtime correctly refuses to fabricate a non-use/control arm and returns `H2_VERIFIED_NON_USE_UNAVAILABLE` with a null effect estimate. It is not a completed causal paper.
- H3 correctly refuses to select K because the minimum BIC lies at the upper boundary and stability has zero refits. It still requires method-source binding and missingness/survivorship sensitivity before a new candidate run.
- M2 already has patient-level split with zero overlap, AUROC/AP/Brier, calibration intercept/slope, and decision curves. Its remaining scientific gaps are external/temporal validation and complete TRIPOD+AI/manuscript reporting.
- M3 has low silhouette and mean bootstrap ARI `0.289`; no clinical naming, outcome claim, or generalizability claim is authorized.
- H1 correctly blocks paper authorization because the global PH diagnostic is violated (`p=7.889e-35`); a non-PH analysis owner must be selected before regeneration.
- E1/E2/E3/M1 share owner-level gaps in post-baseline time anchoring, robustness breadth, and complete evidence-bound writing. E3/M1 additionally retain an endpoint-definition conflict.

## Next execution boundary

Do not rerun all nine runs. Repair only the shared owners identified by the review, then run bounded affected-case replays: time-zero/endpoint authority; association robustness; H1 non-PH; M3/H3 phenotyping/trajectory stability; M2 external-validation boundary; complete reporting suffix. H2 remains unchanged. Qualification12 and Held-out27 stay blocked until these Dev9 owner gaps close and one exact HEAD/image/full-CI checkpoint is frozen.
