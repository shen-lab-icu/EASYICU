# Dev9 article-suite inventory remediation

Date: 2026-08-25 EDT  
Exact renderer HEAD: `b4c5a232038d92251489cc21a815d2f5a3f8c050`  
Authority: development / `analysis_only`; paper authorization remains false.

## Published ICU display benchmark

Frozen comparator pack:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_84f31fd_refresh_20260825/anchor_source_pack.json`
(SHA-256 `91324b6725f55eaeb0171b28b771709f8c516d5f426dd116476fe4b2ff0f9866`).

Thirteen accessible full texts contained 2–10 main figures (median 3) and
1–4 main tables (median 2). Eleven downloaded supplements had a median of one
supplementary figure and four supplementary tables, with wide analysis-family
variation. These counts are planning references, not acceptance gates and not
numeric answer keys.

The resulting generic display policy is:

- split displays by scientific claim rather than force one overloaded composite;
- plan 2–4 complementary main figures and 2–3 main tables for a result-bearing
  development article;
- put routine missingness and measurement-process details in the supplement;
- keep missingness in the main article only when it is the research question or
  changes the construct, denominator, estimand, or interpretation;
- do not apply main-display count targets to a scientifically fail-closed task.

## Repairs

1. Added a digest-bound article display inventory with explicit
   `main|supplementary` placement for figures and tables.
2. Added byte-for-byte EvidenceStore table packaging and table contracts that
   record upstream evidence id/path/digest, support, and claim boundary.
3. Changed the manuscript reader to render separate `Main figures` and
   `Supplementary figures` sections.
4. Split M2 into two main figures: calibration/discrimination and a separate
   two-panel patient-level validation design/stability figure. Decision-curve
   analysis remains supplementary.
5. Preserved H2/H3 as fail-closed with zero main result figures/tables; only
   diagnostic supplementary displays are emitted.

## Exact artifact

Root:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_article_suite_b4c5a23_20260825`

Manifest SHA-256:
`e4502278ea35fc46f4c14d77858765ac9cc51ffe97f8d6531352550b3e3842e5`

Inventory SHA-256:
`10f5e3bd858c7d165650159b4ba958cc2cf62699b1b9e2af44a14bbb11947a60`

| Task | Status | Main figures | Main tables | Supplementary figures | Supplementary tables |
|---|---|---:|---:|---:|---:|
| E1 | analysis_only | 3 | 2 | 1 | 2 |
| E2 | analysis_only | 2 | 2 | 1 | 2 |
| E3 | analysis_only | 2 | 2 | 1 | 2 |
| M1 | analysis_only | 3 | 2 | 0 | 2 |
| M2 | analysis_only | 2 | 3 | 1 | 2 |
| M3 | analysis_only | 3 | 3 | 0 | 2 |
| H1 | analysis_only | 4 | 3 | 0 | 2 |
| H2 | failed_closed | 0 | 0 | 1 | 1 |
| H3 | failed_closed | 0 | 0 | 1 | 2 |

All 59 display placements are resolved. Every result-bearing task falls within
the non-binding article planning range; H2/H3 correctly waive that range.

E2 reader:
`output/pdf/e2_article_suite_b4c5a23/manuscript_scaffold.pdf`
(SHA-256 `3ca2bbee0bbeee61c2ecb9821f1e5030615bd788761226bd9a518a8721baffbf`).
Pages 8–9 were visually inspected: the two main figures are separate and
readable, the measurement-process audit is supplementary, and no clipping or
overlap was observed. The final M2 validation PNG was also visually inspected
after correcting an arrow-label collision and obsolete panel lettering.

## Verification and boundary

- Focused tests: `22 passed, 1 deselected`.
- Ruff format/check and `git diff --check`: passed.
- Provider/Planner/Coder calls: 0; scientific recomputation: false.
- No case-specific condition was added to a shared prompt or shared execution
  decision. Benchmark-specific display selections remain in the Dev9 renderer.
- This closes the article display-structure gap only. It does not close external
  validation/reproduction, chart review, clinical-definition validation, stable
  phenotype selection, or paper authorization. Dev9 remains publication-ready
  0/9.
