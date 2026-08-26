# Dev9 accessible-anchor and Writer quality remediation — 2026-08-25

## Scope and authority boundary

This pass repaired the evaluator/source-access and report-only resume path for
the nine development cases. It did not rerun or change their scientific
analyses. The immutable execution baseline remains exact HEAD `84f31fd44f2e`
with image `easyicu-research-agent:84f31fd` (digest
`sha256:fbb75a5a1812b184f40f71a365ed7dc8b854dc4dbd556eb5c69f6534d2c4cf75`),
76/76 required steps, no missing or failed steps, and zero Provider calls for
the execution replay.

All regenerated manuscripts remain `development_diagnostic` / `analysis_only`.
Neither an accessible comparison paper, a complete manuscript scaffold, nor a
successful AI review grants publication authority.

## Accessible comparison-source replacement

The evaluator protocol now requires accessible full text for every anchor and
requires published supplementary material to be recorded and reviewed. An
inaccessible anchor must be replaced, not silently assessed from title or
abstract alone.

Four inaccessible anchors were replaced with same-purpose, high-quality open
full-text studies:

- E1: PMID 29121281 -> PMID 34259454 / PMC8508729.
- E3: PMID 30819553 -> PMID 35674748 / PMC9543500.
- M2: PMID 36378565 -> PMID 32383124 / PMC7223438.
- H2: PMID 35227959 -> PMID 37073334 / PMC10106031.

Hydrated source pack:
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_84f31fd_refresh_20260825/anchor_source_pack.json`
(SHA-256 `02174551dc05ff1021fc02c4915c087ac4c0b14934c950ce4c13b49725234770`).
Coverage is 14/14 PMC full texts. Eleven anchors published supplementary
material; all 11 archives were downloaded, with 27 PDF/DOCX files converted to
Markdown and one standalone TIF inspected visually. The aggregate supplement
inventory digest is
`e4b2641b9fd68a0298a58f2bf5c31f5fd3d0b1df881560abcd133835d859b64b`.

The 9 x 7 exact-run comparison is under
`/Volumes/外置硬盘/easyicu_data/figure2_dev9_anchor_shadow_review_84f31fd_refresh_20260825/run_bound_reviews/`.
It contains 63 dimensions: 31 `meets_anchor`, 25 `actionable_gap`, 6
`fail_closed_appropriate`, and 1 `not_applicable`. All nine cases therefore
remain `changes_required` for paper review. Published effect sizes or directions
were never used as expected answers.

## General owner repairs

1. Provider hard-stop preflight now rejects a run whose total token ceiling
   cannot fund the Provider's conservative completion floor plus prompt
   overhead. The former 120,000-token configuration could never reserve a
   128,000-token completion floor plus 4,096 prompt overhead, so the Writer
   previously failed only after analysis.
2. Resume plan authority now normalizes `cohort=None` to the same canonical
   implicit primary cohort used by cohort locking and execution. This repairs
   H2/H3 report-only resume without weakening rejection of a changed or dropped
   non-default cohort.
3. Execution registration and Writer recovery now share one physical-artifact
   EvidenceStore-kind mapping. A declared `artifact:*` backed by Parquet is
   stored and recovered as a verified `table`, rather than being incorrectly
   searched as a `log`. The real H3 checkpoint then recovered four Writer
   records and 21 immutable artifact bindings, each checked by evidence id,
   digest, filename, and byte size.

No case id, Sepsis term, H2/H3 reason code, expected result, or effect direction
was added to shared implementation logic.

## Final selected Writer overlays

Every selected manuscript has substantive prose, `numeric_verified=true` with
zero numeric errors, a passing manuscript-literature audit, and a passing
manuscript critic. Plans were byte-identical to the exact execution plans and
all completed analysis steps were skipped.

| Case | Words | Numeric audit | Literature audit | Maturity | Article contract | Figure strategy | Display suite |
|---|---:|---|---|---:|---|---|---|
| E1 | 2,812 | pass, 0 errors | pass | 59, analysis_only | complete | incomplete | complete |
| E2 | 2,974 | pass, 0 errors | pass | 71, analysis_only | complete | complete | complete |
| E3 | 3,471 | pass, 0 errors | pass | 56, analysis_only | complete | incomplete | complete |
| M1 | 2,970 | pass, 0 errors | pass | 71, analysis_only | complete | complete | complete |
| M2 | 2,439 | pass, 0 errors | pass | 71, analysis_only | complete | incomplete | complete |
| M3 | 2,932 | pass, 0 errors | pass | 67, analysis_only | complete | complete | complete |
| H1 | 3,060 | pass, 0 errors | pass | 59, analysis_only | complete | incomplete | complete |
| H2 | 3,085 | pass, 0 errors | pass | 52, analysis_only | incomplete by correct causal fail-close | incomplete | incomplete |
| H3 | 2,687 | pass, 0 errors | pass | 63, analysis_only | complete | incomplete | incomplete |

Selected overlay roots:

- E1:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_resume_84f31fd_20260825/e1_trusted_proxy/`
- E2-E3 and M1-M3:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_resume_84f31fd_20260825/batch/`
- H1 and H3:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_resume_exact_e7c652e_20260825/`
- H2:
  `/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_resume_exact_63534a5_20260825/`

The selected nine Writer overlays used 81 completed `role=writer` calls,
786,808 reported tokens, and a ledger-estimated USD 10.11784. A discarded H1
draft used 9 calls / 86,244 tokens / USD 1.11596. One earlier E1 request failed
with HTTP 401 and was conservatively accounted as 171,615 tokens / USD 4.27615;
that reservation is not evidence of actual Provider billing. H2/H3 resume-plan
failures and the old-image kernel mismatch stopped before Provider calls.

## Published-anchor comparison: real remaining gaps

These are not implementation regressions that can be erased by adding prose:

- E1/E3: post-baseline ascertainment and early-event opportunity need a
  prospectively authorized landmark/descriptive decision; patient identity is
  also required for repeated-stay dependence.
- E2/M1: measurement-by-indication remains after transparent measurement-status
  auditing; adjustment authority and external reproduction remain open.
- M2: calibration, Brier score, DCA, and grouped patient-level repetitions are
  present, but no temporal/external validation or recalibration exists.
- M3: low resampling stability and low GMM-versus-k-means agreement correctly
  block biological naming; a second robustness axis and external replication
  are absent.
- H1: the severe proportional-hazards violation is detected and a constant HR
  is withheld; stronger informative-censoring/missingness sensitivity and
  external reproduction remain absent.
- H2: verified non-use/delayed comparator and positivity are unavailable. The
  correct result is a terminal fail-close with no propensity model, balance
  table, or treatment-effect estimate.
- H3: AIC and BIC minimize at the upper candidate boundary, so no K, phenotype
  name, or outcome association is authorized. Baseline characterization,
  alternative-algorithm agreement, and external replication are still absent.

Across all cases, independent clinical/methodological review and source-bound
novelty positioning remain open. Dev9 paper readiness is therefore 0/9 even
though development execution is 9/9 and substantive Writer output is 9/9.

## Verification

- Comparator and Provider/resume focused suite: 73 passed.
- Evidence-registration / sealed-envelope / resume suite: 95 passed.
- Ruff and `git diff --check`: pass.
- Resume authority fix commit: `63534a5b123ab715b29f7f3ec92c15d6263af247`.
- Artifact binding fix commit and current HEAD:
  `e7c652e96b76cb1b3fe2e7e9ce02f8155c93934a`.
- Current exact image:
  `easyicu-research-agent:e7c652e`, digest
  `sha256:30d2acf8cbba4eb0c0c031cc853902205aa5c80795f71dc2a1bb7c7d569073f0`.

Qualification12 and Held-out27 were not started during this remediation.
