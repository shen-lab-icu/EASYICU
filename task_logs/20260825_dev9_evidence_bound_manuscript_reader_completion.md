# Dev9 evidence-bound manuscript reader completion

Date: 2026-08-25 EDT  
Task: FIG2-DEV9-HELDOUT27 development writing and reader QA  
Implementation branch: `codex/writer-runtime-compat-20260825`  
Reader/export HEAD: `ecacf6b`  
Analysis/reporting replay HEAD and image: `1a0d53c` / `easyicu-research-agent:1a0d53c` (`sha256:e8743ec95022...`)

## Authority boundary

The nine items are development diagnostics. They remain `analysis_only`,
`publication_authorized=false`, and paper authority is 0/9. The checks below
validate execution completion, numeric traceability, deterministic manuscript
structure, and reader/export behavior. They do not establish clinical validity,
novelty, external validity, independent peer review, or submission readiness.

## Exact isolated replay

An isolated output root was created after another concurrent Planner branch
rewrote one status file in the earlier shared aggregate directory:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_reader_exact_1a0d53c_20260825/`

All nine saved plans were reused with replanning disabled. The isolated status
matrix is:

| Item | Completed / required | Numeric verified | Analysis validated | Writing audit | Exact code |
|---|---:|---:|---:|---:|---|
| E1 | 12/12 | true | true | pass | `1a0d53c` |
| E2 | 11/11 | true | true | pass | `1a0d53c` |
| E3 | 13/13 | true | true | pass | `1a0d53c` |
| M1 | 11/11 | true | true | pass | `1a0d53c` |
| M2 | 11/11 | true | true | pass | `1a0d53c` |
| M3 | 10/10 | true | true | pass | `1a0d53c` |
| H1 | 3/3 | true | true | pass | `1a0d53c` |
| H2 | 1/1 | true | true | pass | `1a0d53c` |
| H3 | 4/4 | true | true | pass | `1a0d53c` |

Aggregate provider-free writing audit:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_reader_exact_1a0d53c_20260825/quality_audit/summary.json`

It reports 9/9 pass, zero quality errors, and zero Provider calls. Every run has
zero numeric-audit errors. This is a development writing contract pass, not a
top-journal or publication pass.

## Reader and PDF repair

PDF visual inspection found two deterministic export defects that the first
quality replay did not expose:

1. An evidence-only `**Background:**` line could be misclassified as substantive
   because the next abstract label satisfied a cross-line whitespace regex.
2. Grouped Pandoc citations such as `[@source_a; @source_b]` were left as raw
   text in LaTeX/PDF.

Generic owner fixes were committed as:

- `5a4e569` — same-line, reader-visible abstract label validation and grouped
  citation rendering;
- `ecacf6b` — apply provider-free structural repair at the reader export entry.

The final reader packages are:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_writer_reader_exact_1a0d53c_20260825/readers_ecacf6b/`

Each item contains `manuscript_scaffold_bound.md`, `manuscript_provenance.json`,
LaTeX, BibTeX, PDF, PDF receipt, copied registered figures, and a build receipt.
All nine receipts report `provider_calls=0`, `claim_ceiling=analysis_only`, and
`publication_authorized=false`. Claim counts are E1 29, E2 31, E3 25, M1 35,
M2 14, M3 15, H1 7, H2 3, and H3 4. H2 correctly has zero result figures because
its comparator/causal contrast is not identified.

The corrected M2 PDF has a non-empty Background, renders grouped citations as
normal bibliography numbers, has no raw `[@key]` marker or final undefined
citation, and retains the draft-not-for-submission watermark.

## Web evidence preview

The Copilot preview owner supports:

- highlighted manuscript numbers that open the exact JSON field/pointer,
  analysis step, evidence ID, and SHA-256;
- digest-pinned read-only previews for registered code, aggregate tables, JSON,
  and metadata-only files;
- multiple closable evidence tabs inside the article preview;
- privacy withholding for patient rows, identifier-bearing tables, host paths,
  oversized files, and unsupported encodings;
- code display without execution.

Focused verification passed:

- manuscript/evidence Web contracts: `4 passed`;
- evidence-preview JS security: `3/3`;
- manuscript renderer JS security: `12/12`;
- writer/LaTeX focused tests: `35 passed, 1 deselected`;
- Ruff and `git diff --check`: pass.

A fresh local FastAPI server loaded the exact Guided Copilot static bundle and
completed a real-browser local-workflow smoke. No fake research run was inserted
to manufacture an evidence-click demonstration. The temporary QA project was
moved recoverably to
`~/.Trash/EasyICU_QA_guided-my-first-study-e3caa0_20260825` and the test server
was stopped.

## Writing-quality conclusion

The nine manuscripts are now technically readable and traceable, but not all
are top-journal prose. Remaining material limitations include generic abstract
language and repeated Results/Conclusions in E1/E3, temporal opportunity and
measurement-by-indication issues, missing external validation/recalibration for
M2, insufficient stability/transportability for M3/H3, informative censoring
for H1, and the absent H2 comparator. Literature-to-plan/novelty authority and
independent clinical/statistical review also remain open. These gaps must not be
closed by stylistic rewriting or by borrowing published effect estimates.

