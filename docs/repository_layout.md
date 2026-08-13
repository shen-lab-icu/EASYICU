# Repository layout and ownership

EasyICU keeps one owner for each kind of repository content.  This document is
the durable boundary; `tools/audit_repository_hygiene.py` enforces the parts
that can be checked mechanically.

| Path | Owner and allowed content |
|---|---|
| `src/easyicu/` | Installable product code and package data only. Never run output. |
| `tests/` | Unit, contract, integration, and browser-contract tests. |
| `tools/` | Maintainer and experiment launchers; no reusable product policy. |
| `scripts/` | Small operator entry scripts. |
| `benchmarks/` | The only benchmark owner: cases, catalogs, evaluation fixtures, and frozen formal suites. |
| `docs/` | Maintainer documentation, reviews, QA records, and checked-in evidence summaries. |
| `examples/` | Curated, runnable onboarding examples. |
| `baselines/` | Checked-in comparison baselines. |
| `sources/` | Source definitions and source-facing metadata. |

Local-only directories such as `.venv/`, `.codegraph/`, `output/`,
`research_output/`, `task_logs/`, and tool caches are not source-of-truth.
They may exist in a working copy, but their generated payloads must stay
ignored.  CodeGraph keeps only `.codegraph/.gitignore` in Git.

## Rules

1. Do not recreate a singular top-level `benchmark/`; use a typed owner under
   `benchmarks/`.
2. Do not place QA notes or ad-hoc scripts at the repository root.  QA records
   belong in `docs/qa/`; reusable maintenance code belongs in `tools/`.
3. Do not write generated artifacts beneath `src/`.
4. A capability with no production caller must be registered in
   `docs/research_agent_capability_inventory.md` as awaiting wiring, optional,
   support surface, compatibility surface, or CLI entry point before it is
   kept.  Static zero-reference counts alone never justify deletion.
5. Generated build trees and stale canaries should be moved to a dated,
   recoverable workspace cleanup directory before permanent deletion.
