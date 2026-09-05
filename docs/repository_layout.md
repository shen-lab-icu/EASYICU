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
| `docs/` | User guides, maintainer documentation, reviews, QA records, and checked-in evidence summaries. |
| `desktop/` | Desktop shell, icons, dependency locks, and installer build scripts. |
| `examples/` | Curated, runnable onboarding examples. |
| `baselines/` | Checked-in comparison baselines. |
| `sources/` | Source definitions and source-facing metadata. |

Local-only directories such as `.venv/`, `.codegraph/`, `output/`,
`research_output/`, `task_logs/`, and tool caches stay out of the public Git
tree. Run outputs and receipts can be irreplaceable evidence: being ignored
does not make them disposable. Check ownership, active processes, and recovery
requirements before moving them. CodeGraph keeps only `.codegraph/.gitignore`
in Git.

## What users receive

| Delivery | Contents |
|---|---|
| Git checkout | Product source, tests and fixtures, tools, examples, public documentation, and reproducibility material. |
| Source distribution (`sdist`) | The tracked source and verification material, with private and generated paths excluded. |
| Python wheel | Installable product modules, dictionaries, templates, frontend resources, and required notices. Tests and repository tools remain outside it. |
| Desktop app | The installed product, frozen Python dependencies, Node runtime, native shell, and required notices. Users need no development checkout. |

Build the original sdist from a clean Git checkout. The setuptools-scm file
finder carries tracked support files into the archive; the archive test also
collects representative tests from that extracted tree. New public guides,
scripts, and fixtures must not require `git add -f`.

Keep new private documentation in `docs/_internal/` and local scripts in
`scripts/_local/`. Existing private files have explicit exclusions until their
owners choose to migrate them. See [installation.md](installation.md) for the
user entrypoints and [../desktop/README.md](../desktop/README.md) for build inputs.

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
6. Every Python file directly under `research_agent/` must have a justified
   public/shared/frozen owner in
   `tools/arch_baselines/research_agent_top_level_ownership.json`. New private
   implementation belongs in a responsibility package, not at the top level.
