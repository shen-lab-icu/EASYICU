# Repository layout and hygiene remediation

Date: 2026-08-13 EDT

Branch: `fix/pi-workspace-review-20260809`

Task: `AGENT-REPOSITORY-HYGIENE-20260813`

## Decision

The repository was not globally disordered: the research-agent package graph
has no dependency cycle, and many apparently isolated modules are public CLI,
Web, plugin, compatibility, or frozen benchmark surfaces.  The remediation
therefore removed duplicate ownership and generated clutter without deleting
code merely because an internal import graph showed zero inbound edges.

## Changes

1. Consolidated the competing `benchmark/` and `benchmarks/` roots.  Cases now
   live under `benchmarks/cases/`, catalogs under `benchmarks/catalogs/`, and
   idea-mining benchmark assets under `benchmarks/idea_mining/`.  Production
   imports, tools, tests, and documentation were updated together.
2. Moved the tracked root design-QA note to `docs/qa/` and the workspace root
   figure/input and Playwright output to their owned `outputs/` locations.
3. Added `docs/repository_layout.md`, a machine-readable top-level
   research-agent ownership manifest, and `tools/audit_repository_hygiene.py`.
   A competing benchmark root, tracked build/cache output, or unowned top-level
   agent module now fails locally.
4. Added `tools/audit_capability_inventory.py`.  Every zero-inbound
   non-initializer module must now have an explicit external-consumer,
   entry-point, compatibility, owner, retirement, or review decision in
   `docs/research_agent_capability_inventory.md`.
5. Kept `evaluation_scorecard.py` and `icu_agent_bench.py` at their existing
   top-level paths because frozen Canonical9 scorer/rubric bundles bind those
   exact paths and hashes.  Moving them would change scientific authority.
6. Restored and versioned every recoverable task log referenced by current
   project dashboards.  Five historical Web logs that never existed in Git or
   the workspace were not fabricated; they are listed in
   `docs/evidence/historical_task_log_gaps_20260813.md` and require rerun before
   being used as evidence.  `tools/lint_progress.py` now rejects missing current
   task-log pointers.
7. Reduced the agent, benchmark, and Web CURRENT dashboards from 62,820,
   58,272, and 63,715 bytes to 4,109, 3,583, and 4,094 bytes.  The old pages are
   preserved byte-for-byte under each module's `history/` directory and linked
   from both CURRENT and HISTORY.
8. Moved approximately 3 GiB of old build, distribution, canary scratch, and
   local generated data into the recoverable workspace quarantine
   `_cleanup_20260813/`.  Removed cache-only `.DS_Store`, pytest, Ruff, mypy,
   import-linter, and Playwright state using the macOS Trash.  Active outputs,
   `.venv`, CodeGraph, Pi runtime dependencies, and Claude's in-flight
   `scratch_check_*.py` files were preserved.

## Commits

- `e743498 refactor(repo): consolidate benchmark ownership`
- `3292d9d chore(agent): govern module ownership and latent capabilities`
- `0653794 chore(repo): make current evidence reproducible`

## Focused verification

- Benchmark/catalog/idea-mining focused tests: `76 passed, 2 skipped`.
- Repository hygiene, progress evidence, and capability inventory tests:
  `7 passed`.
- `tools/audit_repository_hygiene.py`: `repository hygiene: OK`.
- `tools/audit_capability_inventory.py`:
  `research-agent capability inventory: OK`.
- `tools/lint_progress.py`: six dashboards pass with zero warnings.
- Ruff on all new/changed governance Python files: passed.
- CodeGraph was used before direct code search to inspect the benchmark and
  research-agent dependency/call paths; the resulting source projection shows
  the current canonical benchmark suite under `benchmarks/figure2_canonical9/`.

## Deliberately not run

No full exact-head CI matrix was started.  This is an E1 development iteration,
so validation stayed at the directly affected unit/contract and minimum
adjacent tests.  Full exact-head CI remains reserved for E1 11/11 freeze,
merge, release, or formal experiment preparation.
