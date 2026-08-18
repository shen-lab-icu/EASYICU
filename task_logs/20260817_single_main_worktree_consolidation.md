# Single-main-worktree consolidation

- Date: 2026-08-17
- Source: `integration/figure2-e1-h3-20260816@7ac301d`
- Pre-merge main: `063a5e4`
- Merge commit: `4ebcddb`

## Result

The complete Figure 2 development history was merged into `main`. The source
worktree and its local and remote branch were removed only after `7ac301d` was
verified as an ancestor of the pushed `main` merge.

The repository now has one worktree and one development branch:

- `/Users/haibo/Documents/GitHub/EASYICU`
- `main`

The earlier prediction V5.1 maturity ceiling remains `analysis_only`; branch
consolidation does not promote Planner, production, formal benchmark, or paper
authority. No E1 canary was run during this Git operation.

## Preserved local change

`tools/run_analysis_bench_overnight.py` had a real two-line model-default change
hidden by Git `skip-worktree`. The hidden state was removed and the change was
committed as `063a5e4` before merging. Conflict resolution kept the Figure 2
multi-provider discovery logic while changing the OpenRouter fallback to
`openai/gpt-oss-120b:free`.

The `pytest.ini` conflict kept the `main` development policy: slow tests are
declared once and excluded from bare local pytest runs; checkpoint jobs cancel
the filter explicitly.

## Verification

- Conflict-file Ruff and Python compilation: passed.
- Repository contract, provider, CLI, E1 model-grid, prediction owner/evidence/
  persisted-runtime/provenance, and capability governance: `224 passed`.
- Capability inventory audit: passed.
- Module graph: `547 modules / 2,177 edges / 0 cyclic SCCs`.
- `main` push: successful.

This is a development consolidation, not an E1 scientific completion receipt or
a formal Held-out27 release checkpoint.
