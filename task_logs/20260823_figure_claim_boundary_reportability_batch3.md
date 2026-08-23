# Per-figure claim boundary + reportability binding (batch 3)

- Date: 2026-08-23
- Parent checkpoint: `66d828b7ae762146d0b521cfb4c6d126784c624c`
- Worktree: `/private/tmp/easyicu-merge-final`
- Scope: post-E1 architecture optimization only; no E1/E2/Qualification12/Held-out27 execution
- Provider calls / tokens / cost: `0 / 0 / 0`

## Outcome

The reporting owner now builds one typed
`easyicu.figure_claim_boundary_audit/1` from current FigureContracts and the
pre-result research-design selection. Every figure records its core supported
claim, cannot-prove boundary, figure role, tier, contract digest, plan digest,
and design-selection digest. Every panel independently records the claim it
supports and inherits the selected design's cannot-prove ceiling.

For a fresh selection-aware plan, every readable FigureContract must have a
core claim and complete panel claims before the boundary is marked `complete`.
If a primary result figure is present and its selected-design boundary is not
complete, the display/reportability audit fails closed. The persisted display
suite audit is now `easyicu.display_suite_audit/3` and includes the full
per-figure boundary packet.

Historical plans without research-design selection remain replayable. Their
figures receive an explicit `legacy_analysis_only` boundary stating that the
figure cannot authorize a manuscript claim beyond the exact registered
evidence and source data; `boundary_ready` remains false. The claim ceiling is
fixed at `analysis_only` for both routes.

## Verification

- Focused figure/reportability matrix: `208 passed, 1 warning`
  - selected-design and legacy boundary contracts
  - selected-design malformed primary boundary fails the display gate closed
  - display-suite integration
  - scientific maturity and write-phase boundary
  - publication figures and review artifacts
  - design-selection and Progressive Planner contracts
- Additional pipeline/display matrix: `60 passed, 290 deselected, 1 warning`
- Architecture gates: all 5 green
  - architecture ratchet: no lower-is-better metric regression
  - module graph: acyclic; intentional `577 -> 578` modules and `2321 -> 2325` edges
  - import contracts: 7 kept, 0 broken
  - Ruff: green
  - size/budget guards: `141 passed`
- `git diff --check`: green

## Claim boundary

This is a local architecture checkpoint, not a fresh E1 result, not full
exact-head CI, and not manuscript/benchmark readiness. No case-specific or
Sepsis-specific logic was introduced.
