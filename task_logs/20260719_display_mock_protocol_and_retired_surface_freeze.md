# Research-agent display/mock protocol and retired-surface freeze

Date: 2026-07-19
Branch: `refactor/agent-control-plane`

## Outcome

This pre-release bundle closes the two remaining places where test/demo or
case-shaped presentation behavior could acquire paper-facing authority.

1. The additive submission profile `npj_dm/20260719` requires a real provider,
   disables deterministic Planner/Coder fallbacks, keeps cross-run memory off,
   and preserves every older profile's serialized replay contract unchanged.
   A mock aware-arm remains available only as an explicitly profile-less
   plumbing smoke test.
2. Human-facing figure labels are now declared by the Planner/run in
   `AnalysisPlan.display_labels`. Publication renderers consume those exact
   labels. When no label is declared, the fallback only title-cases the raw
   identifier; for example, `death` becomes `Death`, never an invented ICU,
   hospital, or fixed-day mortality endpoint.
3. Display-label authority is included in the immutable plan-level signature.
   Once a step has completed, replanning restores the original display labels
   together with the scientific scope so sealed evidence cannot be shown under
   a newly substituted endpoint label.

## Case-neutrality and scientific ownership

- The old shared maps for SOFA/KDIGO/Sepsis, lactate/temperature abbreviations,
  robustness variants, and `death -> ICU mortality` were removed from
  `figures/skill.py`.
- The Agent still owns exposure, outcome, cohort, method, estimand, robustness
  specifications, and their human-facing descriptions. The renderer only
  formats the already declared objects.
- No Canonical9 item, database, or manuscript figure ID was added to a shared
  prompt or execution route.

## Retired-surface inventory

The Finder view that previously showed more than 150 flat implementation and
shim files is no longer the current architecture:

- top-level `research_agent/*.py`: **21** files;
- total Python modules across responsibility packages: **228**;
- explicitly retired pre-v1 top-level modules locked absent by
  `test_retired_top_level_imports.py`: **137**;
- retired modules with no canonical replacement (dormant primary runners and
  obsolete state/migration surfaces): **11**;
- tiny files under 700 bytes are package `__init__.py` files or small protocols,
  not compatibility shims at the top level;
- the module-graph baseline reports no drift and no reintroduced import cycle.

The 21 remaining top-level modules are deliberate public/orchestration or
cross-cutting contracts (`pipeline`, `schema`, CLI/MCP, ICU rules, benchmark,
literature/viability/evaluation, and small shared utilities). Physically moving
them again would not remove a responsibility boundary and is not part of the
release freeze.

## Verification

- submission-profile/orchestration/dictionary focused suite: 55 passed;
- display/plan/parser/profile focused suite: 145 passed after the new protocol;
- pipeline + resume + golden + meta segmented milestone: **366 passed** in
  387.92 seconds;
- `ruff` clean, `black` clean, `git diff --check` clean;
- `tools/research_agent_module_graph.py --diff` exited 0.

No Canonical9 experiment was run and no archived run was modified.

## Remaining release action

Run the final responsibility-package boundary segments and architecture gates,
record the clean release-freeze commit, then start fresh Canonical9 development
runs with `npj_dm/20260719` and a real provider. Further physical reshuffling is
post-freeze work unless a release gate identifies a concrete ownership defect.
