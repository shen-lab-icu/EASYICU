# Research-agent pre-experiment release freeze

Date: 2026-07-19
Branch: `refactor/agent-control-plane`
Code freeze commit: `c1032ae`

## Outcome

The pre-v1 research-agent architecture is frozen for fresh Canonical9
development runs. The release boundary is responsibility-based rather than a
cosmetic relocation of every public module:

- implementation ownership is split across explicit authority, execution,
  gates, repairs, planning, research-context, reporting, robustness, and
  orchestration packages;
- the 21 remaining top-level Python files are deliberate public entry points
  or cross-cutting contracts, not obsolete compatibility shims;
- 137 retired top-level implementation paths are locked absent;
- the module-top-level import graph has no drift and no reintroduced cycle;
- mock Planner/Coder fixtures cannot acquire submission authority under the
  additive `npj_dm/20260719` profile;
- publication labels are Planner-owned immutable run inputs, and shared
  renderers no longer infer case-specific clinical meanings from identifiers.

No Canonical9 run or archived experiment was modified during this freeze.

## Quantitative architecture gate

Compared with the committed decomposition baseline:

- `execution/phase.py`: 14,631 -> 11,126 lines (-3,505);
- `_execute_one_step`: 6,694 -> 6,058 lines (-636), direct nested functions
  26 -> 24, callable closure captures 37 -> 31;
- `run_execute_phase`: 8,451 -> 7,699 lines (-752), total nested functions
  48 -> 45;
- every lower-is-better metric is unchanged or improved;
- module-graph baseline diff exits successfully.

These measurements are structural. Runtime performance remains bound to the
real E3 Step02 acceptance result: one provider call, zero LLM repair, 26.8 s
active wall time. Fresh experiments must measure the new runs independently.

## Verification

- display/plan/parser/profile focused regression: **190 passed**;
- pipeline/resume/golden/meta milestone segment: **366 passed**;
- responsibility-package, retired-path, gate-order, and architecture segment:
  **442 passed**;
- Ruff, Black, compile/import, diff-check, module-graph diff, and architecture
  measurement gates passed.

## Freeze boundary and next action

Further movement of the remaining 21 public/cross-cutting modules is not a
pre-experiment requirement. Reopen shared-engine code only for a reproducible
general fail-open/fail-closed correctness defect, not to make one development
question positive or to reduce a directory file count.

Next, run fresh E3/H2/E2 development cases using the real-provider
`npj_dm/20260719` profile and `--arms aware`, then continue the remaining A
tasks. After A-task development is frozen, evaluate the independently sealed B
and C variants and 3-6 held-out full workflows without modifying the shared
engine from their results.
