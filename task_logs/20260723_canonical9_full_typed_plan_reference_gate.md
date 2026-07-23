# Canonical9 typed plan reference gate and E1 real-run diagnostic

Date: 2026-07-23 EDT  
Branch: `refactor/agent-control-plane`  
Implementation commit: `8a37ec3c779342ec14635c9a8b47a8d2b15438b9`

## Outcome

The first Luna/MIMIC-IV run on `423be9c` reached the real LangGraph execute
phase and exposed a second Planner binding gap. The cohort predicate was valid,
but the plan used the semantic label `sepsis3` as an executable dataframe field
while the sealed physical column was `sep3_sofa2_max`.

The diagnostic run was stopped during E1 step 3 of 6 before another full code
rewrite. It is not paper evidence, and E2 was not started:

`/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_423be9c/e1_sepsis3_prevalence_mortality/aware/run_20260723T084704_2cb160`

## General fix

`8a37ec3` expands the typed Planner boundary from cohort predicates to every
Planner-owned executable raw reference:

- `steps[*].inputs`;
- model requirement outcome and exposure fields;
- robustness missingness variables and audit flags;
- robustness outcome column/time fields.

Raw fields must copy an exact sealed cohort column. Typed `kind:name` products
remain artifact-graph references. Table 1 fields remain covered because the
schema requires every Table 1 group/row variable to be an explicit step input.
Legacy untyped contexts retain their previous behavior.

Structured-retry feedback groups repeated invalid names by location category so
the actionable diagnosis fits within the retry transport's 400-character error
projection. For the archived E1 plan it reports one invalid name, `sepsis3`,
across four step inputs, two model exposures, and two robustness missingness
variables. The original Planner prompt was independently checked to contain
`sep3_sofa2_max`.

## Verification

- Archived invalid E1 plan: rejected before execution.
- Same archived plan with every executable `sepsis3` reference replaced by
  `sep3_sofa2_max`: passed the expanded gate.
- Focused parser/cohort matrix: `58 passed`.
- Typed/context/Table 1/outbound/trajectory matrix: `166 passed`.
- Expanded execution/model matrix: `325 passed, 1 deselected`.
- Planner/Provider adjacency: `122 passed`; two baseline-existing failures were
  reproduced separately and are not attributed to this patch.
- Ruff, Black, `py_compile`, and `git diff --check`: passed.
- Resource/context baseline: passed after recording the intentional 543-byte
  Planner contract increase; maximum Planner request remains 38,344/80,000
  bytes.
- Architecture baseline and zero-cycle module graph: passed.

The deselected primary-model assertion and the other reported red tests were
reproduced on clean `423be9c`; one conflicts with the newer execution-phase
authority assertion, and the scripts factory static gate was already recorded.
They were not concealed as regressions from this increment.

## Data-lane scheduling

The remaining native-v2 database extraction policy is now fixed:

- eICU: 200,859 stays in exactly three batches of 66,953;
- MIMIC-III, AmsterdamUMCdb, HiRID, and SICdb: one full-cohort batch each.

An eICU launch using external output and temporary roots was stopped when the
child reached about 2 GB RSS while macOS already held roughly 17 GB swap on the
internal disk. No incomplete directory is accepted as an export package.
Extraction must run separately from the memory-intensive real Canonical9
process and continue to keep output, spill, and temporary files on the external
disk.

## Next action

Build an immutable image from the new clean HEAD, regenerate execution identity
and operator-freeze authority in a new batch root, then rerun E1 with the aware
arm only. Continue the remaining eight tasks only after E1's plan contains
sealed physical columns and its deterministic gates pass.
