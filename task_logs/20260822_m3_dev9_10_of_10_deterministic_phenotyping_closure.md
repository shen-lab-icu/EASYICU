# M3 Dev9 10/10 deterministic phenotyping closure

Date: 2026-08-22

## Outcome

- Task: `m3_sepsis_subphenotype` (Dev9 development diagnostic only).
- Final execution: required/completed `10/10`; missing steps `[]`; failed steps `[]`; `execution_complete=true`.
- Final run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_931fcc1_m3_checkpoint009_execution_r7_20260822/m3_sepsis_subphenotype/aware/run_20260822T171151_c43df1`.
- Exact clean Git HEAD: `931fcc11d5cdffce689302937754a67f26f851fb`.
- Exact runner image: `easyicu-research-agent:931fcc1`, digest `sha256:3f56bbd1b8a9ad9a406e49beca0617f8735f824c1cfd214ad6314ca40429f0bb`.
- Execution identity records `git_dirty=false`, Docker network policy `none`, and `paper_eligible=false`.
- Provider use in the successful replay: one Planner-labelled call, 20,200 prompt + 1,247 completion = 21,447 provider-reported tokens, estimated cost `$0.23941`; the hard one-attempt limit then correctly prevented Writer generation.
- Runtime: 112.10 seconds.

The task-defined plan has ten required steps, so the truthful completion count is `10/10`, not `11/11`. This is a Dev9 architecture execution success, not a paper-ready result. The run remains `development_diagnostic`, maturity `analysis_only`, `artifact_valid=false`, `scientific_requirement_complete=false`, and `paper_authorized=false`. The absent deterministic scientific validator, incomplete robustness replay, literature authority, complete manuscript, and independent review remain fail-closed.

## Checkpoint boundary

- Reused terminal Planner checkpoint: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_779699f_m3_checkpoint007_execution_r4_20260822/m3_sepsis_subphenotype/aware/run_20260822T170005_64512f/progressive_planner_checkpoint_009.json`.
- Exact file SHA-256 before and after replay: `8406ddbc82f563a25715d9504ebd3c0f439828523f2c58d9513d5d8b53e16cff`.
- Internal checkpoint SHA-256: `6da3b06cadcfee7307880f9fad4d8134b00cb2b79e86fe84268d173f7bd53591`.
- Outline SHA-256: `bccb867db2165d2054c914ea5f59787cecb3bbee17da1d103ee902499eba7001`.
- Planner outline, variable roster, step order, selection/rejection reasons, and checkpoint bytes were not changed by the owner-contract migrations.

## Generic owner-contract closure

No M3, sepsis, benchmark-item, or named-dataset branch was added. The reusable phenotyping path was closed through owner-local contracts:

- `779699f` added typed deterministic cross-sectional phenotype solution, cluster-number selection, cluster-stability, and source-bound multi-panel figure owners. The adapter excludes typed identifier, outcome, and time roles from unsupervised features and remains `analysis_only`.
- `51b6055` bound exact runtime action outputs to their published article roles.
- `c3d179c` allowed a digest-bound migrated checkpoint to recompile only the affected runtime schema under the current strict contract; ordinary checkpoints remain strict.
- `931fcc1` added explicit `source_table`, `source_step_id`, and source-row provenance to each independent phenotype figure source-data table, so the FigureContract verifier checks every typed parent against its own bytes rather than searching for a false shared key across unrelated tables.

Focused/adjacent verification covered 250-, 166-, 137-, and final 170-test slices across the changed owners. Ruff and `git diff --check` passed. The final deterministic figure step emitted exact profile, assignment, and stability source CSVs plus PNG/SVG/PDF/TIFF and a FigureContract. The promoted publication figure contract SHA-256 is `e43a7e58c0aee27cbcd4c270171193e076d6fc3df992144ddb24c181167419c8`; the PNG SHA-256 is `fee69f58d6a8bb85f95cd6a679698d36de6d2f83c5f70f5f59e005ace5cc58d7`.

## Development accounting boundary

The final one-call replay is separated from the planning/diagnostic work used to discover the generic gaps:

- Fresh planning and checkpoint continuation used 12 bounded Planner calls before checkpoint 007, then 2 further Planner calls to materialize the final two steps and produce checkpoint 009.
- The successful checkpoint-009 execution used one Planner-labelled call and no Coder call. Writer generation was blocked by the one-attempt stop-loss after execution had already reached `10/10`; this is why manuscript gates remain false without invalidating execution completeness.
- Earlier r1/r2/r3/r5 diagnostics made zero Provider calls; r4 used the 2 calls that completed checkpoint 009; r6 used one bounded call and exposed only the figure-lineage contract defect.

These are development costs, not formal benchmark accounting and not part of Held-out27.

## Input authority

- Input root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_cc90a62_m3_input_20260822`.
- JSONL SHA-256: `7f2478aa3ba19851827d50026b4793e807d4826e2a2d6ec16c6a09023ab0e110`.
- Cohort: 94,458 rows; development-only authority.
- Binding receipt: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_cc90a62_m3_input_20260822/development_binding_receipt.json`, SHA-256 `9fda1688cdd81cd39e760013d3f906c10ea257a2728021307fbfa16e4a8eecba`.

## Next action

Start H1 Dev9 only. Keep Qualification12, Held-out27, broad Web work, and full exact-head CI deferred. Continue the same efficiency route: verify the case protocol and input owner first, run one bounded development canary, fix only an attributable generic owner contract, and preserve all publication gates.
