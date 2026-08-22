# M2 Dev9 11/11 deterministic prediction closure

Date: 2026-08-22

## Outcome

- Task: `m2_mortality_prediction` (Dev9 development diagnostic only).
- Final execution: required/completed `11/11`; missing steps `[]`; failed steps `[]`; `execution_ok=true`.
- Final run: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_41b22d1_m2_checkpoint010_execution_20260822/m2_mortality_prediction/aware/run_20260822T155734_80a1a2`.
- Exact clean Git HEAD: `41b22d15bf030ee6d4fb5946382caf9c87029ea5`.
- Exact runner image: `easyicu-research-agent:41b22d1`, digest `sha256:aac84bdda61ab43e0d82840ef793343fb9af27f50513004811f60b063a47c93c`.
- Final execution identity records `git_dirty=false`, Docker network policy `none`, and the final repair ledger is empty.
- Provider use in the successful execution-only replay: Writer only, 1 completed call, 5,723 prompt + 932 completion = 6,655 provider-reported tokens, estimated cost `$0.08519`. Planner and Coder calls were zero.
- Runtime: 105.60 seconds.

This is a Dev9 architecture execution success, not a paper-ready result. The run remains `development_diagnostic`, maturity `analysis_only`, `artifact_valid=false`, `scientific_requirement_complete=false`, and `paper_authorized=false`. The absence of a registered deterministic scientific validator, current direct-comparator literature authority, a complete manuscript, and independent review is preserved as a fail-closed boundary.

## Fresh-plan and checkpoint boundary

The accepted plan was produced in bounded segments and then reused immutably:

- Final Planner checkpoint: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_fc6699d_m2_checkpoint005_planfinish_20260822/m2_mortality_prediction/aware/run_20260822T154303_1b628c/progressive_planner_checkpoint_010.json`.
- Checkpoint SHA-256: `2cc4900294521cb2038c01fe659e9dc411db38ac0e8aac8c50cc04ade0cbaca9`.
- Accepted outline: 9 steps; compiler-owned required execution path: 11 steps after adding the cohort-accounting and data-quality figures.
- The exact checkpoint digest remained unchanged for the final execution-only replay.

This evidence is a segmented fresh plan plus exact checkpoint execution, not one uninterrupted fresh run.

## Generic owner-contract closure

No M2, mortality, death, MIMIC-IV, or benchmark-item-specific branch was added. Four generic owner fixes closed the reusable prediction-family path:

- `e8a6cbd` added typed prediction runtime products, a deterministic prediction model/validation/calibration/decision adapter, and a deterministic four-table prediction figure renderer.
- `fc6699d` made the host bind only uniquely implied exact runtime-product dependencies; it does not infer case-specific scientific choices.
- `32bc50b` allowed execution evidence to resolve a verified run-bound cohort authority and its parent patient-group derivation, without guessing identifiers from strings.
- `41b22d1` made the prediction figure owner accept the exact typed input set independent of Planner ordering, eliminating the Coder fallback while retaining exact count and membership checks.

Focused and adjacent verification covered 198, 38, 175, 154, 257, and 21-test slices across the changed owners; Ruff and `git diff --check` passed. A real 94,458-row execution oracle also passed with development/validation patient groups disjoint; its metric values remain oracle-only and are not manuscript findings.

The final run-level publication FigureContract SHA-256 is `ff4c039d5f9897f3d35a752ed6166c613414d4c6b297036a39bd89d8573bfac1`; final publication PNG SHA-256 is `bf6bf250fc47d68da4cc00968e3025b7f98072af023d1f936ad1615880d4134b`. Publication figure stems are `cohort_flow`, `data_quality`, `easyicu_publication_figure`, and `prediction_figure_suite`.

## Provider accounting boundary

The final successful replay accounting above is intentionally separated from development diagnostics:

- The final plan-completion root used 5 Planner calls (45,199 tokens, `$0.51133`) and 2 Repair calls (16,950 tokens, `$0.20468`).
- One completed Coder diagnostic used 14,436 tokens and `$0.26422` before the generic figure-input-order defect was fixed.
- One interrupted Coder request has provider usage unknown. Its durable reservation is a worst-case guardrail, not reported or charged usage, and is not added to successful-call totals.
- Earlier failed planning attempts include a 3-call 31,514-token / `$0.41878` semantic-dependency diagnostic and a 6-call 65,282-token / `$0.74754` bounded strict-planning segment, plus local zero-call preflight failures and one HTTP 401 with unknown usage.

These costs are development evidence for finding reusable owner defects. They are not part of the 1-call final execution replay and must not be presented as formal benchmark accounting.

## Input authority

- Input root: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_9391248_m2_input_20260822`.
- JSONL SHA-256: `508751b7e09d09fd4382de909dd013f8559fff3d269ad1fc350372e24c01ba81`.
- Cohort: 94,458 rows; materialization authority remains development-only and is not paper authority.
- Development binding receipt: `/Volumes/外置硬盘/easyicu_data/figure2_dev9_9391248_m2_input_20260822/development_binding_receipt.json`.

## Next action

Start M3 Dev9 only. Keep Qualification12, Held-out27, expert scoring, broad Web work, and full exact-head CI deferred. Reuse the same sequence: zero-Provider preflight, bounded fresh plan, immutable checkpoint reuse at the efficiency boundary, owner-local generic repair, focused tests, and one final exact-image execution.
