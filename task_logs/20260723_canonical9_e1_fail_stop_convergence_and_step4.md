# Canonical9 E1 fail-stop, convergence, and Step 4 closure

Date: 2026-07-23 EDT
Task: `FIG2-CANONICAL9-REALRUN`
Scope: MIMIC-IV full0717-v2, aware arm, local Luna Provider

## Honest status

- Paper-facing Canonical9 remains **0/9** until a fresh source-bound run passes
  the full scorer. No diagnostic run is promoted.
- The latest diagnostic run is preserved at
  `/Volumes/外置硬盘/easyicu_data/canonical9_runs/batch_20260723_luna_miiv_adaptive_63c8711/e1_sepsis3_prevalence_mortality/aware/run_20260723T112040_81f957`.
- Steps 01--03 succeeded. Step 04 was sent to the Coder and failed before
  transport because its initial prompt was 43,097 bytes, above the 42,000-byte
  gate. The Paper workflow then incorrectly started Step 05 despite the failed
  required predecessor. The request was interrupted manually.
- That interrupted historical Step 05 receipt remains visibly pending inside
  the invalid diagnostic run. It is retained as failure evidence, not repaired
  in place or reused as paper authority.

## Framework fixes

1. Paper submission profiles are sequential and fail immediately after any
   non-`ok` required step. Later steps, transition callbacks, and replanning are
   suppressed, so a failed Coder step cannot spend a Writer/model call.
2. Initial-generation interruptions now terminalize the Provider transport
   receipt as failed even for `KeyboardInterrupt`/`SystemExit`, while preserving
   the original interruption.
3. The structured Step 04 contract
   `missingness_and_measurement_frequency_audit` with exact analysis-cohort
   input and two declared table products is owned by the deterministic
   missingness runner. It emits a concrete `measurement_availability.csv` and
   reads the manifest-selected revised plan rather than a stale original plan.
   The Coder prompt is eliminated rather than compressed close to the limit.
4. The deterministic penalized-convergence repair is versioned to
   `penalized_convergence_contract_v2`. It accepts only a boolean traced to the
   `success` field of a reviewed `scipy.optimize` result. A free variable,
   literal, iteration-count heuristic, or custom optimizer cannot be promoted
   to convergence authority. Historical v1 receipt ids remain parseable.
5. A Planner step declared as the primary adjusted-association analysis cannot
   carry a secondary-only typed model roster. It must contain at least one
   primary requirement; a proxy remains secondary instead of being relabelled.
6. Current Canonical9 input selection now binds `npj_dm/20260719`, matching the
   runtime default. The archived `20260718` profile remains immutable. The
   Figure 2 scorer-tree digest was reauthorized after the schema change.
7. The five legacy integration fixtures now use factory-registered built-in
   offline mocks. Unknown custom clients remain fail-closed.

## Exact data replay

The deterministic Step 04 implementation was replayed on the exact sealed
94,458-stay E1 cohort without a Provider call:

`/Volumes/外置硬盘/easyicu_data/e1_step4_replay.5IdbCM`

Result: `ok`; 21/21 declared inputs resolved; zero missing inputs; both declared
table products were materialized.

## Verification

- Integrated non-Docker matrix: **511 passed**.
- Figure 2 scoring-input authority after reauthorization: **35 passed**.
- Canonical9 typed selector `--check`: ready, 9 tasks,
  SHA-256 `7c1421ade83561d7727a8f6865cbbe99ffbe312437587064d64614bade793210`.
- Ruff, Black, Python compilation, `git diff --check`: passed.
- Architecture lower-is-better gate: exact baseline, no regression.
- Module graph: no new cycle.

The immutable image `easyicu-research-agent:source-5e567eb` was built from the
clean package source, with image digest
`sha256:04a0650bd576b02af6890a347dac6303fcf41cc7b43f0b78260cc9cf56fd2467`.
The full post-repair source-bound integration file then passed **13/13** with
its pytest temp root on the Colima-mounted external drive. The script-integrity
case now simulates a host-side post-execution digest change because the
production container correctly mounts the executable script read-only.

A fresh E1 run must create new execution identity and operator freeze evidence;
the interrupted batch must not be resumed.
