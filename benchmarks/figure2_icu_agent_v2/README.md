# Figure 2 ICU Research Agent benchmark v2

## 2026-09-01 design freeze

The prospective paper design is frozen in `experiment_protocol_v2.json` and
`design_freeze_manifest_v2.json`.  It replaces the future formal design in the
historical aware-only v1 protocol without modifying or reinterpreting any prior
development evidence.  The frozen paper program separates three questions:

1. deterministic six-database substrate validation without LLM trajectories;
2. a paired 27-task comparison of `easyicu_full` with a matched
   `generic_code_agent` using the same model and scientifically equivalent data;
3. a paired 12-task challenge set for prespecified fail-closed behavior.

The core design contains 78 formal runs.  A prespecified nine-task repeatability
stage adds 36 runs only after the core batch.  The primary endpoint is binary
task-level `reportable_without_postrun_repair`; rubric subitems and artifacts are
not independent sample-size units.

This is a **design freeze, not a formal-run freeze**.  Provider, Planner, formal
batch, and paper-result authority remain false until fresh inputs, Safety12
fixtures, dual human sign-off, exact clean code/image/model/provider identities,
budgets, network policy, CI, and the atomic batch declaration are sealed.  Run:

```bash
python -c "from benchmarks.figure2_icu_agent_v2.freeze_v2 import validate_design_freeze; print(validate_design_freeze())"
```

The v2 design assets are:

- `heldout27_evaluation_rubric_v1.json` — implementation-neutral primary and
  secondary evaluation contract;
- `statistical_analysis_plan_v1.json` — paired analysis, denominator, retry,
  reliability, and interpretation rules;
- `formal_safety12_taskbank_v1.jsonl` and `formal_safety12_rubric_v1.json` —
  twelve distinct end-to-end challenge categories;
- `data_platform_validation_protocol_v1.json` — deterministic data-foundation
  validation, kept outside the LLM A/B denominator;
- `freeze_v2.py` — fail-closed identity, digest, schedule, and authority
  validator.

This directory is the versioned experiment owner for the Biomni-aligned EasyICU
evaluation design:

- **Dev9**: the existing E1–H3 questions. They may expose architecture defects
  and drive general fixes, but their results are development-only.
- **Qualification12**: the existing `meta_generalization` probes. They exercise
  off-canonical generalization and safe failure, but have already informed
  development and therefore cannot enter the primary denominator.
- **Held-out27**: 27 distinct ICU research questions spanning six databases,
  six study-design families, and nine basic/intermediate/advanced tasks each.
  These are eligible for the primary Figure 2 experiment only after the entire
  execution and evaluation environment is frozen.

The tracked held-out taskbank is evaluator material, not a secret answer key.
"Held out" means that it is never used for architecture repair, shared-prompt
tuning, method selection, canaries, or result-based reruns. At runtime the Agent
receives only the current item. The taskbank contains no expected numeric result
or effect direction.

## Owner and public contract

`protocol.py` continues to own strict loading and validation of the historical
v1 bundle:

- `action_space_v1.json` — the 11-stage research workflow, owner boundaries,
  expected artifacts, and stable failure codes;
- `experiment_protocol_v1.json` — split identity, scoring dimensions,
  contamination firewall, and the superseded aware-only formal run policy;
- `heldout27_taskbank_v1.jsonl` — the complete item-level scientific contract.

Call `validate_experiment_bundle()` for historical v1 development operations and
`validate_design_freeze()` for the prospective v2 design.  Neither receipt
grants Provider or formal-batch authority.
It fails closed on digest, path, task identity/order, stage coverage, database,
analysis-family, difficulty, or prompt-leakage drift.

`tools/audit_figure2_icu_agent_v2_readiness.py` separately compiles a per-task
development receipt from the production scientific-action catalog, production
concept catalog, and Parquet footer schemas in an explicitly supplied full6
development vintage. A task can be `development_ready` while remaining
`formal_ready=false`; the receipt always records the outstanding native-v2
input, clinical-review, methods-review, and environment-freeze gates.

## Authority boundary

Passing the bundle validator does **not** authorize a Provider call, patient-data
load, paper claim, or formal batch. Formal authority additionally requires:

1. exact code/image/model/provider/execution-policy identity;
2. fresh native-v2 input artifacts and per-task typed input receipts;
3. frozen evaluator/rubric and blinded-review package;
4. current clinical and methods sign-off;
5. one atomic batch declaration that forbids resume, reuse, memory, development
   sampling, posthoc retries, and result-driven changes.

The original `benchmarks/figure2_canonical9/` tree remains unchanged and is the
historical Dev9 authority. Do not overwrite its frozen artifacts to implement
this experiment.
