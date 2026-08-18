# Figure 2 ICU Research Agent benchmark v2

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

`protocol.py` owns strict loading and validation of:

- `action_space_v1.json` — the 11-stage research workflow, owner boundaries,
  expected artifacts, and stable failure codes;
- `experiment_protocol_v1.json` — split identity, scoring dimensions,
  contamination firewall, and aware-only formal run policy;
- `heldout27_taskbank_v1.jsonl` — the complete item-level scientific contract.

Call `validate_experiment_bundle()` before any development or formal operation.
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
