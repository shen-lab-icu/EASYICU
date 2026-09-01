# Figure 2 ICU Research Agent benchmark

## Current design status (v2.1 review candidate)

`experiment_protocol_v2_1.json` is the current review candidate. It is not an
external preregistration and grants no Provider-call, formal-batch, or paper
result authority. The tracked v1 owner remains historical and executable only
for its original development contract.

The v2.1 candidate adds:

- a competitive generic coding-agent arm with a frozen plan/execute/inspect/
  repair/finalize loop and a minimum Qualification12 competence floor, with no
  B-only model-turn cap and two sealed reserve qualification sets;
- an implementation-neutral review bundle, normalizer audit, blinding pilot,
  post-score arm-guess assessment, and a blinded receipt projection that hides
  architecture-diagnostic resource profiles until scoring is locked;
- one binary task-level primary endpoint: answering the frozen question as
  specified and being reportable without postrun repair;
- an estimation-first paired SAP with exact McNemar sensitivity calculations,
  explicit low-power language, and no continuous co-primary endpoint;
- an all-or-none six-database WP1 gate, an external implementation comparator
  on its five supported databases, a source-native SICdb audit, risk-tiered
  35/50-record semantic audits, and mandatory fresh re-audit after correction;
- a secondary Safety12 characterization with neutral dispositions, external
  rationale sources, and independent reviewer sign-off;
- a separate three-case Idea-to-Evidence demonstration that starts from
  externally sealed broad clinical briefs, retains every candidate and failed
  case, selects one idea before patient-level analysis, and follows it through
  deterministic feasibility, governed execution, EvidenceStore, and bounded
  manuscript text;
- an external preregistration plan and a formal launch contract that defaults
  to denial, including a post-registration Qualification12-only authorization
  scope before any core or paper-facing run;
- partially independent Heldout27 review: at least one scoring reviewer and
  every adjudicator are external to implementation and manuscript authorship,
  with conflicts and workload capacity sealed before qualification.

WP5 is deliberately not a third arm or an extension of the Heldout27
denominator. It has no comparator and supports only a descriptive multiple-case
workflow claim. Three externally authored briefs must all be shown; safe
nonlanding, workflow failure, null findings, and restricted reports cannot be
replaced by a more attractive case. Phase-A idea mining uses only declared
literature retrieval and nonpatient metadata. Each selected idea must then pass
a deterministic WP1 extension before phase-B patient-data analysis can begin.

Run the no-Provider design check with:

```bash
python -m pytest tests/benchmarks/figure2_icu_agent_v2/test_design_v2_1.py -q
```

`design_v2_1.validate_review_candidate_bundle()` validates asset digests,
task identity and coverage, schedule reproduction, rubric neutrality, safety
rationale coverage, exact power scenarios, the generic-arm floor, WP1 scope,
and fail-closed launch status. It cannot call a model or authorize a run.

Before any formal call, follow `preregistration_plan_v1.json`, satisfy every
receipt in `formal_launch_contract_v1.json`, and replace the internal review
candidate with an externally timestamped immutable package.

## Historical v1 owner

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
