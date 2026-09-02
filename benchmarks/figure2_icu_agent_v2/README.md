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
- a separate Idea-to-Evidence flagship case study that permits bounded
  human-Agent generate/critique/revise cycles, retains an append-only discovery
  trace, locks one purposively selected feasible idea before patient-level
  outcome analysis, and follows it through governed execution, EvidenceStore,
  and bounded manuscript text;
- an external preregistration plan and a formal launch contract that defaults
  to denial, including a post-registration Qualification12-only authorization
  scope before any core or paper-facing run;
- a two-host execution-acceptance contract for one server and one laptop: work
  is split by complete task pair, both arms stay on the same host and run in
  frozen order, and no more than one trajectory is active per host;
- partially independent Heldout27 review: at least one scoring reviewer and
  every adjudicator are external to implementation and manuscript authorship,
  with conflicts and workload capacity sealed before qualification.

WP5 is deliberately not a third arm or an extension of the Heldout27
denominator. It has no comparator and supports only a descriptive, purposively
selected flagship workflow claim. Phase A may iterate over candidate ideas with
human clinical and methods feedback, declared literature retrieval, nonpatient
metadata, and neutral deterministic feasibility receipts. Every candidate
version and rejection reason remains visible. The final idea and plan are
locked only after feasibility passes and before patient-level outcome analysis.
After that lock, a failed, null, or restricted analysis cannot be replaced by a
more attractive candidate or result.
Once the phase-B declaration is signed, the registered terminal disposition is
reported even if it is `safe_nonlanding` or `workflow_failure`; WP5 cannot be
withdrawn from the manuscript because it did not land.

Run the no-Provider design check with:

```bash
python -m pytest \
  tests/benchmarks/figure2_icu_agent_v2 -q
```

`design_v2_1.validate_review_candidate_bundle()` validates asset digests,
task identity and coverage, schedule reproduction, rubric neutrality, safety
rationale coverage, exact power scenarios, the generic-arm floor, WP1 scope,
and fail-closed launch status. It cannot call a model or authorize a run.

`generic_code_agent_harness.py` implements the frozen generic baseline loop
and adapts the existing isolated DockerRunner to Python and in-container shell.
`formal_generic_runner.py` and `formal_easyicu_runner.py` are the two formal
arm entry points; both route every model turn through
`formal_provider_gate.py` and the same durable budget ledger. The EasyICU arm
is projected into the shared seven-file contract only by
`easyicu_review_bundle_adapter.py`; both producers use
`review_bundle_semantics.py`, and `review_bundle_normalizer.py` performs the
arm-neutral reviewer projection without repairing scientific content.

`formal_scheduler.py` reproduces all 78 core task-arm trajectories, creates the
post-unsealing Qualification12 assignment deterministically, rejects nonempty
per-site output roots, and issues single-use site-bound leases without Provider
access. A formal runner requires the matching lease before construction.
`multi_host_acceptance.py` accepts exactly one server and one laptop preflight
receipt only when the frozen release, model route, input set, budgets, container
limits, and network policy match exactly; a warning, drift, Provider access, or
missing field is a hard NO-GO. `blinded_evaluator.py`
mechanically instantiates Heldout27 sheets from the frozen rubric and taskbank,
then atomically locks two eligible reviewers' scores and arm guesses before
unblinding. `formal_authority.py` verifies an Ed25519-signed atomic declaration,
the exact call coordinate, and the registered SHA-256 of every critical runner,
gate, producer, normalizer, scheduler, evaluator, and test owner before a
transport can be reached. The current launch contract still denies every
Provider call because this review candidate intentionally has no registered
signer key; offline test keys grant no qualification or formal-run authority.

Before any formal call, follow `preregistration_plan_v1.json`, satisfy every
conjunctive gate in `execution_acceptance_contract_v1.json` and every receipt in
`formal_launch_contract_v1.json`, and replace the internal review candidate
with an externally timestamped immutable package. Scientific A/B outcomes may
never decide whether to launch, continue, migrate, exclude, or rerun work.

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
