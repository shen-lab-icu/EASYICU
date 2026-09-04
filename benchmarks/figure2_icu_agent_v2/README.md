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
arm entry points. Both are constructed through `FormalExecutionSession`, and
every Provider-bearing collaborator receives the same arm-bound
`FormalProviderSession`. That session owns call numbering, budget receipts,
hard-stop state, and the two-phase authority sequence: validate the signed
coordinate without consuming it, verify production transport trust, then
atomically consume the coordinate immediately before the one permitted
logical model turn. Every raw transport retry carries that same logical call
identifier while receiving its own pre-transport budget reservation and ledger
entry. A replay or a concurrent second coordinate consumer fails closed.
`PipelineServices` owns the complete list of Provider-bearing collaborators,
and `formal_collaborator_adapter.py` projects that list through formal
authority. An opaque prebuilt visual-QA adapter is rejected because its
internal transport cannot be proven governed; formal visual QA must supply its
client through `vlm_client`.

The EasyICU arm is projected into the shared seven-file contract only by
`easyicu_review_bundle_adapter.py`. Both arms produce the same closed
`TerminalOutcome` and exact `easyicu.figure2_resource_receipt/1` schema through
`review_bundle_semantics.py`; unknown statuses, failure categories, fields, or
invalid numeric and boolean values are rejected. `review_bundle_writer.py`
stages and verifies an exact bundle in a sibling directory, then publishes it
with one atomic rename; competing writers cannot mix files or remove the
winner. `review_bundle_normalizer.py` performs the arm-neutral reviewer
projection without repairing scientific content.

`formal_scheduler.py` reproduces all 78 core task-arm trajectories, creates the
post-unsealing Qualification12 assignment deterministically, rejects nonempty
per-site output roots, and issues single-use site-bound leases without Provider
access. The same `site_assignment_sha256` is bound into the dry run, lease,
runner, every launch receipt, and the signed atomic declaration; qualification
and core authority reject an unknown task, split pair, wrong site, missing arm,
or mismatched assignment digest before transport. The signed declaration also
binds the exact output root for each site; both qualification and core leases
must match the registered task, site, pair, global sequence, and site output
root. `formal_trajectory_lifecycle.py` validates the lease without consuming it,
derives the only permitted scratch workdir under the signed site root,
constructs the arm implementation, and only then commits the single-use lease.
Initialization may create only paths inside that owned workdir. A failed
attempt is moved to a unique `.trajectory-failed` quarantine path so the exact
task can be retried without hiding partial state; a pre-initialization failure
leaves the lease unconsumed. After the lease is committed, the same execution
session terminalizes unexpected runner, plan-review, executor, resource, or
projection failures into the neutral seven-file failure contract. The final
review bundle is written only to the exact leased output directory.
`multi_host_acceptance.py` accepts exactly one server and one laptop preflight
receipt as unparsed JSON bytes only when the frozen release, model route, input
set, budgets, container limits, and network policy match exactly; duplicate
keys, nonfinite values, a warning, drift, Provider access, or missing field are
a hard NO-GO. The normalizer also requires both exact host markers and both
formal output roots on every call, and hides generic POSIX/Windows paths and
registered site identifiers before scoring without interpreting clinical slash
units as paths. `blinded_evaluator.py` mechanically instantiates Heldout27
sheets from the frozen rubric and taskbank, normalizes both source bundles
through one `BlindedReviewPackage`, verifies every post-normalization file
digest, verifies both source receipts name the same frozen task as the review
sheet, and binds the exact pre/post digest maps, redaction-log digest, package
digest, and per-bundle digest into each eligible review and the final score
lock before unblinding. The lock is published from complete fsynced staging
bytes with atomic no-overwrite linking, so a failed write leaves no partial
final file and remains retryable. Reviewers consume the
package's sealed bytes rather than reopening mutable source paths.
`immutable_publication.py` owns that same staged, verified, no-overwrite,
file-and-directory-fsynced publication primitive for trajectory leases,
started markers, signed-coordinate consumption markers, and score locks. The
seven-file review bundle keeps a separate directory-level atomic transaction.
The same seam creates split-specific Heldout27 and Qualification12 sheets;
Qualification12 sheets bind the frozen `meta_generalization` row and its
`bound_result` or `fail_closed` contract so the preregistered blinding pilot
and binary reviewer-reliability gate can run before core launch.
`formal_release_identity.py` is the single owner of required
registration fields, implementation-owner paths, and registered source paths.
`formal_authority.py` verifies an Ed25519-signed atomic declaration, the exact
call coordinate, and the registered SHA-256 of every critical runner, gate,
producer, normalizer, scheduler, evaluator, and test owner before a transport
can be reached. Each accepted coordinate also creates one durable,
create-if-absent consumption marker beneath the signed output root, making
authorization single-use across processes rather than only within one Python
object. The current launch contract still denies every
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
