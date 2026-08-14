# Research-agent layer ownership: authority / gates / audits / repairs / contracts

**What this is.** The owner contract for the four governance subpackages plus
`contracts/`. These layers look interchangeable from the outside ("validation
code"), but each owns a distinct question and a distinct moment in the run.
New code goes in exactly one of them; when a change seems to belong in two,
the usual answer is that one side should *call* the other, not grow a copy.

**What this is not.** It is not a module catalog (see
`docs/research_agent_capability_inventory.md` for reachability status) and not
a dependency audit (see `tools/research_agent_module_graph.py`).

## The one-line version

| layer | owns the question | moment | typical artifact |
| --- | --- | --- | --- |
| `contracts/` | "what shape is the data?" | always (types) | Pydantic / typed schemas |
| `authority/` | "who is allowed to decide, and what does the decision bind?" | before + during execution | evidence records, receipts, locks |
| `gates/` | "may this artifact/step proceed?" | admission checkpoints | `ValidationFinding` verdicts |
| `audits/` | "is the produced artifact/claim consistent with the evidence?" | after production | validator findings, claim audits |
| `repairs/` | "how do we fix what a gate/audit rejected?" | after rejection | patches, `RepairLedger` entries |

## contracts/

Typed data shapes shared across layers: `result_envelope`, `declared_product`,
`runtime` phase results, method/figure/table-one contracts. Rules:

1. No I/O, no LLM calls, no run state — shapes and pure helpers only.
2. If a module needs to *enforce* its contract, that enforcement belongs in
   `gates/` (admission) or `audits/` (post-hoc), importing the shape from here.

## authority/

Stateful runtime ownership: `EvidenceStore` (hash-register-bind),
plan lifecycle and review authority, run locks and heartbeats, execution-input
identity, provider budgets and hard stops, typed input receipts, provenance
and secret redaction, step capsules and attempt bookkeeping. Rules:

1. Authority modules *fail closed*: when identity/provenance cannot be
   established they raise, they never degrade to a warning.
2. They do not decide scientific validity (that is `gates/` + `audits/`);
   they decide *binding* — what a decision, artifact, or resume is bound to.
3. A new receipt/lock/budget always lands here, never beside the consumer
   that motivated it.

## gates/

Admission checks consulted *before or during* execution: `preflight`,
`typed_input`/`typed_schema`/`typed_binding_identity`, `contract`,
`semantics`, `method_compatibility`, `plausibility_obligation`,
`figure_egress`/`figure_privacy`, `visual_qa`. Rules:

1. A gate is a pure verdict: consume typed inputs, emit `ValidationFinding`s
   (or raise). No repairs, no retries, no LLM calls unless the gate's named
   contract says so (e.g. audited concept checks).
2. Gates answer "proceed / block / needs human review" — they never rewrite
   the artifact themselves.
3. Fail-closed is the default posture; an explicit opt-in is required for any
   advisory (non-blocking) gate.

## audits/

Post-hoc verification of *produced* artifacts and claims against registered
evidence: `validators.py` (statistical/clinical validator suite),
`patterns` (analysis-pattern auditing), `envelope_shadow` /
`envelope_consumers`, `manuscript_claims`, `outcome_semantics`,
`step_summary_integrity`, `aggregate_row`. Rules:

1. Audits run after the artifact exists and read sealed evidence ids — they
   must not mutate the run state or the artifact.
2. An audit failure feeds the repair loop; the audit itself never repairs.
3. Claim-level audits (manuscript, summaries) verify that every number and
   sentence traces to a registered evidence record.

## repairs/

Reactive fix machinery invoked after a gate/audit rejection: `patch`
transport, `coordination` (`RepairCoordinator`), per-cause handlers
(`typed_input`, `lossy_coercion`, `merge_collision`, `binary_feasibility`,
…), `reasons` (repair prompt authority), `attempt_record`. Rules:

1. Every applied repair is recorded in the run's `RepairLedger` — an
   unrecorded repair is a defect, not an optimization.
2. Repairs may propose new code/plans through the sanctioned transports
   (`PatchTransportUnavailable` boundaries) but never bypass authority
   binding or re-run gates implicitly.
3. A repair handler owns one failure cause; cross-cause sequencing belongs
   to `coordination`.

## Interaction rules (the part that keeps them from merging)

1. Dependency direction is one-way for domain implementations:
   `repairs → {gates, audits, contracts}` and `gates/audits → contracts`.
   Core authority storage/schema owners remain lower-level. Four explicit
   integration adapters currently consume narrow cross-layer contracts:
   `typed_binding → audits.step_summary_integrity`,
   `{step_runtime, step_capsule} → gates.semantics`, and
   `diagnostic_envelope → repairs.reasons`. These exceptions must not expand
   without an import-contract review.
2. A gate that starts repairing, an audit that starts rewriting, or a repair
   that starts binding evidence is a layering defect — split it, don't grow it.
3. `ValidationFinding` is the shared vocabulary, not a shared owner: gates
   emit it as a verdict, audits emit it as an inconsistency report, repairs
   consume it as a cause. The producer field names the owning layer.
