# Canonical9 owner activation packet

This is a decision packet for the paper-facing Canonical9 batch.  It contains
no patient data, mapping values, secrets, API keys, model outputs, or inferred
clinical decisions.  Filling it does **not** authorize a run.  Its purpose is
to collect the owner-controlled inputs that the existing P4 gate verifies
before any Provider, Docker runner, or cohort data can start.

Current status: **input route selected; run not authorized**.  The canonical
run remains `0/9`.

## 1. Select exactly one input route

Choose one route and record the approving owner and immutable evidence
reference.  Do not combine the routes or add an identity column to the
historical parquet in place.

| Choice | Owner decision required | What must later be verified |
| --- | --- | --- |
| Fresh native data lane | Approve a new, versioned source snapshot and native EasyICU export. | Native export manifest, physical file digests, six-source semantics, typed cohort and trajectory authorities. |
| Controlled identity bridge | Approve protected source-snapshot access solely to create an ICU-stay-to-patient mapping for the exact historical `full6_20260717` bytes. | `identity_bridge_contract.py` descriptor, source semantic attestations, mapping artifact digests/cardinality, then native typed-materialization review. |

The controlled bridge is not a shortcut around review.  Its descriptor must
continue to report `real_run_authorized=false`; P4 will reject any structural
retrofit as production authority.

### Recorded owner-delegated default (2026-07-22)

The owner selected the controlled-bridge route: `full6_20260717` remains the
only clinical payload, with source access limited to the six stay-to-patient
relations needed to prevent patient-level leakage.  The private host artifact
`full6_20260717_identity_bridge_20260722/identity_bridge_contract.json` was
built with digest
`4092104a40d2b22d80a93cf00323be0f3c048bfc3d9ea6bb002124361ecce794`.
It covers all six sources with zero unmapped or duplicate full0717 stays.

This owner decision authorizes the *bridge-production data lane* only.  It
does not turn the descriptor into typed materialization authority, a clinical
review, an execution permit, or a Provider credential.

## 2. Freeze source and identity evidence

Before a clinical task is activated, the owner must supply the following in a
protected host location (not in prompts and not in this packet):

1. The exact data-lane authorization reference and source snapshot identity.
2. An owner attestation for each database's ICU-stay-to-patient semantics.
   HiRID and SICdb require an explicit attestation; their key names cannot be
   treated as patient identity by inference.
3. A `source_attestation_contract.py` handoff pinned to the exact full0717
   export and bridge identities, with six independently reviewed typed-column
   inventories and data/transformation/identity owner references.  This is a
   review handoff only, not a P4 permit.
4. A completed native typed materialization review, with exact cohort authority
   for all nine tasks and trajectory authority for tasks that require it.
5. A full-nine `ProductionInputAuthority` whose ordered task digests match the
   materialized files and sidecars actually selected by the JSONL handoff.

No manual JSON edit, content hash, source-sidecar, or bridge descriptor can
substitute for these artifacts.

## 3. Close the three human/scientific blockers

These decisions must be made before the final P4 freeze.  They are case
protocols, not global agent-prompt text.

| Task | Required decision | Minimum review evidence | Forbidden shortcut |
| --- | --- | --- | --- |
| E2 lactate–mortality | Prespecify measurement/outlier handling, transformation, primary estimand, and sensitivity analyses. | A clinically and methods-reviewed ProtocolCard bound to its content digest. | Dropping values after inspecting the result, or converting the current concept-audit block into a green result. |
| H2 vasopressor causal analysis | Define data coverage and whether absence is `no exposure`, `missing capture`, or `unavailable`; then prespecify target-trial contrast and positivity checks. | Owner data contract plus clinical/methods review. | Inventing a control arm or treating unavailable medication capture as zero. |
| H3 trajectory clustering | Replace the failed stability plan with a new scientific design, including the fixed target population, representation, model-selection rule, stability criterion, and reportability rule. | A clinical/methods-reviewed redesign card. | Changing seed, k, threshold, or post-hoc exclusion to make the prior ARI result pass. |

The existing `KnowHowCard` mechanism binds a clinical/methods review
attestation to exact card content.  Do not set `clinical_reviewed` without a
real reviewer decision.

The real-run launcher now enforces this boundary before reading any cohort
bytes.  The operator must supply
`--figure2-scientific-protocol-authority <absolute-path>`; that authority must
bind the exact ordered E2/H2/H3 cards, their file SHA-256 values, their reviewed
content SHA-256 values, and their versions.  The freeze declaration pins the
authority digest.  Missing, reordered, unsigned, `curated_mvp`, or modified
cards fail closed.  The verifier never generates an attestation, so this
control cannot substitute for the three real review decisions.

### Owner-delegated defaults awaiting formal attestation

These choices let implementation and data-quality work proceed without
silently changing a scientific result.  They are deliberately not labelled
`clinical_reviewed`.

| Task | Default chosen for the next typed review | Consequence now |
| --- | --- | --- |
| E2 | ICU-admission anchor; first-24-hour maximum among unit-verified, finite, positive lactate values; preserve all such values in the primary analysis, summarize with median/IQR, assess nonlinearity on the raw mmol/L scale, and use a predeclared log-scale sensitivity analysis. Values with unresolved units or invalid measurement state are reported as unavailable, never deleted based on an estimated effect. | The existing peak-lactate card and review packet are the implementation starting point; formal clinical/methods sign-off is still required before paper authority. |
| H2 | Medication non-recording is **unknown**, not non-exposure. A causal contrast may include a source only after its first-window medication capture and timing are evidenced; otherwise it is an explicit feasibility failure, not an invented control arm. | The content-bound pre-review packet is `docs/reviews/vasopressor_comparative_effectiveness_20260722.json`; current manifests establish a vasopressor module was exported without errors, not a complete absence-as-nonuse contract. |
| H3 | The observed stability failure is terminal for the current design. Do not retry with a new seed, k, threshold, or post-hoc exclusion. | The content-bound pre-review packet is `docs/reviews/longitudinal_icu_phenotyping_20260722.json`; H3 remains a documented negative feasibility result until an independently reviewed redesign is available. |

## 4. Prepare the P4 operation freeze

Only after Sections 1–3 are complete, prepare these paths and pins for the
existing `OperatorFreezeDeclaration`:

- exact ordered Canonical9 JSONL (`E1`, `E2`, `E3`, `M1`, `M2`, `M3`, `H1`,
  `H2`, `H3`) and its SHA-256;
- full-nine production input authority and digest;
- ordered E2/H2/H3 scientific-protocol authority and digest;
- expected execution identity, clean code commit, Docker image digest,
  submission-profile reference, runner and network-policy pins;
- one real Provider/model pair and the approved cost/runtime limit.  The local
  Luna endpoint may be selected only in the final operator confirmation; never
  record its credential in a file or prompt;
- an empty absolute output directory named by a new `batch_...` identifier;
- `arms=aware`, cross-run memory disabled, no development sample, no resume,
  no writer probe, and paper acceptance enabled.

The P4 gate independently compares every one of these declarations to the
actual parsed invocation.  A mismatch exits before any provider call or data
load.

## 5. Final operator confirmation

The final confirmation must name:

1. the selected route and exact approved source/mapping evidence;
2. all three reviewed E2/H2/H3 protocol decisions;
3. the pinned model, output root, batch identifier, and cost/runtime ceiling;
4. the exact fresh, full-nine `--arms aware` invocation.

After confirmation, run one batch only.  Each task must either produce a
verified aware-arm attempt that reaches the evaluator or be fail-closed with
its evidence retained.  A process exit code alone is not a scientific success.
