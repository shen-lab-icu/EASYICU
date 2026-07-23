# Canonical9 source re-materialization boundary

This card is a paper-input boundary, not a data-processing instruction and not
an authorization to launch Canonical9.  It records why the current historical
six-database development export cannot become a paper-facing input by adding
metadata, and what a separately authorized future data lane would need.

## What the metadata audit established

The `full6_20260717` development export contains ICU-stay-level identifiers
but no patient-level identity mapping.  It may support offline development,
but it cannot by itself supply patient-level split or repeated-admission
authority.  A structural sidecar therefore remains development provenance and
is rejected by `realrun_authority.py`.

The retained source conversion libraries contain candidate identity sources:

| Source database | Candidate ICU identity | Candidate patient identity | Status |
| --- | --- | --- | --- |
| MIMIC-IV | `stay_id` | `subject_id` | source mapping schema observed; re-materialization required |
| MIMIC-III | `icustay_id` | `subject_id` | source mapping schema observed; re-materialization required |
| eICU | `patientunitstayid` | `uniquepid` | source mapping schema observed; re-materialization required |
| AmsterdamUMCdb | `admissionid` | `patientid` | source mapping schema observed; re-materialization required |
| HiRID | `patientid` | owner must attest its stay/patient semantics | not inferable from the development export |
| SICdb | `CaseID` | `PatientID` | source relation available; semantic attestation still required |

“Schema observed” means column names and table metadata were inspected only.
It is not evidence that a particular join, cardinality, cohort, or scientific
analysis is valid.

## Current decision

The owner selected a controlled identity bridge on 2026-07-22.  It preserves
the existing `full6_20260717` clinical bytes and reads only six frozen source
relations to make host-only stay-to-patient mappings.  The builder produced a
private contract with digest
`4092104a40d2b22d80a93cf00323be0f3c048bfc3d9ea6bb002124361ecce794` and
zero unmapped/duplicate full0717 stays in every source.  No clinical export was
re-extracted, modified, or made available to a Provider.

This changes the next action from “await a route choice” to “perform native
typed-materialization review.”  It does **not** change the production state:
the bridge descriptor remains a handoff artifact and P4 correctly blocks any
run until typed cohort/trajectory authorities, the three case decisions, and
the final operation freeze exist.

The non-authorizing evidence inventory for that review is
`docs/reviews/full0717_source_attestation_20260722.json`.  It records the
full-export and bridge identities, plus the decisive limitation: historical
module manifests have empty `concept_meta` and the recorded EasyICU commit is
empty.  The next materialization must therefore be a specifically reviewed,
source-bound typed path; it cannot infer clinical semantics from names or
patch full0717 in place.

`source_attestation_contract.py` now supplies the strict, snapshot-pinned
handoff format that this future review must use: it accepts only the exact
full0717 export and bridge digests, requires six source-attested typed
inventories plus data/transformation/identity owner references, and reports
review eligibility only.  It remains intentionally absent from P4's import
graph; an attested handoff still requires a separately reviewed native typed
materialization and the existing P4 authority/final-operation gates.

## Conditional routes if a future data lane is explicitly approved

There are two mutually exclusive routes.  The controlled bridge is now the
selected route; neither route changes the P4 launch gate.

1. **Fresh native export.** Follow the native source-to-export route below.
2. **Controlled identity bridge.** Preserve the historical full6 clinical
   bytes, but separately derive a protected ICU-stay-to-patient relation from
   frozen source snapshots.  The bridge must be bound to the full6 manifest and
   content digests, every mapping artifact digest, per-source cardinality
   evidence, and an explicit semantic attestation for HiRID/SICdb.  Its
   repository contract is `identity_bridge_contract.py`; it intentionally
   reads only the small descriptor, not a mapping or a patient row.  A complete
   bridge may be handed to native typed materialization review, but cannot
   authorize a run, replace a trajectory/cohort authority, or bypass the final
   operator freeze.

### Fresh native export route

1. The owner explicitly approves a new, versioned data lane, freezes the
   approved source snapshots, and records the
   conversion/provenance manifest for each participating database.
2. For every database, the owner records the ICU-to-patient mapping rule,
   cardinality policy, and the treatment of unavailable mappings.  HiRID and
   SICdb require an explicit semantic attestation; the pipeline must not infer
   that a stay-level key is a patient key.
3. A fresh EasyICU native export is materialized from those approved sources.
   The native `_manifest.json`, physical file digests, concept selection,
   identity column, and mapping provenance must be written together.  The
   historical `full6_20260717` files are not patched in place and are not used
   as paper authority.
4. The normal `ExportPackage` and typed cohort/trajectory materialization path
   produces verified authority sidecars from the fresh native export.  These
   sidecars are evidence of a newly materialized source; they cannot upgrade a
   structural retrofit.
5. Only after all typed authorities, scientific identities, and the three
   human-owned case decisions are frozen may the owner replace the blocked
   entries in `canonical_run_input_bindings_v2.json` with exact digests.
6. The final operator confirmation names the source snapshot, output directory,
   model/cost limit, and all-nine aware-arm invocation.  The P4 gate then
   validates the batch before any provider call or patient-data execution.

## Non-negotiable exclusions

* Do not treat an added `subject_id` column, a content hash, or a structural
  `seal_kind` as source authority.
* Do not use a seed, cluster count, threshold, or fallback cohort to make H3
  pass after a failed stability criterion.
* Do not fabricate an H2 exposure/control definition when the owner has not
  chosen what absence means.
* Do not run a historical `naive` arm for a paper-facing Canonical9 batch.
  The frozen workflow requires `--arms aware`.
