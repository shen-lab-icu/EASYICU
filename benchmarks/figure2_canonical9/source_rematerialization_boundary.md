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
| SICdb | `CaseID` | owner must attest its case/patient semantics | not inferable from the development export |

“Schema observed” means column names and table metadata were inspected only.
It is not evidence that a particular join, cardinality, cohort, or scientific
analysis is valid.

## Current decision

The active project plan forbids re-extracting the six databases and permits
development only from the existing `full6_20260717` export.  Therefore this
audit does **not** open, schedule, or implement a re-materialization job.  With
that constraint, a production Canonical9 run remains correctly blocked: no
code patch may turn the historical export into patient-level paper authority.

## Conditional route if a future data lane is explicitly approved

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
