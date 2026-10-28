# Data provenance in demo and real runs

Audit: 2026-10-28. Inspected packaged metadata and synthetic fixtures only.
No UI files, dictionary entries, patient data or scorer files were changed.

## Existing surfaces and their inputs

| Surface | Input and host | Before this change |
| --- | --- | --- |
| Guided demo dictionary/animated evidence chain (`screens-guided-pi-demo-trace.js`) | Packaged `static/assets/demo/sofa2-run-trace.json`; `build_demo_walkthrough_trace.py` loads dictionary bytes at the recorded Git versions. Static host serves the assets. | Shows source tables/item IDs, callback text, canonical units, definitions and dictionary commit identities for a curated recorded demo. Animation is presentation of that static trace. |
| Demo run files (`static/assets/demo/run-files/`) | Packaged index and review projections, including run context and source manifest. | Shows aggregate run/result/code/cohort evidence; does not establish native extraction mappings for another run. |
| Extraction screens (`screens-extraction*.js`) | Catalog, extraction readiness/resource/cohort previews, `/api/jobs/extract` in `routes/jobs.py`; cohort preview in `routes/extraction.py`. | Selected concepts, units, supported sources and cohort configuration are visible. Extraction writes source column metadata and manifest receipts, but these are not joined into a run provenance view. |
| Dictionary (`screens-dict.js`, `data-catalog.js`) | `/api/catalog` and `/api/catalog/lineage/{concept_id}` in `routes/system.py`; packaged JS catalog fallback. | Live catalog shows declared mappings, selectors, callbacks, units and descriptions. These are current declarations, not a frozen record of a selected run. |
| ICD filter (`screens-icd.js`) | User-entered include/exclude tokens, normalized native cohort contract and database-specific filter owner. | Shows the filter criteria. There is no packaged authoritative ICD vocabulary with descriptions for every filter token. A diagnosis-text regex is not an ICD list. |
| Real run files and evidence preview (`screens-guided-pi-run-files.js`, `screens-guided-pi-evidence-preview.js`) | Governed artifact allowlist, wrapper projections and digest-checked evidence registry; `routes/agent.py` artifact/download/bundle and `routes/pi_copilot.py` evidence routes. | Generic JSON/code/table evidence exists, but no typed extraction provenance artifact joined these facts to a run's variables. |

## Fact inventory

The middle column describes the state before the change. “Available” means
already present in an owner or receipt, without inferring facts from patient rows.

| Fact | Before | New run projection |
| --- | --- | --- |
| Concept → database/table/item IDs | Available but not projected: dictionary, resolver and content-addressed column metadata sidecar. | Captures the selected loader's actual dictionary, including cached versions and inherited/specific source precedence. |
| Canonical unit and callback | Available but not projected per run; live catalog could display current declarations. | Frozen per-concept and per-source fields. |
| Source → target unit, factor/offset/rule | Not recorded as typed run metadata; callback owner knows the conditional rule. | Parsed from the same callback expression grammar; affine conversions carry factors/offsets, other rules retain their expression with unknown factors. |
| Derivation inputs/callback | Available but not projected in a run-level source view. | Dependency closure, forwarding, source callbacks and derivation callbacks are frozen. |
| Clinical definition/source/version | Available but not projected together with source facts. | Uses the clinical-contract owner when declared, otherwise dictionary definition metadata. |
| Diagnosis code list/descriptions | Lists available for declared code selectors; descriptions not recorded when no metadata owner declares them. | Publishes declared codes and descriptions only; missing descriptions have `description_status: not_recorded`. Native cohort ICD tokens use the same explicit absence. |
| Cohort filters | Shown in real runs through cohort summaries, native manifest contract/execution and materialization receipts. | Adds metadata-only native filters and digest-bound materialization predicates/windows, with no observed counts. |
| Dictionary/resolver identity | Dictionary fingerprint already registered, but separate from extraction mappings. | Normalized actual dictionary digest, packaged resource digest, selected source-config digest, resolver input/implementation digests and snapshot digest travel with the mapping. |

## Binding and disclosure

`concept/provenance.py` owns the closed immutable types and callback projection.
`concept/provenance_capture.py` captures inputs before extraction reads and
rejects changes before publication. The API hook observes the selected loader,
so a stale cached dictionary cannot be relabelled as the current catalog.

Native exports always put `data_provenance` in `_manifest.json`, even when
feature-definition export is disabled. Export intake verifies snapshot digests
and the source database. Existing export authority hashing includes the manifest
bytes. Materialization carries this into sealed semantic provenance. Direct
native materialization captures the same metadata and uses the existing verified
cohort-file/provenance binding. Research context copies only verified receipts.

Pipeline startup registers `data_provenance.json` as evidence with
`research_context` as its parent. The registered context digest and either
export authority or materialization/cohort-file digests bind it to the run.
Resume verifies the existing artifact rather than regenerating from a catalog.
Wrapper projection verifies the registered artifact digest and publishes a
closed, disclosure-scanned JSON view with its original `evidence_ref`.
Compact JSON publication preserves the existing evidence-preview size budget.

The artifact contains no patient rows, identifiers, observed unit values,
population counts or arbitrary dictionary parameters. A conversion is labelled
`application: source_callback_rule`: it records the rule used by extraction,
not a claim that a particular source unit occurred. Missing-unit-column and
blank-unit behavior are explicit. Existing row and browser disclosure scanners
apply to preview, download and bundles. Historical/unbound inputs and unknown
concepts yield typed `not_recorded`; no retrospective provenance is invented.

## Renderer handoff

The backend publishes these fields; the lead can use the existing renderers:

| Renderer | Artifact/fields |
| --- | --- |
| Run files (`screens-guided-pi-run-files.js`) | `data_provenance.json` is allowlisted, included in wrapper artifacts, downloadable and described by the bilingual run-file guide. |
| Evidence preview (`screens-guided-pi-evidence-preview.js`) | Existing JSON renderer; use `evidence_ref.evidence_id` / `evidence_ref.sha256` to open the original registered evidence. |
| Variable readers | Join `variables[].column` → `variables[].concept_id` → `data_provenance.snapshots[].concepts[].concept_id`. |
| Dictionary source/derivation display (`screens-dict.js`) | Render frozen `sources[].table`, `item_ids`, selector/value/unit/time fields, `callback`, `unit_conversions[]`, `codes[].code/description/description_status`, `canonical_unit`, `derivation_callback/derivation_inputs`, and `definition/definition_source/definition_version`. |
| Run identity/cohort detail | Display `dictionary_sha256`, `resolver_inputs_sha256`, `implementation_sha256`, `resource_inputs_sha256`, `source_config_sha256`, `snapshot_sha256`, native `cohort_filters[]` and run `cohort_selection`. Respect each `status` and explicit missing descriptions. |

Authoritative missing code descriptions require a separately supplied vocabulary
owner before they can be displayed. This change does not fabricate or fetch one.
