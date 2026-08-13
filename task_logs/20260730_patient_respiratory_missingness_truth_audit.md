# Patient respiratory missingness truth audit

Date: 2026-07-30
Task: `PATIENT-RESPIRATORY-MISSINGNESS-TRUTH-AUDIT`
Status: audit complete; production code unchanged

## Question

Audit whether the Patient Review → Data Quality respiratory feature chart is
computed from the official MIMIC-IV Demo data, and define what a displayed
`0%` means.

## Source identity

- Source: official MIMIC-IV Clinical Database Demo 2.2 prepared by the normal
  EasyICU all-module export pipeline.
- Cached archive SHA-256:
  `97301a03820e8f41af211cf3462ddc19aefe75bbed05f11753859affaafeb8ec`.
  This matches the allowlisted release hash and the extracted/converted/prepared
  receipts.
- Prepared export: 140 ICU stays, 19 modules, 151,373 rows.
- Respiratory table: 11,962 rows, SHA-256
  `5232d56dd86f855914d48ef84a2c041f8a8878a0cee3010e2d2206167b6cd596`.

## Current metric contract

The backend in `patient_drilldown._quality_metrics_payload` calculates:

`entity missingness = ICU stays with zero non-null rows for the feature / 140`

This is an entity-presence metric. It is not cell-level missingness, expected
hourly-slot completeness, or continuous-monitoring completeness.

## Independent prepared-table recomputation

The API values exactly match an independent count over
`respiratory.parquet`:

| concept | non-null rows | stays with ≥1 value | stays with no value | displayed |
|---|---:|---:|---:|---:|
| Oxygenation Index (`oxygenation_index`) | 133 | 45 | 95 | 67.9% |
| Arterial oxygen saturation (`sao2`) | 177 | 56 | 84 | 60.0% |
| Intubation/tracheostomy status (`ett_gcs`) | 1,332 | 60 | 80 | 57.1% |
| Advanced respiratory support (`adv_resp`) | 1,676 | 73 | 67 | 47.9% |
| Supplemental oxygen (`supp_o2`) | 5,252 | 76 | 64 | 45.7% |
| FiO2 (`fio2`) | 1,779 | 77 | 63 | 45.0% |
| PaO2/FiO2 (`pafi`) | 949 | 86 | 54 | 38.6% |
| SpO2/FiO2 (`safi`) | 11,731 | 140 | 0 | 0.0% |
| SpO2 (`o2sat`) | 11,710 | 140 | 0 | 0.0% |

Therefore the arithmetic is reproducible, but the chart combines metrics whose
semantics are not interchangeable.

## Raw-source cross-check

Aggregated checks against the downloaded official raw Parquet files support the
direct measurement counts:

- SpO2 item IDs `220227/220277/226253`: 14,820 non-null chartevents across all
  140 ICU stays.
- FiO2 item ID `223835`: 1,746 non-null chartevents across 76 stays; the
  prepared FiO2 feature covers 77 stays after the declared lab source is also
  included.
- Arterial saturation item ID `220227`: 104 non-null chartevents across 35
  stays; the prepared feature covers 56 stays after the declared lab source is
  included.

No direct identifiers or row payloads were copied into this audit.

## What `0%` currently means

- SpO2 `0%`: every one of the 140 ICU stays has at least one valid SpO2
  observation. It does **not** mean every expected time point is present.
  The per-stay record count ranges from 1 to 472, with median 48.5.
- SaFi `0%`: every stay has at least one derived SaFi value. This is not the
  same as direct FiO2 completeness. The SaFi callback intentionally fills
  unmatched FiO2 with 21% room air; all 63 stays with no exported FiO2 still
  receive SaFi values. PaFi is also available in 17 stays with no direct FiO2
  because of the same default.

## Semantic defects

1. `supp_o2`, `ett_gcs`, and `adv_resp` are positive-only `event_status`
   concepts in the typed column metadata. Their null complement is currently
   labelled missingness. Those bars should instead be reported as presence /
   exposure rates:
   - supplemental oxygen: 76/140 = 54.3%
   - intubation/tracheostomy status: 60/140 = 42.9%
   - advanced respiratory support: 73/140 = 52.1%
2. Direct observations and derived values are shown in one missingness series
   without provenance. In particular, SaFi `0%` visually overstates source-data
   completeness because FiO2 is directly observed in only 77/140 stays.
3. The axis label `实体级缺失率（%）` is too broad. For direct measurements the
   exact meaning is `无任何有效观测的 ICU stay 占比`.

## Recommended correction boundary

- Owner: Patient Review quality backend, consuming an immutable concept-metadata
  classification from the concept catalog/column-metadata owner.
- Backend payload should expose `metric_kind`, `availability_basis`,
  `observed_entities`, `denominator_entities`, and whether a derived concept
  used a default/imputation rule.
- The chart should separate direct measurement availability, derived
  availability, and event/exposure prevalence. Tooltips should show counts
  (`140/140`) alongside percentages.
- Do not fix this by adding more frontend-only feature-name conditionals; the
  semantic owner is the typed concept metadata.

## Commands/evidence

- `POST /api/patient-review/drilldown` against the active registered official
  demo.
- Independent pandas/pyarrow aggregation over the prepared respiratory table.
- Aggregated raw Parquet item-ID checks over chartevents/labevents and the
  official ICU-stay denominator.
