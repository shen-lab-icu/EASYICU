# MIMIC-IV 19-module extraction resource standard

Date: 2026-08-27 EDT
Code baseline: `dfc1687` plus the current profile-registration patch
Dataset: `/Volumes/外置硬盘/databases/mimiciv`
New evidence root: `/Volumes/外置硬盘/tmp/easyicu-6db-resource-profile-30f5228-ak9lbI/mimiciv/`

## Question and method

Complete the missing 11 of 19 MIMIC-IV extraction-module measurements and
determine whether full-cohort one-shot is safe when 8 GiB is currently
available. Each module ran against all 94,458 stays in an isolated monitored
process with native-v2 output and temporary spill on the external disk,
`EASYICU_DUCKDB_THREADS=2`, a 2 GiB DuckDB limit, 0.05-second process-tree RSS
sampling and a 7,447 MiB hard stop. Patient batching was to be tested only after
a one-shot failure.

## New results

All 11 previously unmeasured modules completed one-shot; no batch search was
needed.

| Module | Wall seconds | Peak process-tree RSS MiB | Rows or package scope |
|---|---:|---:|---|
| sepsis_shared | 21.724 | 4,627.9 | 647,053 |
| other_scores | 190.594 | 4,821.9 | 8,561,955 |
| respiratory | 73.137 | 5,077.9 | 7,812,260 |
| circulatory | 334.370 | 4,721.5 | 7,521,507 |
| neurological | 70.912 | 4,604.9 | 5,324,011 |
| medications | 88.702 | 6,749.9 | 3,445,620 |
| renal | 281.666 | 7,362.0 | 8,356,138 |
| sofa1_score | 165.533 | 6,259.9 | 23,244,352 |
| sofa2_score | 437.698 | 6,315.5 | 13,070,521 |
| sepsis3_sofa1 | 267.852 | 5,749.6 | full dependency package; endpoint 36,818 |
| sepsis3_sofa2 | 428.684 | 7,286.7 | full dependency package; endpoint 8,223 |

The two Sepsis-3 runs included `outcome`, `sepsis_shared`, the corresponding
SOFA module and the endpoint module; they are not invalid endpoint-only probes.
The prior eight light-module profiles remain the authority for demographics,
outcome, blood gas, vasopressors, hematology, chemistry, ventilator and vitals.

## Product decision

- Register all 11 new one-shot profiles in the same typed resource owner.
- The full 19-module request at 8,192 MiB available runs one-shot. The strictest
  selected threshold is `renal`: 7,362.0 x 1.10 = 8,098.2 MiB.
- Show no cleanup warning when that threshold fits. Below a selected module's
  threshold, fall back to patient batches and explain the speed penalty.
- Do not copy eICU batch decisions into MIMIC-IV: `respiratory` and
  `circulatory` both fit one-shot here.

## Optimisation findings and limits

Logs repeatedly scan the same MIMIC-IV bucket sets within `other_scores`,
SOFA-2 and the Sepsis-3 dependency graphs. The high-value optimisation is to
materialise/reuse owner-governed dependency results or coalesce compatible
same-table concept scans. Patient batching would repeat those scans and is a
fallback for insufficient memory, not the preferred speed optimisation.

The receipts establish extraction resource selection for this source layout,
cohort ceiling and revision. They do not promise the same wall time on another
CPU/disk, prove the proposed dependency-cache optimisation, or provide
Qualification12, Held-out27, clinical, causal or publication authority.
