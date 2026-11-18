# Frozen thesis support exports

Select these modules explicitly in
`easyicu.extract_database(..., modules=[...], native_export_v2=True)`.
Pass the approved MIIV stay IDs explicitly for the hourly modules. The native
manifest records source availability in `files[].thesis_support`. The pinned
D0 projections take the explicit `pinned_inputs` argument described below.

| Module | Database | Raw inputs or availability |
| --- | --- | --- |
| `first_in_hosp_frozen` | AUMC | `admissions.admissioncount = 1`, preserving SQL NULL |
| `first_in_hosp_frozen` | SIC | `COALESCE(cases.OffsetAfterFirstAdmission, 0) = 0` |
| `first_in_hosp_frozen` | Other databases | `UNAVAILABLE_IN_SCOPE` with `no_frozen_predicate_in_scope` |
| `observed_peep_set` | AUMC | `admissions` and `numericitems` |
| `observed_peep_set` | HiRID | `general` and `observations` |
| `observed_peep_set` | MIIV | `icustays` and `chartevents` |
| `observed_peep_set` | SIC | `cases.ICUOffset` and `data_float_h` with DataID 2278 and Val in [0, 40] |
| `observed_peep_set`, `imv_recorded_compat` | eICU, MIMIC-III | Typed absence by spec |
| `imv_recorded_compat` | HiRID | Pinned `respiratory.mech_vent` and `outcome.los_icu`. One row per eligible stay, with the floored first and last invasive hours. |
| `imv_recorded_compat` | AUMC, MIIV, SIC | Pinned `respiratory.mech_vent`, `outcome.los_icu`, and the S1345 release-parameter receipt. Publishes the start and last invasive hour of the first frozen episode per eligible stay. |
| `rrt_time` | AUMC, eICU, HiRID, SIC | Pinned `renal.rrt` and `outcome.los_icu`. One row per active RRT hour on the frozen LOS spine. |
| `rrt_time` | MIIV | `UNAVAILABLE_IN_SCOPE` with `no_exact_producer`. D0 does not carry the corrected actual-end interval authority. |
| `rrt_time` | MIMIC-III | Typed absence by spec |

PEEP publishes the median in cmH2O for each observed ICU hour. SIC uses
`floor(("Offset" - ICUOffset) / 3600.0)`. A readable raw PEEP source with
no qualifying events publishes zero rows and reports `projected`.
Absent producers publish typed empty modules. The
`UNAVAILABLE_IN_SCOPE` receipts give the reason and name required inputs
under `missing_inputs`.

`first_in_hosp_frozen` publishes a nullable Boolean and the native stay key.
The column metadata and manifest receipt record the predicate, frozen
reference, resolved source file and SHA-256.

## Pinned D0 exports

Use module files from `20261117_d93795ce4_default_r14` (manifest SHA-16
`7f14448b9d9d1870`). Supply each module's resolved `<database>/<module>.parquet`
path, full SHA-256 from the pinned release, and exact projected columns.

```python
from pathlib import Path

from easyicu import extract_database
from easyicu.concept.thesis_support import PinnedModuleInput

extract_database(
    "hirid",
    pinned_inputs={"rrt_time": (
        PinnedModuleInput("renal", Path(renal_path), renal_sha256, ("stay_id", "charttime", "rrt")),
        PinnedModuleInput("outcome", Path(outcome_path), outcome_sha256, ("stay_id", "los_icu")),
    )},
    patient_ids=allowed_stay_ids,
    output_dir=output_dir,
)
```

Use an empty output directory for publication. For in-memory access to
one module, call `load_thesis_support` with that module's tuple of
`PinnedModuleInput` objects. The pinned mode resolves stay filters
directly against the D0 key. MIIV requires explicit allowed members.
When two projections use the same D0 module, pass the same resolved file
and digest to both.
For the HiRID IMV span, use `respiratory` with `stay_id,charttime,mech_vent`
in place of `renal`. Missing files and SHA-256 mismatches raise
`PinnedInputError` with reason codes before any Parquet decode.

For AUMC, MIIV and SIC IMV, bind that respiratory projection and pass
`frozen_release_parameter=FrozenReleaseParameter(Path(receipt_path), receipt_sha256)`
to `extract_database` or `load_thesis_support`. Import `FrozenReleaseParameter`
from `easyicu.concept.thesis_support`. The default common version is `v0.2`
and the key path is `("cohort", "episode_gap_tolerance_h")`. The module
receipt records the selected version, value, full receipt SHA-256, and
defining source location under `frozen_release_parameter`, with the rule
“S1345 narrow config read, 6 files, sha-pinned”.

## Equivalence check

Use an approved raw sample root. For PEEP the tool reads the SHA-256-pinned
`sources.py` or `new_sources.py` from the default frozen checkout or from
a checkout supplied with `--frozen-src-dir`. It binds the frozen PEEP SQL
to the same raw files and cohort as the EasyICU rebuild. For
`first_in_hosp_frozen` it runs the copied DuckDB predicates from
`build_c565v6.py:99-107` on the same cohort and requires
`--stay-ids-file` for explicit selection.

```bash
PYTHONPATH=src python tools/thesis_support/frozen_equivalence.py \
  --database miiv --db-root "$DB_ROOT" --stay-ids-file "$PRIVATE_COHORT_FILE" \
  --module observed_peep_set
```

For the pinned projections, pass the same file bindings to the harness. It
loads the relevant definitions and SQL from the exact frozen source files.

```bash
PYTHONPATH=src python tools/thesis_support/frozen_equivalence.py \
  --database hirid --module rrt_time --stay-ids-file "$PRIVATE_COHORT_FILE" \
  --pinned-input renal "$RENAL_PATH" "$RENAL_SHA256" stay_id,charttime,rrt \
  --pinned-input outcome "$OUTCOME_PATH" "$OUTCOME_SHA256" stay_id,los_icu
```

For AUMC, MIIV and SIC first-episode IMV equivalence, use
`--module imv_recorded_compat`, replace the renal binding with
`--pinned-input respiratory "$RESPIRATORY_PATH" "$RESPIRATORY_SHA256" stay_id,charttime,mech_vent`,
and add `--frozen-release-parameter "$RECEIPT_PATH" "$RECEIPT_SHA256"`.
Use `--common-version` to select a different version from the receipt.
The harness executes the frozen first-episode SQL with the resolved gap.

Keep real stay IDs in an approved file in the private layer. Parquet
cohorts use a non-null integer column named for the database's native
stay key or `stay_id`, with the native key taking precedence. Text
cohorts use one integer per line. `--stay-ids` accepts synthetic IDs and
is mutually exclusive with `--stay-ids-file`. MIIV requires a cohort.
Omitting both options for other databases uses the input stay set.

The tool compares every column by the module's row key and includes duplicate
row multiplicity. It prints status and aggregate counts only. Counts
below ten are JSON null and other counts round to the nearest ten. Exit
codes are 0 for agreement, 1 for a mismatch, and 2 when the comparison
cannot run.
