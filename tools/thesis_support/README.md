# Frozen thesis support exports

Select `imv_recorded_compat`, `observed_peep_set`, and `rrt_time` explicitly
in `easyicu.extract_database(..., modules=[...], native_export_v2=True)`.
Pass the approved MIIV stay IDs explicitly. The native manifest records
source availability in `files[].thesis_support`.

| Module | Database | Raw inputs or availability |
| --- | --- | --- |
| `observed_peep_set` | AUMC | `admissions` and `numericitems` |
| `observed_peep_set` | HiRID | `general` and `observations` |
| `observed_peep_set` | MIIV | `icustays` and `chartevents` |
| `observed_peep_set` | SIC | `cases.ICUOffset` and `data_float_h` with DataID 2278 and Val in [0, 40] |
| `observed_peep_set`, `imv_recorded_compat` | eICU, MIMIC-III | Typed absence by spec |
| `imv_recorded_compat` | AUMC, HiRID, MIIV, SIC | `UNAVAILABLE_IN_SCOPE`. The named builders consume precomputed respiratory inputs. |
| `rrt_time` | AUMC, eICU, HiRID, SIC | `UNAVAILABLE_IN_SCOPE`. The named builders consume the precomputed hourly RRT flag or its dynamic module under the frozen contract. |
| `rrt_time` | MIIV | `UNAVAILABLE_IN_SCOPE`. The named code copies precomputed actual-end hours. |
| `rrt_time` | MIMIC-III | `UNAVAILABLE_IN_SCOPE`. The frozen CareVue producer requires MIMIC-III raw tables outside the approved #641 view. |

PEEP publishes the median in cmH2O for each observed ICU hour. SIC uses
`floor(("Offset" - ICUOffset) / 3600.0)`. A readable raw PEEP source with
no qualifying events publishes zero rows and reports `projected`.
Absent producers publish typed empty modules. The
`UNAVAILABLE_IN_SCOPE` receipts give `reason: no_raw_producer` and list
the missing inputs under `missing_inputs`.

## Equivalence check

Use an approved raw sample root. The tool reads the SHA-256-pinned
`sources.py` or `new_sources.py` from the default frozen checkout or from
a checkout supplied with `--frozen-src-dir`. It binds the frozen PEEP SQL
to the same raw files and cohort as the EasyICU rebuild.

```bash
PYTHONPATH=src python tools/thesis_support/frozen_equivalence.py \
  --database miiv --db-root "$DB_ROOT" --stay-ids-file "$PRIVATE_COHORT_FILE" \
  --module observed_peep_set
```

Keep real stay IDs in an approved file in the private layer. Parquet
cohorts use a non-null integer column named for the database's native
stay key or `stay_id`, with the native key taking precedence. Text
cohorts use one integer per line. `--stay-ids` accepts synthetic IDs and
is mutually exclusive with `--stay-ids-file`. MIIV requires a cohort.
Omitting both options for other databases selects all raw stays.

The tool compares every column by stay and hour and includes duplicate
row multiplicity. It prints status and aggregate counts only. Counts
below ten are JSON null and other counts round to the nearest ten. Exit
codes are 0 for agreement, 1 for a mismatch, and 2 when the comparison
cannot run.
