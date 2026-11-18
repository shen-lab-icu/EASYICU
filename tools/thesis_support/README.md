# Frozen thesis support exports

Select `imv_recorded_compat`, `observed_peep_set`, and `rrt_time` explicitly
in `easyicu.extract_database(..., modules=[...], native_export_v2=True)`.
Each module publishes hours on the frozen source clock. Pass the approved
MIIV stay IDs explicitly. The native manifest's `files[].thesis_support`
receipt identifies the source and its availability.

| Module | Database | Input under the supplied DB root | Output |
| --- | --- | --- | --- |
| `imv_recorded_compat` | AUMC, MIIV, SIC | Authoritative `{db}_hourly_core.parquet` with `stay_id, icu_hour, imv` | `charttime` and `imv=True` for each invasive hour |
| `imv_recorded_compat` | HiRID | Authoritative `hirid_vent_anchor.parquet` with `stay_id, imv_t0_h, imv_end_h` | `charttime` as the inclusive first hour and `imv_recorded_end_hour` as the inclusive last hour |
| `observed_peep_set` | AUMC, HiRID, MIIV | Raw `numericitems`, `observations`, or `chartevents` and the associated stay table | Median of the valid raw setting records at each observed ICU hour |
| `rrt_time` | AUMC, eICU, HiRID, SIC | Authoritative `{db}_hourly_core.parquet` with `stay_id, icu_hour, rrt` | `charttime` and `rrt_active=True` for each active hour |
| `rrt_time` | MIIV | Authoritative `miiv_rrt_time.parquet` with `stay_id, charttime, rrt` | Active hours from the corrected actual-end clock |
| `rrt_time` | MIMIC-III | Authoritative CareVue `mimic3cv_rrt_hours.parquet` with `stay_id, icu_hour` | Active hours on the supplied CareVue clock |

The staged authority tables use the frozen input schemas. For HiRID IMV and
MIIV RRT, the available frozen files specify the executable projection of
the upstream tables. Raw reconstruction requires the original generators
for the anchor's native airway and mode codes and for RRT's actual ends.
A missing authority publishes a typed empty module. A present, valid
authority with no qualifying hours publishes zero rows and records that
the projection ran.

The permitted frozen `sources.py` dispatches SIC PEEP to
`new_sources.py`. SIC PEEP remains unavailable until that reducer's
channel, units, and hour rules can be verified. The eICU and MIMIC-III
IMV and PEEP modules publish typed absence by contract.

## Equivalence check

Point `--frozen-src-dir` at the directory containing frozen `cohort.py`,
`sources.py`, and `readers.py`. Provide the same database root for both
sides, including any staged inputs in the table above.

```bash
PYTHONPATH=src python tools/thesis_support/frozen_equivalence.py \
  --database miiv --db-root "$DB_ROOT" --stay-ids 101 102 \
  --module rrt_time --frozen-src-dir "$FROZEN_SRC"
```

The output includes sorted-row SHA-256 digests over all projection columns.
It aligns rows by stay and hour, or by stay for the HiRID span. An unmatched
row contributes one mismatch to each column. Counts below ten are shown as
JSON null. Other counts round to the nearest ten. Exit codes are 0 for
agreement, 1 for a mismatch, and 2 when the comparison cannot run.

For a staged authority, the check establishes agreement of the consuming
projection. The staged table determines the upstream clinical observation
and clock semantics on both sides.
