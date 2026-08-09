# SOFA partition-invariance evidence

Machine-generated records behind the claim that SOFA and SOFA-2 do not depend
on chunk size or worker count. Regenerate with:

```bash
python tools/record_partition_invariance_evidence.py \
  --data-path /path/to/prepared/mimiciv --database miiv --cohort 3000 \
  --out docs/evidence/partition_invariance/miiv_3000.json
```

## What the comparison actually is

A SHA-256 over the **canonicalised frame** — columns sorted, rows sorted, CSV
rendering — which is equivalent to `assert_frame_equal(check_exact=True)` on
that canonical ordering. It is **not** a byte comparison of stored Parquet
files, and these records do not hash any output file.

## Privacy

No patient data. A cohort appears only as `cohort_id_sha256`, a SHA-256 over
its sorted stay ids; results appear only as row counts and frame hashes.

## Current records

| file | database | cohort | concepts | configurations | result |
|---|---|---|---|---|---|
| `miiv_3000.json` | MIMIC-IV | 3,000 | sofa, sofa2 | chunk 250/500/1000/2000/4000 + workers 1/4 | 7/7 match per concept |
| `eicu_3000.json` | eICU | 3,000 | sofa | chunk 500/2000 + workers 1/4 | 4/4 match |

Larger runs measured but not recorded as JSON: MIMIC-IV 10,000 stays
(2,351,306 rows) across chunk 500/2000/4000, all matching. **Full-database
scale (~94k stays) is not measured.**

The pytest harness is `tests/test_sofa_partition_invariance.py` (13 cases,
`needs_real_data`); it is skipped without `--run-real` and a real
`EASYICU_DATA_PATH`, so ordinary CI does not execute it.
