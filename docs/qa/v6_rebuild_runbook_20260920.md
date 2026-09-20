# Full6 v6 clean rebuild runbook (2026-09-20)

## Decision

Do not patch or resume
`full6_native_v6_dictfix_aumc_miiv_fe382262_20260918`. It mixes extraction
commits, lacks final provenance, and contains the confirmed MIMIC-IV urine
staging failure. Keep it as diagnostic evidence.

Build the replacement from the sealed v5 release using one clean EasyICU
checkout that contains all of the following:

- coherent SOFA-1 total and rolled-component publication (`147f52d3`);
- longitudinal `rrt_criteria` handling (`abae4d1b`);
- MIMIC identity-table/time-axis correction and renal null-output sealer gate
  (`a3191f74`).

The raw-derived refresh request for every database is:

- `demographics`;
- `chemistry`;
- `blood_gas`;
- `vasopressors`;
- `medications` (the two WinTbl resolver routes now share the audited raw-start
  endpoint rule; this directly affects the numeric `dex` window concept).

The audited dependency expansion adds `respiratory`, `renal`, `sofa1_score`,
`sofa2_score`, `sepsis3_sofa1`, and `sepsis3_sofa2`. The other eight modules
must be republished from v5 without a logical table-content change.

## Resource evidence required before the formal run

The read-only 8-GiB plan was generated successfully. It is not yet formally
admissible because four database/module groups use unmeasured or invalidated
fallback profiles:

- eICU: `sofa2_score`, `sepsis3_sofa2`;
- HiRID: the requested five modules and their score/Sepsis closure;
- MIMIC-III: `sofa1_score`, `sofa2_score` and both Sepsis consumers;
- SICdb: the requested five modules and their score/Sepsis closure.

AUMC and MIMIC-IV already have measured plans for the complete closure. Run the
four missing benchmark candidates from the same clean checkout, one database at
a time. The initial batch sizes below are the guarded sizes emitted by the plan:

```bash
SOURCE=/home/zhuhb/workspace/phd-thesis/00-data-foundation/easyicu_full6_runs/releases/full6_native_v5_hirid_aki_rate_e0621aa1_20260906
CANDIDATES=/home/zhuhb/workspace/phd-thesis/00-data-foundation/easyicu_full6_runs/candidates

python scripts/releases/EX-A03_refresh_selected_modules.py \
  --source-run-root "$SOURCE" \
  --output-root "$CANDIDATES/benchmark_eicu_v6_rebuild_sofa2_20260920" \
  --database eicu --module sofa2_score --benchmark-only \
  --batch-size 25000 --allow-resource-policy-override \
  --resource-policy-override-reason "Measure current SOFA-2 closure before formal v6 rebuild"

python scripts/releases/EX-A03_refresh_selected_modules.py \
  --source-run-root "$SOURCE" \
  --output-root "$CANDIDATES/benchmark_hirid_v6_rebuild_fullclosure_20260920" \
  --database hirid --module demographics --module chemistry \
  --module blood_gas --module vasopressors --module medications \
  --benchmark-only \
  --batch-size 14000 --allow-resource-policy-override \
  --resource-policy-override-reason "Measure current HiRID closure before formal v6 rebuild"

python scripts/releases/EX-A03_refresh_selected_modules.py \
  --source-run-root "$SOURCE" \
  --output-root "$CANDIDATES/benchmark_mimic_v6_rebuild_scores_20260920" \
  --database mimic --module sofa1_score --module sofa2_score --benchmark-only \
  --batch-size 20000 --allow-resource-policy-override \
  --resource-policy-override-reason "Measure current MIMIC-III score closure before formal v6 rebuild"

python scripts/releases/EX-A03_refresh_selected_modules.py \
  --source-run-root "$SOURCE" \
  --output-root "$CANDIDATES/benchmark_sic_v6_rebuild_fullclosure_20260920" \
  --database sic --module demographics --module chemistry \
  --module blood_gas --module vasopressors --module medications \
  --benchmark-only \
  --batch-size 16000 --allow-resource-policy-override \
  --resource-policy-override-reason "Measure current SICdb closure before formal v6 rebuild"
```

Review each `resource_benchmark_provenance.json`, register only successful
measured profiles in `src/easyicu/api/extraction.py`, and rerun the plan. Do not
start the formal extraction until `formal_release_admissible` is `true` and
`unmeasured_or_overridden_modules` is empty.

## Formal replacement candidate

From a clean checkout, run:

```bash
python scripts/releases/EX-A03_refresh_selected_modules.py \
  --source-run-root \
  /home/zhuhb/workspace/phd-thesis/00-data-foundation/easyicu_full6_runs/releases/full6_native_v5_hirid_aki_rate_e0621aa1_20260906 \
  --output-root \
  /home/zhuhb/workspace/phd-thesis/00-data-foundation/easyicu_full6_runs/candidates/full6_native_v6_clean_rebuild_20260920 \
  --module demographics \
  --module chemistry \
  --module blood_gas \
  --module vasopressors \
  --module medications
```

The refresh command must produce one `module_refresh_provenance.json`, one
updated `run_manifest.json`, six native manifests bound to the same clean Git
commit, and complete receipts for all 114 Parquet files.

## Acceptance before sealing

Require all of the following:

1. MIMIC-IV `uo_6h`, `uo_12h`, `uo_24h`, and `aki_stage_uo_reference` are
   non-empty and their stay-level coverage is reconciled with v5 and raw urine
   availability.
2. SOFA totals equal the six exported rolled components. The retained 50-stay
   R ricu comparison remains near the audited v6 agreement and no database has
   an unexplained first-24-hour distribution shift.
3. Creatinine 15--25 mg/dL and PaO2 20--40 mmHg recoveries trace to raw rows;
   values outside the accepted bounds have an exclusion receipt.
4. Vasopressin and phenylephrine tails stay within the declared bounds, and the
   norepinephrine-equivalent formula sensitivity is retained.
5. Demographic service categories reflect the corrected dictionary.
6. The nine inherited modules pass the order-independent logical-content audit;
   row counts, hashes, manifests, and runtime provenance have no stale v5/v6
   mixture.
7. The release sealer, targeted core/governance tests, and downstream foundation
   status all pass without bypasses.

Seal only after review:

```bash
python scripts/releases/EX-A01_seal_full6_release.py \
  --run-root \
  /home/zhuhb/workspace/phd-thesis/00-data-foundation/easyicu_full6_runs/candidates/full6_native_v6_clean_rebuild_20260920 \
  --execution-profile portable-low-memory
```

After sealing, point `00-data-foundation/preprocessing` at the new immutable
release, rebuild the derived release, and refresh only study outputs whose input
contracts changed. Never refresh thesis results from an unsealed candidate.
