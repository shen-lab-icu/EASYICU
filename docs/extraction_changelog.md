# Extraction changelog

Changes that move extracted values between versions, for anyone comparing an
export with an earlier one. Each entry names its commit; the audit item it
answers is in brackets.

lead-r13 (0.2.0) is validated on SICdb 1.0.8, the default release, and
remains compatible with SICdb 1.0.6 (the release is detected from the schema).

## 2026-11-06 — the DuckDB and pandas load paths publish the same rows [lead-r13, `f486570e2`]

**What changed.** A concept loaded through DuckDB hourly aggregation now
publishes byte-identical rows, values and dtypes to the same concept loaded
through `load_table` (parity census: `tools/duckdb_raw_parity_census.py`).

- **Episode window before the hour floor.** DuckDB floored each event to its
  hour and the ICU-episode clip then judged the hour, so the hour containing
  discharge + 24 h kept events up to 59 minutes past the bound. Every DuckDB
  entry point now filters the raw event hour against the window first, and
  the window has one owner (`ConceptResolver._icu_episode_window`) shared
  with the pandas clip.
- **Hourly sums are not bounded twice.** ricu bounds each value, then sums;
  the DuckDB path bounded the hourly total again, dropping hours whose
  in-bound records add up past the maximum (eICU urine above 5000 mL from
  several in-bound Foley records). The MIIV/MIMIC signed urine sum keeps its
  documented post-sum bound (`SIGNED_SUM_THEN_BOUNDS_V1`).
- **Integer sums are float64** on both paths (MIIV integer urine values).
- **Failures are typed.** A failed hourly aggregation
  (`interval_aggregation_failed`), a failed DuckDB wide or batch read, and a
  failed episode-bound read (`icu_episode_window`) stop the extraction;
  before, they fell back to keep-first rows, a raw re-read, or an unclipped
  frame.
- **Batch sums.** The wide and long batch loaders sum and average with the
  same order-independent exact sum as the single-concept path.
- **HiRID same-time duplicates** (`neut`, segmented + band) keep the latest
  entry, then the value; before, input row order decided.
- **MIMIC numeric clip** reads its own stay bounds; it used the shared
  icustays cache, which could hold only `intime` and then widened the clip
  to the 366-day fallback.
- **Multi-source pooling.** A regex source (`rgx_itm`) counts as a numeric
  source in the pooling decision, so a concept with a regex source and
  another numeric source pools raw values into one hourly median, as ricu
  does (eICU `etco2`: respiratory charting plus the monitor column). Before,
  each source was pre-aggregated and the per-source medians were averaged.
- **Batch rows need a value.** The wide and long batch loaders no longer
  publish hours where none of the requested concepts has a value.
- **A failed batch read is a failure.** The wide and long batch pre-loads
  re-read each concept on the per-concept path after any error; they now
  stop with a typed error (an absent table still takes the per-concept
  path, which reports it).

**Errata (demo counts versus `1962535eb`).** eICU demo: urine loses 37 hours
past the episode bound, regains 5 hours whose in-bound records total more
than 5000 mL, and changes 1 hourly total; 180 lab and vital hours past the
episode bound leave 40 concepts (at most 10 per concept); 25 `etco2` (and
`etco2_raw`) hours change to the pooled median; `co`, `icp`, `pap_*` and
`pawp` drop 116,000–138,000 rows that carried no value (their values are
unchanged apart from one `co` hour past the bound). MIMIC-IV demo: `cl`, `k`
and `na` each lose fewer than 10 hours past the bound. Parity census after
the change: eICU 78 of 78 and MIMIC-IV 113 of 113 DuckDB-routed concepts
byte-identical to the pandas path.

## 2026-11-06 — Sepsis-3 counts unscored SOFA-2 hours as 0 in every execution mode [lead-r13, `c58852fb9`]

**What changed.** The Sepsis-3 modules (`sepsis3_sofa2`, `sepsis3_sofa1`)
always derive from the published SOFA score modules. They read the
published files first, then the group's own outputs; otherwise the module
owner extracts the score modules into a private staging folder. They are never
recomputed. A missing or failed dependency raises `SpecialDependencyError`
(`sepsis_dependency_module_failed`, `sepsis_dependency_output_missing`,
`sepsis_dependency_manifest_missing`, `sepsis_special_cohort_missing`).
One rule, `sofa2_total_for_sepsis_delta`, sets the SOFA-2 total for both the
extraction path and the concept callback.

**Why.** The one-shot path, used whenever outputs were not streamed and the
score modules were not yet published (the in-process scoring group),
recomputed SOFA-2 through `load_concepts`. The merged frame had no row at all
for hours where no SOFA-2 component had a value, so those hours were missing
instead of scored 0 as in ricu. That raised the baseline before suspected
infection, and onsets moved later or disappeared. `8c948c740` had fixed hours
that were present but empty, not hours that were absent. On the p9b MIMIC-III
CareVue sample, `sepsis3_sofa2` had 928 rows in this mode and 929 when
streamed. A hand check of the two stays involved confirmed the streamed result.
`sepsis3_sofa1` showed no difference on that sample.

**SICdb.** The concept callback used to keep SICdb's partial SOFA-2 total and
could produce Sepsis-3 rows. It now gives an empty total and no rows, matching
extraction; the formal SICdb export already had 0 rows.

**Formal exports.** The formal v6 exports ran this one-shot path for every
database: `sepsis3_sofa2` rows AUMC 5,178, MIIV 42,368, eICU 24,569, HiRID 0,
SICdb 0. No `2786ea872` receipts were found; its code selects the same path.

**Errata.** `sepsis3_sofa2` and `sepsis3_sofa1` rows changed versus
`2786ea872`, per database (counts of 10 or more rounded to tens): FILL.

## 2026-11-06 — HiRID values no longer depend on the batch (no hourly pre-aggregation) [lead-r13, `20c71d62e`]

**What changed.** The raw loader no longer pre-aggregates HiRID frames to
hourly medians. The old step ran when a loaded frame held more than 1,000 rows
and the median time gap across all patients' rows was under half the
interval. Whether it ran therefore depended on which stays shared a batch: a
stay loaded alone kept its raw rows, but in a full batch it was collapsed. It
also applied to raw (`aggregator=False`) and static (`id_tbl`) loads, and it
aggregated before the concept bounds. Values are now aggregated once, after
bounds, in the shared hourly owner, as in ricu.

**Why.** ricu takes a stay's weight as the median of all in-bound raw
measurements. The collapsed frame moved that median: on the p9b sample, 7
weight rows of 2 stays became 5, and their weight moved by up to 2.4%. Every
per-kg drug rate derived from it moved too. The formal v6 and `2786ea872`
HiRID extractions ran one batch over all stays, so they carry the collapsed
values.

**Affected (HiRID only).** `dobu_rate`, `epi_rate`, `norepi_rate`,
`propofol_rate` (`hirid_rate_kg`); `norepi_equiv` and the `*60` concepts;
`weight`, `height` and `bmi`; SOFA cardiovascular and the scores and
sepsis definitions derived from it.

**Errata.** Changed cells and stays per concept versus `2786ea872` (counts of
10 or more rounded to tens): FILL.

## 2026-11-06 — values stay float64; decimal thresholds are decided with a tolerance [lead-r13, `206439895`, `eae58e67b`]

**What changed.** Concept values are no longer down-cast to float32 on load
or after aggregation; every value column stays float64. Every comparison of
an arithmetic-derived quantity (a difference, ratio, product, rate or window
length) with a decimal clinical threshold goes through
`easyicu.utils.thresholds` (`ge`, `gt`, `le`, `lt`), which treats a value
within `1e-6 × max(1, |threshold|)` of the threshold as equal: `ge`/`le`
include an exact boundary and `gt`/`lt` exclude it. Missing values behave as
before (NaN compares False; nullable columns keep `<NA>`).

**Why.** Binary floating point puts an exact decimal boundary a few ulps to
either side: `1.1 + 0.3` is `1.4000000000000001` and `0.8 × 1.5` is
`1.2000000000000002`, so a creatinine of 1.4 after 1.1, or 1.2 against a
baseline of 0.8, was not KDIGO stage 1. Over all 0.1 mg/dL creatinine pairs
from 0.1 to 14.9 the plain operators were wrong for 29 (+0.3) and 17 (×1.5,
×3) pairs in float64 and 55/20/20 in float32; with the tolerance, none in
either. Results therefore no longer depend on the load precision.

**Where.** KDIGO creatinine and urine output; the AKI database profiles at
their native operators (SICdb strict `>`, AUMC ratio `> 3`); SOFA-1 and
SOFA-2 respiratory, liver, cardiovascular and renal components; circEWS drug
levels; urine coverage windows and the 12 h record gap; the ventilation
episode gap (8 h); outcome horizons (days); CVP plausibility; persistent
critical illness (`los_icu ≥ 10`); SIRS `pco2 < 32`; the RRT creatinine
gate (`> 1.2`), the public `sofa2_liver` callback, the MIMIC-III AKI-profile
creatinine window (−6 h to 162 h) and the AUMC death-after-discharge window
(< 72 h) (`e04b63235`). The SOFA-2 gap grid, which carried the source
assessment time in float32, is float64 too (`9aed79a4b`). Full site list:
lead-r13 threshold audit (groups A–G and I).

**Zero comparisons now carry the tolerance.** Four "no urine" tests compared
a derived value with zero: the MIMIC-III native urine staging and the SOFA-2
renal component (`== 0`), and SOFA-2 and KDIGO anuria (KDIGO was
`<= 1e-9`). They now use `le(x, 0)`, which counts a derived value up to 1e-6
as zero. Below 1e-6 mL/kg/h a rate can only be float noise: charted urine has
at least 1 mL resolution, so the smallest real rate (one 1 mL record over
24 h at 300 kg) is 1.4e-4 mL/kg/h. This is a definition change at the zero
boundary, recorded here and in the threshold audit (D1).

**Hourly sums in a canonical order (`1962535eb`).** Sums and means are
computed with an order-independent exact sum, so batch size, thread count
and load path can no longer move the last bit. On the demos this moves
`total_input_ml` by 1 ulp in 22 (eICU) and 578 (MIIV) cells and
`norepi_equiv` in 10 and 17 cells; fluid balances that inherit them move by
at most 4.6e-13 (eICU) and 7.3e-12 (MIIV) absolute.

**Errata.** Float64 with the tolerance is the correct value. Stays whose
KDIGO stage, AKI occurrence, SOFA components or Sepsis-3 changed versus
v6 (`97f508c6`) and `2786ea872`, per database: FILL.

## 2026-11-06 — eICU intake rates: the largest same-time rate owns the segment [lead-r13, `1962535eb`; pinned by `43cbe4aca`]

**Rule.** eICU `intakeoutput` rows labelled as a rate (`mL/hr`) are integrated
from their offset to the next record of the same stay and channel (`cellpath`),
at most six hours. When one channel has two or more different rates at the same
offset, the records are now ordered by (stay, channel, offset, rate) with a
stable sort. The **largest** rate owns the segment to the next record. The
smaller same-time rates get duration 0 and add no volume. A same-time 0 does
not stop a larger rate.

**Rationale.** Only one of the same-time records can own the following
segment. Before `1962535eb` the sort key ended at the offset, so the record the
loader delivered last owned it. That record depends on batch size, bucket
layout and thread count, so `total_input_ml` (and the fluid balances built on
it) could change with how the extraction was split. A total order on the value
makes the result layout-independent. The rule takes the largest rate and
does not sum same-time rates: those are alternative entries for one channel,
not separate infusions. This is a definition change accepted in review (Codex
delta2 UNEXPECTED_CHANGE systemic_intake: recorded here, not reverted).

**Impact.** None measured:
- **eICU demo (v2.0.1):** after deduplication its `intakeoutput` has 0
  rate-labelled systemic rows. Hourly intake is identical under the old and
  new keys: 60,510 stay-hours, 0 differing, 0 stays affected.
- **eICU 3,000-stay server sample:** 0 rate-labelled rows; 95,864 stay-hours,
  0 differing.
- **Full eICU (180,855 stays):** 6,192 rate-labelled rows in 75 stays, but
  0 same-time groups with conflicting rates. 6,676,112 stay-hours, 0
  differing.

Under the old key, reversing the input order changed 0 stay-hours on this
data. The rule matters only for a source that charts different rates for one
channel at the same minute.

**Pinned by** `tests/core/test_eicu_rate_same_time_tie_rule.py`:
- the largest rate wins in every input order;
- the rule applies per channel and per stay;
- a same-time zero does not stop a larger rate.

Four of its six cases fail under the old key.

## 2026-11-06 — creatinine, bilirubin and urea use exact SI factors [lead-r13, `c747d1900`] (dictionary reseal)

**What changed.** AUMC and HiRID creatinine (µmol/L → mg/dL) is multiplied by
1/88.4 (was ricu's 0.011309 for AUMC, 0.011312 for HiRID); bilirubin and
direct bilirubin by 1/17.1 (was 0.058467); urea (mmol/L → BUN mg/dL) by 2.802
(= 28.02/10, was 2.8); SICdb urea (mg/dL → BUN mg/dL) by 28.02/60.06 (was
0.467). The community sources (jinhua, zhejiang_eicu, zigong) follow.

**Why.** ricu's rounded factors bias the converted value by about 1e-4,
beyond the threshold tolerance: 106.08 µmol/L became 1.19966 mg/dL instead
of 1.2, 442 µmol/L 4.9986 instead of 5.0, 34.2 µmol/L 1.99957 instead of
2.0, so SOFA classed them one level low. Values that change class on their
recorded precision: creatinine 442 µmol/L (SOFA-1 renal 3 → 4; AUMC, HiRID),
HiRID creatinine 176.8, 309.4, 353.6 µmol/L, and bilirubin 34.2, 102.6,
205.2 µmol/L (SOFA-1 liver). All other converted values shift by at most
1e-3 relative. Deviation from ricu, recorded in the deviation table.

**Errata.** AUMC and HiRID stays with a changed SOFA renal or liver
component versus v6 and `2786ea872`: FILL.

## 2026-11-06 — SICdb 1.0.8 is the default release [lead-r13, `130ad381a`, `0a64917e7`, `ec2f10a6f`, `d6f89bee9`] (dictionary reseal)

**What changed.** SICdb 1.0.8 is read by default; 1.0.6 directories are still
found and the release is detected from the schema. Hospital discharge types
are read by their `d_references` label (1.0.6: 2026 Survived / 2028 Deceased;
1.0.8: 3129–3134, death 3130 "Sterbefall"), never by number. An empty
survivor follow-up category (ESOT, empty for every 1.0.8 survivor) is
rebuilt from the admission year (2013–2020: one year; 2021: six months;
any other or missing year: unknown, so every horizon and the follow-up are
NA, flagged and warned), flagged assumption-based; `TimeOfStay` is taken as published. RASS (3123)
and O₂ flow (3144) are mapped for SICdb.

Survival is a whitelist of `d_references` labels (`443b7a233`): a discharge
type whose label is neither a known death nor a known survival label (empty,
renamed or unknown) is unknown, warned (`sic_discharge_label_unknown`), and
gets no follow-up or horizon. A case with an empty discharge type is not
"unknown label"; when it has no recorded death its follow-up category is
rebuilt like a survivor's (1.0.8: 49 such cases), disclosed here. A recorded
death with an empty follow-up category also gets the category bound from its
admission year (`b0c995204`), so its 365-day horizon stays capped as in
1.0.6. The official-source registry default moved from `/2` to `/3` (the
1.0.8 DOI); receipts record `registry_version` accordingly.

**Why.** 1.0.8 renumbered the discharge types and left ESOT empty for
survivors; with the 1.0.6 codes every 1.0.8 outcome was unknown. Details and
evidence: deviation table, "SICdb 1.0.8".

**Errata.** On the same 1.0.6 data, baseline and lead-r13 agree on 28/90/365-day
mortality, follow-up and assumption flags (1.0.6 empty categories are rebuilt
too; no value changes). Across releases, on the cases common to 1.0.6 and
1.0.8 the values are not all equal: `mort_28d`/`mort_90d` do not differ,
`mort_365d` differs on FILL (<10 before `b0c995204`); `followup_days` for
28/90/365 days differs on 25/25/26 cases, because 1.0.8 changed
`TimeOfStay` (taken as published) and a case without a death time ends at its
stay end; the assumption flags differ on 22,229/22,229/8,335 cases, because
every 1.0.8 survivor's category is rebuilt.

## 2026-11-06 — conversion follows symlinked source folders and never converts nothing silently [lead-r13, `316bceeca`, `53064634a`]

Conversion only; extracted values do not change. Source folders linked from
another drive (`hosp -> /Volumes/drive/hosp`) were not walked, so the
converter found no CSV, wrote an empty manifest and reported success. Links
are now followed, and each real folder is visited once, so a link cycle ends.
A folder with no CSV and no converted Parquet raises
`conversion_no_source_files` (usually a broken link, an unmounted drive or the
wrong folder). A converted folder whose CSVs were removed is left alone, and
its conversion manifest is kept. An empty manifest would disown every
converted table. If that folder also has no manifest, the converter warns
`conversion_manifest_missing_rebuild_required`: restore or link the CSVs and
convert again to write it. Until then the loader reads the tables with its
recorded `table_scan_unmanifested` fallback.

## 2026-11-06 — urine receipt hours sit on the millisecond grid [lead-r13, `0cf09721d`]

**What changed.** Urine availability, chart and first-record hours are rounded
to the millisecond where the SQL and pandas load paths join.

**Why.** The two paths add the same second clocks in a different order and
published `*_available_at` hours 1e-14 h apart (float32 had hidden it).
Source clocks are whole seconds (AUMC milliseconds), so nothing is lost and
every load path publishes identical bytes.

## 2026-11-06 — urine24 keeps an exact 12 h record gap on every path [lead-r13, `d14e628ba`]

Deviation from ricu, by ruling. ricu 0.6.3 `urine24` fills the hourly grid
(`fill_gaps`) and slides 24 h; its `urine_sum` is NA below `min_steps`
(default `min_win` 12 h) and otherwise `sum(x, na.rm = TRUE) * step_factor /
length(x)`, so an uncharted hour counts as 0 and there is no gap limit. Round
4 ruled otherwise (R4-001, `ade4802d7`, 2026-11-04; extended to KDIGO and
SOFA-2 by RU-01/02/09): an uncharted hour is missing, a record is spread back
to the previous one over at most 12 h, and `urine24` needs 18 covered hours,
scaled by 24 / covered hours. v0.1.x and v6 predate this rule.

`ade4802d7` compared the gap in whole hour cells. `1f7e11164` (2026-11-04, in
`2786ea872` and r12) added the raw record gap in float hours, so that a
multi-record hour does not create a false gap. The record gap is a difference
of float hours (hour bin plus a sub-hour lag), so a gap of exactly 12 h comes
out as 12.000000000000007 or 11.99999999999999 depending on the lag.
`collection_intervals` (HiRID) has allowed 1e-6 h for this since `1a3b6ea76`.
The hourly-grid path used by every other database compared without it, so an
exact 12 h gap was a break or not by float noise. The DuckDB aggregation path
also stored the urine lag receipts as float32, which moved
`*_available_at` by up to 3e-5 h against the raw path and decided one such gap
the other way. Both paths now keep the receipts in float64 and use the
threshold helper, whose tolerance at 12 h is 12e-6 h (1e-6 × 12). Source clocks are whole seconds, so a real excess (≥ 1 s)
is still a break.

Hand check (MIMIC-IV sample, one stay, from raw `outputevents`): records
12 h 0 min 0 s apart in `charttime`; EasyICU's rule (gap ≤ 12 h kept) keeps the
24 h windows that span it; the float64 raw path dropped 35 hourly values, the
float32 DuckDB path kept them. Both now keep them.

**Errata.** FILL (stays and hours with changed urine24; SOFA renal and total
flips; KDIGO stage changes), per database, full cohort, against `2786ea872`.

## 2026-11-06 — vaso60 expands only rows that carry an infusion id [lead-r13, `cc20f06a3`]

ricu 0.6.3 expands each infusion once, in its item callback:
`mimic_rate_cv` is `expand_intervals(x, c(val_var, unit_var), grp_var)` per
CareVue `linkorderid`, `mimic_rate_mv` expands MetaVision rows from start to
`endtime`, and `hirid_rate_kg` expands per HiRID infusion. `vaso60` itself
(non-equi join of the rate rows onto the merged duration ranges,
`change_interval`, `aggregate(res, "max")`) carries nothing forward.
EasyICU's `vaso60` repeats the per-infusion minute expansion for rows that
come with an infusion id (`infusionid`, `orderid` or `linkorderid`). It
decided this from the column, not the row: MIMIC-III's MetaVision rate source
keeps `linkorderid` only while it has no rows, so a batch without MetaVision
rows handed CareVue rows (already expanded by `mimic_rate_cv`) an all-NA
`linkorderid` and expanded them a second time, as one null-id infusion. The
same stays in a batch that held MetaVision rows were not expanded again, so
the MIMIC-III result depended on batch composition (3,000-stay sample, batch
1,000 vs one shot: `epi60` in 16 stays, SOFA cardiovascular in 2). A row is
now expanded only when it has an infusion id. On the CareVue-only view the
second expansion of already hourly rows changed nothing: the 3,000-stay
CareVue sample is byte-identical before and after the fix (20 of 20 module
files); full view below.

**Errata.** Affected: MIMIC-III exports from FILL on (`2786ea872`, r12; v6
CareVue `97f508c6`: FILL; v0.1.2: FILL). CareVue full view (37,776 stays),
`2786ea872` vs this fix: FILL (stays and hours with a changed vasopressor
60-min rate, by magnitude; SOFA cardiovascular and total flips; Sepsis-3
onset changes). Full MIMIC-III with MetaVision: FILL.

## 2026-11-06 — a table stored as several files is read whole [lead-r13, `92d1876ed`]

A lead-r13 candidate regression, never in a release: since `3c7619188` the
storage policy resolved a table configured as several files (MIMIC-IV
`labevents/1..10.parquet`, `chartevents/1..30.parquet`) to its first file, so
DuckDB aggregations read one chunk. On a 3,000-stay MIMIC-IV sample most lab
concepts came back empty (potassium in 0 instead of 2,970 stays) without an
error; chartevents was affected only on a root without a chartevents bucket.
`2786ea872`, r12 and v6 are not affected. A multi-file table now resolves to
its directory, as `load_table` always did.

Every flat scan is now checked against the table's whole visible on-disk
inventory (files and Parquet footer row counts); a subset fails the load with
`TableScanError` (`table_scan_incomplete`). Bucket layouts keep their manifest
inventory check. The registry's file list (ricu's CSV-era chunk names) and
`num_rows` (MIMIC-IV 2.x) describe no particular installation and are not used
as the inventory.

## 2026-11-06 — Sepsis-3 (SOFA-2) reads unscored hours as 0 on every path [lead-r13, `8c948c740`]

ricu 0.6.3 `sofa_score()` fills the hourly grid (`fill_gaps`), takes the
24 h worst value (`slide`) and sums the available components
(`rowSums(.SD, na.rm = TRUE)`), so an hour without any component scores 0;
`sep3()` compares against the running minimum (`delta_cummin`) over
[suspected infection − 48 h, + 24 h] (`si_lwr`, `si_upr`).
EasyICU's published `sofa2_score` total follows this. `sep3_sofa2` read that
total when SOFA-2 had been published by an earlier worker (streamed or
separately batched exports), but read the concept, where such hours are
missing, when both modules shared one worker. A stay whose look-back reached
unscored hours could then get a later onset or none. The callback now reads
unscored hours as 0 on both paths. SOFA-1 Sepsis-3 was not affected.

**Errata** (re-derived from each export's own published SOFA-2 total):
r12 full MIMIC-IV 0 of 42,364 onsets changed; v6 AUMCdb <10 of 5,178 changed
(1-24 h, all earlier after the fix); v6 HiRID has no suspected infection.
`2786ea872` full: FILL.

## 2026-11-06 — SICdb death times are not realigned per batch [lead-r13, `9764fb7e9`]

Pre-existing (also in `2786ea872`, r12 and v6). SICdb death time is
`(OffsetOfDeath − ICUOffset) / 3600` hours from ICU admission, floored to the
hour: ricu 0.6.3 `data-sources.json` makes `ICUOffset` the start of the SICdb
ICU stay and lists `OffsetOfDeath` among the `cases` time variables, and
`load_difftime.sic_tbl` re-bases those second offsets to that start. ricu has
no SICdb item for `death` itself. The SICdb alignment fallback treated any
undeclared time column whose batch maximum exceeded 5,000 as raw seconds and
subtracted `ICUOffset` again, so every death in a batch holding one late
registry death moved earlier by its ICU offset. `sic_death` now declares its
column as ICU-relative hours.

**Errata.** v6 and r12 full SICdb: 2,089 of 2,107 registry deaths had an early
`death_time` (median 180 h early; 1-24 h 276, 1-7 d 731, 7-30 d 792,
> 30 d 290). Derived columns: FILL (mort_28d/90d/365d, followup_days).

## 2026-11-06 — HiRID death and length-of-stay timing skip the full sort [lead-r13, `30088a639`]

The order-restoring read of `observations` sorted the whole table before the
cohort filter (161 s and 21 GiB for one stay). These readers take the per-stay
maximum, which does not depend on order, so they now read unordered (0.6 s,
0.3 GiB). No value change.

## 2026-11-05 — id_tbl concepts keep every row of the stay [lead-r13, `bae85e40f`]

Static (id_tbl) concepts such as weight and height are now aggregated over all
rows of the stay, as ricu does; they are no longer cut to the ICU episode
window [-24 h, LOS + 24 h]. Time-series concepts keep the window. The
aggregation rule itself is unchanged: the per-stay median of the bounded
rows, as in ricu.

Full cohorts, final lead-r13 against `2786ea872` (identical against the r12
tip `088501402`); counts 1-9 are shown as <10:

| Database | Concept | Stays gaining a value | Stays with a changed value (<1 / 1-5 / 5-20 / >=20) | Cause |
|---|---|---|---|---|
| MIMIC-IV | weight | 0 | <10 (<10 / 0 / 0 / <10) | read-column fix |
| MIMIC-IV | height | 0 | 0 | — |
| eICU | weight | 256 | 2,091 (1,092 / 843 / 143 / 13) | no ICU window for id_tbl |
| eICU | height | 0 | 0 | — |
| MIMIC-III (directories) | weight | 142 | <10 (0 / <10 / 0 / 0) | read-column fix |
| MIMIC-III (directories) | height | 0 | 0 | — |
| HiRID, AUMCdb, SICdb | weight, height | 0 | 0 | — |

No stay lost a value. Splitting through the windowed r13 build (`110b37ea5`):
the read-column fix alone gives the MIMIC-IV and MIMIC-III rows above, and
the window removal alone gives the eICU row (eICU id_tbl concepts always took
the windowed raw path before). The windowed build had dropped 1,681 MIMIC-IV
heights and 119 weights; the no-window rule restores exactly those.

Not changed, listed separately: id_tbl output can still hold values outside
the concept bounds where a stay has only such rows (for example MIMIC-IV
weight above 500 kg: 14 stays; SICdb weight below 1 kg: 33 stays). ricu
applies the bounds before the median too; this predates lead-r13.

**Errata.**
1. Before lead-r13 the single-file layout cut id_tbl rows to the ICU window
   while directory and bucket layouts did not, so the same source could give
   different weight/height by layout (eICU was windowed in every layout).
2. The v6 CareVue export (`97f508c6`, single-file view) was cut to the
   window too, but the #641 view holds no weight or height row outside it
   (0 of 1,842,168 weight and 0 of 168,127 height chartevents rows), so v6
   lost nothing: final lead-r13, `2786ea872`, r12 and the v6 run641 export
   agree on all 37,776 view stays (25,200 weights, 17,068 heights; 0 gained,
   lost or changed). Aggregation uses only rows inside the view: the view's
   MetaVision `inputevents_mv` has no `icustay_id` and contributes no weight,
   and no stay outside the view is read.

## 2026-11-05 — one load path for every Parquet table: follow-ups [lead-r13, `397853b79`, `4c4090923`, `7c3e292a5`, `f96f9e122`, `aa79c777a`, `83d26bc0e`]

No value changes where nothing failed; the full-cohort exports are
byte-identical across layouts, engine threads and the no-perf build.

- An all-empty concept keeps its time axis; `circ_failure` treats an empty
  lactate or MAP as absent data instead of `circ_failure_load_failed`. Empty
  optional drug streams still stay outside the evidence (graded levels
  unchanged).
- MIMIC-III hospital tables joined to ICU stays cast the event time, so a
  CSV-converted string `charttime` no longer fails the join.
- The MIMIC alignment helper column `intime` no longer leaks into concept
  output.
- New failure codes: `mimic_age_intime_load_failed` (the shipped MIMIC-III
  age path made every age missing when ICU admission times could not load;
  this never triggered in 2786, r12, r13 or the v6 CareVue export: age is
  complete for all 61,532 and 37,776 stays), `vasopressor_weight_load_failed`,
  `rrt_dependency_load_failed`, `raw_concept_load_failed` (`0532e9a30`).
- An absent eICU optional table is `MissingTableSourceError`; any other
  `KeyError` raises `eicu_optional_channel_read_failed`.
- Only a source aggregated by DuckDB supplies the time axis of an all-empty
  concept (`f96f9e122`); a raw source's untyped `datetime` column no longer
  stands in for it.
- A storage resolver error fails the load instead of silently switching it to
  the raw path (`aa79c777a`).
- A module worker killed for memory is retried with half its batch, down to a
  single stay, and the retry is logged (`stream_batch_oom_retry`) and recorded
  in `stream_retry_history`. A one-shot the planner chose from a stale profile
  retries streamed from half the cohort when the caller has an output
  directory and allows streaming (`83d26bc0e`). eICU medications needed this:
  since `089701e34` its single-file tables take DuckDB without the table
  cache, and the one-shot measured at `486bb169` (11.2 GiB) no longer fits
  14 GiB.
- Dictionary (sealed): the MIMIC-III and MIMIC-III demo `other_output_ml`
  lists named 229413/229414 (Chest Tube #3/#4), which exist only in MIMIC-IV
  (MIMIC-III v1.4 ends at itemid 228647; 0 of 4,349,218 MIMIC-III
  outputevents rows). Removed; no value change.

## 2026-11-05 — results no longer depend on the storage layout [lead-r13, class a; `245e22dbc`, `c6fa1e4be`, `089701e34`]

The same source could give different values as a single parquet file, a flat
directory or a bucket cache. Every database with directory or bucketed tables
was affected; the counts below are from the CareVue 3,000-stay sample, and
full-cohort counts per database follow in this entry once the lead-r13 full
runs finish.

- **id_tbl weight and height.** A DuckDB fast path, used only for directory
  and bucketed tables, read the string `value` column instead of the table's
  `val_var` (`valuenum` on MIMIC chartevents), so rows carrying only
  `valuenum` were dropped. It also took the per-stay median before the concept
  bounds. Affected: MIMIC-IV weight; MIMIC-III weight and height read from a
  directory layout. eICU weight was not affected (the missing `value` column
  failed the fast path, which then fell back to the raw path). HiRID
  weight/height, AUMC weight and MIMIC-IV height read the right column; they
  differ only where a stay has an out-of-bounds value. `245e22dbc` fixed the
  column and `c6fa1e4be` removes the fast path: id_tbl concepts always take the
  raw path. CareVue sample, bucketed vs single-file: 12 stays had lost their
  weight, <10 their height, and <10 stays lost per-kg norepinephrine rows.
  The v6 CareVue export (`97f508c6`) read a single file and was not affected
  by the column error.
- **Path and cache by layout.** A single parquet file took the raw pandas path
  (float64) and a per-patient table cache, while directories and buckets took
  DuckDB (float32). Derived columns (shock indices, white-cell ratios, P/F,
  driving pressure, fluid balance, eGFR) differed in the last float32 digit,
  and with the cache a later concept on the same table bypassed DuckDB (AUMC
  differential counts). `089701e34` chooses the path and the cache by whether
  the table is parquet-backed, whatever its layout. The cache also no longer
  holds whole patient batches of a single-file chartevents, which ran the
  CareVue single-file view out of memory at 14 GiB.
- Tests: every dictionary source on a bucketed table, with values only in its
  own value column, gives the same concept output and byte-identical module
  exports in the single-file, directory and bucket layouts.

## 2026-11-05 — swallowed failures now fail loudly [lead-r13, `55dc5b853`]

Callbacks and loaders that caught an exception and published a whole batch as
NA (or an empty frame) now raise with a reason code; a stay that truly has no
data keeps its documented NA rule. No value changes where nothing failed.
Codes: `hirid_rate_kg_weight_load_failed`, `sic_rate_kg_weight_load_failed`,
`wbc_ratio_calculation_failed`, `vent_mode_map_unavailable`,
`callback_concept_load_failed`, `urine24_urine_load_failed`,
`concept_component_merge_failed`, `sub_concept_load_failed`,
`eicu_optional_channel_read_failed`, `sicdb_cvp_arterial_reference_read_failed`,
`circ_failure_dependency_load_failed`, `aumc_rate_units_expand_failed`,
`renal_aki_bundle_load_failed`, `circ_failure_load_failed`,
`comorbidity_load_failed`, `outcomes_load_failed`, `microbiology_load_failed`,
`special_concept_merge_failed`, `sepsis_dependency_load_failed`,
`hospital_table_hadm_mapping_failed`, `hospital_table_stay_id_fill_failed`,
`mimic3_chartevents_csv_read_failed`, `partitioned_read_failed`. The sepsis
dependency load no longer retries SOFA-2 after a failure. This layer does not
re-raise: a failed SOFA-2 dependency is written to the structured failure
ledger (`sepsis_dependency_load_failed`) and the module returns an empty
result.

- `circ_failure` asked for `levo_rate` and `theo_rate`, which are not in the
  dictionary; the old per-concept fallback hid that. Optional drugs absent
  from the dictionary are now skipped (`710538306`), and a stay without a
  boolean drug record (HiRID milrinone) is filled with False, not 0.0
  (`7a0ed4ba1`).

## 2026-11-05 — MIMIC-III per-kg rates: a weight-load failure is an error [lead-r13]

- `mimic_kg_rate` loads each stay's weight before dividing a rate by it. When
  that load raised, the callback logged a warning and published every per-kg
  rate of the batch as NA. It now fails the module with
  `mimic_kg_rate_weight_load_failed`. Stays that truly have no weight keep the
  round-3 NA rule (R3-20).
- The failure seen was in the CareVue view: its MetaVision `inputevents_mv`
  has no `icustay_id`, so that weight source returns an empty table without
  the value column, and `convert_unit` raised `KeyError: 'weight'`. An empty
  source now passes through `convert_unit`; rows without the value column
  still raise.

## 2026-11-05 — round-12b tightenings: eICU antibiotic HICL and icd_abx timing [`98b58496d`, `1a94804e0`]

- **eICU antibiotic HICL branch** (eICU and eICU-demo). It now admits only
  orders without a `drugname` (`require_empty_var`); named orders are counted
  only by the explicit name source. Before, a named order whose name failed the
  antibiotic pattern also entered through its HICL code: under 10093, <10 rows
  labelled as a 0.9% saline bag. All 21 codes are antibacterial (round-12
  composition check), so S6 (`63ee291db`) stays a deliberate change.
- **icd_abx suspected infection (R3).** The ICD-plus-antibiotic branch took the
  time of every antibiotic row. It now takes only rows whose antibiotic value
  is true or > 0, as the antibiotic/sampling path already did; a False, zero or
  unknown row no longer times a suspected infection.

## 2026-11-05 — round-12 basis records: bounds vs v6 and P/F (no value change)

### Bounds that differ from v6 (97f508c6c)

R2 freezes the bounds of the round-5 baseline (`cd2bae05d`). Every change
below is an ancestor of that baseline and came from an audit correction, so
none breaks R2. Each concept's `_comment` states the physiological basis.

| Concept | v6 min/max | Now min/max | Commit (audit) | Basis |
|---|---|---|---|---|
| `dbp` | 0/200 | 1/200 | `3a296c1f9` (round 3 M) | <= 0 mmHg is a disconnected or zeroing transducer or cuff |
| `map` | 0/250 | 1/250 | `3a296c1f9` (round 3 M) | as `dbp` |
| `sbp` | 0/300 | 1/300 | `3a296c1f9` (round 3 M) | as `dbp` |
| `plateau_pres` | 0/100 | 1/100 | `3a296c1f9` (round 3 M) | <= 0 cmH2O during an inspiratory hold is not a measurement |
| `tidal_vol` | 0/2000 | 1/2000 | `5a5eac84b` (round 3) | 0 mL is no breath or a disconnected sensor |
| `epi_rate` | 0/1.5 | 0/5 | `5a5eac84b` (round 3) | refractory shock exceeds 1.5 mcg/kg/min; > 5 is a unit error |
| `norepi_rate` | 0/3 | 0/5 | `5a5eac84b` (round 3) | refractory shock exceeds 3 mcg/kg/min; > 5 is a unit error |
| `ins` | 0/none | 0/50 | `5a5eac84b` (round 3) | IV insulin rarely exceeds 20-30 U/h; > 50 is a bolus or unit error |
| `icp` | 0/100 | -10/100 | `ade4802d7` (R4-112) | ICP is negative upright or with CSF over-drainage |
| `plt` | 5/1200 | 1/1200 | `ade4802d7` (R4-158) | counts of 1-4 x10^9/L are real in severe thrombocytopenia |
| `hbco` | none | 0/100 | `b21030f26` (R4-135/R4-174) | a percentage of total haemoglobin |

New split/raw concepts that have no v6 counterpart carry the same thresholds
as their parent: `map_invasive`, `map_nibp` (min 1) and `plateau_pres_raw`
(min 1). The AUMC blood-pressure sources keep `na_below(1)` (6641/6642/6643,
6678/6679/6680, 8841/8842/8843); it equals the concept minimum and is not a
separate cut.

### P/F FiO2 aggregation deviates from ricu [G5, `474f34ddb`]

- ricu's `pafi` pairs the hourly minimum PaO2 with the hourly **maximum**
  FiO2. EasyICU uses the hourly **median** FiO2 (PaO2 stays the minimum).
- Reason: the median is closer to the FiO2 at gas sampling, and it resists
  transient peaks such as pre-oxygenation before suctioning.
- Direction against ricu: in any hour, P/F is equal or higher, so the SOFA-1
  and SOFA-2 respiratory components are equal or lower. The effect is largest
  where FiO2 is charted often (HiRID P/F median 184 -> 221 mmHg; AUMC
  215 -> 220; MIIV and eICU < 2%).

## 2026-11-05 — follow-up assumptions are cited or flagged [round 12; `951d566b6`, `9802ce1fa`] (new columns only)

New outcome columns `mort_28d_assumption_based`, `mort_90d_assumption_based`
and `mort_365d_assumption_based` are TRUE where a survivor is known only
through an undocumented follow-up assumption. They are FALSE for recorded
deaths and documented follow-up, and missing where the endpoint is missing.
`mort_*` and `followup_days_*` values do not change. A paper should use the
flagged cells of these databases only in sensitivity analyses.

- **AmsterdamUMCdb.** A missing `dateofdeath` is taken as one year of survival
  from ICU admission. No official source supports that horizon: the admissions
  table documents `dateofdeath` as "the date on which the patient died ... if
  applicable (non-null)" (AmsterdamUMCdb wiki, admissions). The descriptor
  paper (Thoral et al., Crit Care Med 2021) reports deaths < 1 year after
  discharge but no follow-up method. Every AUMC survivor with no valid
  `dateofdeath` after the existing death-date checks is flagged; this includes
  <10 stays whose raw date is present but fails those checks.
- **SICdb.** The categories are documented, so they keep the values unflagged:
  - `d_references` labels 3076 "6 Months" and 3077 "1 Year" for
    `EstimatedSurvivalObservationTime`.
  - The SICdb wiki (Mortality): "For technical reasons observation time is only
    6 month for some patients".
  - PhysioNet SICdb 1.0.8: `OffsetOfDeath` "is capped at one year" from
    admission and includes out-of-hospital deaths from governmental data.
  - The 2023 descriptor paper counts the year from hospital discharge; the
    admission start used here is the shorter, conservative reading.
  - The one-day grace that lets a survivor whose ICU transfer came hours after
    admission reach the horizon is not documented, so survivors who need it
    are flagged.

Flagged stays in the full databases (round-12 code, aggregates only):

| Database | Stays | Horizon | Flagged (TRUE) | Survivors (`mort_*` FALSE) | `mort_*` missing |
|---|---|---|---|---|---|
| AUMC | 23,106 | 28 d / 90 d / 365 d | 15,364 at each | 20,275 / 19,369 / 18,331 | 0 |
| SICdb | 27,386 | 28 d / 90 d | 0 | 25,200 / 24,085 | 0 |
| SICdb | 27,386 | 365 d | 11,263 | 19,608 | 3,189 (6-month category) |

No recorded death is flagged in either database.

## 2026-11-05 — SOFA-2 all-missing flag and one SOFA owner for web exports [GW1, GW2, `4cc1de6bf`]

- Native-v2 research export: `sofa2_score` gains `sofa2_all_components_missing`,
  TRUE in hours where all six SOFA-2 components are missing, so the total is a
  pure normal-value imputation (0, as ricu's na.rm sum). Existing columns keep
  identical values, dtypes, order and rows (6-database regression).
- Web export job (`webserver/dataio.py` `make_export_runner`): it used to call
  `api.load_concepts(..., use_sofa2=..., merge=True)` for the SOFA modules. That
  requested the six organs as separate point concepts, so point states replaced
  the aggregate callback's rolling companions and changed the row set. On the
  eICU demo the grouped and web exports disagreed: SOFA-1 189,380 vs 189,401
  rows and SOFA-2 179,859 vs 178,536 rows (GW1).
- The web job now calls the grouped export's owner,
  `api/extraction._load_module_concepts`, and reuses the native total rebuild.
  Web and native SOFA-1/SOFA-2 exports therefore share one definition: the same
  row keys, totals, six organs and flag
  (`tests/core/test_grouped_web_sofa_gw1.py`). Web receipt dtypes are kept.
  This is intended (GW1 ruling: "keep the web-side fix"), not a side effect.

## 2026-11-05 — AUMC D-dimer is published in ng/mL [round 12, R7]

- The `d_dimer` column's canonical unit is ng/mL. AUMC 10393/12163 values
  (mg/l, R5C-AUMC-011) were published unconverted under that label. They are
  now converted by mass (1 mg/l = 1000 ng/mL), so label and value agree.
- Only the mass unit changes. The FEU/DDU reporting basis stays unverified and
  unconverted, so AUMC and MIMIC assay scales are still not harmonized.
- Values above 50 mg/l now exceed the concept's 50,000 ng/mL bound, as in every
  source, and are dropped (bound kept, round-12 ruling). Two denominators
  (round-12b recount):
  - full source, items 10393/12163: 25 of 932 rows, 20 stays, maximum 239 mg/l;
  - within 24 h before ICU admission to 24 h after discharge: 24 of 782 rows.

## 2026-11-05 — eICU order-derived norepinephrine is marked [V1, `fa80e951e`] (new column only)

- `norepi_dur` in eICU takes the order window of a norepinephrine order for
  stays whose hospital charts no norepinephrine infusion (V1). That span is
  when the order was active, not when the pump ran, and `vaso_ind` and the
  SOFA cardiovascular components inherit it.
- New `norepi_order_dur` (vasopressors module, eICU only) reads the same order
  sources with the same callback and repeats only those order-derived spans,
  on the same rows and clock as `norepi_dur`. A `norepi_dur` row with a
  missing `norepi_order_dur` comes from a charted infusion, so an analysis can
  keep charted infusions only as a sensitivity analysis.
- Existing columns do not change. Other databases have no order-derived spans
  and no values in the new column.

## 2026-11-05 — diltiazem is IV only [R5C-HIRID-012 partly withdrawn] (pending lead reseal)

- Round-11 ruling: `diltiazem` is intravenous exposure in every database,
  as S8 (`fb09aea7a`) and eICU `eicu_iv_exposure` already assume. Oral
  diltiazem is mostly chronic rate control, not an acute haemodynamic
  exposure, and mixing routes makes the column incomparable across databases.
- HiRID 1001071/1000331 (tablets), which R5C-HIRID-012 added under an
  any-route reading, are unmapped again. HiRID has no IV diltiazem item, so
  the concept is unavailable for HiRID. No separate oral concept is created
  while nothing consumes it.
- AUMC 7174 'Diltiazem (Tildiem)' is ordered under 'Niet iv
  Circulatie/Diurese' (not IV), so it is unmapped too.
- SICdb 1817 is inferred to be oral from its dose strengths, because the
  medication table has no route column: 121 rows in 35 cases, all single
  doses, Amount p10/p50/p90 = 45/90/120 mg. These match 60/90/120 mg tablets
  rather than an IV bolus of about 20 mg, so it is unmapped too. Diltiazem remains
  for MIMIC-IV/MetaVision 221468, CareVue 30115 (mg/hr infusions) and eICU
  IV exposure (catalog coverage 6 -> 3).

## 2026-11-05 — MIMIC-IV ED lab attribution [ED1, ED2-1]

- NULL-`hadm_id` labevents use `subject_id` and the existing inclusive window
  from 24 h before ICU admission to 24 h after ICU discharge. Round 5 already
  recovered rows with one candidate; overlapping windows now use the hospital
  rolling rule instead of dropping the row: choose the first candidate stay
  whose outtime is not before charttime, or the last candidate if all outtimes
  precede charttime. Equal outtimes use the lowest stay ID, as admission-linked
  labs already do. Each source row goes to exactly one stay. Consider all
  stays/admissions of the subject before applying requested stay/admission
  filters.
- **[ED2-1]** Hospital admission intervals take precedence over this subject
  fallback: if a NULL-admission lab's charttime lies in a source `admissions`
  interval `[admittime, dischtime]` (inclusive), restrict candidates to ICU
  stays of that admission, keeping the same inclusive ICU window and rolling
  rule. With overlapping hospital admissions, roll over the eligible stays of
  those admissions only. If none of their ICU windows contains the lab (including
  admissions without an ICU stay), drop it; never fall back to an ICU stay of
  another admission. Outside all admission intervals, keep the ED1 subject/window
  rule. Source admission timestamps may be strings and use `TRY_CAST` to
  timestamps; partial/demo sources without admissions have no known intervals.
- This hybrid rule is an explicit exception to ricu: ricu loads MIIV labevents
  by `hadm_id` and `upgrade_id` rolls on `(hadm_id, time)`, so NULL-admission
  rows never acquire a stay. It also differs from MIT-LCP mimic-code's
  subject/window attribution, which counts a row for every overlapping stay.
  The rationale is to recover pre-ICU ED labs, including baseline creatinine,
  without double counting or moving an inpatient lab to another admission.
- Pandas, preloaded, single-concept and multi-concept DuckDB loads consult the
  same `_lab_ctes` owner. Subject-level pushdown retains NULL-admission labs;
  raw `hadm_id` stays NULL and `attributed_hadm_id` records the selected admission.
  Admission-linked assignment and cross-channel deduplication are unchanged.
  `ambiguous_null_hadm_rows` still counts rows with overlapping ICU windows,
  and `unmatched_null_hadm_rows` still counts rows with no such window, both
  before the admission restriction. Two additional counters record its effect:
  `null_hadm_rows_in_other_admission_reassigned` counts rows whose selected
  admission changes from the unrestricted ED1 winner, before cross-channel
  deduplication; `null_hadm_rows_in_other_admission_dropped` counts rows with
  ICU-window candidates but no eligible stay after the admission restriction.
  `attributed_null_hadm_rows` and `deduplicated_null_hadm_rows` retain their
  meanings for the final assignment.
  Validation uses synthetic records only; real-data coverage awaits lead checks.

## 2026-11-05 — UC2 separate urine chemistry module (pending lead reseal)

The 20-module catalog publishes the five UC1 concepts only in
`renal_urine_chem.parquet`: `urine_na`, `urine_crea`, `urine_osm`, `urine_urea`
and `sic_precondition_renal_dysfunction`. The original `renal` roster and its
73 concepts remain unchanged from the pre-UC1 catalog. Urine-only observations
must not extend its hour grid. Other module Parquet bytes also remain unchanged.

The additional module is available through the ordinary lab extraction group,
native-v2 manifests, release controller and sealer, static catalog and both
extraction screens. It is not a core module and starts unselected. Unsupported
sources publish an empty typed file; observed rows require at least one value.
The SICdb case-level flag retains the ordinary `id_tbl` publication semantics.
UC1 definitions, mappings, units, bounds and the complete Planner allowlist are
unchanged. Dictionary category `renal` describes the concepts; the publication
module is owned by the shared catalog, independently of that category.

## 2026-11-05 — UC1 urine chemistry and renal precondition (pending lead reseal)

- **[UC1-K5]** The urine chemistry module adds `urine_na` (mmol/L, 0–400),
  `urine_crea` (mg/dL, 0–1,000), `urine_osm` (mOsm/kg, 0–2,000) and
  `urine_urea` (urea nitrogen mg/dL, 0–5,000). MIMIC-IV (including its demo
  loaded as `miiv`) and MIMIC-III/demo read labevents 51100, 51082, 51093
  and 51104 respectively. eICU/demo read the exact `urinary sodium` and
  `urinary creatinine` labels. These are concentration assays; serum assays
  and daily excretion totals cannot enter these columns. Existing declared
  row filters require compatible unit labels before bounds and aggregation.
  Sodium mEq/L is explicitly relabelled by the identity conversion to mmol/L;
  creatinine, osmolality and urea nitrogen remain on their native scales.
- eICU `urinary osmolality` uses mOsm/L. The repository has no policy treating
  osmolarity as mOsm/kg, so it remains unmapped. AUMC, HiRID, SICdb and community
  sources remain unmapped where no checked-in dictionary verifies the name,
  urine specimen and unit together. No source identifiers are guessed.
  FENa/FEUrea are deferred: existing arithmetic callbacks do not pair each urine
  specimen with a nearest plasma sodium/creatinine sample within 24 hours.
- **[UC1-K6]** `sic_precondition_renal_dysfunction` reads data_ref FieldID
  3104 as a case-level precondition. The existing SICdb source callback owner
  decodes RefID using d_references Yes/No labels. Missing, unrecognised and
  ambiguous references, or conflicting Yes/No records, stay unknown. The
  source schema now retains FieldID and explicitly identifies CaseID.
  Older converted tables lacking FieldID report the concept unavailable.
  This concept is SICdb-specific: Charlson renal and Elixhauser renal-failure
  flags in `scores/comorbidity.py` include hospital ICD codes for unspecified
  failure, transplant and dialysis status without proving pre-ICU onset.
- Catalog membership, audited coverage, static catalog and community coverage
  are updated. The Planner's concept allowlist uses a comma-separated roster
  to retain every key within the existing resource ceiling. Source, dictionary,
  resource-context and Figure 2 digest pins are left for the lead's approved reseal.

## 2026-11-05 — extraction audit round 8 (last batch; no dictionary change)

- **[R7-F005 follow-up]** MIMIC-III source alignment floors minutes from whole
  seconds again (`floor(seconds / 60)`); `floor(seconds / 3600 * 60)` put an
  8h10m offset at 8h09m when the record seconds equalled the intime seconds.
  Exact hours for the availability remainder are computed separately
  (`ecdc3840d`). Sweep of every minute-flooring path in `src/`: no other hit
  (the others divide whole seconds directly or floor at hour level, where
  whole hours convert exactly).
- **[R7-F018]** eICU: a recorded Sample Type that names neither specimen
  (Unknown, Unspecified, blank) is `excluded_recorded_unknown` before the
  label-defined and same-panel fallbacks, which now apply only where no sample
  was recorded (`9eb5b084f`).
- **[R7-F009]** HiRID vasopressor episodes (NE, DOBU, EPI) merge gaps of at
  most five minutes on the source clock, not on float hours (`c7b166e92`).
- **[R7-F010]** HiRID ventilation intervals are differenced on the source
  clock before conversion to hours (`c7b166e92`).
- **[R7-F029]** HiRID fluid-counter readings are restored to whole source
  seconds before the noon reset day is assigned (`c7b166e92`).
- **[R7-F003]** AUMC: the high-frequency pre-sampling no longer re-buckets on
  the absolute clock (it moved potassium into the preceding hour above 1,000
  rows), so the formal SOFA-2 module and the single and batch owners agree
  (`13adc5539`). AUMC high-frequency concepts are aggregated once, after
  admission alignment and bounds.
- **[R8-WIDE-MERGE]** Multi-concept requests that include exact-start
  duration, interval or rate concepts use the window materializer even when
  no numeric wide batch was produced (`13adc5539`).
- **[R7-F030, ruling R8]** MIMIC hospital-level rows (for example MIIV
  `prescriptions`) in an admission with several ICU stays follow the ricu
  rolling join on `starttime` (roll = -Inf): each row belongs to exactly one
  stay, the first stay of the admission whose `outtime` is not before the
  row's `starttime`; a row between two stays goes to the next stay, a row after
  the last discharge to the last stay. That stay's episode window (24 h before
  admission, 24 h after discharge) then applies. A prescription is not split
  across the stays its interval overlaps. Verified with a synthetic
  two-stay admission through the real loader; no code change.

## 2026-11-05 — extraction audit round 7 (eleventh dictionary reseal)

- **[R7-F001]** MIMIC-III hours whose urine rows all lack a value stay missing
  instead of summing to 0 (`951987380`, landed with the round-5 reviews).
- **[R7-F002]** The AUMC/HiRID high-frequency pre-sampling keeps the concept's
  requested aggregator; HiRID PaO2 reaches P/F as the hourly minimum, not a
  median (`b64dfb4d0`). P/F can fall where an hour has several PaO2 values.
- **[R7-F003]** SOFA-2 renal: hyperkalaemia and acidosis are independent RRT
  criteria routes; an absent optional input no longer disables the other
  (`b64dfb4d0`). Scores of 4 can appear where one route was blocked.
- **[R7-F005]** MIMIC-III values stay on the ricu minute clock relative to ICU
  intime, but urine availability keeps the sub-minute intime remainder, so
  `*_available_at` never precedes the source record (`e8684f304`).
- **[R7-F008, F010, F011, F012, F030]** CareVue 582 RRT starts/stops replayed
  as courses; HiRID ventilation keeps exact starts; MIMIC-IV windows selected
  by overlap; CareVue vasopressin/phenylephrine support shares the rate
  callback's timing and stops; MIIV prescriptions need a positive dose or a
  given status on a valid interval (`f4e4a848a`).
- **[R7-F009]** HiRID norepinephrine durations keep exact starts, so an hour
  with several infusion starts publishes each (`930ca2d28`).
- **[R7-F013]** eICU hours with invasive and non-invasive ventilation evidence
  publish `invasive|noninvasive` instead of the last row (`5ce8c9ea5`).
- **[R7-F014]** MIMIC-III CareVue rates are numeric and ordered on the raw clock
  before hourly pooling; results no longer depend on the table scan order
  (`3dfd00a2b`).
- **[R7-F029]** HiRID output-counter rises faster than 5,000 mL/h that the next
  reading undoes are spikes (`b64dfb4d0`).
- **[R7 rulings]** `cai_ph74` (AUMC pH 7.4 corrected ionized calcium) is its own
  blood_gas column (`110380205`); eICU D50 dose text reads gram spellings
  (`7b49661c9`); `dex`, `mech_vent` and `vent_ind` record the eICU D50 and
  ventilation-clock rulings in `_comment`.

## 2026-11-05 — extraction audit round 7 urine receipts

- **[R7-F005]** MIMIC-III mixed numeric/datetime source batches now retain
  seconds when aligning chart times to admission, matching the ordinary
  datetime path already fixed by `09dffb5341`. Hourly pooling retains the
  latest contributing chart/store time; availability comparisons use an
  absolute one-second tolerance, independent of length of stay.
- **[R7-F006]** Native renal packages in all six databases retain
  `uo_6h/12h/24h_assessment_rate`, `_assessment_covered_h` and
  `_assessment_available_at` from the scoring owner. Typed metadata records
  each assessment window, rate units, covered hours and ICU-relative
  availability. Descriptive partial windows keep their existing meaning.
- **[R7-F007]** KDIGO retains null and negative urine records as collection
  breakpoints. A same-time invalid channel also breaks coverage, matching
  shared scoring assessments; valid zero volumes still count as observation.
- **[R7-F024, REFUTED]** Retrospective urine values remain paired with their
  actual availability; prediction analyses must enforce `available_at <= t`
  at their declared evaluation time. No second real-time value is required.
- **[R7-F025, REFUTED]** The eICU subsecond clock warning can result from
  float32 storage quantization. Compare source clocks using absolute seconds
  and check float32 representability before attributing a discrepancy to
  lost source timestamps. Strict subsecond gates can use higher precision or
  a conservative margin accounting for storage quantization.
- **[R7-F026, REFUTED]** A rejected collection interval over 12 h contributes
  no volume or coverage. Historical accepted intervals can still support
  the separately flagged 18–23 h `urine24_est` early in a record gap;
  clearing these valid partial estimates would change the agreed R1 rule.

## 2026-11-04 — landed-code reviews of round-5 lanes (tenth dictionary reseal)

Commits `930ca2d28` through the tenth reseal commit on dev9.

- **[RI-01..03, R6-F005/F012/F013/F015/F018/F021..F023/F035, RR-01..03]**
  A partial wide batch (four or more concepts on one long table) now merges
  the remaining concepts in every database, not only AmsterdamUMCdb.
  Previously mapped measurements, arterial gases, urine values and their
  receipts, and interval concepts such as `dex` could be published empty
  beside the batch. Requests holding windows, durations or rates take the
  window materialiser, so duration starts are no longer floored to the hour
  in merged output.
- **[RI-04]** Furosemide equivalents: MetaVision item IDs and AmsterdamUMCdb
  `iscontinuous` separate infusions from boluses; a malformed continuous
  window contributes no dose instead of a bolus at its start.
- **[R6-F022]** Missing signed urine terms are dropped before summing rather
  than counted as a measured zero.
- **[RR-01..03]** An eICU paO2/paCO2 result outside its concept bounds no
  longer anchors co-measured pH, BE or saturation.
- **[RR-07]** MIMIC-IV eMAR copies of inputevents doses are matched one to
  one within 60 minutes; bucketed `emar_detail` is read as well.
- **Disclosures (auditor round 7):** eICU has no arterial tCO2 source;
  `vaso_ind` and `norepi_equiv` combine different vasopressors by database
  (AmsterdamUMCdb records no vasopressin, HiRID no dopamine or
  phenylephrine); terlipressin is not mapped as vasopressin.

## 2026-11-04 — extraction audit round 5 residue batch and round 6 (ninth dictionary reseal)

Commits `1823ea413` through the ninth reseal commit on dev9.

- **[RB-01..RB-07, R6-F009]** Row existence no longer makes a value TRUE.
  MIIV auxiliary vasopressors, respiratory flow/volume/pressure settings,
  circulatory-support numeric channels, 168 remaining medication and blood
  product sources, eICU infusion drug names and eICU blood product/dialysis
  intake-output rows all require a positive (pressures: finite non-zero) value.
  MIIV/MetaVision `Rewritten` rows are excluded throughout.
- **[RB-08]** AUMC 6658/20662 treat `Uit` (off) as negative.
- **[RV-01..RV-05]** MIIV `Rewritten`, MIMIC-III `ISERROR=1`, and cancelled
  procedure rows are excluded on every path (projection, CSV, Arrow, bucketed
  and batch SQL); NULL status rows are kept. Static weight applies its chart
  quality condition before the median. A HiRID medication table without
  `recordstatus` raises a typed unavailability error.
- **[RI-01..RI-07, R6-F011..F015]** Interval expansion publishes only hours
  with positive half-open overlap (no zero-length or whole-hour-endpoint
  hours) on per-concept, batch and native paths. 228340/7244 doses are split by
  ICU-hour overlap; SOFA-2 scores the t-23..t window without unbounded fill;
  LOCF/LOCB `max_gap` is measured from the last valid observation. eICU
  infusion durations check for a positive rate before segmentation.
- **[RR-01..RR-03, R6-F016; R6 refined by the auditor]** Arterial gas
  concepts (`po2`, `pco2`, `ph`, `sao2`, `be`, `tco2`) admit a row in this
  order. (1) A recorded specimen decides first: venous, capillary, mixed or
  central samples are excluded, arterial ones admitted. (2) Without a
  per-sample specimen field, an item or analyte that is arterial by name is
  admitted (eICU paO2/paCO2; MIMIC/MIIV "Arterial ..." chart items; HiRID
  arterial variables). (3) pH, base excess/deficit and O2 saturation, which do
  not name a specimen, enter only beside an admitted paO2/paCO2 at the same
  stay and time; otherwise they are unknown and excluded. (4) eICU `Total CO2`
  is the chemistry panel and no longer feeds `tco2`. AUMC gases without a
  specimen record and SICdb rows without a laboratory type stay excluded
  (their items are not arterial by name). Each published hour carries
  `<concept>_specimen_status` with its weakest admitted evidence:
  `explicit_arterial`, `label_defined_arterial` or `co_measured_arterial`.
  Excluded rows (`excluded_venous`, `excluded_unknown`) are counted in the
  source audit, not published. Filtering runs before base-deficit negation
  and unit conversion.
- **[RR-04..RR-06]** eICU medication sources need route and positive dose
  evidence. Dose text is read with numeric unit codes (`81 3`), thousands
  separators (`1,000 mg`) and ranges (`2-4 mg`); a range from zero (`0-2 mg`)
  is unknown. AUMC IV drugs exclude `Niet iv`; HiRID IV drugs use an IV
  whitelist.
- **[RR-07]** MIIV eMAR doses duplicated in inputevents are removed only when
  admission, drug and positive mass dose match within 60 minutes and the
  input is explicitly IV; each input removes at most one eMAR dose.
- **[RR-08]** AUMC 8843 is a second arterial-line channel behind the main line
  for `map`/`map_invasive`; zeros no longer mask fallback channels.
- **[R6-F001, F020]** AUMC airway pressures convert mbar to cmH2O
  (×100/98.0665) before bounds, source priority and hourly aggregation; 6694
  is a measured PEEP fallback below the set source.
- **[R6-F003]** HiRID infusion rates find a strictly earlier predecessor for
  all same-time records, so row order no longer changes midazolam rates.
- **[R6-F017]** eICU `Mechanical Ventilator Mode` and `Ventilator Support Mode`
  feed `vent_mode`/`vent_breath_seq`; AC/CMV stay unknown.
- **[R6-F006]** eICU wide-table vitals pool raw values and apply bounds before
  aggregation, matching the ordinary path.
- **[R6-F008]** CVP 30–50 is kept with high/arterial-contamination/isolated-
  peak flags in all six databases; above 50 is excluded.
- **[R6-F018..F023, F035, F046, F047]** HiRID bedside FiO2 2010, eICU measured
  peak pressure and nursing invasive SBP/DBP, MIMIC-III nephrostomy volume
  41184 and SICdb enzymatic creatinine 368 are added. AUMC FiO2 source items
  are published as companion columns. Legacy GCS components expire 6 h after
  their own observation. HiRID insulin is described as total U per hour bin.
- **[R6-F040, F041]** An empty SICdb setting source no longer discards airway
  points. AKI display smoothing is sustained only by surviving measurements or
  active RRT, and empty creatinine/urine components no longer fail.
- **pandas 3:** eICU infusion durations no longer split at every row
  (boolean shift turned object-typed), and GCS components recorded at
  different time resolutions merge again.
- **Module export of companions:** when four or more concepts share a long
  table, the partial wide merge aligned only the value column and dropped
  `<concept>_specimen_status` (eICU, MIMIC-III, MIMIC-IV blood_gas) and other
  published companions. Such concepts now take the full merge.
- **Retrospective availability (all six databases):** values stay
  retrospective; real-time use must filter `available_at <= t`.

## 2026-11-04 — RU5 review, urine availability and assessment residues

- **[RU-01, RU-02, RU-09]** All six databases use the shared R1 collection
  kernel for descriptive urine output, KDIGO and SOFA-2/RRT urine assessment.
  Collection intervals are clipped to the window, and chart gaps over 12 hours
  break coverage. AUMC follows the same owner in per-concept, batch and native
  export paths. Window and duration assessments publish availability companions.
- **[RU-03, RU-08]** `urine24`, `urine24_est`, `uo_6h/12h/24h` and KDIGO urine
  stages remain **retrospective values; real-time use must filter
  `available_at <= t`**. Availability is the latest true availability of every
  contributing interval's records, including the record defining its left
  endpoint. A late entry can precede a later charted record in chart order; the
  entire window's maximum is retained. Entry delays do not change chart-gap
  segmentation or retrospective values. No real-time value column is added.
  eICU uses max of `intakeoutputoffset` and `intakeoutputentryoffset`;
  MIMIC-III/IV use chart/store time; HiRID uses `datetime`/`entertime`;
  AUMC uses `measuredat`/`registeredat`/`updatedat`. SICdb `data_float_h` has
  only `Offset`, so chart time is its available source clock. Missing per-row
  entry/store times fall back to chart time. Times are exported in ICU-relative
  hours through raw and unbinned, SQL, compatibility and native paths. Daily back-allocation
  may publish a value before its availability; its companion exposes that delay.
- **[RU-04]** The public legacy rate helper delegates to the shared kernel;
  obsolete scalar/daily/RRT helpers raise explicit deprecation errors.
- **[RU-05, RU-06]** eICU respiratory-care intervals require recognized invasive
  airway evidence and cover only hours with positive half-open overlap. HiRID
  NIV modes 12–14 keep their pressure control label but block invasive bridges.
- **[RU-07]** SOFA-1's description discloses its fallback to daily volume
  extrapolated from 18 to under 24 covered hours. Scores are unchanged.

## 2026-11-04 — extraction audit round 5 (eighth dictionary reseal)

The round-5 audit of cd2bae05d is fixed in eight lanes; the per-cluster
receipt is `outputs/extraction_audit_r5_receipt_20261104.md`. The loader and
HiRID lanes keep their detailed sections below. The other lanes:

- **Urine chain.** KDIGO collection intervals are clipped to the exact
  evaluation window. `uo_6h/12h/24h` divide by the hours actually covered
  and publish `*_covered_h` and `*_partial_window`. `urine24` is the full
  daily volume; an 18 to under 24 h estimate goes to the separate, flagged
  `urine24_est` (SOFA-1 keeps its fallback). `urine24_covered_h`,
  `urine24_observed_ml` and `*_available_at` report coverage and when the
  source records became available. HiRID urine follows its rate branch with
  gap segmentation. These companions are hidden from Planner menus.
- **eICU.** I/O rows are deduplicated before projection and hourly
  aggregation on every path. Systemic volumes are kept apart from
  cumulative and irrigation channels. Specimen, height, infection, device,
  route, concentration and NIV handling are corrected, and FiO2 channels
  stay independent.
- **AUMC.** Source timestamps are kept until half-open hourly expansion,
  and bag amounts are conserved. Actual-pH calcium is separate from
  `cai_ph74`. Tidal-volume units, ScvO2, FiO2, ECMO, PEEP and RRT evidence
  are corrected. Round-11 review maps blood D-dimer items 10393/12163 only
  when reported in mg/l, keeping native values unchanged. Its FEU/DDU basis
  remains unverified; AUMC and MIMIC D-dimer values are not harmonized.
- **SICdb.** Intervals, rates and diuretic amounts are aligned to
  right-open hours. Pressure units are corrected before bounds.
  `cvp_raw`/`cvp_qc` and `minute_vol_raw`/`minute_vol_qc` are published,
  and an opioid-only ventilation sensitivity column is kept apart. The
  legacy mass-rate and `sofa_score` APIs delegate to the current owners.
- **MIMIC dictionary.** ECMO requires positive evidence, and antibacterial
  exposure must be systemic. MIIV laboratory priority applies. CareVue
  vasoactive support is expanded and BiPAP/CPAP devices are added. SOFA-2
  `vaso_ind` takes `other_vaso` as boolean support, for MIMIC-III only.
  Fentanyl weight indexing fails closed. HbCO units are corrected.
  Round-11 review removes CareVue feeding items 45911/46082 from non-urine
  output in full and demo mappings; they are not reassigned to intake.
- **Bounds.** Nine concepts publish `*_raw`/`*_qc` siblings through the
  opt-in `bounds_audit` module. Public bounds, units, specimens and source
  priorities are unchanged.
- **Resolver.** A sub-concept cached on its parent's grid no longer answers
  a request on another grid. Before this fix the result depended on request
  order.

## 2026-11-04 — extraction audit round 5, MIMIC loader lane

Synthetic regressions only; real-data coverage and export deltas have not
been measured in this lane.

- **[R5C-IV-008, formerly R4-012]** MIIV labs with NULL `hadm_id` use
  `subject_id` and the bounded window from 24 h before ICU admission to 24 h
  after ICU discharge. Attribution considers all stays of the subject before
  selecting requested stays. Multiple candidate stays or admissions reject
  the row in round 5; ED1 above supersedes that rejection with rolling
  attribution. Requested-stay filters survive subject-level read pushdown.
  Admission-linked duplicate events take precedence; available event and
  specimen identifiers distinguish independent equal results, and text-only
  specimen labels remain distinct when they disagree. Missing optional units
  do not disable attribution. Raw and SQL
  loaders retain `lab_attribution_basis` and attribution counters in the
  `easyicu_miiv_lab_attribution` frame attribute and logs, including ambiguous,
  unmatched and deduplicated NULL-admission rows. Candidate counts are not
  estimates of net added stay coverage or of emergency-department activity.
- **[R5C-III-008, formerly R4-073]** MIMIC-III `chartevents` rejects
  `ERROR=1` before source projection or hourly aggregation; NULL and zero
  remain usable. Per-concept, bucketed and multi-concept loads share the rule,
  including lab/chart channel selection and CVP-derived values. Uppercase
  native headers work in preloaded and PyArrow dataset projections too.
- **[R5C-IV-010]** MetaVision medication rates use an explicit unit
  whitelist and mass/time/weight conversion before bounds. Conversion that
  changes the weight dimension requires a charted dosing weight of 20–300 kg.
  `mg/kg/min` is interpreted literally only with a valid dosing weight;
  unsupported labels and invalid required weights are rejected. Minimal
  projection retains source rate units and dosing weights.
- **[R5C-IV-011]** MetaVision rates occupy only hourly cells with positive
  overlap with `[start, end)`. An exact stop boundary contributes no rate to
  the next cell; multiple rate components within one hour retain the existing
  hourly rate aggregation.
- **[R5C-IV-012]** MIIV and MIMIC-III MetaVision insulin amounts represent total administered units.
  Infusion components are apportioned by actual overlap with ICU-hour cells;
  zero-duration pushes and subcutaneous doses contribute once to their start
  cell. Rewritten orders are excluded before summing the dose components
  inside the amount callback and before the generic numeric aggregation.
  Other databases' rate-valued insulin sources retain
  their aggregation semantics.
- **[R5C-IV-014]** The interval allocator tests duration against an absolute
  tolerance of `1e-12` hours with `rtol=0`. A long-stay clock therefore cannot
  turn a short positive infusion into a bolus. The conserved hourly volumes
  feed fluid balance and cumulative balance.

## 2026-11-04 — extraction audit round 5, HiRID

These changes are checked with synthetic sources. The audit's real-data
aggregate counts have not been rerun in this lane.

- **[R5C-HIRID-002/019]** Non-weight drug rates split each converted dose
  over the preceding valid same-infusion interval, conserving amount. ADH
  divides the hourly amount by 60; fentanyl and midazolam use their declared
  mass/time units. Intervals over six hours are not spread across the gap.
  Pharma `recordstatus` bits 2 and 32 exclude invalidated and not-administered
  rows before aggregation, predecessor selection and episode boundaries.
- **[R5C-HIRID-003]** Infusion duration and start retain the exact clock.
  `*60` qualifies episodes by at least 60 actual minutes; occupancy covers
  the original half-open interval, including short cross-hour infusions.
  The public loader also preserves the start, so its hourly alignment cannot
  shift the episode end before the indicator and qualification callbacks run.
- **[R5C-HIRID-004/005]** Invasive airway windows expand in full. NIV,
  no-airway and standby intervals stop inferred ventilation bridges.
  Mixed invasive/NIV evidence within an hour retains its actual invasive
  overlap. Advanced support includes reliable airway intervals, alongside
  existing mode sources, and unknown support remains missing.
  A transition from unknown to supported within one hour retains the support.
  Interval boundaries are scanned in bulk instead of repeatedly searching
  the full evidence table for each stay.
- **[R5C-HIRID-006]** An isolated counter up-spike that immediately returns
  to the preceding trajectory is excluded from both the running peak and
  the allocation clock. Sustained increases and genuine restarts remain.
- **[R5C-HIRID-007–012]** Add the audited antibiotic, LMWH, insulin,
  magnesium, milrinone/terlipressin and diltiazem preparations. Positive-dose
  thresholds remain; antibiotic/magnesium IV and LMWH SC eligibility are
  explicit. The added insulin preparations affect the administration flag
  only. HiRID diltiazem is enteral under the existing any-route definition,
  and must not be interpreted as IV exposure.
- **[R5C-HIRID-013]** HiRID CK-MB is available separately as
  `ckmb_hirid_raw` in its native U/L. It no longer enters the `ckmb`
  ng/mL pool. Assay identity remains unresolved; no fixed conversion is used.
- **[R5C-HIRID-023]** The dex IV route filter now runs once before the
  grouped callback. Both `route` and `infusionid` are loaded, and the complete
  public extraction path is tested with concurrent IV and enteral groups.

Round-5 secondary registrations do not establish clinical validity:

- **[016/017]** Lymphocyte G/L identity and zero coding, and drainage volume
  time semantics/overlap with the total-output counter, remain unresolved.
  Neither candidate is mapped without that evidence.
- **[025]** `plateau_pres_raw` remains the charted measurement. The PIP/PEEP
  consistency screen in `plateau_pres` is a plausibility check, not a
  mode-independent validation of plateau pressure. Assisted breaths or
  active effort require special caution; controlled passive measurements
  still need an adequate inspiratory hold. No clinical revalidation was run.
- **[026]** Descriptive urine half-window coverage is distinct from a fully
  assessable SOFA-2 evaluation window. Coverage columns and the HiRID urine
  strategy are described under the urine chain above.
- **[029]** The legacy `easyicu.callbacks.sofa_score` now delegates to the
  current SOFA owner (SICdb lane, above).
- **[022/024/027/028/030/031/032]** Reference-only zero-event IDs, P/F
  metadata, platelet/ICP/respiratory/neutrophil candidates and insulin timing
  notes are registered without remapping or a scientific-status change.
  HiRID `ins` is an hourly amount sum; it is not proof of continuous IV
  insulin delivery.

## 2026-11-04 — extraction audit round 4

Numbers are 3,000-stay native-v2 samples, before (`5d0ed1256`) → after.
Commits: `ade4802d7` (fixes), `6d3d52e92` (seventh reseal), `b21030f26`
(follow-ups from the server regression). Every item with its status is in
`outputs/extraction_audit_r4_receipt_20261104.md`.

### Trunk fixes (R4)

- **[R4-001/R4-010]** Uncharted urine hours are missing, not zero. A record
  is spread back to the previous one over at most 12 h; `urine24` needs 18
  covered hours of the last 24 cells and is scaled to 24 h. urine24 rows MIIV
  197,138 → 158,643 (p05 49.9 → 330 mL), eICU 113,129 → 72,644 (p05 0 → 286).
- **[R4-002]** MIMIC-III CareVue phenylephrine 30127/30128 is `other_vaso`
  (TRUE rows 7,691 → 15,246; stays 355 → 604).
- **[R4-003]** eICU `other_output_ml` never selects a urine row.
- **[R4-004]** A failed native-v2 publication raises
  `NativeExportPublicationError` (`native_export_v2_publication_failed`) and
  moves the unpublished files to `.easyicu-unpublished/`.
- **[R4-006]** MIMIC-III prescription points stay in `delirium_tx_proxy`
  (rows 238 → 1,042).
- **[R4-007]** AUMC insulin has a rate in every hour of [start, stop) (rows
  31,953 → 185,789; median 2.7 → 2.0 U/h).
- **[R4-008]** Pooled multi-source concepts take one median over all raw rows.

### Trunk M items

- **[R4-009]** SOFA-1/SOFA-2 worst-value windows are 24 hourly cells.
- **[R4-013]** CVP above 50 mmHg is dropped; 30–50 is dropped only within
  10 mmHg of the same-hour arterial diastolic or as an isolated spike (kept
  (30,50] rows: MIIV 108, MIMIC-III 224, eICU 258, AUMC 90). SICdb and HiRID
  keep their 30 mmHg caps; `cvp_raw` keeps the charted values.
- **[R4-014/R4-024]** `vent_ind` takes invasive windows only; NIV stays a
  `mech_vent` level. eICU vent_ind rows 48,502 → 44,600.
- **[R4-016]** Lab concepts prefer the lab channel: a chart transcription
  within 6 h of a lab result is dropped (MIMIC-III crea 26,236 → 21,042 rows).
- **[R4-018]** MIMIC-III MetaVision `mort_365d` is restored (non-null 1,814 →
  2,997), reversing R3-08.
- **[R4-020/R4-021]** MIMIC gases without a specimen record are not arterial
  (ph 26,796 → 20,529; sao2 8,043 → 5,807).
- **[R4-034]** MIMIC-III `adm` is the service active at ICU intime (stays
  2,530 → 2,999).
- **[R4-035]** OR/PACU intake is not an ICU transfusion, in MIMIC-III and MIIV
  (MIIV ffp 603 → 517).
- **[R4-041/R4-043]** eICU 'Delirium Scale/Score' Yes/No is CAM-ICU
  (delirium TRUE 95 → 425); coded GCS responses ('4-->(E4) …') are parsed (egcs
  stays 2,003 → 2,116).
- **[R4-046]** eICU per-kg infusion rates are multiplied by weight (fentanyl
  rate median 50 → 73 µg/h).
- **Route filters.** IV-defined drugs drop enteral orders in MIMIC-III, AUMC
  and HiRID (HiRID levetiracetam 1,571 → 332 rows; AUMC pantoprazole 6,942 →
  5,687).

### Documented, not changed (round 4)

- **MIIV ED labs** with `hadm_id` NULL (R4-012) and **chartevents ERROR rows**
  (R4-073) were deferred in round 4; the round-5 loader fixes are recorded above.
- **HiRID dex route** (R4-164): the route filter broke the grouped callback
  and was reverted.

## 2026-11-04 — extraction audit round 3

Numbers are 3,000-stay samples, before (`014aac14f`) → after. Commits:
`edb7da2aa` (R3-01), `5a5eac84b` (R3-02..R3-20, F items, first M items),
`3a296c1f9`, `8c8605337` and `4b2664127` (full-dictionary M items).

### Trunk fixes (R3)

- **[R3-01]** The id-table DuckDB path converts every row for a
  two-argument `convert_unit`: 226531 'Admission Weight (lbs.)' is now kg.
  MIIV weight p95 149.6 → 121.0 kg; MIMIC-III MetaVision 131.9 → 120.2.
- **[R3-02]** HiRID and AUMC vasopressor durations start on the hour as the
  rates do, so `*60` finds its rates. HiRID norepi60 rows 15,873 → 16,644,
  AUMC 78,265 → 79,977.
- **[R3-03]** eICU cancelled orders are dropped: `drugordercancelled` holds
  'Yes'/'No', which now normalise to the declared `true`.
- **[R3-04]** eICU noradrenaline order spans end at unit discharge. norepi_dur
  p95 117.7 → 109.8 h; norepi60 rows 3,889 → 4,113.
- **[R3-05]** eICU ventilation windows come from charted ventilator settings
  and care intervals. vent_ind hours 3,754 → 50,061 (stays 824 → 985); PEEP > 0
  hours without vent_ind 17,796 → 535.
- **[R3-06/R3-07]** `adv_resp` is limited to ventilation windows (eICU hours
  outside vent_ind 19,279 → 171) and HiRID 'Stand By' is not ventilation
  (11,667 → 7,187).
- **[R3-08]** MIMIC-III MetaVision `mort_365d` is NA: follow-up ends at 90 days
  (non-null 8,131 → 0).
- **[R3-09]** `urine24` windows end at the last hour, not at a duration used
  as an end. MIIV urine24 stays 1,655 → 1,718; AUMC 1,751 → 1,806.
- **[R3-10..R3-12]** AUMC 8903 arterial SO2 feeds `sao2` (rows 74,123 →
  76,478); `minute_vol` excludes spontaneous 12578 (zero rows 268 → 195);
  `ett_gcs` reads the tube and tracheostomy intervals (0 → 2,207 windows).
- **[R3-13/R3-15]** AUMC death after discharge and SICdb deaths without an
  offset are timed at discharge (SICdb death TRUE 2,095 → 2,154 in the full
  database).
- **[R3-14]** AUMC vasopressor bounds allow high doses (norepi > 3: 0 → 19
  rows).
- **[R3-16]** eICU and SICdb `sbp`/`dbp` carry source channels.
- **[R3-17]** Driving pressure matches PEEP within ±1 h. MIIV rows 9,129 →
  9,034; eICU 5,291 → 5,044.
- **[R3-18]** `plateau_pres` drops plateaus above PIP + 1 or below PEEP
  (HiRID 30,517 → 28,176 rows); `plateau_pres_raw` keeps the charted values.
- **[R3-19]** A HiRID zero dose is a pump marker, not a rate: norepi_rate rows
  17,352 → 16,799, vaso_ind 29,552 → 29,477; 37 boolean HiRID drug sources need
  a positive dose.
- **[R3-20]** MIMIC-III CareVue mcg/min vasopressor rows without a weight are
  NA, not mcg/kg/min.

### F items

F-01..F-31 are listed with their numbers in
`outputs/extraction_audit_r3_receipt_20261104.md`. Examples: AUMC antibiotics
rows 27,568 → 30,200; eICU ammonia no longer divided (median 22.6 → 37.0
µmol/L); MIIV D-dimer FEU not halved (median 1,618 → 2,576); MIMIC-III CareVue
vancomycin < 10 → 1,708 rows; SICdb albumin 171 added; the merged export
expands windows that carry an hours duration (SICdb mech_vent 338 → 6,459).

### Full-dictionary M items

- **Fluid balance.** `fluid_balance` is input minus urine minus non-urine
  output (`other_output_ml`: drains, gastric, stool, blood loss; RRT removal and
  urine excluded) for MIIV, MIMIC-III, eICU, AUMC and SICdb. This replaces the
  round-2 [F1] note. Median hourly balance MIIV 36.0 → 30.1 mL, AUMC 11 → 6,
  SICdb 30 → 25.
- **Rates.** HiRID mass rates sum concurrent syringes per hour; zero doses are
  dropped (fentanyl median 50.6 → 51.8 µg/h); diuretic zero amounts are not
  doses; AUMC continuous rates read running infusions only (flushes and single
  doses out).
- **Ventilation.** `vaso_ind` loads on a minute grid and covers the hour an
  episode ends (SICdb 55,580 → 56,953 hours). A HiRID mech_vent window without
  a level ('-') is not ventilation. MIMIC-III 640 'extubated' rows end
  ventilation; MIMIC/MIIV 'Standby' is not a mode.
- **Mappings.** eICU GCS labels (gcs stays 2,123 → 2,248), base deficit (be
  stays 970 → 1,116), differential percentages only, oxygen flow (rows 9,683 →
  23,379), charted NIBP, HICL dedupe, IV-only route filters for IV-defined
  drugs; MIMIC-III blood-gas items, sao2 without pulse-oximetry 834;
  MIIV 'High flow neb' out of adv_resp, bicarbonate 50803, delirium from the
  overall CAM-ICU result (stays 1,057 → 698); AUMC IABP intervals (rows 2,118
  → 2,886), ASB pressure support (stays 1,899 → 2,020), measured PEEP 8879,
  cardiac output 20955; SICdb inspiratory tidal volume for 2013, manual
  differentials (stays 214 → 370), heart-rate channels; eICU SOFA-2 motor
  response labels (stays 1,816 → 2,030).
- **SICdb perfusor furosemide.** A continuous row's syringe amount is shared
  over its running hours (furosemide-equivalent rows 3,927 → 8,709; hours above
  40 mg 205 → 27).
- **eICU monitor etCO2.** `etco2` reads `vitalperiodic.etco2`; the per-table
  offset merge no longer loses a source beside an empty one (rows 149 → 3,878,
  stays < 10 → 116).
- **MIIV admission type.** `adm` takes the hospital service active at ICU
  intime (the first service when the stay starts earlier); a service chosen more
  than a day before the stay or a second ICU stay no longer loses its type
  (stays 2,440 → 2,999 of 3,000).
- **Bounds.** Arterial pressures and plateau below 1 mmHg are not measurements.

### Documented, not changed (round 3)

- **MIIV ED labs.** Subject/time attribution of NULL-`hadm_id` labs was deferred
  in round 3. Round 5 added bounded single-candidate recovery across loaders;
  ED1 above completes overlapping-window attribution with the rolling rule.
- **MIMIC-III prescriptions** are day-resolution (noon); eMAR does not exist.
- **Lab specimens.** `lact`, `be`, `tco2`, `hbco`, `methb`, `cai` and `o2sat`
  are not arterial-only concepts; arterial-only variables carry
  `arterial_specimen_only`.
- **Charted lab transcriptions** (MIMIC-III crea/hgb/plt/wbc chartevents) stay
  as a second source; hourly aggregation merges same-hour copies.
- **SICdb boolean drug infusions** mark the start hour, as boolean drug
  concepts do in every database.
- **SICdb pulmonary-artery and temperature artefacts** cannot be separated from
  physiology by bounds.
- **eICU dopamine/dobutamine '()' rows** carry no unit and stay unconverted;
  'mg/kg/min' rows outside bounds are dropped, not relabelled.

## 2026-11-03 — extraction audit round 2

### Deliberate changes since v6 (`97f508c6`), confirmed

- **[A1] AUMC P/F and S/F.** `57ed20701` (2026-09-27) stopped resampling
  FiO₂ and PaO₂ a second time on the absolute clock. For admissions that do not
  start on the hour, that second pass had moved observations into the preceding
  patient hour. P/F and S/F therefore change in about half of AUMC hours (median
  about −3), and SOFA-1 respiration is one point higher in about 12% of hours.
  See `docs/qa/aumc_po2_source_repair.md`.
- **[A2] SOFA-2 level, all databases.** `aa5e4c697` (2026-10-22) loads the
  SOFA-2 total and its rolling/last-observation organ states from one aggregate
  callback, as SOFA-1 already did. Native publication no longer recomputes the
  total from point components, which had replaced correct stay maxima. The
  median SOFA-2 rises from about 1 to 5.
- **[A3] MIMIC-IV age.** `f1bb708b2` (2026-10-26) computes age at ICU admission
  (`anchor_age + year(intime) − anchor_year`) instead of the anchor-year age.
  1,032 stays become older (maximum 91 → 96), and eGFR follows.
- **[A4] Other deliberate changes.**
  - SICdb Charlson/Elixhauser are not computed (`d36f0e322`): SICdb records one
    principal diagnosis per case, which describes the reason for admission, not a
    comorbidity burden.
  - The RRT evidence status of an unresolved stay is `indeterminate`, not
    `negative` (`b81c04ad6`).
  - New columns: `icu_death`, `fio2_chart` and `fio2_lab`.
  - NEWS moves with its vital-sign inputs (channel-ranked MAP, the eICU and SIC
    vital-sign repairs).

### Fixed in this round

- **[R2-01]** MIMIC-III: the id-table DuckDB fast path aliases the id column.
  Upper-case parquet returned `ICUSTAY_ID`, so `weight` and then demographics,
  renal and SOFA-2 failed, and publication was refused.
- **[R2-02]** `vaso_ind` covers `seq(start, start + duration)`. The ricu form
  read the duration as an end time and dropped every course that started after
  its own length.
- **[R2-03]** Death and follow-up:
  - `death` is False for survivors on HiRID (at the last observation) and SICdb
    (at ICU discharge), not missing.
  - On AUMC, a discharge to "Overleden" is an ICU death timed at `dischargedat`,
    also when it has no `dateofdeath`.
  - The ICU-episode clip keeps post-discharge rows of `death`.
  - eICU `death` sits at `hospitaldischargeoffset`, and the export issues its
    `death_time`.
  - `mort_28d/90d/365d` and `followup_days_*` are restored for MIMIC-III and
    AUMC, with documented follow-up bounds (see `scores/outcomes.py`).
  - `mort_*` counts all-cause deaths, including after discharge. A stay can
    therefore have `mort_28d` without the in-hospital `death`.
- **[R2-04]** Window concepts without durations (eICU `mech_vent` point
  evidence) are floored to the hour. They no longer sit at minute offsets that
  meet no hourly row.
- **[R2-05]** Batched MIMIC lab and chart loads keep the [−24 h, LOS + 24 h]
  window before derivation. They no longer rely on publication to trim them.
- **[V4]** HiRID: the noradrenaline bolus 1000658 is removed from the infusion
  rate and duration.
- **[V8, F2]** MIMIC-III MetaVision rates and `total_input_ml` drop `Rewritten`
  orders.
- **[V9]** MIMIC-III CareVue 30047/30044 (mcg/min) rates are divided by weight.
- **[G3]** AUMC: 8903 blood-gas SO₂ is removed from `spo2` and `o2sat`.
- **[G4]** AUMC: PaCO₂ 21213 is converted from kPa.
- **[G9]** SICdb: 667 is removed from `bicarb`.
- **[P5b]** AUMC: set tidal volume 8851 is limited to 100–1,500 mL.
- **[D2]** eICU HICL-only orders are read: thiazides, torsemide and
  acetazolamide.
- **[K1]** MIMIC-III: CareVue 582 (HD/CVVHD/CAVH start) is RRT evidence, in
  `concept-dict` and in the `sofa2-dict` overlay that the loader prefers
  (`03cd2c0c6`).
- **[P3]** Driving pressure drops a plateau pressure above the nearest PIP + 1
  (`3bdfcd4a3`).
- **[W1, W2]** Weight reads the further charted channels: MIMIC-III 763,
  580/581, 224639, 226531 (lbs) and `patientweight`; MIIV 226531/224639; eICU
  intake/output body weight after the admission weight. One channel is chosen
  per stay (`608cf549d`, `afd679a73`).
- **[P1]** SICdb: in ICU units 2-4, a set-rate or PressureOverPeep hour under
  propofol is invasive ventilation without an airway interval; the airway-only
  rule stays the strict sensitivity rule (`e47722909`).
- **[V6, V7]** HiRID SOFA cardiovascular scores the dose rate per kg per minute
  (`551aaaab2`). SOFA takes the hour's highest dose and lowest MAP, while the
  exported `*60` dose columns are hourly medians, so a residual 1.4% of AUMC
  hours with a 24 h maximum exported dose of 0.07-0.098 still score 4.
- **[D5]** MIMIC-III prescriptions without a time are placed at noon of their
  start date, as `mimic_abx_presc` already was (`76fb2792e`). They remain
  day-resolution exposures: not for hourly landmarks, and a multi-day order
  marks its first day only.
- **[V3, D3]** eICU: `other_vaso` reads HICL 2839/2087; every eICU
  `medication` source drops `drugordercancelled` orders (`30643d33e`).
- **[V2]** eICU noradrenaline rows in ml/hr without a concentration are not
  converted with a constant (`624e28ba7`).
- **[V1]** eICU stays with a noradrenaline order and no infusion charting get
  `norepi_dur`/`vaso_ind` exposure from the order span. The span is the order
  validity, not pump time, and gives no dose (`fa80e951e`).
- **[D1, D3]** MIMIC-IV loop-diuretic doses add administered eMAR doses not
  already in inputevents (±60 min, same drug); MIMIC-IV diuretic exposures read
  administered eMAR rows instead of prescriptions (`2acfa3c97`).
- **[P4, P9, G6, G7, G8, H5, H6, H7]** (`afd679a73`):
  - Set and total PEEP for AUMC, HiRID, eICU and SICdb.
  - MIMIC-III TCPCV is pressure-controlled ventilation.
  - `o2_flow` for MIIV, MIMIC-III and eICU, and `supp_o2` depends on it.
  - The native-v2 export publishes `o2_flow` in the respiratory module. Its
    `supp_o2` check now accepts flow > 0 L/min as support; before, it dropped
    flow-only claims (eICU demo: 6,651 TRUE hours, 874 rows).
  - `safi` is SpO₂/FiO₂; MIIV/MIMIC-III `tco2` keeps arterial samples.
  - eICU vital signs fill hours without a monitor value from nurse charting.
  - MIIV `sbp`/`dbp` add the arterial line 225309/225310.
  - MIMIC-III CareVue IABP 225 is mechanical circulatory support.
  - MAP−CVP has no derived concept; an analysis computing it should use
    `map_invasive`, not the channel-ranked `map`.
- **[V5]** HiRID infusion rates add concurrent syringes in the hour, and
  syringes less than 5 minutes apart are one episode (`2441cf22b`).
- **[P6]** SICdb tidal volume below 100 mL is missing (`04efe52ae`).
- **[P5]** AUMC `tidal_vol` is expiratory (12277/12360 out). Inspiratory
  minute volume (AUMC 12276, MIMIC-III 6934) is out of `minute_vol`;
  spontaneous minute volume stays pooled (`9b01621e0`).
- **[S8]** HiRID (`86bfe7228`):
  - `adm` comes from the APACHE II/IV group (HiRID-ICU-Benchmark Table 7;
    II 107-114 and IV 199-205 surgical).
  - `neut` reads 24000556.
  - Terlipressin 1000793 is `other_vaso`. Diltiazem stays unmapped for HiRID:
    the 2026-07-17 concept is intravenous only, and HiRID 1000331/1001071 are oral.
  - `driving_pres_controlled` aggregates have one entry per component.
  - `epi_rate`/`epi_dur` keep the perfusor items 1000649/1000650.
  - `ett_gcs` is described as an invasive airway.
- **[S1]** SICdb `adv_resp` reuses the invasive-ventilation windows of
  `mech_vent`, so SOFA-2 respiration can score 3-4 (`63affaa70`). SICdb SOFA-1 is
  a five-organ sum (no GCS); SICdb SOFA-2 has no total, by design.
- **[S6]** eICU orders without a drug name match by HICL for 32 drug concepts.
  The HICL codes are those whose named rows match the concept's name pattern in
  at least 95% of rows; albumin also matches brand names (`63ee291db`).
- **[G5]** P/F pairs the hour's lowest PaO₂ with its median FiO₂. HiRID P/F
  median rises from 184 to 221 mmHg; other databases change by less than 3%
  (`474f34ddb`).
- **[G2]** AUMC 2003-2008 blood gases read the specimen site from list item
  7432; PAP is a mixed-venous sample (`65518683f`).
- **[H4]** `sbp`/`dbp` prefer the invasive channel per hour; HiRID cuff
  600/620 fill hours without an arterial line (`741345867`).
- **[F1]** HiRID fluid counters restart on a drop below half of the day's
  running maximum, and a net balance hour needs both counters on that 12:00 day
  (`d26e0f67c`).

### Documented, not changed

- **[F1]** `fluid_balance` differs by database: HiRID is input minus total
  output, every other database input minus urine; AUMC input is intravenous
  only. Compare balances across databases only stratified.
- **[F3]** MIMIC-IV 3.1 has inputevents for 42% of 2020-2022 stays (other years
  95-96%). `total_input_ml` cannot describe that period.
- **[K3]** eICU hourly urine is the volume charted in that hour. Intermittent
  records are not spread backwards (the urine assessment does not assume a
  distribution).
- **[S2]** SOFA-2 components carry forward within the stay without a 24 h cap,
  then take the 24 h worst value. A 24 h-capped recomputation is 2-3% lower.
- **[S3]** SICdb `epi_rate` keeps continuous infusions only (264 of 913 cases
  with adrenaline); single pushes are not rates.
- **[S4]** HiRID and SICdb `mech_vent` are sparse state records; use
  `vent_ind` for ventilation exposure.
- **[S5]** SICdb CVP coverage falls from 64% to 36% across admission years.
  Stratify CVP analyses by year.
