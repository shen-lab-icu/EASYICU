# Extraction changelog

Changes that move extracted values between versions, for anyone comparing an
export with an earlier one. Each entry names its commit; the audit item it
answers is in brackets.

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
  (R4-073) need loader changes and are deferred.
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

- **MIIV ED labs.** Labs with `hadm_id` NULL drawn in the ED before admission
  are not attributed to the ICU stay (ricu convention; 10-30% of stays for ~30
  lab concepts per the audit). Attribution by subject and time needs a loader
  change on three query paths and is tracked separately.
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
