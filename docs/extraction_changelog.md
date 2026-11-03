# Extraction changelog

Changes that move extracted values between versions, for anyone comparing an
export with an earlier one. Each entry names its commit; the audit item it
answers is in brackets.

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
