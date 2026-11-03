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
