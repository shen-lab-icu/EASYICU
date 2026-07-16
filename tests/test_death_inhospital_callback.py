"""Regression tests for the AUMC / SICdb in-hospital `death` callbacks.

Both source callbacks were producing near-annual mortality instead of the
declared in-hospital mortality (concept-dict `death.description = "in hospital
mortality"`):

- AUMC: the ricu-style 72h-of-discharge window was a silent no-op because the
  loader converts admission times to minutes while the threshold was hard-coded
  in milliseconds, so every patient with any (up to 14.5-yr-later) registry
  `dateofdeath` was flagged (~33% vs the true ~10%).
- SICdb: death was flagged whenever `OffsetOfDeath` was non-null, capturing
  registry deaths up to ~1yr of follow-up (~18.6% == mort_365d) rather than the
  ~7.8% in-hospital rate.

The fix reads the source's own discharge disposition (`destination == 'Overleden'`
for AUMC; `HospitalDischargeType == 2028` for SICdb), which is unit-independent.
"""

import types

import pandas as pd

from easyicu.concept.callback_apply import _apply_callback


def _src(callback, index_var=None, value_var=None):
    return types.SimpleNamespace(
        callback=callback, index_var=index_var, value_var=value_var,
        sub_var=None, unit_var=None, table=None, ids=None,
    )


def test_aumc_death_matches_ricu_72h_window():
    # ricu aumc_death: in-hospital = died within 72h of ICU discharge
    # (is_true(dateofdeath - dischargedat < hours(72))). Times in MINUTES. Keep mortality
    # < 25% so the anti-unit-bug guard does not trip.
    #   id 1: died 100 min before ICU discharge (in ICU)      -> death
    #   id 2: died 1 day (1440 min) after ICU discharge        -> death (< 72h)
    #   id 3: died 200 days after ICU discharge                -> NOT death (long-term)
    #   id 4: survivor (no dateofdeath)                        -> NOT death
    #   id 5-20: survivors
    n = 20
    frame = pd.DataFrame({
        "admissionid": list(range(1, n + 1)),
        # value_var dischargedat renamed to concept_name by the loader (minutes)
        "death": [1000] * 4 + [1000] * 16,
        "dateofdeath": [900, 1000 + 1440, 1000 + 200 * 24 * 60, None] + [None] * 16,
        "destination": ["Overleden", "15", "16", "Home"] + ["Home"] * 16,
    })
    out = _apply_callback(frame, _src("aumc_death", "dateofdeath", "dischargedat"), "death")
    died = out["death"] == True  # noqa: E712 (object dtype True/None)
    assert int(died.sum()) == 2, "in-ICU death (id 1) + within-72h post-discharge (id 2)"
    assert pd.isna(out.loc[out["admissionid"] == 3, "death"].iloc[0])  # long-term excluded


def test_aumc_death_guard_falls_back_on_unit_mismatch():
    # Simulate the historical ms-vs-minute bug: every registry death lands just above
    # discharge, so a unit-broken window would flag all of them (>25% mortality). The guard
    # must degrade to the unit-independent ICU-death disposition (destination=='Overleden').
    n = 10
    frame = pd.DataFrame({
        "admissionid": list(range(1, n + 1)),
        "death": [1] * n,          # dischargedat
        "dateofdeath": [2] * n,    # diff = 1 < any threshold -> would flag all 10
        "destination": ["Overleden", "Overleden"] + ["Home"] * 8,
    })
    out = _apply_callback(frame, _src("aumc_death", "dateofdeath", "dischargedat"), "death")
    died = out["death"] == True  # noqa: E712
    assert int(died.sum()) == 2, "guard degrades to destination=='Overleden' only"


def test_sic_death_uses_hospital_discharge_type_not_offset():
    # HospitalDischargeType: 2028=Deceased, 2026=Survived.
    # Row 4 has an OffsetOfDeath (long-term death) but was NOT deceased in hospital.
    frame = pd.DataFrame({
        "CaseID": [1, 2, 3, 4, 5],
        # OffsetOfDeath (val_var==index_var) renamed to concept_name by loader (seconds)
        "death": [3600, 7200, None, 3_700_000, None],
        "HospitalDischargeType": [2028, 2028, 2026, 2026, None],
    })
    out = _apply_callback(frame, _src("sic_death", "OffsetOfDeath", "OffsetOfDeath"), "death")
    died = out["death"] == True  # noqa: E712
    assert int(died.sum()) == 2, "only HospitalDischargeType==2028 rows are in-hospital deaths"
    # row 4: has OffsetOfDeath but discharged alive -> not an in-hospital death
    assert pd.isna(out.loc[out["CaseID"] == 4, "death"].iloc[0])
    # in-hospital deaths carry a charttime (OffsetOfDeath hours); survivors do not
    assert out.loc[out["CaseID"] == 1, "charttime"].iloc[0] == 1.0
    assert pd.isna(out.loc[out["CaseID"] == 4, "charttime"].iloc[0])
