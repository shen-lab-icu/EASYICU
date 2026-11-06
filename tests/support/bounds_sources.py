"""Synthetic SICdb and eICU sources for the bounds-drop tests (one all-out-of-bounds stay).

Shared by tests/core/test_bounds_drops_batch_invariant.py and
tests/core/test_source_bounds_audit_failures.py. Synthetic rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.support.urine_raw_source import _source

_STAYS = 6
_SUSPECT = 1  # every value out of bounds, 120 of them
_SUSPECT_VALUES = 120
_ORDINARY_VALUES = 5  # plus one out-of-bounds value per ordinary stay


def _hours(n):
    return np.arange(n, dtype=float) + 1.25


def _sic(root):
    ids = np.arange(1, _STAYS + 1)
    cases = pd.DataFrame({
        "CaseID": ids, "PatientID": ids, "ICUOffset": 0.0,
        "TimeOfStay": 200.0 * 3600, "WeightOnAdmission": 70000.0,
    })
    rows = []
    for stay in ids:
        if stay == _SUSPECT:
            hours, temp = _hours(_SUSPECT_VALUES), np.full(_SUSPECT_VALUES, 98.6)
        else:
            hours = _hours(_ORDINARY_VALUES + 1)
            temp = np.r_[np.full(_ORDINARY_VALUES, 36.5 + stay / 10), 45.0]
        rows.append(pd.DataFrame({"CaseID": stay, "DataID": 709, "Offset": hours * 3600, "Val": temp}))
        rows.append(pd.DataFrame({"CaseID": stay, "DataID": 710, "Offset": hours * 3600, "Val": 96.0}))
    return _source(root, "sic", cases=cases, data_float_h=pd.concat(rows, ignore_index=True))


def _eicu(root):
    ids = np.arange(1, _STAYS + 1)
    rows = []
    for stay in ids:
        if stay == _SUSPECT:
            minutes, pas = _hours(_SUSPECT_VALUES) * 60, np.full(_SUSPECT_VALUES, 500.0)
        else:
            minutes = _hours(_ORDINARY_VALUES + 1) * 60
            pas = np.r_[np.full(_ORDINARY_VALUES, 20.0 + stay), 300.0]
        rows.append(pd.DataFrame({
            "patientunitstayid": stay, "observationoffset": minutes,
            "pasystolic": pas, "padiastolic": 15.0, "pamean": 30.0,
        }))
    vitals = pd.concat(rows, ignore_index=True)
    vitals.insert(0, "vitalperiodicid", np.arange(1, len(vitals) + 1))
    return _source(
        root, "eicu",
        patient=pd.DataFrame({
            "patientunitstayid": ids, "patienthealthsystemstayid": ids,
            "unitdischargeoffset": 12000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
        }),
        vitalperiodic=vitals,
    )
