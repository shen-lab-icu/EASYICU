"""A window concept without durations is put on the hourly grid.

Audit round 2 (R2-04): eICU ``mech_vent`` is point evidence with no auditable
stop.  Its rows skipped window expansion and ``change_interval`` and stayed at
minute offsets, so ``vent_ind`` never shared an hour with ventilator settings
and SOFA respiration came out as 0.5-steps.  Synthetic frames only.
"""

from __future__ import annotations

import pandas as pd

from easyicu.concept import _floor_point_windows_to_interval


def test_points_move_to_their_own_hour_and_are_not_extended():
    frame = pd.DataFrame({
        "patientunitstayid": [1, 1, 1, 2],
        "charttime": [3.25, 3.75, 4.0, -0.5],
        "mech_vent": ["invasive", "invasive", "invasive", "noninvasive"],
    })

    out = _floor_point_windows_to_interval(frame, "charttime", pd.Timedelta(hours=1))

    assert out.to_dict("list") == {
        "patientunitstayid": [1, 1, 2],
        "charttime": [3.0, 4.0, -1.0],
        "mech_vent": ["invasive", "invasive", "noninvasive"],
    }


def test_a_non_numeric_index_is_left_alone():
    frame = pd.DataFrame({"id": [1], "charttime": [pd.Timestamp("2020-01-01 03:15")]})

    out = _floor_point_windows_to_interval(frame, "charttime", pd.Timedelta(hours=1))

    assert out.equals(frame)
