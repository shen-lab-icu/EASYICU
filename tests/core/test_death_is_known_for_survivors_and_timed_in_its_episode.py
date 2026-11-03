"""In-hospital death is True or False for every stay with a recorded status.

Audit round 2 (R2-03) found the ``death`` status null for every survivor on
HiRID and SICdb (a crude mortality of 100%), AmsterdamUMCdb ICU deaths without
a ``dateofdeath`` read as survivors and timed before ICU admission, and the
ICU-episode clip dropping deaths recorded after ICU discharge (SICdb, AUMC,
and every eICU death, which sits at the hospital discharge offset).
Synthetic frames only.
"""

from __future__ import annotations

import types

import pandas as pd

from easyicu.concept import POST_DISCHARGE_OUTCOME_CONCEPTS, _quarantine_rows_outside_icu_episode
from easyicu.concept.callback_apply import _apply_callback


def _src(callback, index_var=None, value_var=None):
    return types.SimpleNamespace(
        callback=callback, index_var=index_var, value_var=value_var,
        sub_var=None, unit_var=None, table=None, ids=None,
    )


def test_an_aumc_icu_death_is_a_death_at_icu_discharge():
    minute = 60_000
    frame = pd.DataFrame({
        "admissionid": [1, 2, 3],
        # The loader renames the value column (dischargedat) to the concept.
        "death": [160 * minute, 300 * minute, 300 * minute],
        "dateofdeath": [100 * minute, None, None],
        "destination": ["Overleden", "Overleden", "Huis"],
    })

    out = _apply_callback(frame, _src("aumc_death", "dateofdeath", "dischargedat"), "death")
    out = out.set_index("admissionid")

    assert out.loc[[1, 2], "death"].tolist() == [True, True]
    assert bool(out.loc[3, "death"]) is False
    # Each is timed at its ICU discharge, not at the registry date.
    assert out["dateofdeath"].tolist() == [160 * minute, 300 * minute, 300 * minute]


def test_an_outcome_recorded_after_icu_discharge_survives_the_episode_clip():
    frame = pd.DataFrame({"stay": [1, 2, 3], "death": [True, True, False]})
    hours = pd.Series([-30.0, 200.0, 10.0])
    upper = pd.Series([50.0, 50.0, 50.0])

    clipped = _quarantine_rows_outside_icu_episode(
        frame, event_hours=hours, upper_hours=upper, allowed_pre_hours=24, db_label="eICU",
    )
    kept = _quarantine_rows_outside_icu_episode(
        frame, event_hours=hours, upper_hours=upper, allowed_pre_hours=24, db_label="eICU",
        keep_post_discharge=True,
    )

    assert clipped["stay"].tolist() == [3]
    # Only the pre-admission side still applies to an outcome status.
    assert kept["stay"].tolist() == [2, 3]
    assert "death" in POST_DISCHARGE_OUTCOME_CONCEPTS
