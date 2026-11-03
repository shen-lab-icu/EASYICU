"""SICdb invasive ventilation is attributed to a documented airway.

2026-11-03 extraction audit 2 (#13): ventilator settings cover far more
SICdb cases than the airway intervals, but SICdb has no NIV interface item,
so a setting hour counts as invasive ventilation only next to an airway.
"""

from __future__ import annotations

import json
from importlib import resources

import pandas as pd

from easyicu.concept.sicdb_ventilation import (
    UNATTRIBUTED_ATTR,
    attributed_ventilation_windows,
)
from easyicu.table.duration import UNIT_SECONDS, get_dur_var_unit

H = 3600


def _settings(case, hours):
    return pd.DataFrame({"CaseID": case, "Offset": [h * H for h in hours]})


def test_setting_hours_count_only_next_to_an_airway():
    settings = pd.concat(
        [
            # case 1: tracheal cannula 10-20 h; settings 9..12 and 30..31.
            _settings(1, [9, 10, 11, 12, 30, 31]),
            # case 2: no airway at all (post-operative or NIV: unknown).
            _settings(2, [0, 1, 2, 3]),
        ]
    )
    airway = pd.DataFrame({"CaseID": [1], "Offset": [10 * H], "OffsetEnd": [20 * H]})

    out = attributed_ventilation_windows(settings, airway, concept_name="mech_vent")

    assert out[["CaseID", "Offset", "dur_var"]].values.tolist() == [[1, 9 * H, 4 * H]]
    assert out["mech_vent"].tolist() == ["invasive"]
    # 30-31 h (case 1) and all four hours of case 2 are not attributed.
    assert out.attrs[UNATTRIBUTED_ATTR] == 6
    assert get_dur_var_unit(out) == UNIT_SECONDS


def test_a_gap_in_settings_splits_the_window():
    settings = _settings(1, [0, 1, 3])
    airway = pd.DataFrame({"CaseID": [1], "Offset": [0], "OffsetEnd": [10 * H]})

    out = attributed_ventilation_windows(settings, airway, concept_name="mech_vent")

    assert out[["Offset", "dur_var"]].values.tolist() == [[0, 2 * H], [3 * H, H]]


def test_dictionary_reads_the_tube_whole_and_the_cannula_only_with_settings():
    dictionary = json.loads(
        resources.files("easyicu.data").joinpath("concept-dict.json").read_text(encoding="utf-8")
    )
    tube, settings = dictionary["mech_vent"]["sources"]["sic"]

    assert tube["ids"] == [720] and tube["table"] == "data_range"
    assert settings["table"] == "data_float_h"
    assert settings["callback"] == "sicdb_airway_attributed_ventilation"
    assert 2283 not in settings["ids"]  # FiO2 alone is not ventilation
    assert set(settings["ids"]) == {2278, 2282, 2284, 3040, 2285, 3121}
    ett = dictionary["ett_gcs"]["sources"]["sic"][0]
    assert ett["ids"] == [720, 3041] and ett["dur_var"] == "OffsetEnd"


def test_sedated_set_rate_hours_in_icu_units_count_without_an_airway():
    # 2026-11-03 extraction audit round 2 (P1): case 2 has no airway, but its
    # hours 1-2 have a set rate under propofol in an ICU unit.
    settings = pd.concat([_settings(1, [9, 10]), _settings(2, [0, 1, 2, 3])])
    airway = pd.DataFrame({"CaseID": [1], "Offset": [10 * H], "OffsetEnd": [20 * H]})
    sedated = pd.DataFrame({"CaseID": [2, 2, 2], "_h": [1.0, 2.0, 7.0]})

    out = attributed_ventilation_windows(
        settings, airway, concept_name="mech_vent", sedated_hours=sedated
    )

    # Hour 7 has no setting, so it is not ventilation.
    assert out[["CaseID", "Offset", "dur_var"]].values.tolist() == [
        [1, 9 * H, 2 * H], [2, H, 2 * H]]
    assert out.attrs[UNATTRIBUTED_ATTR] == 2
    strict = attributed_ventilation_windows(settings, airway, concept_name="mech_vent")
    assert strict["CaseID"].tolist() == [1]
