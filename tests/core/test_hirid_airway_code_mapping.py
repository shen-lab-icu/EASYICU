"""HiRID ventilator airway code 6 is '-' (nothing recorded), not non-invasive.

2026-11-03 extraction audit 2 (P2): ordinal_vars_ref lists 15001552 codes
1 intubiert, 2 tracheotomiert, 3 Maske, 4 Helm, 5 Mundstück(Stellar), 6 '-'.
"""

from __future__ import annotations

import json
from importlib import resources

import pandas as pd

from easyicu.concept import ConceptSource, _apply_callback


def _hirid_mech_vent_source() -> ConceptSource:
    dictionary = json.loads(
        resources.files("easyicu.data").joinpath("concept-dict.json").read_text(encoding="utf-8")
    )
    return ConceptSource.from_mapping(dictionary["mech_vent"]["sources"]["hirid"][0])


def test_airway_code_six_is_missing_and_closes_the_previous_window():
    frame = pd.DataFrame(
        {
            "patientid": [1, 1, 1, 1],
            "datetime": [0.0, 2.0, 3.0, 10.0],
            "variableid": 15001552,
            "mech_vent": [1, 6, 3, 5],
        }
    )

    result = _apply_callback(frame, _hirid_mech_vent_source(), concept_name="mech_vent")

    values = result["mech_vent"].tolist()
    assert values[0] == "invasive" and values[2] == "noninvasive" and values[3] == "noninvasive"
    assert pd.isna(values[1])
    # The '-' row still ends the invasive record's window at hour 2.
    assert result["dur_var"].tolist()[0] == 2.0
