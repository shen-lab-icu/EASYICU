"""Published eICU mech_vent hour category is pooled by membership, not by row order.

Audit round 7 (R7-F013): eICU mech_vent is point evidence.  Its concept table
keeps every level of an hour (R6-F039 pins this), but the wide merge kept the
last row per stay-hour, so the published category followed source/row order
after the minute clock was gone (a 3,000-stay sample: all 56,575 published
hours equal the last row; 2,238 of 5,241 mixed hours differ from the earliest
raw category).  Rule R5 allows NIV and IMV in the same hour: such an hour
publishes the declared mixed level, as MIMIC/MIIV windows already do.
Synthetic frames and invented eICU tables only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.utils.compat import merge_concepts_r_style

ID = "patientunitstayid"


def _published(rows) -> dict:
    frame = pd.DataFrame(rows, columns=[ID, "charttime", "mech_vent"])
    merged = merge_concepts_r_style({"mech_vent": frame}, id_col=ID, time_col="charttime")
    return {(int(r[ID]), float(r["charttime"])): r["mech_vent"] for _, r in merged.iterrows()}


@pytest.mark.parametrize(
    "rows",
    [
        [(1, 3.0, "invasive"), (1, 3.0, "noninvasive")],
        [(1, 3.0, "noninvasive"), (1, 3.0, "invasive")],
        [(1, 3.0, "invasive|noninvasive"), (1, 3.0, "invasive")],
    ],
)
def test_merge_publishes_the_mixed_level_whatever_the_row_order(rows):
    assert _published(rows) == {(1, 3.0): "invasive|noninvasive"}


def test_merge_keeps_single_categories_and_ignores_missing_levels():
    rows = [
        (1, 3.0, "noninvasive"), (1, 3.0, "noninvasive"),  # pure NIV hour
        (1, 4.0, "invasive"), (1, 4.0, None),              # a missing level adds nothing
        (2, 0.0, "invasive"),                              # untouched single row
    ]
    assert _published(rows) == {(1, 3.0): "noninvasive", (1, 4.0): "invasive", (2, 0.0): "invasive"}


# --- resolver level: invented eICU tables through the real concept loader ---------------

def _load(tmp_path, offsets_values, names=("mech_vent", "vent_ind")):
    from easyicu.concept import ConceptDictionary, ConceptResolver
    from tests.support.eicu_source import DICTIONARY, _eicu_source

    resp = pd.DataFrame({
        "respchartoffset": [offset for offset, _ in offsets_values],
        "respchartvaluelabel": ["O2 Device"] * len(offsets_values),
        "respchartvalue": [value for _, value in offsets_values],
    })
    source = _eicu_source(tmp_path, "eicu", respiratorycharting=resp)
    merged = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        list(names), source, merge=True, r_compatible=True, interval=pd.Timedelta(hours=1),
        patient_ids={ID: [1]}, concept_workers=1, verbose=False,
    )
    frame = merged.data if hasattr(merged, "data") else merged
    time_col = next(c for c in ("charttime", "respchartoffset", "time") if c in frame.columns)
    time = frame[time_col]
    if pd.api.types.is_timedelta64_dtype(time):
        time = time / pd.Timedelta(hours=1)
    return frame.assign(_h=time.astype(float)).set_index("_h")


@pytest.mark.parametrize(
    "rows",
    [
        [(70, "Ventilator"), (100, "BiPAP")],   # IMV -> NIV in hour 1
        [(100, "BiPAP"), (70, "Ventilator")],   # same evidence, other row order
        [(90, "BiPAP"), (90, "Ventilator")],    # exact same-minute conflict
        [(90, "Ventilator"), (90, "BiPAP")],
        [(70, "BiPAP"), (100, "Ventilator")],   # NIV -> IMV in hour 1
    ],
)
def test_published_eicu_mixed_hour_is_mixed_and_keeps_vent_ind(tmp_path, rows):
    out = _load(tmp_path, rows)
    assert out.loc[1.0, "mech_vent"] == "invasive|noninvasive"
    assert bool(out.loc[1.0, "vent_ind"]) is True


def test_published_eicu_pure_niv_hour_is_niv_without_vent_ind(tmp_path):
    out = _load(tmp_path, [(70, "BiPAP"), (100, "CPAP")])
    assert out.loc[1.0, "mech_vent"] == "noninvasive"
    assert not bool(out["vent_ind"].fillna(False).any())
