"""Tier-1 derived-index concepts (2026-06-22).

Pure row-wise derivations over 6/6 primitives: shock indices, BUN/creatinine
ratio, NLR/PLR, corrected calcium, oxygenation index, and CKD-EPI 2021 eGFR.
Offline tests exercise the real callbacks on synthetic tables (incl. the eGFR
formula against published values); real-data bounds checks are gated behind
``needs_real_data``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept_callbacks import (
    ConceptCallbackContext,
    _callback_bun_creatinine_ratio,
    _callback_corrected_calcium,
    _callback_egfr,
    _callback_nlr,
    _callback_oxygenation_index,
    _callback_persistent_critical_illness,
    _callback_shock_index,
)
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable

DERIVED = [
    "shock_index", "modified_shock_index", "diastolic_shock_index",
    "bun_creatinine_ratio", "nlr", "plr", "corrected_calcium",
    "oxygenation_index", "egfr", "persistent_critical_illness",
]


def _ctx(name: str) -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name=name, target=None, interval=pd.Timedelta(hours=1),
        resolver=None, data_source=None, patient_ids=None,
    )


def _ts(col: str, values, times=None) -> ICUTable:
    times = times or [pd.Timestamp("2026-01-01 08:00")] * len(values)
    return ICUTable(
        pd.DataFrame({"stay_id": [1] * len(values), "charttime": times, col: values}),
        id_columns=["stay_id"], index_column="charttime", value_column=col,
    )


def _id(col: str, value) -> ICUTable:
    return ICUTable(
        pd.DataFrame({"stay_id": [1], col: [value]}),
        id_columns=["stay_id"], index_column=None, value_column=col,
    )


# --- structure ---------------------------------------------------------------


def test_derived_concepts_present_in_main_dict():
    d = load_dictionary()
    for name in DERIVED:
        assert name in d, f"{name} not merged into concept-dict.json"
        assert d[name].sources == {} or d[name].sources is not None  # rec_cncpt


# --- offline callback logic --------------------------------------------------


def test_shock_index_divides_hr_by_sbp():
    times = [pd.Timestamp("2026-01-01 08:00"), pd.Timestamp("2026-01-01 12:00")]
    out = _callback_shock_index(
        {
            "hr": _ts("hr", [90.0, 120.0], times=times),
            "sbp": _ts("sbp", [120.0, 80.0], times=times),
        },
        _ctx("shock_index"),
    )
    assert out.value_column == "shock_index"
    assert out.data["shock_index"].tolist() == pytest.approx([0.75, 1.5])


def test_bun_creatinine_ratio():
    out = _callback_bun_creatinine_ratio(
        {"bun": _ts("bun", [40.0]), "crea": _ts("crea", [2.0])},
        _ctx("bun_creatinine_ratio"),
    )
    assert out.data["bun_creatinine_ratio"].tolist() == pytest.approx([20.0])


def test_nlr():
    out = _callback_nlr(
        {"neut": _ts("neut", [8.0]), "lymph": _ts("lymph", [2.0])}, _ctx("nlr")
    )
    assert out.data["nlr"].tolist() == pytest.approx([4.0])


def test_corrected_calcium_adjusts_for_albumin():
    # Ca 8.0, albumin 2.0 -> 8.0 + 0.8*(4-2) = 9.6
    out = _callback_corrected_calcium(
        {"ca": _ts("ca", [8.0]), "alb": _ts("alb", [2.0])}, _ctx("corrected_calcium")
    )
    assert out.data["corrected_calcium"].tolist() == pytest.approx([9.6])


def test_oxygenation_index_normalises_fio2_fraction():
    # FiO2 0.5 (fraction) -> 50%; OI = (50 * 12) / 60 = 10
    out = _callback_oxygenation_index(
        {
            "fio2": _ts("fio2", [0.5]),
            "mean_airway_pres": _ts("mean_airway_pres", [12.0]),
            "po2": _ts("po2", [60.0]),
        },
        _ctx("oxygenation_index"),
    )
    assert out.data["oxygenation_index"].tolist() == pytest.approx([10.0])


@pytest.mark.parametrize(
    "sex,age,crea,expected",
    [
        ("F", 50, 0.9, 78.0),  # CKD-EPI 2021 published ~78
        ("M", 60, 1.2, 69.0),  # ~69
    ],
)
def test_egfr_ckd_epi_2021_matches_published(sex, age, crea, expected):
    out = _callback_egfr(
        {
            "crea": _ts("crea", [crea]),
            "age": _id("age", age),
            "sex": _id("sex", sex),
        },
        _ctx("egfr"),
    )
    assert out.data["egfr"].iloc[0] == pytest.approx(expected, abs=2.0)


def test_persistent_critical_illness_thresholds_los():
    out = _callback_persistent_critical_illness(
        {"los_icu": ICUTable(
            pd.DataFrame({"stay_id": [1, 2, 3], "los_icu": [5.0, 10.0, 21.0]}),
            id_columns=["stay_id"], index_column=None, value_column="los_icu",
        )},
        _ctx("persistent_critical_illness"),
    )
    # >=10 days -> 1, else 0
    assert sorted(out.data["persistent_critical_illness"].tolist()) == [0.0, 1.0, 1.0]


def test_egfr_broadcasts_demographics_across_timeseries():
    # Two creatinine timepoints, one age/sex row -> two eGFR rows.
    out = _callback_egfr(
        {
            "crea": _ts(
                "crea", [0.9, 1.8],
                times=[pd.Timestamp("2026-01-01 08:00"), pd.Timestamp("2026-01-02 08:00")],
            ),
            "age": _id("age", 50),
            "sex": _id("sex", "F"),
        },
        _ctx("egfr"),
    )
    assert len(out.data) == 2
    # higher creatinine -> lower eGFR
    assert out.data["egfr"].iloc[0] > out.data["egfr"].iloc[1]


# --- gated real-data bounds --------------------------------------------------


@pytest.mark.needs_real_data
@pytest.mark.parametrize("database,concept,lo,hi", [
    ("miiv", "shock_index", 0.1, 3.0),
    ("miiv", "bun_creatinine_ratio", 1.0, 120.0),
    ("miiv", "nlr", 0.1, 200.0),
    ("miiv", "corrected_calcium", 4.0, 16.0),
    ("miiv", "oxygenation_index", 0.5, 80.0),
    ("miiv", "egfr", 1.0, 200.0),
    ("eicu", "shock_index", 0.1, 3.0),
    ("sic", "egfr", 1.0, 200.0),
])
def test_derived_extracts_within_bounds(database, concept, lo, hi):
    from easyicu import load_concepts

    df = load_concepts([concept], database=database)
    assert concept in df.columns
    vals = pd.to_numeric(df[concept], errors="coerce").dropna()
    assert len(vals) > 0
    assert vals.min() >= lo
    assert vals.max() <= hi
