"""Regression tests for the 2026-08-16 data-foundation review fixes."""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.scores import sepsis_sofa2


def test_sepsis_sofa2_first_si_event_is_time_sorted() -> None:
    sofa2 = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0.0, 35.0],
            "sofa2": [0.0, 2.0],
        }
    )
    susp = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [20.0, 5.0],  # deliberately unsorted
            "susp_inf": [True, True],
        }
    )

    result = sepsis_sofa2.sep3_sofa2(
        sofa2,
        susp,
        id_cols=["stay_id"],
        index_col="charttime",
        si_window="first",
    )

    # The earliest SI event is t=5; its [t-48h, t+24h] window closes at t=29,
    # so the t=35 SOFA rise is outside the window and no sepsis is detected.
    assert "sep3_sofa2" in result.columns
    assert not result["sep3_sofa2"].any()


def test_si_and_includes_exact_window_boundary() -> None:
    from easyicu.scores.sepsis import _si_and

    abx = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "abx": [1]})
    samp = pd.DataFrame({"stay_id": [1], "charttime": [24.0], "samp": [1]})

    result = _si_and(
        abx,
        samp,
        id_cols=["stay_id"],
        index_col="charttime",
        abx_win=pd.Timedelta(hours=24),
        samp_win=pd.Timedelta(hours=72),
        keep_components=False,
    )

    # A sample exactly 24.0 h after the first antibiotic is a valid SI link.
    assert result["susp_inf"].tolist() == [True]


def test_eicu_v_codes_are_icd9_not_icd10() -> None:
    from easyicu.scores.comorbidity import _explode_eicu_codes

    long = _explode_eicu_codes(pd.Series(["427.31,V45.1"]))
    assert long["code"].tolist() == ["427.31", "V45.1"]
    assert long["version"].tolist() == [9, 9]


def test_circ_failure_missing_map_fails_closed() -> None:
    from easyicu.scores.circ_failure import calculate_circ_failure_status

    df = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0, 5],
            "lact": [3.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="MAP"):
        calculate_circ_failure_status(df)


def test_circ_failure_first_event_level_matches_first_time() -> None:
    from easyicu.scores.circ_failure import get_circ_failure_incidence

    df = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [300, 100],
            "circ_event": [3, 1],
        }
    )

    out = get_circ_failure_incidence(df).set_index("stay_id")

    assert out.loc[1, "first_circ_failure_time"] == 100
    assert out.loc[1, "first_event_level"] == 1
