"""Sepsis SI window tests.

Contract under test: when ``index_col`` is numeric, ``susp_inf`` interprets
those values as **hours since ICU admission** — the post
``concept._align_time_to_admission`` invariant used everywhere else (notably
``sepsis_sofa2.sep3_sofa2``).  A 24h ABX window therefore means
``|abx_t - samp_t| <= 24`` when the times are numeric.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver, ConceptSource, _apply_callback
from easyicu.concept.callbacks import (
    ConceptCallbackContext,
    _callback_susp_inf,
)
from easyicu.scores.sepsis import compute_sepsis3_onset, sep3, susp_inf
from easyicu.table import ICUTable
from easyicu.scores import sepsis_sofa2


def _eicu_context() -> ConceptCallbackContext:
    return ConceptCallbackContext(
        concept_name="susp_inf",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="eicu")),
        patient_ids=None,
        kwargs={"si_mode": "auto"},
    )


def test_mimic_sampling_event_is_true_even_when_culture_is_negative() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "chartdate": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "charttime": [pd.NaT, pd.Timestamp("2025-01-02 08:00")],
            "samp": [pd.NA, 12345],
        }
    )
    source = ConceptSource.from_mapping(
        {
            "table": "microbiologyevents",
            "val_var": "org_itemid",
            "index_var": "chartdate",
            "callback": "mimic_sampling",
            "aux_time": "charttime",
        }
    )

    result = _apply_callback(frame, source, concept_name="samp")

    assert result["samp"].tolist() == [True, True]
    assert result["chartdate"].tolist() == [
        pd.Timestamp("2025-01-01 12:00"),
        pd.Timestamp("2025-01-02 08:00"),
    ]


def test_recursive_suspicion_fails_closed_for_unavailable_timed_positivity() -> None:
    tables = {
        "abx": ICUTable(
            pd.DataFrame({"stay_id": [1], "charttime": [0.0], "abx": [True]}),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="abx",
        ),
        "samp": ICUTable(
            pd.DataFrame({"stay_id": [1], "charttime": [1.0], "samp": [True]}),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="samp",
        ),
    }
    context = ConceptCallbackContext(
        concept_name="susp_inf",
        target=None,
        interval=pd.Timedelta(hours=1),
        resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="miiv")),
        patient_ids=None,
        kwargs={"si_mode": "and", "positive_cultures": True},
    )

    with pytest.raises(ValueError, match="event-level culture-result timing"):
        _callback_susp_inf(tables, context)


def _eicu_icd_abx_tables(abx_times: list[float]) -> dict[str, ICUTable]:
    return {
        "infection_icd": ICUTable(
            pd.DataFrame(
                {
                    "patientunitstayid": [141203, 141203],
                    "diagnosisoffset": [1.0, 19.0],
                    "infection_icd": [True, True],
                }
            ),
            id_columns=["patientunitstayid"],
            index_column="diagnosisoffset",
            value_column="infection_icd",
        ),
        "abx": ICUTable(
            pd.DataFrame(
                {
                    "patientunitstayid": [141203] * len(abx_times),
                    "charttime": abx_times,
                    "abx": [True] * len(abx_times),
                }
            ),
            id_columns=["patientunitstayid"],
            index_column="charttime",
            value_column="abx",
        ),
    }


def test_eicu_icd_abx_uses_antibiotic_time_axis_without_sampling() -> None:
    """The infection diagnosis selects the stay; ABX alone owns SI timing."""
    abx_times = [-3.0, 1.0, 2.0, 18.0, 21.0, 30.0, 39.0]

    result = _callback_susp_inf(
        _eicu_icd_abx_tables(abx_times),
        _eicu_context(),
    )

    assert result.index_column == "charttime"
    assert list(result.data.columns) == [
        "patientunitstayid",
        "charttime",
        "susp_inf",
    ]
    assert result.data["charttime"].tolist() == abx_times
    assert result.data["susp_inf"].tolist() == [True] * len(abx_times)


def test_eicu_icd_abx_is_not_relocated_to_sampling_times_when_merged() -> None:
    """A sepsis_shared merge must retain ABX times, not broadcast onto samp."""
    abx_times = [340.0, 352.0, 401.0]
    samp_times = [2.0, 124.0, 133.0]
    tables = _eicu_icd_abx_tables(abx_times)
    suspicion = _callback_susp_inf(tables, _eicu_context())
    sampling = ICUTable(
        pd.DataFrame(
            {
                "patientunitstayid": [141203] * len(samp_times),
                "charttime": samp_times,
                "samp": [True] * len(samp_times),
            }
        ),
        id_columns=["patientunitstayid"],
        index_column="charttime",
        value_column="samp",
    )
    resolver = ConceptResolver({})
    data_source = SimpleNamespace(config=SimpleNamespace(name="eicu"))

    merged = resolver._to_r_format_merged_enhanced(
        {
            "susp_inf": suspicion,
            "infection_icd": tables["infection_icd"],
            "samp": sampling,
        },
        ["susp_inf", "infection_icd", "samp"],
        pd.Timedelta(hours=1),
        data_source=data_source,
    )

    positive_times = merged.loc[merged["susp_inf"].eq(True), "charttime"]
    assert positive_times.tolist() == abx_times
    sample_rows = merged[merged["charttime"].isin(samp_times)]
    assert sample_rows["samp"].eq(True).all()
    assert sample_rows["susp_inf"].isna().all()


def test_susp_inf_numeric_hour_offsets_match_within_abx_window():
    # samp 10h after abx → inside 24h abx_win
    abx = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "abx": [1]})
    samp = pd.DataFrame({"stay_id": [1], "charttime": [10.0], "samp": [1]})

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result["susp_inf"].tolist() == [True]
    assert result["charttime"].tolist() == [0.0]


def test_susp_inf_numeric_hour_offsets_match_within_samp_window():
    # abx 50h after samp → inside 72h samp_win, outside 24h abx_win
    abx = pd.DataFrame({"stay_id": [1], "charttime": [50.0], "abx": [1]})
    samp = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "samp": [1]})

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result["susp_inf"].tolist() == [True]
    # SI time is the earlier of the pair (samp_time here)
    assert result["charttime"].tolist() == [0.0]


def test_susp_inf_numeric_hour_offsets_outside_both_windows():
    # samp 100h after abx → outside both 24h abx_win and 72h samp_win
    abx = pd.DataFrame({"stay_id": [1], "charttime": [0.0], "abx": [1]})
    samp = pd.DataFrame({"stay_id": [1], "charttime": [100.0], "samp": [1]})

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result.empty


def test_susp_inf_datetime_offsets_use_timedelta_windows():
    # The datetime path is unaffected by the numeric refactor; sanity check.
    abx = pd.DataFrame(
        {"stay_id": [1], "charttime": [pd.Timestamp("2024-01-01 00:00")], "abx": [1]}
    )
    samp = pd.DataFrame(
        {"stay_id": [1], "charttime": [pd.Timestamp("2024-01-01 10:00")], "samp": [1]}
    )

    result = susp_inf(abx, samp, ["stay_id"], "charttime")

    assert result["susp_inf"].tolist() == [True]


def test_compute_sepsis3_onset_requires_explicit_true_susp_inf():
    sofa = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2, 3, 3],
            "charttime": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "sofa": [0, 3, 0, 3, 0, 3],
        }
    )
    si = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "charttime": [0.0, 0.0, 0.0],
            "susp_inf": [True, pd.NA, False],
        }
    )

    result = compute_sepsis3_onset(
        sofa,
        si,
        id_col="stay_id",
        sofa_time_col="charttime",
        si_time_col="charttime",
    )

    assert result["stay_id"].tolist() == [1]


def test_compute_sepsis3_onset_missing_susp_inf_column_fails_closed():
    sofa = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0.0, 1.0],
            "sofa": [0, 3],
        }
    )
    si = pd.DataFrame({"stay_id": [1], "charttime": [0.0]})

    result = compute_sepsis3_onset(
        sofa,
        si,
        id_col="stay_id",
        sofa_time_col="charttime",
        si_time_col="charttime",
    )

    assert result.empty


def test_sep3_accepts_nullable_boolean_suspicion_flags():
    sofa = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0.0, 1.0],
            "sofa": [0.0, 3.0],
        }
    )
    si = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0.0, 1.0],
            "susp_inf": pd.Series([True, pd.NA], dtype="boolean"),
        }
    )

    result = sep3(sofa, si, id_cols=["stay_id"], index_col="charttime")

    assert result[["stay_id", "charttime", "sep3"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 1.0, "sep3": True}
    ]


# --- SI window ordering and boundary (2026-08-16 data review) ---

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
