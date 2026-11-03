"""Infusion rate/episode callbacks align on the ICU stay and its clock.

Regressions from the 2026-11 extraction audit:

* MIMIC-III MetaVision ``mimic_rate_mv`` joined ICU ``intime`` on the first
  column containing "id" -- ``row_id`` -- so most infusion rows matched no
  icustays row and were dropped (norepinephrine: thousands of stays raw, a few
  dozen exported).  A MIMIC-IV frame carrying ``subject_id`` would likewise
  join a multi-stay patient's rows to every stay's clock.
* SICdb ``sic_dur`` collapsed every administration of a stay into one
  ``min(start)..max(stop)`` row on the primary-admission clock instead of
  per-episode durations on the ICU clock.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource
from easyicu.table import ICUTable
from easyicu.utils.callback_utils import mimic_rate_mv


class _TableSource:
    """Minimal data source serving fixed tables to the callback dispatcher."""

    def __init__(self, **tables: pd.DataFrame) -> None:
        self._tables = tables

    def load_table(self, table_name, columns=None, verbose=False):
        del verbose
        frame = self._tables[table_name]
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]
        return ICUTable(data=frame.copy(), id_columns=[frame.columns[0]])


def _mv_source() -> ConceptSource:
    return ConceptSource.from_mapping(
        {
            "ids": 221906,
            "table": "inputevents_mv",
            "sub_var": "itemid",
            "stop_var": "endtime",
            "callback": "mimic_rate_mv",
        }
    )


def _mimic_iii_inputevents_mv() -> pd.DataFrame:
    # Native MIMIC-III column order: row_id precedes the stay identifier and
    # is unrelated to icustays.row_id.
    return pd.DataFrame(
        {
            "row_id": [901, 902, 903],
            "subject_id": [10, 10, 20],
            "hadm_id": [100, 100, 200],
            "icustay_id": [1000, 1000, 2000],
            "starttime": pd.to_datetime(
                ["2101-01-01 01:30", "2101-01-01 03:10", "2101-02-01 00:20"]
            ),
            "endtime": pd.to_datetime(
                ["2101-01-01 03:10", "2101-01-01 03:40", "2101-02-01 01:50"]
            ),
            "itemid": [221906, 221906, 221906],
            "rate": [0.05, 0.08, 0.2],
            "rateuom": ["mcg/kg/min"] * 3,
            "cgid": [5, 5, 6],
            "orderid": [1, 2, 3],
            "linkorderid": [1, 1, 3],
        }
    )


def _mimic_iii_icustays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [1, 2],
            "subject_id": [10, 20],
            "hadm_id": [100, 200],
            "icustay_id": [1000, 2000],
            "intime": pd.to_datetime(["2101-01-01 00:45", "2101-02-01 00:00"]),
        }
    )


def test_mimic_iii_mv_rate_keeps_every_stay_on_its_icu_clock() -> None:
    result = _apply_callback(
        _mimic_iii_inputevents_mv(),
        _mv_source(),
        "norepi_rate",
        data_source=_TableSource(icustays=_mimic_iii_icustays()),
    )

    assert sorted(result["icustay_id"].unique()) == [1000, 2000]
    assert {"row_id", "subject_id", "orderid"}.isdisjoint(result.columns)
    stay = result.loc[result["icustay_id"].eq(1000)].sort_values("starttime", kind="stable")
    # Floors run on ICU-relative hours from intime 00:45:
    # 01:30 -> 0.75 h, 03:10 -> 2.42 h, 03:40 -> 2.92 h.
    assert stay["starttime"].tolist() == list(
        pd.to_datetime(
            ["2101-01-01 00:45", "2101-01-01 01:45", "2101-01-01 02:45", "2101-01-01 02:45"]
        )
    )
    assert stay["norepi_rate"].tolist() == pytest.approx([0.05, 0.05, 0.05, 0.08])
    other = result.loc[result["icustay_id"].eq(2000)].sort_values("starttime", kind="stable")
    assert other["starttime"].tolist() == list(
        pd.to_datetime(["2101-02-01 00:00", "2101-02-01 01:00"])
    )


def test_miiv_mv_rate_does_not_cross_a_patients_stays() -> None:
    inputevents = pd.DataFrame(
        {
            "subject_id": [10, 10],
            "hadm_id": [100, 101],
            "stay_id": [1000, 1001],
            "caregiver_id": [7, 7],
            "starttime": pd.to_datetime(["2150-01-01 02:30", "2150-06-01 05:10"]),
            "endtime": pd.to_datetime(["2150-01-01 02:59", "2150-06-01 05:20"]),
            "itemid": [221906, 221906],
            "rate": [0.1, 0.3],
            "rateuom": ["mcg/kg/min"] * 2,
            "orderid": [1, 2],
            "linkorderid": [1, 2],
        }
    )
    icustays = pd.DataFrame(
        {
            "subject_id": [10, 10],
            "hadm_id": [100, 101],
            "stay_id": [1000, 1001],
            "intime": pd.to_datetime(["2150-01-01 00:00", "2150-06-01 04:30"]),
        }
    )

    result = _apply_callback(
        inputevents,
        _mv_source(),
        "norepi_rate",
        data_source=_TableSource(icustays=icustays),
    ).sort_values(["stay_id", "starttime"])

    # One row per infused ICU hour; a subject-keyed join attached both
    # intimes to every row and duplicated each infusion onto the other clock.
    assert result["stay_id"].tolist() == [1000, 1001]
    assert result["starttime"].tolist() == list(
        pd.to_datetime(["2150-01-01 02:00", "2150-06-01 04:30"])
    )
    assert result["norepi_rate"].tolist() == pytest.approx([0.1, 0.3])


def test_mimic_rate_mv_infers_only_the_stay_identifier() -> None:
    # The full icustays frame shares row_id/subject_id/hadm_id with
    # inputevents_mv; only icustay_id may key the intime join.
    admission_times = _mimic_iii_icustays()

    result = mimic_rate_mv(
        _mimic_iii_inputevents_mv(),
        val_col="rate",
        unit_col="rateuom",
        stop_var="endtime",
        admission_times=admission_times,
    )

    assert sorted(result["icustay_id"].unique()) == [1000, 2000]
    assert len(result) == 6
    assert "row_id" not in result.columns


def _sic_dur_source() -> ConceptSource:
    return ConceptSource.from_mapping(
        {
            "ids": [1562],
            "table": "medication",
            "sub_var": "DrugID",
            "stop_var": "OffsetDrugEnd",
            "callback": "sic_dur",
        }
    )


def _sic_cases() -> pd.DataFrame:
    return pd.DataFrame({"CaseID": [1, 2], "ICUOffset": [1_800, 0]})


def test_sic_duration_reports_per_episode_spans_on_the_icu_clock() -> None:
    frame = pd.DataFrame(
        {
            "CaseID": [1, 1, 1, 1, 2],
            "DrugID": [1562] * 5,
            # Seconds from the primary admission.  Case 1: two contiguous
            # rate segments, a nested segment, then a 2 h pause and a new
            # infusion.  Case 2: one segment.
            "Offset": [1_800, 5_400, 6_000, 16_200, 0],
            "OffsetDrugEnd": [5_400, 9_000, 6_600, 19_800, 2_700],
            "norepi_dur": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    result = _apply_callback(
        frame,
        _sic_dur_source(),
        "norepi_dur",
        data_source=_TableSource(cases=_sic_cases()),
    )

    assert result["CaseID"].tolist() == [1, 1, 2]
    # ICU-relative start hours: (Offset - ICUOffset) / 3600, not floored.
    assert result["Offset"].tolist() == pytest.approx([0.0, 4.0, 0.0])
    assert result["norepi_dur"].tolist() == pytest.approx([2.0, 1.0, 0.75])
    assert result.attrs["easyicu_time_columns"] == {"Offset": "icu_relative_hours"}


def test_sic_duration_merges_five_minute_gaps_and_splits_longer_ones() -> None:
    frame = pd.DataFrame(
        {
            "CaseID": [2, 2, 2],
            "DrugID": [1562] * 3,
            "Offset": [0, 3_900, 7_560],
            "OffsetDrugEnd": [3_600, 7_200, 9_000],
            "norepi_dur": [1.0, 1.0, 1.0],
        }
    )

    result = _apply_callback(
        frame,
        _sic_dur_source(),
        "norepi_dur",
        data_source=_TableSource(cases=_sic_cases()),
    )

    # 300 s gap joins the first episode; a 360 s gap starts a new one.
    assert result["Offset"].tolist() == pytest.approx([0.0, 2.1])
    assert result["norepi_dur"].tolist() == pytest.approx([2.0, 0.4])


def test_sic_duration_refuses_a_case_without_an_icu_origin() -> None:
    frame = pd.DataFrame(
        {
            "CaseID": [3],
            "DrugID": [1562],
            "Offset": [0],
            "OffsetDrugEnd": [3_600],
            "norepi_dur": [1.0],
        }
    )

    with pytest.raises(ValueError, match="ICUOffset"):
        _apply_callback(
            frame,
            _sic_dur_source(),
            "norepi_dur",
            data_source=_TableSource(cases=_sic_cases()),
        )
