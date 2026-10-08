"""An export records the stays each of its criteria excluded, and how it capped them.

The export's cohort report counted the source's stays and the stays left after
all demographic criteria together, so its age bound, first-stay restriction
and minimum ICU stay could not be told apart.  An export without criteria did
not count the source at all, and a capped export did not say which stays its
cap kept.  The report now lists each demographic criterion with the stays it
excluded (``demographic_steps``), counts the source wherever the export holds
all of it, names its unit (ICU stays), and records a cap with the rule that
chose the stays it kept: neither rule is a random sample.  The stays selected
are unchanged.  Synthetic demographics; the export API is a stand-in.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.patient_filter import PatientFilter
from easyicu.webserver import dataio


_ADULT_FIRST = {
    "preset": "adult_first",
    "age_min": 18,
    "min_icu_los_hours": 24,
    "observation_window_hours": 48,
    "exclude_readmissions": True,
}


def _demographics() -> pd.DataFrame:
    # Stay 12 is a child, 13 has no age, 14 is a readmission, 15 stayed 10 h;
    # stays 11, 16, 17 and 18 meet every criterion.
    return pd.DataFrame(
        {
            "patient_id": [11, 12, 13, 14, 15, 16, 17, 18],
            "age": [70.0, 12.0, None, 55.0, 60.0, 45.0, 80.0, 33.0],
            "first_icu_stay": [True, True, True, False, True, True, True, True],
            "los_hours": [48.0, 72.0, 30.0, 96.0, 10.0, 26.0, 25.0, 100.0],
        }
    )


def _api(ids: list[int], order: str | None = "source_file_order") -> SimpleNamespace:
    def get_all_patient_ids(path, database=None, max_patients=None, listing=None):
        if listing is not None and order is not None:
            listing["order"] = order
        return (ids[:max_patients] if max_patients else list(ids)), "stay_id"

    return SimpleNamespace(
        get_id_col_for_database=lambda database: "stay_id",
        get_all_patient_ids=get_all_patient_ids,
    )


@pytest.fixture
def demographics(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    frame = _demographics()
    monkeypatch.setattr(PatientFilter, "_load_demographics", lambda self: frame)
    return frame


def _report(cohort, *, max_patients=None, ids=(), order="source_file_order"):
    return dataio._resolve_export_cohort(
        "/unused", "miiv", cohort, max_patients, _api(list(ids), order)
    )["cohort_report"]


def test_each_demographic_criterion_records_the_stays_it_excluded(demographics) -> None:
    report = _report(_ADULT_FIRST)

    assert report["count_unit"] == "icu_stay"
    assert report["source_total"] == len(demographics)
    assert [
        (step["criterion"], step["n_before"], step["n_excluded"], step["n_remaining"])
        for step in report["demographic_steps"]
    ] == [("age", 8, 2, 6), ("first_icu_stay", 6, 1, 5), ("los", 5, 1, 4)]
    assert report["demographic_steps"][0]["n_excluded_missing"] == 1
    assert report["demographic_steps"][0]["parameters"] == {"age_min": 18}
    assert (
        report["demographic_steps"][-1]["n_remaining"]
        == report["selected_before_concept_prefilter"]
        == report["selected"]
        == 4
    )
    assert "cap" not in report


def test_a_cap_records_that_it_kept_the_first_stays_by_identifier(demographics) -> None:
    resolved = dataio._resolve_export_cohort(
        "/unused", "miiv", _ADULT_FIRST, 3, _api([])
    )
    report = resolved["cohort_report"]

    assert resolved["patient_ids"] == {"stay_id": [11, 16, 17]}
    assert report["selected_before_cap"] == 4
    assert report["selected"] == 3
    assert report["cap"] == {
        "max_patients": 3,
        "rule": "identifier_text_order",
        "cut": True,
    }
    assert _report(_ADULT_FIRST, max_patients=10)["cap"] == {
        "max_patients": 10,
        "rule": "identifier_text_order",
        "cut": False,
    }


def test_an_export_of_every_stay_counts_the_source_it_holds_whole() -> None:
    report = _report({"preset": "all_icu"}, ids=[1, 2, 3])

    assert report["count_unit"] == "icu_stay"
    assert report["source_total"] == report["selected_before_cap"] == 3
    assert report["selected"] == 3
    assert "cap" not in report


def test_a_full_cap_on_every_stay_leaves_the_source_uncounted() -> None:
    full = _report({"preset": "all_icu"}, max_patients=2, ids=[1, 2, 3])
    short = _report({"preset": "all_icu"}, max_patients=5, ids=[1, 2, 3])

    assert full["selected"] == 2
    assert "source_total" not in full
    assert full["cap"] == {
        "max_patients": 2,
        "rule": "source_file_order",
        "cut": None,
    }
    # Fewer stays than the cap: the export holds every stay of the source.
    assert short["source_total"] == short["selected"] == 3
    assert short["cap"]["cut"] is False


def test_rows_sharing_a_stay_are_counted_as_the_filter_read_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.concat([_demographics(), _demographics().iloc[[0]]], ignore_index=True)
    monkeypatch.setattr(PatientFilter, "_load_demographics", lambda self: frame)

    report = _report(_ADULT_FIRST)

    # The steps count rows; the selection counts distinct stays.  The report
    # keeps both as read, so a reader can see that they disagree.
    assert report["source_total"] == 9
    assert report["demographic_steps"][-1]["n_remaining"] == 5
    assert report["selected_before_concept_prefilter"] == 4


def test_a_cap_names_the_order_the_discovery_read_the_stays_in() -> None:
    sorted_sample = _report(
        {"preset": "all_icu"}, max_patients=2, ids=[1, 2, 3], order="identifier_order"
    )
    unnamed = _report({"preset": "all_icu"}, max_patients=2, ids=[1, 2, 3], order=None)

    assert sorted_sample["cap"]["rule"] == "identifier_order"
    assert unnamed["cap"]["rule"] == "unrecorded"


def test_a_filter_that_does_not_count_its_criteria_reports_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    class UncountedFilter:
        def __init__(
            self, database: str, data_path: str, verbose: bool = False
        ) -> None:
            self._last_original_count = 3

        def filter(self, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame({"patient_id": [11, 16]})

    monkeypatch.setattr(patient_filter_module, "PatientFilter", UncountedFilter)

    report = _report(_ADULT_FIRST)

    # Its criteria then stand as one step: the source's stays and those left.
    assert "demographic_steps" not in report
    assert report["source_total"] == 3
    assert report["selected_before_concept_prefilter"] == 2
