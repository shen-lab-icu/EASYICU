"""An export records the stays each of its criteria excluded, and how it capped them.

The export's cohort report counted the source's stays and the stays left after
all demographic criteria together, so its age bound, first-stay restriction
and minimum ICU stay could not be told apart.  An export without criteria did
not count the source at all, and a capped export did not say which stays its
cap kept.  The report now lists each demographic criterion with the stays it
excluded (``demographic_steps``), counts the source wherever the export holds
all of it, names its unit (ICU stays), and records a cap with the rule that
chose the stays it kept.  Binding caps use a deterministic hash sample;
other requests preserve the original selection and report.  Synthetic
demographics; the export API is a stand-in.
"""

from __future__ import annotations

import json
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


def test_a_binding_cap_records_its_sample_and_a_nonbinding_cap_is_unchanged(
    demographics,
) -> None:
    resolved = dataio._resolve_export_cohort(
        "/unused", "miiv", _ADULT_FIRST, 3, _api([])
    )
    report = resolved["cohort_report"]

    assert resolved["patient_ids"] == {"stay_id": [11, 16, 17]}
    assert report["selected_before_cap"] == 4
    assert report["selected"] == 3
    assert report["cap"] == {
        "max_patients": 3,
        "rule": "seeded_random_sample",
        "cut": True,
    }
    assert report["sampling"] == {
        "method": "seeded_hash", "seed": 42, "eligible": 4, "selected": 3,
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


def test_a_binding_cap_on_every_stay_counts_the_whole_eligible_source() -> None:
    full = _report({"preset": "all_icu"}, max_patients=2, ids=[1, 2, 3])
    short = _report({"preset": "all_icu"}, max_patients=5, ids=[1, 2, 3])

    assert full["selected"] == 2
    assert full["source_total"] == full["selected_before_cap"] == 3
    assert full["cap"] == {
        "max_patients": 2,
        "rule": "seeded_random_sample",
        "cut": True,
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


def test_a_binding_cap_records_sampling_independently_of_discovery_order() -> None:
    sorted_sample = _report(
        {"preset": "all_icu"}, max_patients=2, ids=[1, 2, 3], order="identifier_order"
    )
    unnamed = _report({"preset": "all_icu"}, max_patients=2, ids=[1, 2, 3], order=None)

    assert sorted_sample["cap"]["rule"] == "seeded_random_sample"
    assert unnamed["cap"]["rule"] == "seeded_random_sample"
    assert sorted_sample["sampling"] == unnamed["sampling"]


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


def test_a_hash_sample_is_stable_across_input_orders_and_seeds() -> None:
    ids = list(range(1, 21))

    sampled, metadata = dataio._sample_export_cohort_ids(ids, 5)

    # Frozen SHA-256 ranking for seed 42, independent of Python's hash seed.
    assert dataio._EXPORT_COHORT_SAMPLE_SEED == 42
    assert sampled == [1, 6, 11, 13, 20]
    assert dataio._sample_export_cohort_ids(ids, 5) == (sampled, metadata)
    assert dataio._sample_export_cohort_ids(list(reversed(ids)), 5) == (
        sampled, metadata
    )
    assert dataio._sample_export_cohort_ids(ids[::2] + ids[1::2], 5) == (
        sampled, metadata
    )
    assert dataio._sample_export_cohort_ids(ids, 5, seed=43)[0] == [2, 6, 14, 15, 19]


def test_a_capped_sample_spans_the_id_range_and_grows_by_inclusion() -> None:
    ids = list(range(1, 10_001))

    small, _ = dataio._sample_export_cohort_ids(ids, 100)
    large, _ = dataio._sample_export_cohort_ids(list(reversed(ids)), 200)

    assert len(small) == len(set(small)) == 100
    assert len(large) == len(set(large)) == 200
    assert small != ids[:100]
    assert min(small) <= 5_000 < max(small)
    assert set(small) < set(large) <= set(ids)
    assert small == sorted(small)
    assert large == sorted(large)


@pytest.mark.parametrize("filtered", [False, True], ids=["all_icu", "filtered"])
def test_both_export_paths_sample_and_report_only_aggregate_counts(
    monkeypatch: pytest.MonkeyPatch, filtered: bool
) -> None:
    import easyicu.patient_filter as patient_filter_module

    ids = list(range(20, 0, -1))
    listing_caps = []
    api = _api(ids)
    original_listing = api.get_all_patient_ids

    def list_ids(path, **kwargs):
        listing_caps.append(kwargs.get("max_patients"))
        return original_listing(path, **kwargs)

    api.get_all_patient_ids = list_ids

    class FakeFilter:
        def __init__(self, **kwargs):
            self._last_original_count = len(ids)

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": ids})

    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakeFilter)
    events = []
    resolved = dataio._resolve_export_cohort(
        "/unused", "eicu", {"age_min": 18} if filtered else None, 5, api,
        progress=lambda stage, extra: events.append((stage, extra)),
    )

    assert resolved["patient_ids"] == {"stay_id": [1, 6, 11, 13, 20]}
    assert listing_caps == ([] if filtered else [None])
    assert resolved["cohort_size"] == 5
    report = resolved["cohort_report"]
    assert report["source_total"] == report["selected_before_cap"] == 20
    assert report["selected"] == 5
    assert report["max_patients_applied"] is True
    assert report["cap"] == {
        "max_patients": 5, "rule": "seeded_random_sample", "cut": True,
    }
    assert report["sampling"] == {
        "method": "seeded_hash", "seed": 42, "eligible": 20, "selected": 5,
    }
    selection_event = dict(events)["cohort_selected" if filtered else "all_icu_selected"]
    assert selection_event == {
        "selected": 5,
        **({"selected_before_cap": 20} if filtered else {}),
        "max_patients_applied": True,
        "sampling": report["sampling"],
    }
    for payload in [report, *[extra for _, extra in events]]:
        assert "patient_ids" not in payload
        assert "stay_id" not in payload


def test_a_cap_samples_only_the_stays_left_after_every_eligibility_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.patient_filter as patient_filter_module

    class FakeFilter:
        def __init__(self, **kwargs):
            self._last_original_count = 50

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": list(range(1, 41))})

    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakeFilter)
    monkeypatch.setattr(
        dataio, "_match_concept_derived_cohort_ids", lambda *args: set(range(1, 31))
    )
    monkeypatch.setattr(
        dataio, "_match_icd_ids",
        lambda *args: (set(range(1, 26)), set(range(21, 26))),
    )

    resolved = dataio._resolve_export_cohort(
        "/unused", "miiv", {
            "preset": "sepsis3", "icd_enabled": True,
            "icd_include": ["A123"], "icd_exclude": ["B456"],
        }, 5, _api([]),
    )

    assert resolved["patient_ids"] == {"stay_id": [1, 6, 11, 13, 20]}
    report = resolved["cohort_report"]
    assert report["source_total"] == 50
    assert report["selected_before_concept_prefilter"] == 40
    assert report["concept_matches"] == report["selected_before_icd"] == 30
    assert report["selected_before_cap"] == report["sampling"]["eligible"] == 20
    assert report["selected"] == report["sampling"]["selected"] == 5


@pytest.mark.parametrize("filtered", [False, True], ids=["all_icu", "filtered"])
@pytest.mark.parametrize("max_patients", [None, 0, 3, 10])
def test_an_export_not_reduced_by_its_cap_keeps_its_serialized_report(
    monkeypatch: pytest.MonkeyPatch, filtered: bool, max_patients: int | None
) -> None:
    import easyicu.patient_filter as patient_filter_module

    ids = [20, 3, 11]

    class FakeFilter:
        def __init__(self, **kwargs):
            self._last_original_count = 3

        def filter(self, **kwargs):
            return pd.DataFrame({"patient_id": ids})

    monkeypatch.setattr(patient_filter_module, "PatientFilter", FakeFilter)
    events = []
    resolved = dataio._resolve_export_cohort(
        "/unused", "miiv", {"age_min": 18} if filtered else None,
        max_patients, _api(ids),
        progress=lambda stage, extra: events.append((stage, extra)),
    )
    assert resolved["patient_ids"] == {"stay_id": [11, 20, 3] if filtered else ids}
    expected = {
        "mode": "all_icu",
        "count_unit": "icu_stay",
        **({"source_total": 3} if filtered else {}),
        "selected": 3,
        **({
            "selected_before_cap": 3,
            "selected_before_concept_prefilter": 3,
            "concept_matches": None,
            "selected_before_icd": 3,
        } if filtered else {}),
        "max_patients_applied": False if filtered else bool(max_patients),
        **({"applied_filters": []} if not filtered else {}),
        **({"source_total": 3, "selected_before_cap": 3}
           if not filtered and max_patients != 3 else {}),
        **({"cap": {
            "max_patients": max_patients,
            "rule": "identifier_text_order" if filtered else "source_file_order",
            "cut": None if not filtered and max_patients == 3 else False,
        }} if max_patients else {}),
        **({
            "applied_filters": ["demographics"],
            "icd": {
                "enabled": False, "include_tokens": [], "exclude_tokens": [],
                "include_matches": 0, "exclude_matches": 0,
            },
        } if filtered else {}),
    }
    # Compare JSON bytes, including field order and the historical full-cap case.
    assert json.dumps(resolved["cohort_report"]) == json.dumps(expected)
    selected_event = dict(events)["cohort_selected" if filtered else "all_icu_selected"]
    assert json.dumps(selected_event) == json.dumps({
        "selected": 3,
        **({"selected_before_cap": 3} if filtered else {}),
        "max_patients_applied": False if filtered else bool(max_patients),
    })
