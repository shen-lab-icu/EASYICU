"""The patient filter counts the stays each criterion excludes, in the order it applies them.

A flow diagram states, for each selection criterion, how many stays it
excluded.  The filter applied every requested criterion as one mask and kept
only the stays before and after it, so an export could not say how many stays
its age bound, first-stay restriction or minimum ICU stay excluded.  It now
records one step per requested criterion: the stays before it, the stays it
excluded (and how many of them for want of a value) and the stays left.  A
stay is counted under the first criterion that excludes it.  The selected
stays are unchanged.  Synthetic demographics only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.patient_filter import PatientFilter, PatientFilterCriterionError


def _filter_with(frame: pd.DataFrame) -> PatientFilter:
    patient_filter = PatientFilter(database="miiv", data_path="/unused")
    patient_filter._demographics = frame
    return patient_filter


def _stays() -> pd.DataFrame:
    # Stay 2 is a child; stay 3 has no age; stay 4 is a readmission; stay 5 is
    # a readmitted child (excluded by age first); stay 6 stayed 10 h; stay 7
    # has no ICU length of stay; stays 1, 8 and 9 meet every criterion.
    return pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "age": [70.0, 12.0, None, 55.0, 15.0, 60.0, 45.0, 80.0, 33.0],
            "first_icu_stay": [True, True, True, False, False, True, True, True, True],
            "los_hours": [48.0, 72.0, 30.0, 96.0, 50.0, 10.0, None, 25.0, 100.0],
            "gender": ["M", "F", "F", "M", "M", "F", "M", "F", None],
            "survived": [True, True, False, True, True, True, False, True, True],
        }
    )


def _step(
    criterion, n_before, n_excluded, n_remaining, n_excluded_missing, **parameters
):
    return {
        "criterion": criterion,
        "parameters": parameters,
        "n_before": n_before,
        "n_excluded": n_excluded,
        "n_remaining": n_remaining,
        "n_excluded_missing": n_excluded_missing,
    }


def test_each_criterion_counts_the_stays_it_excluded_in_order() -> None:
    patient_filter = _filter_with(_stays())

    selected = patient_filter.filter(age_min=18, first_icu_stay=True, los_min=24)

    assert selected == [1, 8, 9]
    assert patient_filter.last_selection_steps() == (
        _step("age", 9, 3, 6, 1, age_min=18),
        _step("first_icu_stay", 6, 1, 5, 0, first_icu_stay=True),
        _step("los", 5, 2, 3, 1, los_min=24),
    )


def test_the_steps_add_up_to_the_stays_the_filter_dropped() -> None:
    frame = _stays()
    patient_filter = _filter_with(frame)

    result = patient_filter.filter(
        age_min=18,
        age_max=75,
        first_icu_stay=True,
        los_min=24,
        los_max=99,
        gender="M",
        survived=True,
        return_dataframe=True,
    )
    steps = patient_filter.last_selection_steps()

    assert [step["criterion"] for step in steps] == [
        "age",
        "first_icu_stay",
        "los",
        "gender",
        "survived",
    ]
    assert steps[0]["n_before"] == len(frame)
    for earlier, later in zip(steps, steps[1:]):
        assert later["n_before"] == earlier["n_remaining"]
    for step in steps:
        assert step["n_excluded"] == step["n_before"] - step["n_remaining"]
    assert steps[-1]["n_remaining"] == len(result)
    assert sum(step["n_excluded"] for step in steps) == len(frame) - len(result)
    # The selection itself is the one mask it always was.
    expected = frame[
        (frame["age"] >= 18)
        & (frame["age"] <= 75)
        & (frame["first_icu_stay"] == True)  # noqa: E712 - the filter's own test
        & (frame["los_hours"] >= 24)
        & (frame["los_hours"] <= 99)
        & (frame["gender"].str.upper() == "M")
        & (frame["survived"] == True)  # noqa: E712
    ]
    assert result["patient_id"].tolist() == expected["patient_id"].tolist() == [1]


def test_a_published_age_interval_counts_a_stay_without_one_as_missing() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4],
            "age": [None, 20.0, 40.0, None],
            "age_lower": pd.array([None, 20.0, 40.0, 90.0], dtype="Float64"),
            "age_upper": pd.array([None, 20.0, 40.0, None], dtype="Float64"),
        }
    )
    patient_filter = _filter_with(frame)

    assert patient_filter.filter(age_min=18) == [2, 3, 4]
    assert patient_filter.last_selection_steps() == (
        _step("age", 4, 1, 3, 1, age_min=18),
    )

    # Stay 4 is 90 or over: excluded by its stated age, not for want of one.
    assert patient_filter.filter(age_min=18, age_max=65) == [2, 3]
    assert patient_filter.last_selection_steps() == (
        _step("age", 4, 2, 2, 1, age_min=18, age_max=65),
    )


def test_a_missing_first_stay_flag_is_excluded_and_counted_as_missing() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "first_icu_stay": pd.array([True, None, False], dtype="boolean"),
        }
    )
    patient_filter = _filter_with(frame)

    assert patient_filter.filter(first_icu_stay=True) == [1]
    assert patient_filter.last_selection_steps() == (
        _step("first_icu_stay", 3, 2, 1, 1, first_icu_stay=True),
    )


def test_a_stay_already_excluded_is_not_counted_as_missing_again() -> None:
    # Stay 1 is a child without an ICU length of stay: the age bound excludes it.
    frame = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "age": [12.0, 40.0, 50.0],
            "los_hours": [None, 30.0, None],
        }
    )
    patient_filter = _filter_with(frame)

    assert patient_filter.filter(age_min=18, los_min=24) == [2]
    assert patient_filter.last_selection_steps() == (
        _step("age", 3, 1, 2, 0, age_min=18),
        _step("los", 2, 1, 1, 1, los_min=24),
    )


def test_a_stay_without_a_sepsis_status_is_counted_as_missing(monkeypatch) -> None:
    patient_filter = _filter_with(pd.DataFrame({"patient_id": [1, 2, 3, 4]}))
    monkeypatch.setattr(patient_filter, "_get_sepsis_status_ids", lambda: ({1}, {2, 3}))

    assert patient_filter.filter(has_sepsis=False) == [2, 3]
    assert patient_filter.last_selection_steps() == (
        _step("has_sepsis", 4, 2, 2, 1, has_sepsis=False),
    )
    # Either way, only stay 4 lacks a status; stays 2 and 3 are known negative.
    assert patient_filter.filter(has_sepsis=True) == [1]
    assert patient_filter.last_selection_steps() == (
        _step("has_sepsis", 4, 3, 1, 1, has_sepsis=True),
    )


def test_no_step_without_a_criterion_and_none_left_from_a_failed_filter() -> None:
    patient_filter = _filter_with(_stays())

    assert patient_filter.last_selection_steps() == ()
    assert patient_filter.filter() == list(range(1, 10))
    assert patient_filter.last_selection_steps() == ()

    patient_filter.filter(age_min=18)
    assert len(patient_filter.last_selection_steps()) == 1
    patient_filter._demographics = _stays().drop(columns=["los_hours"])
    with pytest.raises(PatientFilterCriterionError):
        patient_filter.filter(los_min=24)
    assert patient_filter.last_selection_steps() == ()


def test_the_steps_returned_are_copies() -> None:
    patient_filter = _filter_with(_stays())
    patient_filter.filter(age_min=18)

    returned = patient_filter.last_selection_steps()
    returned[0]["n_excluded"] = 0
    returned[0]["parameters"]["age_min"] = 0

    assert patient_filter.last_selection_steps() == (
        _step("age", 9, 3, 6, 1, age_min=18),
    )
