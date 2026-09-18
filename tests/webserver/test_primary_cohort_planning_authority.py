"""Unselected source scope is not an approved all-row analysis population."""

import pytest

from easyicu.webserver import primary_cohort


@pytest.mark.parametrize(
    ("cohort", "expected"),
    [
        ({}, None),
        (None, None),
        ({"label": "Sepsis ICU patients", "review": "Adults only"}, None),
        ({"max_patients": 1000}, None),
        ({"exclude_readmissions": False}, None),
        ({"preset": "all_icu"}, "all_input_rows"),
        ({"preset": "all_icu", "age_min": 18}, "predicate_filtered"),
        ({"age_min": 18}, "predicate_filtered"),
        ({"preset": "sepsis3"}, "predicate_filtered"),
        ({"include_diagnoses": ["I50"]}, "predicate_filtered"),
        ({"exclude_readmissions": True}, "predicate_filtered"),
    ],
)
def test_candidate_population_requirement_preserves_unresolved_scope(
    cohort: object, expected: str | None,
) -> None:
    assert primary_cohort.planning_selection_mode(cohort) == expected
    # Existing execution normalization remains deterministic and unchanged.
    assert primary_cohort.normalize_primary_cohort_scope(cohort).selection_mode in {
        "all_input_rows", "predicate_filtered",
    }


def test_candidate_population_does_not_hide_invalid_explicit_scope() -> None:
    with pytest.raises(primary_cohort.PrimaryCohortContractError):
        primary_cohort.planning_selection_mode({"preset": "unregistered"})
