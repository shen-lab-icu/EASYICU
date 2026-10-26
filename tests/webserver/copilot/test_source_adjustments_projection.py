"""The approval card receives a bounded copy of the database capability review."""

from easyicu.webserver.pi_copilot.source_adjustments import (
    current_source_adjustments,
    project_source_adjustments,
)


def _review(decision, unapplied=()):
    return {"decision": decision, "unapplied_substitutions": list(unapplied)}


def test_an_exact_or_absent_review_projects_nothing():
    assert project_source_adjustments(None) is None
    assert project_source_adjustments(_review({"status": "exact", "database": "miiv"})) is None
    assert current_source_adjustments({}) is None
    assert current_source_adjustments({"scientific_plan_review": {"facts": {}}}) is None


def test_a_substitution_keeps_terms_reasons_and_fields_only():
    projected = current_source_adjustments({"scientific_plan_review": {"facts": {"database_capability_review": _review({
        "status": "substitution", "database": "aumc",
        "requirements": {"family": "association", "secret": "not projected"},
        "substitutions": [
            {"capability": "threshold:age", "original": {"concept": "age", "operator": ">=", "value": 65},
             "substitute": {"concept": "age", "operator": ">=", "value": 60},
             "reason_code": "numeric_threshold_aligned_to_band",
             "source_field": {"source": "/abs/path/data-sources.json", "table": "admissions", "column": "agegroup"}},
            {"capability": "threshold:age", "original": {"concept": "age", "operator": "~", "value": 1},
             "substitute": {"concept": "age", "operator": ">=", "value": float("nan")}, "reason_code": "x"},
        ],
        "missing": [],
    }, ["threshold:age"])}}})
    assert projected == {
        "decision": {"status": "substitution", "database": "aumc", "substitutions": [{
            "capability": "threshold:age",
            "original": {"concept": "age", "operator": ">=", "value": 65},
            "substitute": {"concept": "age", "operator": ">=", "value": 60},
            "reason_code": "numeric_threshold_aligned_to_band",
            "source_field": {"table": "admissions", "column": "agegroup"},
        }], "missing": []},
        "unapplied_substitutions": ["threshold:age"],
    }


def test_a_refusal_keeps_its_codes_and_a_substitution_without_rows_is_dropped():
    refusal = project_source_adjustments(_review({
        "status": "refusal", "database": "hirid", "substitutions": [],
        "missing": [{"capability": "hospital_los", "code": "hospital_los_unavailable", "source": "x"}, {"code": ""}],
    }))
    assert refusal["decision"]["missing"] == [{"capability": "hospital_los", "code": "hospital_los_unavailable"}]
    assert project_source_adjustments(_review({"status": "substitution", "database": "sic", "substitutions": [{}]})) is None
