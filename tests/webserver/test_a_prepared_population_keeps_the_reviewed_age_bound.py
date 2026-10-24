"""A real export proof may satisfy a candidate predicate without changing rows."""
from copy import deepcopy

import pytest

from easyicu.webserver.research_launch_population import prepared_candidate_selection_mode


def definition():
    return {"name": "reviewed", "selection_mode": "predicate_filtered", "inclusion": [{
        "concept_id": "age", "time_window": {"anchor": "icu_admission",
        "start_offset_hours": 0, "end_offset_hours": "inf"},
        "aggregation": "first", "op": ">=", "value": 18,
    }], "exclusion": [], "derived_from_named": None, "locked_at": "not_locked"}


def report():
    return {"source_total": 12, "selected": 10, "mode": "all_icu",
            "applied_filters": ["demographics"], "count_unit": "icu_stay",
            "demographic_steps": [{"criterion": "age", "n_before": 12,
                "n_excluded": 2, "n_remaining": 10, "n_excluded_missing": 1,
                "parameters": {"age_min": 18}}]}


def mode(candidate=None, source=None, *, basis="export_contract", required="predicate_filtered"):
    return prepared_candidate_selection_mode(
        candidate_definition=definition() if candidate is None else candidate,
        required_mode=required, source_basis=basis,
        source_report=report() if source is None else source, database="eicu_demo",
    )


def test_an_executed_age_bound_keeps_the_same_population_without_a_second_filter():
    candidate, source = definition(), report()
    before_candidate, before_source = deepcopy(candidate), deepcopy(source)
    assert mode(candidate, source) == "all_input_rows"
    assert candidate == before_candidate and source == before_source


@pytest.mark.parametrize("basis", [None, "package_declaration", "question", ""])
def test_unverified_source_declarations_cannot_relax_the_candidate(basis):
    assert mode(basis=basis) == "predicate_filtered"


@pytest.mark.parametrize("change", ["missing_step", "weaker_bound", "missing_counts", "wrong_filter"])
def test_the_export_must_prove_the_exact_requested_range(change):
    source = report()
    if change == "missing_step":
        source["demographic_steps"] = []
    elif change == "weaker_bound":
        source["demographic_steps"][0]["parameters"]["age_min"] = 17
    elif change == "missing_counts":
        source.pop("source_total")
    else:
        source["demographic_steps"][0]["criterion"] = "los"
    assert mode(source=source) == "predicate_filtered"


@pytest.mark.parametrize("change", ["extra_predicate", "exclusion", "different_window", "mean",
                                    "strict_bound", "upper_bound", "unapplied"])
def test_other_scientific_predicates_and_windows_keep_the_existing_guard(change):
    candidate = definition()
    predicate = candidate["inclusion"][0]
    if change == "extra_predicate":
        candidate["inclusion"].append({**deepcopy(predicate), "concept_id": "death", "value": 0})
    elif change == "exclusion":
        candidate["exclusion"] = [deepcopy(predicate)]
    elif change == "different_window":
        predicate["time_window"]["end_offset_hours"] = 24
    elif change == "mean":
        predicate["aggregation"] = "mean"
    elif change == "strict_bound":
        predicate["op"] = ">"
    elif change == "upper_bound":
        candidate["inclusion"].append({**deepcopy(predicate), "op": "<=", "value": 90})
    else:
        candidate["unapplied_population_criteria"] = ["unresolved population"]
    assert mode(candidate) == "predicate_filtered"


def test_two_proven_age_bounds_and_caller_modes_are_preserved():
    candidate, source = definition(), report()
    candidate["inclusion"].append({**deepcopy(candidate["inclusion"][0]), "op": "<=", "value": 89})
    source["demographic_steps"][0]["parameters"]["age_max"] = 89
    assert mode(candidate, source) == "all_input_rows"
    assert mode(required="all_input_rows") == "all_input_rows"
    assert prepared_candidate_selection_mode(
        candidate_definition=None, required_mode="predicate_filtered",
        source_basis="export_contract", source_report=source, database="eicu_demo",
    ) == "predicate_filtered"


@pytest.mark.parametrize("selected", [None, 0, True, "10"])
def test_an_invalid_or_empty_parent_count_does_not_establish_a_bound(selected):
    source = report()
    source["selected"] = selected
    assert mode(source=source) == "predicate_filtered"


def test_unverified_source_never_reaches_population_compilation(monkeypatch):
    from easyicu.research_agent.planning import population_compile

    def forbidden(*args, **kwargs):
        raise AssertionError("unverified source reached the compiler")

    monkeypatch.setattr(population_compile, "compile_population", forbidden)
    assert mode(basis="package_declaration") == "predicate_filtered"


@pytest.mark.parametrize("disposition", ["applied_by_plan", "requires_extraction"])
def test_a_nonblocking_compiler_result_still_requires_source_application(monkeypatch, disposition):
    from types import SimpleNamespace
    from easyicu.research_agent.planning import population_compile

    monkeypatch.setattr(population_compile, "compile_population", lambda *args, **kwargs:
                        SimpleNamespace(blocking=False, criteria=[SimpleNamespace(disposition=disposition)]))
    assert mode() == "predicate_filtered"


def test_inherited_population_criteria_need_their_own_proof():
    candidate = definition()
    candidate["derived_from_named"] = "reviewed_named_population"
    assert mode(candidate) == "predicate_filtered"


@pytest.mark.parametrize("change", [
    "stronger_lower", "stronger_upper", "weaker_upper", "extra_filter", "extra_step",
    "wrong_mode", "patient_unit", "missing_unit", "missing_unknown_count",
    "invalid_unknown_count", "cap", "hidden_icd", "hidden_concept_filter",
])
def test_age_proof_cannot_hide_other_selection_or_change_the_count_unit(change):
    candidate, source = definition(), report()
    step = source["demographic_steps"][0]
    if change == "stronger_lower":
        step["parameters"]["age_min"] = 19
    elif change in {"stronger_upper", "weaker_upper"}:
        candidate["inclusion"].append({**deepcopy(candidate["inclusion"][0]), "op": "<=", "value": 89})
        step["parameters"]["age_max"] = 88 if change == "stronger_upper" else 90
    elif change == "extra_filter":
        source["applied_filters"].append("diagnosis")
    elif change == "extra_step":
        source["demographic_steps"].append({
            "criterion": "first_icu_stay", "parameters": {}, "n_before": 10,
            "n_remaining": 10, "n_excluded": 0, "n_excluded_missing": 0,
        })
    elif change == "wrong_mode":
        source["mode"] = "adult_first"
    elif change == "patient_unit":
        source["count_unit"] = "patient"
    elif change == "missing_unit":
        source.pop("count_unit")
    elif change == "missing_unknown_count":
        step.pop("n_excluded_missing")
    elif change == "invalid_unknown_count":
        step["n_excluded_missing"] = -1
    elif change == "cap":
        source["max_patients_applied"] = True
    elif change == "hidden_icd":
        source["icd"] = {"enabled": True}
        source["selected_before_icd"] = source["selected_before_cap"] = 10
    else:
        source["concept_matches"] = 10
    assert mode(candidate, source) == "predicate_filtered"


@pytest.mark.parametrize("field,value", [
    ("source_total", 9), ("selected", 11), ("n_before", 13),
    ("n_remaining", 11), ("n_excluded", 1), ("n_excluded_missing", 3),
])
def test_age_proof_requires_a_consistent_stay_count_chain(field, value):
    source = report()
    target = source if field in {"source_total", "selected"} else source["demographic_steps"][0]
    target[field] = value
    assert mode(source=source) == "predicate_filtered"
