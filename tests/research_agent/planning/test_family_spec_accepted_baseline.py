"""A family template keeps the accepted baseline roster in its own Table 1.

After the Host compiles a reviewed candidate's runtime coordinates, the next
plan must keep that candidate's Table 1 content
(``accepted_baseline_requirements``).  The family templates described only
their model roster, so a row the candidate showed beside it (a severity
score, the outcome by exposure level) was dropped and the plan failed the
outline's baseline gate after the Provider call
(``progressive_outline_accepted_baseline_incomplete``).  The 9/25 E1 run
stopped this way.  The fixtures here are generic variables, not that study.
"""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.planning.baseline_requirements import (
    baseline_requirement_coverage,
    bind_baseline_requirements,
)
from easyicu.research_agent.planning.family_spec import FamilySpecError
from easyicu.research_agent.planning.family_spec.contract import table_one_group_column
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

from .family_spec_fixtures import (
    _context,
    _descriptive_context,
    _prediction_context,
    _prediction_payload,
    _request,
    _run,
    _spec_payload,
)

_LANDMARK_ROWS = ("age", "sex", "severity_score_24h", "death")


def _accepting(context, *, group_by, rows):
    return bind_baseline_requirements(
        context,
        {
            # An ungrouped summary roster needs the second schema version.
            "schema_version": "easyicu.accepted_baseline_requirements/"
            + ("2" if group_by is None else "1"),
            "source_plan_sha256": "c" * 64,
            "tables": [
                {
                    "source_step_id": "baseline_context",
                    "group_by": None if group_by is None else {"name": group_by, "source_concept": None},
                    "variables": [
                        {"name": row, "source_concept": None}
                        if isinstance(row, str)
                        else {"name": row[0], "source_concept": row[1]}
                        for row in rows
                    ],
                }
            ],
        },
    )


def _table_one(plan):
    return next(step.table_one_spec for step in plan.steps if step.table_one_spec is not None)


def test_the_landmark_family_keeps_rows_the_candidate_showed_beside_its_model_roster() -> None:
    context = _accepting(_context(exact=True), group_by="injury_stage", rows=_LANDMARK_ROWS)
    request = _request(context)

    assert [(row.required, row.columns, row.summary) for row in request.accepted_baseline_rows] == [
        ("age", ["age"], "both"),
        ("sex", ["sex"], "count_percent"),
        ("severity_score_24h", ["severity_score_24h"], "both"),
        ("death", ["death"], "count_percent"),
    ]
    llm, result = _run(context, [json.dumps(_spec_payload(request))])

    assert len(llm.calls) == 1
    table = _table_one(result.output)
    assert table.group_by == "injury_stage" == table_one_group_column(request)
    # The model roster keeps its own rows; only the rows it lacked are added.
    assert [(item.name, item.summary) for item in table.variables] == [
        ("age", "both"),
        ("sex", "count_percent"),
        ("comorbidity_index", "both"),
        ("severity_score_24h", "both"),
        ("death", "count_percent"),
    ]
    assert baseline_requirement_coverage(context, result.output)["status"] == "complete"
    assert result.output.steps[3].model_requirements[0].covariates == [
        "age", "sex", "comorbidity_index",
    ]


def test_rows_the_model_roster_already_describes_add_nothing() -> None:
    context = _accepting(
        _context(exact=True), group_by="injury_stage", rows=("age", "comorbidity_index")
    )
    request = _request(context)
    _llm, result = _run(context, [json.dumps(_spec_payload(request))])

    assert [item.name for item in _table_one(result.output).variables] == [
        "age", "sex", "comorbidity_index",
    ]


def test_the_named_column_describes_its_row_before_a_companion() -> None:
    # The owner lists a concept's value columns in name order, so the
    # companion comes first unless the row's own name is preferred.
    base = _context(exact=True)
    companion = ConceptDescriptor(
        name="severity_score_12h", description="severity score at 12 h",
        role=VariableRole.COMPOSITE_SCORE, dtype="float64", source_concept="severity_score",
    )
    variables = [*base.variables, companion]
    context = _accepting(
        base.model_copy(update={"variables": variables}),
        group_by="injury_stage",
        rows=(("severity_score_24h", "severity_score"),),
    )
    request = _request(context)

    assert [row.columns for row in request.accepted_baseline_rows] == [
        ["severity_score_24h", "severity_score_12h"]
    ]
    _llm, result = _run(context, [json.dumps(_spec_payload(request))])
    names = [item.name for item in _table_one(result.output).variables]
    assert "severity_score_24h" in names and "severity_score_12h" not in names


def test_an_ungrouped_accepted_summary_is_kept_in_table_one() -> None:
    context = _accepting(_context(exact=True), group_by=None, rows=("severity_score_24h",))
    request = _request(context)
    _llm, result = _run(context, [json.dumps(_spec_payload(request))])

    assert "severity_score_24h" in [item.name for item in _table_one(result.output).variables]
    assert baseline_requirement_coverage(context, result.output)["status"] == "complete"


def test_the_prediction_family_keeps_accepted_rows_in_its_outcome_grouped_table() -> None:
    context = _accepting(_prediction_context(), group_by="death", rows=("age", "sex", "score_first"))
    request = _request(context, cohort_mode=None)
    features = ["hr_max", "lactate_max", "age"]
    _llm, result = _run(
        context,
        [json.dumps(_prediction_payload(request, features=features))],
        required_primary_cohort_selection_mode=None,
    )

    table = _table_one(result.output)
    assert table.group_by == "death" == table_one_group_column(request)
    assert [item.name for item in table.variables] == [*features, "sex", "score_first"]
    assert baseline_requirement_coverage(context, result.output)["status"] == "complete"


_DESCRIPTIVE_LABELS = {
    "phenotype_flag": "Phenotype present in the first 24 h",
    "phenotype_flag=0": "Phenotype absent",
    "phenotype_flag=1": "Phenotype present",
    "death": "In-hospital death",
    "age": "Age at ICU admission (years)",
    "sex": "Patient sex",
    "score_first": "Chronic disease score (first value)",
    "readmit_flag": "ICU readmission indicator",
}


def _descriptive_payload(request, *, baseline_variables):
    return {
        "schema_version": "easyicu.family_plan_spec/1",
        "family_id": request.family_id,
        "request_sha256": request.request_sha256,
        "adjustment_set": [],
        "baseline_variables": baseline_variables,
        "reader_display_labels": [
            {"key": key, "value": _DESCRIPTIVE_LABELS[key]}
            for key in [*request.required_reader_label_keys, *request.level_label_keys]
        ],
        "comparator_applications": [
            {
                "citation_key": key,
                "application": (
                    f"Compare this description with {key} on population, exposure definition, "
                    "time zero, and estimand without copying its design or claiming novelty."
                ),
            }
            for key in request.direct_comparator_literature_keys
        ],
        "roster_decision_note": "Descriptive family: baseline variables chosen from host-timed candidates.",
    }


def test_the_descriptive_family_keeps_accepted_rows_beside_the_planner_choice() -> None:
    context = _accepting(
        _descriptive_context(), group_by="phenotype_flag", rows=("age", "score_first", "death")
    )
    request = _request(context, cohort_mode="all_input_rows")
    _llm, result = _run(
        context,
        [json.dumps(_descriptive_payload(request, baseline_variables=["age", "sex"]))],
        required_primary_cohort_selection_mode="all_input_rows",
    )

    table = _table_one(result.output)
    assert table.group_by == "phenotype_flag" == table_one_group_column(request)
    assert [item.name for item in table.variables] == ["age", "sex", "score_first", "death"]
    assert baseline_requirement_coverage(context, result.output)["status"] == "complete"


def test_a_roster_grouped_by_another_column_is_refused_before_the_provider() -> None:
    context = _accepting(_context(exact=True), group_by="death", rows=("age",))

    with pytest.raises(FamilySpecError) as caught:
        _request(context)
    assert caught.value.reason_code == "family_spec_accepted_baseline_grouping_unsupported"
    # The scripted client holds no answer, so any Provider call would fail
    # differently: the planner refuses while sealing the request, as a typed
    # planning stop.
    with pytest.raises(ProgressivePlanCompileError) as planned:
        _run(context, [])
    assert planned.value.reason_code == "progressive_family_spec_accepted_baseline_grouping_unsupported"
    assert planned.value.__cause__.reason_code == "family_spec_accepted_baseline_grouping_unsupported"


def test_a_row_with_no_prepared_column_is_refused_before_the_provider() -> None:
    context = _accepting(_context(exact=True), group_by="injury_stage", rows=("albumin_min",))

    with pytest.raises(FamilySpecError) as caught:
        _request(context)
    assert caught.value.reason_code == "family_spec_accepted_baseline_row_unavailable"
    assert "albumin_min" in str(caught.value)


def test_the_accepted_roster_is_part_of_the_request_digest() -> None:
    plain = _request(_context(exact=True))
    accepting = _request(
        _accepting(_context(exact=True), group_by="injury_stage", rows=_LANDMARK_ROWS)
    )
    payload = plain.model_dump(mode="json")

    assert plain.accepted_baseline_rows == []
    # A request without an accepted roster keeps the digest it had before
    # the field existed; one with a roster binds it.
    payload.pop("accepted_feature_groups")
    payload.pop("accepted_baseline_rows")
    assert plain.request_sha256 == canonical_sha256(payload)
    assert accepting.request_sha256 != plain.request_sha256
