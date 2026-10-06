"""A population criterion that no predicate applies is a review finding.

The Planner states each restriction the question places on whom the study
includes, with the allowed cohort concepts that express it; the host refuses a
foundation in which no predicate reads a listed criterion's concepts.  A
criterion that no allowed concept expresses is stated with none, and nothing
applies it: the plan analyses a broader population than the one it states.
That criterion was kept only in the planning record, so neither the plan nor
its review showed it.  The plan's cohort now carries it, outside the cohort's
digest, and the review reports it as a major finding that the study, not a
plan revision, resolves.  The host owns it: the Planner's transport leaves it
out, and a runtime replan keeps it.  Fixtures are generic.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from easyicu.research_agent.agents.plan_payload import planner_structured_output_request
from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.authority.plan_authority import normalize_replan_candidate
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    cohort_concept_id_scope,
    cohort_definition_sha,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    remediation_route_for_finding,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import AnalysisPlan

from tests.research_agent.planning.progressive_planner_fixtures import (
    _context,
    _foundation_payload,
    _materialization_payloads,
    _outline_payload,
)

_CODE = "POPULATION_CRITERION_NOT_APPLIED"
_LEGACY_COHORT_KEYS = {
    "name",
    "inclusion",
    "exclusion",
    "derived_from_named",
    "locked_at",
    "selection_mode",
}
_ADULTS = {
    "concept_id": "age_years",
    "anchor": "icu_admission",
    "start_offset_hours": 0,
    "end_offset_hours": 24,
    "aggregation": "first",
    "op": ">=",
    "value": {
        "mode": "number",
        "string_value": None,
        "number_value": 18,
        "boolean_value": None,
        "string_list": [],
        "number_list": [],
    },
}
_FILTERED = {
    "name": "Adults after cardiac surgery",
    "selection_mode": "predicate_filtered",
    "inclusion": [_ADULTS],
    "exclusion": [],
}
_EVERY_ROW = {
    "name": "Every input row",
    "selection_mode": "all_input_rows",
    "inclusion": [],
    "exclusion": [],
}


def _plan(cohort: dict[str, Any]) -> AnalysisPlan:
    """Plan through the progressive Planner with this foundation cohort."""

    foundation = _foundation_payload()
    foundation["foundation"]["cohort"] = cohort
    responses = [_outline_payload(), foundation, *_materialization_payloads()]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    return ProgressivePlannerAgent(llm).run(_context())


def _review(plan: AnalysisPlan):
    return build_plan_scientific_review(context=_context(), plan=plan, literature=None)


def _found(plan: AnalysisPlan) -> list:
    return [item for item in _review(plan).findings if item.code == _CODE]


def test_the_plan_states_a_criterion_that_no_allowed_concept_expresses() -> None:
    plan = _plan(
        {
            **_EVERY_ROW,
            "population_criteria": [{"criterion": "after cardiac surgery", "concept_ids": []}],
        }
    )

    assert plan.cohort is not None
    assert plan.cohort.unapplied_population_criteria == ("after cardiac surgery",)


def test_a_criterion_that_a_predicate_applies_is_not_carried() -> None:
    plan = _plan(
        {
            **_FILTERED,
            "population_criteria": [
                {"criterion": "adults", "concept_ids": ["age_years"]},
                {"criterion": "after cardiac surgery", "concept_ids": []},
            ],
        }
    )

    assert plan.cohort is not None
    assert [item.concept_id for item in plan.cohort.inclusion] == ["age_years"]
    assert plan.cohort.unapplied_population_criteria == ("after cardiac surgery",)


@pytest.mark.parametrize(
    "cohort",
    [
        _EVERY_ROW,
        {**_FILTERED, "population_criteria": [{"criterion": "adults", "concept_ids": ["age_years"]}]},
    ],
)
def test_a_plan_that_applies_every_criterion_serializes_as_before(cohort: dict) -> None:
    plan = _plan(cohort)

    assert plan.cohort is not None
    assert plan.cohort.unapplied_population_criteria == ()
    assert set(plan.model_dump(mode="json")["cohort"]) == _LEGACY_COHORT_KEYS
    assert set(plan.cohort.to_dict()) <= _LEGACY_COHORT_KEYS
    assert not _found(plan)


def test_the_criterion_is_kept_by_the_plan_and_left_out_of_the_cohort_digest() -> None:
    plan = _plan(
        {
            **_FILTERED,
            "population_criteria": [
                {"criterion": "adults", "concept_ids": ["age_years"]},
                {"criterion": "after cardiac surgery", "concept_ids": []},
            ],
        }
    )
    dumped = plan.model_dump(mode="json")

    assert dumped["cohort"]["unapplied_population_criteria"] == ["after cardiac surgery"]
    # The run's own concepts are known while its plan is read back.
    with cohort_concept_id_scope(["age_years"]):
        reread = AnalysisPlan.model_validate(json.loads(json.dumps(dumped)))
        # It selects no row: the cohort's own record and digest are those
        # of the predicates alone.
        bare = CohortDefinition.from_dict(plan.cohort.to_dict())
    assert reread.cohort is not None
    assert reread.cohort.unapplied_population_criteria == ("after cardiac surgery",)
    assert "unapplied_population_criteria" not in plan.cohort.to_dict()
    assert bare.unapplied_population_criteria == ()
    assert cohort_definition_sha(bare) == cohort_definition_sha(plan.cohort)


@pytest.mark.parametrize("value", ["after cardiac surgery", [""], ["  "], [3]])
def test_an_unreadable_criteria_record_is_refused(value: Any) -> None:
    with pytest.raises(CohortSchemaError):
        CohortDefinition.from_dict(
            {"name": "primary", "inclusion": [], "exclusion": [], "unapplied_population_criteria": value}
        )


def test_the_review_reports_each_criterion_no_predicate_applies() -> None:
    plan = _plan(
        {
            **_FILTERED,
            "population_criteria": [
                {"criterion": "adults", "concept_ids": ["age_years"]},
                {"criterion": "after cardiac surgery", "concept_ids": []},
                {"criterion": "with a first episode", "concept_ids": []},
            ],
        }
    )

    findings = _found(plan)

    assert [item.severity for item in findings] == ["major", "major"]
    assert ["'after cardiac surgery'" in item.message for item in findings] == [True, False]
    assert ["'with a first episode'" in item.message for item in findings] == [False, True]
    for item in findings:
        assert item.dimension == "icu_clinical_design"
        assert "no predicate applies it" in item.message
        assert remediation_route_for_finding(item) == "study_authority_change"
        assert item.requires_user_authorization is False
        assert "analysis_plan.json.cohort" in item.evidence_refs


def test_the_finding_does_not_ask_the_planner_to_revise() -> None:
    plan = _plan(
        {
            **_EVERY_ROW,
            "population_criteria": [{"criterion": "after cardiac surgery", "concept_ids": []}],
        }
    )

    review = _review(plan)

    [finding] = [item for item in review.findings if item.code == _CODE]
    assert remediation_route_for_finding(finding) != "agent_plan_revision"
    assert not [
        item
        for item in review.findings
        if item.severity == "blocker" and item.code == _CODE
    ]


def test_the_planner_transport_leaves_the_criteria_to_the_host() -> None:
    request = planner_structured_output_request()
    cohort = json.loads(request.schema_json)["$defs"]["CohortDefinition"]

    assert "unapplied_population_criteria" not in request.schema_json
    assert {"name", "selection_mode", "inclusion", "exclusion"} <= set(cohort["properties"])


@pytest.mark.parametrize(
    ("stated", "written"),
    [
        (("after cardiac surgery",), ()),
        (("after cardiac surgery",), ("with a first episode",)),
        ((), ("with a first episode",)),
    ],
)
def test_a_replan_keeps_the_criteria_its_plan_states(
    stated: tuple[str, ...], written: tuple[str, ...]
) -> None:
    current = _plan(
        {
            **_EVERY_ROW,
            **(
                {"population_criteria": [{"criterion": item, "concept_ids": []} for item in stated]}
                if stated
                else {}
            ),
        }
    )
    assert current.cohort is not None
    assert current.cohort.unapplied_population_criteria == stated
    candidate = current.model_copy(
        update={
            "revision": current.revision + 1,
            "cohort": replace(current.cohort, unapplied_population_criteria=written),
        }
    )

    result = normalize_replan_candidate(
        current_plan=current,
        candidate_plan=candidate,
        completed_records=[],
        context=_context(),
        max_total_steps=0,
        locked_robustness_specs=[],
    )

    # The candidate was accepted, not replaced by the current plan.
    assert result.plan.revision == candidate.revision
    assert result.plan.cohort is not None
    assert result.plan.cohort.unapplied_population_criteria == stated
