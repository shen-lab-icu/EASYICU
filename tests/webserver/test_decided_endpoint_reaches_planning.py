"""An endpoint the researcher decides reaches the candidate plan's review.

The plan review closes its endpoint question only when the planning context
carries a typed outcome with an endpoint contract.  The answer used to stay a
reader label, candidate planning read the endpoint from the question text
alone, and the metadata-only catalog gave derived fixed-horizon mortality no
event role, so no answer could close the question.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog
from easyicu.webserver import research_launch_scientific
from easyicu.webserver.pi_copilot import plan_decisions
from easyicu.webserver.study_scientific_configuration import ScientificConfiguration

_ENDPOINT = "OUTCOME_DEFINITION_UNRESOLVED"
_QUESTION = "Do early lactate trajectory classes differ in outcome?"
_WEBSERVER = Path(research_launch_scientific.__file__).parent


def _study(**fields) -> dict:
    return {
        "id": "study-endpoint-choice", "question": _QUESTION,
        "data_source": {"database": "miiv"}, "execution_concepts": {}, **fields,
    }


def test_derived_event_outcomes_carry_the_owner_event_role():
    roles = {item.concept_id: item.column_role for item in build_database_capability_catalog("miiv").concepts}

    assert roles["mort_28d"] == roles["mort_90d"] == roles["death"] == "event_status"
    assert roles["followup_days_28d"] == ""
    assert roles["los_icu"] == ""


@pytest.mark.parametrize(
    ("question", "configured", "target", "endpoint"),
    [
        (_QUESTION, "mort_90d", "mort_90d", "mort_90d"),
        (_QUESTION, "los_icu", "los_icu", None),
        ("Do early lactate trajectory classes differ in in-hospital mortality?", "mort_90d", "mort_90d", "mort_90d"),
        ("Do early lactate trajectory classes differ in in-hospital mortality?", None, "death", "death"),
        (_QUESTION, None, None, None),
    ],
)
def test_a_configured_outcome_becomes_the_planning_endpoint(question, configured, target, endpoint):
    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database="miiv", configured_outcome=configured,
    )

    assert coordinates["target_outcome"] == target
    assert (coordinates["endpoint"].name if coordinates["endpoint"] else None) == endpoint
    if endpoint:
        assert coordinates["endpoint"].kind == "binary"
    assert coordinates["execution_authorized"] is False


@pytest.mark.parametrize("module", ["research_pipeline_run_preparation.py", "agent_pipeline_runs.py"])
def test_every_planning_caller_passes_the_configured_outcome(module):
    tree = ast.parse((_WEBSERVER / module).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_metadata_only_planning_coordinates"
    ]

    assert calls
    assert all("configured_outcome" in {kw.arg for kw in call.keywords} for call in calls)


def test_the_endpoint_question_offers_the_bindable_outcomes():
    context = plan_decisions.plan_decision_context({}, _ENDPOINT, _study())
    concepts = [row["concept"] for row in context["endpoint_options"]]

    assert {"death", "mort_28d", "mort_90d"} <= set(concepts)
    assert not {"los_icu", "followup_days_28d"} & set(concepts)
    assert plan_decisions.plan_decision_context({}, _ENDPOINT, _study(data_source={})) == {}


def test_choosing_an_endpoint_saves_the_typed_concept_and_replans():
    study = _study(execution_concepts={"primary_exposure": "lact"})
    compiled = plan_decisions.compile_plan_decision(
        decision_code=_ENDPOINT, option_id="mort_90d", study=study, agent_plan={},
    )

    assert compiled.patch["execution_concepts"] == {"primary_exposure": "lact", "outcome": "mort_90d"}
    assert "mort_90d" in compiled.patch["outcome"]
    assert compiled.next_action == "replan"
    assert not ScientificConfiguration.inspect(study).decision_is_resolved(_ENDPOINT)
    assert ScientificConfiguration.inspect({**study, **compiled.patch}).decision_is_resolved(_ENDPOINT)


def test_an_outcome_without_an_endpoint_contract_is_refused():
    with pytest.raises(plan_decisions.PlanDecisionError) as caught:
        plan_decisions.compile_plan_decision(
            decision_code=_ENDPOINT, option_id="los_icu", study=_study(), agent_plan={},
        )

    assert caught.value.code == "plan_decision_option_unknown"
