"""Planner literature keys must be exact members of the pre-plan bundle."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.core import PlannerAgent, ReplannerAgent
from easyicu.research_agent.planning.replan_gate import (
    replan_candidate_contract_findings,
)
from easyicu.research_agent.providers.mocks import MockLLMClient, ScriptedMockLLMClient
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Does an aggregate ICU exposure predict mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=10,
            n_stays=10,
        ),
        variables=[],
    )


def _raw(keys: list[str]) -> str:
    bindings = [
        {
            "citation_key": key,
            "design_elements": ["reporting"],
            "application": "Apply this source prospectively to the reporting design.",
        }
        for key in keys
    ]
    return json.dumps(
        {
            "research_question": _context().research_question,
            "steps": [
                {
                    "step_id": "primary",
                    "planned_analysis_role": "primary",
                    "intent": "Estimate the prespecified primary association.",
                    "inputs": [],
                    "expected_outputs": ["table:primary_summary"],
                    "method": "descriptive",
                    "icu_rule_refs": [],
                    "literature_citation_keys": keys,
                    "literature_design_bindings": bindings,
                }
            ],
        }
    )


def test_planner_accepts_exact_preplan_literature_key() -> None:
    plan = PlannerAgent.__new__(PlannerAgent)._parse(
        _raw(["strobe_2007"]),
        _context(),
        allowed_literature_citation_keys=["strobe_2007"],
    )

    assert plan.steps[0].literature_citation_keys == ["strobe_2007"]


def test_method_source_cannot_claim_a_design_element_outside_its_card() -> None:
    payload = json.loads(_raw(["strobe_2007"]))
    payload["steps"][0]["literature_design_bindings"][0][
        "design_elements"
    ] = ["adjustment"]

    with pytest.raises(ValueError, match="do not support"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            json.dumps(payload),
            _context(),
            allowed_literature_citation_keys=["strobe_2007"],
        )


def test_strobe_dependence_card_is_not_confused_with_reporting_card() -> None:
    payload = json.loads(_raw(["strobe_2007"]))
    payload["steps"][0]["literature_design_bindings"][0][
        "design_elements"
    ] = ["reporting", "dependence"]

    plan = PlannerAgent.__new__(PlannerAgent)._parse(
        json.dumps(payload),
        _context(),
        allowed_literature_citation_keys=["strobe_2007"],
    )

    assert plan.steps[0].literature_design_bindings[0].design_elements == [
        "reporting",
        "dependence",
    ]


def test_planner_rejects_unknown_literature_key() -> None:
    with pytest.raises(ValueError, match="outside this run's pre-plan"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            _raw(["invented_paper_2099"]),
            _context(),
            allowed_literature_citation_keys=["strobe_2007"],
        )


def test_scientific_plan_cannot_ignore_available_literature_bundle() -> None:
    with pytest.raises(ValueError, match="must bind an exact key"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            _raw([]),
            _context(),
            allowed_literature_citation_keys=["strobe_2007"],
        )


def test_citation_key_without_typed_design_application_is_rejected() -> None:
    payload = json.loads(_raw(["strobe_2007"]))
    payload["steps"][0]["literature_design_bindings"] = []

    with pytest.raises(
        ValueError, match="missing or incomplete literature_design_bindings"
    ):
        PlannerAgent.__new__(PlannerAgent)._parse(
            json.dumps(payload),
            _context(),
            allowed_literature_citation_keys=["strobe_2007"],
        )


def test_design_binding_cannot_name_an_unbound_source() -> None:
    payload = json.loads(_raw(["strobe_2007"]))
    payload["steps"][0]["literature_design_bindings"][0]["citation_key"] = "other"

    with pytest.raises(ValueError, match="must also appear"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            json.dumps(payload),
            _context(),
            allowed_literature_citation_keys=["strobe_2007", "other"],
        )


def test_every_cited_source_needs_an_explicit_design_contribution() -> None:
    payload = json.loads(_raw(["singer_sepsis3_2016", "strobe_2007"]))
    payload["steps"][0]["literature_design_bindings"] = payload["steps"][0][
        "literature_design_bindings"
    ][:1]

    with pytest.raises(ValueError, match="unexplained: strobe_2007"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            json.dumps(payload),
            _context(),
            allowed_literature_citation_keys=[
                "singer_sepsis3_2016",
                "strobe_2007",
            ],
        )


def test_each_scientific_step_must_bind_a_preplan_source() -> None:
    payload = json.loads(_raw(["strobe_2007"]))
    payload["steps"].append(
        {
            **payload["steps"][0],
            "step_id": "sensitivity",
            "planned_analysis_role": "sensitivity",
            "literature_citation_keys": [],
            "literature_design_bindings": [],
        }
    )

    with pytest.raises(ValueError, match="unbound steps: sensitivity"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            json.dumps(payload),
            _context(),
            allowed_literature_citation_keys=["strobe_2007"],
        )


def test_topic_paper_alone_cannot_govern_a_scientific_method() -> None:
    with pytest.raises(ValueError, match="must bind at least one method-source"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            _raw(["singer_sepsis3_2016"]),
            _context(),
            allowed_literature_citation_keys=[
                "singer_sepsis3_2016",
                "strobe_2007",
            ],
        )


def test_topic_and_method_sources_can_jointly_bind_one_step() -> None:
    plan = PlannerAgent.__new__(PlannerAgent)._parse(
        _raw(["singer_sepsis3_2016", "strobe_2007"]),
        _context(),
        allowed_literature_citation_keys=[
            "singer_sepsis3_2016",
            "strobe_2007",
        ],
    )

    assert plan.steps[0].literature_citation_keys == [
        "singer_sepsis3_2016",
        "strobe_2007",
    ]


def test_planner_receives_exact_preplan_literature_authority() -> None:
    llm = ScriptedMockLLMClient(
        [_raw(["invented_internal_label"]), _raw(["strobe_2007"])]
    )
    plan = PlannerAgent(llm).run(
        _context(),
        allowed_literature_citation_keys=[
            "strobe_2007",
            "strobe_2007",
            "record_2015",
        ],
    )

    assert plan.steps[0].literature_citation_keys == ["strobe_2007"]
    prompt = "\n".join(message.content for message in llm.messages)
    assert "PRE-PLAN LITERATURE CITATION AUTHORITY (exact, run-bound)" in prompt
    assert '["strobe_2007", "record_2015"]' in prompt
    assert "Do not cite an evidence artifact" in prompt
    assert "method_decision_cards" in prompt
    assert "What must this study report for a reader to appraise it?" in prompt
    assert "Allowed literature_citation_keys for this run are exactly" in prompt
    assert "Method-source bindings may use ONLY the design elements" in prompt
    assert '"record_2015": ["reporting"]' in prompt
    assert (
        '"strobe_2007": ["dependence", "estimand", "outcome", "reporting"]'
        in prompt
    )
    assert len(llm.calls) == 2


def test_primary_plan_must_bind_a_screened_direct_comparator() -> None:
    allowed = ["paper_direct", "strobe_2007"]
    with pytest.raises(ValueError, match="does not bind any screened direct"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            _raw(["strobe_2007"]),
            _context(),
            allowed_literature_citation_keys=allowed,
            direct_comparator_literature_keys=["paper_direct"],
        )

    plan = PlannerAgent.__new__(PlannerAgent)._parse(
        _raw(["paper_direct", "strobe_2007"]),
        _context(),
        allowed_literature_citation_keys=allowed,
        direct_comparator_literature_keys=["paper_direct"],
    )
    assert plan.steps[0].literature_citation_keys == allowed


def test_builtin_mock_obeys_preplan_literature_authority() -> None:
    context = _context()
    plan = PlannerAgent(MockLLMClient(context=context)).run(
        context,
        allowed_literature_citation_keys=["strobe_2007"],
    )

    scientific_steps = [
        step
        for step in plan.steps
        if step.planned_analysis_role in {"primary", "secondary", "sensitivity"}
    ]
    assert scientific_steps
    assert all(
        step.literature_citation_keys == ["strobe_2007"]
        for step in scientific_steps
    )


def test_replanner_retries_when_revision_drops_literature_binding() -> None:
    current = PlannerAgent.__new__(PlannerAgent)._parse(
        _raw(["strobe_2007"]),
        _context(),
        allowed_literature_citation_keys=["strobe_2007"],
    )
    llm = ScriptedMockLLMClient([_raw([]), _raw(["strobe_2007"])])

    revised = ReplannerAgent(llm).run(
        context=_context(),
        current_plan=current,
        allowed_literature_citation_keys=["strobe_2007"],
    )

    assert revised.steps[0].literature_citation_keys == ["strobe_2007"]
    assert len(llm.calls) == 2
    prompt = "\n".join(message.content for message in llm.messages)
    assert "PRE-PLAN LITERATURE CITATION AUTHORITY" in prompt


def test_replan_gate_rejects_unbound_scientific_step() -> None:
    plan = PlannerAgent.__new__(PlannerAgent)._parse(_raw([]), _context())

    findings = replan_candidate_contract_findings(
        plan=plan,
        context=_context(),
        allowed_literature_citation_keys=["strobe_2007"],
    )

    assert any(
        finding.severity == "error"
        and finding.validator == "replanner_literature_authority"
        for finding in findings
    )
