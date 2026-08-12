"""Planner literature keys must be exact members of the pre-plan bundle."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.core import PlannerAgent
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


def test_each_scientific_step_must_bind_a_preplan_source() -> None:
    payload = json.loads(_raw(["strobe_2007"]))
    payload["steps"].append(
        {
            **payload["steps"][0],
            "step_id": "sensitivity",
            "planned_analysis_role": "sensitivity",
            "literature_citation_keys": [],
        }
    )

    with pytest.raises(ValueError, match="unbound steps: sensitivity"):
        PlannerAgent.__new__(PlannerAgent)._parse(
            json.dumps(payload),
            _context(),
            allowed_literature_citation_keys=["strobe_2007"],
        )


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
    assert "Allowed literature_citation_keys for this run are exactly" in prompt
    assert len(llm.calls) == 2


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
