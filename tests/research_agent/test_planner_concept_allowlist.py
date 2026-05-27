"""Canaries for planner CTAS concept-id grounding.

The Phase-1 real-LLM pilot failed in plan generation when the planner
invented ``sofa2_admission`` as a concept_id. These tests pin the prompt
contract that prevents that failure mode from coming back.
"""

from __future__ import annotations

from easyicu.research_agent import schema
from easyicu.research_agent.agents import _build_planner_user_prompt


def _context() -> schema.ResearchContext:
    return schema.ResearchContext(
        research_question="Estimate admission SOFA association with ICU mortality.",
        cohort=schema.CohortDescriptor(
            cohort_name="case_b",
            database="miiv",
            n_patients=200,
            n_stays=200,
        ),
        variables=[
            schema.ConceptDescriptor(
                name="sofa",
                role=schema.VariableRole.COMPOSITE_SCORE,
                dtype="float64",
                is_ordinal=True,
            ),
            schema.ConceptDescriptor(
                name="death",
                role=schema.VariableRole.OUTCOME,
                dtype="int64",
            ),
        ],
        target_outcome="death",
    )


def test_planner_prompt_contains_concept_allowlist() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert "ALLOWED concept_ids" in prompt
    assert prompt.index("ALLOWED concept_ids") < prompt.index(
        "Every cohort/exposure/outcome concept"
    )
    listed_ids = [
        line.strip()[3:-1]
        for line in prompt.splitlines()
        if line.strip().startswith("- `") and line.strip().endswith("`")
    ]
    assert len(listed_ids) >= 50
    assert "sofa" in listed_ids
    assert "death" in listed_ids


def test_planner_prompt_forbids_concept_id_synthesis() -> None:
    prompt = _build_planner_user_prompt(_context())

    assert "Synthesizing new names" in prompt
    assert '"sofa2_admission"' in prompt
    assert '"kdigo_aki_max"' in prompt
    assert '"sepsis_onset_window"' in prompt


def test_planner_prompt_has_non_null_cohort_override_example() -> None:
    prompt = _build_planner_user_prompt(_context())

    cohort_start = prompt.index('"spec_id": "alt_cohort_max_during_stay"')
    missing_start = prompt.index('"spec_id": "alt_missing_complete_case"')
    cohort_example = prompt[cohort_start:missing_start]
    assert '"axis": "cohort"' in cohort_example
    assert '"cohort_override": {' in cohort_example
    assert '"concept_id": "sofa"' in cohort_example
    assert '"cohort_override": null' not in cohort_example
