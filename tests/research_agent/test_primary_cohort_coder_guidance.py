"""Scoped Coder guidance for the canonical primary-cohort product schema."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents import (
    CoderAgent,
    _primary_analysis_cohort_output_contract,
)
from easyicu.research_agent.plan_utils import _step_contract_repair_guidance
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


class _CaptureLLM:
    name = "primary-cohort-guidance-test"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((list(messages), dict(kwargs)))
        return self.responses.pop(0)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Apply the Planner-owned eligibility definition.",
        cohort=CohortDescriptor(
            cohort_name="planned_cohort",
            database="synthetic",
            n_stays=10,
            n_patients=9,
        ),
        variables=[],
    )


def _primary_cohort_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="01_cohort",
        intent="Materialize the planned cohort and report attrition.",
        inputs=["stay_id", "registered_eligibility_concept"],
        expected_outputs=[
            "artifact:analysis_cohort",
            "table:cohort_flow",
            "table:cohort_attrition",
        ],
        method="cohort_definition_and_attrition",
    )


def _assert_canonical_schema_guidance(text: str) -> None:
    assert "PRIMARY ANALYSIS-COHORT PRODUCT SCHEMA" in text or (
        "exact top-level integer fields `n_universe`" in text
    )
    assert "`n_final_analysis_cohort`" in text
    assert "`criterion_id`" in text
    assert "`n_at_start_rows`" in text
    assert "`n_remaining_rows`" in text
    assert "`n_excluded_rows`" in text
    assert "`{include|exclude}_{order:02d}_{normalized_concept_id}`" in text
    assert "Do not split a predicate" in text


def test_initial_coder_prompt_receives_primary_cohort_canonical_schema() -> None:
    llm = _CaptureLLM(["import os\nresult = 1\n"])

    CoderAgent(llm).run(context=_context(), step=_primary_cohort_step())

    assert len(llm.calls) == 1
    _assert_canonical_schema_guidance(llm.calls[0][0][-1].content)


def test_repair_prompt_and_contract_guidance_share_primary_cohort_schema() -> None:
    patch = json.dumps(
        {
            "format": "easyicu.code_patch/1",
            "edits": [
                {
                    "old": "result = 1",
                    "new": "result = 2",
                    "expected_count": 1,
                }
            ],
        }
    )
    llm = _CaptureLLM([patch])
    step = _primary_cohort_step()
    repair_guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"status": "contract_failed"},
        code="import os\nresult = 1\n",
    )

    repaired = CoderAgent(llm).repair(
        context=_context(),
        step=step,
        code="import os\nresult = 1\n",
        run_log=repair_guidance,
    )

    assert repaired.strip().endswith("result = 2")
    _assert_canonical_schema_guidance(repair_guidance)
    _assert_canonical_schema_guidance(llm.calls[0][0][-1].content)


def test_primary_cohort_schema_guidance_tracks_host_product_aliases() -> None:
    step = _primary_cohort_step().model_copy(
        update={
            "expected_outputs": [
                "dataset:analysis_cohort",
                "table:attrition",
            ]
        }
    )

    _assert_canonical_schema_guidance(
        _primary_analysis_cohort_output_contract(step)
    )


def test_primary_cohort_schema_guidance_requires_host_method_family() -> None:
    step = _primary_cohort_step().model_copy(
        update={"method": "mixed_effects_regression"}
    )

    assert _primary_analysis_cohort_output_contract(step) == ""


@pytest.mark.parametrize(
    "outputs",
    [
        ["artifact:analysis_cohort"],
        ["table:cohort_flow", "table:cohort_attrition"],
        ["artifact:adult_analysis_cohort", "table:cohort_flow"],
        ["artifact:analysis_cohort", "table:table_one"],
        ["table:ordinary_summary"],
    ],
)
def test_primary_cohort_schema_guidance_does_not_leak_to_other_products(
    outputs: list[str],
) -> None:
    step = AnalysisStep(
        step_id="cohort_words_are_not_authority",
        intent=(
            "Mention analysis_cohort, cohort_flow, canonical attrition, and "
            "missingness in prose only."
        ),
        expected_outputs=outputs,
        method="cohort_definition_and_attrition",
    )

    assert _primary_analysis_cohort_output_contract(step) == ""
    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"status": "contract_failed"},
        code="",
    )
    assert "n_final_analysis_cohort" not in guidance
    assert "normalized_concept_id" not in guidance
