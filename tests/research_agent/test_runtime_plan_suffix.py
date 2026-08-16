"""Executed-prefix and runtime observation-loop regressions."""

from __future__ import annotations

import json

import pytest

from easyicu.research_agent.agents.replanner import ReplannerAgent
from easyicu.research_agent.execution.phase_support import (
    _successful_step_requests_replan,
)
from easyicu.research_agent.planning.runtime_suffix import (
    RuntimePlanSuffixError,
    RuntimePlanSuffixRevision,
    merge_runtime_plan_suffix,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Describe a sealed cohort and report its result.",
        analysis_type="descriptive_epidemiology",
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Bind the authorized analysis cohort.",
                expected_outputs=["artifact:analysis_cohort"],
            ),
            AnalysisStep(
                step_id="02_summary",
                intent="Summarize the authorized analysis cohort.",
                inputs=["artifact:analysis_cohort"],
                expected_outputs=["table:descriptive_summary"],
                scientific_action_id="descriptive.descriptive_summary",
            ),
            AnalysisStep(
                step_id="03_report",
                intent="Report the verified descriptive result.",
                inputs=["table:descriptive_summary"],
                expected_outputs=["report:analysis_report"],
            ),
        ],
        rationale="Bind, summarize, and report in dependency order.",
    )


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Describe a sealed cohort and report its result.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=12,
            id_columns=["stay_id"],
        ),
        variables=[ConceptDescriptor(name="stay_id", dtype="int64")],
    )


def _revision(plan: AnalysisPlan) -> RuntimePlanSuffixRevision:
    revised_summary = plan.steps[1].model_copy(
        update={"intent": "Summarize the executed cohort observation."}
    )
    return RuntimePlanSuffixRevision(
        replace_from_step_id="02_summary",
        replacement_step=revised_summary,
        rationale="The executed cohort observation now governs the remaining suffix.",
    )


def test_runtime_suffix_merge_preserves_executed_prefix_exactly() -> None:
    plan = _plan()

    merged = merge_runtime_plan_suffix(
        current_plan=plan,
        completed_step_ids=["01_cohort"],
        revision=_revision(plan),
    )

    assert merged.steps[0] == plan.steps[0]
    assert merged.steps[1].intent == "Summarize the executed cohort observation."
    assert merged.steps[2] == plan.steps[2]
    assert merged.research_question == plan.research_question
    assert merged.analysis_type == plan.analysis_type
    assert merged.rationale == plan.rationale
    assert merged.revision == plan.revision + 1


def test_runtime_suffix_rejects_wrong_coordinate_and_noncontiguous_prefix() -> None:
    plan = _plan()
    wrong = _revision(plan).model_copy(update={"replace_from_step_id": "03_report"})

    with pytest.raises(RuntimePlanSuffixError) as coordinate_error:
        merge_runtime_plan_suffix(
            current_plan=plan,
            completed_step_ids=["01_cohort"],
            revision=wrong,
        )
    assert coordinate_error.value.reason_code == "runtime_suffix_coordinate_mismatch"

    with pytest.raises(RuntimePlanSuffixError) as prefix_error:
        merge_runtime_plan_suffix(
            current_plan=plan,
            completed_step_ids=["01_cohort", "03_report"],
            revision=_revision(plan),
        )
    assert prefix_error.value.reason_code == (
        "runtime_completed_prefix_noncontiguous"
    )


def test_progressive_success_requests_observation_replan_without_truthy_coercion() -> None:
    assert not _successful_step_requests_replan({"status": "ok"})
    assert _successful_step_requests_replan(
        {"status": "ok"},
        progressive_observation_loop=True,
    )
    assert not _successful_step_requests_replan(
        {"status": "failed"},
        progressive_observation_loop=True,
    )
    assert not _successful_step_requests_replan(
        {"status": "ok", "replan_requested": "true"}
    )


def test_replanner_returns_only_unexecuted_suffix_and_merges_host_side() -> None:
    plan = _plan()
    response = _revision(plan).model_dump(mode="json")
    llm = ScriptedMockLLMClient([json.dumps(response)])
    capsule_sha256 = "a" * 64
    agent = ReplannerAgent(llm)

    revised = agent.run(
        context=_context(),
        current_plan=plan,
        completed_step_records=[
            {
                "step_id": "01_cohort",
                "status": "ok",
                "step_summary": {"n_rows": 12},
                "step_authority_capsule_ref": {
                    "schema_version": "easyicu.step_authority_capsule_ref/1",
                    "step_id": "01_cohort",
                    "capsule_sha256": capsule_sha256,
                },
            }
        ],
        suffix_only=True,
    )

    assert revised.steps[0] == plan.steps[0]
    assert revised.steps[1].intent == "Summarize the executed cohort observation."
    assert revised.revision == plan.revision + 1
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0][-1].content
    assert "IMMUTABLE EXECUTED PREFIX OBSERVATION AUTHORITY" in prompt
    assert capsule_sha256 in prompt
    assert '"replace_from_step_id":"02_summary"' not in prompt
    assert "CURRENT UNEXECUTED STEP" in prompt
    assert "FUTURE OUTLINE" in prompt
    assert plan.steps[0].intent not in prompt
    assert "Return a RuntimePlanSuffixRevision beginning at '02_summary'" in prompt
