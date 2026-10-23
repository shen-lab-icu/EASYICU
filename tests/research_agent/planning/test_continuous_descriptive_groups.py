"""A grouped continuous question keeps both its endpoint and its grouping."""

import json

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_planner import ProgressivePlannerAgent
from easyicu.research_agent.execution.runners.continuous_outcome_summary_executor import run_continuous_outcome_summary
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, EndpointSpec, ResearchContext


def context(*, measured):
    return ResearchContext(
        research_question="Describe synthetic recovery duration in each admission ward category.",
        cohort=CohortDescriptor(
            cohort_name="synthetic", database="synthetic", n_stays=6 if measured else 0,
            id_columns=["stay_id"], outcome_columns=["recovery"], requested_outcome_columns=["recovery"],
            provenance={} if measured else {"evidence_stage": "metadata_only_planning", "patient_rows_read": False},
        ),
        primary_exposure="ward_group", target_outcome="recovery",
        endpoint=EndpointSpec(name="recovery", kind="continuous", absence_semantics="absent_row_is_unmeasured"),
        variables=[
            ConceptDescriptor(name="stay_id", role="id", dtype="int64"),
            ConceptDescriptor(name="ward_group", role="demographic", dtype="float64", source_concept="synthetic_factor",
                              observed_domain={"levels": [10, 20], "n_unique": 2} if measured else None),
            ConceptDescriptor(name="recovery", role="outcome", dtype="float64", unit="hours",
                              description="Whole episode recovery duration"),
        ],
    )


def compile_plan(ctx):
    return compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate({
        "analysis_type": "descriptive_epidemiology",
        "rationale": "Describe the overall distribution and every requested group without inference.",
        "cohort": {"name": "primary", "selection_mode": "all_input_rows"},
        "display_labels": [{"key": "recovery", "value": "Recovery duration"},
                           {"key": "ward_group", "value": "Admission ward category"}],
        "steps": [
            {"step_id": "cohort", "module_id": "cohort_definition", "planned_analysis_role": "auxiliary",
             "objective": "Account for the analysis stay denominator."},
            {"step_id": "overall", "module_id": "custom_analysis", "planned_analysis_role": "primary",
             "objective": "Describe the overall empirical recovery distribution.",
             "scientific_action_id": "descriptive.descriptive_summary", "custom_method": "empirical_summary",
             "depends_on": ["cohort"], "raw_inputs": ["recovery"],
             "outputs": [{"product_id": "table:cohort_summary", "semantic_role": "custom"},
                         {"product_id": "table:distribution_summary", "semantic_role": "custom"}]},
            {"step_id": "groups", "module_id": "continuous_outcome_summary", "planned_analysis_role": "secondary",
             "objective": "Describe recovery in every admission ward category.",
             "depends_on": ["cohort"], "primary_exposure": "ward_group", "raw_inputs": ["ward_group", "recovery"]},
        ],
    }), context=ctx)[0]


def findings(ctx, plan):
    return {item.code for item in build_plan_scientific_review(
        context=ctx, plan=plan, require_reportable_capability=True,
    ).findings}


@pytest.mark.parametrize("measured", [False, True])
def test_typed_plan_keeps_the_group_and_passes_endpoint_coverage_and_capability_review(monkeypatch, measured):
    from easyicu.research_agent.research_context import typed
    monkeypatch.setitem(typed._CONCEPT_FACTOR_CACHE, "synthetic_factor", True)
    ctx = context(measured=measured)
    plan = compile_plan(ctx)
    review = build_plan_scientific_review(context=ctx, plan=plan, require_reportable_capability=True)
    assert review.approval_allowed
    assert not {"OUTCOME_DEFINITION_UNRESOLVED", "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
                "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE"} & {item.code for item in review.findings}
    assert plan.steps[-1].inputs == ["artifact:analysis_cohort", "ward_group", "recovery"]
    assert select_standard_executor(plan.steps[-1], plan=plan).analysis_kind == "continuous_outcome_summary"
    # The overall primary result alone cannot answer a question about groups.
    ungrouped = plan.model_copy(update={"steps": plan.steps[:-1]})
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" in findings(ctx, ungrouped)


def test_unknown_levels_are_planning_only_and_post_extraction_groups_execute(monkeypatch, tmp_path):
    from easyicu.research_agent.research_context import typed
    monkeypatch.setitem(typed._CONCEPT_FACTOR_CACHE, "synthetic_factor", True)
    initial = context(measured=False)
    compile_plan(initial)
    frame = pd.DataFrame({"ward_group": [10, 10, 10, 20, 20, 20], "recovery": [1., 3., None, 2., 4., 8.]})
    (tmp_path / "research_context.json").write_text(initial.model_dump_json())
    args = dict(frame=frame, grouping_variable="ward_group", outcome_variables=["recovery"],
                typed_cohort_input="artifact:analysis_cohort", source_cohort=tmp_path / "synthetic.parquet",
                run_dir=tmp_path, out_dir=tmp_path / "out")
    with pytest.raises(RuntimeError, match="closed domain"):
        run_continuous_outcome_summary(**args)
    assert not (tmp_path / "out/continuous_outcome_summary.csv").exists()
    measured = context(measured=True)
    plan = compile_plan(measured)
    (tmp_path / "research_context.json").write_text(measured.model_dump_json())
    summary = run_continuous_outcome_summary(**args)
    assert summary["inferential_tests"] == []
    table = pd.read_csv(tmp_path / "out/continuous_outcome_summary.csv")
    assert table["group"].tolist() == ["Overall", "10", "20"]
    assert table["median"].tolist() == [3., 2., 4.]
    assert table.n_nonmissing.tolist() == [5, 2, 3]
    assert table.unit.tolist() == ["hours"] * 3
    assert not {"OUTCOME_DEFINITION_UNRESOLVED", "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
                "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE"} & findings(measured, plan)


def test_continuous_grouping_is_not_a_deferred_categorical_domain():
    with pytest.raises(ValueError, match="closed domain"):
        compile_plan(context(measured=False).model_copy(update={"variables": [
            ConceptDescriptor(name="stay_id", role="id", dtype="int64"),
            ConceptDescriptor(name="ward_group", role="demographic", dtype="float64"),
            ConceptDescriptor(name="recovery", role="outcome", dtype="float64", unit="hours"),
        ]}))


def test_another_grouping_cannot_cover_the_requested_comparison():
    ctx = context(measured=True)
    plan = compile_plan(ctx)
    ctx = ctx.model_copy(update={"variables": [*ctx.variables, ConceptDescriptor(
        name="another_group", role="demographic", dtype="int64", observed_domain={"levels": [0, 1]},
    )]})
    wrong = plan.steps[-1].model_copy(update={"inputs": ["artifact:analysis_cohort", "another_group", "recovery"]})
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" in findings(ctx, plan.model_copy(update={"steps": [*plan.steps[:-1], wrong]}))


@pytest.mark.parametrize("change", [
    {"n_stays": 6}, {"provenance": {"evidence_stage": "extracted", "patient_rows_read": False}},
    {"provenance": {"evidence_stage": "metadata_only_planning", "patient_rows_read": True}},
])
def test_only_row_free_planning_can_defer_levels(monkeypatch, change):
    from easyicu.research_agent.research_context import typed
    monkeypatch.setitem(typed._CONCEPT_FACTOR_CACHE, "synthetic_factor", True)
    ctx = context(measured=False)
    with pytest.raises(ValueError, match="closed domain"):
        compile_plan(ctx.model_copy(update={"cohort": ctx.cohort.model_copy(update=change)}))


def test_prompt_requires_answering_summary_and_preserves_unknown_groups():
    ctx = context(measured=False)
    families, variables, actions, rows = ProgressivePlannerAgent._request_authorities(ctx)
    prompt = ProgressivePlannerAgent._user_prompt(ctx, analysis_types=families, variables=variables, action_rows=rows)
    assert "answering secondary" in prompt
    assert "keep the grouping" in prompt
    assert json.loads(ctx.model_dump_json())["primary_exposure"] == "ward_group"
