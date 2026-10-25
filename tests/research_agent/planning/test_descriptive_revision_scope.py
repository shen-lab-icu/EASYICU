"""Synthetic summaries keep their declared science through host shaping and repair."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.research_agent.authority.plan_input_closure import close_measurement_companion_inputs
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.contracts.cohort_summary import descriptive_summary_result_receipt_valid
from easyicu.research_agent.contracts.continuous_outcome_summary import continuous_outcome_summary_verdict
from easyicu.research_agent.execution.runners.cohort_summary_executor import cohort_summary_executor_code
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ResearchContext
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.research_plan_revision import PlanRevisionGuard


def _study(grouped=True):
    context = ResearchContext.model_validate({
        "research_question": "Summarize observed signal measurements across ward categories.",
        "cohort": {"cohort_name": "synthetic", "database": "synthetic", "n_stays": 8,
                   "id_columns": ["stay_id"], "outcome_columns": ["signal_max"],
                   "requested_outcome_columns": ["signal_max"]},
        "primary_exposure": None,
        "target_outcome": "signal_max",
        "endpoint": {"name": "signal_max", "kind": "continuous", "absence_semantics": "absent_row_is_unmeasured"},
        "variables": [
            {"name": "stay_id", "role": "id", "dtype": "int64"},
            {"name": "admission_category", "role": "other", "dtype": "str",
             "observed_domain": {"levels": ["A", "B", "C"], "n_unique": 3}},
            {"name": "signal_max", "role": "outcome", "dtype": "float64",
             "description": "Maximum observed signal during the first day."},
        ] + [{"name": name, "role": "meta", "dtype": "int64"}
             for name in ("signal_measured", "signal_n")],
    })
    plan = AnalysisPlan(
        research_question=context.research_question, analysis_type="descriptive_epidemiology",
        endpoint=context.endpoint, steps=[
            AnalysisStep(
                step_id="define_cohort", planned_analysis_role="auxiliary",
                intent="Account for every synthetic stay.", method="cohort_definition_and_attrition",
                inputs=["stay_id", "signal_max"],
                expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
                cohort_definition_spec={"identity_column": "stay_id", "analysis_set": "bound_universe"},
            ),
            AnalysisStep(
                step_id="grouped_primary", planned_analysis_role="primary",
                intent="Describe the measurement distribution by category.", method="empirical_distribution_summary",
                scientific_action_id="descriptive.descriptive_summary",
                inputs=["stay_id", "admission_category", "signal_max", "artifact:analysis_cohort"],
                expected_outputs=["table:cohort_summary", "table:distribution_summary"],
            ),
            AnalysisStep(
                step_id="continuous_secondary", planned_analysis_role="secondary",
                intent="Summarize the continuous measurement within categories.", method="continuous_outcome_summary",
                inputs=["artifact:analysis_cohort", "admission_category", "signal_max"],
                expected_outputs=["table:continuous_outcome_summary"],
            ),
            AnalysisStep(
                step_id="availability", planned_analysis_role="auxiliary",
                intent="Audit measurement completeness.", method="missing_data",
                inputs=["stay_id", "admission_category", "signal_max", "artifact:analysis_cohort"],
                expected_outputs=["table:measurement_audit", "table:measurement_process_audit"],
                measurement_audit_spec={"products": [
                    {"product_id": "measurement_audit", "audit": "measurement_missingness"},
                    {"product_id": "measurement_process_audit", "audit": "measurement_process"},
                ]},
            ),
        ],
    )
    if not grouped:
        plan = plan.model_copy(update={"steps": [
            step.model_copy(update={"inputs": [v for v in step.inputs if v != "admission_category"]})
            for step in plan.steps if step.step_id != "continuous_secondary"
        ]})
    return context, plan


def _guard(context, plan):
    review = build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)
    assert review.plan_sha256 == canonical_sha256(plan.model_dump(mode="json"))
    return PlanRevisionGuard.from_review(plan=plan.model_dump(mode="json"), review=review)


@pytest.mark.parametrize("grouped", [False, True])
def test_companion_closure_keeps_summaries_executable_and_keeps_the_audit(grouped):
    context, plan = _study(grouped)
    closed, findings = close_measurement_companion_inputs(plan=plan, context=context)
    summaries = [step for step in closed.steps if step.planned_analysis_role in {"primary", "secondary"}]
    for step in summaries:
        assert step.inputs == next(old.inputs for old in plan.steps if old.step_id == step.step_id)
        assert select_standard_executor(step, plan=closed) is not None
        if step.planned_analysis_role == "secondary":
            assert continuous_outcome_summary_verdict(step).claimed
    audit = next(step for step in closed.steps if step.step_id == "availability")
    assert {"signal_max", "signal_measured", "signal_n"} <= set(audit.inputs)
    assert len(audit.measurement_audit_spec.products) == 2
    assert findings
    review = build_plan_scientific_review(context=context, plan=closed, require_reportable_capability=True)
    assert review.facts["scientific_capability"]["claim_ceiling"] == "reportable"
    assert not {"SCIENTIFIC_CAPABILITY_NOT_REPORTABLE", "OUTCOME_DEFINITION_UNRESOLVED"} & {
        finding.code for finding in review.findings
    }
    # Publication evidence stays visible, and cannot demand new optional science.
    robustness = next(f for f in review.findings if f.code == "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED")
    assert (robustness.severity, robustness.remediation_route) == ("major", "runtime_capability")
    assert review.facts["robustness_readiness"]["effect_style_grid_required"] is False


@pytest.mark.parametrize("grouped", [False, True])
def test_owned_summary_method_wording_is_a_repair_not_new_science(grouped):
    context, plan = _study(grouped)
    guard = _guard(context, plan)
    revised = plan.model_copy(update={"steps": [
        step.model_copy(update={"method": "empirical_measurement_summary_with_categories"})
        if step.planned_analysis_role == "primary" else step for step in plan.steps
    ]})
    guard.validate_plan(plan=revised, context=context)
    assert guard.previous_plan == plan.model_dump(mode="json")


@pytest.mark.parametrize("companion_role", ["meta", "other"])
def test_summary_repair_removes_only_typed_audit_companions(companion_role):
    context, plan = _study()
    context = context.model_copy(update={"variables": [
        variable.model_copy(update={"role": type(variable.role)(companion_role)}) if variable.role.value == "meta" else variable
        for variable in context.variables
    ]})
    primary = plan.steps[1]
    source = plan.model_copy(update={"steps": [plan.steps[0], primary.model_copy(update={
        "inputs": [*primary.inputs, "signal_measured", "signal_n"],
    }), *plan.steps[2:]]})
    revised = plan.model_copy(update={"steps": [plan.steps[0], primary.model_copy(update={
        "method": "reworded_empirical_measurement_summary",
    }), *plan.steps[2:]]})
    guard = _guard(context, source)
    assert "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE" in {f.code for f in guard.review.findings}
    if companion_role == "meta":
        guard.validate_plan(plan=revised, context=context)
    else:
        with pytest.raises(ResearchPipelineRunError) as raised:
            guard.validate_plan(plan=revised, context=context)
        assert raised.value.code == "revision_added_unrequested_analysis"


@pytest.mark.parametrize("change", [
    "extra_step", "new_input", "known_input", "removed_group", "removed_measurement",
    "new_product", "new_role", "untyped", "model",
])
def test_summary_rewording_cannot_authorize_additional_science(change):
    context, plan = _study()
    primary = plan.steps[1]
    repaired = primary.model_copy(update={"method": "reworded_empirical_measurement_summary"})
    updates = {
        "new_input": {"inputs": [*primary.inputs, "unrequested_measurement"]},
        "known_input": {"inputs": [*primary.inputs, "signal_n"]},
        "removed_group": {"inputs": [name for name in primary.inputs if name != "admission_category"]},
        "removed_measurement": {"inputs": [name for name in primary.inputs if name != "signal_max"]},
        "new_product": {"expected_outputs": [*primary.expected_outputs, "table:extra"],
                        "method": "descriptive.descriptive_summary"},
        "new_role": {"planned_analysis_role": "secondary"},
        "untyped": {"scientific_action_id": None},
    }
    if change == "model":
        from tests.research_agent.planning.scientific_review_fixtures import _plan

        repaired = repaired.model_copy(update={"model_requirements": _plan().steps[0].model_requirements})
    elif change in updates:
        repaired = repaired.model_copy(update=updates[change])
    steps = [repaired if step.step_id == primary.step_id else step for step in plan.steps]
    if change == "known_input":
        audit = next(step for step in plan.steps if step.step_id == "availability")
        audit.inputs.append("signal_n")
    if change == "extra_step":
        steps.append(repaired.model_copy(update={"step_id": "duplicate_summary"}))
    with pytest.raises(ResearchPipelineRunError) as raised:
        _guard(context, plan).validate_plan(plan=plan.model_copy(update={"steps": steps}), context=context)
    assert raised.value.code == "revision_added_unrequested_analysis"
    if change == "extra_step":
        assert raised.value.details["step_id"] == "duplicate_summary"


@pytest.mark.parametrize("grouped", [False, True])
def test_repaired_summary_produces_validated_synthetic_results(tmp_path, monkeypatch, grouped):
    context, plan = _study(grouped)
    closed, _ = close_measurement_companion_inputs(plan=plan, context=context)
    primary = closed.steps[1].model_copy(update={"method": "empirical_measurement_summary_with_categories"})
    revised = closed.model_copy(update={"steps": [closed.steps[0], primary, *closed.steps[2:]]})
    _guard(context, closed).validate_plan(plan=revised, context=context)
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    frame = pd.DataFrame({
        "stay_id": range(8), "admission_category": ["A", "A", "B", "B", "C", "C", None, None],
        "signal_max": [1., 3., 2., None, 4., 8., 7., None],
        "signal_measured": [1, 1, 1, 0, 1, 1, 1, 0], "signal_n": [1, 1, 1, 0, 1, 1, 1, 0],
    })
    source = tmp_path / "synthetic.parquet"
    frame.to_parquet(source, index=False)
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    from easyicu.research_agent.execution.runners import cohort_summary_executor as owner

    monkeypatch.setattr(owner, "load_cohort_summary_frame", lambda **kwargs: (frame, source))
    exec(compile(cohort_summary_executor_code(primary), "<synthetic-summary>", "exec"), {})
    result = json.loads((tmp_path / "out/step_summary.json").read_text())
    assert descriptive_summary_result_receipt_valid(result)
    assert result["declared_columns"] == ["signal_max"]
    assert result["cohort_n"] == 8
    summary = pd.read_csv(tmp_path / "out/cohort_summary.csv")
    assert summary.loc[summary.statistic == "median", "value"].tolist() == (
        [3.5, 2., 2., 6., 7.] if grouped else [3.5]
    )
    assert summary.loc[summary.statistic == "cohort_n", "value"].tolist() == (
        [8, 2, 2, 2, 2] if grouped else [8]
    )


def test_proportion_summary_keeps_its_missingness_and_sensitivity_boundary():
    from tests.research_agent.planning.scientific_review_fixtures import (
        _absolute_risk_distribution_step, _context,
    )

    context = _context()
    primary = _absolute_risk_distribution_step(descriptive=True)
    plan = AnalysisPlan(research_question=context.research_question,
                        analysis_type="descriptive_epidemiology", steps=[primary])
    closed, _ = close_measurement_companion_inputs(plan=plan, context=context)
    _guard(context, plan).validate_plan(plan=closed, context=context)
    assert closed.steps[0].exposure_outcome_distribution_spec == primary.exposure_outcome_distribution_spec
    assert select_standard_executor(closed.steps[0], plan=closed).analysis_kind == "exposure_outcome_distribution"


def test_association_review_retains_the_required_sensitivity():
    from tests.research_agent.planning.scientific_review_fixtures import _context, _plan

    context, plan = _context(), _plan()
    primary = plan.steps[0]
    requirement = primary.model_requirements[0].model_copy(update={"analysis_set": "complete_case"})
    plan = plan.model_copy(update={"steps": [
        primary.model_copy(update={"model_requirements": [requirement]}), *plan.steps[1:],
    ]})
    closed, _ = close_measurement_companion_inputs(plan=plan, context=context)
    review = build_plan_scientific_review(context=context, plan=closed)
    assert "MISSINGNESS_UNEXAMINED_COMPLETE_CASE" in {f.code for f in review.findings}
    assert review.facts["robustness_readiness"]["effect_style_grid_required"] is True


@pytest.mark.parametrize("identity", ["stay_id", "new_identity"])
def test_cohort_can_restore_only_its_already_declared_identity_input(identity):
    context, plan = _study()
    plan = plan.model_copy(update={"steps": [
        step.model_copy(update={"inputs": [name for name in step.inputs if name != "stay_id"]})
        for step in plan.steps
    ]})
    cohort = plan.steps[0]
    revised = plan.model_copy(update={"steps": [
        cohort.model_copy(update={"inputs": [identity, *cohort.inputs]}), *plan.steps[1:],
    ]})
    guard = _guard(context, plan)
    if identity == "stay_id":
        guard.validate_plan(plan=revised, context=context)
    else:
        with pytest.raises(ResearchPipelineRunError) as raised:
            guard.validate_plan(plan=revised, context=context)
        assert raised.value.code == "revision_added_unrequested_analysis"
