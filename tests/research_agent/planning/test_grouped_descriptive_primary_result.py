"""Replay the grouped primary shape using only synthetic metadata and rows."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from easyicu.research_agent.contracts.cohort_summary import (
    descriptive_summary_result_receipt_valid,
)
from easyicu.research_agent.execution.runners.cohort_summary_executor import (
    cohort_summary_executor_code,
    validate_primary_descriptive_tables,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.capability_registry import (
    assess_scientific_capability,
    resolve_primary_capability,
)
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, ConceptDescriptor, ResearchContext


def _context(*, measured=True):
    return ResearchContext.model_validate({
        "research_question": "Describe synthetic recovery duration in each admission category.",
        "cohort": {
            "cohort_name": "synthetic", "database": "synthetic", "n_stays": 8 if measured else 0,
            "id_columns": ["stay_id"], "outcome_columns": ["recovery"],
            "requested_outcome_columns": ["recovery"],
            "provenance": {} if measured else {
                "evidence_stage": "metadata_only_planning", "patient_rows_read": False,
            },
        },
        "primary_exposure": "admission_category", "target_outcome": "recovery",
        "endpoint": {"name": "recovery", "kind": "continuous",
                     "absence_semantics": "absent_row_is_unmeasured"},
        "variables": [
            {"name": "stay_id", "role": "id", "dtype": "int64"},
            {"name": "admission_category", "role": "other", "dtype": "float64",
             "source_concept": "synthetic_factor",
             "observed_domain": {"levels": [10, 20], "n_unique": 2} if measured else None},
            {"name": "recovery", "role": "outcome", "dtype": "float64", "unit": "hours"},
        ],
    })


def _plan():
    steps = [
        AnalysisStep(
            step_id="define_cohort", planned_analysis_role="auxiliary",
            intent="Account for the analysis denominator.", method="cohort_definition_and_attrition",
            inputs=["stay_id", "recovery"],
            expected_outputs=["artifact:analysis_cohort", "table:cohort_flow"],
            cohort_definition_spec={"identity_column": "stay_id", "analysis_set": "bound_universe"},
        ),
        AnalysisStep(
            step_id="grouped_primary", planned_analysis_role="primary",
            intent="Describe the empirical distribution within the declared categories.",
            method="grouped_empirical_distribution_summary",
            scientific_action_id="descriptive.descriptive_summary",
            inputs=["stay_id", "admission_category", "recovery", "artifact:analysis_cohort"],
            expected_outputs=["table:cohort_summary", "table:distribution_summary"],
        ),
        AnalysisStep(
            step_id="continuous_secondary", planned_analysis_role="secondary",
            intent="Report continuous outcomes in each category.", method="continuous_outcome_summary",
            inputs=["artifact:analysis_cohort", "admission_category", "recovery"],
            expected_outputs=["table:continuous_outcome_summary"],
            descriptive_claim={"unresolved_limitations": ["post_baseline_exposure_opportunity_unresolved"]},
        ),
        AnalysisStep(
            step_id="availability", planned_analysis_role="auxiliary",
            intent="Audit measurement completeness.", method="missing_data",
            inputs=["stay_id", "admission_category", "recovery", "artifact:analysis_cohort"],
            expected_outputs=["table:measurement_audit", "table:measurement_process_audit"],
            measurement_audit_spec={"products": [
                {"product_id": "measurement_audit", "audit": "measurement_missingness"},
                {"product_id": "measurement_process_audit", "audit": "measurement_process"},
            ]},
        ),
    ]
    for output, sources, role, chart in [
        ("primary_result", ["distribution_summary"], "distribution", "distribution_plot"),
        ("cohort_flow", ["cohort_flow"], "cohort_accounting", "cohort_flow"),
        ("data_quality", ["measurement_audit", "measurement_process_audit"], "data_quality", "availability_panel"),
    ]:
        steps.append(AnalysisStep(
            step_id=f"render_{output}", planned_analysis_role="auxiliary",
            intent="Render the bound result products.", method="visualization",
            inputs=[f"table:{source}" for source in sources], expected_outputs=[f"figure:{output}"],
            input_consumption_contracts=[{"input_key": f"table:{source}", "mode": "all_rows"} for source in sources],
            figure_panels=[{
                "panel_id": source, "figure_output": f"figure:{output}",
                "article_role": role, "chart_type": chart, "source_products": [f"table:{source}"],
            } for source in sources],
        ))
    return AnalysisPlan(research_question=_context().research_question,
                        analysis_type="descriptive_epidemiology", steps=steps, endpoint=_context().endpoint)


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("exposure", [None, "admission_category"])
def test_a_grouped_primary_is_reportable_with_the_real_plan_structure(monkeypatch, measured, exposure):
    from easyicu.research_agent.research_context import typed

    monkeypatch.setitem(typed._CONCEPT_FACTOR_CACHE, "synthetic_factor", True)
    context, plan = _context(measured=measured), _plan()
    context = context.model_copy(update={"primary_exposure": exposure})
    verdict = resolve_primary_capability(analysis_type=plan.analysis_type, plan=plan)
    assert verdict.owner_claimed is True
    assert verdict.capability_id == "descriptive_measurement_v1"
    assessment = assess_scientific_capability(analysis_type=plan.analysis_type, plan=plan, context=context)
    assert assessment.claim_ceiling == "reportable"
    assert assessment.scientific_validator_available
    review = build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)
    assert not {"SCIENTIFIC_CAPABILITY_NOT_REPORTABLE", "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE"} & {
        finding.code for finding in review.findings
    }
    assert select_standard_executor(plan.steps[1], plan=plan).analysis_kind == "descriptive_cohort_summary"


def test_a_typed_secondary_cannot_promote_an_untyped_generated_primary():
    plan = _plan()
    plan.steps[1] = plan.steps[1].model_copy(update={"scientific_action_id": None})
    review = build_plan_scientific_review(context=_context(), plan=plan, require_reportable_capability=True)
    assert "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE" in {finding.code for finding in review.findings}
    assert resolve_primary_capability(analysis_type=plan.analysis_type, plan=plan).owner_claimed is not True


def test_the_secondary_still_must_cover_the_requested_outcome_and_grouping():
    plan = _plan()
    plan.steps = [step for step in plan.steps if step.step_id != "continuous_secondary"]
    review = build_plan_scientific_review(context=_context(), plan=plan, require_reportable_capability=True)
    assert "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE" in {finding.code for finding in review.findings}


@pytest.mark.parametrize("change", [
    {"primary_exposure": "unknown_category"},
    {"primary_exposure": "recovery"},
])
def test_grouping_must_be_bound_by_typed_metadata(change):
    context, plan = _context().model_copy(update=change), _plan()
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    ).claim_ceiling != "reportable"


@pytest.mark.parametrize("change", [
    {"observed_domain": None, "source_concept": None},
    {"role": "id"},
])
def test_a_second_numeric_column_is_not_an_untyped_grouping(change):
    context, plan = _context(), _plan()
    context.variables[1] = ConceptDescriptor.model_validate({**context.variables[1].model_dump(), **change})
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    ).claim_ceiling != "reportable"


def test_a_factor_code_is_not_a_single_numeric_measurement(monkeypatch):
    from easyicu.research_agent.research_context import typed

    monkeypatch.setitem(typed._CONCEPT_FACTOR_CACHE, "synthetic_factor", True)
    context, plan = _context(measured=False), _plan()
    plan.steps[1] = plan.steps[1].model_copy(update={
        "inputs": ["stay_id", "admission_category", "artifact:analysis_cohort"],
    })
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    ).claim_ceiling != "reportable"


@pytest.mark.parametrize("levels", [[10, 20], ["Overall", "Missing"], ["NA", "null"]])
@pytest.mark.parametrize("outcome_domain", [None, {"levels": [1., 2., 3., 4., 7., 8.], "n_unique": 6}])
def test_grouped_primary_executes_and_validates_every_partition(tmp_path, monkeypatch, levels, outcome_domain):
    context, plan = _context(), _plan()
    context = context.model_copy(update={"primary_exposure": None})
    context.variables[1] = context.variables[1].model_copy(update={
        "dtype": "object" if isinstance(levels[0], str) else "float64",
        "observed_domain": {"levels": levels, "n_unique": 2},
    })
    context.variables[2] = context.variables[2].model_copy(update={"observed_domain": outcome_domain})
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    frame = pd.DataFrame({
        "stay_id": range(8), "admission_category": [*([levels[0]] * 3), *([levels[1]] * 3), None, None],
        "recovery": [1., 3., None, 2., 4., 8., 7., None],
    })
    source = tmp_path / "synthetic.parquet"
    frame.to_parquet(source, index=False)
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    from easyicu.research_agent.execution.runners import cohort_summary_executor as owner

    monkeypatch.setattr(owner, "load_cohort_summary_frame", lambda **kwargs: (frame, source))
    exec(compile(cohort_summary_executor_code(plan.steps[1]), "<grouped-summary>", "exec"), {})
    result = json.loads((tmp_path / "out/step_summary.json").read_text())
    assert descriptive_summary_result_receipt_valid(result)
    for field in ("grouping_variable", "grouping_levels"):
        corrupt = {**result, field: None}
        assert not descriptive_summary_result_receipt_valid(corrupt)
    assert result["grouping_variable"] == "admission_category"
    assert result["declared_columns"] == ["recovery"]
    summary = pd.read_csv(tmp_path / "out/cohort_summary.csv", converters={"group": str})
    distribution = pd.read_csv(tmp_path / "out/distribution_summary.csv", converters={"group": str})
    medians = summary.loc[summary.statistic == "median"]
    assert medians["group"].tolist() == ["Overall", *map(str, levels), "Missing"]
    assert medians["value"].tolist() == [3.5, 2., 4., 7.]
    counts = summary.loc[summary.statistic == "cohort_n", "value"]
    assert counts.tolist() == [8, 3, 3, 2]
    def validate(s, d):
        validate_primary_descriptive_tables(
            frame[["admission_category", "recovery"]], s, d,
            grouping_variable="admission_category", grouping_levels=levels,
        )

    validate(summary, distribution)
    for column in ("count", "denominator", "value", "cumulative_proportion", "group", "row_role", "group_variable"):
        corrupt = distribution.copy()
        idx = corrupt.index[corrupt["row_role"] == "exposure_level"][0]
        corrupt.loc[idx, column] = "invented" if column in {"group", "row_role", "group_variable"} else corrupt.loc[idx, column] + 1
        with pytest.raises(ValueError, match="descriptive"):
            validate(summary, corrupt)
    corrupt = summary.copy()
    corrupt.loc[(corrupt["group"] == str(levels[0])) & (corrupt.statistic == "median"), "value"] = 99
    with pytest.raises(ValueError, match="descriptive"):
        validate(corrupt, distribution)
    extra = distribution.iloc[[0]].assign(group="invented")
    with pytest.raises(ValueError, match="descriptive"):
        validate(summary, pd.concat([distribution, extra], ignore_index=True))
    wrong_source = frame.copy()
    wrong_source.loc[0, "admission_category"] = 99
    with pytest.raises(ValueError, match="closed metadata domain"):
        validate_primary_descriptive_tables(
            wrong_source[["admission_category", "recovery"]], summary, distribution,
            grouping_variable="admission_category", grouping_levels=levels,
        )
    with pytest.raises(ValueError, match="closed metadata domain"):
        owner.run_cohort_summary_from_env(
            declared_columns=["stay_id", "admission_category", "recovery"],
            typed_cohort_input="artifact:analysis_cohort", frame=wrong_source,
            cohort_path=source, primary_summary=True,
        )


def test_declared_levels_cannot_count_the_same_stay_in_two_groups(tmp_path, monkeypatch):
    from easyicu.research_agent.execution.runners import cohort_summary_executor as owner

    context = _context()
    context.variables[1] = context.variables[1].model_copy(update={
        "observed_domain": {"levels": [10, 10.0], "n_unique": 2},
    })
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    frame = pd.DataFrame({"stay_id": [1, 2], "admission_category": [10, 10], "recovery": [1., 3.]})
    source = tmp_path / "synthetic.parquet"
    frame.to_parquet(source)
    with pytest.raises(ValueError, match="partition"):
        owner.run_cohort_summary_from_env(
            declared_columns=["stay_id", "admission_category", "recovery"],
            typed_cohort_input="artifact:analysis_cohort", frame=frame, cohort_path=source, primary_summary=True,
        )
