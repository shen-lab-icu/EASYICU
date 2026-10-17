"""Synthetic end-to-end owner checks; no benchmark question or patient data."""

import json

import pandas as pd
import pytest

from easyicu.research_agent.agents.progressive_payload import progressive_step_materialization_request
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.contracts.continuous_outcome_summary import (
    CONTINUOUS_OUTCOME_OUTPUT, continuous_outcome_summary_verdict,
)
from easyicu.research_agent.execution.runners.continuous_outcome_summary_executor import run_continuous_outcome_summary
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import ProgressiveOutlineStep, ProgressivePlanSkeleton, ProgressiveSkeletonStep
from easyicu.research_agent.planning.scientific_review import planned_model_outcomes
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole
from tests.research_agent.planning.progressive_planner_fixtures import _context, _payload


def _study():
    context = _context()
    variables = [*context.variables, *(ConceptDescriptor(
        name=name, role=VariableRole.OUTCOME, dtype="float64", unit=unit,
        observed_domain={"n_unique": 80, "min": 0.0, "max": 40.0},
    ) for name, unit in (("duration_a", "minutes"), ("duration_b", "hours")))]
    return context.model_copy(update={"research_question": "Describe two recovery durations across the exposure groups.",
                                      "variables": variables, "cohort": context.cohort.model_copy(update={
                                          "outcome_columns": ["outcome_flag", "duration_a", "duration_b"],
                                      })})


def _skeleton_step():
    return {"step_id": "04_durations", "planned_analysis_role": "secondary",
            "module_id": "continuous_outcome_summary", "objective": "Describe the two requested recovery durations within each exposure group.",
            "depends_on": ["01_cohort"], "primary_exposure": "exposure_flag",
            "raw_inputs": ["duration_b", "exposure_flag", "duration_a"], "outputs": []}


def _compiled(context=None, step=None):
    context = context or _study()
    payload = _payload()
    payload.update(analysis_type="descriptive_epidemiology", robustness_intents=[],
                   steps=[payload["steps"][0], payload["steps"][2], step or _skeleton_step()])
    skeleton = ProgressivePlanSkeleton.model_validate(payload)
    return compile_progressive_plan(skeleton=skeleton, context=context)[0]


def test_requested_continuous_outcomes_compile_to_a_selected_exact_owner():
    context = _study()
    plan = _compiled(context)
    step = plan.steps[-1]
    assert step.expected_outputs == [CONTINUOUS_OUTCOME_OUTPUT]
    assert step.inputs == ["artifact:analysis_cohort", "exposure_flag", "duration_b", "duration_a"]
    assert continuous_outcome_summary_verdict(step).claimed
    assert set(planned_model_outcomes(plan, context)) == {"outcome_flag", "duration_a", "duration_b"}
    scope = FlagOnlyPlausibilityScope(step_id=step.step_id, expected_columns=("duration_a",),
                                     source_contracts_sha256="a" * 64, authority_kind="resolved_raw_input_contracts")
    selected = select_standard_executor(step, plan=plan, plausibility_scope=scope)
    assert selected is not None and selected.analysis_kind == "continuous_outcome_summary"
    assert selected.consumed_input_keys == ("artifact:analysis_cohort",)
    assert 'summary["plausibility_audit"]' in selected.code
    compile(selected.code, "<continuous-summary>", "exec")


def test_provider_transport_offers_only_the_executable_secondary_shape():
    step = _skeleton_step()
    outline = ProgressiveOutlineStep(**{k: step[k] for k in (
        "step_id", "planned_analysis_role", "module_id", "objective", "depends_on",
    )}, variable_names=step["raw_inputs"])
    request = progressive_step_materialization_request(
        outline_step=outline, outline_step_sha256=canonical_sha256(outline.model_dump(mode="json")),
        variable_names=step["raw_inputs"], scientific_action_ids=[],
        available_product_refs=[("01_cohort", "artifact:analysis_cohort")],
    )
    properties = json.loads(request.schema_json)["$defs"]["ProgressiveSkeletonStep"]["properties"]
    assert properties["planned_analysis_role"]["const"] == "secondary"
    assert properties["primary_exposure"]["type"] == "string"
    assert properties["raw_inputs"]["minItems"] == 2
    assert properties["outputs"]["maxItems"] == 0
    for field in ("outcome", "outcome_type", "confidence_level", "denominator_policy"):
        assert properties[field] == {"type": "null"}
    assert properties["model_terms"]["maxItems"] == 0


@pytest.mark.parametrize("change", [
    {"planned_analysis_role": "primary"}, {"confidence_level": .95},
    {"outcome": "duration_a"}, {"raw_inputs": ["duration_a", "duration_b"]},
    {"model_terms": [{"name": "duration_a", "role": "covariate", "coding": "continuous"}]},
])
def test_summary_module_refuses_inference_or_an_implicit_grouping(change):
    with pytest.raises(ValueError):
        ProgressiveSkeletonStep.model_validate({**_skeleton_step(), **change})


@pytest.mark.parametrize("bad_outcome", ["outcome_flag", "age_years", "sex_code", "unavailable"])
def test_compiler_rejects_binary_baseline_or_unavailable_outcomes(bad_outcome):
    step = {**_skeleton_step(), "raw_inputs": ["exposure_flag", bad_outcome]}
    with pytest.raises(ValueError):
        _compiled(step=step)


def test_continuous_values_cannot_be_invented_as_closed_group_levels():
    step = {**_skeleton_step(), "primary_exposure": "age_years",
            "raw_inputs": ["age_years", "duration_a"]}
    with pytest.raises(ValueError, match="closed domain"):
        _compiled(step=step)


@pytest.mark.parametrize("change", [
    {"planned_analysis_role": "primary"}, {"method": "linear_regression"},
    {"expected_outputs": ["table:other"]},
    {"inputs": ["exposure_flag", "duration_a"]},
    {"inputs": ["artifact:analysis_cohort", "exposure_flag", "duration_a", "duration_a"]},
    {"scientific_capability": "association_adjusted_v1"},
])
def test_stale_or_handwritten_steps_do_not_gain_outcome_coverage(change):
    context = _study(); plan = _compiled(context)
    step = plan.steps[-1].model_copy(update=change)
    verdict = continuous_outcome_summary_verdict(step)
    assert not verdict.claimed and verdict.declaration_is_incomplete
    altered = plan.model_copy(update={"steps": [step]})
    assert planned_model_outcomes(altered, context) == ()


def test_a_summary_cannot_replace_a_requested_association():
    context = _study(); plan = _compiled(context)
    association = plan.model_copy(update={"analysis_type": "association_study"})
    assert not {"duration_a", "duration_b"} & set(planned_model_outcomes(association, context))


def test_each_outcome_retains_its_own_missingness_unit_and_partition(tmp_path):
    context = _study(); (tmp_path / "research_context.json").write_text(context.model_dump_json())
    frame = pd.DataFrame({"exposure_flag": [0, 0, 0, 1, 1, 1],
                          "duration_a": [1., 3., None, 2., 4., 8.],
                          "duration_b": [10., 20., 30., None, 60., 80.]})
    original = frame.copy(deep=True)
    summary = run_continuous_outcome_summary(
        frame=frame, grouping_variable="exposure_flag", outcome_variables=["duration_a", "duration_b"],
        typed_cohort_input="artifact:analysis_cohort", source_cohort=tmp_path / "synthetic.parquet",
        out_dir=tmp_path / "out", run_dir=tmp_path,
    )
    table = pd.read_csv(tmp_path / "out/continuous_outcome_summary.csv")
    assert len(table) == 6
    for variable in ("duration_a", "duration_b"):
        rows = table[table.variable == variable]
        assert rows.iloc[0].group_n == rows.iloc[1:].group_n.sum() == 6
        assert rows.iloc[0].n_nonmissing == 5
    a = table[(table.variable == "duration_a") & (table["group"] == "0")].iloc[0]
    b = table[(table.variable == "duration_b") & (table["group"] == "1")].iloc[0]
    assert (a["median"], a.missing_n, a.unit) == (2., 1, "minutes")
    assert (b["median"], b.missing_n, b.unit) == (70., 1, "hours")
    assert summary["adjusted_effect"] is None and summary["inferential_tests"] == []
    assert summary["source_row_count_reconciliation"]["filtering_performed"] is False
    assert set(summary["output_files"]) == {CONTINUOUS_OUTCOME_OUTPUT}
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("fault", ["missing_column", "non_numeric", "infinite", "unknown_group", "empty_group", "binary_metadata"])
def test_runtime_metadata_and_data_faults_leave_no_partial_product(tmp_path, fault):
    context = _study()
    payload = json.loads(context.model_dump_json())
    frame = pd.DataFrame({"exposure_flag": [0, 0, 1, 1], "duration_a": [1., 2., 3., 4.], "duration_b": [2., 3., 4., 5.]})
    if fault == "missing_column":
        frame = frame.drop(columns="duration_b")
    elif fault == "non_numeric":
        frame["duration_b"] = ["bad", "3", "4", "5"]
    elif fault == "infinite":
        frame.loc[0, "duration_b"] = float("inf")
    elif fault == "unknown_group":
        frame.loc[0, "exposure_flag"] = 2
    elif fault == "empty_group":
        frame["exposure_flag"] = 0
    else:
        payload["variables"][-1]["observed_domain"] = {"n_unique": 2, "is_binary": True, "levels": [0, 1]}
    (tmp_path / "research_context.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError):
        run_continuous_outcome_summary(frame=frame, grouping_variable="exposure_flag", outcome_variables=["duration_a", "duration_b"],
                                      typed_cohort_input="artifact:analysis_cohort", source_cohort=tmp_path / "synthetic.parquet",
                                      out_dir=tmp_path / "out", run_dir=tmp_path)
    assert not (tmp_path / "out/continuous_outcome_summary.csv").exists()
