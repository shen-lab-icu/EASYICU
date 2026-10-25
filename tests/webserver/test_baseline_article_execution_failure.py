"""Synthetic replay of host shaping, signed grouped fitting, and safe refusals."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.baseline_association_runtime import BaselineAssociationAuthorityError
from easyicu.research_agent.execution.runners.baseline_association_executor import run_baseline_step
from easyicu.research_agent.pipeline import _shape_fresh_plan
from easyicu.webserver import agent_pipeline_runs as web
from tests.support.baseline_association import baseline_plan_attempt, grouped_case


def _article(tmp_path):
    context, runtime, catalog = grouped_case(tmp_path, source="age", middle_band=False)
    runtime = runtime.for_planning_context(context=context, run_dir=tmp_path, cohort_path=catalog)
    draft = baseline_plan_attempt(context)[0].output
    runtime = runtime.seal_for_plan(draft)
    shaped = _shape_fresh_plan(
        SimpleNamespace(_scientific_runtime_authorities=runtime, _enable_publication_figure_skill=True, _max_total_steps=20),
        plan=draft, context=context, agent_context=context, long_trajectory_bound=False, findings=[],
    )
    return context, runtime, shaped


def test_grouped_article_survives_host_shaping_and_runs_the_signed_column(tmp_path):
    import statsmodels.formula.api as smf
    from easyicu.research_agent.contracts.exposure_group_rules import evaluate_grouping, read_grouping_rules
    from easyicu.research_agent.orchestration.exposure_grouping_phase import planning_run_groupings
    from easyicu.research_agent.planning.final_plan_shape import validate_final_plan_shape
    from easyicu.research_agent.execution.runners.selection import resolve_standard_executor
    import pyarrow as pa

    context, runtime, shaped = _article(tmp_path)
    # The ordinary article shaper appends two source-bound display steps.
    assert len(shaped.steps) == 9
    bound, _ = runtime.bind_plan(shaped)
    runtime.validate_plan(bound)
    validate_final_plan_shape(bound)
    authority = runtime.current_case
    assert authority.exposure_column == context.primary_exposure
    assert authority.required_adjustment_columns == ("sex",)
    columns = {
        "table:cohort_flow": ["step_order", "predicate_kind", "n_before", "n_excluded", "n_remaining"],
        "table:measurement_audit_missingness": ["variable", "label", "n_total", "eligible_n", "not_applicable_n",
                                                "measured_one_n", "value_missing_n", "value_missing_pct", "indicator_semantics"],
        "table:measurement_audit_process": ["variable", "n_total", "eligible_n", "measured_one_n", "repeat_measured_n"],
    }
    for step in bound.steps[-2:]:
        bindings = {key: {
            "declared_kind": "table", "evidence_kind": "table", "product": key.partition(":")[2],
            "product_contract": {"columns": columns[key]}, "consumption_contract": {"mode": "all_rows"},
        } for key in step.inputs}
        decision = resolve_standard_executor(step, plan=bound, current_case_scientific_runtime_authority=authority,
                                             scientific_runtime_projection_sha256="a" * 64, resolved_bindings=bindings)
        assert decision.claimed_by is not None
        compile(decision.render_selection().code, "<synthetic-display>", "exec")
    frame = pd.DataFrame([
        {"age": age, "sex": sex, "death": index < events}
        for age, sex, events in [(18, 0, 4), (18, 1, 8), (24, 0, 12), (24, 1, 16)]
        for index in range(40)
    ])
    groups = planning_run_groupings(tmp_path, cohort_sha256=hashlib.sha256((tmp_path / "cohort.parquet").read_bytes()).hexdigest())
    rules = read_grouping_rules(groups.columns[0].parameters)
    frame[context.primary_exposure] = evaluate_grouping(rules, pa.Table.from_pandas(frame)).to_pylist()
    frame = frame.drop(columns="age")  # A source-column read cannot pass this replay.
    primary = run_baseline_step(step_id="adjusted_association", plan=bound, authority=authority,
                                runtime_projection_sha256="a" * 64, frame=frame, output_dir=tmp_path / "primary")
    estimates = pd.read_csv(tmp_path / "primary/adjusted_association_estimates.csv")
    independent = smf.logit(f"event ~ C({context.primary_exposure}, Treatment(reference=2)) + sex",
                            data=frame.assign(event=frame.death.astype(int))).fit(disp=False)
    assert estimates.loc[estimates.is_primary_contrast, "estimate"].item() == pytest.approx(np.exp(independent.params.iloc[1]))
    assert primary["status"] == "ok"
    assert set(estimates.exposure) == {authority.exposure_column}
    risk = run_baseline_step(step_id="absolute_risk_context", plan=bound, authority=authority,
                             runtime_projection_sha256="a" * 64, frame=frame, primary_table=estimates,
                             output_dir=tmp_path / "risk")
    assert risk["status"] == "ok"


@pytest.mark.parametrize("change", ["extra_model", "missing_core", "reordered_core"])
def test_a_roster_refusal_retains_its_owner_phase_and_step(tmp_path, change):
    context, runtime, _ = _article(tmp_path)
    draft = baseline_plan_attempt(context)[0].output
    if change == "extra_model":
        draft.steps.append(draft.steps[3].model_copy(update={"step_id": "unexpected_model", "planned_analysis_role": "secondary"}))
    elif change == "missing_core":
        draft.steps.pop(2)
    else:
        draft.steps[1:3] = reversed(draft.steps[1:3])
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        runtime.bind_plan(draft)
    wrapper = RuntimeError("synthetic private message")
    wrapper.__cause__ = caught.value
    code = web._pipeline_failure_code(wrapper)
    assert code == "research_pipeline_scientific_authority_refused"
    path = web._record_pipeline_failure(wrapper_dir=tmp_path, study={"id": "synthetic"}, provider={},
                                       exc=wrapper, code=code, execution_retry_id=None)
    receipt = json.loads((tmp_path / path).read_text())
    diagnostic = receipt["typed_failure"]
    assert diagnostic["owner"] == "easyicu.current_case_scientific_authority_v1"
    assert diagnostic["reason_code"] == "baseline_association_step_roster_mismatch"
    assert diagnostic["exception_type"] == "BaselineAssociationAuthorityError"
    assert diagnostic["step_id"] == "plan"
    assert diagnostic["runner_kind"] == "not_started"
    assert diagnostic["phase"] == "plan_validation"
    assert diagnostic["mismatch_fields"] == ["step_roster"]
    assert receipt["exception_types"] == ["BaselineAssociationAuthorityError"]
    assert "synthetic private message" not in json.dumps(receipt)


@pytest.mark.parametrize("change", ["raw_input", "model", "method", "output", "panel", "consumption", "duplicate", "duplicate_id", "source_missing", "source_ambiguous"])
def test_article_companions_cannot_expand_the_signed_analysis(tmp_path, change):
    _, runtime, shaped = _article(tmp_path)
    display = shaped.steps[-1]
    if change == "raw_input":
        display.inputs.append("age")
    elif change == "model":
        display.model_requirements = shaped.steps[3].model_requirements
    elif change == "method":
        display.method = "unreviewed_method"
    elif change == "output":
        display.expected_outputs = ["table:unreviewed_estimate"]
    elif change == "panel":
        display.figure_panels = []
    elif change == "consumption":
        display.input_consumption_contracts = []
    elif change == "duplicate":
        shaped.steps.append(display.model_copy(update={"step_id": "extra_display"}))
    elif change == "duplicate_id":
        display.step_id = shaped.steps[0].step_id
    elif change == "source_missing":
        shaped.steps[2].expected_outputs.remove("table:measurement_audit_process")
    else:
        shaped.steps[1].expected_outputs.append("table:measurement_audit_process")
    with pytest.raises(BaselineAssociationAuthorityError):
        runtime.bind_plan(shaped)


def test_first_result_does_not_acquire_article_companions(tmp_path):
    from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions

    context, runtime, article = _article(tmp_path)
    plan = baseline_plan_attempt(context, planning_contract_context=first_result_instructions())[0].output
    plan.steps.append(article.steps[-1])
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        runtime.bind_plan(plan)
    assert caught.value.reason_code == "baseline_association_step_roster_mismatch"


@pytest.mark.parametrize("step_id,change", [
    ("adjusted_association", "method"), ("adjusted_association", "capability"),
    ("adjusted_association", "reference"), ("adjusted_association", "output"),
    ("adjusted_association", "inputs"), ("absolute_risk_context", "method"),
    ("adjusted_association", "outcome"), ("absolute_risk_context", "outcome"),
    ("adjusted_association", "consumption"), ("absolute_risk_context", "consumption"),
])
def test_each_signed_binding_refusal_is_typed(tmp_path, step_id, change):
    _, runtime, draft = _article(tmp_path)
    bound, _ = runtime.bind_plan(draft)
    step = next(step for step in bound.steps if step.step_id == step_id)
    if change == "method":
        step.method = "unreviewed_method"
    elif change == "capability":
        step.scientific_capability = None
    elif change == "reference":
        step.icu_rule_refs = []
    elif change == "output":
        step.expected_outputs = ["table:other_result"]
    elif change == "inputs":
        step.inputs.append("unrequested_column")
    elif change == "outcome":
        step.runtime_outcome_contract = None
    else:
        step.input_consumption_contracts = []
    with pytest.raises(BaselineAssociationAuthorityError) as caught:
        runtime.current_case.validate_plan(bound)
    diagnostic = web._safe_pipeline_typed_failure(caught.value)
    assert diagnostic["step_id"] == step_id
    assert diagnostic["phase"] == "plan_validation"
    assert diagnostic["runner_kind"] == "not_started"
    assert diagnostic["mismatch_fields"]
