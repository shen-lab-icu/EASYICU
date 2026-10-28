"""Synthetic treatment-confounder feedback and governed g-method boundaries."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
import statsmodels.api as sm

from easyicu.research_agent.contracts.time_varying_exposure import (
    G_METHOD_OUTPUTS, GMethodError, GMethodSpec, g_method_for_step, g_method_serialization,
)
from easyicu.research_agent.methods.msm import fit_longitudinal_msm
from easyicu.research_agent.planning.target_trial_spec import require_g_method_source


def specification(**updates):
    body = dict(
        estimator="longitudinal_msm", database="miiv", identity_column="person",
        period_column="period", treatment_column="treatment", outcome_column="event",
        confounder_history=["severity"], baseline_columns=["baseline"], treatment_lags=[1],
        confounder_lags=[0, 1], intervals=3, interval_hours=6, censoring_policy="complete_followup",
        censoring_column=None, ps_bounds=(0.001, 0.999), truncation_quantiles=(0.01, 0.99),
        n_bootstrap=None, random_seed=17, require_exchangeability=True,
        require_correct_specification=True, require_consistency=True,
    )
    return GMethodSpec(**(body | updates))


def feedback_panel(n=6000, seed=1709):
    """Fresh severity shocks affect treatment and death; earlier treatment affects severity."""
    rng = np.random.default_rng(seed)
    baseline = rng.normal(size=n)
    previous = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    rows = []
    for period in range(3):
        shock = rng.binomial(1, 0.5, n)
        severity = shock + 0.6 * previous
        treatment = rng.binomial(1, expit(-1.6 + 2.0 * severity + 0.25 * baseline))
        event = rng.binomial(1, expit(-2.0 - 0.8 * treatment + 1.8 * shock))
        rows.append(pd.DataFrame(dict(person=np.arange(n)[alive], period=period,
            treatment=treatment[alive], event=event[alive], severity=severity[alive],
            baseline=baseline[alive])))
        alive &= event == 0
        previous = treatment
    # Integrating the independent binary severity shock gives the exact marginal hazard OR.
    risk0 = (expit(-2.0) + expit(-0.2)) / 2
    risk1 = (expit(-2.8) + expit(-1.0)) / 2
    return pd.concat(rows, ignore_index=True), float(logit(risk1) - logit(risk0))


def source(spec, **changes):
    return dict(schema_version="easyicu.g_methods_source/1", producer="time_varying",
        identity_unit="person", identity_column=spec.identity_column,
        database=spec.database, structure="person_period", columns=list(spec.required_columns),
        pre_treatment_columns=[*spec.confounder_history, *spec.baseline_columns], history_columns=list(spec.confounder_history),
        interval_hours=spec.interval_hours, time_zero=spec.time_zero,
        source_sha256="a" * 64, specification_sha256=spec.sha256) | changes


def test_msm_recovers_known_effect_under_treatment_confounder_feedback():
    panel, truth = feedback_panel()
    spec = specification(truncation_quantiles=None)
    result = fit_longitudinal_msm(panel, spec)
    estimate = result.estimates.set_index("term").loc["treatment", "coefficient"]
    initial_severity = panel.loc[panel.period.eq(0)].set_index("person").severity
    baseline_design = panel[["treatment", "baseline", "period"]].assign(
        initial_severity=panel.person.map(initial_severity))
    baseline = sm.GLM(panel.event, sm.add_constant(baseline_design),
        family=sm.families.Binomial()).fit().params["treatment"]
    assert abs(estimate - truth) < 0.12
    assert abs(baseline - truth) > 0.35
    assert abs(estimate - truth) < abs(baseline - truth) / 3
    assert len(result.diagnostics) == 3
    assert (result.diagnostics.weight_ess > 0).all()
    assert result.receipt["variance_estimator"] == "person_cluster_sandwich"
    assert result.receipt["claim_ceiling"] == "analysis_only"


def test_cumulative_weights_and_truncation_are_auditable():
    panel, _ = feedback_panel(n=700)
    raw = fit_longitudinal_msm(panel, specification(truncation_quantiles=None))
    clipped = fit_longitudinal_msm(panel, specification(truncation_quantiles=(0.1, 0.9)))
    for _, group in raw.weights.groupby("person"):
        np.testing.assert_allclose(group.treatment_weight, group.increment.cumprod())
    for period, weights in clipped.weights.groupby("period"):
        untrimmed = raw.weights.loc[raw.weights.period.eq(period), "combined_weight"].to_numpy()
        bounds = np.quantile(untrimmed, [0.1, 0.9])
        np.testing.assert_allclose(weights.analysis_weight, np.clip(untrimmed, *bounds))
        row = clipped.diagnostics.set_index("period").loc[period]
        assert row.trunc_low == pytest.approx(bounds[0])
        assert row.trunc_high == pytest.approx(bounds[1])
        assert row.truncated_n > 0
        assert row.weight_ess == pytest.approx(weights.analysis_weight.sum() ** 2 / (weights.analysis_weight ** 2).sum())
        assert 0 < row.weight_ess <= len(weights)
        assert row.weight_mean == pytest.approx(weights.analysis_weight.mean())
        assert row.weight_min == pytest.approx(weights.analysis_weight.min())
        assert row.weight_max == pytest.approx(weights.analysis_weight.max())
    assert clipped.receipt["truncation_quantiles"] == [0.1, 0.9]
    assert not np.allclose(raw.weights.analysis_weight, clipped.weights.analysis_weight)


@pytest.mark.parametrize("change,code", [
    ({"structure": "one_row_per_stay"}, "g_methods_person_period_unavailable"),
    ({"history_columns": []}, "g_methods_confounder_history_unavailable"),
    ({"pre_treatment_columns": []}, "g_methods_history_timing_unavailable"),
    ({"database": "eicu"}, "g_methods_source_database_mismatch"),
])
def test_source_capability_refuses_missing_or_mistimed_history(change, code):
    spec = specification()
    with pytest.raises(GMethodError) as error:
        require_g_method_source(spec, source(spec, **change), database="miiv")
    assert error.value.code == code


def test_missing_history_never_becomes_a_baseline_model():
    panel, _ = feedback_panel(n=300)
    with pytest.raises(GMethodError) as error:
        fit_longitudinal_msm(panel.drop(columns="severity"), specification())
    assert error.value.code == "g_methods_confounder_history_unavailable"


def test_positivity_is_a_typed_interval_refusal():
    panel, _ = feedback_panel(n=500)
    panel["treatment"] = (panel.severity > 0.5).astype(int)
    with pytest.raises(GMethodError) as error:
        fit_longitudinal_msm(panel, specification(ps_bounds=(0.15, 0.85)))
    assert "positivity" in error.value.code
    assert error.value.period == 0


def test_method_component_is_available_and_explicit_about_scope():
    from easyicu.research_agent.method_skills import method_skill_catalog
    row = next(row for row in method_skill_catalog(enabled=True)["components"]
        if row["id"] == "causal_emulation.g_methods")
    assert row["implementation"] == "deterministic"
    assert set(row["kernel_modules"]) == {"msm", "gformula"}
    assert "multi-interval g-formula is not implemented" in row["description"].lower()


def test_typed_method_serialization_preserves_all_reviewed_choices():
    from easyicu.research_agent.schema import AnalysisStep
    spec = specification()
    step = AnalysisStep(step_id="msm", intent="Estimate a longitudinal causal contrast",
        planned_analysis_role="primary", scientific_action_id="causal_emulation.g_methods",
        method=g_method_serialization(spec), inputs=["table:person_period", *spec.required_columns],
        expected_outputs=list(G_METHOD_OUTPUTS))
    assert g_method_for_step(step) == spec
    damaged = step.model_copy(update={"method": "g_methods({})"})
    with pytest.raises(GMethodError):
        g_method_for_step(damaged)
    assert json.loads(step.method.removeprefix("g_methods(")[:-1])["treatment_lags"] == [1]


def test_point_gformula_reports_rd_and_rr_bootstrap_intervals():
    from easyicu.research_agent.methods.gformula import gformula_ate
    rng = np.random.default_rng(81)
    x = rng.normal(size=(500, 1))
    a = rng.binomial(1, expit(0.3 * x[:, 0]))
    y = rng.binomial(1, expit(-1 + 0.7 * a + 0.4 * x[:, 0]))
    result = gformula_ate(x, a, y, n_bootstrap=30, random_state=4,
        require_exchangeability=True, require_correct_specification=True)
    assert result.risk_ratio == pytest.approx(result.mu1_mean / result.mu0_mean)
    assert 0 < result.rr_ci_low < result.rr_ci_high
    assert result.rr_ci_low < result.risk_ratio < result.rr_ci_high
    assert result.to_json()["risk_ratio"] == result.risk_ratio


@pytest.mark.parametrize("database", ["miiv", "mimic", "eicu", "hirid", "aumc", "sic"])
def test_no_database_gets_an_implicit_history(database):
    spec = specification(database=database)
    with pytest.raises(GMethodError) as error:
        require_g_method_source(spec, None, database=database)
    assert error.value.code == "g_methods_person_period_unavailable"


@pytest.mark.parametrize("skill_id", ["target-trial-emulation", "time-varying-exposure-survival-study"])
def test_workflow_has_an_optional_connected_msm_history_branch(skill_id):
    from easyicu.webserver.research_workflows import research_workflow_catalog
    row = next(item for item in research_workflow_catalog(enabled=True)["items"] if item["id"] == skill_id)
    nodes = {node["id"]: node for node in row["steps"]}
    assert nodes["longitudinal_msm"]["optional"] is True
    assert nodes["g_method_history"]["optional"] is True
    assert any(edge["source"] == "g_method_history" and edge["target"] == "longitudinal_msm" for edge in row["edges"])
    assert any(edge["source"] == "longitudinal_msm" and edge["target"] == "report" for edge in row["edges"])


def g_step(spec):
    from easyicu.research_agent.schema import AnalysisStep
    return AnalysisStep(step_id="msm", intent="Fit a reviewed longitudinal MSM",
        planned_analysis_role="primary", scientific_action_id="causal_emulation.g_methods",
        method=g_method_serialization(spec), inputs=["cohort:person_period", *spec.required_columns],
        expected_outputs=list(G_METHOD_OUTPUTS))


def test_review_gate_and_executor_share_the_exact_specification(tmp_path):
    from easyicu.research_agent.acquisition.time_varying_execution_input import prepare_g_method_execution_input
    from easyicu.research_agent.execution.runners.selection import select_standard_executor
    from easyicu.research_agent.execution.runners.time_varying_executor import run_time_varying_g_method
    from easyicu.research_agent.planning.target_trial_spec import require_g_method_plan
    from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext
    from easyicu.research_agent.contracts.time_varying_exposure import G_METHOD_SOURCE_KEY
    spec = specification()
    step = g_step(spec)
    plan = AnalysisPlan(research_question="Does a changing treatment affect events?", analysis_type="causal_inference", steps=[step])
    context = ResearchContext(research_question=plan.research_question, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=500,
            provenance={G_METHOD_SOURCE_KEY: source(spec)}))
    require_g_method_plan(context, plan)
    selection = select_standard_executor(step, plan=plan)
    assert selection.analysis_kind == "time_varying_g_methods"
    assert selection.consumed_input_keys == ("cohort:person_period",)
    compile(selection.code, "g_method_executor.py", "exec")
    panel, _ = feedback_panel(n=500)
    prepared = prepare_g_method_execution_input(panel, specification=spec, source_receipt=source(spec))
    path = tmp_path / "person_period.parquet"
    prepared.to_parquet(path)
    # Source metadata survives the existing digest-bound parquet loading path.
    loaded = pd.read_parquet(path)
    summary = run_time_varying_g_method(frame=loaded, specification=spec, out_dir=tmp_path / "out")
    assert set(summary["output_files"]) == set(G_METHOD_OUTPUTS)
    assert all((tmp_path / "out" / filename).is_file() for filename in summary["output_files"].values())
    receipt = summary["scientific_runtime_receipt"]
    assert receipt["specification_sha256"] == spec.sha256
    assert "conditional" in receipt["estimand"]
    assert receipt["source_construction"]["source_sha256"] == "a" * 64
    assert receipt["publication_ready"] is False
    table = pd.read_csv(tmp_path / "out/g_method_weight_diagnostics.csv")
    assert {"period", "weight_ess", "trunc_low", "trunc_high", "overlap_status"} <= set(table)
    checklist = (tmp_path / "out/g_method_checklist.md").read_text()
    assert "analysis_only" in checklist and "exchangeability" in checklist
    assert "person" not in table.columns


def test_review_and_execution_refuse_an_unbound_or_baseline_only_source(tmp_path):
    from easyicu.research_agent.execution.runners.time_varying_executor import run_time_varying_g_method
    panel, _ = feedback_panel(n=200)
    with pytest.raises(GMethodError):
        run_time_varying_g_method(frame=panel, specification=specification(), out_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_invalid_g_method_never_falls_through_to_coder():
    from easyicu.research_agent.execution.runners.selection import select_standard_executor
    from easyicu.research_agent.schema import AnalysisPlan
    step = g_step(specification()).model_copy(update={"method": "baseline_logistic"})
    plan = AnalysisPlan(research_question="Causal longitudinal contrast", analysis_type="causal_inference", steps=[step])
    with pytest.raises(GMethodError) as error:
        select_standard_executor(step, plan=plan)
    assert error.value.code == "g_methods_declaration_required"


def test_msm_primary_capability_names_its_actual_owner_and_analysis_ceiling():
    from easyicu.research_agent.planning.capability_registry import resolve_primary_capability
    from easyicu.research_agent.schema import AnalysisPlan
    step = g_step(specification())
    plan = AnalysisPlan(research_question="Causal longitudinal contrast", analysis_type="causal_inference", steps=[step])
    verdict = resolve_primary_capability(analysis_type="causal_inference", plan=plan)
    assert verdict.capability.capability_id == "causal_g_methods_v1"
    assert verdict.capability.primary_analysis == "deterministic"
    assert verdict.capability.scientific_validation == "analysis_only"
    assert verdict.owner_claimed is True


@pytest.mark.parametrize("changes", [
    {"treatment_lags": []}, {"treatment_lags": [0]}, {"treatment_lags": [1, 1]},
    {"confounder_lags": [1]}, {"confounder_lags": [1, 0]}, {"intervals": 1},
    {"treatment_lags": [3]}, {"ps_bounds": [0.8, 0.2]}, {"ps_bounds": [0, 1]},
    {"truncation_quantiles": [0.9, 0.1]}, {"interval_hours": 0},
    {"censoring_policy": "stabilized_ipcw"}, {"censoring_column": "censored"},
    {"outcome_column": "treatment"}, {"require_exchangeability": False},
    {"require_correct_specification": False}, {"require_consistency": False},
    {"n_bootstrap": 0}, {"n_bootstrap": 30}, {"random_seed": -1},
    {"baseline_columns": ["intercept"]},
])
def test_scientific_choices_are_closed_and_explicit(changes):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        specification(**changes)


@pytest.mark.parametrize("defect,code", [
    ("duplicate", "g_methods_person_period_invalid"),
    ("gap", "g_methods_person_period_invalid"),
    ("missing_value", "g_methods_history_incomplete"),
    ("nonbinary", "g_methods_nonbinary"),
    ("baseline_changes", "g_methods_baseline_changed"),
    ("unexplained_dropout", "g_methods_censoring_unavailable"),
    ("event_before_last", "g_methods_followup_invalid"),
])
def test_person_period_integrity_fails_closed(defect, code):
    panel, _ = feedback_panel(n=400)
    survivor = next(person for person, rows in panel.groupby("person") if len(rows) == 3)
    index = panel.loc[panel.person.eq(survivor)].index
    if defect == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
    elif defect == "gap":
        panel = panel.drop(index[1])
    elif defect == "missing_value":
        panel.loc[index[1], "severity"] = np.nan
    elif defect == "nonbinary":
        panel.loc[index[1], "treatment"] = 2
    elif defect == "baseline_changes":
        panel.loc[index[1], "baseline"] += 2
    elif defect == "unexplained_dropout":
        panel = panel.drop(index[2])
        panel.loc[index[1], "event"] = 0
    else:
        panel.loc[index[0], "event"] = 1
    with pytest.raises(GMethodError) as error:
        fit_longitudinal_msm(panel, specification())
    assert error.value.code == code


def test_censoring_weights_accumulate_only_after_observed_intervals():
    from easyicu.research_agent.methods.msm import stabilized_iptw
    panel, _ = feedback_panel(n=1700)
    panel["censored"] = 0
    rng = np.random.default_rng(19)
    at0 = panel.loc[panel.period.eq(0) & panel.event.eq(0)]
    censor = rng.binomial(1, expit(-1.8 + 1.2 * at0.severity))
    panel.loc[at0.index, "censored"] = censor
    dropped = set(at0.loc[censor == 1, "person"])
    panel = panel.loc[~(panel.person.isin(dropped) & panel.period.gt(0))].copy()
    spec = specification(censoring_policy="stabilized_ipcw", censoring_column="censored", truncation_quantiles=None)
    result = fit_longitudinal_msm(panel, spec)
    reference = stabilized_iptw([1 - censor],
        [np.column_stack([at0.baseline, np.zeros(len(at0)), at0.severity, np.zeros(len(at0)), at0.treatment])],
        [np.column_stack([at0.baseline, np.zeros(len(at0)), at0.treatment])],
        random_state=spec.random_seed, ps_trim=spec.ps_bounds,
        require_sequential_exchangeability=True, require_correct_history_encoding=True)
    expected = dict(zip(at0.person, reference.weights))
    assert result.weights.loc[result.weights.period.eq(0), "censor_weight"].eq(1).all()
    later = result.weights.loc[result.weights.period.gt(0)]
    np.testing.assert_allclose(later.censor_weight, later.person.map(expected))
    np.testing.assert_allclose(result.weights.combined_weight,
        result.weights.treatment_weight * result.weights.censor_weight)
    assert result.diagnostics.iloc[0].censor_model == "stabilized_ipcw"
    assert result.diagnostics.iloc[0].censor_ps_min < 1


@pytest.mark.parametrize("change,code", [
    ({"producer": "planner"}, "g_methods_person_period_unavailable"),
    ({"source_sha256": "broken"}, "g_methods_source_receipt_mismatch"),
    ({"specification_sha256": "b" * 64}, "g_methods_source_receipt_mismatch"),
    ({"columns": []}, "g_methods_source_columns_unavailable"),
    ({"interval_hours": 12}, "g_methods_history_timing_unavailable"),
    ({"time_zero": "hospital_discharge"}, "g_methods_history_timing_unavailable"),
    ({"identity_unit": "icu_stay"}, "g_methods_person_identity_unavailable"),
    ({"identity_column": "another_person"}, "g_methods_person_identity_unavailable"),
    ({"patient_rows": [[1, 2]]}, "g_methods_source_receipt_mismatch"),
    ({"columns": None}, "g_methods_source_receipt_mismatch"),
])
def test_source_provenance_cannot_be_renamed_or_reused(change, code):
    spec = specification()
    with pytest.raises(GMethodError) as error:
        require_g_method_source(spec, source(spec, **change), database=spec.database)
    assert error.value.code == code


def test_point_gformula_runtime_uses_its_kernel_and_outputs_both_contrasts(tmp_path):
    from easyicu.research_agent.contracts.time_varying_exposure import G_METHOD_SOURCE_KEY
    from easyicu.research_agent.execution.runners.time_varying_executor import run_time_varying_g_method
    rng = np.random.default_rng(92)
    n = 400
    panel = pd.DataFrame(dict(person=np.arange(n), period=0,
        baseline=rng.normal(size=n), severity=rng.normal(size=n)))
    panel["treatment"] = rng.binomial(1, expit(0.2 * panel.severity))
    panel["event"] = rng.binomial(1, expit(-1 + .7 * panel.treatment + .3 * panel.severity))
    spec = specification(estimator="point_gformula", intervals=1, treatment_lags=[], confounder_lags=[0], truncation_quantiles=None, n_bootstrap=30)
    panel.attrs[G_METHOD_SOURCE_KEY] = source(spec)
    summary = run_time_varying_g_method(frame=panel, specification=spec, out_dir=tmp_path / "point")
    estimates = pd.read_csv(tmp_path / "point/g_method_estimates.csv").set_index("term")
    assert set(estimates.index) == {"risk_difference", "risk_ratio"}
    assert np.isfinite(estimates.to_numpy()).all()
    receipt = summary["scientific_runtime_receipt"]
    assert receipt["n_successful"] == spec.n_bootstrap
    assert receipt["claim_ceiling"] == "analysis_only"
    assert receipt["risk_ratio"] == pytest.approx(receipt["mu1_mean"] / receipt["mu0_mean"])


def test_planner_action_binds_kernels_without_coder_resource_slot_eviction():
    from easyicu.research_agent.planning.scientific_action_catalog import scientific_action_for_id
    from easyicu.research_agent.contracts.method_kernels import CURATED_METHOD_KERNELS
    action = scientific_action_for_id(analysis_type="causal_inference", action_id="causal_emulation.g_methods")
    assert action.execution_mode == "host_owned"
    assert action.adapter_status == "full_action"
    assert action.runtime_contract.standard_executor == "time_varying_g_methods"
    assert {name.rsplit(".", 1)[-1] for name in action.kernel_imports} == {"msm", "gformula"}
    assert {"msm", "gformula"}.isdisjoint(kernel.module for kernel in CURATED_METHOD_KERNELS)


def test_rendered_executor_consumes_digest_bound_history_and_rejects_tampering(tmp_path, monkeypatch):
    import hashlib
    from easyicu.research_agent.acquisition.time_varying_execution_input import prepare_g_method_execution_input
    from easyicu.research_agent.execution.runners.selection import select_standard_executor
    from easyicu.research_agent.execution.runners.typed_input_binding import TypedInputBindingError
    from easyicu.research_agent.schema import AnalysisPlan
    spec = specification()
    panel, _ = feedback_panel(n=450)
    prepared = prepare_g_method_execution_input(panel, specification=spec, source_receipt=source(spec))
    bound = tmp_path / "history.parquet"
    prepared.to_parquet(bound, index=False)
    digest = hashlib.sha256(bound.read_bytes()).hexdigest()
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps(dict(inputs={"cohort:person_period": dict(
        relative_path=bound.name, sha256=digest, declared_kind="dataset", product="person_period",
        evidence_kind="table", product_contract=dict(columns=list(prepared), row_count=len(prepared)))})))
    out = tmp_path / "out"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest))
    monkeypatch.setenv("STEP_OUT_DIR", str(out))
    step = g_step(spec)
    plan = AnalysisPlan(research_question="Longitudinal causal effect", analysis_type="causal_inference", steps=[step])
    selection = select_standard_executor(step, plan=plan)
    exec(compile(selection.code, "msm_executor", "exec"), {})
    summary = json.loads((out / "step_summary.json").read_text())
    assert summary["n_total"] == 450 and summary["n_interval_rows"] == len(prepared)
    assert summary["scientific_runtime_receipt"]["source_construction"]["source_sha256"] == "a" * 64
    prepared.loc[0, "severity"] += 0.25
    prepared.to_parquet(bound, index=False)
    with pytest.raises(TypedInputBindingError) as error:
        exec(compile(selection.code, "msm_executor", "exec"), {})
    assert error.value.reason_code == "digest_mismatch"


def test_scientific_review_stops_before_offering_missing_history_for_approval():
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext
    step = g_step(specification())
    plan = AnalysisPlan(research_question="Longitudinal causal effect", analysis_type="causal_inference", steps=[step])
    context = ResearchContext(research_question=plan.research_question, variables=[],
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=10))
    with pytest.raises(GMethodError) as error:
        build_plan_scientific_review(context=context, plan=plan)
    assert error.value.code == "g_methods_person_period_unavailable"


@pytest.mark.parametrize("updates", [
    {"expected_outputs": ["table:g_method_estimates"]},
    {"inputs": ["severity", "baseline", "treatment", "event", "person", "period"]},
    {"planned_analysis_role": "auxiliary"},
    {"scientific_capability": "association_adjusted_v1"},
    {"inputs": ["cohort:person_period", "treatment", "event", "person", "period", "baseline"]},
    {"cohort_definition_spec": {"identity_column": "person"}},
])
def test_incomplete_or_foreign_step_contracts_are_not_owned(updates):
    from easyicu.research_agent.planning.target_trial_spec import require_g_method_step
    with pytest.raises(GMethodError) as error:
        require_g_method_step(g_step(specification()).model_copy(update=updates))
    assert error.value.code == "g_methods_step_contract_invalid"


def test_reporting_checklist_recognizes_actual_g_method_products():
    from types import SimpleNamespace
    from easyicu.research_agent.reporting.reporting_checklist import build_strobe_checklist
    report = build_strobe_checklist(evidence_records=[SimpleNamespace(evidence_id=name) for name in
        ("g_method_weight_diagnostics", "g_method_receipt", "g_method_estimates")], bound_manuscript="")
    items = {item.item_id: item for item in report.items}
    for item_id in ("9", "12a", "16"):
        assert items[item_id].status == "addressed"
    assert items["12e"].status == "open"  # No sensitivity analysis was executed.


def test_linear_gformula_does_not_label_continuous_means_as_risks():
    from easyicu.research_agent.methods.gformula import gformula_ate
    rng = np.random.default_rng(35)
    x = rng.normal(size=(180, 1))
    a = rng.binomial(1, 0.5, len(x))
    y = 10 + .5 * a + .2 * x[:, 0] + rng.normal(size=len(x))
    result = gformula_ate(x, a, y, outcome_model="linear", n_bootstrap=20,
        require_exchangeability=True, require_correct_specification=True)
    assert result.risk_ratio is None and result.rr_ci_low is None and result.rr_ci_high is None


def test_a_typed_g_method_cannot_borrow_a_different_scientific_action():
    step = g_step(specification()).model_copy(update={"scientific_action_id": "association.adjusted_association"})
    with pytest.raises(GMethodError) as error:
        g_method_for_step(step)
    assert error.value.code == "g_methods_declaration_required"


@pytest.mark.parametrize("identity", [None, " "])
def test_missing_person_identity_is_not_an_analysis_unit(identity):
    panel, _ = feedback_panel(n=100)
    panel["person"] = panel.person.astype(object)
    panel.loc[0, "person"] = identity
    with pytest.raises(GMethodError) as error:
        fit_longitudinal_msm(panel, specification())
    assert error.value.code == "g_methods_person_period_invalid"


def test_array_kernel_multiplies_weights_from_every_visit(monkeypatch):
    from easyicu.research_agent.methods import msm
    # Known propensity oracle isolates the sequential weight contract from model fitting.
    monkeypatch.setattr(msm, "_fit_visit_propensity", lambda features, treatment, **kwargs: features[:, 0])
    result = msm.stabilized_iptw(
        [np.array([0, 1, 0, 1]), np.array([1, 1, 0, 0])],
        [np.array([[.25], [.3], [.6], [.7]]), np.array([[.2], [.4], [.5], [.8]])],
        [None, None], ps_trim=(.1, .9), require_sequential_exchangeability=True,
        require_correct_history_encoding=True,
    )
    np.testing.assert_allclose(result.weights, [5 / 3, 25 / 12, 5 / 4, 25 / 14])
    assert result.per_visit[0].cumul_mean == pytest.approx(np.mean([2 / 3, 5 / 3, 5 / 4, 5 / 7]))


@pytest.mark.parametrize("updates", [
    {"intervals": 2}, {"treatment_lags": [1]}, {"confounder_lags": [0, 1]},
    {"censoring_policy": "stabilized_ipcw", "censoring_column": "censored"},
    {"truncation_quantiles": [.01, .99]}, {"n_bootstrap": None},
])
def test_point_gformula_never_claims_longitudinal_or_unimplemented_options(updates):
    from pydantic import ValidationError
    point = dict(estimator="point_gformula", intervals=1, treatment_lags=[],
        confounder_lags=[0], truncation_quantiles=None, n_bootstrap=20)
    with pytest.raises(ValidationError):
        specification(**(point | updates))


@pytest.mark.parametrize("field,value", [
    ("primary_exposure", "different_treatment"), ("target_outcome", "different_event"),
])
def test_g_method_keeps_the_research_questions_treatment_and_outcome(field, value):
    from easyicu.research_agent.contracts.time_varying_exposure import G_METHOD_SOURCE_KEY
    from easyicu.research_agent.planning.target_trial_spec import require_g_method_plan
    from easyicu.research_agent.schema import AnalysisPlan, CohortDescriptor, ResearchContext
    spec = specification()
    plan = AnalysisPlan(research_question="Longitudinal causal effect", analysis_type="causal_inference", steps=[g_step(spec)])
    context = ResearchContext(research_question=plan.research_question, variables=[],
        primary_exposure="treatment", target_outcome="event",
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=10,
            provenance={G_METHOD_SOURCE_KEY: source(spec)}))
    with pytest.raises(GMethodError) as error:
        require_g_method_plan(context.model_copy(update={field: value}), plan)
    assert error.value.code == "g_methods_question_mismatch"
