"""A sensitivity must preserve which term, population, and model it changes."""

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.functional_form import FunctionalFormSpec
from easyicu.research_agent.execution.runners.landmark_spline_functional_form_executor import (
    landmark_spline_functional_form_executor_owns_step,
    run_landmark_spline_functional_form,
)
from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError, ProgressivePlanSkeleton,
)
from easyicu.research_agent.schema import AnalysisStep

from .test_current_case_scientific_runtime_authority import _authority, _e2_plan
from ..planning.progressive_planner_fixtures import _context, _payload


def _form(target):
    return FunctionalFormSpec(target_column=target, knot_quantiles=(0.1, 0.5, 0.9))


def _functional_payload(target="age_years"):
    payload = _payload()
    step = payload["steps"][5]
    step["custom_method"] = "restricted_cubic_spline_sensitivity"
    step["functional_form_spec"] = _form(target).model_dump(mode="json")
    return payload


def test_compiler_preserves_explicit_target_not_a_name_in_the_sensitivity_id():
    payload = _functional_payload()
    payload["steps"][5]["sensitivity_spec_ids"] = ["not_a_variable_name"]
    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context(),
    )
    assert plan.steps[5].functional_form_spec.target_column == "age_years"


@pytest.mark.parametrize("target", ["exposure_flag", "outcome_flag", "sex_code", "not_available"])
def test_compiler_rejects_noncontinuous_or_nonmodel_targets(target):
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(_functional_payload(target)), context=_context(),
        )
    assert caught.value.reason_code == "progressive_functional_form_target_invalid"


def test_compiler_rejects_a_functional_method_without_a_target():
    payload = _functional_payload()
    payload["steps"][5]["functional_form_spec"] = None
    with pytest.raises(ProgressivePlanCompileError) as caught:
        compile_progressive_plan(
            skeleton=ProgressivePlanSkeleton.model_validate(payload), context=_context(),
        )
    assert caught.value.reason_code == "progressive_functional_form_target_missing"


def test_new_optional_contract_does_not_change_legacy_step_serialization():
    skeleton = ProgressivePlanSkeleton.model_validate(_payload())
    assert all("functional_form_spec" not in step.model_dump(mode="json") for step in skeleton.steps)
    step = AnalysisStep(step_id="legacy", intent="A previously saved step.")
    assert "functional_form_spec" not in step.model_dump(mode="json")


@pytest.mark.parametrize("variable", ["age", "sex_code"])
def test_declared_covariate_mapping_is_honored_but_other_targets_are_not(variable):
    from easyicu.research_agent.planning.scientific_review import _sensitivity_facts
    from easyicu.research_agent.schema import UserPreferences

    payload = _functional_payload()
    payload["steps"][5]["sensitivity_spec_ids"] = ["one_covariate_form"]
    preferences = UserPreferences(
        covariates=["age", "sex_code"], covariate_selection="exact",
        covariate_operationalizations={"age": "age_years"},
        sensitivity_specs=[{
            "spec_id": "one_covariate_form", "axis": "functional_form",
            "strategy": "restricted_cubic_spline", "execution_variables": [variable],
        }],
    )
    context = _context().model_copy(update={"user_preferences": preferences})
    if variable == "sex_code":
        with pytest.raises(ProgressivePlanCompileError) as caught:
            compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context)
        assert caught.value.reason_code == "progressive_functional_form_authority_mismatch"
    else:
        plan, _ = compile_progressive_plan(skeleton=ProgressivePlanSkeleton.model_validate(payload), context=context)
        assert not _sensitivity_facts(context, plan)["missing_spec_ids"]


def _bound_sensitivity(target):
    projection, authority = _authority("e2_lactate_mortality")
    primary = _e2_plan(authority).steps[0]
    draft = primary.model_copy(update={
        "method": "adjusted_association_models", "expected_outputs": ["table:adjusted_association_estimates"],
    })
    child = AnalysisStep(
        step_id="check_form", planned_analysis_role="sensitivity", intent="Check one declared model term.",
        method="restricted_cubic_spline_sensitivity", inputs=["table:adjusted_association_estimates"],
        expected_outputs=["table:functional_form"], sensitivity_spec_ids=["one_term_form"],
        scientific_capability="association_freeform_v1", functional_form_spec=_form(target),
    )
    bound = authority.bind_plan(_e2_plan(authority).model_copy(update={"steps": [draft, child]}))
    return projection, authority, bound


def test_covariate_binding_retains_cohort_for_a_real_refit():
    _, authority, bound = _bound_sensitivity("age")
    child = bound.steps[1]
    authority.validate_plan(bound)
    assert "dataset:analysis_cohort" in child.inputs
    assert set(authority.required_columns) <= set(child.inputs)
    assert child.functional_form_spec.target_column == "age"


def test_timing_method_cannot_claim_the_functional_form_executor():
    _, authority, bound = _bound_sensitivity("lact_max")
    wrong = bound.steps[1].model_copy(update={"method": "landmark_analysis", "functional_form_spec": None})
    assert not landmark_spline_functional_form_executor_owns_step(wrong, plan=bound, authority=authority)
    with pytest.raises(ValueError, match="target-bound execution owner"):
        authority.validate_plan(bound.model_copy(update={"steps": [bound.steps[0], wrong]}))


def test_functional_form_refit_keeps_its_signed_temporal_closure():
    from easyicu.research_agent.planning.scientific_review import timing_design_closed, _sensitivity_facts

    _, _, bound = _bound_sensitivity("age")
    assert timing_design_closed(bound)
    assert "functional_form" in _sensitivity_facts(_context(), bound)["typed_executable"]
    wrong = bound.steps[1].model_copy(update={"icu_rule_refs": []})
    assert not timing_design_closed(bound.model_copy(update={"steps": [bound.steps[0], wrong]}))


def test_native_exposure_spline_does_not_credit_a_covariate_form_spec():
    from easyicu.research_agent.planning.scientific_review import _sensitivity_facts
    from easyicu.research_agent.schema import UserPreferences

    _, _, bound = _bound_sensitivity("age")
    preferences = UserPreferences(sensitivity_specs=[{
        "spec_id": "one_term_form", "axis": "functional_form",
        "strategy": "restricted_cubic_spline", "execution_variables": ["age"],
    }])
    context = _context().model_copy(update={"primary_exposure": "lact_max", "user_preferences": preferences})
    primary_only = bound.model_copy(update={"steps": [bound.steps[0]]})
    assert _sensitivity_facts(context, primary_only)["missing_spec_ids"] == ["one_term_form"]
    assert _sensitivity_facts(context, bound)["missing_spec_ids"] == []


def test_covariate_check_cannot_project_an_exposure_only_comparison(tmp_path):
    projection, authority, bound = _bound_sensitivity("age")
    source = pd.DataFrame([{
        "n": 100, "events": 15, "linear_aic": 110., "spline_aic": 105.,
        "linear_bic": 120., "spline_bic": 117., "likelihood_ratio_statistic": 7.,
        "additional_spline_parameters": 2, "nonlinearity_p_value": .03,
    }])
    with pytest.raises(ValueError, match="refit|cohort|target"):
        run_landmark_spline_functional_form(
            step=bound.steps[1], authority=authority, runtime_projection_sha256=projection.runtime_projection_sha256,
            linear_sensitivity=source, linear_evidence_id="exposure_only", out_dir=tmp_path,
        )
    assert not (tmp_path / "functional_form.csv").exists()


def test_exposure_check_does_not_discharge_an_unchecked_covariate():
    from easyicu.research_agent.planning.scientific_review import _continuous_linearity_facts

    plan, _ = compile_progressive_plan(
        skeleton=ProgressivePlanSkeleton.model_validate(_functional_payload()), context=_context(),
    )
    child = plan.steps[5].model_copy(update={"functional_form_spec": _form("another_continuous_term")})
    wrong = plan.model_copy(update={"steps": [*plan.steps[:5], child, *plan.steps[6:]]})
    assert not _continuous_linearity_facts(wrong)["functional_form_sensitivity_executable"]
    assert _continuous_linearity_facts(plan)["functional_form_sensitivity_executable"]


def _synthetic_frame():
    rng = np.random.default_rng(73051)
    n = 600
    age = rng.uniform(22., 88., n)
    exposure = rng.lognormal(0., .55, n)
    logit = -2.8 + .25 * exposure + .004 * (age - 52.) ** 2
    events = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    frame = pd.DataFrame({
        "lact_max": exposure, "age": age, "sex": rng.choice(["F", "M"], n),
        "death": events, "death_time": np.where(events, 72., np.nan),
        "los_icu": 5., "patient_key": np.repeat(np.arange(n // 2), 2),
        "charlson_first": rng.integers(0, 6, n).astype(float),
    })
    frame.loc[0, ["death", "death_time"]] = [1, 12.]
    frame.loc[1, "los_icu"] = .5
    frame.loc[2, "age"] = np.nan
    return frame


@pytest.mark.parametrize("clustered", [False, True])
def test_covariate_sensitivity_really_refits_on_the_primary_population(tmp_path, monkeypatch, clustered):
    import hashlib
    import json
    import statsmodels.api as sm
    from easyicu.research_agent.authority.current_case_scientific_runtime import LandmarkSplineRuntimeAuthority
    from easyicu.research_agent.canonical_json import canonical_sha256
    from easyicu.research_agent.execution.runners.landmark_spline_executor import run_landmark_spline_association
    from easyicu.research_agent.execution.runners.landmark_spline_fit import compare_covariate_functional_form

    projection, authority, bound = _bound_sensitivity("age")
    if clustered:
        body = authority.model_dump(mode="json", exclude={"execution_contract_sha256"})
        body.update(schema_version="easyicu.landmark_spline_runtime_authority/4", dependence={
            "schema_version": "easyicu.planned_dependence/1", "variance_estimator": "cluster_robust",
            "cluster_unit": "patient", "group_source": "patient_key", "group_derivation": "identity", "delimiter": None,
        })
        authority = LandmarkSplineRuntimeAuthority.model_validate_json(json.dumps({**body, "execution_contract_sha256": canonical_sha256(body)}))
        bound = authority.bind_plan(bound)
        authority.validate_plan(bound)
    frame = _synthetic_frame()
    before = frame.copy(deep=True)
    original_fit = sm.GLM.fit
    covariance_calls = []

    def tracking_fit(self, *args, **kwargs):
        covariance_calls.append(kwargs.get("cov_type", "nonrobust"))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(sm.GLM, "fit", tracking_fit)
    run_landmark_spline_association(
        frame=frame, authority=authority, runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path / "primary",
    )
    source = pd.read_csv(tmp_path / "primary" / f"{authority.linear_sensitivity_product.partition(':')[2]}.csv")
    frame.to_parquet(tmp_path / "cohort.parquet", index=False)
    contrast_path = tmp_path / "primary" / f"{authority.downstream_parent_product.partition(':')[2]}.csv"
    contrast_source = pd.read_csv(contrast_path)
    source_paths = {
        "dataset:analysis_cohort": tmp_path / "cohort.parquet",
        authority.downstream_parent_product: contrast_path,
        authority.linear_sensitivity_product: tmp_path / "primary" / f"{authority.linear_sensitivity_product.partition(':')[2]}.csv",
    }
    receipts = [{
        "input_key": key, "evidence_id": "primary_comparison" if key == authority.linear_sensitivity_product else key,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "loaded": True,
        "row_count": len(frame) if key.startswith("dataset:") else 2 if key == authority.downstream_parent_product else 1,
    } for key, path in source_paths.items()]
    result = run_landmark_spline_functional_form(
        step=bound.steps[1], authority=authority, runtime_projection_sha256=projection.runtime_projection_sha256,
        linear_sensitivity=source, linear_evidence_id="primary_comparison", out_dir=tmp_path / "sensitivity",
        cohort_frame=frame, primary_contrasts=contrast_source, input_bindings=receipts,
    )
    table = pd.read_csv(tmp_path / "sensitivity" / "functional_form.csv")
    assert result["target_column"] == "age"
    assert result["execution_mode"] == "covariate_refit_same_primary_population"
    assert result["n_complete_case"] == 597
    assert table.loc[0, "linear_aic"] == pytest.approx(source.loc[0, "spline_aic"])
    assert table.loc[0, "spline_aic"] != pytest.approx(source.loc[0, "spline_aic"])
    assert table.loc[0, "additional_spline_parameters"] == 1
    assert table.loc[0, "nonlinearity_p_value"] != pytest.approx(source.loc[0, "nonlinearity_p_value"])
    assert json.loads(table.loc[0, "target_knots"]) != json.loads(table.loc[0, "primary_exposure_knots"])
    assert covariance_calls[-2:] == ["cluster" if clustered else "nonrobust"] * 2
    if clustered:
        assert table.loc[0, "method"] == "cluster_robust_nested_wald_chi2"
        assert table.loc[0, "cluster_count"] == 299
        assert table.loc[0, "information_criteria_basis"] == "working_independence_loglikelihood_descriptive_only"
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="reproduce the source primary"):
        compare_covariate_functional_form(
            frame=frame.iloc[3:].copy(), authority=authority, form=_form("age"),
            primary_diagnostics={**source.iloc[0].to_dict(), "n": 599},
        )


def test_covariate_refit_matches_an_independent_truncated_power_spline_oracle(tmp_path):
    import statsmodels.api as sm
    from scipy.stats import chi2
    from easyicu.research_agent.execution.runners.landmark_spline_executor import run_landmark_spline_association
    from easyicu.research_agent.execution.runners.landmark_spline_fit import compare_covariate_functional_form

    projection, authority, _ = _bound_sensitivity("age")
    frame = _synthetic_frame()
    run_landmark_spline_association(
        frame=frame, authority=authority, runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path / "primary",
    )
    source = pd.read_csv(tmp_path / "primary" / f"{authority.linear_sensitivity_product.partition(':')[2]}.csv")
    actual = compare_covariate_functional_form(
        frame=frame, authority=authority, form=_form("age"), primary_diagnostics=source.iloc[0],
    )
    # Independent row selection and truncated-power basis: no shared population,
    # Patsy basis, fitting helper, or nested-comparison implementation is used.
    rows = frame.loc[((frame.death == 0) | (frame.death_time > 24)) & (frame.los_icu >= 1)].dropna(
        subset=["lact_max", "death", "age", "sex", "charlson_first"]
    )

    def nonlinear_basis(values):
        lower, middle, upper = np.quantile(values, [.1, .5, .9])
        return (
            np.maximum(values - lower, 0) ** 3
            - np.maximum(values - middle, 0) ** 3 * (upper - lower) / (upper - middle)
            + np.maximum(values - upper, 0) ** 3 * (middle - lower) / (upper - middle)
        ) / (upper - lower) ** 2

    restricted = pd.DataFrame({
        "const": 1., "exposure": rows.lact_max, "exposure_nonlinear": nonlinear_basis(rows.lact_max),
        "age": rows.age, "sex_M": (rows.sex == "M").astype(float), "charlson": rows.charlson_first,
    })
    full = restricted.assign(age_nonlinear=nonlinear_basis(rows.age))
    base_fit = sm.GLM(rows.death, restricted, family=sm.families.Binomial()).fit(maxiter=200)
    full_fit = sm.GLM(rows.death, full, family=sm.families.Binomial()).fit(maxiter=200)
    statistic = 2 * (full_fit.llf - base_fit.llf)
    assert actual["n_complete_case"] == len(rows) == 597
    assert actual["event_n"] == rows.death.sum()
    assert actual["linear_aic"] == pytest.approx(base_fit.aic, abs=1e-7)
    assert actual["spline_aic"] == pytest.approx(full_fit.aic, abs=1e-7)
    assert actual["statistic"] == pytest.approx(statistic, abs=1e-7)
    assert actual["nonlinearity_p_value"] == pytest.approx(chi2.sf(statistic, 1), rel=1e-7)


@pytest.mark.parametrize("mutation", [None, "missing_cohort", "digest_drift", "wrong_step", "cohort_alias"])
def test_bound_covariate_refit_verifies_all_three_artifact_inputs(tmp_path, mutation):
    import hashlib
    import json
    from easyicu.research_agent.execution.runners.landmark_spline_executor import run_landmark_spline_association
    from easyicu.research_agent.execution.runners.landmark_spline_functional_form_executor import run_bound_landmark_spline_functional_form
    from easyicu.research_agent.execution.runners.typed_input_binding import TypedInputBindingError

    projection, authority, plan = _bound_sensitivity("age")
    cohort_input = "dataset:analysis_cohort"
    if mutation == "cohort_alias":
        cohort_input = "cohort:analysis_set"
        primary = plan.steps[0].model_copy(update={
            "inputs": [cohort_input if key == "dataset:analysis_cohort" else key for key in plan.steps[0].inputs],
        })
        plan = authority.bind_plan(plan.model_copy(update={"steps": [primary, plan.steps[1]]}))
    frame = _synthetic_frame()
    frame.to_parquet(tmp_path / "cohort.parquet", index=False)
    run_landmark_spline_association(
        frame=frame, authority=authority, runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path / "primary",
    )
    paths = {cohort_input: tmp_path / "cohort.parquet"}
    paths.update({key: tmp_path / "primary" / f"{key.partition(':')[2]}.csv" for key in (
        authority.downstream_parent_product, authority.linear_sensitivity_product,
    )})
    manifest = {"step_id": plan.steps[1].step_id, "inputs": {}}
    for key, path in paths.items():
        data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["inputs"][key] = {
            "relative_path": str(path.relative_to(tmp_path)), "sha256": digest,
            "declared_kind": "dataset" if key.startswith("cohort:") else key.partition(":")[0],
            "evidence_kind": "table", "product": key.partition(":")[2],
            "evidence_id": key, "product_contract": {"columns": list(data.columns), "row_count": len(data)},
            "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest},
        }
    expected = None
    if mutation == "missing_cohort":
        manifest["inputs"].pop("dataset:analysis_cohort")
        expected = "binding_absent"
    elif mutation == "digest_drift":
        frame.assign(age=frame.age + 1).to_parquet(tmp_path / "cohort.parquet", index=False)
        expected = "digest_mismatch"
    elif mutation == "wrong_step":
        manifest["step_id"] = "a_different_step"
        expected = "manifest_step_mismatch"
    manifest_path = tmp_path / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def execute():
        return run_bound_landmark_spline_functional_form(
            step=plan.steps[1], authority=authority, runtime_projection_sha256=projection.runtime_projection_sha256,
            run_dir=tmp_path, resolved_inputs=manifest_path, out_dir=tmp_path / "sensitivity",
        )

    if expected:
        with pytest.raises(TypedInputBindingError) as caught:
            execute()
        assert caught.value.reason_code == expected
        assert not (tmp_path / "sensitivity" / "functional_form.csv").exists()
    else:
        result = execute()
        assert {item["input_key"] for item in result["input_bindings"]} == set(paths)
        assert all(item["loaded"] for item in result["input_bindings"])
        assert result["target_column"] == "age"
