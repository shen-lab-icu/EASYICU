from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.association_execution import (
    association_binary_sensitivity_plan_verdict,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.sensitivity_plan_shaping import (
    ensure_prespecified_sensitivity_steps,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    LiteratureDesignBinding,
    PlannedModelRequirement,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is exposure associated with the binary outcome?",
        cohort=CohortDescriptor(
            cohort_name="adult ICU stays",
            database="miiv",
            n_stays=100,
            id_columns=["stay_id"],
        ),
        variables=[
            ConceptDescriptor(name="exposure", role=VariableRole.OTHER, dtype="int64"),
            ConceptDescriptor(name="outcome", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
        ],
        target_outcome="outcome",
        primary_exposure="exposure",
        user_preferences=UserPreferences(
            covariates=["age"],
            sensitivity_specs=[
                {
                    "spec_id": "age_functional_form",
                    "axis": "functional_form",
                    "strategy": "restricted_cubic_spline",
                    "execution_variables": ["age"],
                },
                {
                    "spec_id": "complete_case_primary",
                    "axis": "missing_data",
                    "strategy": "complete_case",
                    "execution_variables": ["exposure", "outcome", "age"],
                },
            ],
        ),
    )


def _plan() -> AnalysisPlan:
    spline_binding = LiteratureDesignBinding(
        citation_key="spline_method",
        design_elements=["adjustment", "robustness"],
        application="Use the sealed spline method card for the functional-form check.",
    )
    primary = AnalysisStep(
        step_id="primary_adjusted_association",
        planned_analysis_role="primary",
        intent="Estimate the adjusted association.",
        method="adjusted_association_models",
        inputs=["exposure", "outcome", "age", "artifact:analysis_cohort"],
        expected_outputs=["table:adjusted_association_estimates"],
        sensitivity_spec_ids=["age_functional_form", "complete_case_primary"],
        literature_citation_keys=["spline_method"],
        literature_design_bindings=[spline_binding],
        model_requirements=[
            PlannedModelRequirement(
                requirement_id="primary",
                outcome="outcome",
                outcome_type="binary",
                method_family="statsmodels_logit_mle",
                exposure_source="exposure",
                analysis_role="primary",
                analysis_set="source_aware",
                covariates=["age"],
                model_terms=[
                    {
                        "name": "exposure",
                        "role": "exposure",
                        "coding": "binary",
                        "levels": ["0", "1"],
                        "reference_level": "0",
                        "transform": "treatment_contrast",
                    },
                    {
                        "name": "age",
                        "role": "covariate",
                        "coding": "continuous",
                        "transform": "identity",
                    },
                ],
            )
        ],
    )
    replay = AnalysisStep.model_validate(
        {
            "step_id": "robustness_grid",
            "planned_analysis_role": "sensitivity",
            "intent": "Replay the locked complete-case specification.",
            "method": "robustness_sensitivity",
            "inputs": [
                "artifact:analysis_cohort",
                "table:adjusted_association_estimates",
            ],
            "expected_outputs": ["table:robustness_matrix"],
            "sensitivity_spec_ids": [
                "age_functional_form",
                "complete_case_primary",
            ],
            "literature_citation_keys": ["spline_method"],
            "literature_design_bindings": [spline_binding.model_dump(mode="json")],
            "robustness_replay_spec": {
                "products": [
                    {
                        "product_id": "robustness_matrix",
                        "output": "robustness_matrix",
                    }
                ]
            },
        }
    )
    return AnalysisPlan(
        research_question="Is exposure associated with the binary outcome?",
        analysis_type="association_study",
        steps=[primary, replay],
        robustness_specs=[
            RobustnessSpec(
                spec_id="complete_case_primary",
                axis="missing",
                description="Locked complete-case replay.",
                missing_override={
                    "strategy": "complete_case",
                    "variables": ["exposure", "outcome", "age"],
                },
            )
        ],
    )


def test_missing_typed_functional_form_becomes_explicit_analysis_only_step() -> None:
    shaped, findings = ensure_prespecified_sensitivity_steps(
        plan=_plan(),
        context=_context(),
    )

    inserted = [
        step
        for step in shaped.steps
        if step.step_id == "sensitivity_age_functional_form"
    ]
    assert len(inserted) == 1
    step = inserted[0]
    assert step.method == "restricted_cubic_spline_sensitivity"
    assert step.sensitivity_spec_ids == ["age_functional_form"]
    assert step.scientific_capability == "association_freeform_v1"
    assert step.expected_outputs == ["table:sensitivity_age_functional_form"]
    assert "complete_case_primary" not in step.sensitivity_spec_ids
    verdict = association_binary_sensitivity_plan_verdict(
        step,
        plan_steps=shaped.steps,
    )
    assert verdict.claimed is True
    assert findings[0].detail["deterministic_method_adapter"] is False

    again, repeated = ensure_prespecified_sensitivity_steps(
        plan=shaped,
        context=_context(),
    )
    assert again == shaped
    assert repeated == []


def test_missing_linear_per_unit_becomes_explicit_analysis_only_step() -> None:
    context = _context()
    preferences = UserPreferences.model_validate(
        {
            **context.user_preferences.model_dump(mode="json"),
            "sensitivity_specs": [
                {
                    "spec_id": "exposure_linear_per_unit",
                    "axis": "functional_form",
                    "strategy": "linear_per_unit",
                    "execution_variables": ["exposure"],
                }
            ],
        }
    )
    context = context.model_copy(update={"user_preferences": preferences})

    shaped, findings = ensure_prespecified_sensitivity_steps(
        plan=_plan(),
        context=context,
    )

    step = next(
        item for item in shaped.steps
        if item.step_id == "sensitivity_exposure_linear_per_unit"
    )
    assert step.method == "linear_per_unit_sensitivity"
    assert step.expected_outputs == ["table:sensitivity_exposure_linear_per_unit"]
    assert findings[0].detail["strategy"] == "linear_per_unit"


def test_landmark_and_cluster_choices_become_separate_executable_steps() -> None:
    context = _context()
    preferences = UserPreferences.model_validate(
        {
            **context.user_preferences.model_dump(mode="json"),
            "sensitivity_specs": [
                {
                    "spec_id": "landmark_24h",
                    "axis": "timing",
                    "strategy": "landmark",
                    "execution_variables": ["event_time", "observation_hours"],
                    "landmark_hours": 24,
                    "require_alive_at_landmark": True,
                    "exclude_negative_event_times": True,
                    "event_time_variable": "event_time",
                    "observation_duration_variable": "observation_hours",
                    "observation_duration_unit": "hours",
                },
                {
                    "spec_id": "repeated_stays_cluster_robust",
                    "axis": "repeated_stays",
                    "strategy": "cluster_robust",
                },
            ],
        }
    )
    context = context.model_copy(update={"user_preferences": preferences})
    plan = _plan()
    primary = plan.steps[0]
    requirement = primary.model_requirements[0].model_copy(
        update={
            "dependence": {
                "schema_version": "easyicu.planned_dependence/1",
                "variance_estimator": "cluster_robust",
                "cluster_unit": "patient",
                "group_source": "patient_stay_id",
                "group_derivation": "prefix_before_delimiter",
                "delimiter": ":s",
            }
        }
    )
    primary = primary.model_copy(
        update={
            "inputs": [*primary.inputs, "patient_stay_id"],
            "model_requirements": [requirement],
        }
    )
    plan = plan.model_copy(update={"steps": [primary, *plan.steps[1:]]})

    shaped, findings = ensure_prespecified_sensitivity_steps(
        plan=plan,
        context=context,
    )

    landmark = next(
        step for step in shaped.steps if step.step_id == "sensitivity_landmark_24h"
    )
    clustered = next(
        step
        for step in shaped.steps
        if step.step_id == "sensitivity_repeated_stays_cluster_robust"
    )
    assert landmark.method == "landmark_analysis"
    assert {"event_time", "observation_hours"} <= set(landmark.inputs)
    assert clustered.method == "cluster_robust_association"
    assert "patient_stay_id" in clustered.inputs
    assert {item.detail["spec_id"] for item in findings} == {
        "landmark_24h",
        "repeated_stays_cluster_robust",
    }


def test_first_stay_spec_becomes_an_executable_reestimation_step() -> None:
    context = _context()
    context = context.model_copy(
        update={
            "variables": [
                *context.variables,
                ConceptDescriptor(
                    name="icu_readmission",
                    role=VariableRole.OTHER,
                    dtype="int64",
                ),
            ],
            "user_preferences": UserPreferences.model_validate(
                {
                    **context.user_preferences.model_dump(mode="json"),
                    "sensitivity_specs": [
                        {
                            "spec_id": "first_icu_stay",
                            "axis": "repeated_stays",
                            "strategy": "first_stay",
                            "execution_variables": ["icu_readmission"],
                        }
                    ],
                }
            ),
        }
    )

    shaped, findings = ensure_prespecified_sensitivity_steps(
        plan=_plan(),
        context=context,
    )

    step = next(
        item for item in shaped.steps if item.step_id == "sensitivity_first_icu_stay"
    )
    assert step.method == "first_stay_association"
    assert "icu_readmission" in step.inputs
    assert step.scientific_capability == "association_freeform_v1"
    assert step.sensitivity_spec_ids == ["first_icu_stay"]
    assert findings[0].detail["strategy"] == "first_stay"


def test_time_varying_choice_is_not_projected_without_a_registered_runtime() -> None:
    context = _context()
    preferences = UserPreferences.model_validate(
        {
            **context.user_preferences.model_dump(mode="json"),
            "sensitivity_specs": [
                {
                    "spec_id": "time_varying_exposure",
                    "axis": "timing",
                    "strategy": "time_varying",
                    "execution_variables": ["exposure"],
                }
            ],
        }
    )
    context = context.model_copy(update={"user_preferences": preferences})

    shaped, findings = ensure_prespecified_sensitivity_steps(
        plan=_plan(),
        context=context,
    )

    assert shaped.steps == _plan().steps
    assert findings == []


def _landmark_shaping_case():
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        build_current_case_scientific_runtime_authority,
    )

    authority = build_current_case_scientific_runtime_authority({
        "schema_version": "easyicu.landmark_spline_runtime_authority/1",
        "authority_kind": "landmark_spline_association",
        "protocol_content_sha256": "a" * 64,
        "plan_method": "signed_landmark_restricted_cubic_spline",
        "plan_intent": "Estimate the prespecified landmark association.",
        "plan_outputs": ["table:risk_curve", "table:exposure_contrasts", "table:linear_sensitivity", "log:runtime_receipt"],
        "exposure_column": "exposure", "outcome_column": "outcome",
        "outcome_time_column": "event_hours",
        "observation_duration_column": "followup_hours",
        "observation_duration_unit": "hours", "landmark_hours": 24,
        "required_adjustment_columns": ["age"], "categorical_adjustment_columns": [],
        "alternative_exposure_columns": [], "dependence": None,
        "adjusted_absolute_risk_product": None, "population_flow_product": None,
        "variable_opportunity_sensitivity_product": None,
        "spline_knot_quantiles": [.1, .5, .9], "curve_quantile_range": [.1, .9],
        "spline_reference": "median_in_primary_population", "curve_points": 41,
        "linear_sensitivity_per_unit": 1.,
        "interpretation": "descriptive_prognostic_association_not_causal",
    })
    spec = {
        "spec_id": "predeclared_time_zero", "axis": "timing", "strategy": "landmark",
        "landmark_hours": 24., "require_alive_at_landmark": True,
        "exclude_negative_event_times": True,
        "event_time_variable": "event_hours", "observation_duration_variable": "followup_hours",
        "observation_duration_unit": "hours", "execution_variables": ["event_hours", "followup_hours"],
    }
    context = _context().model_copy(update={"user_preferences": UserPreferences(sensitivity_specs=[spec])})
    primary = _plan().steps[0].model_copy(update={"sensitivity_spec_ids": [spec["spec_id"]]})
    plan = _plan().model_copy(update={"steps": [primary], "robustness_specs": []})
    return authority, context, plan


@pytest.mark.parametrize("mutation", [None, "no_authority", "different_hour", "different_event", "different_unit", "extra_variable", "different_adjustment", "unreferenced", "different_eligibility"])
def test_landmark_obligation_uses_exact_primary_runtime_owner(mutation):
    from easyicu.research_agent.planning.figure_plan_shaping import apply_required_plan_obligations

    authority, context, plan = _landmark_shaping_case()
    payload = context.user_preferences.model_dump(mode="json")
    spec = payload["sensitivity_specs"][0]
    if mutation == "no_authority":
        authority = None
    elif mutation == "different_hour":
        spec["landmark_hours"] = 48
    elif mutation == "different_event":
        spec["event_time_variable"] = "other_event_hours"
    elif mutation == "different_unit":
        spec["observation_duration_unit"] = "days"
    elif mutation == "extra_variable":
        spec["execution_variables"].append("other_exposure")
    elif mutation == "different_eligibility":
        spec["require_alive_at_landmark"] = False
    elif mutation == "different_adjustment":
        primary = plan.steps[0]
        requirement = primary.model_requirements[0].model_copy(update={"covariates": ["other_age"]})
        plan = plan.model_copy(update={"steps": [primary.model_copy(update={"model_requirements": [requirement]})]})
    elif mutation == "unreferenced":
        plan = plan.model_copy(update={"steps": [plan.steps[0].model_copy(update={"sensitivity_spec_ids": []})]})
    context = context.model_copy(update={"user_preferences": UserPreferences.model_validate(payload)})
    findings = []
    shaped = apply_required_plan_obligations(plan, context, findings, runtime_authority=authority)
    added = [step for step in shaped.steps if step.method == "landmark_analysis"]
    if mutation is not None:
        assert len(added) == 1
        assert added[0].sensitivity_spec_ids == [spec["spec_id"]]
        return
    assert added == []
    assert shaped.steps[0] == plan.steps[0]
    assert any(f.detail.get("reason_code") == "typed_landmark_obligation_owned_by_primary" for f in findings)
    bound = authority.bind_plan(shaped)
    authority.validate_plan(bound)
    assert bound.steps[0].method == authority.plan_method
    assert bound.steps[0].sensitivity_spec_ids == [spec["spec_id"]]


def test_primary_landmark_coverage_does_not_remove_other_sensitivity_obligations():
    authority, context, plan = _landmark_shaping_case()
    preferences = context.user_preferences.model_dump(mode="json")
    preferences["sensitivity_specs"].append({
        "spec_id": "different_missingness", "axis": "missing_data",
        "strategy": "multiple_imputation", "execution_variables": ["exposure", "age"],
    })
    context = context.model_copy(update={"user_preferences": UserPreferences.model_validate(preferences)})
    shaped, _ = ensure_prespecified_sensitivity_steps(plan=plan, context=context, runtime_authority=authority)
    assert [s.method for s in shaped.steps] == ["adjusted_association_models", "multiple_imputation_sensitivity"]


@pytest.mark.parametrize("mutation", [None, "plausibility", "cohort_digest", "missing_parent", "wrong_step", "different_population"])
def test_primary_population_risk_executes_same_rows_and_refuses_drift(tmp_path, monkeypatch, mutation):
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    from easyicu.research_agent.execution.runners.landmark_spline_executor import run_landmark_spline_association
    from easyicu.research_agent.execution.runners.selection import select_standard_executor
    from easyicu.research_agent.execution.runners.typed_input_binding import TypedInputBindingError

    authority, context, plan = _landmark_shaping_case()
    descriptive = AnalysisStep(
        step_id="risk_in_primary_population", planned_analysis_role="secondary",
        method="primary_population_absolute_risk_context", intent="Describe risk in the primary model population.",
        inputs=["exposure", "outcome", "artifact:analysis_cohort", "table:adjusted_association_estimates"],
        expected_outputs=["table:absolute_risk_context"],
    )
    plan = authority.bind_plan(plan.model_copy(update={"steps": [plan.steps[0], descriptive]}))
    authority.validate_plan(plan)
    step = plan.steps[1]
    scope = None
    raw_contracts = {"contracts": {"age": {"analysis_plausibility_range": {"minimum": 0, "maximum": 120}, "plausibility_policy": {"range_policy": "flag_only", "out_of_range_action": "retain_and_flag"}}}}
    if mutation == "plausibility":
        from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
        digest = hashlib.sha256(json.dumps(raw_contracts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        raw_contracts["contracts_sha256"] = digest
        scope = FlagOnlyPlausibilityScope(step_id=step.step_id, expected_columns=("age",), source_contracts_sha256=digest, authority_kind="test")
    selected = select_standard_executor(step, plan=plan, current_case_scientific_runtime_authority=authority, scientific_runtime_projection_sha256="b" * 64, plausibility_scope=scope)
    assert selected is not None
    assert selected.analysis_kind == "primary_population_absolute_risk_context"
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"exposure": rng.uniform(1, 9, 60), "outcome": [int(i % 3 == 0) for i in range(60)], "age": rng.uniform(20, 80, 60), "event_hours": 100., "followup_hours": 120.})
    frame.loc[:4, ["outcome", "event_hours"]] = [1, 12.]
    frame.loc[5:9, "followup_hours"] = 12.
    frame.loc[10:14, "age"] = np.nan
    cohort_path = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    run_landmark_spline_association(frame=frame, authority=authority, runtime_projection_sha256="b" * 64, out_dir=tmp_path / "primary")
    parent_path = tmp_path / "primary" / f"{authority.linear_sensitivity_product.partition(':')[2]}.csv"
    if mutation == "different_population":
        data = pd.read_csv(parent_path)
        data.loc[0, "n"] += 1
        data.to_csv(parent_path, index=False)
    manifest = {"step_id": step.step_id, "inputs": {}, "raw_input_contracts": raw_contracts}
    for key, path in [("artifact:analysis_cohort", cohort_path), (authority.linear_sensitivity_product, parent_path)]:
        data = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["inputs"][key] = {
            "relative_path": str(path.relative_to(tmp_path)), "sha256": digest,
            "declared_kind": key.partition(':')[0], "evidence_kind": "table", "product": key.partition(':')[2], "evidence_id": key,
            "product_contract": {"columns": list(data.columns), "row_count": len(data)},
            "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest},
        }
    if mutation == "cohort_digest":
        frame.assign(age=frame.age + 1).to_parquet(cohort_path, index=False)
    if mutation == "missing_parent":
        manifest["inputs"].pop(authority.linear_sensitivity_product)
    if mutation == "wrong_step":
        manifest["step_id"] = "another_step"
    (tmp_path / "resolved_inputs.json").write_text(json.dumps(manifest))
    (tmp_path / "analysis_plan.json").write_text(plan.model_dump_json())
    (tmp_path / "manifest_partial.json").write_text(json.dumps({"plan_path": "analysis_plan.json"}))
    (tmp_path / "research_context.json").write_text(context.model_dump_json())
    out = tmp_path / "risk"
    for key, value in {"EASYICU_RUN_DIR": tmp_path, "EASYICU_STEP_ID": step.step_id, "COHORT_PARQUET": cohort_path, "EASYICU_RESOLVED_INPUTS_JSON": tmp_path / "resolved_inputs.json", "STEP_OUT_DIR": out, "OUTCOME_COL": "outcome"}.items():
        monkeypatch.setenv(key, str(value))
    if mutation not in (None, "plausibility"):
        reason = {"cohort_digest": "digest_mismatch", "missing_parent": "binding_absent", "wrong_step": "manifest_step_mismatch", "different_population": "does not match"}[mutation]
        with pytest.raises((TypedInputBindingError, ValueError), match=reason):
            exec(compile(selected.code, "<primary_population_risk>", "exec"), {})
        assert not (out / "step_summary.json").exists()
        return
    exec(compile(selected.code, "<primary_population_risk>", "exec"), {})
    summary = json.loads((out / "step_summary.json").read_text())
    assert summary["n_total"] == 45
    assert summary["population_binding"]["source_cohort_n"] == 60
    assert summary["population_binding"]["event_n"] == 15
    assert summary["exposure_columns"] == ["exposure"]
    assert len(summary["input_bindings"]) == 2
    table = pd.read_csv(out / "absolute_risk_context.csv")
    assert set(table["population_scope"]) == {"primary_model_complete_cases"}
    assert set(table["population_n"]) == {45}

    if mutation == "plausibility":
        assert summary["plausibility_audit"]["age"]["compared_n"] == 55


def test_unbound_primary_population_cannot_pass_scientific_review():
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    authority, context, plan = _landmark_shaping_case()
    step = AnalysisStep(step_id="risk", method="primary_population_absolute_risk_context", planned_analysis_role="secondary", intent="Use primary population", inputs=["artifact:analysis_cohort", "table:adjusted_association_estimates"], expected_outputs=["table:absolute_risk_context"])
    plan = plan.model_copy(update={"steps": [plan.steps[0], step]})
    review = build_plan_scientific_review(context=context, plan=plan)
    assert not review.approval_allowed
    assert "PRIMARY_POPULATION_EXECUTION_OWNER_MISSING" in {f.code for f in review.findings}
    bound = authority.bind_plan(plan)
    review = build_plan_scientific_review(context=context, plan=bound)
    assert "PRIMARY_POPULATION_EXECUTION_OWNER_MISSING" not in {f.code for f in review.findings}
