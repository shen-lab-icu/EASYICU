from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.figure2_canonical9.case_scientific_protocol import (
    build_runtime_scientific_projection,
    load_default_case_protocol,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    CurrentCaseScientificAuthorityError,
    LandmarkSplineRuntimeAuthority,
    LandmarkSurvivalRuntimeAuthority,
    SourceFeasibilityRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.contracts.capability_ids import (
    LANDMARK_SPLINE_ANALYSIS_KIND,
    LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID,
    SOURCE_FEASIBILITY_ANALYSIS_KIND,
    SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID,
)
from easyicu.research_agent.contracts.landmark_spline_validation import (
    landmark_spline_runtime_receipt_valid,
)
from easyicu.research_agent.contracts.source_feasibility_validation import (
    source_feasibility_runtime_bundle_errors,
)
from easyicu.research_agent.execution.final_validation import (
    _primary_runner_core_estimate_present,
)
from easyicu.research_agent.reporting.readiness import (
    _compute_readiness_gates,
    _deterministic_primary_estimate_bound,
)
from easyicu.research_agent.execution.runners.landmark_spline_executor import (
    run_landmark_spline_association,
)
from easyicu.research_agent.execution.runners.landmark_spline_functional_form_executor import (
    LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND,
    run_landmark_spline_functional_form,
)
from easyicu.research_agent.execution.runners.landmark_spline_robustness_executor import (
    _matching_complete_case_spec_id,
    run_landmark_spline_robustness,
)
from easyicu.research_agent.execution.runners.landmark_survival_executor import (
    run_landmark_survival_suite,
    run_landmark_survival_figure,
)
from easyicu.research_agent.execution.runners.source_feasibility_executor import (
    run_source_feasibility_fail_closed,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.plan_utils import (
    _typed_plan_dag_findings,
    effect_output_authorized,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ResearchContext,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.capability_registry import (
    assess_scientific_capability,
    resolve_primary_capability,
)


def _authority(task_id: str):
    projection = build_runtime_scientific_projection(
        load_default_case_protocol(task_id)
    )
    authority = load_current_case_scientific_runtime_authority(
        projection.deterministic_execution_contract or {}
    )
    return projection, authority


def _e2_plan(authority: LandmarkSplineRuntimeAuthority) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Estimate the signed landmark association.",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "01_signed_primary",
                    "planned_analysis_role": "primary",
                    "intent": authority.plan_intent,
                    "inputs": [
                        "dataset:analysis_cohort",
                        *authority.required_columns,
                    ],
                    "expected_outputs": list(authority.plan_outputs),
                    "method": authority.plan_method,
                    "scientific_capability": LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID,
                    "icu_rule_refs": [authority.plan_rule_ref],
                }
            ],
        }
    )


def test_landmark_complete_case_row_binds_only_to_equivalent_locked_spec() -> None:
    _projection, authority = _authority("e2_lactate_mortality")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    variables = [
        authority.exposure_column,
        authority.outcome_column,
        *authority.required_adjustment_columns,
    ]
    specs = [
        RobustnessSpec(
            spec_id="primary_complete_case",
            axis="missing",
            description="Use complete observations for the signed primary model.",
            missing_override={"strategy": "complete_case", "variables": variables},
        )
    ]

    assert (
        _matching_complete_case_spec_id(specs=specs, authority=authority)
        == "primary_complete_case"
    )
    specs[0] = RobustnessSpec(
        spec_id="narrow_complete_case",
        axis="missing",
        description="Uses a scientifically different analysis set.",
        missing_override={
            "strategy": "complete_case",
            "variables": variables[:-1],
        },
    )
    with pytest.raises(ValueError, match="matched 0"):
        _matching_complete_case_spec_id(specs=specs, authority=authority)


def _h2_plan(authority: SourceFeasibilityRuntimeAuthority) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Assess whether the contrast is identifiable.",
            "analysis_type": "causal_inference",
            "steps": [
                {
                    "step_id": "01_signed_feasibility",
                    "planned_analysis_role": "auxiliary",
                    "intent": authority.plan_intent,
                    "inputs": [],
                    "expected_outputs": list(authority.plan_outputs),
                    "method": authority.plan_method,
                    "icu_rule_refs": [authority.plan_rule_ref],
                }
            ],
        }
    )


def _h1_draft_plan() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Estimate the ventilation survival association.",
            "analysis_type": "survival",
            "steps": [
                {
                    "step_id": "01_primary_survival",
                    "planned_analysis_role": "primary",
                    "intent": "Draft the primary survival analysis.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:cox_summary"],
                    "method": "cox_proportional_hazards",
                },
                {
                    "step_id": "02_figure",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Draft a survival figure.",
                    "inputs": ["table:cox_summary"],
                    "expected_outputs": ["figure:survival"],
                    "method": "visualization",
                },
            ],
        }
    )


@pytest.mark.parametrize(
    ("step_updates", "record_updates"),
    [
        ({"method": "descriptive_summary"}, {}),
        ({"planned_analysis_role": "sensitivity"}, {}),
        ({"expected_outputs": ["figure:survival"]}, {}),
        ({"icu_rule_refs": []}, {}),
        ({}, {"deterministic_standard_analysis": "association_model_grid"}),
        (
            {},
            {
                "deterministic_standard_selection_reason": (
                    "signed_association_model_grid_contract_preflight"
                )
            },
        ),
        ({}, {"standard_executor_candidates": {"claimed_by": "coder"}}),
    ],
    ids=[
        "wrong-method",
        "wrong-role",
        "no-table-output",
        "no-runtime-contract",
        "wrong-analysis-owner",
        "wrong-selection-reason",
        "coder-claimed",
    ],
)
def test_h1_effect_output_authority_fails_closed_on_near_miss(
    step_updates: dict[str, object],
    record_updates: dict[str, object],
) -> None:
    _projection, authority = _authority("h1_ventilation_survival")
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    bound, _findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(_h1_draft_plan())
    step = bound.steps[1].model_copy(update=step_updates)
    step_record = {
        "deterministic_standard_analysis": "signed_landmark_survival_suite",
        "deterministic_standard_selection_reason": (
            "signed_landmark_survival_suite_contract_preflight"
        ),
        "standard_executor_candidates": {
            "claimed_by": "signed_landmark_survival_suite"
        },
    }
    step_record.update(record_updates)

    assert effect_output_authorized(step, step_record=step_record) is False


def test_h1_effect_output_authority_requires_mapping_record() -> None:
    _projection, authority = _authority("h1_ventilation_survival")
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    bound, _findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(_h1_draft_plan())

    assert effect_output_authorized(bound.steps[1], step_record=None) is False


def test_e2_plan_and_runtime_are_bound_to_one_signed_contract(tmp_path: Path) -> None:
    projection, authority = _authority("e2_lactate_mortality")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    plan = _e2_plan(authority)
    authority.validate_plan(plan)
    verdict = resolve_primary_capability(
        analysis_type=plan.analysis_type,
        plan=plan,
    )
    assert verdict.capability_id == LANDMARK_SPLINE_ASSOCIATION_CAPABILITY_ID
    assert verdict.execution_owner == "host_deterministic"
    assert verdict.scientific_validation == "reportable"
    assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type,
        context=SimpleNamespace(
            research_question=plan.research_question,
            primary_exposure=authority.exposure_column,
            target_outcome=authority.outcome_column,
            variables=[],
            cohort=object(),
            endpoint=None,
        ),
        plan=plan,
    )
    assert assessment.scientific_validator_available
    assert assessment.claim_ceiling == "reportable"
    assert effect_output_authorized(
        plan.steps[0],
        step_record={
            "deterministic_standard_analysis": LANDMARK_SPLINE_ANALYSIS_KIND,
            "deterministic_standard_selection_reason": (
                "signed_landmark_spline_contract_preflight"
            ),
            "standard_executor_candidates": {
                "claimed_by": LANDMARK_SPLINE_ANALYSIS_KIND
            },
        },
    )

    drifted_step = plan.steps[0].model_copy(
        update={"method": "linear_logistic_regression"}
    )
    with pytest.raises(CurrentCaseScientificAuthorityError, match="method"):
        authority.validate_plan(plan.model_copy(update={"steps": [drifted_step]}))

    rng = np.random.default_rng(20260810)
    n = 320
    lactate = rng.lognormal(mean=0.55, sigma=0.45, size=n)
    age = rng.normal(64.0, 13.0, size=n)
    sex = rng.choice(["F", "M"], size=n).astype(object)
    sex[0] = None
    charlson = rng.poisson(3.0, size=n).astype(float)
    probability = 1.0 / (1.0 + np.exp(-(-3.6 + 0.45 * lactate + 0.018 * (age - 60.0))))
    death = rng.binomial(1, probability, size=n)
    death_time = np.where(death == 1, rng.uniform(30.0, 180.0, size=n), np.nan)
    frame = pd.DataFrame(
        {
            "lact_max": lactate,
            "death": death,
            "death_time": death_time,
            "los_icu": rng.uniform(1.2, 8.0, size=n),
            "age": age,
            "sex": sex,
            "charlson_first": charlson,
        }
    )
    summary = run_landmark_spline_association(
        frame=frame,
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path,
    )
    assert summary["status"] == "ok"
    assert set(summary["output_files"]) == set(authority.plan_outputs)
    receipt = summary["scientific_runtime_receipt"]
    assert receipt["execution_contract_sha256"] == (authority.execution_contract_sha256)
    assert landmark_spline_runtime_receipt_valid(summary)
    assert _primary_runner_core_estimate_present(LANDMARK_SPLINE_ANALYSIS_KIND, summary)
    records = [
        {
            "step_id": "01_signed_primary",
            "deterministic_standard_analysis": LANDMARK_SPLINE_ANALYSIS_KIND,
            "step_summary": summary,
        }
    ]
    assert _deterministic_primary_estimate_bound(records)
    broken = json.loads(json.dumps(summary))
    broken["scientific_runtime_receipt"]["observed_knots"] = [2.0, 1.0, 3.0]
    assert not landmark_spline_runtime_receipt_valid(broken)
    assert not _primary_runner_core_estimate_present(
        LANDMARK_SPLINE_ANALYSIS_KIND, broken
    )
    assert receipt["runtime_projection_sha256"] == (
        projection.runtime_projection_sha256
    )
    assert summary["n_primary_population"] == n
    assert summary["n_complete_case"] == n - 1
    sensitivity = pd.read_csv(tmp_path / "e2_linear_sensitivity.csv")
    assert len(sensitivity) == 1
    assert int(sensitivity.loc[0, "n"]) == n - 1
    assert int(sensitivity.loc[0, "additional_spline_parameters"]) > 0
    assert 0.0 <= sensitivity.loc[0, "nonlinearity_p_value"] <= 1.0
    assert sensitivity.loc[0, "likelihood_ratio_statistic"] >= 0.0
    absolute_risk = pd.read_csv(tmp_path / "e2_adjusted_absolute_risk.csv")
    assert len(absolute_risk) == authority.curve_points
    assert absolute_risk["adjusted_absolute_risk"].between(0.0, 1.0).all()
    assert (absolute_risk["ci_low"] <= absolute_risk["adjusted_absolute_risk"]).all()
    assert (absolute_risk["adjusted_absolute_risk"] <= absolute_risk["ci_high"]).all()
    assert absolute_risk["standardization_n"].eq(n - 1).all()
    population_flow = pd.read_csv(tmp_path / "e2_landmark_population_flow.csv")
    assert population_flow["n"].tolist() == [n, n, n, n - 1]
    variable_opportunity = pd.read_csv(
        tmp_path / "e2_variable_opportunity_sensitivity.csv"
    )
    assert int(variable_opportunity.loc[0, "n"]) == n - 1
    assert variable_opportunity.loc[0, "adjusted_odds_ratio"] > 0


def test_m1_reuses_generic_landmark_spline_with_definition_sensitivity(
    tmp_path: Path,
) -> None:
    projection, authority = _authority("m1_hepatobiliary_missingness")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    rng = np.random.default_rng(20260824)
    n = 420
    bilirubin_first = rng.lognormal(mean=-0.1, sigma=0.6, size=n)
    bilirubin_max = bilirubin_first + rng.gamma(shape=1.2, scale=0.35, size=n)
    age = rng.normal(64.0, 13.0, size=n)
    sex = rng.choice(["F", "M"], size=n)
    components = {
        column: rng.integers(0, 4, size=n).astype(float)
        for column in authority.required_adjustment_columns
        if column not in {"age", "sex"}
    }
    logit = -4.0 + 0.35 * bilirubin_max + 0.018 * (age - 60.0)
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)), size=n)
    frame = pd.DataFrame(
        {
            "bili_max": bilirubin_max,
            "bili_first": bilirubin_first,
            "death": death,
            "death_time": np.where(
                death == 1, rng.uniform(30.0, 180.0, size=n), np.nan
            ),
            "los_icu": rng.uniform(1.2, 8.0, size=n),
            "age": age,
            "sex": sex,
            **components,
        }
    )

    summary = run_landmark_spline_association(
        frame=frame,
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path,
    )

    assert set(summary["output_files"]) == set(authority.plan_outputs)
    definitions = pd.read_csv(tmp_path / "m1_exposure_definition_sensitivity.csv")
    assert definitions["exposure_column"].tolist() == ["bili_max", "bili_first"]
    assert definitions["is_primary_definition"].tolist() == [True, False]
    curve = pd.read_csv(tmp_path / "m1_landmark_bilirubin_curve.csv")
    assert {"exposure_value", "reference_exposure_value"}.issubset(curve.columns)


def test_e2_runtime_authority_mechanically_compiles_the_primary_draft() -> None:
    _projection, authority = _authority("e2_lactate_mortality")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    exact = _e2_plan(authority)
    draft_step = exact.steps[0].model_copy(
        update={
            "method": "adjusted_association_models",
            "intent": "Estimate a generic adjusted association.",
            "expected_outputs": ["table:adjusted_association_estimates"],
            "scientific_capability": "association_adjusted_v1",
            "icu_rule_refs": [],
        }
    )
    consumer = exact.steps[0].model_copy(
        update={
            "step_id": "02_display",
            "planned_analysis_role": "auxiliary",
            "method": "visualization",
            "intent": "Render the signed primary result.",
            "inputs": ["table:adjusted_association_estimates"],
            "expected_outputs": ["figure:primary_result"],
            "scientific_capability": None,
            "icu_rule_refs": [],
        }
    )
    draft = exact.model_copy(update={"steps": [draft_step, consumer]})

    bound, findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(draft)

    authority.validate_plan(bound)
    assert bound.steps[0].method == authority.plan_method
    assert bound.steps[0].expected_outputs == list(authority.plan_outputs)
    assert set(authority.required_columns).issubset(bound.steps[0].inputs)
    assert bound.steps[1].inputs == [authority.downstream_parent_product]
    assert findings[0].detail["reason_code"] == "landmark_spline_host_compiled"


def test_e2_runtime_clears_rebound_binary_sensitivity_capability(
    tmp_path: Path,
) -> None:
    projection, authority = _authority("e2_lactate_mortality")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    exact = _e2_plan(authority)
    draft_step = exact.steps[0].model_copy(
        update={
            "method": "adjusted_association_models",
            "intent": "Estimate a generic adjusted association.",
            "expected_outputs": ["table:adjusted_association_estimates"],
            "scientific_capability": "association_adjusted_v1",
            "icu_rule_refs": [],
        }
    )
    sensitivity = exact.steps[0].model_copy(
        update={
            "step_id": "02_sensitivity",
            "planned_analysis_role": "sensitivity",
            "method": "prespecified_functional_form_check",
            "intent": "Check the declared functional form.",
            "inputs": ["table:adjusted_association_estimates"],
            "expected_outputs": ["table:functional_form_check"],
            "scientific_capability": "association_freeform_v1",
            "sensitivity_spec_ids": ["functional_form_check"],
            "icu_rule_refs": [],
        }
    )
    draft = exact.model_copy(update={"steps": [draft_step, sensitivity]})

    bound = authority.bind_plan(draft)

    rebound = bound.steps[1]
    assert rebound.scientific_capability is None
    assert rebound.inputs == [
        authority.downstream_parent_product,
        authority.linear_sensitivity_product,
    ]
    AnalysisPlan.model_validate(bound.model_dump(mode="json"))
    selected = select_standard_executor(
        rebound,
        plan=bound,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=(projection.runtime_projection_sha256),
    )
    assert selected is not None
    assert selected.analysis_kind == LANDMARK_SPLINE_FUNCTIONAL_FORM_ANALYSIS_KIND
    summary = run_landmark_spline_functional_form(
        step=rebound,
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        linear_sensitivity=pd.DataFrame(
            [
                {
                    "n": 44095,
                    "events": 6200,
                    "linear_aic": 310.0,
                    "spline_aic": 300.0,
                    "linear_bic": 340.0,
                    "spline_bic": 338.0,
                    "likelihood_ratio_statistic": 14.0,
                    "additional_spline_parameters": 2,
                    "nonlinearity_p_value": 0.00091,
                }
            ]
        ),
        linear_evidence_id="table_linear",
        out_dir=tmp_path,
    )
    assert summary["n_complete_case"] == 44095
    projected = pd.read_csv(tmp_path / "functional_form_check.csv")
    assert projected.loc[0, "n_complete_case"] == 44095
    assert projected.loc[0, "nonlinearity_p_value"] == pytest.approx(0.00091)


def test_h1_runtime_compiles_and_executes_one_deterministic_survival_suite(
    tmp_path: Path,
) -> None:
    pytest.importorskip("lifelines")
    projection, authority = _authority("h1_ventilation_survival")
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    bound, findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(_h1_draft_plan())
    authority.validate_plan(bound)
    assert len(bound.steps) == 3
    assert bound.steps[0].method == "host_materialized_locked_cohort"
    assert bound.steps[0].expected_outputs == ["table:analysis_cohort"]
    assert bound.steps[1].method == authority.plan_method
    assert set(bound.steps[1].expected_outputs) == set(authority.analysis_plan_outputs)
    assert bound.steps[2].inputs == list(authority.figure_input_products)
    assert bound.steps[2].expected_outputs == [authority.figure_product]
    assert _typed_plan_dag_findings(bound) == []
    cohort_selection = select_standard_executor(bound.steps[0], plan=bound)
    assert cohort_selection is not None
    assert cohort_selection.analysis_kind == "host_bound_analysis_cohort"
    assert cohort_selection.consumed_input_keys == ()
    assert findings[0].detail["reason_code"] == (
        "landmark_survival_suite_host_compiled"
    )

    execution_only = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).development_execution_only_plan(
        research_question="Run the sealed landmark survival development suite."
    )
    assert execution_only is not None
    execution_only_plan, execution_only_finding = execution_only
    authority.validate_plan(execution_only_plan)
    assert len(execution_only_plan.steps) == 3
    assert _typed_plan_dag_findings(execution_only_plan) == []
    assert execution_only_finding.detail["reason_code"] == (
        "development_execution_only_authority_compiled"
    )
    endpoint = authority.research_context_endpoint()
    assert endpoint.kind == "time_to_event"
    assert endpoint.event_column == "mort_28d"
    assert endpoint.time_column == "followup_days_28d"
    assert execution_only_plan.endpoint == endpoint

    selected = select_standard_executor(
        bound.steps[1],
        plan=bound,
        plausibility_scope=FlagOnlyPlausibilityScope(
            step_id=bound.steps[1].step_id,
            expected_columns=("age",),
            source_contracts_sha256="a" * 64,
            authority_kind="test",
        ),
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
    )
    assert selected is not None
    assert selected.analysis_kind == "signed_landmark_survival_suite"
    assert "analysis_frame = bound.frame" in selected.code
    assert "frame=analysis_frame" in selected.code
    figure_selected = select_standard_executor(
        bound.steps[2],
        plan=bound,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
    )
    assert figure_selected is not None
    assert figure_selected.analysis_kind == "signed_landmark_survival_figure"
    assert figure_selected.host_sealed_renderer is True
    assert effect_output_authorized(
        bound.steps[1],
        step_record={
            "deterministic_standard_analysis": "signed_landmark_survival_suite",
            "deterministic_standard_selection_reason": (
                "signed_landmark_survival_suite_contract_preflight"
            ),
            "standard_executor_candidates": {
                "claimed_by": "signed_landmark_survival_suite"
            },
        },
    )

    rng = np.random.default_rng(20260822)
    n = 900
    exposed = rng.binomial(1, 0.38, n)
    onset = np.where(exposed == 1, rng.uniform(1.0, 23.9, n), np.nan)
    onset[:30] = 0.0
    exposed[:30] = 1
    age = rng.normal(64.0, 14.0, n)
    sex = rng.choice(["Female", "Male"], n)
    charlson = rng.poisson(3.0, n).astype(float)
    sofa2 = rng.integers(0, 18, n).astype(float)
    rate = np.exp(-3.8 + 0.55 * exposed + 0.018 * (age - 64.0))
    event_time = rng.exponential(1.0 / rate)
    event = event_time <= 28.0
    followup = np.minimum(event_time, 28.0)
    frame = pd.DataFrame(
        {
            "mech_vent_max": exposed,
            "mech_vent_first_time": onset,
            "mort_28d": event.astype(int),
            "followup_days_28d": followup,
            "age": age,
            "sex": sex,
            "charlson_first": charlson,
            "sofa2_max": sofa2,
        }
    )
    summary = run_landmark_survival_suite(
        frame=frame,
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path,
        input_product="dataset:analysis_cohort",
        input_evidence_id="sha256:" + "a" * 64,
        input_sha256="a" * 64,
    )
    assert summary["status"] == "ok"
    assert summary["analysis_only"] is True
    assert summary["effect_measure"] == "hazard_ratio"
    assert " versus " in summary["contrast"]
    assert "primary_predictor" not in summary
    assert summary["n_landmark_population"] < n
    assert summary["missingness_measurement_audit"]["source_n"] == n
    assert set(
        summary["missingness_measurement_audit"]["source_missing_n_by_column"]
    ) == set(authority.required_columns)
    assert set(
        summary["missingness_measurement_audit"]["landmark_missing_n_by_model_column"]
    ) == set(authority.adjustment_columns)
    assert set(summary["output_files"]) == set(authority.analysis_plan_outputs)
    assert authority.rmst_product is not None
    assert (tmp_path / "landmark_rmst_summary.csv").is_file()
    assert (tmp_path / "landmark_time_varying_cox_summary.csv").is_file()
    rmst = pd.read_csv(tmp_path / "landmark_rmst_summary.csv")
    assert len(rmst) == 1
    assert rmst.loc[0, "tau_days_from_landmark"] == pytest.approx(27.0)
    assert rmst.loc[0, "ci_low"] <= rmst.loc[0, "rmst_difference_days"]
    assert rmst.loc[0, "rmst_difference_days"] <= rmst.loc[0, "ci_high"]
    time_varying = pd.read_csv(tmp_path / "landmark_time_varying_cox_summary.csv")
    exposure_intervals = time_varying.loc[time_varying["is_exposure"]]
    assert exposure_intervals["interval_start_days"].tolist() == [0.0, 7.0, 14.0]
    assert exposure_intervals["interval_end_days"].tolist() == [7.0, 14.0, 27.0]
    reporting = summary["reportable_survival_results"]
    assert reporting["schema_version"] == "easyicu.survival_reporting/1"
    assert reporting["constant_hazard_ratio_authorized"] is (
        not summary["proportional_hazards_status"].startswith("violation_")
    )
    assert reporting["rmst"]["difference_days"] == pytest.approx(
        rmst.loc[0, "rmst_difference_days"]
    )
    assert [
        row["hazard_ratio"]
        for row in reporting["time_varying_adjusted_association"]["intervals"]
    ] == pytest.approx(exposure_intervals["hazard_ratio"].tolist())
    projection = reporting["manuscript_projection"]
    assert projection["schema_version"] == "easyicu.manuscript_projection/1"
    assert [claim["claim_id"] for claim in projection["claims"]] == [
        "primary_rmst_contrast",
        "time_varying_association_intervals",
    ]
    from easyicu.research_agent.reporting.manuscript_projection import (
        project_owner_issued_manuscript_claims,
    )

    projected, repairs = project_owner_issued_manuscript_claims(
        """## Abstract

**Results:** Owner values omitted.

## Results

### Primary association

Owner values omitted.

### Sensitivity and subgroup analyses

Owner values omitted.
""",
        per_step_records=[
            {
                "step_id": "01_survival",
                "generation_mode": "deterministic_standard",
                "step_summary_evidence_id": "survival_summary",
                "step_summary": summary,
            }
        ],
    )
    assert len(repairs) == 4
    assert f"{reporting['rmst']['p_value']:.6g}" in projected
    for row in reporting["time_varying_adjusted_association"]["intervals"]:
        assert f"{row['p_value']:.6g}" in projected
    assert not (tmp_path / "landmark_survival_suite.svg").exists()
    risk = pd.read_csv(tmp_path / "landmark_risk_set_flow.csv")
    final_count = risk.loc[
        risk["stage"] == "landmark_analysis_population", "count"
    ].item()
    assert final_count == summary["n_landmark_population"]
    assert risk["excluded_since_prior_stage"].sum() >= 30

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    figure_sources = {}
    for product, source_name in (
        (authority.km_product, "landmark_km_curve.csv"),
        (authority.cox_product, "landmark_cox_summary.csv"),
        (authority.risk_set_product, "landmark_risk_set_flow.csv"),
        (authority.ph_product, "landmark_ph_diagnostics.csv"),
        (authority.rmst_product, "landmark_rmst_summary.csv"),
        (
            authority.time_varying_cox_product,
            "landmark_time_varying_cox_summary.csv",
        ),
    ):
        evidence_path = evidence_dir / f"table_step_artifact_deadbeef__{source_name}"
        evidence_path.write_bytes((tmp_path / source_name).read_bytes())
        figure_sources[product] = evidence_path
    figure_dir = tmp_path / "figure"
    figure_summary = run_landmark_survival_figure(
        km_table=pd.read_csv(tmp_path / "landmark_km_curve.csv"),
        cox_table=pd.read_csv(tmp_path / "landmark_cox_summary.csv"),
        rmst_table=rmst,
        time_varying_table=time_varying,
        risk_flow=risk,
        ph_table=pd.read_csv(tmp_path / "landmark_ph_diagnostics.csv"),
        source_paths=figure_sources,
        authority=authority,
        out_dir=figure_dir,
    )
    assert figure_summary["status"] == "ok"
    assert (figure_dir / "landmark_survival_suite.svg").is_file()
    figure_receipt = json.loads(
        (figure_dir / "landmark_survival_figure_runtime_receipt.json").read_text()
    )
    assert figure_receipt["adjustment_columns"] == list(authority.adjustment_columns)
    if summary["proportional_hazards_status"].startswith("violation_"):
        assert figure_receipt["promoted_adjustment_columns"] == list(
            authority.adjustment_columns
        )
        assert figure_receipt["promoted_effect_measure"] == (
            "interval_specific_hazard_ratio"
        )
    else:
        assert figure_receipt["promoted_adjustment_columns"] == list(
            authority.adjustment_columns
        )
        assert figure_receipt["promoted_effect_measure"] == "hazard_ratio"
    assert figure_summary["figure_assets"]["runtime_receipt"] == (
        "landmark_survival_figure_runtime_receipt.json"
    )
    assert set(figure_summary["source_data_files"]) == {
        "landmark_km_curve.csv",
        "landmark_cox_summary.csv",
        "landmark_risk_set_flow.csv",
        "landmark_ph_diagnostics.csv",
        "landmark_rmst_summary.csv",
        "landmark_time_varying_cox_summary.csv",
    }
    contract = json.loads(
        (figure_dir / "landmark_survival_suite.figure_contract.json").read_text()
    )
    assert contract["panels"][0]["title"].startswith("Unadjusted landmark")
    if summary["proportional_hazards_status"].startswith("violation_"):
        assert contract["panels"][1]["title"] == ("Time-varying adjusted association")
        assert contract["panels"][1]["role"] == "survival_effect"
        assert contract["panels"][1]["metadata"]["chart_type"] == (
            "time_varying_hazard_ratio_forest"
        )
        assert "withheld" in contract["core_claim"]
    else:
        assert contract["panels"][1]["role"] == "survival_effect"
    assert contract["panels"][3]["role"] == "diagnostics"
    assert contract["panels"][3]["metadata"]["chart_type"] == "schoenfeld_plot"
    assert "direction can differ" in contract["statistics_note"]

    violation_ph = pd.read_csv(tmp_path / "landmark_ph_diagnostics.csv")
    violation_ph["ph_status"] = "violation_block_paper_authorization"
    violation_ph["paper_authorization_allowed"] = False
    violation_ph.loc[0, "p_value"] = 0.001
    violation_dir = tmp_path / "violation_figure"
    run_landmark_survival_figure(
        km_table=pd.read_csv(tmp_path / "landmark_km_curve.csv"),
        cox_table=pd.read_csv(tmp_path / "landmark_cox_summary.csv"),
        rmst_table=rmst,
        time_varying_table=time_varying,
        risk_flow=risk,
        ph_table=violation_ph,
        source_paths=figure_sources,
        authority=authority,
        out_dir=violation_dir,
    )
    violation_contract = json.loads(
        (violation_dir / "landmark_survival_suite.figure_contract.json").read_text()
    )
    assert violation_contract["panels"][1]["title"] == (
        "Time-varying adjusted association"
    )
    assert violation_contract["panels"][1]["metadata"]["chart_type"] == (
        "time_varying_hazard_ratio_forest"
    )
    assert "landmark_rmst_summary.csv" in violation_contract["source_data"]

    invalid_ph = pd.read_csv(tmp_path / "landmark_ph_diagnostics.csv")
    invalid_ph.loc[0, "p_value"] = float("nan")
    with pytest.raises(ValueError, match="finite p values"):
        run_landmark_survival_figure(
            km_table=pd.read_csv(tmp_path / "landmark_km_curve.csv"),
            cox_table=pd.read_csv(tmp_path / "landmark_cox_summary.csv"),
            rmst_table=rmst,
            time_varying_table=time_varying,
            risk_flow=risk,
            ph_table=invalid_ph,
            source_paths=figure_sources,
            authority=authority,
            out_dir=tmp_path / "invalid_figure",
        )


def test_host_bound_cohort_root_publishes_exact_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _projection, authority = _authority("h1_ventilation_survival")
    assert isinstance(authority, LandmarkSurvivalRuntimeAuthority)
    plan = authority.development_execution_only_plan(
        research_question="Run the sealed landmark survival development suite."
    )
    selected = select_standard_executor(plan.steps[0], plan=plan)
    assert selected is not None

    source = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "outputs"
    pd.DataFrame({"stay_id": [1, 2], "value": [3.0, 4.0]}).to_parquet(
        source,
        index=False,
    )
    monkeypatch.setenv("COHORT_PARQUET", str(source))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))

    exec(compile(selected.code, "<host_bound_cohort>", "exec"), {})

    output = out_dir / "analysis_cohort.parquet"
    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert output.read_bytes() == source.read_bytes()
    assert summary["status"] == "ok"
    assert summary["n_analysis_cohort"] == 2
    assert summary["analysis_cohort_n"] == 2
    assert summary["output_files"] == {
        "table:analysis_cohort": "analysis_cohort.parquet"
    }


def test_landmark_survival_executor_keeps_case_labels_in_authority() -> None:
    from easyicu.research_agent.execution.runners import landmark_survival_executor

    source = inspect.getsource(landmark_survival_executor)
    assert "Incident ventilation" not in source
    assert "ICU stays" not in source


def test_e2_runtime_authority_binds_and_executes_deterministic_robustness(
    tmp_path: Path,
) -> None:
    projection, authority = _authority("e2_lactate_mortality")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    primary = _e2_plan(authority).steps[0]
    robustness = AnalysisPlan.model_validate(
        {
            "research_question": "Project signed sensitivity results.",
            "analysis_type": "association_study",
            "robustness_specs": [
                {
                    "spec_id": "primary_complete_case",
                    "axis": "missing",
                    "description": "Complete cases for the draft primary model.",
                    "missing_override": {
                        "strategy": "complete_case",
                        "variables": [
                            authority.exposure_column,
                            authority.outcome_column,
                            "draft_adjustment",
                        ],
                    },
                }
            ],
            "steps": [
                primary.model_dump(mode="json"),
                {
                    "step_id": "02_robustness",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Summarize signed robustness results.",
                    "inputs": [
                        "dataset:analysis_cohort",
                        "unrelated_raw_covariate",
                    ],
                    "expected_outputs": [
                        "statistic:primary_or",
                        "statistic:complete_case_n",
                        "table:robustness_summary",
                        "log:missingness_strategy_notes",
                        "table:robustness_matrix",
                    ],
                    "method": "robustness_sensitivity",
                    "sensitivity_spec_ids": ["primary_complete_case"],
                    "robustness_replay_spec": {
                        "products": [
                            {"product_id": "primary_or", "output": "primary_effect"},
                            {
                                "product_id": "complete_case_n",
                                "output": "complete_case_n",
                            },
                            {
                                "product_id": "robustness_summary",
                                "output": "robustness_summary",
                            },
                            {
                                "product_id": "missingness_strategy_notes",
                                "output": "missingness_strategy_notes",
                            },
                            {
                                "product_id": "robustness_matrix",
                                "output": "robustness_matrix",
                            },
                        ]
                    },
                },
                {
                    "step_id": "03_absolute_risk",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Describe absolute-risk context.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:absolute_risk_context"],
                    "method": "descriptive_statistics",
                },
                {
                    "step_id": "04_measurement_audit",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Audit the measurement process.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:measurement_process"],
                    "method": "missing_data",
                },
                {
                    "step_id": "05_robustness_figure",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render the registered evidence.",
                    "inputs": [
                        "statistic:primary_or",
                        "table:robustness_summary",
                    ],
                    "expected_outputs": ["figure:robustness_plot"],
                    "method": "visualization",
                    "input_consumption_contracts": [
                        {
                            "input_key": "table:robustness_summary",
                            "mode": "all_rows",
                        }
                    ],
                },
                {
                    "step_id": "06_article_display",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render the primary association with article context.",
                    "inputs": [
                        "table:absolute_risk_context",
                        "table:adjusted_association_estimates",
                        "table:robustness_summary",
                    ],
                    "expected_outputs": ["figure:article_display"],
                    "method": "visualization",
                },
            ],
        }
    )
    bound = authority.bind_plan(robustness)
    step = bound.steps[1]
    assert authority.downstream_parent_product in step.inputs
    assert authority.linear_sensitivity_product in step.inputs
    assert step.inputs == [
        authority.downstream_parent_product,
        authority.linear_sensitivity_product,
    ]
    assert {item.input_key for item in step.input_consumption_contracts} == {
        authority.downstream_parent_product,
        authority.linear_sensitivity_product,
    }
    assert bound.robustness_specs[0].missing_override == {
        "strategy": "complete_case",
        "variables": list(authority.model_complete_case_columns),
    }
    robustness_figure = bound.steps[4]
    assert robustness_figure.inputs == [
        "statistic:primary_or",
        "table:robustness_summary",
    ]
    figure = bound.steps[5]
    assert figure.inputs == [
        authority.curve_product,
        authority.adjusted_absolute_risk_product,
        "table:robustness_summary",
        "table:measurement_process",
    ]
    assert {item.input_key for item in figure.input_consumption_contracts} == set(
        figure.inputs
    )
    assert [panel.article_role for panel in figure.figure_panels] == [
        "primary_estimand",
        "descriptive_result",
        "robustness",
        "data_quality",
    ]

    selected = select_standard_executor(
        step,
        plan=bound,
        current_case_scientific_runtime_authority=authority,
        scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
    )
    assert selected is not None
    assert selected.analysis_kind == "signed_landmark_spline_robustness"

    contrasts = pd.DataFrame(
        {
            "exposure_value": [1.0, 5.0],
            "reference_value": [2.1, 2.1],
            "adjusted_odds_ratio": [0.8, 2.0],
            "ci_low": [0.7, 1.8],
            "ci_high": [0.9, 2.2],
        }
    )
    linear = pd.DataFrame(
        {
            "per_unit": [1.0],
            "adjusted_odds_ratio": [1.25],
            "ci_low": [1.2],
            "ci_high": [1.3],
            "n": [44095],
            "events": [5480],
        }
    )
    summary = run_landmark_spline_robustness(
        step=step,
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        contrasts=contrasts,
        linear_sensitivity=linear,
        contrast_evidence_id="contrast_evidence",
        linear_evidence_id="linear_evidence",
        out_dir=tmp_path,
        complete_case_spec_id="primary_complete_case",
        input_bindings=[
            {
                "input_key": authority.downstream_parent_product,
                "evidence_id": "contrast_evidence",
                "sha256": "a" * 64,
                "loaded": True,
                "row_count": 2,
            },
            {
                "input_key": authority.linear_sensitivity_product,
                "evidence_id": "linear_evidence",
                "sha256": "b" * 64,
                "loaded": True,
                "row_count": 1,
            },
        ],
    )
    assert summary["status"] == "ok"
    assert summary["primary_or"] == 2.0
    assert summary["primary_effect_is_nonlinear_curve_summary"] is False
    assert "exposure_value=5" in summary["primary_effect_label"]
    assert "reference_value=2.1" in summary["primary_effect_label"]
    assert summary["complete_case_n"] == 44095
    assert len(summary["input_bindings"]) == 2
    matrix = pd.read_csv(tmp_path / "robustness_matrix.csv")
    assert set(matrix["axis"]) == {"primary", "functional_form", "missing"}
    assert (
        matrix.loc[matrix["axis"] == "missing", "spec_id"].item()
        == "primary_complete_case"
    )
    assert matrix.loc[matrix["axis"] == "missing", "independent_variant"].item() in (
        False,
        0,
    )


def test_e2_runtime_authority_rejects_missing_referenced_complete_case_spec() -> None:
    _projection, authority = _authority("e2_lactate_mortality")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    primary = _e2_plan(authority).steps[0]
    plan = AnalysisPlan.model_validate(
        {
            "research_question": "Project signed sensitivity results.",
            "analysis_type": "association_study",
            "robustness_specs": [
                {
                    "spec_id": "unreferenced_complete_case",
                    "axis": "missing",
                    "description": "A complete-case spec not selected by the step.",
                    "missing_override": {
                        "strategy": "complete_case",
                        "variables": list(authority.model_complete_case_columns),
                    },
                }
            ],
            "steps": [
                primary.model_dump(mode="json"),
                {
                    "step_id": "02_robustness",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Summarize signed robustness results.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:robustness_summary"],
                    "method": "robustness_sensitivity",
                    "sensitivity_spec_ids": ["unrelated_robustness_axis"],
                    "robustness_replay_spec": {
                        "products": [
                            {
                                "product_id": "robustness_summary",
                                "output": "robustness_summary",
                            }
                        ]
                    },
                },
            ],
        }
    )

    with pytest.raises(
        CurrentCaseScientificAuthorityError,
        match="exactly one referenced complete-case specification",
    ):
        authority.bind_plan(plan)


def test_landmark_authority_migrates_one_legacy_missingness_axis() -> None:
    _projection, authority = _authority("m1_hepatobiliary_missingness")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    primary = _e2_plan(authority).steps[0]
    plan = AnalysisPlan.model_validate(
        {
            "research_question": "Project the signed legacy sensitivity.",
            "analysis_type": "association_study",
            "robustness_specs": [
                {
                    "spec_id": "primary_complete_case_replay",
                    "axis": "missing",
                    "description": "Complete cases for the primary model.",
                    "missing_override": {
                        "strategy": "complete_case",
                        "variables": ["draft_variable"],
                    },
                }
            ],
            "steps": [
                primary.model_dump(mode="json"),
                {
                    "step_id": "02_legacy_robustness",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Retain the legacy missingness axis.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:robustness_summary"],
                    "method": "robustness_sensitivity",
                    "sensitivity_spec_ids": ["informative_measurement_missingness"],
                    "robustness_replay_spec": {
                        "products": [
                            {
                                "product_id": "robustness_summary",
                                "output": "robustness_summary",
                            }
                        ]
                    },
                },
            ],
        }
    )

    bound = authority.bind_plan(plan)

    assert bound.steps[1].sensitivity_spec_ids == [
        "informative_measurement_missingness",
        "primary_complete_case_replay",
    ]
    assert bound.robustness_specs[0].missing_override == {
        "strategy": "complete_case",
        "variables": list(authority.model_complete_case_columns),
    }


def test_landmark_authority_prefers_four_table_hero_over_robustness_figure() -> None:
    _projection, authority = _authority("m1_hepatobiliary_missingness")
    assert isinstance(authority, LandmarkSplineRuntimeAuthority)
    primary = _e2_plan(authority).steps[0]
    plan = AnalysisPlan.model_validate(
        {
            "research_question": "Render robustness and a primary display.",
            "analysis_type": "association_study",
            "steps": [
                primary.model_dump(mode="json"),
                {
                    "step_id": "02_absolute_risk",
                    "intent": "Report absolute-risk context.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:absolute_risk_context"],
                    "method": "descriptive_statistics",
                },
                {
                    "step_id": "03_measurement",
                    "intent": "Audit measurement.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:measurement_process_audit"],
                    "method": "missing_data",
                },
                {
                    "step_id": "04_robustness",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Summarize robustness.",
                    "inputs": ["dataset:analysis_cohort"],
                    "expected_outputs": ["table:robustness_summary"],
                    "method": "robustness_sensitivity",
                },
                {
                    "step_id": "05_robustness_figure",
                    "intent": "Render robustness only.",
                    "inputs": ["table:robustness_summary"],
                    "expected_outputs": ["figure:robustness_plot"],
                    "method": "visualization",
                },
                {
                    "step_id": "06_publication_figure",
                    "intent": "Render the four-table publication display.",
                    "inputs": [
                        "table:adjusted_association_estimates",
                        "table:absolute_risk_context",
                        "table:robustness_summary",
                        "table:measurement_process_audit",
                    ],
                    "expected_outputs": ["figure:publication_figure"],
                    "method": "visualization",
                },
            ],
        }
    )

    bound = authority.bind_plan(plan)
    robustness_figure = next(
        step for step in bound.steps if step.step_id == "05_robustness_figure"
    )
    publication_figure = next(
        step for step in bound.steps if step.step_id == "06_publication_figure"
    )

    assert robustness_figure.inputs == ["table:robustness_summary"]
    assert publication_figure.inputs == [
        authority.curve_product,
        authority.adjusted_absolute_risk_product,
        "table:robustness_summary",
        "table:measurement_process_audit",
    ]
    assert len(publication_figure.figure_panels) == 4


def test_h2_plan_forbids_effect_work_and_runtime_emits_no_estimate(
    tmp_path: Path,
) -> None:
    projection, authority = _authority("h2_vasopressor_causal")
    assert isinstance(authority, SourceFeasibilityRuntimeAuthority)
    plan = _h2_plan(authority)
    authority.validate_plan(plan)
    execution_only_plan = authority.development_execution_only_plan(
        research_question="Record whether the current source identifies the contrast."
    )
    authority.validate_plan(execution_only_plan)
    assert len(execution_only_plan.steps) == 1
    assert execution_only_plan.steps[0].planned_analysis_role == "auxiliary"
    assert execution_only_plan.steps[0].model_requirements == []
    assert execution_only_plan.steps[0].family_primary_result_requirement is None
    assert not any(
        step.planned_analysis_role == "primary" for step in execution_only_plan.steps
    )
    projected = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).development_execution_only_plan(
        research_question="Record whether the current source identifies the contrast."
    )
    assert projected is not None
    projected_plan, projected_finding = projected
    authority.validate_plan(projected_plan)
    assert projected_finding.detail["reason_code"] == (
        "source_feasibility_development_execution_only_authority_compiled"
    )

    generic_article_step = plan.steps[0].model_copy(
        update={
            "step_id": "02_generic_article_figure",
            "method": "visualization",
            "intent": "Add a generic article figure.",
            "expected_outputs": ["figure:generic_article_figure"],
        }
    )
    rebound, rebound_findings = ScientificRuntimeAuthorities(
        trajectory=None,
        current_case=authority,
    ).bind_plan(
        plan.model_copy(update={"steps": [plan.steps[0], generic_article_step]})
    )
    authority.validate_plan(rebound)
    assert len(rebound.steps) == 1
    assert rebound.steps[0].method == authority.plan_method
    assert rebound_findings[0].detail["reason_code"] == (
        "source_feasibility_fail_closed_host_compiled"
    )

    forbidden = plan.steps[0].model_copy(
        update={
            "step_id": "02_psm",
            "method": "propensity_score_matching",
            "intent": "Construct a control arm and estimate a causal effect.",
        }
    )
    with pytest.raises(CurrentCaseScientificAuthorityError, match="forbidden"):
        authority.validate_plan(
            plan.model_copy(update={"steps": [plan.steps[0], forbidden]})
        )

    source_out_dir = tmp_path / "steps" / plan.steps[0].step_id / "outputs"
    summary = run_source_feasibility_fail_closed(
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=source_out_dir,
    )
    assert summary["status"] == "ok"
    assert summary["scientific_decision"] == "blocked_by_source_authority"
    assert summary["reason_code"] == "H2_VERIFIED_NON_USE_UNAVAILABLE"
    assert summary["effect_estimate"] is None
    table = pd.read_csv(source_out_dir / "h2_source_feasibility.csv")
    assert table.loc[0, "causal_contrast_authorized"] in (False, 0)
    assert pd.isna(table.loc[0, "effect_estimate"])

    record = {
        "step_id": plan.steps[0].step_id,
        "status": "ok",
        "generation_mode": "deterministic_standard",
        "deterministic_standard_analysis": SOURCE_FEASIBILITY_ANALYSIS_KIND,
        "step_summary": summary,
    }
    assert (
        source_feasibility_runtime_bundle_errors(
            plan=plan,
            records=[record],
            run_dir=tmp_path,
        )
        == []
    )
    verdict = resolve_primary_capability(
        analysis_type=plan.analysis_type,
        plan=plan,
    )
    assert verdict.capability_id == SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID
    assert verdict.execution_owner == "host_deterministic"
    wrong_family = plan.model_copy(update={"analysis_type": "association_study"})
    wrong_verdict = resolve_primary_capability(
        analysis_type=wrong_family.analysis_type,
        plan=wrong_family,
    )
    assert wrong_verdict.failure_reason == "source_feasibility_family_mismatch"
    assessment = assess_scientific_capability(
        analysis_type=plan.analysis_type,
        context=ResearchContext(
            research_question=plan.research_question,
            variables=[],
            cohort=CohortDescriptor(
                cohort_name="source_audit",
                database="synthetic",
                n_patients=1,
                n_stays=1,
            ),
        ),
        plan=plan,
    )
    assert assessment.capability_id == SOURCE_FEASIBILITY_NON_USE_CAPABILITY_ID
    assert assessment.claim_ceiling == "reportable"

    tampered = deepcopy(record)
    tampered["step_summary"]["scientific_runtime_receipt"]["effect_estimate"] = 1.0
    assert (
        "runtime receipt is invalid"
        in source_feasibility_runtime_bundle_errors(
            plan=plan,
            records=[tampered],
            run_dir=tmp_path,
        )[0]
    )

    gates = _compute_readiness_gates(
        context=ResearchContext(
            research_question=plan.research_question,
            variables=[],
            cohort=CohortDescriptor(
                cohort_name="source_audit",
                database="synthetic",
                n_patients=1,
                n_stays=1,
            ),
        ),
        plan=plan,
        per_step_records=[record],
        findings=[],
        evidence=EvidenceStore(tmp_path),
        run_dir=tmp_path,
        manuscript_path=tmp_path / "manuscript.md",
        stop_after_analysis=True,
    )
    assert gates["execution_complete"] is True
    assert gates["analysis_validated"] is True
    assert gates["paper_authorized"] is False


def test_signed_current_case_contracts_are_selected_by_the_real_execution_router() -> (
    None
):
    for task_id, expected_kind, plan_factory in (
        (
            "e2_lactate_mortality",
            "signed_landmark_spline_association",
            _e2_plan,
        ),
        (
            "h2_vasopressor_causal",
            "signed_source_feasibility_fail_closed",
            _h2_plan,
        ),
    ):
        projection, authority = _authority(task_id)
        plan = plan_factory(authority)
        selected = select_standard_executor(
            plan.steps[0],
            plan=plan,
            current_case_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256=(projection.runtime_projection_sha256),
        )
        assert selected is not None
        assert selected.analysis_kind == expected_kind
        assert authority.execution_contract_sha256 in selected.code
        assert projection.runtime_projection_sha256 in selected.code

        rebuilt_step = plan.steps[0].model_copy(deep=True)
        assert rebuilt_step is not plan.steps[0]
        rebuilt = select_standard_executor(
            rebuilt_step,
            plan=plan,
            current_case_scientific_runtime_authority=authority,
            scientific_runtime_projection_sha256=(projection.runtime_projection_sha256),
        )
        assert rebuilt is not None
        assert rebuilt.analysis_kind == expected_kind


def test_pipeline_config_requires_the_signed_contract_and_projection_as_a_pair(
    tmp_path: Path,
) -> None:
    projection, authority = _authority("e2_lactate_mortality")
    config = PipelineConfig(
        workdir=tmp_path,
        current_case_scientific_runtime_authority=authority.model_dump(mode="json"),
        scientific_runtime_projection_sha256=projection.runtime_projection_sha256,
    )
    assert config.current_case_scientific_runtime_authority is not None

    with pytest.raises(ValueError, match="configured together"):
        PipelineConfig(
            workdir=tmp_path,
            current_case_scientific_runtime_authority=authority.model_dump(mode="json"),
        )
