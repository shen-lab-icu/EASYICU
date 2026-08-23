from __future__ import annotations

import inspect
import json
from pathlib import Path

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
from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.execution.runners.landmark_spline_executor import (
    run_landmark_spline_association,
)
from easyicu.research_agent.execution.runners.landmark_spline_robustness_executor import (
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
from easyicu.research_agent.schema import AnalysisPlan


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
                    "scientific_capability": "association_freeform_v1",
                    "icu_rule_refs": [authority.plan_rule_ref],
                }
            ],
        }
    )


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
    probability = 1.0 / (
        1.0 + np.exp(-(-3.6 + 0.45 * lactate + 0.018 * (age - 60.0)))
    )
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
    assert receipt["execution_contract_sha256"] == (
        authority.execution_contract_sha256
    )
    assert receipt["runtime_projection_sha256"] == (
        projection.runtime_projection_sha256
    )
    assert summary["n_primary_population"] == n
    assert summary["n_complete_case"] == n - 1


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
    assert summary["n_landmark_population"] < n
    assert set(summary["output_files"]) == set(authority.analysis_plan_outputs)
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
    ):
        evidence_path = evidence_dir / f"table_step_artifact_deadbeef__{source_name}"
        evidence_path.write_bytes((tmp_path / source_name).read_bytes())
        figure_sources[product] = evidence_path
    figure_dir = tmp_path / "figure"
    figure_summary = run_landmark_survival_figure(
        km_table=pd.read_csv(tmp_path / "landmark_km_curve.csv"),
        cox_table=pd.read_csv(tmp_path / "landmark_cox_summary.csv"),
        risk_flow=risk,
        source_paths=figure_sources,
        authority=authority,
        out_dir=figure_dir,
    )
    assert figure_summary["status"] == "ok"
    assert (figure_dir / "landmark_survival_suite.svg").is_file()
    assert set(figure_summary["source_data_files"]) == {
        "landmark_km_curve.csv",
        "landmark_cox_summary.csv",
        "landmark_risk_set_flow.csv",
    }


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
            "steps": [
                primary.model_dump(mode="json"),
                {
                    "step_id": "02_robustness",
                    "planned_analysis_role": "sensitivity",
                    "intent": "Summarize signed robustness results.",
                    "inputs": [
                        "dataset:analysis_cohort",
                        "table:adjusted_association_estimates",
                    ],
                    "expected_outputs": [
                        "statistic:primary_or",
                        "statistic:complete_case_n",
                        "table:robustness_summary",
                        "log:missingness_strategy_notes",
                        "table:robustness_matrix",
                    ],
                    "method": "robustness_sensitivity",
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
            ],
        }
    )
    bound = authority.bind_plan(robustness)
    step = bound.steps[1]
    assert authority.downstream_parent_product in step.inputs
    assert authority.linear_sensitivity_product in step.inputs
    assert {item.input_key for item in step.input_consumption_contracts} == {
        authority.downstream_parent_product,
        authority.linear_sensitivity_product,
    }

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
    assert summary["complete_case_n"] == 44095
    assert len(summary["input_bindings"]) == 2
    matrix = pd.read_csv(tmp_path / "robustness_matrix.csv")
    assert set(matrix["axis"]) == {"primary", "functional_form", "missing"}
    assert matrix.loc[matrix["axis"] == "missing", "independent_variant"].item() in (
        False,
        0,
    )


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
    ).bind_plan(plan.model_copy(update={"steps": [plan.steps[0], generic_article_step]}))
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

    summary = run_source_feasibility_fail_closed(
        authority=authority,
        runtime_projection_sha256=projection.runtime_projection_sha256,
        out_dir=tmp_path,
    )
    assert summary["status"] == "ok"
    assert summary["scientific_decision"] == "blocked_by_source_authority"
    assert summary["reason_code"] == "H2_VERIFIED_NON_USE_UNAVAILABLE"
    assert summary["effect_estimate"] is None
    table = pd.read_csv(tmp_path / "h2_source_feasibility.csv")
    assert table.loc[0, "causal_contrast_authorized"] in (False, 0)
    assert pd.isna(table.loc[0, "effect_estimate"])


def test_signed_current_case_contracts_are_selected_by_the_real_execution_router() -> None:
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
            scientific_runtime_projection_sha256=(
                projection.runtime_projection_sha256
            ),
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
            scientific_runtime_projection_sha256=(
                projection.runtime_projection_sha256
            ),
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
            current_case_scientific_runtime_authority=authority.model_dump(
                mode="json"
            ),
        )
