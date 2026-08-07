"""The primary survival result is computed and receipt-bound by the host."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from easyicu.research_agent.contracts.survival import (
    SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
    SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
    SURVIVAL_PRIMARY_OWNER,
)
from easyicu.research_agent.contracts.survival_execution import (
    SURVIVAL_PRIMARY_ANALYSIS_KIND,
)
from easyicu.research_agent.execution.runners import survival_primary_executor as owner
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.gates.family_primary_result import (
    family_primary_result_reconciliation_findings,
)
from easyicu.research_agent.planning.primary_result_contract import (
    validate_required_primary_result,
)
from easyicu.research_agent.robustness.primary_effect import (
    _extract_primary_effect_payload_from_summary,
    _primary_effect_payload_is_complete,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    EndpointSpec,
    FamilyPrimaryResultRequirement,
    ResearchContext,
    VariableRole,
)


def _requirement() -> FamilyPrimaryResultRequirement:
    return FamilyPrimaryResultRequirement(
        analysis_family="survival",
        exposure_source="treatment",
        outcome="death",
        expected_result_product="table:survival_effect_estimates",
        input_product="table:analysis_cohort",
        estimator="cox_proportional_hazards",
        effect_scale="hazard_ratio",
        uncertainty_method="wald_95ci",
        population="eligible cohort",
        time_origin="ICU admission",
        time_column="follow_up_days",
        time_unit="days",
        event_column="death",
        event_value=1,
        event_definition="death",
        censoring_strategy="administrative at 28 days",
        competing_risk_strategy="none",
        time_horizon="28 days",
        time_horizon_value=28,
        effect_measure="hazard ratio",
        proportional_hazards_diagnostic="schoenfeld_global_test",
        proportional_hazards_alpha=0.05,
        proportional_hazards_policy="block_paper_authorization",
        covariates=["age"],
        model_terms=[
            {
                "name": "treatment",
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
        exposure_encoding="declared_model_terms",
        missing_data_policy="complete_case",
    )


def _step() -> AnalysisStep:
    return AnalysisStep(
        step_id="04_primary",
        intent="Estimate the declared survival contrast.",
        inputs=["table:analysis_cohort"],
        expected_outputs=[
            "table:survival_effect_estimates",
            SURVIVAL_PH_DIAGNOSTIC_PRODUCT,
            SURVIVAL_ANALYSIS_RECEIPT_PRODUCT,
        ],
        method="cox_proportional_hazards",
        planned_analysis_role="primary",
        family_primary_result_requirement=_requirement(),
    )


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Does treatment change 28-day mortality?",
        cohort=CohortDescriptor(
            cohort_name="cohort", database="synthetic", n_patients=8, n_stays=8
        ),
        variables=[
            ConceptDescriptor(
                name="treatment", role=VariableRole.INTERVENTION, dtype="int64"
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(
                name="follow_up_days",
                role=VariableRole.TIME,
                dtype="float64",
                unit="days",
            ),
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"
            ),
        ],
        primary_exposure="treatment",
        target_outcome="death",
        endpoint=EndpointSpec(
            name="death",
            kind="time_to_event",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
            event_column="death",
            time_column="follow_up_days",
            time_origin="ICU admission",
            censoring_rule="administrative at 28 days",
        ),
    )


def _runtime_kwargs(
    requirement: FamilyPrimaryResultRequirement,
    *,
    input_path,
    input_sha256: str,
) -> dict:
    return {
        "input_path": input_path,
        "input_product": str(requirement.input_product),
        "input_evidence_id": "evidence-analysis-cohort",
        "input_sha256": input_sha256,
        "result_product": requirement.expected_result_product,
        "exposure": requirement.exposure_source,
        "outcome": requirement.outcome,
        "effect_scale": requirement.effect_scale,
        "population": requirement.population,
        "time_origin": str(requirement.time_origin),
        "time_column": str(requirement.time_column),
        "time_unit": str(requirement.time_unit),
        "event_column": str(requirement.event_column),
        "event_value": int(requirement.event_value),
        "event_definition": str(requirement.event_definition),
        "censoring_strategy": str(requirement.censoring_strategy),
        "competing_risk_strategy": str(requirement.competing_risk_strategy),
        "time_horizon": str(requirement.time_horizon),
        "time_horizon_value": float(requirement.time_horizon_value),
        "estimator": requirement.estimator,
        "effect_measure": str(requirement.effect_measure),
        "covariates": list(requirement.covariates or ()),
        "model_terms": list(requirement.model_terms or ()),
        "ph_diagnostic": str(requirement.proportional_hazards_diagnostic),
        "ph_alpha": float(requirement.proportional_hazards_alpha),
        "ph_policy": str(requirement.proportional_hazards_policy),
    }


def test_primary_survival_contract_selects_a_fully_sealed_host_executor() -> None:
    step = _step()
    plan = AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="survival",
        steps=[step],
    )

    validate_required_primary_result(plan=plan, context=_context())
    selected = select_standard_executor(step, plan=plan)
    scaffold = owner.survival_primary_executor_scaffold(step)

    assert selected is not None
    assert selected.analysis_kind == SURVIVAL_PRIMARY_ANALYSIS_KIND
    assert selected.consumed_input_keys == ("table:analysis_cohort",)
    assert scaffold.body == ""
    assert scaffold.host_regions_intact(scaffold.assembled())
    assert "run_survival_primary" in selected.code


def test_primary_survival_execution_binds_all_evidence_digests(
    tmp_path, monkeypatch
) -> None:
    frame = pd.DataFrame(
        {
            "follow_up_days": [5.0, 7.0, 10.0, 14.0, 21.0, 35.0, 40.0, 12.0],
            "death": [1, 0, 1, 0, 1, 1, 0, 0],
            "treatment": [0, 1, 1, 0, 1, 0, 1, 0],
            "age": [50, 55, 60, 65, 70, 75, 80, 58],
        }
    )
    input_path = tmp_path / "analysis_cohort.csv"
    frame.to_csv(input_path, index=False)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    out_dir = tmp_path / "out"
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setattr(
        owner,
        "_fit_declared_cox",
        lambda *_args, **_kwargs: owner._CoxEstimate(
            hazard_ratio=0.81,
            ci_low=0.66,
            ci_high=0.98,
            standard_error=0.1,
            coefficient=-0.210721,
            p_value=0.03,
        ),
    )
    monkeypatch.setattr(
        owner,
        "ph_test",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {"covariate": "treatment", "test_statistic": 0.3, "p_value": 0.58},
                {"covariate": "age", "test_statistic": 0.2, "p_value": 0.65},
                {"covariate": "global", "test_statistic": 0.3, "p_value": 1.0},
            ]
        ),
    )
    monkeypatch.setattr(
        owner,
        "_package_versions",
        lambda: {"easyicu": "1.0.0", "lifelines": "0.30.3", "pandas": "2.0.0"},
    )
    requirement = _requirement()

    summary = owner.run_survival_primary(
        input_path=input_path,
        input_product=str(requirement.input_product),
        input_evidence_id="evidence-analysis-cohort",
        input_sha256=input_sha256,
        result_product=requirement.expected_result_product,
        exposure=requirement.exposure_source,
        outcome=requirement.outcome,
        effect_scale=requirement.effect_scale,
        population=requirement.population,
        time_origin=str(requirement.time_origin),
        time_column=str(requirement.time_column),
        time_unit=str(requirement.time_unit),
        event_column=str(requirement.event_column),
        event_value=int(requirement.event_value),
        event_definition=str(requirement.event_definition),
        censoring_strategy=str(requirement.censoring_strategy),
        competing_risk_strategy=str(requirement.competing_risk_strategy),
        time_horizon=str(requirement.time_horizon),
        time_horizon_value=float(requirement.time_horizon_value),
        estimator=requirement.estimator,
        effect_measure=str(requirement.effect_measure),
        covariates=list(requirement.covariates or ()),
        model_terms=list(requirement.model_terms or ()),
        ph_diagnostic=str(requirement.proportional_hazards_diagnostic),
        ph_alpha=float(requirement.proportional_hazards_alpha),
        ph_policy=str(requirement.proportional_hazards_policy),
    )

    receipt_path = out_dir / summary["output_files"][SURVIVAL_ANALYSIS_RECEIPT_PRODUCT]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = out_dir / summary["output_files"][requirement.expected_result_product]
    ph_path = out_dir / summary["output_files"][SURVIVAL_PH_DIAGNOSTIC_PRODUCT]
    assert receipt["issuer"] == SURVIVAL_PRIMARY_OWNER
    assert receipt["input_sha256"] == input_sha256
    assert (
        receipt["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
    )
    assert (
        receipt["ph_diagnostic_sha256"]
        == hashlib.sha256(ph_path.read_bytes()).hexdigest()
    )
    assert receipt["result_evidence_id"] == f"sha256:{receipt['result_sha256']}"
    assert receipt["ph_diagnostic_evidence_id"] == (
        f"sha256:{receipt['ph_diagnostic_sha256']}"
    )
    assert len(receipt["analysis_frame_sha256"]) == 64
    assert receipt["n_censored_at_horizon"] == 2
    assert receipt["formula"] == (
        "Surv(follow_up_days, death==1) ~ treatment__is_1 + age"
    )
    assert receipt["proportional_hazards_status"] == "not_rejected"
    assert receipt["paper_authorization_allowed"] is True
    primary_payload = _extract_primary_effect_payload_from_summary(
        summary,
        path=None,
        preferred_predictor="treatment",
    )
    assert _primary_effect_payload_is_complete(primary_payload)
    assert primary_payload["effect_measure"] == "HR"

    step = _step()
    plan = AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="survival",
        steps=[step],
    )
    assert (
        family_primary_result_reconciliation_findings(
            step=step,
            plan=plan,
            context=_context(),
            step_summary=summary,
            out_dir=out_dir,
        )
        == []
    )

    result_path.write_text(result_path.read_text() + "\n", encoding="utf-8")
    findings = family_primary_result_reconciliation_findings(
        step=step,
        plan=plan,
        context=_context(),
        step_summary=summary,
        out_dir=out_dir,
    )
    assert [finding.detail["issue"] for finding in findings] == [
        "survival_host_receipt_binding_mismatch"
    ]


def test_survival_uses_declared_string_categorical_covariate(
    tmp_path, monkeypatch
) -> None:
    frame = pd.DataFrame(
        {
            "follow_up_days": [5, 7, 10, 14, 21, 35, 40, 12],
            "death": [1, 0, 1, 0, 1, 1, 0, 0],
            "treatment": [0, 1, 1, 0, 1, 0, 1, 0],
            "age": [50, 55, 60, 65, 70, 75, 80, 58],
            "sex": ["female", "male"] * 4,
        }
    )
    input_path = tmp_path / "analysis_cohort.csv"
    frame.to_csv(input_path, index=False)
    payload = _requirement().model_dump(mode="json")
    payload["covariates"] = ["age", "sex"]
    payload["model_terms"].append(
        {
            "name": "sex",
            "role": "covariate",
            "coding": "categorical",
            "levels": ["female", "male"],
            "reference_level": "female",
            "transform": "treatment_contrast",
        }
    )
    requirement = FamilyPrimaryResultRequirement.model_validate(payload)
    captured = {}

    def _fit(analysis, **_kwargs):
        captured["analysis"] = analysis.copy()
        return owner._CoxEstimate(0.81, 0.66, 0.98, 0.1, -0.210721, 0.03)

    def _ph(*_args, covariates, **_kwargs):
        return pd.DataFrame(
            [
                {"covariate": item, "test_statistic": 0.2, "p_value": 0.8}
                for item in covariates
            ]
            + [{"covariate": "global", "test_statistic": 0.4, "p_value": 0.9}]
        )

    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(owner, "_fit_declared_cox", _fit)
    monkeypatch.setattr(owner, "ph_test", _ph)
    monkeypatch.setattr(
        owner,
        "_package_versions",
        lambda: {"easyicu": "1.0.0", "lifelines": "0.30.3", "pandas": "2.0.0"},
    )

    summary = owner.run_survival_primary(
        **_runtime_kwargs(
            requirement,
            input_path=input_path,
            input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        )
    )

    assert list(captured["analysis"].columns) == [
        "follow_up_days",
        "death",
        "treatment__is_1",
        "age",
        "sex__is_male",
    ]
    assert "sex" not in captured["analysis"]
    assert summary["design_columns"][-1] == "sex__is_male"


def test_primary_survival_plan_cannot_fall_back_to_coder_without_bound_input() -> None:
    requirement = _requirement().model_copy(update={"input_product": "table:other"})
    step = _step().model_copy(update={"family_primary_result_requirement": requirement})
    plan = AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="survival",
        steps=[step],
    )

    with pytest.raises(ValueError, match="host-owned survival runner"):
        validate_required_primary_result(plan=plan, context=_context())


def test_penalized_cox_token_cannot_dispatch_to_plain_cox() -> None:
    payload = _requirement().model_dump(mode="json")
    payload["estimator"] = "penalized_cox"
    requirement = FamilyPrimaryResultRequirement.model_validate(payload)
    step = _step().model_copy(update={"family_primary_result_requirement": requirement})
    plan = AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="survival",
        steps=[step],
    )

    with pytest.raises(ValueError, match="host-owned survival runner"):
        validate_required_primary_result(plan=plan, context=_context())


def test_time_unit_must_come_from_the_exact_context_descriptor() -> None:
    context = _context()
    variables = [
        item.model_copy(update={"unit": "hours"})
        if item.name == "follow_up_days"
        else item
        for item in context.variables
    ]
    context = context.model_copy(update={"variables": variables})
    plan = AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="survival",
        steps=[_step()],
    )

    with pytest.raises(ValueError, match="ConceptDescriptor.unit"):
        validate_required_primary_result(plan=plan, context=context)


def test_declared_ph_policy_blocks_paper_when_threshold_is_crossed(
    tmp_path, monkeypatch
) -> None:
    frame = pd.DataFrame(
        {
            "follow_up_days": [5.0, 7.0, 10.0, 14.0, 21.0, 35.0, 40.0, 12.0],
            "death": [1, 0, 1, 0, 1, 1, 0, 0],
            "treatment": [0, 1, 1, 0, 1, 0, 1, 0],
            "age": [50, 55, 60, 65, 70, 75, 80, 58],
        }
    )
    input_path = tmp_path / "analysis_cohort.csv"
    frame.to_csv(input_path, index=False)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    out_dir = tmp_path / "out"
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setattr(
        owner,
        "_fit_declared_cox",
        lambda *_args, **_kwargs: owner._CoxEstimate(
            hazard_ratio=0.81,
            ci_low=0.66,
            ci_high=0.98,
            standard_error=0.1,
            coefficient=-0.210721,
            p_value=0.03,
        ),
    )
    monkeypatch.setattr(
        owner,
        "ph_test",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "covariate": "treatment__is_1",
                    "test_statistic": 4.0,
                    "p_value": 0.04,
                },
                {"covariate": "age", "test_statistic": 3.0, "p_value": 0.08},
                {"covariate": "global", "test_statistic": 7.0, "p_value": 0.01},
            ]
        ),
    )
    monkeypatch.setattr(
        owner,
        "_package_versions",
        lambda: {"easyicu": "1.0.0", "lifelines": "0.30.3", "pandas": "2.0.0"},
    )
    requirement = _requirement()

    summary = owner.run_survival_primary(
        **_runtime_kwargs(
            requirement,
            input_path=input_path,
            input_sha256=input_sha256,
        )
    )

    assert summary["proportional_hazards_status"] == (
        "violation_block_paper_authorization"
    )
    assert summary["paper_authorization_allowed"] is False
    findings = family_primary_result_reconciliation_findings(
        step=_step(),
        plan=AnalysisPlan(
            research_question="Does treatment change 28-day mortality?",
            analysis_type="survival",
            steps=[_step()],
        ),
        context=_context(),
        step_summary=summary,
        out_dir=out_dir,
    )
    assert [finding.detail["issue"] for finding in findings] == [
        "survival_ph_policy_blocks_paper_authorization"
    ]


def test_primary_survival_rejects_lossy_numeric_conversion(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "follow_up_days": [5.0, 7.0, 10.0, 14.0],
            "death": [1, 0, 1, 0],
            "treatment": [0, 1, 1, 0],
            "age": [50, "not-a-number", 60, 65],
        }
    )
    input_path = tmp_path / "analysis_cohort.csv"
    frame.to_csv(input_path, index=False)
    requirement = _requirement()

    with pytest.raises(owner.SurvivalPrimaryExecutionError) as exc_info:
        owner.run_survival_primary(
            input_path=input_path,
            input_product=str(requirement.input_product),
            input_evidence_id="evidence-analysis-cohort",
            input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
            result_product=requirement.expected_result_product,
            exposure=requirement.exposure_source,
            outcome=requirement.outcome,
            effect_scale=requirement.effect_scale,
            population=requirement.population,
            time_origin=str(requirement.time_origin),
            time_column=str(requirement.time_column),
            time_unit=str(requirement.time_unit),
            event_column=str(requirement.event_column),
            event_value=int(requirement.event_value),
            event_definition=str(requirement.event_definition),
            censoring_strategy=str(requirement.censoring_strategy),
            competing_risk_strategy=str(requirement.competing_risk_strategy),
            time_horizon=str(requirement.time_horizon),
            time_horizon_value=float(requirement.time_horizon_value),
            estimator=requirement.estimator,
            effect_measure=str(requirement.effect_measure),
            covariates=list(requirement.covariates or ()),
            model_terms=list(requirement.model_terms or ()),
            ph_diagnostic=str(requirement.proportional_hazards_diagnostic),
            ph_alpha=float(requirement.proportional_hazards_alpha),
            ph_policy=str(requirement.proportional_hazards_policy),
        )

    assert exc_info.value.reason_code == "model_term_numeric_conversion_loss"


def test_primary_survival_requires_the_host_censor_code() -> None:
    requirement = _requirement().model_copy(update={"event_value": 2})
    step = _step().model_copy(update={"family_primary_result_requirement": requirement})
    plan = AnalysisPlan(
        research_question="Does treatment change 28-day mortality?",
        analysis_type="survival",
        steps=[step],
    )
    context = _context().model_copy(
        update={"endpoint": _context().endpoint.model_copy(update={"levels": [1, 2]})}
    )

    with pytest.raises(ValueError, match="censor code 0"):
        validate_required_primary_result(plan=plan, context=context)
