from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import PrimaryModelContractValidator
from easyicu.research_agent.contracts.runtime import ValidationFinding
from easyicu.research_agent.execution.phase import _contract_repair_log
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    PlannedModelRequirement,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Is the laboratory signal associated with death?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_patients=6,
            n_stays=6,
        ),
        variables=[
            ConceptDescriptor(name="lab_max", role="lab", dtype="float64"),
            ConceptDescriptor(
                name="organ_score",
                role="ordinal_score",
                dtype="float64",
                is_ordinal=True,
            ),
            ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        ],
        primary_exposure="lab",
        target_outcome="death",
    )


def _step(
    *,
    complex_contract: bool = True,
    model_requirements: list[PlannedModelRequirement] | None = None,
) -> AnalysisStep:
    intent = "Fit one adjusted association model."
    method = "logistic_regression"
    expected_outputs = ["table:primary_adjusted_association"]
    if complex_contract:
        intent = (
            "Estimate separate source-aware and complete-case regression models "
            "for the primary and corroborative representations."
        )
        method = "adjusted_association_models"
        expected_outputs = ["table:adjusted_association_estimates"]
    return AnalysisStep(
        step_id="05_primary_association",
        intent=intent,
        method=method,
        expected_outputs=expected_outputs,
        model_requirements=model_requirements or [],
    )


def _prior_records() -> list[dict]:
    return [
        {
            "step_id": "02_missingness",
            "status": "ok",
            "step_summary": {
                "planned_adjustment_context": {
                    "candidate_covariates": ["age", "sex", "adm"],
                    "not_adjusted_for": [
                        "lab_n",
                        "lab_measured",
                        "organ_score_n",
                    ],
                }
            },
        },
        {
            "step_id": "04_reconciliation",
            "status": "ok",
            "step_summary": {
                "laboratory": {
                    "representation_locked": (
                        "continuous log1p(lab_max) among valid observed values"
                    )
                }
            },
        },
    ]


def _contracts() -> list[dict]:
    shared = {
        "baseline_missing_policy": "drop_missing_baseline",
        "fit_status": "fitted",
        "converged": True,
        # The deliberately tiny fixture has zero-event categorical cells, so a
        # truthful accepted contract must disclose separation and a penalized
        # fit. Larger real cohorts without zero cells may set both booleans
        # false and use maximum likelihood.
        "separation_detected": True,
        "penalized": True,
        "fit_method": "sklearn_ridge_logistic_regression(C=1)",
        "interval_method": "bootstrap",
        "convergence_method": "optimizer_success",
        "optimizer_success": True,
    }
    return [
        {
            **shared,
            "model_id": "lab_source_aware",
            "exposure_source": "lab_max",
            "exposure_expression": "log1p(lab_max)",
            "exposure_role": "primary",
            "analysis_role": "primary",
            "analysis_set": "source_aware",
            "n": 5,
            "event_n": 2,
        },
        {
            **shared,
            "model_id": "lab_complete_case",
            "exposure_source": "lab_max",
            "exposure_expression": "log1p(lab_max)",
            "exposure_role": "primary",
            "analysis_role": "sensitivity",
            "analysis_set": "complete_case",
            "n": 3,
            "event_n": 1,
        },
        {
            **shared,
            "model_id": "organ_source_aware",
            "exposure_source": "organ_score",
            "exposure_expression": "C(organ_score)",
            "exposure_role": "secondary",
            "analysis_role": "secondary",
            "analysis_set": "source_aware",
            "n": 5,
            "event_n": 2,
        },
        {
            **shared,
            "model_id": "organ_complete_case",
            "exposure_source": "organ_score",
            "exposure_expression": "C(organ_score)",
            "exposure_role": "secondary",
            "analysis_role": "sensitivity",
            "analysis_set": "complete_case",
            "n": 5,
            "event_n": 2,
        },
    ]


def _contracts_with_requirements() -> tuple[list[dict], list[PlannedModelRequirement]]:
    contracts = copy.deepcopy(_contracts())
    requirements: list[PlannedModelRequirement] = []
    for contract in contracts:
        requirement_id = f"planned_{contract['model_id']}"
        required = contract["analysis_role"] in {"primary", "secondary"}
        contract.update(
            {
                "requirement_id": requirement_id,
                "outcome": "death",
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "model_family": "logistic_regression",
            }
        )
        requirements.append(
            PlannedModelRequirement(
                requirement_id=requirement_id,
                outcome="death",
                outcome_type="binary",
                method_family="logistic_regression",
                exposure_source=contract["exposure_source"],
                analysis_role=contract["analysis_role"],
                analysis_set=contract["analysis_set"],
                required_for_step_success=required,
            )
        )
    return contracts, requirements


def _bind_standard_contracts_to_requirements(
    contracts: list[dict],
) -> list[PlannedModelRequirement]:
    """Add the planner-owned ids/fields used by the standard four-model fixture."""

    canonical_contracts, requirements = _contracts_with_requirements()
    canonical_by_id = {
        contract["model_id"]: contract for contract in canonical_contracts
    }
    for contract in contracts:
        canonical = canonical_by_id.get(contract.get("model_id"))
        if canonical is None:
            continue
        for field in (
            "requirement_id",
            "outcome",
            "outcome_type",
            "method_family",
            "model_family",
        ):
            contract.setdefault(field, canonical[field])
    return requirements


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cohort = pd.DataFrame(
        {
            "age": [60, 61, 62, 63, 64, 65],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "adm": ["A", "B", None, "A", "B", "A"],
            "death": [1, 0, 1, 0, 1, 0],
            "lab_max": [1.0, None, 2.0, 3.0, None, 4.0],
            "organ_score": [0.0, 1.0, None, 2.0, 3.0, 4.0],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    rows: list[dict] = []
    for contract in _contracts():
        model_id = contract["model_id"]
        source = contract["exposure_source"]
        rows.extend(
            [
                {
                    "model_id": model_id,
                    "term": "intercept",
                    "term_role": "intercept",
                    "source_variable": "",
                    "odds_ratio": 1.0,
                    "ci_low": 0.8,
                    "ci_high": 1.2,
                    "standard_error": 0.1,
                    "effect_scale": "odds_ratio",
                },
                {
                    "model_id": model_id,
                    "term": contract["exposure_expression"],
                    "term_role": "exposure",
                    "source_variable": source,
                    "odds_ratio": 1.2,
                    "ci_low": 1.0,
                    "ci_high": 1.4,
                    "standard_error": 0.1,
                    "effect_scale": "odds_ratio",
                },
            ]
        )
        for covariate in ("age", "sex", "adm"):
            rows.append(
                {
                    "model_id": model_id,
                    "term": covariate,
                    "term_role": "adjustment",
                    "source_variable": covariate,
                    "odds_ratio": 1.0,
                    "ci_low": 0.9,
                    "ci_high": 1.1,
                    "standard_error": 0.1,
                    "effect_scale": "odds_ratio",
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "model_coefficients.csv", index=False)
    return cohort_path, out_dir


def _audit(
    tmp_path: Path,
    *,
    contracts: list[dict] | None,
    step: AnalysisStep | None = None,
) -> list:
    cohort_path, out_dir = _write_inputs(tmp_path)
    if step is None and contracts is not None:
        contracts = copy.deepcopy(contracts)
        requirements = _bind_standard_contracts_to_requirements(contracts)
        step = _step(model_requirements=requirements)
    summary = {} if contracts is None else {"model_contracts": contracts}
    return PrimaryModelContractValidator().audit(
        step=step or _step(),
        step_summary=summary,
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )


def _issue_types(findings: list) -> set[str]:
    return {
        issue["issue"]
        for finding in findings
        for issue in (finding.detail or {}).get("issues", [])
    }


def test_primary_model_contract_accepts_separate_verified_models(tmp_path: Path):
    assert _audit(tmp_path, contracts=_contracts()) == []


def test_primary_model_contract_accepts_planner_authorized_secondary_only_step(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts, requirements = _contracts_with_requirements()
    contract = copy.deepcopy(contracts[2])
    requirement = requirements[2]
    coefficients = pd.read_csv(out_dir / "model_coefficients.csv")
    coefficients.loc[coefficients["model_id"].eq(contract["model_id"])].to_csv(
        out_dir / "model_coefficients.csv", index=False
    )

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=[requirement]),
        step_summary={"model_contracts": [contract]},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert findings == []


def test_primary_step_rejects_secondary_only_model_roster_before_execution() -> None:
    _contracts_unused, requirements = _contracts_with_requirements()

    with pytest.raises(
        ValueError,
        match="primary adjusted-association step.*primary model requirement",
    ):
        AnalysisStep(
            step_id="05_primary_association",
            planned_analysis_role="primary",
            intent="Estimate the primary adjusted association.",
            method="adjusted_association_models",
            expected_outputs=["table:adjusted_association_estimates"],
            model_requirements=[requirements[2]],
        )


def test_primary_model_contract_rejects_primary_in_secondary_only_roster(
    tmp_path: Path,
):
    contracts, requirements = _contracts_with_requirements()
    secondary_contract = copy.deepcopy(contracts[2])
    secondary_contract["analysis_role"] = "primary"

    issues = _issue_types(
        _audit(
            tmp_path,
            contracts=[secondary_contract],
            step=_step(model_requirements=[requirements[2]]),
        )
    )

    assert "model_requirement_field_mismatch" in issues
    assert "unplanned_primary_model" in issues


def test_secondary_only_contract_error_does_not_request_a_fake_primary(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts, requirements = _contracts_with_requirements()
    contract = copy.deepcopy(contracts[2])
    contract.pop("convergence_method")
    contract.pop("optimizer_success")
    coefficients = pd.read_csv(out_dir / "model_coefficients.csv")
    coefficients.loc[coefficients["model_id"].eq(contract["model_id"])].to_csv(
        out_dir / "model_coefficients.csv", index=False
    )

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=[requirements[2]]),
        step_summary={"model_contracts": [contract]},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert len(findings) == 1
    assert "secondary-only roster" in findings[0].message
    assert (
        "keep exactly one context-declared primary exposure" not in findings[0].message
    )


@pytest.mark.parametrize("fit_status", ["not_fitted", "separation_no_estimate"])
def test_primary_model_contract_legacy_required_model_must_be_fitted(
    tmp_path: Path,
    fit_status: str,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts = copy.deepcopy(_contracts())
    requirements = _bind_standard_contracts_to_requirements(contracts)
    contracts[2].update(
        {
            "fit_status": fit_status,
            "converged": False,
            "separation_detected": True,
            "penalized": False,
            "fit_method": "attempted_logistic_regression",
            "interval_method": "unavailable",
            "fit_failure_reason": "Design matrix was rank deficient.",
        }
    )
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    target = table["model_id"].eq("organ_source_aware")
    table.loc[target, ["odds_ratio", "ci_low", "ci_high", "standard_error"]] = None
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "required_model_not_fitted" in issues
    assert "inconsistent_not_fitted_estimate" not in issues


@pytest.mark.parametrize("fit_status", ["not_fitted", "separation_no_estimate"])
def test_primary_model_contract_allows_optional_sensitivity_without_fake_rows(
    tmp_path: Path,
    fit_status: str,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts = copy.deepcopy(_contracts())
    requirements = _bind_standard_contracts_to_requirements(contracts)
    contracts[1].update(
        {
            "fit_status": fit_status,
            "converged": False,
            "separation_detected": True,
            "penalized": False,
            "fit_method": "attempted_logistic_regression",
            "interval_method": "unavailable",
            "fit_failure_reason": "Optional sensitivity design was singular.",
        }
    )
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    table = table.loc[~table["model_id"].eq("lab_complete_case")]
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert findings == []


def test_primary_model_contract_rejects_finite_result_for_not_fitted_model(
    tmp_path: Path,
):
    contracts = copy.deepcopy(_contracts())
    contracts[1].update(
        {
            "fit_status": "not_fitted",
            "converged": False,
            "separation_detected": True,
            "penalized": False,
            "fit_method": "attempted_logistic_regression",
            "interval_method": "unavailable",
            "fit_failure_reason": "Optional sensitivity design was singular.",
        }
    )

    issues = _issue_types(_audit(tmp_path, contracts=contracts))

    assert "inconsistent_not_fitted_estimate" in issues


@pytest.mark.parametrize("result_location", ["contract", "nested_model_summary"])
def test_primary_model_contract_rejects_nonfitted_summary_result(
    tmp_path: Path,
    result_location: str,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts = copy.deepcopy(_contracts())
    contracts[1].update(
        {
            "fit_status": "not_fitted",
            "converged": False,
            "separation_detected": True,
            "penalized": False,
            "fit_method": "attempted_logistic_regression",
            "interval_method": "unavailable",
            "fit_failure_reason": "Optional sensitivity design was singular.",
        }
    )
    step_summary = {"model_contracts": contracts}
    finite_result = {"odds_ratio": 9.9, "ci_low": 4.0, "ci_high": 12.0}
    if result_location == "contract":
        contracts[1].update(finite_result)
    else:
        step_summary["models"] = [
            {
                "model_id": "lab_complete_case",
                "exposure_terms": [finite_result],
            }
        ]
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    table = table.loc[~table["model_id"].eq("lab_complete_case")]
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary=step_summary,
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "inconsistent_not_fitted_estimate" in _issue_types(findings)


def test_primary_model_contract_blocks_missing_planner_required_model(
    tmp_path: Path,
):
    contracts, requirements = _contracts_with_requirements()
    contracts = [
        contract
        for contract in contracts
        if contract["model_id"] != "organ_source_aware"
    ]

    findings = _audit(
        tmp_path,
        contracts=contracts,
        step=_step(model_requirements=requirements),
    )

    assert "required_model_missing" in _issue_types(findings)


def test_primary_model_contract_blocks_not_fitted_planner_required_model(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts, requirements = _contracts_with_requirements()
    contracts[2].update(
        {
            "fit_status": "not_fitted",
            "converged": False,
            "separation_detected": True,
            "penalized": False,
            "fit_method": "attempted_logistic_regression",
            "interval_method": "unavailable",
            "fit_failure_reason": "Required model design was singular.",
        }
    )
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    table = table.loc[~table["model_id"].eq("organ_source_aware")]
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "required_model_not_fitted" in issues
    assert "missing_coefficient_rows" not in issues


def test_primary_model_contract_defensively_blocks_optional_primary_nonfit(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts, requirements = _contracts_with_requirements()
    requirements[0] = requirements[0].model_copy(
        update={"required_for_step_success": False}
    )
    contracts[0].update(
        {
            "fit_status": "not_fitted",
            "converged": False,
            "separation_detected": True,
            "penalized": False,
            "fit_method": "attempted_logistic_regression",
            "interval_method": "unavailable",
            "fit_failure_reason": "Primary model design was singular.",
        }
    )
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    table = table.loc[~table["model_id"].eq("lab_source_aware")]
    table.to_csv(table_path, index=False)
    step = _step().model_copy(update={"model_requirements": requirements})

    findings = PrimaryModelContractValidator().audit(
        step=step,
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "required_model_not_fitted" in _issue_types(findings)


def test_primary_model_contract_allows_planner_optional_sensitivity_nonfit(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts, requirements = _contracts_with_requirements()
    contracts[1].update(
        {
            "fit_status": "not_fitted",
            "converged": False,
            "separation_detected": True,
            "penalized": False,
            "fit_method": "attempted_logistic_regression",
            "interval_method": "unavailable",
            "fit_failure_reason": "Optional sensitivity design was singular.",
        }
    )
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    table = table.loc[~table["model_id"].eq("lab_complete_case")]
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert findings == []


@pytest.mark.parametrize(
    ("field", "reported"),
    [
        ("outcome", "different_outcome"),
        ("outcome_type", "continuous"),
        ("method_family", "linear_regression"),
        ("exposure_source", "different_exposure"),
        ("analysis_role", "secondary"),
        ("analysis_set", "complete_case"),
    ],
)
def test_primary_model_contract_blocks_planner_requirement_field_drift(
    tmp_path: Path,
    field: str,
    reported: str,
):
    contracts, requirements = _contracts_with_requirements()
    contracts[0][field] = reported

    findings = _audit(
        tmp_path,
        contracts=contracts,
        step=_step(model_requirements=requirements),
    )

    assert "model_requirement_field_mismatch" in _issue_types(findings)


def test_plan_normalizer_preserves_typed_model_requirements() -> None:
    from easyicu.research_agent.agents.core import _normalise_plan_payload

    payload, dropped = _normalise_plan_payload(
        {
            "research_question": "Is an exposure associated with an outcome?",
            "steps": [
                {
                    "step_id": "01_model",
                    "planned_analysis_role": "primary",
                    "intent": "Fit the planner-selected model.",
                    "method": "adjusted_association_models",
                    "expected_outputs": ["table:adjusted_association_estimates"],
                    "model_requirements": [
                        {
                            "requirement_id": "primary_model",
                            "outcome": "outcome",
                            "outcome_type": "binary",
                            "method_family": "logistic_regression",
                            "exposure_source": "exposure",
                            "analysis_role": "primary",
                            "analysis_set": "complete_case",
                            "required_for_step_success": True,
                            "case_hint": "must be discarded",
                        }
                    ],
                }
            ],
        }
    )

    plan = AnalysisPlan.model_validate(payload)
    requirement = plan.steps[0].model_requirements[0]
    assert requirement.requirement_id == "primary_model"
    assert requirement.required_for_step_success is True
    assert dropped["model_requirements"] == ["primary_model:case_hint"]


@pytest.mark.parametrize(
    ("method", "expected_output", "outcome_type", "method_family"),
    [
        (
            "survival",
            "table:survival_effect",
            "binary",
            "cox_proportional_hazards",
        ),
        (
            "prediction_model",
            "table:model_performance",
            "binary",
            "logistic_regression",
        ),
        (
            "trajectory_clustering",
            "table:phenotype_assignments",
            "continuous",
            "kmeans",
        ),
        (
            "adjusted_association_models",
            "table:adjusted_association_estimates",
            "continuous",
            "mixed_effects_regression",
        ),
    ],
)
def test_model_requirements_reject_unsupported_analysis_family_contracts(
    method: str,
    expected_output: str,
    outcome_type: str,
    method_family: str,
) -> None:
    with pytest.raises(ValueError, match="model_requirements"):
        AnalysisStep(
            step_id="unsupported_model_contract",
            intent="Run a family-specific scientific analysis.",
            method=method,
            expected_outputs=[expected_output],
            model_requirements=[
                {
                    "requirement_id": "unsupported_primary",
                    "outcome": "outcome",
                    "outcome_type": outcome_type,
                    "method_family": method_family,
                    "exposure_source": "exposure",
                    "analysis_role": "primary",
                    "analysis_set": "complete_case",
                    "required_for_step_success": True,
                }
            ],
        )


@pytest.mark.parametrize(
    "expected_output",
    [
        "figure:adjusted_association_estimates",
        "statistic:adjusted_association_estimates",
        "adjusted_association_estimates",
    ],
)
def test_model_requirements_require_exact_table_product(
    expected_output: str,
) -> None:
    _contracts_payload, requirements = _contracts_with_requirements()

    with pytest.raises(ValueError, match="table:adjusted_association_estimates"):
        AnalysisStep(
            step_id="unsupported_model_product",
            intent="Run the supported models but declare the wrong product kind.",
            method="adjusted_association_models",
            expected_outputs=[expected_output],
            model_requirements=requirements,
        )


@pytest.mark.parametrize("analysis_role", ["primary", "secondary"])
def test_primary_and_secondary_model_requirements_cannot_be_optional(
    analysis_role: str,
) -> None:
    with pytest.raises(ValueError, match="must be required for step success"):
        PlannedModelRequirement(
            requirement_id=f"optional_{analysis_role}",
            outcome="death",
            outcome_type="binary",
            method_family="logistic_regression",
            exposure_source="lab_max",
            analysis_role=analysis_role,
            analysis_set="source_aware",
            required_for_step_success=False,
        )


def test_plan_signature_treats_model_requirement_change_as_substantive() -> None:
    from easyicu.research_agent.execution.phase import _plan_signature

    _contracts_payload, requirements = _contracts_with_requirements()
    base = AnalysisPlan(
        research_question="Is the exposure associated with the outcome?",
        steps=[_step(model_requirements=requirements)],
    )
    changed_requirements = copy.deepcopy(requirements)
    changed_requirements[2] = changed_requirements[2].model_copy(
        update={"analysis_set": "complete_case"}
    )
    changed = base.model_copy(
        update={"steps": [_step(model_requirements=changed_requirements)]}
    )

    assert _plan_signature(base) != _plan_signature(changed)


def test_primary_model_contract_requires_fixed_schema(tmp_path: Path):
    findings = _audit(tmp_path, contracts=None)

    assert len(findings) == 1
    assert "missing_model_contracts" in _issue_types(findings)


@pytest.mark.parametrize("raw_contracts", [[], {}, "not-a-contract-list"])
def test_primary_model_contract_activates_when_contract_key_is_present(
    tmp_path: Path, raw_contracts
):
    cohort_path, out_dir = _write_inputs(tmp_path)

    findings = PrimaryModelContractValidator().audit(
        step=_step(complex_contract=False),
        step_summary={"model_contracts": raw_contracts},
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "missing_model_contracts" in _issue_types(findings)


def test_primary_model_contract_configure_step_is_not_misclassified_as_figure():
    configure = AnalysisStep(
        step_id="06_configure_adjusted_models",
        intent="Configure and fit the adjusted models.",
        method="adjusted_association_models",
        expected_outputs=["table:adjusted_association_estimates"],
    )
    figure_only = AnalysisStep(
        step_id="06_adjusted_models_render",
        intent="Render the fitted estimates.",
        method="publication_figure_generation",
        expected_outputs=["figure:adjusted_association_plot"],
    )

    assert PrimaryModelContractValidator._activates(configure, _context(), {}) is True
    assert (
        PrimaryModelContractValidator._activates(figure_only, _context(), {}) is False
    )


def test_primary_model_contract_requires_planner_owned_model_roster(
    tmp_path: Path,
) -> None:
    findings = _audit(
        tmp_path,
        contracts=_contracts(),
        step=_step(model_requirements=[]),
    )

    assert "planned_model_requirements_required" in _issue_types(findings)


def test_current_step_summary_cannot_self_authorize_primary_alias() -> None:
    aliases = PrimaryModelContractValidator._operational_primary_sources(
        declared_primary="primary_signal",
        completed_step_records=[],
        step_summary={
            "primary_exposure": "primary_signal",
            "operational_column": "different_signal",
        },
    )

    assert aliases == []


def test_failed_latest_checkpoint_revokes_old_primary_alias() -> None:
    aliases = PrimaryModelContractValidator._operational_primary_sources(
        declared_primary="primary_signal",
        completed_step_records=[
            {
                "step_id": "01_mapping",
                "status": "ok",
                "step_summary": {
                    "primary_exposure": "primary_signal",
                    "operational_column": "mapped_signal",
                },
            },
            {
                "step_id": "01_mapping",
                "status": "contract_failed",
                "step_summary": {},
            },
        ],
        step_summary={},
    )

    assert aliases == []


def test_planner_model_requirements_activate_without_context_exposure():
    _contracts_payload, requirements = _contracts_with_requirements()
    context = _context().model_copy(update={"primary_exposure": None})

    assert (
        PrimaryModelContractValidator._activates(
            _step(model_requirements=requirements),
            context,
            {},
        )
        is True
    )


def test_planner_primary_source_is_authoritative_operational_alias(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts = copy.deepcopy(_contracts())
    requirements = _bind_standard_contracts_to_requirements(contracts)
    context = _context().model_copy(
        update={"primary_exposure": "clinical_laboratory_signal"}
    )

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=context,
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "primary_exposure_mismatch" not in issues
    assert "alternate_exposure_cannot_be_primary" not in issues


def test_planner_model_requirements_do_not_activate_for_wrong_product_kind():
    _contracts_payload, requirements = _contracts_with_requirements()
    step = _step(model_requirements=requirements).model_copy(
        update={"expected_outputs": ["figure:adjusted_association_estimates"]}
    )

    assert PrimaryModelContractValidator._activates(step, _context(), {}) is False


@pytest.mark.parametrize(
    "method",
    [
        "cox_proportional_hazards",
        "prediction_model",
        "trajectory_clustering",
        "mixed_effects_regression",
    ],
)
def test_primary_model_contract_does_not_claim_other_method_families(
    method: str,
) -> None:
    step = AnalysisStep(
        step_id="family_specific_model",
        intent="Run the requested family-specific model.",
        method=method,
        expected_outputs=["table:family_specific_results"],
    )

    assert (
        PrimaryModelContractValidator._activates(
            step,
            _context(),
            {"model_contracts": [{"model_id": "family_specific"}]},
        )
        is False
    )


def test_primary_model_contract_name_matching_rejects_nearby_concepts():
    assert PrimaryModelContractValidator._names_match("lab", "lab_max") is True
    assert PrimaryModelContractValidator._names_match("sofa2", "sofa3") is False
    assert PrimaryModelContractValidator._names_match("organ", "organ_score") is False


def test_primary_model_contract_blocks_wrong_or_duplicate_primary(tmp_path: Path):
    contracts = _contracts()
    contracts[0]["analysis_role"] = "secondary"
    contracts[2]["analysis_role"] = "primary"
    contracts[2]["exposure_role"] = "primary"

    findings = _audit(tmp_path, contracts=contracts)

    issues = _issue_types(findings)
    assert "primary_exposure_mismatch" in issues
    assert "alternate_exposure_cannot_be_primary" in issues


def test_primary_model_contract_enforces_locked_transform(tmp_path: Path):
    contracts = _contracts()
    contracts[0]["exposure_expression"] = "lab_max"

    findings = _audit(tmp_path, contracts=contracts)

    assert "locked_primary_expression_mismatch" in _issue_types(findings)


def test_primary_model_contract_blocks_mutual_and_forbidden_adjustment(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    extra = pd.DataFrame(
        [
            {
                "model_id": "lab_source_aware",
                "term": "organ_score",
                "term_role": "adjustment",
                "source_variable": "organ_score",
                "odds_ratio": 1.1,
                "ci_low": 0.9,
                "ci_high": 1.3,
            },
            {
                "model_id": "lab_source_aware",
                "term": "lab_n",
                "term_role": "adjustment",
                "source_variable": "lab_n",
                "odds_ratio": 1.0,
                "ci_low": 0.9,
                "ci_high": 1.1,
            },
        ]
    )
    pd.concat([table, extra], ignore_index=True).to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": _contracts()},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "mutual_exposure_adjustment" in issues
    assert "forbidden_adjustment_source" in issues
    assert "adjustment_outside_planned_allowlist" in issues


def test_primary_model_contract_checks_denominators_and_fit_diagnostics(
    tmp_path: Path,
):
    contracts = copy.deepcopy(_contracts())
    contracts[0]["n"] = 6
    contracts[0]["event_n"] = 3
    contracts[0]["converged"] = False
    contracts[0]["separation_detected"] = True
    contracts[0]["penalized"] = False

    findings = _audit(tmp_path, contracts=contracts)

    issues = _issue_types(findings)
    assert "model_denominator_or_event_mismatch" in issues
    assert "fitted_model_must_converge" in issues
    assert "separation_requires_penalized_fit_or_no_estimate" in issues


def test_primary_model_contract_rejects_encoded_term_as_raw_source(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    cohort = pd.read_parquet(cohort_path)
    cohort["group"] = ["A", "B", "A", "B", "A", "B"]
    cohort.to_parquet(cohort_path, index=False)

    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    target = table["model_id"].eq("lab_source_aware") & table["term"].eq("age")
    table.loc[target, "term"] = "group_B"
    table.loc[target, "source_variable"] = "group_B"
    table.to_csv(table_path, index=False)

    prior_records = copy.deepcopy(_prior_records())
    prior_records[0]["step_summary"]["planned_adjustment_context"][
        "candidate_covariates"
    ].append("group")
    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": _contracts()},
        context=_context(),
        completed_step_records=prior_records,
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = [
        issue
        for finding in findings
        for issue in (finding.detail or {}).get("issues", [])
    ]
    lineage_issue = next(
        issue
        for issue in issues
        if issue.get("issue") == "coefficient_source_variable_unresolvable"
        and issue.get("model_id") == "lab_source_aware"
    )
    assert lineage_issue["term"] == "group_B"
    assert lineage_issue["reported_source_variable"] == "group_B"
    assert lineage_issue["missing_raw_source_variables"] == ["group_B"]
    assert lineage_issue["reason"] == (
        "source_variable_missing_from_authoritative_cohort"
    )


def test_primary_model_contract_accepts_encoded_term_with_raw_source(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    cohort = pd.read_parquet(cohort_path)
    cohort["group"] = ["A", "B", "A", "B", "A", "B"]
    cohort.to_parquet(cohort_path, index=False)

    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    target = table["model_id"].eq("lab_source_aware") & table["term"].eq("age")
    table.loc[target, "term"] = "group_B"
    table.loc[target, "source_variable"] = "group"
    table.to_csv(table_path, index=False)

    prior_records = copy.deepcopy(_prior_records())
    prior_records[0]["step_summary"]["planned_adjustment_context"][
        "candidate_covariates"
    ].append("group")
    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": _contracts()},
        context=_context(),
        completed_step_records=prior_records,
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "coefficient_source_variable_unresolvable" not in issues
    assert "denominator_contract_unresolvable" not in issues


def test_primary_model_contract_replays_derived_source_without_range_exclusion(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    contracts = copy.deepcopy(_contracts())
    target = next(
        item for item in contracts if item["model_id"] == "organ_complete_case"
    )
    target.update(
        {
            "exposure_source": "organ_score_quartiles",
            "exposure_expression": "qcut(organ_score, q=4)",
            "n": 5,
            "event_n": 2,
        }
    )
    context = _context()
    organ = next(item for item in context.variables if item.name == "organ_score")
    organ.valid_range = (0.0, 3.0)

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=context,
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    target_issues = [
        issue
        for finding in findings
        for issue in (finding.detail or {}).get("issues", [])
        if issue.get("model_id") == "organ_complete_case"
    ]
    assert "exposure_terms_do_not_match_model_source" not in {
        issue.get("issue") for issue in target_issues
    }
    assert "denominator_contract_unresolvable" not in {
        issue.get("issue") for issue in target_issues
    }
    assert "model_denominator_or_event_mismatch" not in {
        issue.get("issue") for issue in target_issues
    }


def test_primary_model_contract_does_not_expand_legacy_simple_steps(
    tmp_path: Path,
):
    findings = _audit(
        tmp_path,
        contracts=None,
        step=_step(complex_contract=False),
    )

    assert findings == []


def test_primary_model_contract_is_wired_into_all_contract_passes() -> None:
    import ast
    import inspect

    from easyicu.research_agent.execution import phase as pipeline_execute

    def _audit_calls(function, validator_name: str) -> list[ast.Call]:
        tree = ast.parse(inspect.getsource(function))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "audit"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == validator_name
        ]

    def _shared_gate_calls(function) -> list[ast.Call]:
        tree = ast.parse(inspect.getsource(function))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_step_deterministic_contract_findings"
        ]

    # The early repair gate and the final authority gate now evaluate ONE shared
    # deterministic contract sequence (dedup), so the primary-model contract
    # validator is audited exactly once — inside that shared sequence — and it
    # carries the cohort-integrity authority path there. For ordinary analysis
    # steps this is the execution cohort; a cohort-producing step deliberately
    # retains the full universe even when later development execution is sampled.
    shared_audits = _audit_calls(
        pipeline_execute._step_deterministic_contract_findings,
        "primary_model_contract_validator",
    )
    assert len(shared_audits) == 1
    shared_keywords = {kw.arg: kw.value for kw in shared_audits[0].keywords}
    assert isinstance(shared_keywords.get("cohort_path"), ast.Name)
    assert shared_keywords["cohort_path"].id == "integrity_universe_path"

    # The mutable execution loop still owns one early repair gate and the
    # extracted read-only evaluator owns the single final authority gate; each
    # wires in the shared sequence and passes its already-resolved cohort path.
    early_calls = _shared_gate_calls(pipeline_execute.run_execute_phase)
    final_calls = _shared_gate_calls(
        pipeline_execute._evaluate_final_deterministic_gates
    )
    assert len(early_calls) == 1
    assert len(final_calls) == 1
    early_keywords = {keyword.arg: keyword.value for keyword in early_calls[0].keywords}
    final_keywords = {keyword.arg: keyword.value for keyword in final_calls[0].keywords}
    assert isinstance(early_keywords.get("execution_cohort_path"), ast.Name)
    assert early_keywords["execution_cohort_path"].id == "step_execution_cohort_path"
    assert isinstance(final_keywords.get("execution_cohort_path"), ast.Name)
    assert final_keywords["execution_cohort_path"].id == "execution_cohort_path"


def test_primary_model_contract_rejects_noncanonical_machine_fields(
    tmp_path: Path,
):
    contracts = copy.deepcopy(_contracts())
    contracts[0].update(
        {
            "exposure_role": "primary exposure",
            "analysis_role": "main model",
            "analysis_set": "locked full cohort",
            "baseline_missing_policy": "drop rows with missing baseline data",
            "fit_status": "fitted_penalized",
        }
    )

    issues = _issue_types(_audit(tmp_path, contracts=contracts))

    assert "noncanonical_exposure_role" in issues
    assert "noncanonical_analysis_role" in issues
    assert "noncanonical_analysis_set" in issues
    assert "noncanonical_baseline_missing_policy" in issues
    assert "noncanonical_fit_status" in issues


def test_primary_model_contract_accepts_canonical_penalized_fit(tmp_path: Path):
    contracts = copy.deepcopy(_contracts())
    contracts[0]["fit_status"] = "fitted"
    contracts[0]["penalized"] = True
    contracts[0]["fit_method"] = "sklearn_ridge_logistic_regression(C=1)"

    assert _audit(tmp_path, contracts=contracts) == []


def test_primary_model_contract_ignores_non_model_rows_in_wide_figure_data(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    pd.DataFrame(
        [
            {
                "model_id": None,
                "term": None,
                "term_role": None,
                "source_variable": None,
                "odds_ratio": None,
                "ci_low": 0.1,
                "ci_high": 0.2,
                "source_table": "missingness_summary",
            }
        ]
    ).to_csv(out_dir / "figure_source_data.csv", index=False)

    contracts = _contracts()
    requirements = _bind_standard_contracts_to_requirements(contracts)
    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert findings == []


def test_primary_model_contract_cross_checks_model_result_coefficients(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    summary = {
        "model_contracts": _contracts(),
        "models": [
            {
                "model_id": "lab_source_aware",
                "exposure_terms": [
                    {
                        "term": "log1p(lab_max)",
                        "odds_ratio": 9.9,
                        "ci_low": 1.0,
                        "ci_high": 1.4,
                    }
                ],
            }
        ],
    }

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary=summary,
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "coefficient_model_result_mismatch" in _issue_types(findings)


def test_primary_model_contract_detects_unreported_zero_event_category(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    cohort = pd.read_parquet(cohort_path)
    cohort.loc[5, "adm"] = "EYE"
    cohort.to_parquet(cohort_path, index=False)
    contracts = copy.deepcopy(_contracts())
    for contract in contracts:
        contract["separation_detected"] = False
        contract["penalized"] = False
        contract["fit_method"] = "maximum_likelihood_logit"

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "zero_cell_separation_not_reported" in issues
    assert "zero_cell_separation_requires_penalized_fit" in issues


def test_primary_model_contract_replays_actual_adjustment_zero_cell_without_prior_plan(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    cohort = pd.read_parquet(cohort_path)
    cohort.loc[5, "adm"] = "sparse_level"
    cohort.to_parquet(cohort_path, index=False)
    contracts = copy.deepcopy(_contracts())
    for contract in contracts:
        contract["separation_detected"] = False
        contract["penalized"] = False
        contract["fit_method"] = "maximum_likelihood_logit"
        contract.pop("interval_method", None)
        contract.pop("convergence_method", None)
        contract.pop("optimizer_success", None)

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "zero_cell_separation_not_reported" in issues
    assert "zero_cell_separation_requires_penalized_fit" in issues


@pytest.mark.parametrize(
    ("ci_low", "ci_high", "standard_error"),
    [
        (None, None, None),
        (1.4, 1.0, 0.1),
        (1.0, 1.4, float("inf")),
    ],
)
def test_primary_model_contract_blocks_invalid_fitted_term_interval(
    tmp_path: Path, ci_low, ci_high, standard_error
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    target = table["model_id"].eq("lab_source_aware") & table["term"].eq("age")
    table.loc[target, ["ci_low", "ci_high", "standard_error"]] = [
        ci_low,
        ci_high,
        standard_error,
    ]
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": _contracts()},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "fitted_term_missing_or_invalid_interval" in _issue_types(findings)


def test_primary_model_contract_allows_explicit_reference_row_without_interval(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    reference = (
        table.loc[
            table["model_id"].eq("lab_source_aware") & table["term_role"].eq("exposure")
        ]
        .iloc[0]
        .copy()
    )
    reference.update(
        {
            "term": "lab_max_reference_level",
            "odds_ratio": 1.0,
            "ci_low": None,
            "ci_high": None,
            "standard_error": None,
            "interval_method": "not_applicable_reference",
        }
    )
    table = pd.concat([table, reference.to_frame().T], ignore_index=True)
    table.to_csv(table_path, index=False)

    contracts = _contracts()
    requirements = _bind_standard_contracts_to_requirements(contracts)
    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert findings == []


def test_primary_model_contract_allows_truthful_penalized_point_only_output(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    target = table["model_id"].eq("lab_source_aware")
    table.loc[target, ["ci_low", "ci_high", "standard_error"]] = None
    table.to_csv(table_path, index=False)
    contracts = copy.deepcopy(_contracts())
    contracts[0]["interval_method"] = "unavailable"

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    issues = _issue_types(findings)
    assert "fitted_term_missing_or_invalid_interval" not in issues
    assert "penalized_intervals_require_controlled_provenance" not in issues


def test_primary_model_contract_point_only_still_requires_finite_estimate(
    tmp_path: Path,
):
    cohort_path, out_dir = _write_inputs(tmp_path)
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    target = table["model_id"].eq("lab_source_aware")
    table.loc[target, ["odds_ratio", "ci_low", "ci_high", "standard_error"]] = None
    table.to_csv(table_path, index=False)
    contracts = copy.deepcopy(_contracts())
    contracts[0]["interval_method"] = "unavailable"

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=_prior_records(),
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "fitted_term_missing_or_invalid_interval" in _issue_types(findings)


def _write_mixed_outcome_inputs(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    cohort_path, out_dir = _write_inputs(tmp_path)
    cohort = pd.read_parquet(cohort_path)
    cohort["continuous_outcome"] = [1.2, 2.0, 3.5, 4.0, 5.1, 6.2]
    cohort.to_parquet(cohort_path, index=False)
    rows = pd.DataFrame(
        [
            {
                "model_id": "binary_primary",
                "term": "log1p(lab_max)",
                "term_role": "exposure",
                "source_variable": "lab_max",
                "estimate": 1.2,
                "ci_low": 1.0,
                "ci_high": 1.4,
                "standard_error": 0.1,
                "effect_scale": "odds_ratio",
            },
            {
                "model_id": "continuous_secondary",
                "term": "log1p(lab_max)",
                "term_role": "exposure",
                "source_variable": "lab_max",
                "estimate": 0.4,
                "ci_low": 0.2,
                "ci_high": 0.6,
                "standard_error": 0.1,
                "effect_scale": "conditional_quantile_difference",
            },
        ]
    )
    rows.to_csv(out_dir / "model_coefficients.csv", index=False)
    shared = {
        "exposure_source": "lab_max",
        "exposure_expression": "log1p(lab_max)",
        "exposure_role": "primary",
        "analysis_set": "complete_case",
        "baseline_missing_policy": "drop_missing_baseline",
        "fit_status": "fitted",
        "converged": True,
        "separation_detected": False,
        "penalized": False,
    }
    contracts = [
        {
            **shared,
            "model_id": "binary_primary",
            "analysis_role": "primary",
            "outcome": "death",
            "outcome_type": "binary",
            "model_family": "binomial_logistic_regression",
            "n": 4,
            "event_n": 2,
            "fit_method": "statsmodels_logit_mle",
        },
        {
            **shared,
            "model_id": "continuous_secondary",
            "analysis_role": "secondary",
            "outcome": "continuous_outcome",
            "outcome_type": "continuous",
            "model_family": "median_quantile_regression",
            "n": 4,
            "event_n": None,
            "fit_method": "statsmodels_quantreg",
        },
    ]
    return cohort_path, out_dir, contracts


def test_primary_model_contract_supports_mixed_binary_and_continuous_outcomes(
    tmp_path: Path,
):
    cohort_path, out_dir, contracts = _write_mixed_outcome_inputs(tmp_path)
    requirements = [
        PlannedModelRequirement(
            requirement_id=f"planned_{contract['model_id']}",
            outcome=contract["outcome"],
            outcome_type=contract["outcome_type"],
            method_family=contract["model_family"],
            exposure_source=contract["exposure_source"],
            analysis_role=contract["analysis_role"],
            analysis_set=contract["analysis_set"],
            required_for_step_success=True,
        )
        for contract in contracts
    ]
    for contract, requirement in zip(contracts, requirements):
        contract["requirement_id"] = requirement.requirement_id
        contract["method_family"] = contract["model_family"]

    findings = PrimaryModelContractValidator().audit(
        step=_step(model_requirements=requirements),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert findings == []


def test_primary_model_contract_rejects_continuous_event_count(tmp_path: Path):
    cohort_path, out_dir, contracts = _write_mixed_outcome_inputs(tmp_path)
    contracts[1]["event_n"] = 2

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "continuous_outcome_event_n_must_be_null" in _issue_types(findings)


def test_primary_model_contract_rejects_quantile_log_odds_scale(tmp_path: Path):
    cohort_path, out_dir, contracts = _write_mixed_outcome_inputs(tmp_path)
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    table.loc[table["model_id"].eq("continuous_secondary"), "effect_scale"] = "log_odds"
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "effect_scale_model_family_mismatch" in _issue_types(findings)


def test_primary_model_contract_requires_scale_for_continuous_fitted_terms(
    tmp_path: Path,
):
    cohort_path, out_dir, contracts = _write_mixed_outcome_inputs(tmp_path)
    table_path = out_dir / "model_coefficients.csv"
    table = pd.read_csv(table_path)
    table.loc[table["model_id"].eq("continuous_secondary"), "effect_scale"] = None
    table.to_csv(table_path, index=False)

    findings = PrimaryModelContractValidator().audit(
        step=_step(),
        step_summary={"model_contracts": contracts},
        context=_context(),
        completed_step_records=[],
        out_dir=out_dir,
        cohort_path=cohort_path,
    )

    assert "continuous_fitted_term_requires_effect_scale" in _issue_types(findings)


def test_zero_cell_audit_detects_low_cardinality_numeric_category():
    frame = pd.DataFrame(
        {
            "death": [0, 1] * 10 + [0] * 10,
            "admission_code": [0] * 10 + [1] * 10 + [2] * 10,
        }
    )

    cells = PrimaryModelContractValidator._categorical_zero_event_cells(
        frame=frame,
        outcome="death",
        covariates=["admission_code"],
        contract={
            "baseline_missing_policy": "drop_missing_baseline",
            "analysis_set": "source_aware",
        },
    )

    assert any(cell["level"] == "2" and cell["event_n"] == 0 for cell in cells)


def test_zero_cell_audit_respects_explicit_numeric_categorical_declaration():
    frame = pd.DataFrame(
        {
            "death": [0, 1, 0, 1, 0, 0],
            "admission_code": [0, 0, 1, 1, 2, 2],
        }
    )

    cells = PrimaryModelContractValidator._categorical_zero_event_cells(
        frame=frame,
        outcome="death",
        covariates=["admission_code"],
        contract={
            "baseline_missing_policy": "drop_missing_baseline",
            "analysis_set": "source_aware",
            "categorical_covariates": ["admission_code"],
        },
    )

    assert any(cell["level"] == "2" and cell["event_n"] == 0 for cell in cells)


def test_primary_model_contract_rejects_unverified_penalized_provenance(
    tmp_path: Path,
):
    contracts = copy.deepcopy(_contracts())
    contracts[0].pop("interval_method")
    contracts[0].pop("convergence_method")
    contracts[0].pop("optimizer_success")

    issues = _issue_types(_audit(tmp_path, contracts=contracts))

    assert "penalized_intervals_require_controlled_provenance" in issues
    assert "penalized_convergence_not_verified" in issues


def test_primary_model_contract_accepts_model_bound_nested_ridge_diagnostics(
    tmp_path: Path,
):
    contracts = copy.deepcopy(_contracts())
    contract = contracts[0]
    contract.pop("convergence_method")
    contract.pop("optimizer_success")
    contract["diagnostics"] = {
        "ridge_converged": True,
        "ridge_iterations": 98,
    }

    assert _audit(tmp_path, contracts=contracts) == []


@pytest.mark.parametrize(
    ("fit_method", "penalized", "diagnostics"),
    [
        (
            "sklearn_ridge_logistic_regression(C=1)",
            True,
            {"ridge_converged": False, "ridge_iterations": 98},
        ),
        (
            "sklearn_ridge_logistic_regression(C=1)",
            True,
            {"ridge_converged": True},
        ),
        (
            "statsmodels_regularized_logit",
            True,
            {"ridge_converged": True, "ridge_iterations": 98},
        ),
        (
            "sklearn_ridge_logistic_regression(C=1)",
            False,
            {"ridge_converged": True, "ridge_iterations": 98},
        ),
        (
            "sklearn_ridge_logistic_regression(C=1)",
            True,
            {
                "model_id": "organ_source_aware",
                "ridge_converged": True,
                "ridge_iterations": 98,
            },
        ),
        (
            "sklearn_ridge_logistic_regression(C=1)",
            True,
            {"ridge_converged": True, "ridge_iterations": -1},
        ),
    ],
)
def test_primary_model_contract_rejects_unsafe_nested_ridge_aliases(
    tmp_path: Path,
    fit_method: str,
    penalized: bool,
    diagnostics: dict,
):
    contracts = copy.deepcopy(_contracts())
    contract = contracts[0]
    contract.pop("convergence_method")
    contract.pop("optimizer_success")
    contract["fit_method"] = fit_method
    contract["penalized"] = penalized
    contract["diagnostics"] = diagnostics

    issues = _issue_types(_audit(tmp_path, contracts=contracts))

    assert "penalized_convergence_not_verified" in issues


def test_contract_repair_log_preserves_structured_issue_details() -> None:
    payload = _contract_repair_log(
        [
            ValidationFinding(
                validator="primary_model_contract",
                severity="error",
                message="Contract failed.",
                detail={
                    "issues": [
                        {
                            "model_id": "lab_source_aware",
                            "issue": "locked_primary_expression_mismatch",
                            "expected": "log1p(lab_max)",
                            "reported": "verbose prose",
                        }
                    ]
                },
            )
        ]
    )

    assert "locked_primary_expression_mismatch" in payload
    assert "lab_source_aware" in payload
    assert "log1p(lab_max)" in payload


def test_coder_prompt_declares_primary_model_canonical_enums() -> None:
    prompt = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "providers"
        / "prompts"
        / "v1"
        / "coder.txt"
    ).read_text(encoding="utf-8")

    assert "`analysis_set` is exactly `source_aware` or `complete_case`" in prompt
    assert "`drop_missing_baseline` or" in prompt
    assert "`fit_status` is exactly `fitted`" in prompt
    assert 'keep `exposure_role="primary"`' in prompt
    assert "use and report `alpha <= 1/n`" in prompt
    assert "planner-owned `model_requirements`" in prompt
    assert "matching `requirement_id`" in prompt
    assert "`fit_failure_reason`" in prompt
    assert "not a generic contract for" in prompt


def test_planner_prompt_declares_typed_model_requirements() -> None:
    from easyicu.research_agent.agents.core import _build_planner_user_prompt

    prompt = _build_planner_user_prompt(_context())

    assert "`model_requirements` roster" in prompt
    assert "`required_for_step_success`" in prompt
    assert "the execution layer only verifies it" in prompt
    assert "currently covers only" in prompt
    assert "Leave the array empty for survival" in prompt
    assert "only a sensitivity entry may be optional" in prompt


def test_replanner_preserves_requirements_without_cross_family_drift() -> None:
    prompt = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "providers"
        / "prompts"
        / "v1"
        / "replanner.txt"
    ).read_text(encoding="utf-8")

    assert "Preserve planner-owned `model_requirements`" in prompt
    assert "A failed fit is not permission" in prompt
    assert "Do not add this roster to" in prompt


def test_primary_model_contract_rejects_unreported_or_strong_ridge_penalty(
    tmp_path: Path,
):
    contracts = copy.deepcopy(_contracts())
    contracts[0]["fit_method"] = "statsmodels_glm_l2_regularized"
    missing_path = tmp_path / "missing_strength"
    missing_path.mkdir()
    missing_strength = _issue_types(_audit(missing_path, contracts=contracts))
    assert "statsmodels_penalty_strength_not_reported" in missing_strength

    contracts[0]["fit_method"] = "statsmodels_glm_ridge(alpha=1.0)"
    strong_path = tmp_path / "strong_penalty"
    strong_path.mkdir()
    strong_penalty = _issue_types(_audit(strong_path, contracts=contracts))
    assert "statsmodels_penalty_too_strong_for_separation_fallback" in strong_penalty


def test_primary_model_contract_rejects_penalized_method_with_false_flag(
    tmp_path: Path,
) -> None:
    contracts = copy.deepcopy(_contracts())
    contracts[0].update(
        {
            "penalized": False,
            "fit_method": "statsmodels_regularized_logit",
        }
    )

    issues = _issue_types(_audit(tmp_path, contracts=contracts))

    assert "penalized_method_must_report_penalized_true" in issues
    assert "statsmodels_penalty_strength_not_reported" in issues


def test_primary_model_contract_accepts_reported_weak_ridge_penalty(
    tmp_path: Path,
):
    contracts = copy.deepcopy(_contracts())
    for contract in contracts:
        contract["fit_method"] = "statsmodels_glm_ridge(alpha=1e-7)"

    assert _audit(tmp_path, contracts=contracts) == []
