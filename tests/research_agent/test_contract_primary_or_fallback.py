from __future__ import annotations

import math


def _association_step(ra, step_id: str = "04_adjusted_association"):
    return ra.AnalysisStep(
        step_id=step_id,
        intent="Estimate the adjusted association between SOFA-2 and mortality.",
        expected_outputs=[
            "model:logistic_regression_sofa_death",
            "statistic:adjusted_odds_ratio",
        ],
    )


def _errors(findings):
    return [finding for finding in findings if finding.severity == "error"]


def test_accepts_statistic_adjusted_sofa2_odds_ratio_dict(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra, step_id="03b_event_count_check"),
        step_summary={
            "statistic:adjusted_sofa2_odds_ratio": {
                "adjusted_odds_ratio": 1.3114628301617859,
                "ci_lower": 1.1779552418640442,
                "ci_upper": 1.46010195784202,
                "standard_error": 0.054778092768164525,
                "model_converged": True,
            }
        },
    )

    assert _errors(findings) == []


def test_accepts_primary_estimates_list_first_finite_or(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra),
        step_summary={
            "primary_estimates": [
                {
                    "stratum": "overall",
                    "odds_ratio": 1.42,
                    "ci_lower": 1.10,
                    "ci_upper": 1.83,
                }
            ]
        },
    )

    assert _errors(findings) == []


def test_accepts_generic_primary_statistic_value_with_ci(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra),
        step_summary={
            "statistic:primary_adjusted_or": {
                "value": 1.18,
                "ci_low": 1.02,
                "ci_high": 1.36,
            }
        },
    )

    assert _errors(findings) == []


def test_prior_successful_primary_effect_satisfies_later_primary_requirement(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra),
        step_summary={
            "skipped": [
                {
                    "variables": ["sofa_factor", "weight_missing_indicator"],
                    "reason": "column not present",
                }
            ],
            "notes": [],
        },
        completed_step_records=[
            {
                "step_id": "03b_event_count_check",
                "status": "ok",
                "step_summary": {
                    "statistic:adjusted_sofa2_odds_ratio": {
                        "adjusted_odds_ratio": 1.3114628301617859,
                        "ci_lower": 1.1779552418640442,
                        "ci_upper": 1.46010195784202,
                    }
                },
            }
        ],
    )

    assert _errors(findings) == []
    assert any(
        finding.severity == "warning"
        and finding.detail["fallback_step_id"] == "03b_event_count_check"
        for finding in findings
    )


def test_missing_exposure_or_outcome_count_is_not_primary_or(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra),
        step_summary={
            "cohort_definition": {
                "excluded_missing_exposure_or_outcome_n": 0,
            },
            "primary_or": None,
        },
    )

    errors = _errors(findings)
    assert errors
    assert "primary association estimate" in errors[0].message


def test_table_prevalence_step_does_not_satisfy_later_primary_model(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_association_step(ra, step_id="04_primary_adjusted_association_model"),
        step_summary={
            "primary_or": None,
            "skipped": [{"reason": "validated_modeling_cohort_not_found"}],
        },
        completed_step_records=[
            {
                "step_id": "02_table_one_and_prevalence",
                "status": "ok",
                "step_summary": {
                    "cohort_definition": {
                        "excluded_missing_exposure_or_outcome_n": 0,
                    },
                    "prevalence": {
                        "death_by_exposure": {
                            "exposed": {"prevalence": 0.12},
                            "unexposed": {"prevalence": 0.08},
                        }
                    },
                },
            }
        ],
    )

    errors = _errors(findings)
    assert errors
    assert "primary association estimate" in errors[0].message


def _prediction_figure_step(ra, step_id: str = "01_model_training_figure"):
    return ra.AnalysisStep(
        step_id=step_id,
        intent="Render discrimination and calibration panels for the mortality model.",
        expected_outputs=[
            "figure:discrimination_calibration",
            "statistic:auroc",
            "statistic:brier_score",
        ],
    )


def test_prediction_auroc_satisfied_by_sibling_training_step(ra):
    """A figure step that renders an upstream training step's metrics must not
    fail when its own summary lacks the metric under a recognised key but the
    training step genuinely produced and bound it (M2 regression)."""
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_prediction_figure_step(ra),
        step_summary={
            # the figure step's own renderer found nothing under its key list
            "auroc": None,
            "cv_auroc_mean": None,
            "brier_score": None,
            "registered_evidence_step": "01_model_training",
        },
        completed_step_records=[
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {
                    "auroc_test": 0.8267455907381426,
                    "statistic:auroc": 0.8267455907381426,
                    "brier_test": 0.1716274488483539,
                    "statistic:brier_score": 0.1716274488483539,
                    "model_status": "fit_success",
                },
            }
        ],
    )

    assert _errors(findings) == []
    assert any(
        finding.severity == "warning"
        and finding.detail.get("fallback_step_id") == "01_model_training"
        and "AUROC" in finding.message
        for finding in findings
    )
    assert any(
        finding.severity == "warning"
        and finding.detail.get("fallback_step_id") == "01_model_training"
        and "Brier" in finding.message
        for finding in findings
    )


def test_prediction_auroc_missing_everywhere_still_errors(ra):
    """The fallback only credits a genuinely-bound sibling metric — when no step
    produced an AUROC, the requirement must still fail (no silent pass)."""
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_prediction_figure_step(ra),
        step_summary={"auroc": None, "brier_score": None},
        completed_step_records=[
            {
                "step_id": "01_model_training",
                "status": "ok",
                "step_summary": {"model_status": "fit_failed"},
            }
        ],
    )

    assert _errors(findings)


def test_rejects_nonfinite_primary_effect_values(ra):
    from easyicu.research_agent.pipeline import _step_contract_findings

    for value in (math.nan, math.inf, -math.inf):
        findings = _step_contract_findings(
            step=_association_step(ra),
            step_summary={
                "statistic:adjusted_sofa2_odds_ratio": {
                    "adjusted_odds_ratio": value,
                    "ci_lower": 1.1779552418640442,
                    "ci_upper": 1.46010195784202,
                },
                "primary_estimates": [
                    {
                        "odds_ratio": value,
                        "ci_lower": 1.10,
                        "ci_upper": 1.83,
                    }
                ],
            },
        )

        assert _errors(findings)
