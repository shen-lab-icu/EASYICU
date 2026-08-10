from __future__ import annotations

import math


def _association_step(ra, step_id: str = "04_adjusted_association"):
    return ra.AnalysisStep(
        step_id=step_id,
        method="adjusted_logistic_regression",
        intent="Estimate the adjusted association between SOFA-2 and mortality.",
        planned_analysis_role="primary",
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


def test_primary_step_cannot_borrow_a_sibling_effect_estimate(ra):
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

    errors = _errors(findings)
    assert errors
    assert "primary association estimate" in errors[0].message
    assert not any("fallback_step_id" in finding.detail for finding in findings)


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
        method="prediction_model_evaluation",
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


def _feature_freeze_prep_step(ra):
    """The M3 ``01_feature_audit_and_primary_set_freeze`` shape: a
    ``data_quality_audit`` step that freezes the feature set *for* downstream
    clustering. Its intent and expected_outputs mention clustering, but it does
    not fit clusters itself."""
    return ra.AnalysisStep(
        step_id="01_feature_audit_and_primary_set_freeze",
        method="data_quality_audit",
        intent=(
            "Audit feature availability and freeze a primary clustering feature "
            "set that avoids letting sparse variables dominate cluster geometry."
        ),
        expected_outputs=[
            "table:feature_availability_and_missingness",
            "table:feature_selection_for_primary_clustering",
            "manifest:primary_clustering_feature_spec",
        ],
    )


def test_feature_freeze_prep_step_not_subject_to_clustering_contract(ra):
    """Regression (M3): a feature-freeze/audit step that merely *mentions*
    clustering must not be forced to report a silhouette/cluster count. It
    self-declared it "froze the feature set but did not fit clusters" yet its
    null placeholder metrics fail-closed the entire run."""
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_feature_freeze_prep_step(ra),
        step_summary={
            "statistic:silhouette_score": None,
            "statistic:cluster_count": None,
            "clustering_summary": {
                "cluster_count": None,
                "silhouette_score": None,
                "no_ground_truth_note": (
                    "This step froze the feature set but did not fit clusters."
                ),
            },
            "primary_feature_freeze": {"n_features": 15},
        },
    )

    assert _errors(findings) == []


def _clustering_figure_step(
    ra, step_id: str = "02_primary_phenotype_clustering_figure"
):
    return ra.AnalysisStep(
        step_id=step_id,
        method="clustering",
        intent="Render the primary phenotype clustering assignment panel.",
        expected_outputs=[
            "figure:cluster_scatter",
            "statistic:silhouette_score",
            "statistic:cluster_count",
        ],
    )


def test_clustering_metric_satisfied_by_sibling_clustering_step(ra):
    """A clustering figure/render step whose own summary lacks the metric under a
    recognised key must not fail when the dedicated clustering step genuinely
    produced and bound it (M3 cross-step analogue of the AUROC fallback)."""
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=_clustering_figure_step(ra),
        step_summary={
            "silhouette_score": None,
            "cluster_count": None,
            "registered_evidence_step": "02_primary_phenotype_clustering",
        },
        completed_step_records=[
            {
                "step_id": "02_primary_phenotype_clustering",
                "status": "ok",
                "step_summary": {
                    "silhouette_score": 0.32642819634210984,
                    "statistic:silhouette_score": 0.32642819634210984,
                    "cluster_count": 2,
                    "statistic:cluster_count": 2,
                    "cluster_selection": {
                        "criterion": "silhouette_score",
                        "selection_rule": "maximum",
                        "direction": "maximize",
                        "selected_n_clusters": 2,
                        "candidates": [
                            {"n_clusters": 1, "criterion_value": 0.0},
                            {"n_clusters": 2, "criterion_value": 0.32642819634210984},
                        ],
                        "rationale": "Maximum among evaluated candidates.",
                        "candidate_range_boundary_rule": "allow_upper_boundary",
                        "candidate_range_boundary_reason_code": None,
                    },
                },
            }
        ],
    )

    assert _errors(findings) == []
    assert any(
        finding.severity == "warning"
        and finding.detail.get("fallback_step_id") == "02_primary_phenotype_clustering"
        and "cluster" in finding.message.lower()
        for finding in findings
    )


def test_clustering_metric_missing_everywhere_still_errors(ra):
    """The clustering fallback only credits a genuinely-bound sibling metric —
    when no step produced a silhouette/cluster count, the requirement must still
    fail (no silent pass)."""
    from easyicu.research_agent.pipeline import _step_contract_findings

    findings = _step_contract_findings(
        step=ra.AnalysisStep(
            step_id="02_primary_phenotype_clustering",
            method="clustering",
            intent="Fit the primary phenotype clustering.",
            expected_outputs=["statistic:silhouette_score", "statistic:cluster_count"],
        ),
        step_summary={"silhouette_score": None, "cluster_count": None},
        completed_step_records=[
            {
                "step_id": "01_feature_audit_and_primary_set_freeze",
                "status": "ok",
                "step_summary": {"primary_feature_freeze": {"n_features": 15}},
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
