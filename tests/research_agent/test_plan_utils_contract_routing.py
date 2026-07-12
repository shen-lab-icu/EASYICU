from __future__ import annotations

from easyicu.research_agent.plan_utils import (
    _clustering_contract_applies,
    _effect_contract_applies,
    _prediction_contract_applies,
    _step_contract_findings,
    _step_contract_repair_guidance,
    _step_declares_audit_panel,
)
from easyicu.research_agent.schema import AnalysisStep


def _step(
    *,
    method: str,
    outputs: list[str],
    step_id: str = "01_analysis",
    intent: str = "Run the declared analysis.",
) -> AnalysisStep:
    return AnalysisStep(
        step_id=step_id,
        intent=intent,
        method=method,
        expected_outputs=outputs,
    )


def _errors(step: AnalysisStep, summary: dict | None = None):
    return [
        finding
        for finding in _step_contract_findings(
            step=step,
            step_summary=summary or {"status": "ok"},
        )
        if finding.severity == "error"
    ]


def _clustering_step() -> AnalysisStep:
    return _step(
        method="kmeans_clustering",
        outputs=["table:cluster_characteristics"],
        step_id="02_phenotype_discovery",
        intent="Discover phenotypes using the declared clustering method.",
    )


def _cluster_selection(*, selected_n_clusters: int) -> dict:
    return {
        "criterion": "silhouette_score",
        "selection_rule": "maximum",
        "direction": "maximize",
        "selected_n_clusters": selected_n_clusters,
        "candidates": [
            {
                "n_clusters": 2,
                "criterion_value": 0.9 if selected_n_clusters == 2 else 0.7,
            },
            {
                "n_clusters": 3,
                "criterion_value": 0.9 if selected_n_clusters == 3 else 0.7,
            },
        ],
        "rationale": "Selected the candidate with the maximum criterion value.",
    }


def _cluster_stability(*, selected_n_clusters: int | None) -> dict:
    evidence = {
        "n_resamples": 5,
        "mean_adjusted_rand_index": 0.82,
    }
    if selected_n_clusters is not None:
        evidence["selected_n_clusters"] = selected_n_clusters
    return evidence


def test_clustering_contract_requires_exact_method_head_and_closed_product():
    assert _clustering_contract_applies(
        method="k-means with silhouette review",
        expected_outputs=["table:cluster_characteristics"],
    )
    assert not _clustering_contract_applies(
        method="data_quality_audit",
        step_id="02_primary_phenotype_clustering",
        intent="Cluster ICU patients using trajectory features.",
        expected_outputs=[
            "table:cluster_characteristics",
            "statistic:cluster_count",
        ],
    )
    assert not _clustering_contract_applies(
        method="kmeans",
        expected_outputs=["table:cluster_characteristics_review"],
    )
    assert not _clustering_contract_applies(
        method="kmeans",
        expected_outputs=["log:cluster_count"],
    )


def test_invalid_explicit_cluster_selection_cannot_be_laundered_by_stability():
    invalid_manifests = (
        # Malformed: required selection fields/candidates are absent.
        {"selected_n_clusters": 3},
        # Internally valid, but contradictory to cluster_count=3.
        _cluster_selection(selected_n_clusters=2),
    )
    for invalid_manifest in invalid_manifests:
        summary = {
            "status": "ok",
            "cluster_count": 3,
            "cluster_selection": invalid_manifest,
            # A valid alternative must not rescue an explicitly invalid
            # authoritative manifest in the same summary.
            "cluster_stability": _cluster_stability(selected_n_clusters=3),
        }

        assert any(
            "clustering summary" in finding.message
            for finding in _errors(_clustering_step(), summary)
        )


def test_cluster_stability_must_bind_the_reported_cluster_count():
    for selected_n_clusters in (None, 2):
        summary = {
            "status": "ok",
            "cluster_count": 3,
            "cluster_stability": _cluster_stability(
                selected_n_clusters=selected_n_clusters
            ),
        }
        assert any(
            "clustering summary" in finding.message
            for finding in _errors(_clustering_step(), summary)
        )

    valid_summary = {
        "status": "ok",
        "cluster_count": 3,
        "cluster_stability": _cluster_stability(selected_n_clusters=3),
    }
    assert _errors(_clustering_step(), valid_summary) == []


def test_invalid_sibling_cluster_manifest_blocks_fallback_laundering():
    findings = _step_contract_findings(
        step=_clustering_step(),
        step_summary={"status": "ok"},
        completed_step_records=[
            {
                "step_id": "01_clustering_fit",
                "status": "ok",
                "step_summary": {
                    "cluster_count": 3,
                    "cluster_selection": _cluster_selection(
                        selected_n_clusters=2
                    ),
                    "cluster_stability": _cluster_stability(
                        selected_n_clusters=3
                    ),
                },
            }
        ],
    )

    assert any(
        finding.severity == "error" and "clustering summary" in finding.message
        for finding in findings
    )
    assert not any(
        finding.detail.get("fallback_step_id") == "01_clustering_fit"
        for finding in findings
    )


def test_sibling_stability_must_bind_the_sibling_cluster_count():
    for selected_n_clusters in (None, 2):
        findings = _step_contract_findings(
            step=_clustering_step(),
            step_summary={"status": "ok"},
            completed_step_records=[
                {
                    "step_id": "01_clustering_fit",
                    "status": "ok",
                    "step_summary": {
                        "cluster_count": 3,
                        "cluster_stability": _cluster_stability(
                            selected_n_clusters=selected_n_clusters
                        ),
                    },
                }
            ],
        )
        assert any(finding.severity == "error" for finding in findings)

    valid_findings = _step_contract_findings(
        step=_clustering_step(),
        step_summary={"status": "ok"},
        completed_step_records=[
            {
                "step_id": "01_clustering_fit",
                "status": "ok",
                "step_summary": {
                    "cluster_count": 3,
                    "cluster_stability": _cluster_stability(
                        selected_n_clusters=3
                    ),
                },
            }
        ],
    )
    assert not any(finding.severity == "error" for finding in valid_findings)
    assert any(
        finding.detail.get("fallback_step_id") == "01_clustering_fit"
        for finding in valid_findings
    )


def test_effect_contract_ignores_id_intent_and_output_without_method_owner():
    audit = _step(
        method="data_quality_audit",
        step_id="04_primary_association_model",
        intent="Audit the regression inputs and expected odds-ratio output.",
        outputs=["statistic:adjusted_or_ci"],
    )

    assert not _effect_contract_applies(audit)
    assert _errors(audit) == []
    assert "association step must" not in _step_contract_repair_guidance(
        step=audit,
        step_summary={"status": "ok"},
        code="",
    )


def test_effect_contract_requires_closed_product_from_exact_method_owner():
    prep = _step(
        method="logistic_regression",
        outputs=["table:model_input_audit"],
    )
    log_only = _step(
        method="logistic_regression",
        outputs=["log:odds_ratio"],
    )
    owner = _step(
        method="logistic regression with cluster-robust standard errors",
        outputs=["statistic:adjusted_or_ci"],
    )

    assert not _effect_contract_applies(prep)
    assert not _effect_contract_applies(log_only)
    assert _errors(prep) == []
    assert _effect_contract_applies(owner)
    assert any(
        "primary association estimate" in finding.message for finding in _errors(owner)
    )
    assert "association step must" in _step_contract_repair_guidance(
        step=owner,
        step_summary={"status": "ok"},
        code="",
    )


def test_prediction_contract_does_not_claim_feature_prep_or_model_only_output():
    prep = _step(
        method="feature_preparation",
        step_id="03_model_training",
        intent="Prepare features for the mortality prediction model.",
        outputs=["statistic:auroc"],
    )
    model_only = _step(
        method="prediction_model",
        outputs=["model:trained_prediction_model"],
    )
    figure_only = _step(
        method="prediction_model",
        outputs=["figure:auroc"],
    )

    assert not _prediction_contract_applies(prep)
    assert not _prediction_contract_applies(model_only)
    assert not _prediction_contract_applies(figure_only)
    assert _errors(prep) == []
    assert _errors(model_only) == []
    assert "prediction step must" not in _step_contract_repair_guidance(
        step=prep,
        step_summary={"status": "ok"},
        code="",
    )


def test_prediction_contract_requires_metrics_from_exact_method_owner():
    owner = _step(
        method="prediction model with cross validation",
        outputs=["statistic:auroc"],
    )

    assert _prediction_contract_applies(owner)
    messages = [finding.message for finding in _errors(owner)]
    assert any("AUROC" in message for message in messages)
    assert any("Brier" in message for message in messages)
    assert "prediction step must" in _step_contract_repair_guidance(
        step=owner,
        step_summary={"status": "ok"},
        code="",
    )


def test_cohort_change_requires_exact_owner_and_closed_attrition_product():
    owner = _step(
        method="cohort definition sensitivity with binomial glm",
        outputs=["table:cohort_overlap"],
    )
    audit = _step(
        method="data_quality_audit",
        step_id="01_primary_cohort_flow",
        intent="Review alternative eligibility across cohort definitions.",
        outputs=["table:cohort_overlap"],
    )
    reconciliation = _step(
        method="cohort_definition_reconciliation",
        outputs=["table:cohort_overlap"],
    )
    near_match = _step(
        method="cohort_definition_sensitivity",
        outputs=["table:cohort_overlap_review"],
    )
    log_only = _step(
        method="cohort_definition_sensitivity",
        outputs=["log:cohort_overlap"],
    )
    summary = {"status": "ok", "analysis_family": "cohort_definition_sensitivity"}

    assert _errors(owner, summary) == []
    for non_owner in (audit, reconciliation, near_match, log_only):
        assert any(
            finding.detail.get("kind") == "unauthorized_cohort_redefinition"
            for finding in _errors(non_owner, summary)
        )


def test_audit_panel_tokens_use_word_or_snake_case_boundaries():
    positive = _step(
        method="visualization",
        outputs=["figure:data_completeness"],
    )
    near_match = _step(
        method="review",
        intent="Discuss auditability and the calibrationist perspective.",
        outputs=["table:review"],
    )

    assert _step_declares_audit_panel(positive)
    assert not _step_declares_audit_panel(near_match)
