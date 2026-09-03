from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.contracts.step_families import (
    _clustering_contract_applies,
    _effect_contract_applies,
    _prediction_contract_applies,
)
from easyicu.research_agent.gates.step_contract import (
    _step_contract_findings,
)
from easyicu.research_agent.gates.step_repair import (
    _step_contract_repair_guidance,
)
from easyicu.research_agent.planning.figure_plan_shaping import (
    step_declares_audit_panel,
)
from easyicu.research_agent.schema import AnalysisStep


def test_plan_contract_owners_do_not_depend_on_compatibility_catch_all() -> None:
    root = Path(__file__).resolve().parents[3] / "src/easyicu/research_agent"
    gate = (root / "gates/contract.py").read_text(encoding="utf-8")
    final_owner = (root / "planning/final_plan_shape.py").read_text(encoding="utf-8")
    compatibility = (root / "plan_utils.py").read_text(encoding="utf-8")
    execution_owner = (root / "execution/phase_support.py").read_text(
        encoding="utf-8"
    )
    production_consumers = [
        path
        for path in root.rglob("*.py")
        if path.name != "plan_utils.py"
    ]

    assert all(
        "plan_utils import" not in path.read_text(encoding="utf-8")
        and "import plan_utils" not in path.read_text(encoding="utf-8")
        for path in production_consumers
    )
    assert "_plan_contracts" not in gate
    assert "plan_utils" not in final_owner
    assert "plan_utils" not in execution_owner
    assert len(compatibility.splitlines()) < 100
    assert "from .contracts.step_families import (" in compatibility
    assert "from .gates.step_contract import _step_contract_findings" in compatibility
    assert "from .planning.plan_graph import (" in compatibility


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


def test_clustering_contract_accepts_planner_structure_owner_method():
    assert _clustering_contract_applies(
        method="phenotype_clustering_and_structure",
        expected_outputs=[
            "dataset:cluster_assignments",
            "table:phenotype_structure",
        ],
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


def test_clustering_contract_repair_guidance_names_the_accepted_summary_shapes():
    guidance = _step_contract_repair_guidance(
        step=_clustering_step(),
        step_summary={
            "status": "ok",
            "cluster_count": 2,
            "selected_silhouette": 0.31,
            "selected_stability_ari": 0.93,
        },
        code="",
    )

    assert '"cluster_selection"' in guidance
    assert '"selection_rule"' in guidance
    assert '"selected_n_clusters"' in guidance
    assert '"candidates"' in guidance
    assert '"cluster_stability"' in guidance
    assert '"n_resamples"' in guidance
    assert '"mean_adjusted_rand_index"' in guidance


def test_mixed_clustering_figure_repair_guidance_uses_declared_figure_stem():
    step = _step(
        method="kmeans_clustering",
        outputs=[
            "statistic:cluster_count",
            "manifest:cluster_selection",
            "table:cluster_characteristics",
            "figure:clustering_visualization",
        ],
        step_id="06_fit_candidate_clusters",
        intent="Fit candidate clusters and render the declared visualization.",
    )

    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={
            "cluster_count": 2,
            "figure_contract": "clustering_visualization_figure_contract.json",
        },
        code="fig.savefig(out_dir / 'clustering_visualization.png')",
    )

    assert "save_publication_figure" in guidance
    assert 'stem="clustering_visualization"' in guidance
    assert "clustering_visualization.figure_contract.json" in guidance
    assert "clustering_visualization_figure_contract.json" in guidance
    assert "must not" in guidance


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


def test_non_effect_method_does_not_gain_effect_scope_from_id_or_output():
    audit = _step(
        method="data_quality_audit",
        step_id="04_primary_association_model",
        intent="Audit the regression inputs and expected odds-ratio output.",
        outputs=["statistic:adjusted_or_ci"],
    )

    assert not _effect_contract_applies(audit)
    errors = _errors(audit)
    assert any(
        finding.detail.get("kind") == "unauthorized_effect_product"
        for finding in errors
    )
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


def test_cohort_definition_sensitivity_may_own_declared_effect_replay():
    sensitivity = _step(
        method="cohort_definition_sensitivity",
        step_id="robustness",
        outputs=[
            "table:cohort_overlap",
            "statistic:primary_or",
            "table:robustness_summary",
        ],
    )
    summary = {
        "status": "ok",
        "primary_or": 1.2,
        "output_files": {
            "table:cohort_overlap": "cohort_overlap.csv",
            "statistic:primary_or": 1.2,
            "table:robustness_summary": "robustness_summary.csv",
        },
    }

    errors = _errors(sensitivity, summary)

    assert not any(
        finding.detail.get("kind") == "unauthorized_effect_product"
        for finding in errors
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
        method="cohort_definition_sensitivity",
        outputs=["table:cohort_overlap_and_attrition"],
    )
    audit = _step(
        method="data_quality_audit",
        step_id="01_primary_cohort_flow",
        intent="Review alternative eligibility across cohort definitions.",
        outputs=["table:cohort_overlap"],
    )
    reconciliation = _step(
        method="sensitivity_analysis",
        outputs=["table:cohort_overlap_and_attrition"],
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

    assert step_declares_audit_panel(positive)
    assert not step_declares_audit_panel(near_match)
