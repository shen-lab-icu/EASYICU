"""Anti-hijack contract for the optional KMeans clustering auxiliary.

The agent owns the scientific method.  The deterministic implementation is not a
preflight owner and may be considered compatible only when the agent explicitly
planned KMeans and declared a closed clustering-product contract.
"""

from __future__ import annotations

from easyicu.research_agent.pipeline_execute import (
    _trajectory_clustering_step_matches as _matches,
)
from easyicu.research_agent.plan_utils import _step_contract_findings
from easyicu.research_agent.pipeline import _semantic_aliases_for
from easyicu.research_agent.schema import AnalysisStep


def test_kmeans_requires_a_closed_clustering_product_contract():
    assert not _matches("kmeans_clustering", "")
    assert not _matches(
        "kmeans_clustering",
        "discover trajectory phenotypes",
        "table:unowned_custom_contract",
    )
    assert _matches(
        "kmeans_clustering",
        "discover trajectory phenotypes with KMeans",
        "table:cluster_assignments table:cluster_characteristics "
        "statistic:silhouette_score",
    )


def test_auxiliary_does_not_choose_an_unspecified_or_different_algorithm():
    outputs = (
        "table:cluster_assignments table:cluster_characteristics "
        "statistic:silhouette_score"
    )
    assert not _matches(
        "analysis",
        "discover phenotypes with an unsupervised method",
        outputs,
    )
    assert not _matches(
        "latent_class",
        "fit a latent-class model",
        outputs,
    )


def test_kmeans_method_spelling_normalization_keeps_closed_contract():
    outputs = (
        "table:cluster_assignments table:cluster_characteristics "
        "statistic:silhouette_score"
    )
    for method in ("k-means", "k_means", "kmeans clustering"):
        assert _matches(
            method,
            "discover longitudinal phenotypes with KMeans",
            outputs,
        ), method


def test_does_not_hijack_a_primary_effect_step():
    # A primary EFFECT step can mention patient "subgroups"/"phenotype" in prose;
    # the clustering runner emits no effect estimate, so it must NOT claim a step
    # whose estimand is an odds ratio / hazard ratio / AUROC.
    assert (
        _matches(
            "multivariable_association",
            "estimate the adjusted odds ratio of mortality across patient "
            "phenotype subgroups; primary association "
            "adjusted_effect_estimates odds_ratio",
        )
        is False
    )


def test_cluster_robust_mixed_effects_association_is_not_clustering():
    assert not _matches(
        "mixed_effects_regression",
        "Estimate the association with cluster-robust SE and hospital-level "
        "clustering.",
        "table:association_estimates.csv",
    )


def test_effect_output_spellings_and_hybrid_method_head_are_excluded():
    intent = "discover KMeans phenotypes with cluster assignments"
    for output in (
        "table:odds_ratio",
        "statistic:OR",
        "statistic:HR",
        "table:hazard_ratio",
        "statistic:c_statistic",
        "table:adjusted_odds_ratios",
        "table:model_ORs",
        "table:hazard_ratio_estimates",
        "table:association_estimates_by_cluster",
        "table:odds_ratio_by_cluster",
        "statistic:or_adjusted",
    ):
        assert not _matches("kmeans_clustering", intent, output), output
    assert not _matches(
        "mixed_effects_regression_with_kmeans_rider",
        intent,
        "table:cluster_assignments table:cluster_characteristics",
    )
    assert (
        _matches(
            "cox_regression",
            "cox proportional-hazards model of survival by cluster membership; "
            "report the hazard ratio hazard_ratio survival_curve",
        )
        is False
    )


def test_closed_kmeans_products_do_not_launder_a_mixed_effect_contract():
    clustering = "table:cluster_assignments table:cluster_characteristics"
    for effect_output in (
        "table:association_estimates.csv",
        "statistic:OR",
        "statistic:HR",
        "table:hazard_ratio",
    ):
        assert not _matches(
            "kmeans_clustering",
            "Discover trajectory phenotypes with KMeans.",
            f"{clustering} {effect_output}",
        ), effect_output


def test_does_not_fire_on_unrelated_step():
    assert _matches("association_analysis", "adjusted association of lactate") is False
    assert _matches("missingness_audit", "per-concept measured vs missing counts") is False
    assert _matches("", "") is False


def test_cluster_robust_association_does_not_receive_clustering_contract(tmp_path):
    step = AnalysisStep(
        step_id="05_primary_association",
        intent=(
            "Estimate the association with cluster-robust SE and hospital-level "
            "clustering."
        ),
        method="mixed_effects_regression",
        expected_outputs=["table:association_estimates"],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={"status": "ok", "adjusted_effect": 1.2},
    )
    assert not any("clustering summary" in item.message for item in findings)

    aliases = _semantic_aliases_for(step, tmp_path / "step_summary.json")
    assert "cluster_summary" not in aliases
    assert "clustering_performance" not in aliases


def test_real_clustering_contract_still_requires_a_cluster_metric():
    step = AnalysisStep(
        step_id="05_trajectory_phenotyping",
        intent="Discover trajectory phenotypes with KMeans.",
        method="kmeans_clustering",
        expected_outputs=[
            "table:cluster_assignments",
            "table:cluster_characteristics",
            "statistic:silhouette_score",
        ],
    )
    findings = _step_contract_findings(step=step, step_summary={"status": "ok"})
    assert any("clustering summary" in item.message for item in findings)
