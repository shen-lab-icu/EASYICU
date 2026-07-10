"""Routing test for the deterministic trajectory-clustering (phenotyping) runner.

H3's phenotyping runs fail-closed when the multi-step clustering analysis (feature
engineering over trajectory windows + model selection + stability) is left to the
flaky LLM coder. The deterministic ``trajectory_clustering`` runner owns that step,
routed by a MODULE-LEVEL pure predicate ``_trajectory_clustering_step_matches`` so
the routing is verifiable without a full bench.

These tests lock the two properties that matter: the predicate FIRES on a real
phenotyping / trajectory-clustering primary step (however the planner phrases the
method), and it does NOT hijack a primary EFFECT step (which carries an OR/HR/AUROC
estimand the clustering runner cannot produce) nor an unrelated step.
"""

from __future__ import annotations

from easyicu.research_agent.pipeline_execute import (
    _trajectory_clustering_step_matches as _matches,
)


def test_matches_explicit_controlled_method():
    # The planner's controlled method wins outright, whatever the prose says.
    for method in (
        "trajectory_clustering",
        "trajectory_feature_clustering",
        "clustering",
        "kmeans_clustering",
        "phenotyping",
        "unsupervised_clustering",
        "latent_class",
        "cluster_analysis",
    ):
        assert _matches(method, "") is True, method


def test_matches_trajectory_phenotyping_prose_without_controlled_method():
    # A realistic H3 primary step whose method is a generic label but whose intent
    # is unambiguously a trajectory-clustering / subphenotype discovery step.
    assert (
        _matches(
            "analysis",
            "04_trajectory_subphenotypes cluster patients on 0-72h sofa2 "
            "trajectory features; unsupervised k-means with silhouette-selected k; "
            "discover latent subphenotypes "
            "trajectory_features cluster_assignments cluster_characteristics "
            "silhouette_metrics",
        )
        is True
    )


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
    assert (
        _matches(
            "cox_regression",
            "cox proportional-hazards model of survival by cluster membership; "
            "report the hazard ratio hazard_ratio survival_curve",
        )
        is False
    )


def test_does_not_fire_on_unrelated_step():
    assert _matches("association_analysis", "adjusted association of lactate") is False
    assert _matches("missingness_audit", "per-concept measured vs missing counts") is False
    assert _matches("", "") is False
