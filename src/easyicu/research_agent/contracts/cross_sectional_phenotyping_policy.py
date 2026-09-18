"""Published settings consumed by the native phenotyping adapter and Planner.

This is the adapter's fixed analysis-only method, not permission for the model
to choose settings or a claim that conditional agreement proves reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossSectionalPhenotypingPolicy:
    random_seed: int = 1729
    minimum_rows: int = 20
    candidate_k: tuple[int, ...] = (2, 3, 4, 5, 6)
    silhouette_sample_limit: int = 10_000
    kmeans_n_init: int = 10
    kmeans_batch_size_limit: int = 2048
    imputation: str = "median"
    n_resamples: int = 5
    sample_fraction: float = 0.8
    gmm_covariance_type: str = "diag"
    gmm_n_init: int = 5
    gmm_max_iter: int = 500
    gmm_reg_covar: float = 1e-6
    complete_case_assignment_bootstraps: int = 200

    def kmeans_parameters(self, n_rows: int, *, seed_offset: int = 0) -> dict:
        return {
            "random_state": self.random_seed + seed_offset,
            "n_init": self.kmeans_n_init,
            "batch_size": min(self.kmeans_batch_size_limit, n_rows),
        }

    def parameters(self, action_id: str) -> tuple[tuple[str, object], ...]:
        common = (
            ("policy_version", "easyicu.cross_sectional_phenotyping_policy/1"),
            ("random_seed", self.random_seed),
            ("minimum_rows", self.minimum_rows),
            ("kmeans_n_init", self.kmeans_n_init),
            ("kmeans_batch_size_limit", self.kmeans_batch_size_limit),
        )
        if action_id in {"phenotyping.cluster_solution", "phenotyping.k_selection"}:
            selection = (
                ("candidate_k", self.candidate_k),
                ("silhouette_sample_limit", self.silhouette_sample_limit),
                ("silhouette_sampling", "without_replacement_fixed_seed"),
                ("selection_rule", "maximum_silhouette_then_lower_k"),
            )
            if action_id == "phenotyping.k_selection":
                return common + selection + (("input_representation", "sealed_standardized_primary_matrix"),)
            return common + selection + (
                ("imputation", self.imputation),
                ("standardization", "standard_scaler"),
                ("declared_numeric_inputs", "clustering_features_not_profile_only"),
                ("outcome_used_for_fit", False),
                ("complete_case_uncertainty_if_locked", "paired_assignment_bootstrap_not_pipeline_refits"),
                ("complete_case_assignment_bootstraps", self.complete_case_assignment_bootstraps),
            )
        if action_id == "phenotyping.cluster_stability":
            return common + (
                ("resampling_method", "subsampling_without_replacement"),
                ("n_resamples", self.n_resamples),
                ("sample_fraction", self.sample_fraction),
                ("sample_size_rule", "max_minimum_rows_floor_fraction_times_n"),
                ("replicate_seed_rule", "random_seed_plus_one_based_replicate"),
                ("preprocessing_scope", "fixed_full_primary_cohort"),
                ("k_selection_scope", "fixed_primary_selected_k"),
                ("uncertainty_scope", "conditional_agreement_not_full_pipeline_bootstrap"),
                ("gmm_covariance_type", self.gmm_covariance_type),
                ("gmm_n_init", self.gmm_n_init),
                ("gmm_max_iter", self.gmm_max_iter),
                ("gmm_reg_covar", self.gmm_reg_covar),
                ("external_reproducibility_established", False),
            )
        raise ValueError("unsupported cross-sectional phenotyping policy action")


CROSS_SECTIONAL_PHENOTYPING_POLICY = CrossSectionalPhenotypingPolicy()
