"""A signed-free upstream bundle for the trajectory stability executor.

Synthetic, well-separated clusters with opaque identifiers, the representation
and solution schemas, the cluster-selection manifest and their typed input
bindings.  Test modules may not import one another
(``tests/governance/test_test_organization.py``); these helpers moved here from
``test_trajectory_stability_executor`` when a second module needed them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.schema import TrajectoryStabilitySpec


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(run_dir: Path, path: Path, *, evidence_id: str) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": sha256_file(path),
    }


def stability_spec(
    *,
    decision_mode: str = "report_only",
    minimum_mean_stability: float | None = None,
) -> TrajectoryStabilitySpec:
    return TrajectoryStabilitySpec(
        resampling_method="subsample_without_replacement",
        n_resamples=2,
        sample_fraction=0.75,
        sample_fraction_rounding="floor",
        base_seed=271_828,
        seed_derivation="numpy_seedsequence_spawn_uint32_v1",
        cross_resample_membership="distinct_membership_required",
        stability_metric="adjusted_rand_index",
        stability_aggregation="mean",
        metric_label_source="raw_refit_labels_label_invariant",
        evaluation_scope="sampled_overlap",
        label_alignment="hungarian_maximum_overlap",
        label_alignment_reference="frozen_candidate_assignments",
        label_alignment_tie_break="minimum_rank_distance_then_lexicographic_v1",
        final_assignment_policy="copy_selected_candidate_labels",
        minimum_successful_resamples=2,
        failed_refit_policy="record_once_no_retry",
        refit_engine="easyicu_observed_data_diag_gmm_v1",
        refit_initialization="random_balanced_assignments",
        refit_max_iter=500,
        refit_tolerance=1e-5,
        refit_regularization=1e-6,
        decision_mode=decision_mode,
        minimum_mean_stability=minimum_mean_stability,
        threshold_failure_action="fail_closed_require_planner_revision",
    )


def write_upstream_bundle(
    run_dir: Path,
    *,
    n_clusters: int,
    id_column: str,
    representation_columns: tuple[str, ...],
    assignment_column: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(10_000 + n_clusters)
    n_per_cluster = 60
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    centers = np.zeros((n_clusters, len(representation_columns)), dtype=float)
    for cluster in range(n_clusters):
        centers[cluster] = (
            np.arange(1, len(representation_columns) + 1, dtype=float)
            * (cluster + 1)
            * 5.0
        )
    matrix = centers[labels] + rng.normal(
        0.0, 0.18, size=(len(labels), len(centers[0]))
    )
    # Exercise the observed-data implementation without creating all-missing rows.
    matrix[np.arange(len(labels)) % 17 == 0, -1] = np.nan

    identifiers = [
        f"opaque-unit-{n_clusters}-{index:04d}" for index in range(len(labels))
    ]
    representation = pd.DataFrame(matrix, columns=list(representation_columns))
    representation.insert(0, id_column, identifiers)
    reference_labels = [f"group::{100 + int(value)}" for value in labels]
    assignments = (
        pd.DataFrame({id_column: identifiers, assignment_column: reference_labels})
        .sample(frac=1.0, random_state=91)
        .reset_index(drop=True)
    )

    upstream = run_dir / "upstream"
    upstream.mkdir(parents=True)
    representation_path = upstream / "opaque_representation.parquet"
    assignment_path = upstream / "opaque_candidate_labels.csv"
    representation.to_parquet(representation_path, index=False)
    assignments.to_csv(assignment_path, index=False)

    representation_schema = {
        "schema_version": "easyicu.trajectory_representation_schema/2",
        "id_column": id_column,
        "representation_columns": list(representation_columns),
        "frozen_population_n": len(representation),
        "observation_family": "opaque_signal_family",
        "observation_columns": list(representation_columns),
        "min_observed_windows": 1,
        "profile_columns": list(representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "index_event",
        "anchor_provenance": "agent_declared",
        "anchor_source": "synthetic_contract_fixture",
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": {
            "method": "pooled_coordinate_wise_z_score",
            "ddof": 0,
            "observed_value_policy": "direct_or_owner_locf_available",
            "missing_value_policy": "preserve_missing_exclude_from_likelihood",
            "zero_variance_action": "fail_closed",
        },
        "evidence_state_policy": {
            "direct_observed": "include",
            "owner_locf_available": "include_and_audit",
            "unavailable": "exclude",
            "additional_clustering_stage_imputation": "none",
        },
        "representation_sha256": sha256_file(representation_path),
    }
    solution_schema = {
        "schema_version": "easyicu.candidate_cluster_solution_schema/2",
        "id_column": id_column,
        "representation_columns": list(representation_columns),
        "model_family": "latent_class_diagonal_gaussian_mixture",
        "fit_method": "observed_data_em_diagonal_gaussian_mixture",
        "covariance_type": "diag",
        "selected_n_clusters": n_clusters,
        "selected_model_id": f"opaque-model-k{n_clusters}",
        "assignment_column": assignment_column,
        "criterion": "bic",
        "selection_rule": "minimum",
        "direction": "minimize",
        "selected_criterion_value": 123.0,
        "representation_schema_sha256": "pending",
        "candidate_assignments_sha256": sha256_file(assignment_path),
        "coordinate_scaling": representation_schema["coordinate_scaling"],
    }
    representation_schema_path = upstream / "opaque_representation_schema.json"
    solution_schema_path = upstream / "opaque_solution_schema.json"
    representation_schema_path.write_text(
        json.dumps(representation_schema), encoding="utf-8"
    )
    solution_schema["representation_schema_sha256"] = sha256_file(
        representation_schema_path
    )
    solution_schema_path.write_text(json.dumps(solution_schema), encoding="utf-8")
    selection_path = upstream / "cluster_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "criterion": "bic",
                "selection_rule": "minimum",
                "direction": "minimize",
                "selected_n_clusters": n_clusters,
                "candidates": [
                    {"n_clusters": max(1, n_clusters - 1), "criterion_value": 200.0},
                    {"n_clusters": n_clusters, "criterion_value": 123.0},
                    {"n_clusters": n_clusters + 1, "criterion_value": 180.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved_inputs: dict[str, object] = {
        "inputs": {
            "artifact:trajectory_representation": binding(
                run_dir,
                representation_path,
                evidence_id="step_owned_representation_12345678",
            ),
            "artifact:candidate_cluster_assignments": binding(
                run_dir,
                assignment_path,
                evidence_id="step_owned_assignments_23456789",
            ),
            "manifest:trajectory_representation_schema": binding(
                run_dir,
                representation_schema_path,
                evidence_id="step_owned_representation_schema_34567890",
            ),
            "manifest:cluster_selection": binding(
                run_dir,
                selection_path,
                evidence_id="log_opaque_selection_ef56ab78",
            ),
            "manifest:candidate_cluster_solution_schema": binding(
                run_dir,
                solution_schema_path,
                evidence_id="log_opaque_solution_schema_de45fa67",
            ),
        }
    }
    return resolved_inputs, representation, assignments
