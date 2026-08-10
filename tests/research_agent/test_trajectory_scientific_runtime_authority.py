from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.execution.runners.trajectory_scientific_candidate_executor import (
    run_trajectory_scientific_candidate_selection,
)
from easyicu.research_agent.execution.runners.trajectory_scientific_representation_executor import (
    run_trajectory_scientific_representation,
)
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
    run_trajectory_stability,
)
from easyicu.research_agent.schema import TrajectoryStabilitySpec
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    build_trajectory_scientific_runtime_authority,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(run_dir: Path, path: Path, evidence_id: str) -> dict[str, str]:
    return {
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": _sha256(path),
        "evidence_id": evidence_id,
    }


def _authority():
    columns = (
        "sofa2_resp__h0_12",
        "sofa2_resp__h12_24",
        "lact__h0_12",
        "lact__h12_24",
    )
    spec = TrajectoryStabilitySpec(
        n_resamples=2,
        sample_fraction=0.75,
        base_seed=1729,
        minimum_successful_resamples=2,
        refit_max_iter=500,
        refit_tolerance=1e-5,
        refit_regularization=1e-6,
        minimum_mean_stability=0.0,
        decision_mode="minimum_mean_threshold",
    )
    return build_trajectory_scientific_runtime_authority(
        {
            "schema_version": "easyicu.trajectory_scientific_runtime_authority/1",
            "protocol_content_sha256": "1" * 64,
            "coordinate_concepts": ["sofa2_resp", "lact"],
            "descriptive_only_concepts": ["sofa2"],
            "window_start_hours": 0,
            "window_end_hours": 24,
            "grid_width_hours": 12,
            "aggregation": "max",
            "representation_columns": list(columns),
            "minimum_available_windows": 2,
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
            "model_family": "latent_class_diagonal_gaussian_mixture",
            "fit_method": "observed_data_em_diagonal_gaussian_mixture",
            "covariance_type": "diag",
            "candidate_cluster_counts": [2, 3, 4],
            "selection_criterion": "bic",
            "selection_rule": "minimum",
            "candidate_fit_base_seed": 1729,
            "candidate_fit_max_iter": 500,
            "candidate_fit_tolerance": 1e-5,
            "candidate_fit_regularization": 1e-6,
            "bic_sample_size": "frozen_population_rows",
            "bic_parameter_count": (
                "mixture_weights_k_minus_1_plus_2_k_per_coordinate"
            ),
            "bic_tie_break": "smaller_k",
            "upper_boundary_action": "fail_closed_if_selected_at_upper_boundary",
            "upper_boundary_reason_code": "H3_NO_INTERIOR_BIC_OPTIMUM",
            "minimum_cluster_fraction": 0.05,
            "minimum_cluster_fraction_reason_code": (
                "H3_MINIMUM_CLUSTER_FRACTION_NOT_MET"
            ),
            "stability_spec": spec.model_dump(mode="json"),
        }
    )


def test_signed_representation_excludes_owner_unavailable_zero(tmp_path: Path) -> None:
    authority = _authority()
    rows = []
    for stay_id in range(1, 9):
        for charttime, resp, lact in ((0.0, 0.0, 1.0), (12.0, 2.0, 2.0)):
            rows.extend(
                [
                    {
                        "stay_id": stay_id,
                        "charttime": charttime,
                        "concept": "sofa2_resp",
                        "value_num": resp,
                        "value_str": str(resp),
                        "evidence_state": "direct_observed",
                        "owner_observed": 1,
                        "owner_available": 1,
                    },
                    {
                        "stay_id": stay_id,
                        "charttime": charttime,
                        "concept": "lact",
                        "value_num": lact,
                        "value_str": str(lact),
                        "evidence_state": "direct_observed",
                        "owner_observed": 1,
                        "owner_available": 1,
                    },
                ]
            )
        rows.append(
            {
                "stay_id": stay_id,
                "charttime": 6.0,
                "concept": "sofa2_resp",
                "value_num": 99.0,
                "value_str": "99",
                "evidence_state": "unavailable",
                "owner_observed": 0,
                "owner_available": 0,
            }
        )
    trajectory_path = tmp_path / "trajectory.parquet"
    pd.DataFrame(rows).to_parquet(trajectory_path, index=False)
    out_dir = tmp_path / "representation"

    summary = run_trajectory_scientific_representation(
        authority=authority,
        runtime_projection_sha256="2" * 64,
        trajectory_path=trajectory_path,
        out_dir=out_dir,
    )

    assert summary["status"] == "ok"
    representation = pd.read_parquet(out_dir / "trajectory_representation.parquet")
    assert representation["sofa2_resp__h0_12"].max() == 0.0
    schema = json.loads(
        (out_dir / "trajectory_representation_schema.json").read_text("utf-8")
    )
    authority.validate_representation_schema(schema)


def test_signed_candidate_and_stability_share_one_scaling_and_selection_contract(
    tmp_path: Path,
) -> None:
    authority = _authority()
    run_dir = tmp_path
    upstream = run_dir / "upstream"
    upstream.mkdir()
    rng = np.random.default_rng(44)
    labels = np.repeat([0, 1, 2], 40)
    centers = np.asarray(
        [
            [-6.0, -4.0, -2.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [6.0, 4.0, 2.0, 1.0],
        ]
    )
    matrix = centers[labels] + rng.normal(0.0, 0.2, size=(len(labels), 4))
    representation = pd.DataFrame(
        matrix, columns=list(authority.representation_columns)
    )
    representation.insert(0, "stay_id", np.arange(1, len(labels) + 1))
    representation_path = upstream / "trajectory_representation.parquet"
    representation.to_parquet(representation_path, index=False)
    schema = {
        "schema_version": "easyicu.trajectory_representation_schema/2",
        "id_column": "stay_id",
        "observation_family": list(authority.coordinate_concepts),
        "observation_columns": list(authority.representation_columns),
        "min_observed_windows": authority.minimum_available_windows,
        "profile_columns": list(authority.representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "signed_runtime_scientific_projection",
        "source_window_contract": {
            "start_hours": 0,
            "end_hours": 24,
            "grid_width_hours": 12,
            "aggregation": "max",
        },
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": authority.scaling_payload,
        "evidence_state_policy": authority.evidence_payload,
        "representation_columns": list(authority.representation_columns),
        "frozen_population_n": len(representation),
        "representation_sha256": _sha256(representation_path),
        "scientific_runtime_authority": {
            "schema_version": authority.schema_version,
            "protocol_content_sha256": authority.protocol_content_sha256,
            "execution_contract_sha256": authority.execution_contract_sha256,
        },
        "runtime_projection_sha256": "2" * 64,
    }
    schema_path = upstream / "trajectory_representation_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    candidate_inputs = {
        "inputs": {
            "artifact:trajectory_representation": _binding(
                run_dir, representation_path, "signed-representation"
            ),
            "manifest:trajectory_representation_schema": _binding(
                run_dir, schema_path, "signed-representation-schema"
            ),
        }
    }
    candidate_out = run_dir / "candidate"

    candidate_summary = run_trajectory_scientific_candidate_selection(
        authority=authority,
        runtime_projection_sha256="2" * 64,
        out_dir=candidate_out,
        run_dir=run_dir,
        resolved_inputs=candidate_inputs,
    )

    assert candidate_summary["status"] == "ok", candidate_summary
    assert candidate_summary["n_clusters"] == 3
    stability_inputs = {
        "inputs": {
            "artifact:trajectory_representation": _binding(
                run_dir, representation_path, "signed-representation"
            ),
            "manifest:trajectory_representation_schema": _binding(
                run_dir, schema_path, "signed-representation-schema"
            ),
            "artifact:candidate_cluster_assignments": _binding(
                run_dir,
                candidate_out / "candidate_cluster_assignments.csv",
                "signed-candidate-assignments",
            ),
            "manifest:cluster_selection": _binding(
                run_dir,
                candidate_out / "cluster_selection.json",
                "signed-cluster-selection",
            ),
            "manifest:candidate_cluster_solution_schema": _binding(
                run_dir,
                candidate_out / "candidate_cluster_solution_schema.json",
                "signed-candidate-schema",
            ),
        }
    }
    stability_out = run_dir / "stability"

    stability_summary = run_trajectory_stability(
        spec=authority.stability_spec,
        out_dir=stability_out,
        run_dir=run_dir,
        resolved_inputs=stability_inputs,
        scientific_runtime_authority=authority,
        runtime_projection_sha256="2" * 64,
    )

    assert stability_summary["status"] == "ok", stability_summary
    assert (
        stability_summary["coordinate_scaling_sha256"]
        == (candidate_summary["coordinate_scaling_sha256"])
    )

    selection_path = candidate_out / "cluster_selection.json"
    selection = json.loads(selection_path.read_text("utf-8"))
    selection["candidate_range_boundary_rule"] = "allow_upper_boundary"
    selection["candidate_range_boundary_reason_code"] = None
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    stability_inputs["inputs"]["manifest:cluster_selection"] = _binding(
        run_dir, selection_path, "tampered-cluster-selection"
    )

    tampered = run_trajectory_stability(
        spec=authority.stability_spec,
        out_dir=run_dir / "tampered_stability",
        run_dir=run_dir,
        resolved_inputs=stability_inputs,
        scientific_runtime_authority=authority,
        runtime_projection_sha256="2" * 64,
    )

    assert tampered["status"] == "failed_closed"
    assert tampered["reason_code"] == "TRAJECTORY_STABILITY_CONTRACT_INVALID"
    assert "signed authority" in " ".join(tampered["errors"])
