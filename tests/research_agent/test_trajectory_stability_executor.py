from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from easyicu.research_agent.schema import TrajectoryStabilitySpec
from easyicu.research_agent.trajectory_stability_executor import (
    run_trajectory_stability,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(run_dir: Path, path: Path, *, evidence_id: str) -> dict[str, str]:
    return {
        "evidence_id": evidence_id,
        "relative_path": str(path.relative_to(run_dir)),
        "sha256": _sha256(path),
    }


def _spec(
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


def _write_upstream_bundle(
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
    matrix = centers[labels] + rng.normal(0.0, 0.18, size=(len(labels), len(centers[0])))
    # Exercise the observed-data implementation without creating all-missing rows.
    matrix[np.arange(len(labels)) % 17 == 0, -1] = np.nan

    identifiers = [f"opaque-unit-{n_clusters}-{index:04d}" for index in range(len(labels))]
    representation = pd.DataFrame(matrix, columns=list(representation_columns))
    representation.insert(0, id_column, identifiers)
    reference_labels = [f"group::{100 + int(value)}" for value in labels]
    assignments = pd.DataFrame(
        {id_column: identifiers, assignment_column: reference_labels}
    ).sample(frac=1.0, random_state=91).reset_index(drop=True)

    upstream = run_dir / "upstream"
    upstream.mkdir(parents=True)
    representation_path = upstream / "opaque_representation.parquet"
    assignment_path = upstream / "opaque_candidate_labels.csv"
    representation.to_parquet(representation_path, index=False)
    assignments.to_csv(assignment_path, index=False)

    representation_evidence_id = "table_opaque_representation_ab12cd34"
    representation_schema_evidence_id = "log_opaque_representation_schema_bc23de45"
    assignment_evidence_id = "table_opaque_candidate_labels_cd34ef56"
    representation_schema = {
        "schema_version": "easyicu.trajectory_representation_schema/1",
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
        "membership_evidence_id": "table_opaque_membership_de45fa67",
        "membership_sha256": "a" * 64,
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "representation_evidence_id": representation_evidence_id,
        "representation_sha256": _sha256(representation_path),
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
        "candidate_models_evidence_id": "model_opaque_candidates_de45fa67",
        "cluster_selection_evidence_id": "log_opaque_selection_ef56ab78",
        "criterion": "bic",
        "selection_rule": "minimum",
        "direction": "minimize",
        "selected_criterion_value": 123.0,
        "representation_schema_evidence_id": representation_schema_evidence_id,
        "candidate_assignments_evidence_id": assignment_evidence_id,
    }
    representation_schema_path = upstream / "opaque_representation_schema.json"
    solution_schema_path = upstream / "opaque_solution_schema.json"
    representation_schema_path.write_text(
        json.dumps(representation_schema), encoding="utf-8"
    )
    solution_schema_path.write_text(json.dumps(solution_schema), encoding="utf-8")

    resolved_inputs: dict[str, object] = {
        "inputs": {
            "artifact:trajectory_representation": _binding(
                run_dir,
                representation_path,
                evidence_id=representation_evidence_id,
            ),
            "artifact:candidate_cluster_assignments": _binding(
                run_dir,
                assignment_path,
                evidence_id=assignment_evidence_id,
            ),
            "manifest:trajectory_representation_schema": _binding(
                run_dir,
                representation_schema_path,
                evidence_id=representation_schema_evidence_id,
            ),
            "manifest:candidate_cluster_solution_schema": _binding(
                run_dir,
                solution_schema_path,
                evidence_id="log_opaque_solution_schema_de45fa67",
            ),
        }
    }
    return resolved_inputs, representation, assignments


def _sample_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(str(value).strip() for value in values.tolist()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("n_clusters", "id_column", "coordinates", "assignment_column"),
    [
        (2, "opaque_subject_key", ("axis_q", "axis_r", "axis_s"), "partition_x"),
        (
            3,
            "anonymous_record_token",
            ("coordinate_17", "coordinate_29", "coordinate_43", "coordinate_71"),
            "chosen_partition",
        ),
    ],
)
def test_executor_is_case_neutral_and_replayable(
    tmp_path: Path,
    n_clusters: int,
    id_column: str,
    coordinates: tuple[str, ...],
    assignment_column: str,
) -> None:
    resolved, representation, candidate_assignments = _write_upstream_bundle(
        tmp_path,
        n_clusters=n_clusters,
        id_column=id_column,
        representation_columns=coordinates,
        assignment_column=assignment_column,
    )
    out_dir = tmp_path / "step_outputs"
    spec = _spec()

    summary = run_trajectory_stability(
        spec=spec,
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "ok", summary
    assert summary["selected_n_clusters"] == n_clusters
    assert summary["stability_threshold_passed"] is None
    assert (
        summary["freeze_status"]
        == "candidate_labels_preserved_report_only_no_stability_decision"
    )
    assert "stable" not in summary

    final_assignments = pd.read_csv(out_dir / "cluster_assignments.csv")
    expected_by_id = candidate_assignments.set_index(id_column)[assignment_column]
    observed_by_id = final_assignments.set_index(id_column)["cluster"]
    assert observed_by_id.to_dict() == expected_by_id.to_dict()
    assert final_assignments[id_column].tolist() == representation[id_column].tolist()

    resolved_spec = json.loads(
        (out_dir / "cluster_stability_spec.json").read_text(encoding="utf-8")
    )
    assert resolved_spec["selected_n_clusters"] == n_clusters
    assert resolved_spec["representation_columns"] == list(coordinates)
    assert resolved_spec["bound_sample_n"] == math.floor(
        len(representation) * float(spec.sample_fraction)
    )
    expected_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(spec.base_seed).spawn(spec.n_resamples)
    ]
    assert resolved_spec["derived_seeds"] == expected_seeds

    stability = pd.read_csv(out_dir / "cluster_stability.csv")
    row_assignments = pd.read_csv(out_dir / "cluster_stability_assignments.csv")
    assert stability["seed"].astype(int).tolist() == expected_seeds
    for row in stability.to_dict(orient="records"):
        replay = row_assignments.loc[
            row_assignments["resample_id"] == row["resample_id"]
        ]
        assert len(replay) == int(row["sample_n"]) == int(row["n_overlap"])
        assert row["sample_id_hash"] == _sample_hash(replay[id_column])
        replayed_ari = adjusted_rand_score(
            replay["reference_cluster"], replay["resampled_cluster"]
        )
        assert float(row["adjusted_rand_index"]) == pytest.approx(
            replayed_ari, rel=1e-12, abs=1e-12
        )

    assert len(row_assignments) == int(stability["sample_n"].sum())
    replay_counts = (
        row_assignments.assign(
            assignment_agreement=(
                row_assignments["reference_cluster"].astype(str)
                == row_assignments["resampled_cluster"].astype(str)
            ).astype(int)
        )
        .groupby(id_column, sort=False)
        .agg(
            stability_inclusion_n=("resample_id", "nunique"),
            assignment_agreement_n=("assignment_agreement", "sum"),
        )
    )
    provenance = pd.read_csv(out_dir / "cluster_assignment_provenance.csv")
    provenance_by_id = provenance.set_index(id_column)
    for identifier in representation[id_column]:
        expected_inclusion = (
            int(replay_counts.loc[identifier, "stability_inclusion_n"])
            if identifier in replay_counts.index
            else 0
        )
        expected_agreement = (
            int(replay_counts.loc[identifier, "assignment_agreement_n"])
            if identifier in replay_counts.index
            else 0
        )
        observed = provenance_by_id.loc[identifier]
        assert int(observed["stability_inclusion_n"]) == expected_inclusion
        assert int(observed["assignment_agreement_n"]) == expected_agreement
        assert float(observed["stability_inclusion_fraction"]) == pytest.approx(
            expected_inclusion / len(stability)
        )
        expected_fraction = (
            expected_agreement / expected_inclusion if expected_inclusion else math.nan
        )
        if math.isnan(expected_fraction):
            assert pd.isna(observed["assignment_agreement_fraction"])
        else:
            assert float(observed["assignment_agreement_fraction"]) == pytest.approx(
                expected_fraction
            )
    assert not (out_dir / ".cluster_stability_assignments.pending.csv").exists()


@pytest.mark.parametrize(
    "violation",
    [
        "unexpected_outcome",
        "binding_digest_mismatch",
        "schema_digest_mismatch",
        "coordinate_order_mismatch",
        "schema_version_mismatch",
        "fractional_cluster_count",
        "missing_policy_field",
        "min_windows_out_of_bounds",
    ],
)
def test_executor_fails_closed_on_untrusted_input_binding(
    tmp_path: Path,
    violation: str,
) -> None:
    resolved, _representation, _assignments = _write_upstream_bundle(
        tmp_path,
        n_clusters=2,
        id_column="neutral_id",
        representation_columns=("feature_alpha", "feature_beta"),
        assignment_column="reference_partition",
    )
    inputs = resolved["inputs"]
    assert isinstance(inputs, dict)
    if violation == "unexpected_outcome":
        inputs["table:outcome_by_cluster"] = {
            "evidence_id": "table_forbidden_outcome_12345678",
            "relative_path": "upstream/forbidden_outcome.csv",
            "sha256": "0" * 64,
        }
    elif violation == "binding_digest_mismatch":
        representation_binding = inputs["artifact:trajectory_representation"]
        assert isinstance(representation_binding, dict)
        representation_binding["sha256"] = "0" * 64
    elif violation in {
        "schema_digest_mismatch",
        "missing_policy_field",
        "min_windows_out_of_bounds",
    }:
        schema_path = tmp_path / "upstream" / "opaque_representation_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if violation == "schema_digest_mismatch":
            schema["representation_sha256"] = "0" * 64
        elif violation == "missing_policy_field":
            schema.pop("profile_summary_statistic")
        else:
            schema["min_observed_windows"] = len(schema["observation_columns"]) + 1
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        schema_binding = inputs["manifest:trajectory_representation_schema"]
        assert isinstance(schema_binding, dict)
        schema_binding["sha256"] = _sha256(schema_path)
    else:
        schema_path = tmp_path / "upstream" / "opaque_solution_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if violation == "coordinate_order_mismatch":
            schema["representation_columns"] = list(
                reversed(schema["representation_columns"])
            )
        elif violation == "schema_version_mismatch":
            schema["schema_version"] = "easyicu.candidate_cluster_solution_schema/999"
        else:
            schema["selected_n_clusters"] = 2.9
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        schema_binding = inputs["manifest:candidate_cluster_solution_schema"]
        assert isinstance(schema_binding, dict)
        schema_binding["sha256"] = _sha256(schema_path)

    summary = run_trajectory_stability(
        spec=_spec(),
        out_dir=tmp_path / "step_outputs",
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "failed_closed"
    assert summary["freeze_status"] == "not_frozen"
    error_text = " ".join(summary["errors"]).lower()
    if violation == "unexpected_outcome":
        assert "undeclared typed bindings" in error_text
        assert "outcome_by_cluster" in error_text
    elif violation == "binding_digest_mismatch":
        assert "digest mismatch" in error_text
    elif violation == "schema_digest_mismatch":
        assert "schema digest" in error_text
    elif violation == "coordinate_order_mismatch":
        assert "coordinate order" in error_text
    elif violation == "schema_version_mismatch":
        assert "schema version" in error_text
    elif violation == "fractional_cluster_count":
        assert "selected_n_clusters must be an integer" in error_text
    elif violation == "missing_policy_field":
        assert "profile_summary_statistic" in error_text
    else:
        assert "min_observed_windows exceeds" in error_text


def test_executor_fails_closed_when_sampled_reference_has_one_cluster(
    tmp_path: Path,
) -> None:
    resolved, representation, assignments = _write_upstream_bundle(
        tmp_path,
        n_clusters=2,
        id_column="neutral_id",
        representation_columns=("feature_alpha", "feature_beta"),
        assignment_column="reference_partition",
    )
    assignments["reference_partition"] = "group::100"
    assignments.loc[
        assignments["neutral_id"] == representation["neutral_id"].iloc[-1],
        "reference_partition",
    ] = "group::200"
    assignment_path = tmp_path / "upstream" / "opaque_candidate_labels.csv"
    assignments.to_csv(assignment_path, index=False)
    assignment_binding = resolved["inputs"][
        "artifact:candidate_cluster_assignments"
    ]
    assignment_binding["sha256"] = _sha256(assignment_path)

    out_dir = tmp_path / "step_outputs"
    summary = run_trajectory_stability(
        spec=_spec().model_copy(update={"base_seed": 9}),
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "failed_closed"
    failures = json.loads(
        (tmp_path / "step_outputs" / "cluster_stability_refit_failures.json").read_text(
            encoding="utf-8"
        )
    )["failures"]
    assert any("fewer than two clusters" in row["error"] for row in failures)
    assert not (out_dir / ".cluster_stability_assignments.pending.csv").exists()
    assert pd.read_csv(out_dir / "cluster_stability_assignments.csv").empty


def test_threshold_failure_fails_closed_without_changing_selected_solution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved, representation, candidate_assignments = _write_upstream_bundle(
        tmp_path,
        n_clusters=3,
        id_column="uninterpreted_id",
        representation_columns=("dimension_u", "dimension_v", "dimension_w"),
        assignment_column="frozen_label",
    )

    def deliberately_unstable_refit(
        x: np.ndarray,
        *,
        n_components: int,
        seed: int,
        max_iter: int,
        tolerance: float,
        regularization: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del seed, max_iter, tolerance, regularization
        labels = np.arange(len(x), dtype=int) % n_components
        return labels, {
            "converged": True,
            "n_iter": 1,
            "final_log_likelihood": 0.0,
            "parameter_sha256": "synthetic-threshold-probe",
        }

    monkeypatch.setattr(
        "easyicu.research_agent.trajectory_stability_executor._fit_observed_data_diag_gmm",
        deliberately_unstable_refit,
    )
    out_dir = tmp_path / "step_outputs"
    summary = run_trajectory_stability(
        spec=_spec(
            decision_mode="minimum_mean_threshold",
            minimum_mean_stability=0.99,
        ),
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "failed_closed"
    assert summary["stability_threshold_passed"] is False
    assert summary["freeze_status"] == "not_frozen_stability_threshold_failed"
    assert summary["selected_n_clusters"] == 3
    assert "selected k was not changed" in " ".join(summary["errors"])

    final_assignments = pd.read_csv(out_dir / "cluster_assignments.csv")
    assert final_assignments["uninterpreted_id"].tolist() == representation[
        "uninterpreted_id"
    ].tolist()
    assert final_assignments.set_index("uninterpreted_id")["cluster"].to_dict() == (
        candidate_assignments.set_index("uninterpreted_id")["frozen_label"].to_dict()
    )


def test_invalid_spec_still_writes_failed_closed_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "step_outputs"

    summary = run_trajectory_stability(
        spec={"n_resamples": 1},
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs={"inputs": {}},
    )

    assert summary["status"] == "failed_closed"
    assert summary["freeze_status"] == "not_frozen"
    assert "validation" in " ".join(summary["errors"]).lower()
    assert json.loads((out_dir / "step_summary.json").read_text()) == summary
