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
from easyicu.research_agent.execution.runners.trajectory_stability_executor import (
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
        "representation_schema_sha256": "pending",
        "candidate_assignments_sha256": _sha256(assignment_path),
    }
    representation_schema_path = upstream / "opaque_representation_schema.json"
    solution_schema_path = upstream / "opaque_solution_schema.json"
    representation_schema_path.write_text(
        json.dumps(representation_schema), encoding="utf-8"
    )
    solution_schema["representation_schema_sha256"] = _sha256(
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
            "artifact:trajectory_representation": _binding(
                run_dir,
                representation_path,
                evidence_id="step_owned_representation_12345678",
            ),
            "artifact:candidate_cluster_assignments": _binding(
                run_dir,
                assignment_path,
                evidence_id="step_owned_assignments_23456789",
            ),
            "manifest:trajectory_representation_schema": _binding(
                run_dir,
                representation_schema_path,
                evidence_id="step_owned_representation_schema_34567890",
            ),
            "manifest:cluster_selection": _binding(
                run_dir,
                selection_path,
                evidence_id="log_opaque_selection_ef56ab78",
            ),
            "manifest:candidate_cluster_solution_schema": _binding(
                run_dir,
                solution_schema_path,
                evidence_id="log_opaque_solution_schema_de45fa67",
            ),
        }
    }
    return resolved_inputs, representation, assignments


def test_executor_replays_legacy_exact_evidence_links_without_digest_links(
    tmp_path: Path,
) -> None:
    resolved, _representation, _assignments = _write_upstream_bundle(
        tmp_path,
        n_clusters=2,
        id_column="legacy_id",
        representation_columns=("feature_alpha", "feature_beta"),
        assignment_column="legacy_partition",
    )
    inputs = resolved["inputs"]
    assert isinstance(inputs, dict)
    representation_path = tmp_path / "upstream" / "opaque_representation.parquet"
    assignments_path = tmp_path / "upstream" / "opaque_candidate_labels.csv"
    representation_schema_path = (
        tmp_path / "upstream" / "opaque_representation_schema.json"
    )
    solution_schema_path = tmp_path / "upstream" / "opaque_solution_schema.json"

    representation_schema = json.loads(
        representation_schema_path.read_text(encoding="utf-8")
    )
    representation_schema.pop("representation_sha256")
    representation_schema["representation_evidence_id"] = "legacy-representation"
    representation_schema_path.write_text(
        json.dumps(representation_schema), encoding="utf-8"
    )
    solution_schema = json.loads(solution_schema_path.read_text(encoding="utf-8"))
    solution_schema.pop("representation_schema_sha256")
    solution_schema.pop("candidate_assignments_sha256")
    solution_schema["representation_schema_evidence_id"] = "legacy-schema"
    solution_schema["candidate_assignments_evidence_id"] = "legacy-assignments"
    solution_schema_path.write_text(json.dumps(solution_schema), encoding="utf-8")

    legacy_bindings = {
        "artifact:trajectory_representation": (
            representation_path,
            "legacy-representation",
        ),
        "artifact:candidate_cluster_assignments": (
            assignments_path,
            "legacy-assignments",
        ),
        "manifest:trajectory_representation_schema": (
            representation_schema_path,
            "legacy-schema",
        ),
        "manifest:candidate_cluster_solution_schema": (
            solution_schema_path,
            "legacy-solution-schema",
        ),
    }
    for input_key, (path, evidence_id) in legacy_bindings.items():
        binding = inputs[input_key]
        assert isinstance(binding, dict)
        binding["evidence_id"] = evidence_id
        binding["sha256"] = _sha256(path)

    summary = run_trajectory_stability(
        spec=_spec(),
        out_dir=tmp_path / "step_outputs",
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "ok", summary


def _sample_hash(values: pd.Series) -> str:
    payload = "\n".join(sorted(str(value).strip() for value in values.tolist()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rewrite_solution_digest_link(
    resolved: dict[str, object],
    run_dir: Path,
    *,
    field: str,
    target_path: Path,
) -> None:
    inputs = resolved["inputs"]
    assert isinstance(inputs, dict)
    solution_path = run_dir / "upstream" / "opaque_solution_schema.json"
    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    solution[field] = _sha256(target_path)
    solution_path.write_text(json.dumps(solution), encoding="utf-8")
    solution_binding = inputs["manifest:candidate_cluster_solution_schema"]
    assert isinstance(solution_binding, dict)
    solution_binding["sha256"] = _sha256(solution_path)


def _assert_complete_input_receipts(
    *,
    summary: dict[str, object],
    resolved_inputs: dict[str, object],
    representation_n: int,
    assignment_n: int,
) -> None:
    raw_inputs = resolved_inputs["inputs"]
    assert isinstance(raw_inputs, dict)
    raw_receipts = summary["input_bindings"]
    assert isinstance(raw_receipts, list)
    receipts = {receipt["input_key"]: receipt for receipt in raw_receipts}
    assert set(receipts) == set(raw_inputs)
    for input_key, binding in raw_inputs.items():
        assert isinstance(binding, dict)
        assert receipts[input_key]["loaded"] is True
        assert receipts[input_key]["evidence_id"] == binding["evidence_id"]
        assert receipts[input_key]["sha256"] == binding["sha256"]
    assert receipts["artifact:trajectory_representation"]["row_count"] == (
        representation_n
    )
    assert receipts["artifact:candidate_cluster_assignments"]["row_count"] == (
        assignment_n
    )
    assert "row_count" not in receipts["manifest:trajectory_representation_schema"]
    assert "row_count" not in receipts["manifest:cluster_selection"]
    assert "row_count" not in receipts["manifest:candidate_cluster_solution_schema"]


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
    _assert_complete_input_receipts(
        summary=summary,
        resolved_inputs=resolved,
        representation_n=len(representation),
        assignment_n=len(candidate_assignments),
    )

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
        "candidate_assignment_schema_digest_mismatch",
        "representation_schema_link_digest_mismatch",
        "missing_cluster_selection",
        "cluster_selection_k_mismatch",
        "cluster_selection_criterion_mismatch",
        "cluster_selection_value_mismatch",
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
    elif violation == "missing_cluster_selection":
        inputs.pop("manifest:cluster_selection")
    elif violation.startswith("cluster_selection_"):
        selection_path = tmp_path / "upstream" / "cluster_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if violation == "cluster_selection_k_mismatch":
            selection["selected_n_clusters"] = 3
        elif violation == "cluster_selection_criterion_mismatch":
            selection["criterion"] = "aic"
        else:
            selection["candidates"][1]["criterion_value"] = 124.0
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        selection_binding = inputs["manifest:cluster_selection"]
        assert isinstance(selection_binding, dict)
        selection_binding["sha256"] = _sha256(selection_path)
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
        _rewrite_solution_digest_link(
            resolved,
            tmp_path,
            field="representation_schema_sha256",
            target_path=schema_path,
        )
    else:
        schema_path = tmp_path / "upstream" / "opaque_solution_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if violation == "coordinate_order_mismatch":
            schema["representation_columns"] = list(
                reversed(schema["representation_columns"])
            )
        elif violation == "schema_version_mismatch":
            schema["schema_version"] = "easyicu.candidate_cluster_solution_schema/999"
        elif violation == "candidate_assignment_schema_digest_mismatch":
            schema["candidate_assignments_sha256"] = "0" * 64
        elif violation == "representation_schema_link_digest_mismatch":
            schema["representation_schema_sha256"] = "0" * 64
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
    elif violation == "missing_cluster_selection":
        assert "required typed bindings are absent" in error_text
        assert "manifest:cluster_selection" in error_text
    elif violation.startswith("cluster_selection_"):
        assert "cluster selection manifest disagrees" in error_text
    elif violation == "schema_digest_mismatch":
        assert "representation_sha256 does not bind" in error_text
    elif violation == "candidate_assignment_schema_digest_mismatch":
        assert "candidate_assignments_sha256 does not bind" in error_text
    elif violation == "representation_schema_link_digest_mismatch":
        assert "representation_schema_sha256 does not bind" in error_text
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
    assignment_binding = resolved["inputs"]["artifact:candidate_cluster_assignments"]
    assignment_binding["sha256"] = _sha256(assignment_path)
    _rewrite_solution_digest_link(
        resolved,
        tmp_path,
        field="candidate_assignments_sha256",
        target_path=assignment_path,
    )

    out_dir = tmp_path / "step_outputs"
    summary = run_trajectory_stability(
        spec=_spec().model_copy(update={"base_seed": 9}),
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=resolved,
    )

    assert summary["status"] == "failed_closed"
    _assert_complete_input_receipts(
        summary=summary,
        resolved_inputs=resolved,
        representation_n=len(representation),
        assignment_n=len(assignments),
    )
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
        "easyicu.research_agent.execution.runners.trajectory_stability_executor._fit_observed_data_diag_gmm",
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
    assert (
        final_assignments["uninterpreted_id"].tolist()
        == representation["uninterpreted_id"].tolist()
    )
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
