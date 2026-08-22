"""Compute an agent-specified trajectory-cluster stability design.

This is a supporting executor, not a clustering planner.  It is eligible only
for a dedicated stability/freeze step carrying a complete
``TrajectoryStabilitySpec``.  The frozen population, coordinate order, selected
model family, selected cluster count, missing-data likelihood, and reference
assignments come from digest-verified typed upstream manifests.  No variable,
method, cluster count, resampling design, or acceptance threshold is inferred
from prose, filenames, column substrings, or benchmark identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score

from ...schema import (
    AnalysisPlan,
    AnalysisStep,
    ClusterSelectionManifest,
    TrajectoryStabilitySpec,
)
from ...trajectory.plan_contract import (
    OBSERVED_DATA_DIAG_GMM_FIT_METHOD,
    OBSERVED_DATA_DIAG_GMM_MODEL_FAMILY,
    STABILITY_EXECUTOR_INPUTS,
    STABILITY_EXECUTOR_OUTPUTS,
    STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS,
    TRAJECTORY_CANDIDATE_SOLUTION_SCHEMA_VERSION,
    TRAJECTORY_REPRESENTATION_SCHEMA_VERSION,
    TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD,
    TRAJECTORY_STABILITY_METHOD_HEAD,
    trajectory_step_roles,
)
from ...trajectory.scientific_runtime_authority import (
    TrajectoryScientificRuntimeAuthority,
    load_trajectory_scientific_runtime_authority,
)

__all__ = [
    "STABILITY_EXECUTOR_INPUTS",
    "STABILITY_EXECUTOR_OUTPUTS",
    "run_trajectory_stability",
    "trajectory_stability_executor_code",
    "trajectory_stability_executor_owns_step",
    "validate_trajectory_stability_schema_pair",
    "validate_trajectory_stability_upstream",
]


_SUPPORTED_MODEL_FAMILY = OBSERVED_DATA_DIAG_GMM_MODEL_FAMILY
_SUPPORTED_FIT_METHOD = OBSERVED_DATA_DIAG_GMM_FIT_METHOD
_SUPPORTED_COVARIANCE = "diag"
_SUPPORTED_REPRESENTATION_SCHEMA = TRAJECTORY_REPRESENTATION_SCHEMA_VERSION
_SUPPORTED_SOLUTION_SCHEMA = TRAJECTORY_CANDIDATE_SOLUTION_SCHEMA_VERSION
_NATIVE_MATH_THREAD_ENV = (
    "VECLIB_MAXIMUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class _NumericalRefitFailure(ValueError):
    pass


def _step_contract_is_closed(step: AnalysisStep) -> bool:
    """Validate the local calculator contract without granting DAG ownership."""

    if step.trajectory_stability_spec is None:
        return False
    method_head = _normalise(step.method).split("_with_", 1)[0]
    roles = trajectory_step_roles(step)
    inputs = {str(item).strip().lower() for item in step.inputs or []}
    outputs = {str(item).strip().lower() for item in step.expected_outputs or []}
    shared_closed = (
        len(step.inputs or []) == len(STABILITY_EXECUTOR_INPUTS)
        and inputs == STABILITY_EXECUTOR_INPUTS
    )
    if not shared_closed:
        return False
    if method_head == TRAJECTORY_STABILITY_METHOD_HEAD:
        return (
            roles == frozenset({"stability_freeze"})
            and len(step.expected_outputs or []) == len(STABILITY_EXECUTOR_OUTPUTS)
            and outputs == STABILITY_EXECUTOR_OUTPUTS
        )
    return (
        method_head == TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD
        and roles == frozenset({"stability_freeze", "characterization"})
        and len(step.expected_outputs or [])
        == len(STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS)
        and outputs == STABILITY_CHARACTERIZATION_EXECUTOR_OUTPUTS
    )


def trajectory_stability_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
) -> bool:
    """Return whether a complete closed stability contract authorizes execution."""

    if not _step_contract_is_closed(step):
        return False
    candidates = [
        item
        for item in plan.steps
        if "candidate_selection" in trajectory_step_roles(item)
    ]
    stability_owners = [
        item for item in plan.steps if "stability_freeze" in trajectory_step_roles(item)
    ]
    if len(candidates) != 1:
        return False
    if stability_owners != [step] or candidates[0].step_id == step.step_id:
        return False
    order = {item.step_id: index for index, item in enumerate(plan.steps)}
    if order[candidates[0].step_id] >= order[step.step_id]:
        return False
    if "characterization" in trajectory_step_roles(step):
        characterization_owners = [
            item
            for item in plan.steps
            if "characterization" in trajectory_step_roles(item)
        ]
        return characterization_owners == [step]
    return True


def trajectory_stability_executor_code(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    scientific_runtime_authority: (
        TrajectoryScientificRuntimeAuthority | Mapping[str, Any] | None
    ) = None,
    runtime_projection_sha256: str | None = None,
) -> str:
    """Return the short trusted adapter for one planner-owned stability spec."""

    if not trajectory_stability_executor_owns_step(step, plan=plan):
        raise ValueError("step does not satisfy the trajectory stability contract")
    payload = step.trajectory_stability_spec.model_dump(mode="json")
    include_characterization = (
        _normalise(step.method).split("_with_", 1)[0]
        == TRAJECTORY_STABILITY_CHARACTERIZATION_METHOD_HEAD
    )
    authority_payload = (
        load_trajectory_scientific_runtime_authority(
            scientific_runtime_authority
        ).model_dump(mode="json")
        if scientific_runtime_authority is not None
        else None
    )
    return (
        "import json, os\n"
        "os.environ['VECLIB_MAXIMUM_THREADS'] = '1'\n"
        "os.environ['OMP_NUM_THREADS'] = '1'\n"
        "os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
        "os.environ['MKL_NUM_THREADS'] = '1'\n"
        "os.environ['NUMEXPR_NUM_THREADS'] = '1'\n"
        "from pathlib import Path\n"
        "from easyicu.research_agent.execution.runners.trajectory_stability_executor import "
        "run_trajectory_stability\n"
        f"spec = json.loads({json.dumps(json.dumps(payload, sort_keys=True))})\n"
        f"scientific_runtime_authority = json.loads({json.dumps(json.dumps(authority_payload, sort_keys=True))})\n"
        "run_trajectory_stability("
        "spec=spec, out_dir=Path(os.environ['STEP_OUT_DIR']), "
        "run_dir=Path(os.environ['EASYICU_RUN_DIR']), "
        "resolved_inputs=os.environ['EASYICU_RESOLVED_INPUTS_JSON'], "
        "scientific_runtime_authority=scientific_runtime_authority, "
        f"runtime_projection_sha256={runtime_projection_sha256!r}, "
        f"include_characterization={include_characterization!r})\n"
    )


def _normalise(value: Any) -> str:
    return "_".join(
        part
        for part in "".join(
            char.lower() if char.isalnum() else " " for char in str(value or "")
        ).split()
        if part
    )


def _canonical_id(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null:"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{bool(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "null:"
        return f"float:{number:.17g}"
    return f"str:{value}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_resolved_inputs(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        text = str(value)
        path = Path(text)
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else json.loads(text)
        )
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("resolved typed-input manifest must contain an inputs object")
    return inputs


def _binding_path(*, binding: Mapping[str, Any], run_dir: Path) -> Path:
    relative_path = str(binding.get("relative_path") or "").strip()
    if not relative_path:
        raise ValueError("typed input binding lacks relative_path")
    root = run_dir.resolve()
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError("typed input binding escapes the run directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = str(binding.get("sha256") or "").strip().lower()
    if not expected_sha or _sha256(path) != expected_sha:
        raise ValueError(f"typed input digest mismatch for {relative_path}")
    return path


def _read_bound(*, inputs: Mapping[str, Any], key: str, run_dir: Path) -> Any:
    binding = inputs.get(key)
    if not isinstance(binding, Mapping):
        raise ValueError(f"missing typed input binding: {key}")
    path = _binding_path(binding=binding, run_dir=run_dir)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"unsupported typed input format for {key}: {suffix}")


def _binding_provenance(inputs: Mapping[str, Any], key: str) -> dict[str, Any]:
    binding = inputs[key]
    return {
        "evidence_id": binding.get("evidence_id"),
        "sha256": binding.get("sha256"),
        "relative_path": binding.get("relative_path"),
    }


def _loaded_input_receipt(
    *,
    inputs: Mapping[str, Any],
    key: str,
    value: Any,
) -> dict[str, Any]:
    """Return one truthful receipt for an exact input already loaded above."""

    binding = inputs.get(key)
    if not isinstance(binding, Mapping):
        raise ValueError(f"missing typed input binding: {key}")
    evidence_id = str(binding.get("evidence_id") or "").strip()
    sha256 = str(binding.get("sha256") or "").strip()
    if not evidence_id or not sha256:
        raise ValueError(f"typed input binding lacks identity metadata: {key}")
    receipt: dict[str, Any] = {
        "input_key": key,
        "evidence_id": evidence_id,
        "sha256": sha256,
        "loaded": True,
    }
    if isinstance(value, pd.DataFrame):
        receipt["row_count"] = int(len(value))
    return receipt


def _require_legacy_evidence_link(
    *,
    payload: Mapping[str, Any],
    field: str,
    inputs: Mapping[str, Any],
    input_key: str,
) -> None:
    expected = str(payload.get(field) or "").strip()
    binding = inputs.get(input_key)
    actual = (
        str(binding.get("evidence_id") or "").strip()
        if isinstance(binding, Mapping)
        else ""
    )
    if not expected or expected != actual:
        raise ValueError(
            f"typed schema provenance mismatch: {field} does not bind {input_key}"
        )


def _require_artifact_digest_link(
    *,
    payload: Mapping[str, Any],
    digest_field: str,
    legacy_evidence_field: str,
    inputs: Mapping[str, Any],
    input_key: str,
) -> None:
    """Bind a typed schema to one artifact by its producer-computable digest.

    Current evidence identifiers are assigned by the host after the producer
    finishes and may therefore be step-owned rather than content-derived.  A
    producer cannot truthfully predict that identifier.  New schemas bind the
    exact artifact bytes by SHA-256 instead.  The evidence-id fallback exists
    only so already-sealed legacy schemas remain replayable when their original
    binding still carries that exact identifier.
    """

    binding = inputs.get(input_key)
    if not isinstance(binding, Mapping):
        raise ValueError(f"missing typed input binding: {input_key}")
    expected_digest = str(payload.get(digest_field) or "").strip().lower()
    actual_digest = str(binding.get("sha256") or "").strip().lower()
    if expected_digest:
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise ValueError(f"{digest_field} must be a SHA-256 digest")
        if expected_digest != actual_digest:
            raise ValueError(
                "typed schema provenance mismatch: "
                f"{digest_field} does not bind {input_key}"
            )
        return
    _require_legacy_evidence_link(
        payload=payload,
        field=legacy_evidence_field,
        inputs=inputs,
        input_key=input_key,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _empty_outputs(out_dir: Path, id_column: str = "id") -> None:
    pd.DataFrame(columns=[id_column, "cluster"]).to_csv(
        out_dir / "cluster_assignments.csv", index=False
    )
    pd.DataFrame(
        columns=[
            "resample_id",
            "n_overlap",
            "adjusted_rand_index",
            "clustering_method",
            "refit_model_id",
            "seed",
            "sampling_method",
            "sample_n",
            "sample_id_hash",
            "selected_n_clusters",
        ]
    ).to_csv(out_dir / "cluster_stability.csv", index=False)
    pd.DataFrame(
        columns=["resample_id", id_column, "reference_cluster", "resampled_cluster"]
    ).to_csv(out_dir / "cluster_stability_assignments.csv", index=False)
    pd.DataFrame(columns=[id_column, "cluster"]).to_csv(
        out_dir / "cluster_assignment_provenance.csv", index=False
    )


def _sample_hash(values: pd.Series) -> str:
    def contract_token(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value)).lower()
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).strip()
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
        return str(value).strip()

    payload = "\n".join(sorted(contract_token(value) for value in values.tolist()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_observed_data_diag_gmm(
    x: np.ndarray,
    *,
    n_components: int,
    seed: int,
    max_iter: int,
    tolerance: float,
    regularization: float,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Fit the versioned observed-data diagonal-GMM refit engine."""

    x = np.asarray(x, dtype=float)
    n_rows, n_features = x.shape
    if n_rows <= n_components:
        raise ValueError("refit sample is not larger than selected_n_clusters")
    observed = np.isfinite(x)
    if not observed.any(axis=1).all():
        raise ValueError("a refit row has no observed representation coordinate")
    if not observed.any(axis=0).all():
        raise ValueError("a refit coordinate has no observed values")

    # Stack the three sufficient-statistic blocks once so each EM phase uses
    # one dense matrix multiplication instead of three small ones.  This is
    # algebraically identical to the separate observed/x/x-squared products,
    # but materially reduces dispatch overhead for large trajectory matrices.
    sufficient_statistics = np.empty((n_rows, 3 * n_features), dtype=float)
    observed_float = sufficient_statistics[:, :n_features]
    observed_float[:] = observed
    x_work = sufficient_statistics[:, n_features : 2 * n_features]
    x_work[:] = np.where(observed, x, 0.0)
    x_squared = sufficient_statistics[:, 2 * n_features :]
    np.square(x_work, out=x_squared)
    counts = observed_float.sum(axis=0)
    global_mean = x_work.sum(axis=0) / counts
    centered = x_work - global_mean
    global_variance = (observed_float * centered * centered).sum(axis=0) / counts
    global_variance = np.maximum(global_variance, regularization)

    rng = np.random.default_rng(seed)
    initial_labels = np.arange(n_rows, dtype=int) % n_components
    rng.shuffle(initial_labels)
    responsibilities = np.zeros((n_rows, n_components), dtype=float)
    responsibilities[np.arange(n_rows), initial_labels] = 1.0

    previous = -np.inf
    converged = False
    for iteration in range(max_iter):
        weighted_statistics = responsibilities.T @ sufficient_statistics
        effective_observed = weighted_statistics[:, :n_features]
        weighted_values = weighted_statistics[:, n_features : 2 * n_features]
        weighted_squares = weighted_statistics[:, 2 * n_features :]
        means = np.broadcast_to(global_mean, (n_components, n_features)).copy()
        np.divide(
            weighted_values,
            effective_observed,
            out=means,
            where=effective_observed > 0,
        )
        second_moments = np.broadcast_to(
            global_variance + global_mean * global_mean,
            (n_components, n_features),
        ).copy()
        np.divide(
            weighted_squares,
            effective_observed,
            out=second_moments,
            where=effective_observed > 0,
        )
        variances = second_moments - means * means
        variances = np.maximum(variances, regularization)
        weights = np.maximum(responsibilities.sum(axis=0), 1e-12)
        weights /= weights.sum()

        inverse_variances = 1.0 / variances
        observed_constant = (
            np.log(2.0 * math.pi)
            + np.log(variances)
            + means * means * inverse_variances
        )
        likelihood_coefficients = np.concatenate(
            (
                observed_constant.T,
                (-2.0 * means * inverse_variances).T,
                inverse_variances.T,
            ),
            axis=0,
        )
        contribution = sufficient_statistics @ likelihood_coefficients
        log_prob = np.log(weights)[None, :] - 0.5 * contribution
        normalizer = logsumexp(log_prob, axis=1)
        likelihood = float(normalizer.sum())
        if not math.isfinite(likelihood):
            raise ValueError("observed-data refit produced non-finite likelihood")
        responsibilities = np.exp(log_prob - normalizer[:, None])
        if iteration > 0 and abs(likelihood - previous) <= tolerance * (
            1.0 + abs(previous)
        ):
            converged = True
            break
        previous = likelihood

    if not converged:
        raise ValueError("observed-data refit did not converge")
    labels = np.argmax(responsibilities, axis=1).astype(int)
    if np.unique(labels).size != n_components:
        raise ValueError("observed-data refit did not realize every selected cluster")
    parameter_digest = hashlib.sha256()
    for array in (weights, means, variances):
        parameter_digest.update(np.ascontiguousarray(array).tobytes())
    return labels, {
        "converged": True,
        "n_iter": iteration + 1,
        "final_log_likelihood": likelihood,
        "parameter_sha256": parameter_digest.hexdigest(),
    }


def _lexicographic_linear_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimize total cost, then the assigned-column vector lexicographically."""

    cost = np.asarray(cost, dtype=np.int64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("label-alignment cost matrix must be square")
    size = cost.shape[0]
    initial_rows, initial_columns = linear_sum_assignment(cost)
    remaining_target = int(cost[initial_rows, initial_columns].sum())
    remaining_columns = list(range(size))
    chosen: list[int] = []
    for row in range(size):
        selected = None
        for column in remaining_columns:
            future_rows = list(range(row + 1, size))
            future_columns = [value for value in remaining_columns if value != column]
            future_cost = 0
            if future_rows:
                submatrix = cost[np.ix_(future_rows, future_columns)]
                sub_rows, sub_columns = linear_sum_assignment(submatrix)
                future_cost = int(submatrix[sub_rows, sub_columns].sum())
            if int(cost[row, column]) + future_cost == remaining_target:
                selected = column
                remaining_target -= int(cost[row, column])
                break
        if selected is None:
            raise ValueError("label-alignment tie-break could not preserve the optimum")
        chosen.append(selected)
        remaining_columns.remove(selected)
    return np.arange(size, dtype=int), np.asarray(chosen, dtype=int)


def _aligned_labels(
    reference: np.ndarray,
    refit: np.ndarray,
    *,
    reference_universe: list[Any],
) -> np.ndarray:
    reference_values = sorted(reference_universe, key=lambda value: str(value))
    refit_values = sorted(pd.unique(refit).tolist(), key=lambda value: str(value))
    if len(reference_values) != len(refit_values):
        raise ValueError("reference and refit label universes differ in size")
    matrix = np.zeros((len(reference_values), len(refit_values)), dtype=int)
    for row, reference_value in enumerate(reference_values):
        for column, refit_value in enumerate(refit_values):
            matrix[row, column] = int(
                np.sum((reference == reference_value) & (refit == refit_value))
            )
    # Maximise overlap first. For equal-overlap assignments, minimize total
    # sorted-label rank distance, then choose the assigned-column vector
    # lexicographically. The multiplier is larger than the maximum total
    # rank-distance penalty, so secondary preferences cannot sacrifice overlap.
    size = max(len(reference_values), len(refit_values))
    maximum_penalty = max(1, size * size)
    penalty = np.fromfunction(
        lambda row, column: np.abs(row - column),
        matrix.shape,
        dtype=int,
    ).astype(np.int64)
    cost = -matrix.astype(np.int64) * (maximum_penalty + 1) + penalty
    rows, columns = _lexicographic_linear_assignment(cost)
    mapping = {
        refit_values[column]: reference_values[row]
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
    }
    if len(mapping) != len(refit_values):
        raise ValueError("refit labels could not be aligned one-to-one")
    return np.asarray([mapping[value] for value in refit], dtype=object)


def _strict_int(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _nonempty_string_list(value: Any, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty list of strings")
    normalized = [item.strip() for item in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} contains duplicate values")
    return normalized


def _validate_representation_policy(schema: Mapping[str, Any]) -> None:
    observation_columns = _nonempty_string_list(
        schema.get("observation_columns"), field="observation_columns"
    )
    _nonempty_string_list(schema.get("profile_columns"), field="profile_columns")
    min_observed_windows = _strict_int(
        schema.get("min_observed_windows"),
        field="min_observed_windows",
        minimum=1,
    )
    if min_observed_windows > len(observation_columns):
        raise ValueError("min_observed_windows exceeds observation_columns")
    if not str(schema.get("observation_family") or "").strip():
        raise ValueError("observation_family is required")
    if schema.get("profile_summary_statistic") not in {"mean", "median"}:
        raise ValueError("profile_summary_statistic must be mean or median")
    if schema.get("time_axis") != "relative_hours":
        raise ValueError("time_axis must be relative_hours")
    if not str(schema.get("anchor") or "").strip():
        raise ValueError("anchor is required")
    if schema.get("anchor_provenance") not in {"task_contract", "agent_declared"}:
        raise ValueError("anchor_provenance is unsupported")
    if not str(schema.get("anchor_source") or "").strip():
        raise ValueError("anchor_source is required")
    trailing_policy = schema.get("trailing_na_policy")
    required_trailing = {
        "zero_imputation": False,
        "eligibility_uses_observed_window_count": True,
        "profile_summaries_ignore_missing": True,
    }
    if not isinstance(trailing_policy, Mapping) or any(
        trailing_policy.get(key) is not expected
        for key, expected in required_trailing.items()
    ):
        raise ValueError("trailing_na_policy does not preserve observed missingness")
    scaling = schema.get("coordinate_scaling")
    required_scaling = {
        "method": "pooled_coordinate_wise_z_score",
        "ddof": 0,
        "observed_value_policy": "direct_or_owner_locf_available",
        "missing_value_policy": "preserve_missing_exclude_from_likelihood",
        "zero_variance_action": "fail_closed",
    }
    if not isinstance(scaling, Mapping) or dict(scaling) != required_scaling:
        raise ValueError("coordinate_scaling does not match the frozen policy")
    evidence = schema.get("evidence_state_policy")
    required_evidence = {
        "direct_observed": "include",
        "owner_locf_available": "include_and_audit",
        "unavailable": "exclude",
        "additional_clustering_stage_imputation": "none",
    }
    if not isinstance(evidence, Mapping) or dict(evidence) != required_evidence:
        raise ValueError("evidence_state_policy does not preserve owner receipts")


def validate_trajectory_stability_schema_pair(
    *,
    representation_schema: Mapping[str, Any],
    solution_schema: Mapping[str, Any],
) -> tuple[str, str, list[str], int, int]:
    """Validate every schema field consumed by the standard executor."""

    if representation_schema.get("schema_version") != _SUPPORTED_REPRESENTATION_SCHEMA:
        raise ValueError("trajectory representation schema version is unsupported")
    if solution_schema.get("schema_version") != _SUPPORTED_SOLUTION_SCHEMA:
        raise ValueError("candidate solution schema version is unsupported")
    id_column = str(representation_schema.get("id_column") or "")
    if not id_column or id_column != str(solution_schema.get("id_column") or ""):
        raise ValueError("trajectory schemas disagree on id_column")
    representation_columns = _nonempty_string_list(
        representation_schema.get("representation_columns"),
        field="representation_columns",
    )
    solution_representation_columns = solution_schema.get("representation_columns")
    if solution_representation_columns != representation_columns:
        raise ValueError(
            "candidate solution and representation schemas disagree on coordinate order"
        )
    frozen_population_n = _strict_int(
        representation_schema.get("frozen_population_n"),
        field="frozen_population_n",
        minimum=1,
    )
    assignment_column = str(solution_schema.get("assignment_column") or "")
    if not assignment_column:
        raise ValueError("candidate schema lacks assignment_column")
    if not str(solution_schema.get("selected_model_id") or "").strip():
        raise ValueError("candidate schema lacks selected_model_id")
    for field in ("criterion",):
        if not str(solution_schema.get(field) or "").strip():
            raise ValueError(f"candidate schema lacks {field}")
    if solution_schema.get("selection_rule") not in {
        "minimum",
        "maximum",
        "elbow",
        "multi_criteria",
    }:
        raise ValueError("candidate schema has unsupported selection_rule")
    if solution_schema.get("direction") not in {
        "minimize",
        "maximize",
        "not_applicable",
    }:
        raise ValueError("candidate schema has unsupported direction")
    selected_criterion_value = solution_schema.get("selected_criterion_value")
    if (
        isinstance(selected_criterion_value, bool)
        or not isinstance(selected_criterion_value, (int, float))
        or not math.isfinite(float(selected_criterion_value))
    ):
        raise ValueError("candidate schema lacks finite selected_criterion_value")
    selected_n_clusters = _strict_int(
        solution_schema.get("selected_n_clusters"),
        field="selected_n_clusters",
        minimum=2,
    )
    _validate_representation_policy(representation_schema)
    if _normalise(solution_schema.get("model_family")) != _SUPPORTED_MODEL_FAMILY:
        raise ValueError(
            "selected candidate model family is unsupported by this executor"
        )
    if _normalise(solution_schema.get("fit_method")) != _SUPPORTED_FIT_METHOD:
        raise ValueError(
            "selected candidate fit method is unsupported by this executor"
        )
    if _normalise(solution_schema.get("covariance_type")) != _SUPPORTED_COVARIANCE:
        raise ValueError(
            "selected candidate covariance is unsupported by this executor"
        )
    if selected_n_clusters >= frozen_population_n:
        raise ValueError("selected_n_clusters must be smaller than frozen_population_n")
    if solution_schema.get("coordinate_scaling") != representation_schema.get(
        "coordinate_scaling"
    ):
        raise ValueError(
            "candidate solution and representation schemas disagree on scaling"
        )
    return (
        id_column,
        assignment_column,
        representation_columns,
        frozen_population_n,
        selected_n_clusters,
    )


def _validate_cluster_selection_binding(
    *,
    selection_payload: object,
    solution_schema: Mapping[str, Any],
) -> None:
    """Bind the full Planner-owned selection manifest to the chosen schema."""

    selection = ClusterSelectionManifest.model_validate(selection_payload)
    selected_value = next(
        candidate.criterion_value
        for candidate in selection.candidates
        if candidate.n_clusters == selection.selected_n_clusters
    )
    expected = {
        "criterion": selection.criterion,
        "selection_rule": selection.selection_rule,
        "direction": selection.direction,
        "selected_n_clusters": selection.selected_n_clusters,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if solution_schema.get(field) != value
    ]
    try:
        recorded_value = float(solution_schema.get("selected_criterion_value"))
    except (TypeError, ValueError):
        recorded_value = math.nan
    if not math.isclose(
        recorded_value,
        float(selected_value),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        mismatches.append("selected_criterion_value")
    if mismatches:
        raise ValueError(
            "cluster selection manifest disagrees with candidate solution schema: "
            f"{sorted(mismatches)}"
        )


def _scale_coordinates(
    x: np.ndarray,
    *,
    columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen observed-value z-score policy and return its manifest."""

    values = np.asarray(x, dtype=float)
    stats: list[dict[str, Any]] = []
    scaled = values.copy()
    for index, column in enumerate(columns):
        observed = np.isfinite(values[:, index])
        count = int(observed.sum())
        if count == 0:
            raise ValueError(f"scaling coordinate {column!r} has no observed values")
        center = float(np.mean(values[observed, index]))
        scale = float(np.std(values[observed, index], ddof=0))
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"scaling coordinate {column!r} has zero/non-finite variance"
            )
        scaled[observed, index] = (values[observed, index] - center) / scale
        stats.append(
            {
                "coordinate": column,
                "observed_n": count,
                "center": center,
                "scale": scale,
            }
        )
    body = {
        "schema_version": "easyicu.trajectory_coordinate_scaling/1",
        "method": "pooled_coordinate_wise_z_score",
        "ddof": 0,
        "observed_value_policy": "direct_or_owner_locf_available",
        "missing_value_policy": "preserve_missing_exclude_from_likelihood",
        "zero_variance_action": "fail_closed",
        "coordinates": stats,
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return scaled, {**body, "scaling_manifest_sha256": digest}


def validate_trajectory_stability_upstream(
    *,
    representation: pd.DataFrame,
    assignments: pd.DataFrame,
    representation_schema: Mapping[str, Any],
    solution_schema: Mapping[str, Any],
) -> tuple[str, str, list[str], int, np.ndarray, pd.Series]:
    (
        id_column,
        assignment_column,
        representation_columns,
        frozen_population_n,
        selected_n_clusters,
    ) = validate_trajectory_stability_schema_pair(
        representation_schema=representation_schema,
        solution_schema=solution_schema,
    )
    if frozen_population_n != len(representation):
        raise ValueError("representation row count disagrees with frozen_population_n")

    missing_representation = [
        value
        for value in [id_column, *representation_columns]
        if value not in representation
    ]
    missing_assignments = [
        value for value in [id_column, assignment_column] if value not in assignments
    ]
    if missing_representation or missing_assignments:
        raise ValueError(
            "typed trajectory tables do not satisfy their schemas: "
            f"representation={missing_representation}, assignments={missing_assignments}"
        )
    if (
        representation[id_column].isna().any()
        or representation[id_column].duplicated().any()
    ):
        raise ValueError(
            "trajectory representation identifiers must be complete and unique"
        )
    selected = assignments[[id_column, assignment_column]].copy()
    if selected[id_column].isna().any() or selected[id_column].duplicated().any():
        raise ValueError("candidate assignment identifiers must be complete and unique")
    rep_ids = representation[id_column].map(_canonical_id)
    assignment_ids = selected[id_column].map(_canonical_id)
    if rep_ids.duplicated().any() or assignment_ids.duplicated().any():
        raise ValueError("canonicalized trajectory identifiers are not unique")
    assignment_by_id = dict(
        zip(assignment_ids, selected[assignment_column], strict=True)
    )
    if set(rep_ids) != set(assignment_by_id):
        raise ValueError(
            "representation and candidate assignment identifier sets differ"
        )
    reference = pd.Series(
        [assignment_by_id[value] for value in rep_ids], index=representation.index
    )
    if reference.isna().any() or reference.nunique() != selected_n_clusters:
        raise ValueError("candidate labels do not realize selected_n_clusters")

    numeric = representation[representation_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    newly_invalid = representation[representation_columns].notna() & numeric.isna()
    if newly_invalid.any().any():
        raise ValueError("numeric coercion invalidated a representation value")
    x = numeric.to_numpy(dtype=float)
    if np.isinf(x).any() or (~np.isfinite(x)).all(axis=1).any():
        raise ValueError("representation contains infinite or all-missing rows")
    return (
        id_column,
        assignment_column,
        representation_columns,
        selected_n_clusters,
        x,
        reference,
    )


def run_trajectory_stability(
    *,
    spec: TrajectoryStabilitySpec | Mapping[str, Any],
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: str | Path | Mapping[str, Any],
    scientific_runtime_authority: (
        TrajectoryScientificRuntimeAuthority | Mapping[str, Any] | None
    ) = None,
    runtime_projection_sha256: str | None = None,
    include_characterization: bool = False,
) -> Mapping[str, Any]:
    """Execute exactly one frozen stability design and always write a summary."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _empty_outputs(out_dir)
    pending_assignments_path = out_dir / ".cluster_stability_assignments.pending.csv"
    pending_assignments_path.unlink(missing_ok=True)
    summary: dict[str, Any] = {
        "analysis_family": "trajectory_clustering",
        "status": "failed_closed",
        "freeze_status": "not_frozen",
        "deterministic_standard_analysis": "trajectory_cluster_stability",
        "scientific_design_owner": "planner_agent",
        "outcome_binding_received_by_executor": False,
        "outcome_bindings_received": [],
        "eligibility_reapplied": False,
        "errors": [],
        "failure_class": None,
        "reason_code": None,
    }
    sealed_authority = None
    try:
        if scientific_runtime_authority is not None:
            sealed_authority = load_trajectory_scientific_runtime_authority(
                scientific_runtime_authority
            )
            if len(str(runtime_projection_sha256 or "")) != 64:
                raise ValueError("runtime projection digest is required")
            summary["scientific_design_owner"] = "signed_runtime_projection"
            summary["scientific_runtime_authority"] = {
                "schema_version": sealed_authority.schema_version,
                "protocol_content_sha256": sealed_authority.protocol_content_sha256,
                "execution_contract_sha256": (
                    sealed_authority.execution_contract_sha256
                ),
                "runtime_projection_sha256": runtime_projection_sha256,
            }
    except Exception as exc:
        summary["failure_class"] = "input_or_contract_failure"
        summary["reason_code"] = "TRAJECTORY_SCIENTIFIC_AUTHORITY_INVALID"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        _write_json(out_dir / "step_summary.json", summary)
        return summary
    try:
        spec = (
            spec
            if isinstance(spec, TrajectoryStabilitySpec)
            else TrajectoryStabilitySpec.model_validate(spec)
        )
    except Exception as exc:
        summary["failure_class"] = "input_or_contract_failure"
        summary["reason_code"] = "TRAJECTORY_STABILITY_SPEC_INVALID"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
        _write_json(out_dir / "step_summary.json", summary)
        return summary
    summary["trajectory_stability_spec"] = spec.model_dump(mode="json")
    try:
        if sealed_authority is not None and spec.model_dump(mode="json") != (
            sealed_authority.stability_spec.model_dump(mode="json")
        ):
            raise ValueError("stability spec drifted from signed runtime authority")
        inputs = _load_resolved_inputs(resolved_inputs)
        input_receipts: list[dict[str, Any]] = []
        summary["input_bindings"] = input_receipts
        missing_bindings = sorted(STABILITY_EXECUTOR_INPUTS - set(inputs))
        if missing_bindings:
            raise ValueError(f"required typed bindings are absent: {missing_bindings}")
        unexpected_bindings = sorted(set(inputs) - STABILITY_EXECUTOR_INPUTS)
        if unexpected_bindings:
            raise ValueError(
                "trajectory stability executor received undeclared typed bindings: "
                f"{unexpected_bindings}"
            )
        loaded_inputs: dict[str, Any] = {}
        for input_key in sorted(STABILITY_EXECUTOR_INPUTS):
            loaded_value = _read_bound(
                inputs=inputs,
                key=input_key,
                run_dir=run_dir,
            )
            loaded_inputs[input_key] = loaded_value
            input_receipts.append(
                _loaded_input_receipt(
                    inputs=inputs,
                    key=input_key,
                    value=loaded_value,
                )
            )
        representation = loaded_inputs["artifact:trajectory_representation"]
        assignments = loaded_inputs["artifact:candidate_cluster_assignments"]
        representation_schema = loaded_inputs[
            "manifest:trajectory_representation_schema"
        ]
        selection_manifest = loaded_inputs["manifest:cluster_selection"]
        solution_schema = loaded_inputs["manifest:candidate_cluster_solution_schema"]
        if not isinstance(representation, pd.DataFrame) or not isinstance(
            assignments, pd.DataFrame
        ):
            raise ValueError("trajectory representation and assignments must be tables")
        if (
            not isinstance(representation_schema, Mapping)
            or not isinstance(selection_manifest, Mapping)
            or not isinstance(solution_schema, Mapping)
        ):
            raise ValueError("trajectory schemas must be JSON objects")
        _require_artifact_digest_link(
            payload=representation_schema,
            digest_field="representation_sha256",
            legacy_evidence_field="representation_evidence_id",
            inputs=inputs,
            input_key="artifact:trajectory_representation",
        )
        _require_artifact_digest_link(
            payload=solution_schema,
            digest_field="candidate_assignments_sha256",
            legacy_evidence_field="candidate_assignments_evidence_id",
            inputs=inputs,
            input_key="artifact:candidate_cluster_assignments",
        )
        _require_artifact_digest_link(
            payload=solution_schema,
            digest_field="representation_schema_sha256",
            legacy_evidence_field="representation_schema_evidence_id",
            inputs=inputs,
            input_key="manifest:trajectory_representation_schema",
        )
        if sealed_authority is not None:
            sealed_authority.validate_representation_schema(representation_schema)
            sealed_authority.validate_selection(selection_manifest)
            expected_authority_binding = {
                "schema_version": sealed_authority.schema_version,
                "protocol_content_sha256": sealed_authority.protocol_content_sha256,
                "execution_contract_sha256": (
                    sealed_authority.execution_contract_sha256
                ),
                "runtime_projection_sha256": runtime_projection_sha256,
            }
            if solution_schema.get("scientific_runtime_authority") != (
                expected_authority_binding
            ):
                raise ValueError(
                    "candidate solution is not bound to signed runtime authority"
                )

        (
            id_column,
            _assignment_column,
            representation_columns,
            selected_n_clusters,
            x,
            reference,
        ) = validate_trajectory_stability_upstream(
            representation=representation,
            assignments=assignments,
            representation_schema=representation_schema,
            solution_schema=solution_schema,
        )
        _validate_cluster_selection_binding(
            selection_payload=selection_manifest,
            solution_schema=solution_schema,
        )
        x, scaling_manifest = _scale_coordinates(
            x,
            columns=representation_columns,
        )
        if sealed_authority is not None:
            if tuple(representation_columns) != sealed_authority.representation_columns:
                raise ValueError(
                    "candidate representation columns drifted from signed authority"
                )
            if solution_schema.get("coordinate_scaling_manifest") != scaling_manifest:
                raise ValueError(
                    "candidate BIC selection did not bind the deterministic scaling manifest"
                )
            if solution_schema.get("coordinate_scaling_manifest_sha256") != (
                scaling_manifest["scaling_manifest_sha256"]
            ):
                raise ValueError("candidate scaling-manifest digest mismatch")
        _write_json(
            out_dir / "trajectory_coordinate_scaling_manifest.json",
            scaling_manifest,
        )
        summary["coordinate_scaling_sha256"] = scaling_manifest[
            "scaling_manifest_sha256"
        ]
        _empty_outputs(out_dir, id_column=id_column)
        n_rows = len(representation)
        sample_n = (
            int(spec.sample_size)
            if spec.sample_size is not None
            else int(math.floor(n_rows * float(spec.sample_fraction)))
        )
        if sample_n <= selected_n_clusters or sample_n >= n_rows:
            raise ValueError(
                "agent-owned sample size must exceed selected_n_clusters and be "
                "strictly smaller than the frozen population"
            )
        if math.comb(n_rows, sample_n) < spec.n_resamples:
            raise ValueError(
                "the planner-owned distinct-membership design requests more "
                "resamples than the number of possible subsamples"
            )

        seed_children = np.random.SeedSequence(spec.base_seed).spawn(spec.n_resamples)
        seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_children
        ]
        if len(set(seeds)) != len(seeds):
            raise ValueError("seed derivation did not produce distinct seeds")

        spec_payload = spec.model_dump(mode="json")
        spec_digest = hashlib.sha256(
            json.dumps(spec_payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        executor_code_sha256 = _sha256(Path(__file__).resolve())
        resolved_spec = {
            "schema_version": "easyicu.cluster_stability_spec/1",
            "executor_version": "easyicu_observed_data_diag_gmm_v1",
            "scientific_design_owner": "planner_agent",
            "trajectory_stability_spec": spec_payload,
            "trajectory_stability_spec_sha256": spec_digest,
            "executor_code_sha256": executor_code_sha256,
            "bound_sample_n": sample_n,
            "derived_seeds": seeds,
            "selected_n_clusters": selected_n_clusters,
            "model_family": solution_schema.get("model_family"),
            "fit_method": solution_schema.get("fit_method"),
            "covariance_type": solution_schema.get("covariance_type"),
            "representation_columns": representation_columns,
            "coordinate_scaling": scaling_manifest,
            "scientific_runtime_authority": summary.get(
                "scientific_runtime_authority"
            ),
            "execution_parallelism": {
                "method": "ordered_sequential_refits_v1",
                "max_workers": 1,
                "native_math_thread_environment": {
                    key: os.environ.get(key) for key in _NATIVE_MATH_THREAD_ENV
                },
            },
            "input_provenance": {
                key: _binding_provenance(inputs, key)
                for key in sorted(STABILITY_EXECUTOR_INPUTS)
            },
        }
        _write_json(out_dir / "cluster_stability_spec.json", resolved_spec)

        ids = representation[id_column].reset_index(drop=True)
        reference_array = reference.to_numpy(dtype=object)
        reference_universe = pd.unique(reference_array).tolist()
        successful_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        attempt_rows: list[dict[str, Any]] = []
        inclusion_counts = np.zeros(n_rows, dtype=np.int64)
        agreement_counts = np.zeros(n_rows, dtype=np.int64)
        seen_hashes: set[str] = set()
        assignments_header_written = False
        for index, seed in enumerate(seeds, start=1):
            resample_id = f"stability_resample_{index:03d}"
            rng = np.random.default_rng(seed)
            positions = np.sort(rng.choice(n_rows, size=sample_n, replace=False))
            sampled_ids = ids.iloc[positions]
            sample_id_hash = _sample_hash(sampled_ids)
            refit_model_id = (
                f"{resample_id}::{spec.refit_engine}::k{selected_n_clusters}"
            )
            try:
                if sample_id_hash in seen_hashes:
                    raise ValueError("subsample membership duplicated an earlier refit")
                reference_sample = reference_array[positions]
                if np.unique(reference_sample).size < 2:
                    raise ValueError(
                        "sampled reference assignments contain fewer than two clusters"
                    )
                refit_labels, fit_trace = _fit_observed_data_diag_gmm(
                    x[positions],
                    n_components=selected_n_clusters,
                    seed=seed,
                    max_iter=spec.refit_max_iter,
                    tolerance=spec.refit_tolerance,
                    regularization=spec.refit_regularization,
                )
                aligned = _aligned_labels(
                    reference_sample,
                    refit_labels,
                    reference_universe=reference_universe,
                )
                ari = float(adjusted_rand_score(reference_sample, refit_labels))
                if not math.isfinite(ari):
                    raise ValueError("adjusted Rand index is non-finite")
                seen_hashes.add(sample_id_hash)
                success_row = {
                    "resample_id": resample_id,
                    "n_overlap": sample_n,
                    "adjusted_rand_index": ari,
                    "clustering_method": solution_schema["model_family"],
                    "refit_model_id": refit_model_id,
                    "seed": seed,
                    "sampling_method": spec.resampling_method,
                    "sample_n": sample_n,
                    "sample_id_hash": sample_id_hash,
                    "selected_n_clusters": selected_n_clusters,
                    **fit_trace,
                }
                successful_rows.append(success_row)
                attempt_rows.append(
                    {
                        **success_row,
                        "status": "success",
                        "executor_version": spec.refit_engine,
                        "spec_sha256": spec_digest,
                        "representation_sha256": inputs[
                            "artifact:trajectory_representation"
                        ].get("sha256"),
                    }
                )
                assignment_frame = pd.DataFrame(
                    {
                        "resample_id": resample_id,
                        id_column: sampled_ids.to_numpy(),
                        "reference_cluster": reference_sample,
                        "resampled_cluster": aligned,
                    }
                )
                assignment_frame.to_csv(
                    pending_assignments_path,
                    mode="a",
                    header=not assignments_header_written,
                    index=False,
                )
                assignments_header_written = True
                inclusion_counts[positions] += 1
                agreement_counts[positions] += np.asarray(
                    aligned == reference_sample,
                    dtype=np.int64,
                )
            except Exception as exc:
                failure_row = {
                    "resample_id": resample_id,
                    "refit_model_id": refit_model_id,
                    "seed": seed,
                    "sample_n": sample_n,
                    "sample_id_hash": sample_id_hash,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failure_rows.append(failure_row)
                attempt_rows.append(
                    {
                        **failure_row,
                        "status": "failed",
                        "executor_version": spec.refit_engine,
                        "spec_sha256": spec_digest,
                        "representation_sha256": inputs[
                            "artifact:trajectory_representation"
                        ].get("sha256"),
                    }
                )

        _write_json(
            out_dir / "cluster_stability_refit_attempts.json",
            {
                "planned_n_resamples": spec.n_resamples,
                "attempted_n_resamples": len(attempt_rows),
                "successful_n_resamples": len(successful_rows),
                "failed_n_resamples": len(failure_rows),
                "attempts": attempt_rows,
            },
        )
        if len(attempt_rows) != spec.n_resamples:
            raise ValueError("stability attempt ledger does not match n_resamples")

        if len(successful_rows) < spec.minimum_successful_resamples:
            pending_assignments_path.unlink(missing_ok=True)
            _write_json(
                out_dir / "cluster_stability_refit_failures.json",
                {"failures": failure_rows},
            )
            raise _NumericalRefitFailure(
                "successful stability refits are below the planner-owned minimum: "
                f"{len(successful_rows)} < {spec.minimum_successful_resamples}"
            )
        stability = pd.DataFrame(successful_rows)
        stability.to_csv(out_dir / "cluster_stability.csv", index=False)
        if not assignments_header_written or not pending_assignments_path.is_file():
            raise ValueError("successful refits did not produce assignment rows")
        pending_assignments_path.replace(out_dir / "cluster_stability_assignments.csv")

        final_assignments = pd.DataFrame(
            {id_column: ids.tolist(), "cluster": reference_array.tolist()}
        )
        final_assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
        if include_characterization:
            profile_rows: list[dict[str, Any]] = []
            for cluster in sorted(pd.unique(reference_array), key=str):
                mask = reference_array == cluster
                for column in representation_columns:
                    values = pd.to_numeric(
                        representation.loc[mask, column], errors="coerce"
                    ).dropna()
                    match = re.fullmatch(r".+__h(-?\d+)_(-?\d+)", column)
                    if match is None:
                        raise ValueError(
                            f"representation column lacks a signed window: {column}"
                        )
                    profile_rows.append(
                        {
                            "cluster": cluster,
                            "source_column": column,
                            "window_start_hours": int(match.group(1)),
                            "window_end_hours": int(match.group(2)),
                            "summary_statistic": "mean",
                            "value": float(values.mean()),
                            "n_observed": int(len(values)),
                        }
                    )
            pd.DataFrame(profile_rows).to_csv(
                out_dir / "trajectory_profiles.csv", index=False
            )
            (
                final_assignments.groupby("cluster", sort=True)
                .size()
                .rename("n")
                .reset_index()
                .to_csv(out_dir / "cluster_sizes.csv", index=False)
            )
        provenance = final_assignments.copy()
        provenance["stability_inclusion_n"] = inclusion_counts
        provenance["assignment_agreement_n"] = agreement_counts
        provenance["stability_inclusion_fraction"] = provenance[
            "stability_inclusion_n"
        ] / len(successful_rows)
        provenance["assignment_agreement_fraction"] = np.where(
            provenance["stability_inclusion_n"] > 0,
            provenance["assignment_agreement_n"] / provenance["stability_inclusion_n"],
            np.nan,
        )
        provenance["assignment_uncertainty_fraction"] = (
            1.0 - provenance["assignment_agreement_fraction"]
        )
        provenance.to_csv(out_dir / "cluster_assignment_provenance.csv", index=False)

        policy = {
            "schema_version": "easyicu.trajectory_missingness_policy/1",
            "id_column": id_column,
            "observation_family": representation_schema.get("observation_family"),
            "observation_columns": representation_schema.get("observation_columns"),
            "representation_columns": representation_columns,
            "min_observed_windows": representation_schema.get("min_observed_windows"),
            "profile_columns": representation_schema.get("profile_columns"),
            "profile_summary_statistic": representation_schema.get(
                "profile_summary_statistic"
            ),
            "clustering_method": solution_schema.get("model_family"),
            "fit_method": solution_schema.get("fit_method"),
            "covariance_type": solution_schema.get("covariance_type"),
            "n_clusters": selected_n_clusters,
            "time_axis": representation_schema.get("time_axis"),
            "anchor": representation_schema.get("anchor"),
            "anchor_provenance": representation_schema.get("anchor_provenance"),
            "anchor_source": representation_schema.get("anchor_source"),
            "trailing_na_policy": representation_schema.get("trailing_na_policy"),
            "evidence_state_policy": representation_schema.get(
                "evidence_state_policy"
            ),
            "coordinate_scaling": scaling_manifest,
        }
        _write_json(out_dir / "trajectory_missingness_policy.json", policy)

        mean_ari = float(stability["adjusted_rand_index"].mean())
        threshold_passed = (
            None
            if spec.decision_mode == "report_only"
            else mean_ari >= float(spec.minimum_mean_stability)
        )
        freeze_status = (
            "candidate_labels_preserved_report_only_no_stability_decision"
            if threshold_passed is None
            else (
                "candidate_labels_frozen_stability_threshold_passed"
                if threshold_passed
                else "not_frozen_stability_threshold_failed"
            )
        )
        freeze = {
            "schema_version": "easyicu.trajectory_stability_freeze/1",
            "freeze_status": freeze_status,
            "selected_n_clusters": selected_n_clusters,
            "selected_model_id": solution_schema.get("selected_model_id"),
            "model_family": solution_schema.get("model_family"),
            "fit_method": solution_schema.get("fit_method"),
            "covariance_type": solution_schema.get("covariance_type"),
            "reference_assignment_policy": spec.final_assignment_policy,
            "n_successful_resamples": len(successful_rows),
            "n_failed_resamples": len(failure_rows),
            "mean_adjusted_rand_index": mean_ari,
            "minimum_mean_stability": spec.minimum_mean_stability,
            "stability_threshold_passed": threshold_passed,
            "trajectory_stability_spec": spec.model_dump(mode="json"),
            "trajectory_stability_spec_sha256": spec_digest,
            "executor_version": spec.refit_engine,
            "executor_code_sha256": executor_code_sha256,
            "input_provenance": {
                key: _binding_provenance(inputs, key)
                for key in sorted(STABILITY_EXECUTOR_INPUTS)
            },
            "coordinate_scaling_sha256": scaling_manifest[
                "scaling_manifest_sha256"
            ],
            "outcome_binding_received_by_executor": False,
            "outcome_bindings_received": [],
            "eligibility_reapplied": False,
        }
        _write_json(out_dir / "stability_freeze.json", freeze)
        if failure_rows:
            _write_json(
                out_dir / "cluster_stability_refit_failures.json",
                {"failures": failure_rows},
            )

        output_files = {
            "artifact:stability_freeze": "stability_freeze.json",
            "artifact:cluster_assignments": "cluster_assignments.csv",
            "table:cluster_assignments": "cluster_assignments.csv",
            "table:cluster_stability": "cluster_stability.csv",
            "table:cluster_stability_assignments": "cluster_stability_assignments.csv",
            "table:cluster_assignment_provenance": "cluster_assignment_provenance.csv",
            "manifest:trajectory_missingness_policy": "trajectory_missingness_policy.json",
            "manifest:cluster_stability_spec": "cluster_stability_spec.json",
            "manifest:trajectory_coordinate_scaling": (
                "trajectory_coordinate_scaling_manifest.json"
            ),
        }
        if include_characterization:
            output_files.update(
                {
                    "table:trajectory_profiles": "trajectory_profiles.csv",
                    "table:cluster_sizes": "cluster_sizes.csv",
                }
            )
        summary.update(
            {
                "status": ("ok" if threshold_passed is not False else "failed_closed"),
                "freeze_status": freeze_status,
                "selected_n_clusters": selected_n_clusters,
                "n_clusters": selected_n_clusters,
                "clustering_method": solution_schema.get("model_family"),
                "fit_method": solution_schema.get("fit_method"),
                "min_observed_windows": representation_schema.get(
                    "min_observed_windows"
                ),
                "n_successful_resamples": len(successful_rows),
                "n_failed_resamples": len(failure_rows),
                "mean_adjusted_rand_index": mean_ari,
                "cluster_stability": {
                    "selected_n_clusters": selected_n_clusters,
                    "n_resamples": len(successful_rows),
                    "mean_adjusted_rand_index": mean_ari,
                    "metric": spec.stability_metric,
                },
                "stability_threshold_passed": threshold_passed,
                "trajectory_stability_spec_sha256": spec_digest,
                "executor_version": spec.refit_engine,
                "executor_code_sha256": executor_code_sha256,
                "output_files": output_files,
                "outputs": sorted(set(output_files.values())),
            }
        )
        if threshold_passed is False:
            summary["failure_class"] = "scientific_instability"
            summary["reason_code"] = "TRAJECTORY_STABILITY_BELOW_THRESHOLD"
            summary["errors"].append(
                "Mean stability was below the planner-owned threshold; "
                "the selected k was not changed, execution failed closed, and a "
                "new planner revision is required before retrying."
            )
    except _NumericalRefitFailure as exc:
        summary["failure_class"] = "numerical_engine_failure"
        summary["reason_code"] = "TRAJECTORY_REFIT_ENGINE_FAILURE"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        if summary["failure_class"] is None:
            summary["failure_class"] = "input_or_contract_failure"
            summary["reason_code"] = "TRAJECTORY_STABILITY_CONTRACT_INVALID"
        summary["errors"].append(f"{type(exc).__name__}: {exc}")
    pending_assignments_path.unlink(missing_ok=True)
    _write_json(out_dir / "step_summary.json", summary)
    return summary
