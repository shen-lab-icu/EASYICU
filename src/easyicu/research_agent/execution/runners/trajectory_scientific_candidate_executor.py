"""Deterministic candidate-k selection bound to a signed trajectory authority."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...schema import AnalysisPlan, AnalysisStep
from ...trajectory.plan_contract import trajectory_step_roles
from ...trajectory.scientific_runtime_authority import (
    TrajectoryScientificRuntimeAuthority,
    load_trajectory_scientific_runtime_authority,
)
from .trajectory_stability_executor import (
    _fit_observed_data_diag_gmm,
    _load_resolved_inputs,
    _loaded_input_receipt,
    _read_bound,
    _scale_coordinates,
    _sha256,
)


SCIENTIFIC_CANDIDATE_INPUTS = frozenset(
    {
        "artifact:trajectory_representation",
        "manifest:trajectory_representation_schema",
    }
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def trajectory_scientific_candidate_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None or trajectory_step_roles(step) != frozenset(
        {"candidate_selection"}
    ):
        return False
    owners = [
        item
        for item in plan.steps
        if "candidate_selection" in trajectory_step_roles(item)
    ]
    return owners == [step] and SCIENTIFIC_CANDIDATE_INPUTS.issubset(
        {str(value).strip().lower() for value in step.inputs}
    )


def trajectory_scientific_candidate_executor_code(
    *,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
) -> str:
    sealed = load_trajectory_scientific_runtime_authority(authority)
    authority_json = json.dumps(sealed.model_dump(mode="json"), sort_keys=True)
    return (
        "import json, os\n"
        "from pathlib import Path\n"
        "from easyicu.research_agent.execution.runners."
        "trajectory_scientific_candidate_executor import "
        "run_trajectory_scientific_candidate_selection\n"
        f"authority = json.loads({json.dumps(authority_json)})\n"
        "run_trajectory_scientific_candidate_selection("
        "authority=authority, "
        f"runtime_projection_sha256={runtime_projection_sha256!r}, "
        "out_dir=Path(os.environ['STEP_OUT_DIR']), "
        "run_dir=Path(os.environ['EASYICU_RUN_DIR']), "
        "resolved_inputs=os.environ['EASYICU_RESOLVED_INPUTS_JSON'])\n"
    )


def run_trajectory_scientific_candidate_selection(
    *,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Scale once, fit every frozen k, apply BIC/boundary/size gates, and seal."""

    sealed = load_trajectory_scientific_runtime_authority(authority)
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("runtime projection digest is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_resolved_inputs(resolved_inputs)
    missing = sorted(SCIENTIFIC_CANDIDATE_INPUTS - set(inputs))
    unexpected = sorted(set(inputs) - SCIENTIFIC_CANDIDATE_INPUTS)
    if missing or unexpected:
        raise ValueError(
            f"signed candidate inputs mismatch: missing={missing}, unexpected={unexpected}"
        )
    representation = _read_bound(
        inputs=inputs,
        key="artifact:trajectory_representation",
        run_dir=run_dir,
    )
    schema = _read_bound(
        inputs=inputs,
        key="manifest:trajectory_representation_schema",
        run_dir=run_dir,
    )
    if not isinstance(representation, pd.DataFrame) or not isinstance(schema, Mapping):
        raise ValueError("signed candidate inputs have unsupported types")
    sealed.validate_representation_schema(schema)
    binding = inputs["artifact:trajectory_representation"]
    if not isinstance(binding, Mapping) or schema.get("representation_sha256") != (
        binding.get("sha256")
    ):
        raise ValueError("representation schema does not bind exact matrix bytes")
    id_column = str(schema.get("id_column") or "")
    columns = list(sealed.representation_columns)
    if id_column not in representation or any(
        column not in representation for column in columns
    ):
        raise ValueError("signed representation columns are absent")
    if (
        representation[id_column].isna().any()
        or representation[id_column].duplicated().any()
    ):
        raise ValueError("trajectory representation identifiers are not unique")
    x = representation[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(x).any(axis=1).all():
        raise ValueError("a trajectory row has no observed model coordinate")
    scaled, scaling_manifest = _scale_coordinates(x, columns=columns)
    _write_json(
        out_dir / "trajectory_coordinate_scaling_manifest.json", scaling_manifest
    )
    seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(sealed.candidate_fit_base_seed).spawn(
            len(sealed.candidate_cluster_counts)
        )
    ]
    n_rows, n_coordinates = scaled.shape
    candidate_rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k, seed in zip(sealed.candidate_cluster_counts, seeds, strict=True):
        labels, fit = _fit_observed_data_diag_gmm(
            scaled,
            n_components=k,
            seed=seed,
            max_iter=sealed.candidate_fit_max_iter,
            tolerance=sealed.candidate_fit_tolerance,
            regularization=sealed.candidate_fit_regularization,
        )
        parameter_count = (k - 1) + 2 * k * n_coordinates
        final_log_likelihood = float(fit["final_log_likelihood"])
        bic = -2.0 * final_log_likelihood + parameter_count * math.log(n_rows)
        # AIC is an explicitly diagnostic second criterion.  The signed
        # protocol continues to select by BIC; reporting AIC cannot change K
        # or alter a boundary solution after looking at the result.
        aic = -2.0 * final_log_likelihood + 2.0 * parameter_count
        labels_by_k[k] = labels
        candidate_rows.append(
            {
                "model_id": f"signed-observed-data-diag-gmm-k{k}",
                "n_clusters": k,
                "criterion_value": bic,
                "bic": bic,
                "aic": aic,
                "final_log_likelihood": final_log_likelihood,
                "parameter_count": parameter_count,
                "seed": seed,
                "n_iter": int(fit["n_iter"]),
                "parameter_sha256": str(fit["parameter_sha256"]),
            }
        )
    selected = min(candidate_rows, key=lambda item: (item["bic"], item["n_clusters"]))
    selected_k = int(selected["n_clusters"])
    aic_selected = min(
        candidate_rows, key=lambda item: (item["aic"], item["n_clusters"])
    )
    aic_selected_k = int(aic_selected["n_clusters"])
    selection = {
        "criterion": sealed.selection_criterion,
        "selection_rule": sealed.selection_rule,
        "direction": "minimize",
        "selected_n_clusters": selected_k,
        "candidates": [
            {
                "n_clusters": int(row["n_clusters"]),
                "criterion_value": float(row["bic"]),
            }
            for row in candidate_rows
        ],
        "rationale": (
            "Deterministic minimum BIC over the signed candidate grid; exact ties "
            "resolve to smaller k."
        ),
    }
    _write_json(out_dir / "cluster_selection.json", selection)
    authority_binding = {
        "schema_version": sealed.schema_version,
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
    }
    base_summary: dict[str, Any] = {
        "status": "ok",
        "clustering_method": "observed_data_diagonal_gaussian_mixture",
        "n_clusters": selected_k,
        "cluster_selection": selection,
        "diagnostic_criteria": ["bic", "aic"],
        "diagnostic_aic_selected_n_clusters": aic_selected_k,
        "diagnostic_criteria_agree": aic_selected_k == selected_k,
        "coordinate_scaling_sha256": scaling_manifest["scaling_manifest_sha256"],
        "scientific_runtime_authority": authority_binding,
        "input_bindings": [
            _loaded_input_receipt(inputs=inputs, key=key, value=value)
            for key, value in (
                ("artifact:trajectory_representation", representation),
                ("manifest:trajectory_representation_schema", schema),
            )
        ],
    }
    sealed.validate_selection(
        selection,
        allow_prespecified_boundary_rejection=True,
    )
    labels = labels_by_k[selected_k]
    counts = pd.Series(labels).value_counts().sort_index()
    minimum_fraction = float(counts.min() / len(labels))
    scientific_rejection: dict[str, str] | None = None
    if selected_k == max(sealed.candidate_cluster_counts):
        scientific_rejection = {
            "reason_code": sealed.upper_boundary_reason_code,
            "reportable_result": (
                "no_interior_solution_in_prespecified_candidate_range"
            ),
        }
    elif minimum_fraction < sealed.minimum_cluster_fraction:
        scientific_rejection = {
            "reason_code": sealed.minimum_cluster_fraction_reason_code,
            "reportable_result": "no_stable_phenotype_solution",
        }
    selection_table = pd.DataFrame(
        [
            {
                "n_clusters": int(row["n_clusters"]),
                "bic": float(row["bic"]),
                "aic": float(row["aic"]),
                "final_log_likelihood": float(row["final_log_likelihood"]),
                "parameter_count": int(row["parameter_count"]),
                "selected": int(row["n_clusters"]) == selected_k,
                "aic_minimum": int(row["n_clusters"]) == aic_selected_k,
                "upper_boundary": int(row["n_clusters"])
                == max(sealed.candidate_cluster_counts),
                "scientific_status": (
                    "failed_closed" if scientific_rejection else "selected"
                ),
                "reason_code": (
                    scientific_rejection["reason_code"]
                    if scientific_rejection
                    else "NOT_APPLICABLE"
                ),
                "reportable_result": (
                    scientific_rejection["reportable_result"]
                    if scientific_rejection
                    else "candidate_selected_pending_stability"
                ),
            }
            for row in candidate_rows
        ]
    )
    selection_table.to_csv(out_dir / "trajectory_candidate_selection.csv", index=False)
    assignments = pd.DataFrame(
        {id_column: representation[id_column].to_numpy(), "candidate_cluster": labels}
    )
    assignments_path = out_dir / "candidate_cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    _write_json(
        out_dir / "candidate_cluster_models.json",
        {
            "model_family": sealed.model_family,
            "fit_method": sealed.fit_method,
            "covariance_type": sealed.covariance_type,
            "candidates": candidate_rows,
            "scientific_runtime_authority": authority_binding,
        },
    )
    schema_binding = inputs["manifest:trajectory_representation_schema"]
    assert isinstance(schema_binding, Mapping)
    solution = {
        "schema_version": "easyicu.candidate_cluster_solution_schema/2",
        "id_column": id_column,
        "representation_columns": columns,
        "model_family": sealed.model_family,
        "fit_method": sealed.fit_method,
        "covariance_type": sealed.covariance_type,
        "selected_n_clusters": selected_k,
        "selected_model_id": selected["model_id"],
        "assignment_column": "candidate_cluster",
        "criterion": sealed.selection_criterion,
        "selection_rule": sealed.selection_rule,
        "direction": "minimize",
        "selected_criterion_value": float(selected["bic"]),
        "diagnostic_criteria": ["bic", "aic"],
        "diagnostic_aic_selected_n_clusters": aic_selected_k,
        "diagnostic_criteria_agree": aic_selected_k == selected_k,
        "representation_schema_sha256": schema_binding.get("sha256"),
        "candidate_assignments_sha256": _sha256(assignments_path),
        "coordinate_scaling": sealed.scaling_payload,
        "coordinate_scaling_manifest": scaling_manifest,
        "coordinate_scaling_manifest_sha256": scaling_manifest[
            "scaling_manifest_sha256"
        ],
        "scientific_selection_status": (
            "failed_closed" if scientific_rejection else "interior_solution"
        ),
        "stability_authorized": scientific_rejection is None,
        "scientific_selection_reason_code": (
            scientific_rejection["reason_code"] if scientific_rejection else None
        ),
        "scientific_runtime_authority": authority_binding,
    }
    _write_json(out_dir / "candidate_cluster_solution_schema.json", solution)
    # Descriptive-only embedding; never consumed by the model or selector.
    display = np.where(np.isfinite(scaled), scaled, 0.0)
    centered = display - display.mean(axis=0, keepdims=True)
    left, singular, _right = np.linalg.svd(centered, full_matrices=False)
    embedding = pd.DataFrame(
        {
            id_column: representation[id_column].to_numpy(),
            "embedding_1": left[:, 0] * singular[0],
            "embedding_2": (
                left[:, 1] * singular[1] if left.shape[1] > 1 else np.zeros(n_rows)
            ),
            "cluster": labels,
        }
    )
    embedding.to_csv(out_dir / "embedding_plot.csv", index=False)
    summary = {
        **base_summary,
        "minimum_observed_cluster_fraction": minimum_fraction,
        "candidate_assignments_sha256": solution["candidate_assignments_sha256"],
        "candidate_solution_schema_sha256": _sha256(
            out_dir / "candidate_cluster_solution_schema.json"
        ),
        "embedding_role": "descriptive_only_not_model_input",
        "scientific_status": (
            "failed_closed" if scientific_rejection else "selected"
        ),
        "stability_authorized": scientific_rejection is None,
        "output_files": {
            "artifact:candidate_cluster_assignments": (
                "candidate_cluster_assignments.csv"
            ),
            "manifest:cluster_selection": "cluster_selection.json",
            "manifest:candidate_cluster_solution_schema": (
                "candidate_cluster_solution_schema.json"
            ),
            "table:trajectory_candidate_selection": (
                "trajectory_candidate_selection.csv"
            ),
        },
    }
    if scientific_rejection:
        summary.update(
            {
                "failure_class": "scientific_selection_failure",
                **scientific_rejection,
            }
        )
    _write_json(out_dir / "step_summary.json", summary)
    return summary


__all__ = [
    "SCIENTIFIC_CANDIDATE_INPUTS",
    "run_trajectory_scientific_candidate_selection",
    "trajectory_scientific_candidate_executor_code",
    "trajectory_scientific_candidate_executor_owns_step",
]
