"""Deterministic fixed-grid representation owned by a signed trajectory contract."""

from __future__ import annotations

import hashlib
import json
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def trajectory_scientific_representation_executor_owns_step(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any] | None,
) -> bool:
    if authority is None or trajectory_step_roles(step) != frozenset(
        {"representation"}
    ):
        return False
    owners = [
        item for item in plan.steps if "representation" in trajectory_step_roles(item)
    ]
    return owners == [step]


def trajectory_scientific_representation_executor_code(
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
        "trajectory_scientific_representation_executor import "
        "run_trajectory_scientific_representation\n"
        f"authority = json.loads({json.dumps(authority_json)})\n"
        "run_trajectory_scientific_representation("
        "authority=authority, "
        f"runtime_projection_sha256={runtime_projection_sha256!r}, "
        "trajectory_path=Path(os.environ['TRAJECTORY_PARQUET']), "
        "out_dir=Path(os.environ['STEP_OUT_DIR']))\n"
    )


def run_trajectory_scientific_representation(
    *,
    authority: TrajectoryScientificRuntimeAuthority | Mapping[str, Any],
    runtime_projection_sha256: str,
    trajectory_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Build the exact signed concept-by-window matrix without model imputation."""

    sealed = load_trajectory_scientific_runtime_authority(authority)
    if len(str(runtime_projection_sha256)) != 64:
        raise ValueError("runtime projection digest is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_parquet(trajectory_path)
    required = {
        "stay_id",
        "charttime",
        "concept",
        "value_num",
        "evidence_state",
        "owner_observed",
        "owner_available",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"signed trajectory input lacks columns: {missing}")
    frame = trajectory.loc[
        trajectory["concept"].isin(sealed.coordinate_concepts)
        & trajectory["charttime"].ge(sealed.window_start_hours)
        & trajectory["charttime"].le(sealed.window_end_hours)
    ].copy()
    if frame.empty:
        raise ValueError("signed trajectory coordinates have no rows")
    frame["value_num"] = pd.to_numeric(frame["value_num"], errors="coerce")
    frame["owner_available"] = pd.to_numeric(frame["owner_available"], errors="coerce")
    frame["owner_observed"] = pd.to_numeric(frame["owner_observed"], errors="coerce")
    invalid_state = ~frame["evidence_state"].isin(
        ["direct_observed", "owner_locf_available", "unavailable"]
    )
    if bool(invalid_state.any()):
        raise ValueError("trajectory input contains an unknown owner evidence state")
    sofa_rows = frame["concept"].astype(str).str.startswith("sofa2")
    if bool(
        (
            sofa_rows
            & (
                ~frame["owner_available"].isin([0, 1])
                | ~frame["owner_observed"].isin([0, 1])
            )
        ).any()
    ):
        raise ValueError("SOFA-2 coordinate rows lack binary owner receipts")
    usable = (
        frame["owner_available"].eq(1)
        & frame["value_num"].notna()
        & ~frame["evidence_state"].eq("unavailable")
    )
    frame = frame.loc[usable].copy()
    if frame.empty:
        raise ValueError("no owner-available coordinate values remain")
    relative = frame["charttime"] - sealed.window_start_hours
    frame["window_index"] = np.floor(relative / sealed.grid_width_hours).astype(int)
    n_windows = (sealed.window_end_hours - sealed.window_start_hours) // (
        sealed.grid_width_hours
    )
    frame = frame.loc[frame["window_index"].between(0, n_windows - 1)].copy()
    frame["representation_column"] = [
        f"{concept}__h{sealed.window_start_hours + int(index) * sealed.grid_width_hours}_"
        f"{sealed.window_start_hours + (int(index) + 1) * sealed.grid_width_hours}"
        for concept, index in zip(frame["concept"], frame["window_index"], strict=True)
    ]
    matrix = (
        frame.groupby(["stay_id", "representation_column"], sort=False)["value_num"]
        .max()
        .unstack("representation_column")
        .reindex(columns=list(sealed.representation_columns))
    )
    sofa_concepts = tuple(
        concept for concept in sealed.coordinate_concepts if concept.startswith("sofa2")
    )
    sofa_columns = [
        column
        for column in sealed.representation_columns
        if any(column.startswith(f"{concept}__h") for concept in sofa_concepts)
    ]
    available_windows = pd.DataFrame(
        {
            index: matrix[
                [
                    f"{concept}__h{sealed.window_start_hours + index * sealed.grid_width_hours}_"
                    f"{sealed.window_start_hours + (index + 1) * sealed.grid_width_hours}"
                    for concept in sofa_concepts
                ]
            ]
            .notna()
            .any(axis=1)
            for index in range(n_windows)
        }
    )
    observed_window_count = available_windows.sum(axis=1).astype(int)
    eligible = observed_window_count.ge(sealed.minimum_available_windows)
    membership = pd.DataFrame(
        {
            "stay_id": matrix.index,
            "observed_window_count": observed_window_count.to_numpy(),
            "meets_min_observed_windows": eligible.to_numpy(),
            "included_in_clustering": eligible.to_numpy(),
            "exclusion_reason": np.where(
                eligible,
                "",
                "fewer_than_signed_minimum_owner_available_sofa2_windows",
            ),
        }
    )
    membership.to_csv(out_dir / "trajectory_membership.csv", index=False)
    model_matrix = matrix.loc[eligible].reset_index()
    if len(model_matrix) <= max(sealed.candidate_cluster_counts):
        raise ValueError(
            "eligible trajectory population is too small for candidate grid"
        )
    if model_matrix[list(sealed.representation_columns)].isna().all(axis=1).any():
        raise ValueError("an eligible stay has no model coordinate")
    representation_path = out_dir / "trajectory_representation.parquet"
    model_matrix.to_parquet(representation_path, index=False)
    feature_rows = []
    for column in sealed.representation_columns:
        observed_n = int(model_matrix[column].notna().sum())
        feature_rows.append(
            {
                "feature": column,
                "observed_n": observed_n,
                "missing_n": int(len(model_matrix) - observed_n),
                "missing_fraction": float(1.0 - observed_n / len(model_matrix)),
            }
        )
    feature_frame = pd.DataFrame(feature_rows)
    feature_frame.to_csv(out_dir / "feature_availability.csv", index=False)
    feature_frame.to_csv(out_dir / "feature_missingness_heatmap.csv", index=False)
    authority_binding = {
        "schema_version": sealed.schema_version,
        "protocol_content_sha256": sealed.protocol_content_sha256,
        "execution_contract_sha256": sealed.execution_contract_sha256,
    }
    schema = {
        "schema_version": "easyicu.trajectory_representation_schema/2",
        "id_column": "stay_id",
        "observation_family": list(sealed.coordinate_concepts),
        "observation_columns": list(sealed.representation_columns),
        "min_observed_windows": sealed.minimum_available_windows,
        "profile_columns": list(sealed.representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "signed_runtime_scientific_projection",
        "source_window_contract": {
            "start_hours": sealed.window_start_hours,
            "end_hours": sealed.window_end_hours,
            "grid_width_hours": sealed.grid_width_hours,
            "aggregation": sealed.aggregation,
        },
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
        "coordinate_scaling": sealed.scaling_payload,
        "evidence_state_policy": sealed.evidence_payload,
        "representation_columns": list(sealed.representation_columns),
        "frozen_population_n": len(model_matrix),
        "representation_sha256": _sha256(representation_path),
        "scientific_runtime_authority": authority_binding,
        "runtime_projection_sha256": runtime_projection_sha256,
        "descriptive_only_concepts": list(sealed.descriptive_only_concepts),
    }
    sealed.validate_representation_schema(schema)
    _write_json(out_dir / "trajectory_representation_schema.json", schema)
    window_manifest = {
        "schema_version": "easyicu.trajectory_window_manifest/1",
        "panel_product": "artifact:trajectory_representation",
        "families": [
            {
                "family": concept,
                "ordered_source_columns": [
                    {
                        "name": (
                            f"{concept}__h{start}_{start + sealed.grid_width_hours}"
                        ),
                        "window_start_hours": start,
                        "window_end_hours": start + sealed.grid_width_hours,
                    }
                    for start in range(
                        sealed.window_start_hours,
                        sealed.window_end_hours,
                        sealed.grid_width_hours,
                    )
                ],
            }
            for concept in sealed.coordinate_concepts
        ],
        "scientific_runtime_authority": authority_binding,
        "runtime_projection_sha256": runtime_projection_sha256,
    }
    _write_json(out_dir / "trajectory_window_manifest.json", window_manifest)
    state_counts = {
        str(key): int(value)
        for key, value in trajectory["evidence_state"].value_counts().items()
    }
    summary = {
        "status": "ok",
        "id_column": "stay_id",
        "observation_family": list(sealed.coordinate_concepts),
        "observation_columns": list(sealed.representation_columns),
        "min_observed_windows": sealed.minimum_available_windows,
        "profile_columns": list(sealed.representation_columns),
        "profile_summary_statistic": "mean",
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "signed_runtime_scientific_projection",
        "trailing_na_policy": schema["trailing_na_policy"],
        "coordinate_scaling": sealed.scaling_payload,
        "evidence_state_policy": sealed.evidence_payload,
        "representation_columns": list(sealed.representation_columns),
        "frozen_population_n": len(model_matrix),
        "eligible_n": int(eligible.sum()),
        "excluded_n": int((~eligible).sum()),
        "owner_evidence_state_counts": state_counts,
        "scientific_runtime_authority": authority_binding,
        "runtime_projection_sha256": runtime_projection_sha256,
        "representation_sha256": schema["representation_sha256"],
        "sofa_coordinate_columns": sofa_columns,
        "output_files": {
            "artifact:trajectory_representation": (
                "trajectory_representation.parquet"
            ),
            "table:trajectory_membership": "trajectory_membership.csv",
            "table:feature_availability": "feature_availability.csv",
            "manifest:trajectory_representation_schema": (
                "trajectory_representation_schema.json"
            ),
            "manifest:trajectory_window_manifest": (
                "trajectory_window_manifest.json"
            ),
        },
    }
    _write_json(out_dir / "step_summary.json", summary)
    return summary


__all__ = [
    "run_trajectory_scientific_representation",
    "trajectory_scientific_representation_executor_code",
    "trajectory_scientific_representation_executor_owns_step",
]
