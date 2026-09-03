from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.figures import FigureSourceDataValidator
from easyicu.research_agent.execution.runners.trajectory_selection_figure_executor import (
    run_trajectory_selection_figure,
)


def _binding(
    run_dir: Path,
    *,
    step_id: str,
    input_key: str,
    frame: pd.DataFrame,
) -> dict:
    product = input_key.split(":", 1)[1]
    path = run_dir / f"{product}.csv"
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = {
        "relative_path": path.name,
        "sha256": digest,
        "evidence_id": f"table_{product}",
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "produced_by_step": f"producer_{product}",
        "product_contract": {
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        "consumption_contract": {
            "input_key": input_key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
    }
    binding["identity_row"] = {
        "input_key": input_key,
        "declared_kind": "table",
        "product": product,
        "evidence_id": binding["evidence_id"],
        "sha256": digest,
    }
    return binding


def _selection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_clusters": [2, 3, 4],
            "bic": [300.0, 200.0, 100.0],
            "aic": [280.0, 180.0, 80.0],
            "final_log_likelihood": [-120.0, -70.0, -20.0],
            "parameter_count": [20, 20, 20],
            "selected": [False, False, True],
            "aic_minimum": [False, False, True],
            "upper_boundary": [False, False, True],
            "scientific_status": ["failed_closed"] * 3,
            "reason_code": ["NO_INTERIOR_OPTIMUM"] * 3,
            "reportable_result": [
                "no_interior_solution_in_prespecified_candidate_range"
            ]
            * 3,
        }
    )


def _availability() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["resp__h0_12", "resp__h12_24", "lact__h0_12", "lact__h12_24"],
            "observed_n": [80, 60, 90, 70],
            "missing_n": [20, 40, 10, 30],
            "missing_fraction": [0.2, 0.4, 0.1, 0.3],
        }
    )


def test_failed_closed_selection_renders_a_bound_diagnostic_without_labels(
    tmp_path: Path,
) -> None:
    step_id = "trajectory_selection_figure"
    bindings = {
        "table:trajectory_candidate_selection": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:trajectory_candidate_selection",
            frame=_selection(),
        ),
        "table:feature_availability": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:feature_availability",
            frame=_availability(),
        ),
    }

    summary = run_trajectory_selection_figure(
        out_dir=tmp_path / "figure",
        run_dir=tmp_path,
        resolved_inputs={"step_id": step_id, "inputs": bindings},
        step_id=step_id,
    )

    assert summary["status"] == "ok"
    assert summary["scientific_status"] == "failed_closed"
    assert summary["reason_code"] == "NO_INTERIOR_OPTIMUM"
    selection_source = pd.read_csv(
        tmp_path / "figure" / "trajectory_selection_bic_source_data.csv"
    )
    assert selection_source["source_row_index"].tolist() == [0, 1, 2]
    trace = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=selection_source,
        source_path=(
            tmp_path / "figure" / "trajectory_selection_bic_source_data.csv"
        ),
        upstream_path=tmp_path / "trajectory_candidate_selection.csv",
    )
    assert trace["ok"], trace
    availability_source = pd.read_csv(
        tmp_path
        / "figure"
        / "trajectory_selection_availability_source_data.csv"
    )
    assert list(availability_source.columns) == [
        *_availability().columns,
        "source_table",
        "source_step_id",
    ]
    assert set(availability_source["source_table"]) == {"feature_availability.csv"}
    assert set(availability_source["source_step_id"]) == {
        "producer_feature_availability"
    }
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (
            tmp_path / "figure" / f"trajectory_selection_diagnostics.{suffix}"
        ).exists()
    contract = json.loads(
        (
            tmp_path
            / "figure"
            / "trajectory_selection_diagnostics.figure_contract.json"
        ).read_text("utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "phenotype_structure",
        "data_quality",
    ]
    assert "no interior solution" in contract["panels"][0]["claim"].lower()
    assert "aic" in contract["panels"][0]["claim"].lower()
    assert "candidate labels are not displayed" in contract["statistics_note"].lower()


def test_availability_arithmetic_drift_fails_closed(tmp_path: Path) -> None:
    step_id = "trajectory_selection_figure"
    availability = _availability()
    availability.loc[0, "missing_fraction"] = 0.9
    bindings = {
        "table:trajectory_candidate_selection": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:trajectory_candidate_selection",
            frame=_selection(),
        ),
        "table:feature_availability": _binding(
            tmp_path,
            step_id=step_id,
            input_key="table:feature_availability",
            frame=availability,
        ),
    }

    with pytest.raises(ValueError, match="missing fractions"):
        run_trajectory_selection_figure(
            out_dir=tmp_path / "figure",
            run_dir=tmp_path,
            resolved_inputs={"step_id": step_id, "inputs": bindings},
            step_id=step_id,
        )
