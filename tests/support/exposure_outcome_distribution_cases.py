"""Shared exposure-outcome distribution cases: a drug exposure and a readmission.

The executor and renderer suites, and the published-bound rounding test,
build their tables with the real producer from these helpers.  The case is
deliberately not the benchmark item.  Synthetic data only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
    run_exposure_outcome_distribution_from_env,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_render import (
    run_exposure_outcome_distribution_figure,
)

EXPOSURE = "anticoagulant_exposed"
OUTCOME = "readmitted_30d"
EXECUTOR_STEP_ID = "03_drug_readmission_distribution"
RENDER_STEP_ID = "04_drug_readmission_distribution_figure"
INPUT_KEY = "table:exposure_outcome_distribution"
PRODUCT = "exposure_overview"

EXECUTOR_SPEC = {
    "exposure": EXPOSURE,
    "exposure_levels": [0, 1],
    "outcome": OUTCOME,
    "outcome_levels": [0, 1],
    "outcome_positive_value": 1,
    "level_match_policy": "exact_typed",
    "denominator_policy": "all_declared_rows",
    "missing_outcome_policy": "structural_absence_is_non_event",
    "confidence_level": 0.95,
}


def executor_cohort(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    """Write a digest-bound cohort and its resolved-inputs manifest."""

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / EXECUTOR_STEP_ID / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    manifest = {
        "step_id": EXECUTOR_STEP_ID,
        "inputs": {
            "artifact:analysis_cohort": {
                "relative_path": "cohort.parquet",
                "sha256": digest,
                "declared_kind": "artifact",
                "product": "analysis_cohort",
                "evidence_id": "ev-cohort",
                "identity_row": {
                    "input_key": "artifact:analysis_cohort",
                    "declared_kind": "artifact",
                    "product": "analysis_cohort",
                    "evidence_id": "ev-cohort",
                    "sha256": digest,
                },
                "product_contract": {
                    "columns": list(frame.columns),
                    "row_count": int(len(frame)),
                },
            }
        },
    }
    manifest_path = run_dir / "resolved_inputs.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, out_dir


def run_executor(monkeypatch, tmp_path: Path, frame: pd.DataFrame, **spec_updates) -> dict:
    run_dir, out_dir = executor_cohort(tmp_path, frame)
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv(
        "EASYICU_RESOLVED_INPUTS_JSON", str(run_dir / "resolved_inputs.json")
    )
    return run_exposure_outcome_distribution_from_env(
        spec_payload={**EXECUTOR_SPEC, **spec_updates},
        typed_cohort_input="artifact:analysis_cohort",
    )


def produced_table(
    tmp_path: Path,
    monkeypatch,
    *,
    frame: pd.DataFrame | None = None,
    spec_updates: dict | None = None,
    exposure: str = EXPOSURE,
    outcome: str = OUTCOME,
) -> Path:
    """Build the product with the real producer, not a hand-written fixture."""

    if frame is None:
        frame = pd.DataFrame(
            {
                exposure: [1] * 10 + [0] * 10,
                outcome: (
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, None]
                    + [1, 0, 0, 0, 0, 0, 0, 0, 0, None]
                ),
            }
        )
    parent = tmp_path / "parent"
    parent_out = parent / "steps" / "03_parent" / "outputs"
    parent_out.mkdir(parents=True)
    cohort = parent / "cohort.parquet"
    frame.to_parquet(cohort, index=False)
    digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    (parent / "resolved_inputs.json").write_text(
        json.dumps(
            {
                "step_id": "03_parent",
                "inputs": {
                    "artifact:analysis_cohort": {
                        "relative_path": "cohort.parquet",
                        "sha256": digest,
                        "declared_kind": "artifact",
                        "product": "analysis_cohort",
                        "product_contract": {
                            "columns": list(frame.columns),
                            "row_count": int(len(frame)),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("STEP_OUT_DIR", str(parent_out))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(parent))
    monkeypatch.setenv(
        "EASYICU_RESOLVED_INPUTS_JSON", str(parent / "resolved_inputs.json")
    )
    run_exposure_outcome_distribution_from_env(
        spec_payload={
            "exposure": exposure,
            "exposure_levels": [0, 1],
            "outcome": outcome,
            "outcome_levels": [0, 1],
            "outcome_positive_value": 1,
            "level_match_policy": "exact_typed",
            "denominator_policy": "all_declared_rows",
            "missing_outcome_policy": "structural_absence_is_non_event",
            "confidence_level": 0.95,
            **(spec_updates or {}),
        },
        typed_cohort_input="artifact:analysis_cohort",
    )
    return parent_out / "exposure_outcome_distribution.csv"


def bound_table(
    tmp_path: Path, table: Path, *, rows: int | None = None
) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    bound = run_dir / "inputs" / "exposure_outcome_distribution.csv"
    bound.write_bytes(table.read_bytes())
    digest = hashlib.sha256(bound.read_bytes()).hexdigest()
    frame = pd.read_csv(bound)
    manifest = {
        "step_id": RENDER_STEP_ID,
        "inputs": {
            INPUT_KEY: {
                "relative_path": "inputs/exposure_outcome_distribution.csv",
                "sha256": digest,
                "declared_kind": "table",
                "evidence_kind": "table",
                "product": "exposure_outcome_distribution",
                "evidence_id": "ev-distribution",
                "identity_row": {
                    "input_key": INPUT_KEY,
                    "declared_kind": "table",
                    "product": "exposure_outcome_distribution",
                    "evidence_id": "ev-distribution",
                    "sha256": digest,
                },
                "product_contract": {
                    "columns": list(EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS),
                    "row_count": rows if rows is not None else int(len(frame)),
                },
                "consumption_contract": {
                    "input_key": INPUT_KEY,
                    "mode": "all_rows",
                    "artifact_sha256": digest,
                },
            }
        },
    }
    return run_dir, manifest


def render(run_dir: Path, manifest: dict, out_dir: Path):
    return run_exposure_outcome_distribution_figure(
        out_dir=out_dir,
        run_dir=run_dir,
        resolved_inputs=manifest,
        step_id=RENDER_STEP_ID,
        figure_product=PRODUCT,
    )
