"""Recompute subgroup output gates from the host's authorized input artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..canonical_json import canonical_sha256, sha256_bytes
from ..schema import ValidationFinding
from .artifact_consumption import load_typed_input
from .prediction_execution import PREDICTION_SCORES_PRODUCT
from .prediction_subgroup import (
    PREDICTION_SUBGROUP_ACTION, PREDICTION_SUBGROUP_PRODUCT, PredictionSubgroupResult,
    PredictionSubgroupSpec, prediction_subgroup_spec_for_step, prediction_subgroup_step_matches,
)


def subgroup_performance_frame(result: PredictionSubgroupResult) -> pd.DataFrame:
    """A deterministic aggregate-only serialization, shared with its output gate."""
    rows = []
    for cell in result.rows:
        row = cell.model_dump(mode="json")
        row["group_values"] = json.dumps(row["group_values"], ensure_ascii=False, sort_keys=True, allow_nan=False)
        row["reason_codes"] = json.dumps(row["reason_codes"], ensure_ascii=False)
        row.update(evaluation_split=result.evaluation_split, analysis_unit=result.analysis_unit, uncertainty=result.uncertainty)
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_provenance(
    result: PredictionSubgroupResult, spec: PredictionSubgroupSpec,
    prediction_sha256: str, membership_sha256: str,
) -> dict[str, Any]:
    return {
        "prediction_artifact_sha256": prediction_sha256,
        "membership_artifact_sha256": membership_sha256,
        "contract_sha256": result.contract_sha256,
        "result_sha256": canonical_sha256(result.model_dump(mode="json")),
        "evaluation_unit_ids_sha256": result.evaluation_unit_ids_sha256,
        "declaration": spec.model_dump(mode="json"),
        "model_refitted": False,
        "model_identity_verification": "a fixed score artifact is verified; independent model identity is not established",
        "calibration_definition": "joint logistic intercept and slope; probabilities clipped only for the calibration logit",
        "sample_status_definition": "declared count flags; they do not establish statistical precision or clinical adequacy",
        "uncertainty": "point_estimates_only; no subgroup difference tests or confidence intervals",
        "interpretation": result.interpretation,
    }


def prediction_subgroup_output_findings(
    *, step: object, step_summary: Mapping[str, Any], context: object | None = None,
    resolved_input_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    out_dir: Path | None = None,
) -> list[ValidationFinding]:
    """Verify results, CSV and provenance by replay; never trust a claimed hash."""
    if getattr(step, "scientific_action_id", None) != PREDICTION_SUBGROUP_ACTION:
        return []
    from ..methods.prediction_subgroup import bind_subgroup_membership, evaluate_prediction_subgroups

    try:
        if out_dir is None or resolved_input_bindings is None:
            raise ValueError("prediction_subgroup_validation_bindings_missing")
        if not prediction_subgroup_step_matches(step):
            raise ValueError("prediction_subgroup_declaration_invalid")
        spec = prediction_subgroup_spec_for_step(step)
        keys = (PREDICTION_SCORES_PRODUCT, spec.subgroup_input_key)
        if set(resolved_input_bindings) != set(keys):
            raise ValueError("prediction_subgroup_validation_bindings_mismatch")
        # Normal pipeline outputs are <run>/steps/<step>/outputs.
        output = Path(out_dir)
        root = output.parents[2]
        manifest = {"step_id": step.step_id, "inputs": dict(resolved_input_bindings)}
        bound = [load_typed_input(
            input_key=key, run_dir=root, resolved_inputs=manifest, step_id=step.step_id,
            expected_declared_kind=key.partition(":")[0], expected_evidence_kind="table",
            require_consumption_contract=True, minimum_row_count=1,
        ) for key in keys]
        frame = bind_subgroup_membership(bound[0].frame, bound[1].frame, spec)
        expected = evaluate_prediction_subgroups(frame, spec)
        payload = expected.model_dump(mode="json")
        provenance = subgroup_provenance(expected, spec, bound[0].sha256, bound[1].sha256)
        csv_bytes = subgroup_performance_frame(expected).to_csv(index=False).encode("utf-8")
        provenance_bytes = json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        required_summary = {
            "step_id": step.step_id, "status": "ok", "analysis_status": "ok",
            "analysis_family": "prediction", "method": "frozen_prediction_subgroup_performance",
            "deterministic_standard_analysis": "prediction_subgroup_performance",
            "authority_scope": "analysis_only", "paper_authorization_allowed": False,
            "source_inputs": list(keys), "input_bindings": [{"input_key": key, "loaded": True} for key in keys],
            "output_files": {PREDICTION_SUBGROUP_PRODUCT: "subgroup_performance.csv"},
            "subgroup_performance": payload, "provenance": provenance,
            "performance_rows": len(expected.rows), "output_sha256": sha256_bytes(csv_bytes),
            "provenance_sha256": sha256_bytes(provenance_bytes),
        }
        if any(canonical_sha256(step_summary.get(key)) != canonical_sha256(value) for key, value in required_summary.items()):
            raise ValueError("prediction_subgroup_summary_recomputation_mismatch")
        for filename, expected_bytes in (
            ("subgroup_performance.csv", csv_bytes),
            ("subgroup_performance_provenance.json", provenance_bytes),
        ):
            path = output / filename
            if path.is_symlink() or path.read_bytes() != expected_bytes:
                raise ValueError("prediction_subgroup_output_recomputation_mismatch")
        summary_path = output / "step_summary.json"
        if summary_path.is_symlink():
            raise ValueError("prediction_subgroup_summary_symlink")
        saved = json.loads(summary_path.read_text("utf-8"))
        if any(canonical_sha256(saved.get(key)) != canonical_sha256(value) for key, value in required_summary.items()):
            raise ValueError("prediction_subgroup_saved_summary_mismatch")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OSError, RuntimeError) as error:
        reason = getattr(error, "reason_code", None)
        if reason is None:
            reason = str(error) if str(error).startswith("prediction_subgroup_") else type(error).__name__
        return [ValidationFinding(
            validator="prediction_subgroup_output_contract", severity="error",
            message="Subgroup performance could not be verified against the reviewed declaration and bound input bytes.",
            detail={"step_id": getattr(step, "step_id", None), "reason": reason},
        )]
    return []
