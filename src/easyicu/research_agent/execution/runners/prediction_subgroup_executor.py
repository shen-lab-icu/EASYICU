"""Typed, digest-bound deterministic adapter for frozen prediction subgroups."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Mapping

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...canonical_json import sha256_bytes
from ...contracts.prediction_execution import PREDICTION_SCORES_PRODUCT
from ...contracts.prediction_subgroup import (
    PREDICTION_SUBGROUP_ACTION, PREDICTION_SUBGROUP_PRODUCT, PredictionSubgroupError,
    PredictionSubgroupSpec, prediction_subgroup_spec_for_step, prediction_subgroup_step_matches,
)
from ...contracts.prediction_subgroup_output import subgroup_performance_frame, subgroup_provenance
from ...methods.prediction_subgroup import bind_subgroup_membership, evaluate_prediction_subgroups
from ..step_executor_registry import StepExecutor
from .typed_input_binding import load_typed_input


def prediction_subgroup_executor_code(
    step: object, *, plausibility_scope: FlagOnlyPlausibilityScope | None = None,
) -> str:
    if not prediction_subgroup_step_matches(step):
        raise PredictionSubgroupError("prediction_subgroup_declaration_required", "the complete reviewed subgroup declaration is required")
    spec = prediction_subgroup_spec_for_step(step)
    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    code = textwrap.dedent(f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.prediction_subgroup_executor import run_prediction_subgroup_step
        summary = run_prediction_subgroup_step(
            spec={spec.model_dump(mode="json")!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r}, out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    """).strip()
    from .plausibility_receipt import host_plausibility_receipt_injected

    return host_plausibility_receipt_injected(code, scope=plausibility_scope, already_satisfied=False)


def prediction_subgroup_executor() -> StepExecutor:
    return StepExecutor(
        key="prediction_subgroup_performance",
        applicable=lambda c: c.step.scientific_action_id == PREDICTION_SUBGROUP_ACTION,
        owns=lambda c: prediction_subgroup_step_matches(c.step),
        render=lambda c: prediction_subgroup_executor_code(c.step, plausibility_scope=c.plausibility_scope),
        analysis_kind="prediction_subgroup_performance",
        selection_reason="reviewed_frozen_prediction_subgroups",
        progress_message="Evaluating frozen predictions in the declared validation subgroups",
        consumed_input_keys=lambda c: tuple(c.step.inputs),
        host_sealed_renderer=True,
    )


def run_prediction_subgroup_step(
    *, spec: PredictionSubgroupSpec | Mapping[str, Any], run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any], step_id: str, out_dir: Path,
) -> dict[str, Any]:
    """Consume only host-authorized artifacts and write aggregate results."""
    parsed = PredictionSubgroupSpec.model_validate(spec)
    manifest = dict(resolved_inputs) if isinstance(resolved_inputs, Mapping) else json.loads(Path(resolved_inputs).read_text("utf-8"))
    expected = {PREDICTION_SCORES_PRODUCT, parsed.subgroup_input_key}
    if manifest.get("step_id") != step_id or set(manifest.get("inputs", {})) != expected:
        raise PredictionSubgroupError("prediction_subgroup_input_binding_mismatch", "the manifest must bind exactly the reviewed score and membership inputs")
    scores = load_typed_input(
        input_key=PREDICTION_SCORES_PRODUCT, run_dir=run_dir, resolved_inputs=manifest,
        step_id=step_id, expected_declared_kind="table", expected_evidence_kind="table",
        require_consumption_contract=True, minimum_row_count=1,
    )
    membership = load_typed_input(
        input_key=parsed.subgroup_input_key, run_dir=run_dir, resolved_inputs=manifest,
        step_id=step_id, expected_declared_kind=parsed.subgroup_input_key.partition(":")[0], expected_evidence_kind="table",
        require_consumption_contract=True, minimum_row_count=1,
    )
    frame = bind_subgroup_membership(scores.frame, membership.frame, parsed)
    result = evaluate_prediction_subgroups(frame, parsed)
    table = subgroup_performance_frame(result)
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    provenance = subgroup_provenance(result, parsed, scores.sha256, membership.sha256)
    provenance_bytes = json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    summary = {
        "step_id": step_id, "status": "ok", "analysis_status": "ok",
        "analysis_family": "prediction", "method": "frozen_prediction_subgroup_performance",
        "deterministic_standard_analysis": "prediction_subgroup_performance",
        "authority_scope": "analysis_only", "paper_authorization_allowed": False,
        "source_inputs": [PREDICTION_SCORES_PRODUCT, parsed.subgroup_input_key],
        "input_bindings": [{"input_key": key, "loaded": True} for key in (PREDICTION_SCORES_PRODUCT, parsed.subgroup_input_key)],
        "output_files": {PREDICTION_SUBGROUP_PRODUCT: "subgroup_performance.csv"},
        "subgroup_performance": result.model_dump(mode="json"),
        "provenance": provenance,
        "performance_rows": len(table),
        "output_sha256": sha256_bytes(csv_bytes),
        "provenance_sha256": sha256_bytes(provenance_bytes),
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "subgroup_performance.csv").write_bytes(csv_bytes)
    (output / "subgroup_performance_provenance.json").write_bytes(provenance_bytes)
    (output / "step_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    return summary
