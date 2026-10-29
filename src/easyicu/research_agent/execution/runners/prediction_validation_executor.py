"""Prediction-validation adapters with separate internal and external routes.

The historical probability-table API remains a compatibility re-export. The
reviewed external action owns a deterministic, digest-bound transport route.
"""

import json
import textwrap
from pathlib import Path
from typing import Mapping
from ...contracts.model_terms import DataPreparationPlan, PreparationRefusal, preparation_for_step
from ...contracts.prediction_validation import (
    EXTERNAL_ACTION, FROZEN_MODEL_INPUT, ExternalValidationError, ExternalValidationSpec,
    external_cohort_input, external_validation_spec_for_step, external_validation_step_matches,
)
from ...intake.typed_cohort_source import validation_source_from_binding
from ...methods.prediction_validation import run_external_validation
from ...methods.missing_data import ImputationRefusal
from ...reporting.supplement_package import write_external_validation_report
from ..step_executor_registry import StepExecutor
from .typed_input_binding import load_typed_input

from ...prediction_validation_owner import (
    prediction_validation_receipt_findings,
    prediction_validation_result_findings,
    run_prediction_validation,
    run_prediction_validation_csv,
    seal_prediction_validation_receipt,
)

__all__ = [
    "external_validation_executor",
    "prediction_external_executor_code",
    "run_external_validation_step",
    "prediction_validation_receipt_findings",
    "prediction_validation_result_findings",
    "run_prediction_validation",
    "run_prediction_validation_csv",
    "seal_prediction_validation_receipt",
]


def prediction_external_executor_code(step) -> str:
    if not external_validation_step_matches(step):
        raise ExternalValidationError("external_validation_declaration_required")
    return textwrap.dedent(f"""
        import json
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.prediction_validation_executor import run_external_validation_step
        summary = run_external_validation_step(
            spec={external_validation_spec_for_step(step).model_dump(mode="json")!r},
            data_preparation={preparation_for_step(step).model_dump(mode="json") if preparation_for_step(step) else None!r},
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r}, out_dir=Path(os.environ["STEP_OUT_DIR"]),
        )
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    """).strip()


def external_validation_executor() -> StepExecutor:
    return StepExecutor(
        key="prediction_external_validation",
        applicable=lambda c: c.step.scientific_action_id == EXTERNAL_ACTION,
        owns=lambda c: external_validation_step_matches(c.step),
        render=lambda c: prediction_external_executor_code(c.step),
        analysis_kind="prediction_external_validation",
        selection_reason="reviewed_frozen_model_external_validation",
        progress_message="Validating the frozen model in the reviewed external databases",
        consumed_input_keys=lambda c: tuple(c.step.inputs),
    )


def run_external_validation_step(
    *, spec: ExternalValidationSpec | Mapping, run_dir: Path,
    resolved_inputs: Path | Mapping, step_id: str, out_dir: Path,
    data_preparation: DataPreparationPlan | Mapping | None = None,
) -> dict:
    preparation = DataPreparationPlan.model_validate(data_preparation) if data_preparation is not None else None
    if preparation is not None:
        if preparation.missing_data is not None:
            raise ImputationRefusal(
                "mi_external_validation_model_unavailable",
                "The frozen transport model contains development medians, not fitted development MI conditionals; a validation cohort cannot fit an imputer",
            )
        raise PreparationRefusal(
            "preparation_external_frozen_pipeline", "External transport must retain the frozen development preprocessing",
        )
    spec = ExternalValidationSpec.model_validate(spec)
    manifest = dict(resolved_inputs) if isinstance(resolved_inputs, Mapping) else json.loads(resolved_inputs.read_text("utf-8"))
    expected_inputs = {FROZEN_MODEL_INPUT, *(external_cohort_input(db) for db in spec.validation_databases)}
    if manifest.get("step_id") != step_id or set(manifest.get("inputs", {})) != expected_inputs:
        raise ExternalValidationError("external_validation_input_binding_mismatch")
    # This is an aggregate, one-row portable model, never a patient table.
    frozen = load_typed_input(
        input_key=FROZEN_MODEL_INPUT, run_dir=run_dir, resolved_inputs=manifest,
        step_id=step_id, expected_declared_kind="table", expected_evidence_kind="table",
        expected_columns=("model_json",), require_consumption_contract=True, minimum_row_count=1,
    )
    if len(frozen.frame) != 1:
        raise ExternalValidationError("frozen_model_table_invalid")
    payload = str(frozen.frame.iloc[0]["model_json"]).encode("utf-8")

    def sources(model):
        return tuple(validation_source_from_binding(
            database=db, model=model, input_key=external_cohort_input(db), run_dir=run_dir,
            resolved_inputs=manifest, step_id=step_id,
        ) for db in spec.validation_databases)

    result = run_external_validation(model_bytes=payload, spec=spec, sources=sources)
    return write_external_validation_report(result, out_dir)
