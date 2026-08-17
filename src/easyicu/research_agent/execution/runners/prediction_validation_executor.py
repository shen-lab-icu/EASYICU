"""Experimental host adapter for deterministic prediction validation.

This adapter is intentionally not registered in production selection.  It
offers a direct incubator API and validates candidate results by recomputing
them from the same declared frame and contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pydantic import ValidationError

from ...contracts.prediction_validation import (
    PredictionValidationFinding,
    PredictionValidationReason,
    PredictionValidationResult,
    PredictionValidationSpec,
    prediction_validation_result_sha256,
)
from ...methods.prediction_validation import evaluate_binary_predictions


def run_prediction_validation(
    frame: pd.DataFrame,
    spec: PredictionValidationSpec | Mapping[str, Any],
) -> PredictionValidationResult:
    """Run the experimental deterministic owner through its typed boundary."""

    return evaluate_binary_predictions(frame, spec)


def _first_mismatch(
    expected: Any,
    observed: Any,
    *,
    path: str = "$",
) -> tuple[str, Any, Any] | None:
    if type(expected) is not type(observed):
        return path, expected, observed
    if isinstance(expected, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return f"{path}.keys", sorted(expected_keys), sorted(observed_keys)
        for key in sorted(expected_keys):
            mismatch = _first_mismatch(
                expected[key], observed[key], path=f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if len(expected) != len(observed):
            return f"{path}.length", len(expected), len(observed)
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=True)
        ):
            mismatch = _first_mismatch(
                expected_item, observed_item, path=f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    return None if expected == observed else (path, expected, observed)


def prediction_validation_result_findings(
    frame: pd.DataFrame,
    spec: PredictionValidationSpec | Mapping[str, Any],
    candidate: PredictionValidationResult | Mapping[str, Any],
) -> tuple[PredictionValidationFinding, ...]:
    """Return a stable finding if ``candidate`` differs from host recomputation."""

    try:
        parsed_candidate = PredictionValidationResult.model_validate(candidate)
    except ValidationError as error:
        return (
            PredictionValidationFinding(
                reason_code=PredictionValidationReason.RESULT_SCHEMA_INVALID,
                message="candidate prediction-validation result is not schema-valid",
                detail={"error_count": error.error_count()},
            ),
        )

    expected = run_prediction_validation(frame, spec)
    expected_payload = expected.model_dump(mode="json")
    candidate_payload = parsed_candidate.model_dump(mode="json")
    mismatch = _first_mismatch(expected_payload, candidate_payload)
    if mismatch is None:
        return ()
    path, expected_value, observed_value = mismatch
    return (
        PredictionValidationFinding(
            reason_code=PredictionValidationReason.RESULT_MISMATCH,
            message="candidate result does not match host recomputation",
            detail={
                "path": path,
                "expected": expected_value,
                "observed": observed_value,
                "expected_result_sha256": prediction_validation_result_sha256(expected),
                "observed_result_sha256": prediction_validation_result_sha256(
                    parsed_candidate
                ),
            },
        ),
    )


__all__ = [
    "prediction_validation_result_findings",
    "run_prediction_validation",
]
