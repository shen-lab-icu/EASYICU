"""Experimental host adapter for deterministic prediction validation.

This adapter is intentionally not registered in production selection.  It
offers a direct incubator API and validates candidate results by recomputing
them from the same declared frame and contract.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from ...contracts.prediction_validation import (
    PredictionValidationFinding,
    PredictionValidationError,
    PredictionValidationHostValidationSeal,
    PredictionValidationReason,
    PredictionValidationReceipt,
    PredictionValidationResult,
    PredictionValidationRuntimeIdentity,
    PredictionValidationSourceBinding,
    PredictionValidationSpec,
    prediction_validation_receipt_sha256,
    prediction_validation_result_sha256,
    prediction_validation_runtime_identity_sha256,
    prediction_validation_spec_sha256,
)
from ...canonical_json import sha256_bytes
from ...methods.prediction_validation import evaluate_binary_predictions


def run_prediction_validation(
    frame: pd.DataFrame,
    spec: PredictionValidationSpec | Mapping[str, Any],
) -> PredictionValidationResult:
    """Run the experimental deterministic owner through its typed boundary."""

    return evaluate_binary_predictions(frame, spec)


def _raise(
    reason_code: PredictionValidationReason,
    message: str,
    **detail: Any,
) -> None:
    raise PredictionValidationError(reason_code, message, **detail)


def _strict_csv_header(raw: bytes, *, source_name: str) -> tuple[str, ...]:
    try:
        decoded = raw.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded), strict=True), None)
    except (UnicodeDecodeError, csv.Error) as error:
        raise PredictionValidationError(
            PredictionValidationReason.SOURCE_READ_FAILED,
            "digest-bound prediction-validation CSV header could not be parsed",
            source_name=source_name,
        ) from error
    if not header:
        _raise(
            PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
            "prediction-validation CSV requires one header row",
            source_name=source_name,
        )
    columns = tuple(str(column) for column in header)
    duplicate_columns = sorted(
        {column for column in columns if columns.count(column) > 1}
    )
    noncanonical_columns = [
        column for column in columns if not column or column != column.strip()
    ]
    if duplicate_columns or noncanonical_columns:
        _raise(
            PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
            "prediction-validation CSV header must be canonical and unique",
            source_name=source_name,
            duplicate_columns=duplicate_columns,
            noncanonical_columns=noncanonical_columns,
        )
    return columns


def _read_bound_csv(
    source_path: Path,
    *,
    expected_source_sha256: str,
) -> tuple[pd.DataFrame, PredictionValidationSourceBinding]:
    path = Path(source_path)
    expected_digest = str(expected_source_sha256 or "").strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        _raise(
            PredictionValidationReason.SOURCE_DIGEST_INVALID,
            "expected source digest must be one lowercase SHA-256 value",
        )
    if not path.is_file() or path.suffix.lower() != ".csv":
        _raise(
            PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
            "prediction-validation source must be one regular CSV file",
            source_name=path.name,
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PredictionValidationError(
            PredictionValidationReason.SOURCE_READ_FAILED,
            "prediction-validation source bytes could not be read",
            source_name=path.name,
        ) from error
    if not raw:
        _raise(
            PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
            "prediction-validation source CSV is empty",
            source_name=path.name,
        )
    observed_digest = sha256_bytes(raw)
    if observed_digest != expected_digest:
        _raise(
            PredictionValidationReason.SOURCE_DIGEST_MISMATCH,
            "prediction-validation source digest does not match its authority",
            source_name=path.name,
            expected_sha256=expected_digest,
            observed_sha256=observed_digest,
        )
    header = _strict_csv_header(raw, source_name=path.name)
    try:
        frame = pd.read_csv(
            io.BytesIO(raw),
            sep=",",
            header=0,
            engine="c",
            encoding="utf-8-sig",
            encoding_errors="strict",
            keep_default_na=True,
            na_filter=True,
            low_memory=False,
            on_bad_lines="error",
            skip_blank_lines=False,
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise PredictionValidationError(
            PredictionValidationReason.SOURCE_READ_FAILED,
            "digest-bound prediction-validation CSV could not be parsed",
            source_name=path.name,
        ) from error
    if frame.empty:
        _raise(
            PredictionValidationReason.EMPTY_INPUT,
            "digest-bound prediction-validation CSV contains no rows",
            source_name=path.name,
        )
    parsed_columns = tuple(str(column) for column in frame.columns)
    if parsed_columns != header:
        _raise(
            PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
            "parsed prediction-validation columns do not match the raw header",
            source_name=path.name,
            header_columns=list(header),
            parsed_columns=list(parsed_columns),
        )
    binding = PredictionValidationSourceBinding(
        parser_version=pd.__version__,
        source_artifact_name=path.name,
        source_artifact_sha256=observed_digest,
        source_artifact_size_bytes=len(raw),
        source_row_count=int(len(frame)),
        source_columns=parsed_columns,
    )
    return frame, binding


def run_prediction_validation_csv(
    *,
    source_path: Path,
    expected_source_sha256: str,
    spec: PredictionValidationSpec | Mapping[str, Any],
) -> PredictionValidationReceipt:
    """Evaluate the exact bytes of one pre-authorized UTF-8 CSV artifact."""

    parsed_spec = PredictionValidationSpec.model_validate(spec)
    frame, source = _read_bound_csv(
        source_path,
        expected_source_sha256=expected_source_sha256,
    )
    result = run_prediction_validation(frame, parsed_spec)
    return PredictionValidationReceipt(
        source=source,
        contract_sha256=prediction_validation_spec_sha256(parsed_spec),
        result_sha256=prediction_validation_result_sha256(result),
        result=result,
    )


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
        candidate_input = (
            candidate.model_dump(mode="python")
            if isinstance(candidate, PredictionValidationResult)
            else candidate
        )
        parsed_candidate = PredictionValidationResult.model_validate(candidate_input)
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


def prediction_validation_receipt_findings(
    *,
    source_path: Path,
    expected_source_sha256: str,
    spec: PredictionValidationSpec | Mapping[str, Any],
    candidate: PredictionValidationReceipt | Mapping[str, Any],
) -> tuple[PredictionValidationFinding, ...]:
    """Validate a candidate receipt against exact source bytes and recomputation."""

    try:
        candidate_input = (
            candidate.model_dump(mode="python")
            if isinstance(candidate, PredictionValidationReceipt)
            else candidate
        )
        parsed_candidate = PredictionValidationReceipt.model_validate(candidate_input)
    except ValidationError as error:
        return (
            PredictionValidationFinding(
                reason_code=PredictionValidationReason.RECEIPT_SCHEMA_INVALID,
                message="candidate prediction-validation receipt is not schema-valid",
                detail={"error_count": error.error_count()},
            ),
        )
    try:
        expected = run_prediction_validation_csv(
            source_path=source_path,
            expected_source_sha256=expected_source_sha256,
            spec=spec,
        )
    except PredictionValidationError as error:
        return (
            PredictionValidationFinding(
                reason_code=error.reason_code,
                message="source-bound receipt could not be recomputed",
                detail=error.detail,
            ),
        )
    expected_payload = expected.model_dump(mode="json")
    candidate_payload = parsed_candidate.model_dump(mode="json")
    mismatch = _first_mismatch(expected_payload, candidate_payload)
    if mismatch is None:
        return ()
    path, expected_value, observed_value = mismatch
    return (
        PredictionValidationFinding(
            reason_code=PredictionValidationReason.RECEIPT_MISMATCH,
            message="candidate receipt does not match source-bound recomputation",
            detail={
                "path": path,
                "expected": expected_value,
                "observed": observed_value,
                "expected_receipt_sha256": prediction_validation_receipt_sha256(
                    expected
                ),
                "observed_receipt_sha256": prediction_validation_receipt_sha256(
                    parsed_candidate
                ),
            },
        ),
    )


def seal_prediction_validation_receipt(
    *,
    source_path: Path,
    expected_source_sha256: str,
    spec: PredictionValidationSpec | Mapping[str, Any],
    candidate: PredictionValidationReceipt | Mapping[str, Any],
    runtime_identity: PredictionValidationRuntimeIdentity | Mapping[str, Any],
) -> PredictionValidationHostValidationSeal:
    """Seal a receipt only after exact-source host recomputation succeeds."""

    findings = prediction_validation_receipt_findings(
        source_path=source_path,
        expected_source_sha256=expected_source_sha256,
        spec=spec,
        candidate=candidate,
    )
    if findings:
        finding = findings[0]
        _raise(
            finding.reason_code,
            "candidate receipt cannot receive a host validation seal",
            finding=finding.model_dump(mode="json"),
        )
    candidate_input = (
        candidate.model_dump(mode="python")
        if isinstance(candidate, PredictionValidationReceipt)
        else candidate
    )
    parsed_candidate = PredictionValidationReceipt.model_validate(candidate_input)
    try:
        runtime_input = (
            runtime_identity.model_dump(mode="python")
            if isinstance(runtime_identity, PredictionValidationRuntimeIdentity)
            else runtime_identity
        )
        parsed_runtime = PredictionValidationRuntimeIdentity.model_validate(
            runtime_input
        )
    except ValidationError as error:
        raise PredictionValidationError(
            PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
            "host runtime identity is not schema-valid",
            error_count=error.error_count(),
        ) from error
    return PredictionValidationHostValidationSeal(
        receipt_sha256=prediction_validation_receipt_sha256(parsed_candidate),
        source_artifact_sha256=parsed_candidate.source.source_artifact_sha256,
        contract_sha256=parsed_candidate.contract_sha256,
        result_sha256=parsed_candidate.result_sha256,
        runtime_identity_sha256=prediction_validation_runtime_identity_sha256(
            parsed_runtime
        ),
    )


__all__ = [
    "prediction_validation_receipt_findings",
    "prediction_validation_result_findings",
    "run_prediction_validation",
    "run_prediction_validation_csv",
    "seal_prediction_validation_receipt",
]
