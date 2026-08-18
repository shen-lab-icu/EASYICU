"""Dependency-neutral host owner for deterministic prediction validation.

The execution incubator adapter and EvidenceStore authority bridge both call
this module.  Keeping exact-source parsing and recomputation here prevents the
authority layer from importing execution or maintaining a second parser.
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

from .contracts.prediction_validation import (
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
from .canonical_json import sha256_bytes
from .methods.prediction_validation import evaluate_binary_predictions


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


def _canonical_text_values(
    frame: pd.DataFrame,
    *,
    column: str,
    reason_code: PredictionValidationReason,
    role: str,
) -> pd.Series:
    if column not in frame.columns:
        _raise(
            reason_code,
            "lineage table is missing a required semantic coordinate",
            role=role,
            missing_columns=[column],
        )
    raw = frame[column]
    missing = raw.isna() | raw.map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    if bool(missing.any()):
        _raise(
            reason_code,
            "lineage table contains a missing semantic coordinate",
            role=role,
            column=column,
            row_count=int(missing.sum()),
        )
    values = raw.map(str)
    noncanonical = values.ne(values.str.strip())
    if bool(noncanonical.any()):
        _raise(
            reason_code,
            "lineage table coordinates must be whitespace-canonical",
            role=role,
            column=column,
            row_count=int(noncanonical.sum()),
        )
    return values


def _read_lineage_csv(
    *,
    source_path: Path,
    expected_source_sha256: str,
    role: str,
    reason_code: PredictionValidationReason,
) -> pd.DataFrame:
    try:
        frame, _ = _read_bound_csv(
            source_path,
            expected_source_sha256=expected_source_sha256,
        )
    except PredictionValidationError as error:
        raise PredictionValidationError(
            reason_code,
            "digest-bound lineage CSV could not be parsed",
            role=role,
            cause_reason_code=error.reason_code.value,
            cause_detail=error.detail,
        ) from error
    return frame


def _reconcile_cohort_subjects(
    prediction_frame: pd.DataFrame,
    cohort_frame: pd.DataFrame,
    spec: PredictionValidationSpec,
) -> None:
    reason = PredictionValidationReason.LINEAGE_COHORT_MISMATCH
    prediction_subjects = _canonical_text_values(
        prediction_frame,
        column=spec.subject_id_column,
        reason_code=reason,
        role="prediction_table",
    )
    cohort_subjects = _canonical_text_values(
        cohort_frame,
        column=spec.subject_id_column,
        reason_code=reason,
        role="cohort",
    )
    duplicates = cohort_subjects.duplicated(keep=False)
    if bool(duplicates.any()):
        _raise(
            reason,
            "cohort lineage requires exactly one row per prediction subject",
            role="cohort",
            duplicate_subject_count=int(cohort_subjects[duplicates].nunique()),
            duplicate_row_count=int(duplicates.sum()),
        )
    prediction_set = set(prediction_subjects)
    cohort_set = set(cohort_subjects)
    missing = sorted(prediction_set - cohort_set)
    unexpected = sorted(cohort_set - prediction_set)
    if missing or unexpected:
        _raise(
            reason,
            "cohort subjects do not exactly match prediction-table subjects",
            role="cohort",
            prediction_subject_count=len(prediction_set),
            cohort_subject_count=len(cohort_set),
            missing_subject_count=len(missing),
            unexpected_subject_count=len(unexpected),
            missing_subject_examples=missing[:10],
            unexpected_subject_examples=unexpected[:10],
        )


def _reconcile_split_assignments(
    prediction_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    spec: PredictionValidationSpec,
) -> None:
    reason = PredictionValidationReason.LINEAGE_SPLIT_MISMATCH
    prediction_subjects = _canonical_text_values(
        prediction_frame,
        column=spec.subject_id_column,
        reason_code=reason,
        role="prediction_table",
    )
    prediction_splits = _canonical_text_values(
        prediction_frame,
        column=spec.split_column,
        reason_code=reason,
        role="prediction_table",
    )
    split_subjects = _canonical_text_values(
        split_frame,
        column=spec.subject_id_column,
        reason_code=reason,
        role="split_assignment",
    )
    split_values = _canonical_text_values(
        split_frame,
        column=spec.split_column,
        reason_code=reason,
        role="split_assignment",
    )
    duplicate_rows = split_subjects.duplicated(keep=False)
    if bool(duplicate_rows.any()):
        _raise(
            reason,
            "split lineage requires exactly one assignment per prediction subject",
            role="split_assignment",
            duplicate_subject_count=int(split_subjects[duplicate_rows].nunique()),
            duplicate_row_count=int(duplicate_rows.sum()),
        )

    prediction_pairs = pd.DataFrame(
        {"subject": prediction_subjects, "split": prediction_splits}
    ).drop_duplicates()
    leaking = prediction_pairs["subject"].duplicated(keep=False)
    if bool(leaking.any()):
        _raise(
            reason,
            "prediction subjects do not have one canonical split assignment",
            role="prediction_table",
            subject_count=int(prediction_pairs.loc[leaking, "subject"].nunique()),
        )
    expected = dict(
        zip(prediction_pairs["subject"], prediction_pairs["split"], strict=True)
    )
    observed = dict(zip(split_subjects, split_values, strict=True))
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    mismatched = sorted(
        subject
        for subject in set(expected) & set(observed)
        if expected[subject] != observed[subject]
    )
    if missing or unexpected or mismatched:
        _raise(
            reason,
            "split assignments do not exactly match prediction-table subjects",
            role="split_assignment",
            prediction_subject_count=len(expected),
            split_subject_count=len(observed),
            missing_subject_count=len(missing),
            unexpected_subject_count=len(unexpected),
            mismatched_subject_count=len(mismatched),
            missing_subject_examples=missing[:10],
            unexpected_subject_examples=unexpected[:10],
            mismatched_subject_examples=[
                {
                    "subject_id": subject,
                    "prediction_split": expected[subject],
                    "lineage_split": observed[subject],
                }
                for subject in mismatched[:10]
            ],
        )


def _validated_runtime_identity(
    runtime_identity: PredictionValidationRuntimeIdentity | Mapping[str, Any],
) -> PredictionValidationRuntimeIdentity:
    try:
        runtime_input = (
            runtime_identity.model_dump(mode="python")
            if isinstance(runtime_identity, PredictionValidationRuntimeIdentity)
            else runtime_identity
        )
        return PredictionValidationRuntimeIdentity.model_validate(runtime_input)
    except ValidationError as error:
        raise PredictionValidationError(
            PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
            "host runtime identity is not schema-valid",
            error_count=error.error_count(),
        ) from error


def _seal_recomputed_receipt(
    receipt: PredictionValidationReceipt,
    runtime: PredictionValidationRuntimeIdentity,
) -> PredictionValidationHostValidationSeal:
    return PredictionValidationHostValidationSeal(
        receipt_sha256=prediction_validation_receipt_sha256(receipt),
        source_artifact_sha256=receipt.source.source_artifact_sha256,
        contract_sha256=receipt.contract_sha256,
        result_sha256=receipt.result_sha256,
        runtime_identity_sha256=prediction_validation_runtime_identity_sha256(runtime),
    )


def recompute_prediction_validation_analysis(
    *,
    prediction_path: Path,
    prediction_sha256: str,
    cohort_path: Path,
    cohort_sha256: str,
    split_path: Path,
    split_sha256: str,
    spec: PredictionValidationSpec | Mapping[str, Any],
    runtime_identity: PredictionValidationRuntimeIdentity | Mapping[str, Any],
) -> tuple[PredictionValidationReceipt, PredictionValidationHostValidationSeal]:
    """Recompute one analysis from exact bytes and reconcile subject lineage."""

    try:
        spec_input = (
            spec.model_dump(mode="python")
            if isinstance(spec, PredictionValidationSpec)
            else spec
        )
        parsed_spec = PredictionValidationSpec.model_validate(spec_input)
    except ValidationError as error:
        raise PredictionValidationError(
            PredictionValidationReason.LINEAGE_SCHEMA_INVALID,
            "prediction-validation specification is not schema-valid",
            error_count=error.error_count(),
        ) from error
    prediction_frame, source = _read_bound_csv(
        prediction_path,
        expected_source_sha256=prediction_sha256,
    )
    result = run_prediction_validation(prediction_frame, parsed_spec)
    receipt = PredictionValidationReceipt(
        source=source,
        contract_sha256=prediction_validation_spec_sha256(parsed_spec),
        result_sha256=prediction_validation_result_sha256(result),
        result=result,
    )
    cohort_frame = _read_lineage_csv(
        source_path=cohort_path,
        expected_source_sha256=cohort_sha256,
        role="cohort",
        reason_code=PredictionValidationReason.LINEAGE_COHORT_MISMATCH,
    )
    split_frame = _read_lineage_csv(
        source_path=split_path,
        expected_source_sha256=split_sha256,
        role="split_assignment",
        reason_code=PredictionValidationReason.LINEAGE_SPLIT_MISMATCH,
    )
    _reconcile_cohort_subjects(prediction_frame, cohort_frame, parsed_spec)
    _reconcile_split_assignments(prediction_frame, split_frame, parsed_spec)
    runtime = _validated_runtime_identity(runtime_identity)
    return receipt, _seal_recomputed_receipt(receipt, runtime)


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
    parsed_runtime = _validated_runtime_identity(runtime_identity)
    return _seal_recomputed_receipt(parsed_candidate, parsed_runtime)


__all__ = [
    "prediction_validation_receipt_findings",
    "prediction_validation_result_findings",
    "recompute_prediction_validation_analysis",
    "run_prediction_validation",
    "run_prediction_validation_csv",
    "seal_prediction_validation_receipt",
]
