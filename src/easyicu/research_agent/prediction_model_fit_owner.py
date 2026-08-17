"""Experimental owner for one fixed, train-only binary prediction model.

The fitting route consumes only a host-issued :class:`LoadedTypedInput`.  It
fits numeric median imputation, standardization, and L2 logistic regression on
the declared training subjects, applies that frozen state to every row, and
seals the model JSON plus prediction CSV in one immutable bundle.  The same
algorithm owner can re-fit a persisted consumed-column projection for an
authority layer, but that computation-only route registers nothing.  This
module chooses no cohort, split, feature, outcome, threshold, or tuning
coordinate and grants no Planner, EvidenceStore, claim, or paper authority.
"""

from __future__ import annotations

import csv
import importlib.metadata
import io
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from pydantic import ValidationError
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .authority.typed_input_receipt import (
    TypedInputConsumptionReceipt,
    TypedInputRowIdentity,
    typed_input_row_identity_sha256,
)
from .authority.typed_input_sdk import LoadedTypedInput
from .canonical_json import canonical_sha256, sha256_bytes
from .contracts.prediction_model_fit import (
    PredictionLogisticArtifact,
    PredictionModelArtifact,
    PredictionModelFitError,
    PredictionModelFitReason,
    PredictionModelFitReceipt,
    PredictionModelFitSpec,
    PredictionPackageVersion,
    PredictionPreprocessingArtifact,
    prediction_model_artifact_bytes,
    prediction_model_fit_receipt_sha256,
    prediction_model_fit_spec_sha256,
)


_CONSTRUCTION_TOKEN = object()
_ISSUER = "easyicu.prediction_model_fit_owner"


def _raise(
    reason_code: PredictionModelFitReason,
    message: str,
    **detail: Any,
) -> None:
    raise PredictionModelFitError(reason_code, message, **detail)


def _parse_spec(
    spec: PredictionModelFitSpec | Mapping[str, Any],
) -> PredictionModelFitSpec:
    try:
        return PredictionModelFitSpec.model_validate(spec)
    except ValidationError as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.SPEC_INVALID,
            "prediction-model fit declaration is invalid",
            errors=error.errors(include_url=False),
        ) from error


def _canonical_text_column(
    frame: pd.DataFrame,
    *,
    column: str,
    role: str,
    reason_code: PredictionModelFitReason,
) -> pd.Series:
    raw = frame[column]
    missing = raw.isna()
    if bool(missing.any()):
        _raise(
            reason_code,
            "prediction-model coordinate contains missing values",
            role=role,
            column=column,
            row_count=int(missing.sum()),
        )
    values = raw.map(str)
    noncanonical = values.eq("") | values.ne(values.str.strip())
    if bool(noncanonical.any()):
        _raise(
            reason_code,
            "prediction-model coordinate must be canonical non-empty text",
            role=role,
            column=column,
            row_count=int(noncanonical.sum()),
        )
    return values


@dataclass(frozen=True, slots=True)
class _PreparedInput:
    unit_ids: pd.Series
    subject_ids: pd.Series
    splits: pd.Series
    outcomes: np.ndarray
    features: pd.DataFrame
    training_mask: np.ndarray
    evaluation_mask: np.ndarray


def _validated_source_receipt(
    source_input: LoadedTypedInput,
) -> TypedInputConsumptionReceipt:
    if not isinstance(source_input, LoadedTypedInput):
        _raise(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "model fitting requires one host-issued LoadedTypedInput",
        )
    try:
        receipt = TypedInputConsumptionReceipt.model_validate(
            source_input.receipt.model_dump(mode="json")
        )
    except (AttributeError, ValidationError) as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "typed-input receipt is invalid",
        ) from error
    if receipt != source_input.receipt:
        _raise(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "typed-input receipt changed during validation",
        )
    return receipt


def _numeric_outcome(frame: pd.DataFrame, *, column: str) -> np.ndarray:
    series = frame[column]
    if (
        not is_numeric_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
        or is_complex_dtype(series.dtype)
    ):
        _raise(
            PredictionModelFitReason.OUTCOME_INVALID,
            "binary outcome must use a real numeric dtype",
            column=column,
            dtype=str(series.dtype),
        )
    try:
        values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.OUTCOME_INVALID,
            "binary outcome could not be represented numerically",
            column=column,
        ) from error
    if not bool(np.isfinite(values).all()) or not bool(
        np.isin(values, (0.0, 1.0)).all()
    ):
        _raise(
            PredictionModelFitReason.OUTCOME_INVALID,
            "binary outcome must contain only finite 0 and 1 values",
            column=column,
        )
    return values.astype(np.int64, copy=False)


def _numeric_features(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    training_mask: np.ndarray,
) -> pd.DataFrame:
    arrays: list[np.ndarray] = []
    for column in feature_columns:
        series = frame[column]
        if (
            not is_numeric_dtype(series.dtype)
            or is_bool_dtype(series.dtype)
            or is_complex_dtype(series.dtype)
        ):
            _raise(
                PredictionModelFitReason.FEATURE_NONNUMERIC,
                "prediction features must use real numeric dtypes",
                column=column,
                dtype=str(series.dtype),
            )
        try:
            values = series.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as error:
            raise PredictionModelFitError(
                PredictionModelFitReason.FEATURE_NONNUMERIC,
                "prediction feature could not be represented numerically",
                column=column,
            ) from error
        if bool(np.isinf(values).any()):
            _raise(
                PredictionModelFitReason.FEATURE_NONFINITE,
                "prediction features may be missing but cannot be infinite",
                column=column,
                row_count=int(np.isinf(values).sum()),
            )
        if bool(np.isnan(values[training_mask]).all()):
            _raise(
                PredictionModelFitReason.FEATURE_ALL_MISSING_TRAIN,
                "a feature is entirely missing in the training subjects",
                column=column,
            )
        arrays.append(values)
    return pd.DataFrame(
        np.column_stack(arrays),
        columns=list(feature_columns),
        index=frame.index,
    )


def _validated_persisted_source_receipt(
    value: TypedInputConsumptionReceipt | Mapping[str, Any],
) -> TypedInputConsumptionReceipt:
    payload = (
        value.model_dump(mode="python")
        if isinstance(value, TypedInputConsumptionReceipt)
        else value
    )
    try:
        return TypedInputConsumptionReceipt.model_validate(payload)
    except ValidationError as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "persisted model source receipt is invalid",
        ) from error


def _prepare_frame(
    *,
    frame: pd.DataFrame,
    receipt: TypedInputConsumptionReceipt,
    spec: PredictionModelFitSpec,
) -> _PreparedInput:
    if not isinstance(receipt.row_identity, TypedInputRowIdentity):
        _raise(
            PredictionModelFitReason.SOURCE_IDENTITY_MISMATCH,
            "model fitting requires an ordered row-identity contract",
        )
    if receipt.row_identity.column != spec.unit_id_column:
        _raise(
            PredictionModelFitReason.SOURCE_IDENTITY_MISMATCH,
            "declared unit identity is not the typed-input row identity",
            declared_unit_id_column=spec.unit_id_column,
            source_row_identity_column=receipt.row_identity.column,
        )
    if frame.empty:
        _raise(
            PredictionModelFitReason.EMPTY_INPUT,
            "prediction-model source contains no rows",
        )
    if len(frame) != receipt.row_identity.row_count:
        _raise(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "typed-input row count changed before model fitting",
            receipt_row_count=receipt.row_identity.row_count,
            observed_row_count=int(len(frame)),
        )
    required = {
        spec.unit_id_column,
        spec.subject_id_column,
        spec.split_column,
        spec.outcome_column,
        *spec.feature_columns,
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        _raise(
            PredictionModelFitReason.MISSING_COLUMNS,
            "prediction-model source is missing declared columns",
            missing_columns=missing_columns,
        )

    unit_ids = _canonical_text_column(
        frame,
        column=spec.unit_id_column,
        role="unit_id",
        reason_code=PredictionModelFitReason.IDENTITY_INVALID,
    )
    if (
        typed_input_row_identity_sha256(frame[spec.unit_id_column])
        != receipt.row_identity.sha256
    ):
        _raise(
            PredictionModelFitReason.SOURCE_IDENTITY_MISMATCH,
            "persisted unit identity does not match the typed-input receipt",
        )
    subject_ids = _canonical_text_column(
        frame,
        column=spec.subject_id_column,
        role="subject_id",
        reason_code=PredictionModelFitReason.IDENTITY_INVALID,
    )
    splits = _canonical_text_column(
        frame,
        column=spec.split_column,
        role="split",
        reason_code=PredictionModelFitReason.SPLIT_INVALID,
    )
    duplicate_units = sorted(set(unit_ids.loc[unit_ids.duplicated(keep=False)]))
    if duplicate_units:
        _raise(
            PredictionModelFitReason.DUPLICATE_UNIT,
            "unit identity must be unique",
            duplicate_units=duplicate_units[:20],
            duplicate_unit_count=len(duplicate_units),
        )

    allowed_splits = {spec.training_split, spec.evaluation_split}
    unexpected_splits = sorted(set(splits) - allowed_splits)
    if unexpected_splits:
        _raise(
            PredictionModelFitReason.SPLIT_INVALID,
            "v1 model fitting accepts only the declared train and evaluation splits",
            unexpected_splits=unexpected_splits,
        )
    training_mask = splits.eq(spec.training_split).to_numpy(dtype=bool)
    evaluation_mask = splits.eq(spec.evaluation_split).to_numpy(dtype=bool)
    if not bool(training_mask.any()):
        _raise(
            PredictionModelFitReason.TRAINING_SPLIT_MISSING,
            "declared training split contains no subjects",
            training_split=spec.training_split,
        )
    if not bool(evaluation_mask.any()):
        _raise(
            PredictionModelFitReason.EVALUATION_SPLIT_MISSING,
            "declared evaluation split contains no subjects",
            evaluation_split=spec.evaluation_split,
        )

    subject_split = pd.DataFrame({"subject": subject_ids, "split": splits})
    crossing = (
        subject_split.drop_duplicates()
        .groupby("subject", sort=False)["split"]
        .nunique()
    )
    leaked_subjects = sorted(crossing.index[crossing.gt(1)].astype(str))
    if leaked_subjects:
        _raise(
            PredictionModelFitReason.SUBJECT_SPLIT_LEAKAGE,
            "one or more subjects cross training and evaluation splits",
            leaked_subjects=leaked_subjects[:20],
            leaked_subject_count=len(leaked_subjects),
        )
    repeated_subjects = sorted(set(subject_ids.loc[subject_ids.duplicated(keep=False)]))
    if repeated_subjects:
        _raise(
            PredictionModelFitReason.SUBJECT_UNIT_NOT_UNIQUE,
            "subject-level v1 requires exactly one row per subject",
            repeated_subjects=repeated_subjects[:20],
            repeated_subject_count=len(repeated_subjects),
        )

    outcomes = _numeric_outcome(frame, column=spec.outcome_column)
    training_classes = set(outcomes[training_mask].tolist())
    if training_classes != {0, 1}:
        _raise(
            PredictionModelFitReason.TRAINING_SINGLE_CLASS,
            "training subjects must contain both binary outcome classes",
            observed_classes=sorted(training_classes),
        )
    features = _numeric_features(
        frame,
        feature_columns=spec.feature_columns,
        training_mask=training_mask,
    )
    return _PreparedInput(
        unit_ids=unit_ids,
        subject_ids=subject_ids,
        splits=splits,
        outcomes=outcomes,
        features=features,
        training_mask=training_mask,
        evaluation_mask=evaluation_mask,
    )


def _prepare_input(
    *,
    source_input: LoadedTypedInput,
    spec: PredictionModelFitSpec,
) -> tuple[_PreparedInput, TypedInputConsumptionReceipt]:
    receipt = _validated_source_receipt(source_input)
    return (
        _prepare_frame(
            frame=source_input.to_pandas(),
            receipt=receipt,
            spec=spec,
        ),
        receipt,
    )


def _fit_train_only(
    prepared: _PreparedInput,
    spec: PredictionModelFitSpec,
) -> tuple[
    PredictionPreprocessingArtifact,
    PredictionLogisticArtifact,
    np.ndarray,
]:
    training_features = prepared.features.loc[prepared.training_mask]
    training_outcomes = prepared.outcomes[prepared.training_mask]
    imputer = SimpleImputer(strategy="median", copy=True)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    estimator = LogisticRegression(
        penalty="l2",
        C=spec.regularization_c,
        fit_intercept=True,
        class_weight=None,
        solver="lbfgs",
        max_iter=spec.max_iter,
        tol=spec.tolerance,
    )
    try:
        train_imputed = imputer.fit_transform(training_features)
        all_imputed = imputer.transform(prepared.features)
        train_scaled = scaler.fit_transform(train_imputed)
        all_scaled = scaler.transform(all_imputed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(train_scaled, training_outcomes)
        convergence_warnings = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        if convergence_warnings:
            _raise(
                PredictionModelFitReason.FIT_FAILED,
                "L2 logistic regression did not converge",
                warning_count=len(convergence_warnings),
            )
        classes = tuple(int(value) for value in estimator.classes_.tolist())
        if classes != (0, 1):
            _raise(
                PredictionModelFitReason.FIT_FAILED,
                "fitted binary class order is unsupported",
                classes=list(classes),
            )
        probabilities = estimator.predict_proba(all_scaled)[:, 1]
    except PredictionModelFitError:
        raise
    except (FloatingPointError, OverflowError, TypeError, ValueError) as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.FIT_FAILED,
            "fixed preprocessing or L2 logistic regression could not be fitted",
        ) from error

    numeric_state = np.concatenate(
        [
            np.asarray(imputer.statistics_, dtype=np.float64),
            np.asarray(scaler.mean_, dtype=np.float64),
            np.asarray(scaler.scale_, dtype=np.float64),
            np.asarray(estimator.coef_, dtype=np.float64).reshape(-1),
            np.asarray(estimator.intercept_, dtype=np.float64),
            np.asarray(probabilities, dtype=np.float64),
        ]
    )
    if not bool(np.isfinite(numeric_state).all()):
        _raise(
            PredictionModelFitReason.PREDICTION_INVALID,
            "fitted state or probabilities contain non-finite values",
        )
    if not bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all()):
        _raise(
            PredictionModelFitReason.PREDICTION_INVALID,
            "predicted probabilities fall outside [0, 1]",
        )

    preprocessing = PredictionPreprocessingArtifact(
        strategy="numeric_median_then_standardize",
        fit_scope="training_subjects_only",
        feature_columns=spec.feature_columns,
        medians=tuple(float(value) for value in imputer.statistics_),
        means=tuple(float(value) for value in scaler.mean_),
        scales=tuple(float(value) for value in scaler.scale_),
    )
    model = PredictionLogisticArtifact(
        estimator="logistic_regression_l2",
        solver="lbfgs",
        fit_scope="training_subjects_only",
        feature_columns=spec.feature_columns,
        classes=(0, 1),
        coefficients=tuple(float(value) for value in estimator.coef_.reshape(-1)),
        intercept=float(estimator.intercept_[0]),
        regularization_c=spec.regularization_c,
        fit_intercept=True,
        class_weight=None,
        max_iter=spec.max_iter,
        tolerance=spec.tolerance,
        n_iter=int(estimator.n_iter_[0]),
    )
    return preprocessing, model, np.asarray(probabilities, dtype=np.float64)


def _prediction_csv_bytes(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...],
) -> bytes:
    if tuple(str(column) for column in frame.columns) != expected_columns:
        _raise(
            PredictionModelFitReason.BUNDLE_INVALID,
            "prediction payload columns do not match the receipt",
        )
    if len(expected_columns) != 5:
        _raise(
            PredictionModelFitReason.BUNDLE_INVALID,
            "prediction payload requires five fixed columns",
        )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(expected_columns)
    try:
        for unit_id, subject_id, split, outcome, probability in frame.itertuples(
            index=False,
            name=None,
        ):
            writer.writerow(
                (
                    str(unit_id),
                    str(subject_id),
                    str(split),
                    str(int(outcome)),
                    format(float(probability), ".17g"),
                )
            )
    except (TypeError, ValueError, OverflowError) as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.BUNDLE_INVALID,
            "prediction payload cannot be serialized canonically",
        ) from error
    return buffer.getvalue().encode("utf-8")


def _source_projection_csv_bytes(
    prepared: _PreparedInput,
    spec: PredictionModelFitSpec,
) -> bytes:
    columns = (
        spec.unit_id_column,
        spec.subject_id_column,
        spec.split_column,
        spec.outcome_column,
        *spec.feature_columns,
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for unit_id, subject_id, split, outcome, feature_row in zip(
        prepared.unit_ids,
        prepared.subject_ids,
        prepared.splits,
        prepared.outcomes,
        prepared.features.itertuples(index=False, name=None),
        strict=True,
    ):
        features = tuple(
            "" if np.isnan(value) else format(float(value), ".17g")
            for value in feature_row
        )
        writer.writerow(
            (
                unit_id,
                subject_id,
                split,
                int(outcome),
                *features,
            )
        )
    return buffer.getvalue().encode("utf-8")


def prediction_model_fit_source_projection_bytes(
    *,
    source_input: LoadedTypedInput,
    spec: PredictionModelFitSpec | Mapping[str, Any],
) -> bytes:
    """Serialize only the exact model-consumed columns in canonical CSV form."""

    parsed_spec = _parse_spec(spec)
    prepared, _ = _prepare_input(source_input=source_input, spec=parsed_spec)
    return _source_projection_csv_bytes(prepared, parsed_spec)


def _read_persisted_source_projection(
    payload: bytes,
    spec: PredictionModelFitSpec,
) -> pd.DataFrame:
    if not isinstance(payload, bytes) or not payload:
        _raise(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "persisted model source must be non-empty immutable bytes",
        )
    expected_columns = (
        spec.unit_id_column,
        spec.subject_id_column,
        spec.split_column,
        spec.outcome_column,
        *spec.feature_columns,
    )
    try:
        text = payload.decode("utf-8", errors="strict")
        header = next(csv.reader(io.StringIO(text, newline="")), None)
    except (UnicodeDecodeError, csv.Error) as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "persisted model source is not valid UTF-8 CSV",
        ) from error
    if header is None or tuple(header) != expected_columns:
        _raise(
            PredictionModelFitReason.MISSING_COLUMNS,
            "persisted model source columns do not match the fit declaration",
            expected_columns=list(expected_columns),
            observed_columns=list(header or []),
        )
    if len(header) != len(set(header)):
        _raise(
            PredictionModelFitReason.MISSING_COLUMNS,
            "persisted model source contains duplicate columns",
        )
    try:
        return pd.read_csv(
            io.BytesIO(payload),
            dtype={
                spec.unit_id_column: str,
                spec.subject_id_column: str,
                spec.split_column: str,
            },
        )
    except (pd.errors.ParserError, UnicodeDecodeError, TypeError, ValueError) as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.SOURCE_INPUT_INVALID,
            "persisted model source cannot be parsed",
        ) from error


def _package_versions() -> tuple[PredictionPackageVersion, ...]:
    values: list[PredictionPackageVersion] = []
    for distribution in sorted(
        ("easyicu", "numpy", "pandas", "pyarrow", "scikit-learn")
    ):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise PredictionModelFitError(
                PredictionModelFitReason.RUNTIME_IDENTITY_INVALID,
                "a required model-fit distribution has no installed version",
                distribution=distribution,
            ) from error
        values.append(
            PredictionPackageVersion(distribution=distribution, version=version)
        )
    return tuple(values)


class PredictionModelFitBundle:
    """Immutable model/prediction bytes issued only by the fit owner."""

    __slots__ = (
        "__model_artifact",
        "__model_artifact_bytes",
        "__prediction_csv_bytes",
        "__prediction_payload",
        "__receipt",
        "__sealed",
    )

    def __init__(
        self,
        *,
        prediction_payload: pa.Table,
        prediction_csv_bytes: bytes,
        model_artifact_bytes: bytes,
        receipt: PredictionModelFitReceipt,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "PredictionModelFitBundle may only be constructed by the fit owner",
            )
        if not isinstance(prediction_payload, pa.Table):
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction bundle payload must be an Arrow table",
            )
        if not isinstance(prediction_csv_bytes, bytes) or not isinstance(
            model_artifact_bytes, bytes
        ):
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction and model artifacts must be immutable bytes",
            )
        try:
            parsed_receipt = PredictionModelFitReceipt.model_validate(
                receipt.model_dump(mode="python")
            )
            model_artifact = PredictionModelArtifact.model_validate_json(
                model_artifact_bytes
            )
        except (AttributeError, ValidationError) as error:
            raise PredictionModelFitError(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction-model bundle contains an invalid receipt or model artifact",
            ) from error
        if parsed_receipt != receipt:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction-model receipt changed during validation",
            )
        if prediction_model_artifact_bytes(model_artifact) != model_artifact_bytes:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "model artifact is not in canonical byte form",
            )
        if sha256_bytes(model_artifact_bytes) != receipt.model_artifact_sha256:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "model artifact bytes do not match the receipt",
            )
        if len(model_artifact_bytes) != receipt.model_artifact_size_bytes:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "model artifact size does not match the receipt",
            )
        if sha256_bytes(prediction_csv_bytes) != receipt.prediction_table_sha256:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction CSV bytes do not match the receipt",
            )
        if len(prediction_csv_bytes) != receipt.prediction_table_size_bytes:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction CSV size does not match the receipt",
            )
        if prediction_payload.num_rows != receipt.prediction_table_row_count:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction payload row count does not match the receipt",
            )
        observed_csv = _prediction_csv_bytes(
            prediction_payload.to_pandas(),
            expected_columns=receipt.prediction_table_columns,
        )
        if observed_csv != prediction_csv_bytes:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "prediction payload does not match the exact CSV bytes",
            )
        binding_fields = (
            "contract_sha256",
            "source_input_receipt_sha256",
            "source_artifact_sha256",
            "training_subjects_sha256",
            "evaluation_subjects_sha256",
            "split_assignment_sha256",
        )
        mismatches = [
            field
            for field in binding_fields
            if getattr(model_artifact, field) != getattr(receipt, field)
        ]
        if mismatches:
            _raise(
                PredictionModelFitReason.BUNDLE_INVALID,
                "model artifact and fit receipt bindings differ",
                mismatched_fields=mismatches,
            )
        object.__setattr__(
            self,
            "_PredictionModelFitBundle__prediction_payload",
            prediction_payload,
        )
        object.__setattr__(
            self,
            "_PredictionModelFitBundle__prediction_csv_bytes",
            prediction_csv_bytes,
        )
        object.__setattr__(
            self,
            "_PredictionModelFitBundle__model_artifact_bytes",
            model_artifact_bytes,
        )
        object.__setattr__(
            self,
            "_PredictionModelFitBundle__model_artifact",
            model_artifact,
        )
        object.__setattr__(self, "_PredictionModelFitBundle__receipt", receipt)
        object.__setattr__(self, "_PredictionModelFitBundle__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_PredictionModelFitBundle__sealed", False):
            raise AttributeError("PredictionModelFitBundle is immutable")
        object.__setattr__(self, name, value)

    @property
    def prediction_payload(self) -> pa.Table:
        return self.__prediction_payload

    @property
    def prediction_csv_bytes(self) -> bytes:
        return self.__prediction_csv_bytes

    @property
    def model_artifact_bytes(self) -> bytes:
        return self.__model_artifact_bytes

    @property
    def model_artifact(self) -> PredictionModelArtifact:
        return self.__model_artifact

    @property
    def receipt(self) -> PredictionModelFitReceipt:
        return self.__receipt

    def to_pandas(self) -> pd.DataFrame:
        """Materialize a fresh prediction-table copy."""

        return self.__prediction_payload.to_pandas()


def _build_prediction_model_fit_bundle(
    *,
    prepared: _PreparedInput,
    source_receipt: TypedInputConsumptionReceipt,
    parsed_spec: PredictionModelFitSpec,
) -> PredictionModelFitBundle:
    preprocessing, estimator, probabilities = _fit_train_only(prepared, parsed_spec)
    contract_sha256 = prediction_model_fit_spec_sha256(parsed_spec)
    training_subjects = tuple(
        sorted(prepared.subject_ids.loc[prepared.training_mask].tolist())
    )
    evaluation_subjects = tuple(
        sorted(prepared.subject_ids.loc[prepared.evaluation_mask].tolist())
    )
    training_subjects_sha256 = canonical_sha256(
        {"analysis_unit": "subject", "subjects": training_subjects}
    )
    evaluation_subjects_sha256 = canonical_sha256(
        {"analysis_unit": "subject", "subjects": evaluation_subjects}
    )
    assignments = sorted(
        (
            {"subject_id": subject_id, "split": split}
            for subject_id, split in zip(
                prepared.subject_ids.tolist(),
                prepared.splits.tolist(),
                strict=True,
            )
        ),
        key=lambda value: value["subject_id"],
    )
    split_assignment_sha256 = canonical_sha256(
        {"analysis_unit": "subject", "assignments": assignments}
    )
    model_artifact = PredictionModelArtifact(
        issuer=_ISSUER,
        model_identifier=parsed_spec.model_identifier,
        contract_sha256=contract_sha256,
        source_input_receipt_sha256=source_receipt.receipt_sha256,
        source_artifact_sha256=source_receipt.artifact_sha256,
        training_subjects_sha256=training_subjects_sha256,
        evaluation_subjects_sha256=evaluation_subjects_sha256,
        split_assignment_sha256=split_assignment_sha256,
        analysis_unit="subject",
        preprocessing=preprocessing,
        estimator=estimator,
        authority_scope="analysis_only",
        paper_authorization_allowed=False,
    )
    model_bytes = prediction_model_artifact_bytes(model_artifact)

    output_columns = (
        parsed_spec.unit_id_column,
        parsed_spec.subject_id_column,
        parsed_spec.split_column,
        parsed_spec.outcome_column,
        parsed_spec.probability_column,
    )
    prediction_frame = pd.DataFrame(
        {
            parsed_spec.unit_id_column: prepared.unit_ids,
            parsed_spec.subject_id_column: prepared.subject_ids,
            parsed_spec.split_column: prepared.splits,
            parsed_spec.outcome_column: prepared.outcomes,
            parsed_spec.probability_column: probabilities,
        }
    )
    prediction_bytes = _prediction_csv_bytes(
        prediction_frame,
        expected_columns=output_columns,
    )
    try:
        prediction_payload = pa.Table.from_pandas(
            prediction_frame,
            preserve_index=False,
            safe=True,
        )
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.PREDICTION_INVALID,
            "prediction table cannot be represented as an immutable Arrow payload",
        ) from error

    package_versions = _package_versions()
    receipt_payload: dict[str, object] = {
        "schema_version": "easyicu.prediction_model_fit_receipt/1",
        "issuer": _ISSUER,
        "execution_mode": "experimental",
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "source_input_receipt_sha256": source_receipt.receipt_sha256,
        "source_artifact_sha256": source_receipt.artifact_sha256,
        "source_loaded_frame_sha256": source_receipt.loaded_frame_sha256,
        "source_row_identity_sha256": source_receipt.row_identity.sha256,
        "contract_sha256": contract_sha256,
        "training_split": parsed_spec.training_split,
        "evaluation_split": parsed_spec.evaluation_split,
        "input_n": int(len(prepared.outcomes)),
        "training_n": int(prepared.training_mask.sum()),
        "evaluation_n": int(prepared.evaluation_mask.sum()),
        "training_event_n": int(prepared.outcomes[prepared.training_mask].sum()),
        "evaluation_event_n": int(prepared.outcomes[prepared.evaluation_mask].sum()),
        "training_subjects_sha256": training_subjects_sha256,
        "evaluation_subjects_sha256": evaluation_subjects_sha256,
        "split_assignment_sha256": split_assignment_sha256,
        "preprocessing_fit_scope": "training_subjects_only",
        "model_fit_scope": "training_subjects_only",
        "model_artifact_sha256": sha256_bytes(model_bytes),
        "model_artifact_size_bytes": len(model_bytes),
        "prediction_table_sha256": sha256_bytes(prediction_bytes),
        "prediction_table_size_bytes": len(prediction_bytes),
        "prediction_table_row_count": int(len(prediction_frame)),
        "prediction_table_columns": output_columns,
        "package_versions": tuple(
            item.model_dump(mode="python") for item in package_versions
        ),
    }
    receipt_payload["receipt_sha256"] = prediction_model_fit_receipt_sha256(
        receipt_payload
    )
    try:
        receipt = PredictionModelFitReceipt.model_validate(receipt_payload)
    except ValidationError as error:  # pragma: no cover - owner construction invariant
        raise PredictionModelFitError(
            PredictionModelFitReason.BUNDLE_INVALID,
            "owner produced an invalid prediction-model receipt",
            errors=error.errors(include_url=False),
        ) from error
    return PredictionModelFitBundle(
        prediction_payload=prediction_payload,
        prediction_csv_bytes=prediction_bytes,
        model_artifact_bytes=model_bytes,
        receipt=receipt,
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def fit_binary_prediction_model(
    *,
    source_input: LoadedTypedInput,
    spec: PredictionModelFitSpec | Mapping[str, Any],
) -> PredictionModelFitBundle:
    """Fit the fixed v1 binary model using training subjects only."""

    parsed_spec = _parse_spec(spec)
    prepared, source_receipt = _prepare_input(
        source_input=source_input,
        spec=parsed_spec,
    )
    return _build_prediction_model_fit_bundle(
        prepared=prepared,
        source_receipt=source_receipt,
        parsed_spec=parsed_spec,
    )


def revalidate_prediction_model_fit_persisted_artifacts(
    *,
    source_projection_csv_bytes: bytes,
    spec: PredictionModelFitSpec | Mapping[str, Any],
    source_receipt: TypedInputConsumptionReceipt | Mapping[str, Any],
    expected_fit_receipt: PredictionModelFitReceipt | Mapping[str, Any],
    model_artifact_bytes: bytes,
    prediction_csv_bytes: bytes,
) -> PredictionModelFitReceipt:
    """Re-fit current persisted bytes without granting evidence authority."""

    parsed_spec = _parse_spec(spec)
    parsed_source_receipt = _validated_persisted_source_receipt(source_receipt)
    fit_payload = (
        expected_fit_receipt.model_dump(mode="python")
        if isinstance(expected_fit_receipt, PredictionModelFitReceipt)
        else expected_fit_receipt
    )
    try:
        parsed_fit_receipt = PredictionModelFitReceipt.model_validate(fit_payload)
    except ValidationError as error:
        raise PredictionModelFitError(
            PredictionModelFitReason.BUNDLE_INVALID,
            "persisted fit receipt is invalid",
        ) from error
    if not isinstance(model_artifact_bytes, bytes) or not isinstance(
        prediction_csv_bytes, bytes
    ):
        _raise(
            PredictionModelFitReason.BUNDLE_INVALID,
            "persisted model and prediction artifacts must be immutable bytes",
        )
    source_frame = _read_persisted_source_projection(
        source_projection_csv_bytes,
        parsed_spec,
    )
    prepared = _prepare_frame(
        frame=source_frame,
        receipt=parsed_source_receipt,
        spec=parsed_spec,
    )
    expected = _build_prediction_model_fit_bundle(
        prepared=prepared,
        source_receipt=parsed_source_receipt,
        parsed_spec=parsed_spec,
    )
    mismatched_parts: list[str] = []
    if prediction_csv_bytes != expected.prediction_csv_bytes:
        mismatched_parts.append("prediction_csv")
    if model_artifact_bytes != expected.model_artifact_bytes:
        mismatched_parts.append("model_artifact")
    if parsed_fit_receipt != expected.receipt:
        mismatched_parts.append("receipt")
    if mismatched_parts:
        _raise(
            PredictionModelFitReason.RECOMPUTATION_MISMATCH,
            "persisted prediction-model evidence does not match full re-fit",
            mismatched_parts=mismatched_parts,
        )
    return expected.receipt


def revalidate_prediction_model_fit_bundle(
    *,
    bundle: PredictionModelFitBundle,
    source_input: LoadedTypedInput,
    spec: PredictionModelFitSpec | Mapping[str, Any],
) -> PredictionModelFitReceipt:
    """Recompute the complete fit and reject any source or bundle drift."""

    if not isinstance(bundle, PredictionModelFitBundle):
        _raise(
            PredictionModelFitReason.BUNDLE_INVALID,
            "revalidation requires one owner-issued PredictionModelFitBundle",
        )
    validated = PredictionModelFitBundle(
        prediction_payload=bundle.prediction_payload,
        prediction_csv_bytes=bundle.prediction_csv_bytes,
        model_artifact_bytes=bundle.model_artifact_bytes,
        receipt=bundle.receipt,
        _construction_token=_CONSTRUCTION_TOKEN,
    )
    expected = fit_binary_prediction_model(source_input=source_input, spec=spec)
    mismatched_parts: list[str] = []
    if validated.prediction_csv_bytes != expected.prediction_csv_bytes:
        mismatched_parts.append("prediction_csv")
    if validated.model_artifact_bytes != expected.model_artifact_bytes:
        mismatched_parts.append("model_artifact")
    if validated.receipt != expected.receipt:
        mismatched_parts.append("receipt")
    if mismatched_parts:
        _raise(
            PredictionModelFitReason.RECOMPUTATION_MISMATCH,
            "prediction-model bundle does not match full host recomputation",
            mismatched_parts=mismatched_parts,
        )
    return validated.receipt


__all__ = [
    "PredictionModelFitBundle",
    "fit_binary_prediction_model",
    "prediction_model_fit_source_projection_bytes",
    "revalidate_prediction_model_fit_bundle",
    "revalidate_prediction_model_fit_persisted_artifacts",
]
