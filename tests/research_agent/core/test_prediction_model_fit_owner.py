"""Experimental host model fitting is train-only and receipt-bound."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from easyicu.research_agent.authority.typed_input_sdk import (
    LoadedTypedInput,
    load_typed_input,
)
from easyicu.research_agent.contracts.prediction_model_fit import (
    PredictionModelFitError,
    PredictionModelFitReason,
    PredictionModelFitSpec,
)
from easyicu.research_agent.contracts.prediction_validation import (
    PredictionValidationSpec,
)
from easyicu.research_agent.prediction_model_fit_owner import (
    PredictionModelFitBundle,
    fit_binary_prediction_model,
    revalidate_prediction_model_fit_bundle,
)
from easyicu.research_agent.prediction_validation_owner import (
    run_prediction_validation,
)


INPUT_KEY = "table:prediction_model_source"
STEP_ID = "fit_prediction_model"
CODE_SHA256 = "f" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_identity_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype("string"):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": list(range(101, 113)),
            "subject_id": [f"P{index:02d}" for index in range(1, 13)],
            "split": ["train"] * 8 + ["test"] * 4,
            "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "age": [
                40.0,
                50.0,
                60.0,
                70.0,
                45.0,
                55.0,
                65.0,
                75.0,
                48.0,
                58.0,
                68.0,
                78.0,
            ],
            "biomarker": [1.0, 2.0, None, 4.0, 1.5, 2.5, 3.5, 4.5, 2.2, 2.7, 3.2, 3.7],
        }
    )


def _loaded_input(case_root: Path, frame: pd.DataFrame) -> LoadedTypedInput:
    run_root = case_root / "run"
    evidence_dir = run_root / "evidence"
    manifest_dir = run_root / "resolved_inputs"
    evidence_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    artifact = evidence_dir / "prediction_model_source.parquet"
    frame.to_parquet(artifact, index=False)
    artifact_sha256 = _sha256(artifact)
    identity_row = {
        "input_key": INPUT_KEY,
        "declared_kind": "table",
        "product": "prediction_model_source",
        "evidence_id": "ev_prediction_model_source",
        "sha256": artifact_sha256,
        "produced_by_step": "cohort_and_split",
    }
    binding = {
        "evidence_id": "ev_prediction_model_source",
        "declared_kind": "table",
        "product": "prediction_model_source",
        "evidence_kind": "table",
        "relative_path": artifact.relative_to(run_root).as_posix(),
        "absolute_path": str(artifact),
        "sha256": artifact_sha256,
        "produced_by_step": "cohort_and_split",
        "identity_row": identity_row,
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v2",
            "identity_row": identity_row,
            "tabular_format": "parquet",
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "row_identity_column": "record_id",
            "row_count": len(frame),
            "row_identity_sha256": _row_identity_sha256(frame["record_id"]),
        },
    }
    manifest_payload = {
        "schema_version": "2.1",
        "step_id": STEP_ID,
        "planner_declared_inputs": [INPUT_KEY],
        "inputs": {INPUT_KEY: binding},
    }
    manifest = manifest_dir / f"{STEP_ID}.json"
    manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return load_typed_input(
        resolved_inputs_path=manifest,
        expected_resolved_inputs_sha256=_sha256(manifest),
        run_root=run_root,
        input_key=INPUT_KEY,
        consumer_step_id=STEP_ID,
        consumer_code_sha256=CODE_SHA256,
    )


def _spec(**updates: object) -> PredictionModelFitSpec:
    payload: dict[str, object] = {
        "model_identifier": "mortality_logistic_v1",
        "unit_id_column": "record_id",
        "subject_id_column": "subject_id",
        "split_column": "split",
        "outcome_column": "outcome",
        "probability_column": "probability",
        "feature_columns": ("age", "biomarker"),
        "training_split": "train",
        "evaluation_split": "test",
    }
    payload.update(updates)
    return PredictionModelFitSpec.model_validate(payload)


def _assert_reason(
    error: pytest.ExceptionInfo[PredictionModelFitError],
    expected: PredictionModelFitReason,
) -> None:
    assert error.value.reason_code is expected


def test_public_api_accepts_no_path_dataframe_or_caller_receipt() -> None:
    fit_parameters = set(inspect.signature(fit_binary_prediction_model).parameters)
    validation_parameters = set(
        inspect.signature(revalidate_prediction_model_fit_bundle).parameters
    )

    assert fit_parameters == {"source_input", "spec"}
    assert validation_parameters == {"bundle", "source_input", "spec"}
    assert not fit_parameters & {
        "path",
        "source_path",
        "frame",
        "dataframe",
        "receipt",
        "model_artifact",
        "prediction_table",
    }


def test_owner_fits_and_seals_one_downstream_compatible_bundle(tmp_path: Path) -> None:
    source = _loaded_input(tmp_path / "case", _source_frame())
    spec = _spec()

    bundle = fit_binary_prediction_model(source_input=source, spec=spec)
    receipt = bundle.receipt
    artifact = bundle.model_artifact
    predictions = bundle.to_pandas()

    assert receipt.issuer == "easyicu.prediction_model_fit_owner"
    assert receipt.execution_mode == "experimental"
    assert receipt.authority_scope == "analysis_only"
    assert receipt.paper_authorization_allowed is False
    assert receipt.source_input_receipt_sha256 == source.receipt.receipt_sha256
    assert receipt.source_artifact_sha256 == source.receipt.artifact_sha256
    assert receipt.preprocessing_fit_scope == "training_subjects_only"
    assert receipt.model_fit_scope == "training_subjects_only"
    assert receipt.input_n == 12
    assert receipt.training_n == 8
    assert receipt.evaluation_n == 4
    assert receipt.training_event_n == 4
    assert receipt.evaluation_event_n == 2
    assert artifact.preprocessing.fit_scope == "training_subjects_only"
    assert artifact.estimator.fit_scope == "training_subjects_only"
    assert artifact.preprocessing.feature_columns == ("age", "biomarker")
    assert artifact.preprocessing.medians == pytest.approx((57.5, 2.5))
    assert artifact.preprocessing.means == pytest.approx((57.5, 2.6875))
    assert predictions.columns.tolist() == [
        "record_id",
        "subject_id",
        "split",
        "outcome",
        "probability",
    ]
    assert predictions["probability"].between(0.0, 1.0).all()
    source_features = _source_frame().loc[:, ["age", "biomarker"]].to_numpy()
    medians = np.asarray(artifact.preprocessing.medians)
    means = np.asarray(artifact.preprocessing.means)
    scales = np.asarray(artifact.preprocessing.scales)
    imputed = np.where(np.isnan(source_features), medians, source_features)
    logits = ((imputed - means) / scales) @ np.asarray(
        artifact.estimator.coefficients
    ) + artifact.estimator.intercept
    reconstructed_probabilities = 1.0 / (1.0 + np.exp(-logits))
    assert predictions["probability"].tolist() == pytest.approx(
        reconstructed_probabilities.tolist(), rel=1e-14, abs=1e-14
    )
    assert (
        revalidate_prediction_model_fit_bundle(
            bundle=bundle,
            source_input=source,
            spec=spec,
        )
        == receipt
    )

    validation = run_prediction_validation(
        predictions,
        PredictionValidationSpec(
            unit_id_column="record_id",
            subject_id_column="subject_id",
            split_column="split",
            outcome_column="outcome",
            probability_column="probability",
            evaluation_split="test",
            analysis_unit="subject",
            thresholds=(0.25, 0.5, 0.75),
            calibration_bins=2,
        ),
    )
    assert validation.summary.evaluation_n == 4
    assert validation.summary.event_n == 2


def test_test_only_extremes_cannot_change_train_fitted_state(tmp_path: Path) -> None:
    baseline_frame = _source_frame()
    shifted_frame = _source_frame()
    shifted_frame.loc[shifted_frame["split"] == "test", "age"] += 100_000.0
    shifted_frame.loc[shifted_frame["split"] == "test", "biomarker"] *= 10_000.0
    spec = _spec()

    baseline = fit_binary_prediction_model(
        source_input=_loaded_input(tmp_path / "baseline", baseline_frame),
        spec=spec,
    )
    shifted = fit_binary_prediction_model(
        source_input=_loaded_input(tmp_path / "shifted", shifted_frame),
        spec=spec,
    )

    assert baseline.model_artifact.preprocessing == shifted.model_artifact.preprocessing
    assert baseline.model_artifact.estimator == shifted.model_artifact.estimator
    baseline_predictions = baseline.to_pandas()
    shifted_predictions = shifted.to_pandas()
    assert baseline_predictions.loc[:7, "probability"].tolist() == pytest.approx(
        shifted_predictions.loc[:7, "probability"].tolist(), rel=0.0, abs=0.0
    )
    assert baseline_predictions.loc[8:, "probability"].tolist() != pytest.approx(
        shifted_predictions.loc[8:, "probability"].tolist(), rel=0.0, abs=0.0
    )


def test_subject_crossing_train_and_test_fails_closed(tmp_path: Path) -> None:
    frame = _source_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate.loc[:, "record_id"] = 999
    duplicate.loc[:, "split"] = "test"
    frame = pd.concat([frame, duplicate], ignore_index=True)

    with pytest.raises(PredictionModelFitError) as error:
        fit_binary_prediction_model(
            source_input=_loaded_input(tmp_path / "case", frame),
            spec=_spec(),
        )

    _assert_reason(error, PredictionModelFitReason.SUBJECT_SPLIT_LEAKAGE)


def test_subject_level_contract_rejects_repeated_subjects_within_split(
    tmp_path: Path,
) -> None:
    frame = _source_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate.loc[:, "record_id"] = 999
    frame = pd.concat([frame, duplicate], ignore_index=True)

    with pytest.raises(PredictionModelFitError) as error:
        fit_binary_prediction_model(
            source_input=_loaded_input(tmp_path / "case", frame),
            spec=_spec(),
        )

    _assert_reason(error, PredictionModelFitReason.SUBJECT_UNIT_NOT_UNIQUE)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("nonnumeric", PredictionModelFitReason.FEATURE_NONNUMERIC),
        ("boolean", PredictionModelFitReason.FEATURE_NONNUMERIC),
        ("nonfinite", PredictionModelFitReason.FEATURE_NONFINITE),
        ("all_missing_train", PredictionModelFitReason.FEATURE_ALL_MISSING_TRAIN),
    ],
)
def test_invalid_feature_authority_fails_closed(
    tmp_path: Path,
    mutation: str,
    expected: PredictionModelFitReason,
) -> None:
    frame = _source_frame()
    if mutation == "nonnumeric":
        frame["age"] = frame["age"].map(lambda value: f"value-{value}")
    elif mutation == "boolean":
        frame["age"] = frame["age"] > 55.0
    elif mutation == "nonfinite":
        frame.loc[0, "age"] = float("inf")
    elif mutation == "all_missing_train":
        frame.loc[frame["split"] == "train", "biomarker"] = None
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(mutation)

    with pytest.raises(PredictionModelFitError) as error:
        fit_binary_prediction_model(
            source_input=_loaded_input(tmp_path / mutation, frame),
            spec=_spec(),
        )

    _assert_reason(error, expected)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("training_missing", PredictionModelFitReason.TRAINING_SPLIT_MISSING),
        ("evaluation_missing", PredictionModelFitReason.EVALUATION_SPLIT_MISSING),
        ("unknown_split", PredictionModelFitReason.SPLIT_INVALID),
        ("training_single_class", PredictionModelFitReason.TRAINING_SINGLE_CLASS),
    ],
)
def test_invalid_fit_coordinates_fail_closed(
    tmp_path: Path,
    mutation: str,
    expected: PredictionModelFitReason,
) -> None:
    frame = _source_frame()
    if mutation == "training_missing":
        frame.loc[:, "split"] = "test"
    elif mutation == "evaluation_missing":
        frame.loc[:, "split"] = "train"
    elif mutation == "unknown_split":
        frame.loc[0, "split"] = "validation"
    elif mutation == "training_single_class":
        frame.loc[frame["split"] == "train", "outcome"] = 0
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(mutation)

    with pytest.raises(PredictionModelFitError) as error:
        fit_binary_prediction_model(
            source_input=_loaded_input(tmp_path / mutation, frame),
            spec=_spec(),
        )

    _assert_reason(error, expected)


def test_unit_coordinate_must_be_the_typed_input_row_identity(tmp_path: Path) -> None:
    frame = _source_frame()
    frame["alternate_id"] = [f"A{index:02d}" for index in range(1, 13)]
    source = _loaded_input(tmp_path / "case", frame)

    with pytest.raises(PredictionModelFitError) as error:
        fit_binary_prediction_model(
            source_input=source,
            spec=_spec(unit_id_column="alternate_id"),
        )

    _assert_reason(error, PredictionModelFitReason.SOURCE_IDENTITY_MISMATCH)


def test_caller_cannot_construct_a_prediction_fit_bundle(tmp_path: Path) -> None:
    source = _loaded_input(tmp_path / "case", _source_frame())
    real = fit_binary_prediction_model(source_input=source, spec=_spec())

    with pytest.raises(PredictionModelFitError) as error:
        PredictionModelFitBundle(
            prediction_payload=real.prediction_payload,
            prediction_csv_bytes=real.prediction_csv_bytes,
            model_artifact_bytes=real.model_artifact_bytes,
            receipt=real.receipt,
            _construction_token=object(),
        )

    _assert_reason(error, PredictionModelFitReason.BUNDLE_INVALID)


@pytest.mark.parametrize("target", ["prediction", "model", "receipt", "payload"])
def test_full_recomputation_rejects_bundle_tampering(
    tmp_path: Path,
    target: str,
) -> None:
    source = _loaded_input(tmp_path / target, _source_frame())
    bundle = fit_binary_prediction_model(source_input=source, spec=_spec())
    if target == "prediction":
        object.__setattr__(
            bundle,
            "_PredictionModelFitBundle__prediction_csv_bytes",
            bundle.prediction_csv_bytes + b"tampered",
        )
    elif target == "model":
        object.__setattr__(
            bundle,
            "_PredictionModelFitBundle__model_artifact_bytes",
            bundle.model_artifact_bytes + b"tampered",
        )
    elif target == "receipt":
        forged = bundle.receipt.model_copy(update={"paper_authorization_allowed": True})
        object.__setattr__(
            bundle,
            "_PredictionModelFitBundle__receipt",
            forged,
        )
    elif target == "payload":
        changed = bundle.prediction_payload.set_column(
            bundle.prediction_payload.column_names.index("probability"),
            "probability",
            pa.array([0.5] * bundle.prediction_payload.num_rows),
        )
        object.__setattr__(
            bundle,
            "_PredictionModelFitBundle__prediction_payload",
            changed,
        )
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(target)

    with pytest.raises(PredictionModelFitError) as error:
        revalidate_prediction_model_fit_bundle(
            bundle=bundle,
            source_input=source,
            spec=_spec(),
        )

    assert error.value.reason_code in {
        PredictionModelFitReason.BUNDLE_INVALID,
        PredictionModelFitReason.RECOMPUTATION_MISMATCH,
    }


@pytest.mark.parametrize("drift", ["source", "spec"])
def test_full_recomputation_rejects_source_or_contract_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    source = _loaded_input(tmp_path / "baseline", _source_frame())
    spec = _spec()
    bundle = fit_binary_prediction_model(source_input=source, spec=spec)
    candidate_source = source
    candidate_spec = spec
    if drift == "source":
        changed = _source_frame()
        changed.loc[8, "age"] += 1.0
        candidate_source = _loaded_input(tmp_path / "changed", changed)
    elif drift == "spec":
        candidate_spec = _spec(model_identifier="mortality_logistic_v2")
    else:  # pragma: no cover - parametrization contract
        raise AssertionError(drift)

    with pytest.raises(PredictionModelFitError) as error:
        revalidate_prediction_model_fit_bundle(
            bundle=bundle,
            source_input=candidate_source,
            spec=candidate_spec,
        )

    _assert_reason(error, PredictionModelFitReason.RECOMPUTATION_MISMATCH)


def test_experimental_owner_has_no_planner_or_evidence_authority_import() -> None:
    import easyicu.research_agent.prediction_model_fit_owner as owner

    tree = ast.parse(Path(owner.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert not any("planning" in module for module in imported_modules)
    assert not any("evidence_store" in module for module in imported_modules)
    assert not any(
        "prediction_validation_evidence" in module for module in imported_modules
    )
