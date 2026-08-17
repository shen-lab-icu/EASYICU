"""Source provenance and independent-R oracle for prediction validation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.canonical_json import sha256_bytes, sha256_file
from easyicu.research_agent.contracts.prediction_validation import (
    PredictionValidationError,
    PredictionValidationReason,
    PredictionValidationSpec,
    prediction_validation_receipt_sha256,
    prediction_validation_result_sha256,
)
from easyicu.research_agent.execution.runners.prediction_validation_executor import (
    prediction_validation_receipt_findings,
    run_prediction_validation_csv,
)


DATA = Path(__file__).parent / "data"
SOURCE = DATA / "oracle_prediction_validation.csv"
R_SCRIPT = DATA / "oracle_prediction_validation.R"
ORACLE = json.loads(
    (DATA / "prediction_validation_r_oracle.json").read_text(encoding="utf-8")
)


def _spec() -> PredictionValidationSpec:
    return PredictionValidationSpec(
        unit_id_column="row_id",
        subject_id_column="subject_id",
        split_column="split",
        outcome_column="outcome",
        probability_column="probability",
        evaluation_split="test",
        analysis_unit="subject",
        thresholds=(0.5, 0.8),
        calibration_bins=3,
    )


def _receipt():
    return run_prediction_validation_csv(
        source_path=SOURCE,
        expected_source_sha256=sha256_file(SOURCE),
        spec=_spec(),
    )


def _assert_source_reason(
    path: Path,
    expected_sha256: str,
    reason: PredictionValidationReason,
) -> None:
    with pytest.raises(PredictionValidationError) as caught:
        run_prediction_validation_csv(
            source_path=path,
            expected_source_sha256=expected_sha256,
            spec=_spec(),
        )
    assert caught.value.reason_code is reason


def test_csv_receipt_binds_exact_source_contract_and_result() -> None:
    receipt = _receipt()

    assert receipt.paper_authorization is False
    assert receipt.source.source_artifact_name == SOURCE.name
    assert receipt.source.source_artifact_sha256 == sha256_file(SOURCE)
    assert receipt.source.source_artifact_size_bytes == SOURCE.stat().st_size
    assert (
        receipt.source.model_dump(mode="json")["parser_profile"]
        == "easyicu.prediction_validation_csv_strict/1"
    )
    assert receipt.source.source_row_count == 16
    assert receipt.source.source_columns == (
        "row_id",
        "subject_id",
        "split",
        "outcome",
        "probability",
    )
    assert receipt.contract_sha256 == receipt.result.contract_sha256
    assert receipt.result_sha256 == prediction_validation_result_sha256(receipt.result)
    assert len(prediction_validation_receipt_sha256(receipt)) == 64
    assert (
        prediction_validation_receipt_findings(
            source_path=SOURCE,
            expected_source_sha256=sha256_file(SOURCE),
            spec=_spec(),
            candidate=receipt,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("expected_sha256", "reason"),
    [
        ("not-a-digest", PredictionValidationReason.SOURCE_DIGEST_INVALID),
        ("0" * 64, PredictionValidationReason.SOURCE_DIGEST_MISMATCH),
    ],
)
def test_csv_source_digest_fails_closed(
    expected_sha256: str,
    reason: PredictionValidationReason,
) -> None:
    _assert_source_reason(SOURCE, expected_sha256, reason)


def test_csv_source_rejects_invalid_artifacts_and_parse_failures(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.csv"
    _assert_source_reason(
        missing,
        "0" * 64,
        PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
    )

    wrong_suffix = tmp_path / "source.txt"
    wrong_suffix.write_bytes(SOURCE.read_bytes())
    _assert_source_reason(
        wrong_suffix,
        sha256_file(wrong_suffix),
        PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
    )

    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    _assert_source_reason(
        empty,
        sha256_bytes(b""),
        PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
    )

    invalid_utf8 = tmp_path / "invalid.csv"
    invalid_utf8.write_bytes(b"\xff")
    _assert_source_reason(
        invalid_utf8,
        sha256_file(invalid_utf8),
        PredictionValidationReason.SOURCE_READ_FAILED,
    )

    header_only = tmp_path / "header.csv"
    header_only.write_text(
        "row_id,subject_id,split,outcome,probability\n",
        encoding="utf-8",
    )
    _assert_source_reason(
        header_only,
        sha256_file(header_only),
        PredictionValidationReason.EMPTY_INPUT,
    )


def test_csv_source_rejects_duplicate_raw_headers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "row_id,subject_id,split,outcome,probability,probability\n"
        "0,subject-0,test,0,0.1,0.9\n"
        "1,subject-1,test,1,0.4,0.6\n"
        "2,subject-2,test,0,0.6,0.4\n"
        "3,subject-3,test,1,0.9,0.1\n",
        encoding="utf-8",
    )

    _assert_source_reason(
        duplicate,
        sha256_file(duplicate),
        PredictionValidationReason.SOURCE_ARTIFACT_INVALID,
    )


def test_csv_source_drift_is_rejected_after_receipt(tmp_path: Path) -> None:
    source = tmp_path / SOURCE.name
    source.write_bytes(SOURCE.read_bytes())
    original_sha256 = sha256_file(source)
    receipt = run_prediction_validation_csv(
        source_path=source,
        expected_source_sha256=original_sha256,
        spec=_spec(),
    )
    source.write_bytes(source.read_bytes() + b"\n")

    assert receipt.source.source_artifact_sha256 == original_sha256
    _assert_source_reason(
        source,
        original_sha256,
        PredictionValidationReason.SOURCE_DIGEST_MISMATCH,
    )
    findings = prediction_validation_receipt_findings(
        source_path=source,
        expected_source_sha256=original_sha256,
        spec=_spec(),
        candidate=receipt,
    )
    assert len(findings) == 1
    assert findings[0].reason_code is PredictionValidationReason.SOURCE_DIGEST_MISMATCH
    assert findings[0].detail["expected_sha256"] == original_sha256
    assert findings[0].detail["observed_sha256"] == sha256_file(source)


def test_receipt_validator_rejects_invalid_schema_and_valid_tampering() -> None:
    receipt = _receipt()
    invalid = receipt.model_dump(mode="json")
    invalid["result"]["summary"]["brier_score"] += 0.01

    schema_findings = prediction_validation_receipt_findings(
        source_path=SOURCE,
        expected_source_sha256=sha256_file(SOURCE),
        spec=_spec(),
        candidate=invalid,
    )
    assert len(schema_findings) == 1
    assert (
        schema_findings[0].reason_code
        is PredictionValidationReason.RECEIPT_SCHEMA_INVALID
    )

    invalid_instance = receipt.model_copy(update={"paper_authorization": True})
    instance_findings = prediction_validation_receipt_findings(
        source_path=SOURCE,
        expected_source_sha256=sha256_file(SOURCE),
        spec=_spec(),
        candidate=invalid_instance,
    )
    assert len(instance_findings) == 1
    assert (
        instance_findings[0].reason_code
        is PredictionValidationReason.RECEIPT_SCHEMA_INVALID
    )
    with pytest.raises(ValidationError):
        prediction_validation_receipt_sha256(invalid_instance)

    tampered = receipt.model_copy(
        update={
            "source": receipt.source.model_copy(
                update={"source_artifact_name": "renamed.csv"}
            )
        }
    )
    mismatch_findings = prediction_validation_receipt_findings(
        source_path=SOURCE,
        expected_source_sha256=sha256_file(SOURCE),
        spec=_spec(),
        candidate=tampered,
    )
    assert len(mismatch_findings) == 1
    assert (
        mismatch_findings[0].reason_code is PredictionValidationReason.RECEIPT_MISMATCH
    )
    assert mismatch_findings[0].detail["path"] == "$.source.source_artifact_name"


def test_frozen_r_oracle_is_bound_to_exact_fixture_and_script() -> None:
    provenance = ORACLE["_provenance"]

    assert provenance["source_csv_sha256"] == sha256_file(SOURCE)
    assert provenance["script_sha256"] == sha256_file(R_SCRIPT)
    assert "stats::glm" in provenance["reference_scope"]
    assert abs(ORACLE["calibration_intercept"]) > 0.2
    assert abs(ORACLE["calibration_slope"] - 1.0) > 0.2


def test_python_owner_matches_frozen_independent_r_oracle() -> None:
    result = _receipt().result
    summary = result.summary

    assert summary.evaluation_n == ORACLE["evaluation_n"]
    assert summary.auroc == pytest.approx(ORACLE["auroc"], rel=1e-12)
    assert summary.brier_score == pytest.approx(ORACLE["brier_score"], rel=1e-12)
    assert summary.calibration_intercept == pytest.approx(
        ORACLE["calibration_intercept"], abs=1e-10
    )
    assert summary.calibration_slope == pytest.approx(
        ORACLE["calibration_slope"], rel=1e-10
    )
    by_threshold = {str(row.threshold): row for row in result.threshold_metrics}
    for threshold, expected in ORACLE["threshold_metrics"].items():
        observed = by_threshold[threshold]
        assert observed.true_positive_n == expected["tp"]
        assert observed.false_positive_n == expected["fp"]
        assert observed.true_negative_n == expected["tn"]
        assert observed.false_negative_n == expected["fn"]


def test_live_base_r_oracle_matches_frozen_values() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is optional; frozen external oracle remains enforced")
    completed = subprocess.run(
        [rscript, str(R_SCRIPT), str(SOURCE)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    observed = json.loads(completed.stdout)

    assert observed["evaluation_n"] == ORACLE["evaluation_n"]
    assert observed["auroc"] == pytest.approx(ORACLE["auroc"], rel=1e-12)
    assert observed["brier_score"] == pytest.approx(ORACLE["brier_score"], rel=1e-12)
    assert observed["calibration_intercept"] == pytest.approx(
        ORACLE["calibration_intercept"], abs=1e-12
    )
    assert observed["calibration_slope"] == pytest.approx(
        ORACLE["calibration_slope"], rel=1e-12
    )
    assert observed["threshold_metrics"] == ORACLE["threshold_metrics"]
