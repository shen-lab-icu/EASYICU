"""Contracts for privacy-preserving time-varying execution inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.acquisition.time_varying_execution_input import (
    TimeVaryingExecutionInputError,
    build_time_varying_execution_input,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [11, 11, 12, 13],
            "interval_start_hours": [0.0, 2.0, 0.0, 0.0],
            "interval_stop_hours": [2.0, 10.0, 8.0, 4.0],
            "hospital_death": [0, 1, 0, 0],
            "exposure_state": [
                "unmeasured",
                "observed_running_max",
                "observed_running_max",
                "unmeasured",
            ],
            "exposure_running_max": [np.nan, 2.5, 4.0, np.nan],
        }
    )


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [11, 12, 13],
            "age": [60.0, 72.0, 54.0],
            "sex_binary": [0.0, 1.0, 0.0],
        }
    )


def _grouping(tmp_path: Path) -> PatientGroupingBinding:
    mapping = tmp_path / "private-stay-patient.parquet"
    pd.DataFrame(
        {
            "stay_id": [11, 12, 13],
            "patient_key": [1001, 1001, 2002],
        }
    ).to_parquet(mapping, index=False)
    digest = _digest(mapping)
    return PatientGroupingBinding(
        mapping_path=mapping.absolute(),
        mapping_sha256=digest,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "test/time-varying-input/v1",
            "database": "miiv",
            "export_manifest_sha256": "b" * 64,
            "mapping_sha256": digest,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )


def test_execution_input_replaces_source_ids_and_materializes_explicit_state(
    tmp_path: Path,
) -> None:
    result = build_time_varying_execution_input(
        _panel(),
        _baseline(),
        _grouping(tmp_path),
        baseline_columns=("age", "sex_binary"),
        missingness_policy="observed_state_indicator",
    )

    assert result.frame.columns.tolist() == [
        "analysis_stay_index",
        "analysis_cluster_index",
        "interval_start_hours",
        "interval_stop_hours",
        "hospital_death",
        "exposure_running_max_when_observed",
        "exposure_unmeasured_indicator",
        "age",
        "sex_binary",
    ]
    assert result.frame["analysis_stay_index"].tolist() == [1, 1, 2, 3]
    assert result.frame["analysis_cluster_index"].tolist() == [1, 1, 1, 2]
    assert result.frame["exposure_running_max_when_observed"].tolist() == [
        0.0,
        2.5,
        4.0,
        0.0,
    ]
    assert result.frame["exposure_unmeasured_indicator"].tolist() == [1, 0, 0, 1]
    assert result.model_covariates == (
        "exposure_running_max_when_observed",
        "exposure_unmeasured_indicator",
        "age",
        "sex_binary",
    )
    assert "stay_id" not in result.frame
    assert "__private_patient_group" not in result.frame
    assert result.receipt["counts"] == {
        "interval_rows": 4,
        "stay_count": 3,
        "event_count": 1,
        "cluster_count": 2,
        "observed_exposure_interval_rows": 2,
        "unmeasured_exposure_interval_rows": 2,
        "fully_unmeasured_stays": 1,
        "patient_grouping_mapping_rows": 3,
    }
    assert result.receipt["missingness_policy"]["clinical_value_imputed"] is False
    assert result.receipt["patient_grouping"]["provider_visible_values"] is False
    assert result.receipt["privacy"]["source_paths_returned"] is False
    assert "1001" not in repr(result.receipt)


def test_execution_input_requires_declared_supported_missingness_policy(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TimeVaryingExecutionInputError, match="unsupported or undeclared"
    ):
        build_time_varying_execution_input(
            _panel(),
            _baseline(),
            _grouping(tmp_path),
            baseline_columns=("age",),
            missingness_policy="",
        )


def test_execution_input_requires_explicit_categorical_reference_contract(
    tmp_path: Path,
) -> None:
    baseline = _baseline().rename(columns={"sex_binary": "sex"})
    baseline["sex"] = ["Female", "Male", "Female"]
    encoding = {
        "sex": {
            "kind": "binary_indicator",
            "output_column": "sex_male",
            "positive_level": "Male",
            "negative_level": "Female",
            "unknown_or_missing_policy": "reject",
        }
    }

    result = build_time_varying_execution_input(
        _panel(),
        baseline,
        _grouping(tmp_path),
        baseline_columns=("age", "sex"),
        missingness_policy="observed_state_indicator",
        baseline_categorical_encodings=encoding,
    )

    assert result.model_covariates[-2:] == ("age", "sex_male")
    assert result.frame["sex_male"].tolist() == [0, 0, 1, 0]
    assert result.receipt["baseline_categorical_encodings"] == [
        {
            "source_column": "sex",
            "output_column": "sex_male",
            "kind": "binary_indicator",
            "positive_level": "Male",
            "negative_level": "Female",
            "unknown_or_missing_policy": "reject",
        }
    ]

    with pytest.raises(TimeVaryingExecutionInputError, match="must be finite"):
        build_time_varying_execution_input(
            _panel(),
            baseline,
            _grouping(tmp_path),
            baseline_columns=("age", "sex"),
            missingness_policy="observed_state_indicator",
        )

    baseline.loc[2, "sex"] = "Unknown"
    with pytest.raises(TimeVaryingExecutionInputError, match="undeclared levels"):
        build_time_varying_execution_input(
            _panel(),
            baseline,
            _grouping(tmp_path),
            baseline_columns=("age", "sex"),
            missingness_policy="observed_state_indicator",
            baseline_categorical_encodings=encoding,
        )


def test_execution_input_rejects_unverified_or_uncovered_grouping(
    tmp_path: Path,
) -> None:
    binding = _grouping(tmp_path)
    unverified = PatientGroupingBinding(
        mapping_path=binding.mapping_path,
        mapping_sha256=binding.mapping_sha256,
        mapping_stay_column=binding.mapping_stay_column,
        mapping_patient_column=binding.mapping_patient_column,
        authority_coordinates={},
    )
    with pytest.raises(TimeVaryingExecutionInputError, match="authority is not valid"):
        build_time_varying_execution_input(
            _panel(),
            _baseline(),
            unverified,
            baseline_columns=("age",),
            missingness_policy="observed_state_indicator",
        )

    mapping = tmp_path / "uncovered.parquet"
    pd.DataFrame({"stay_id": [11, 12], "patient_key": [1, 1]}).to_parquet(
        mapping, index=False
    )
    uncovered_digest = _digest(mapping)
    uncovered = PatientGroupingBinding(
        mapping_path=mapping.absolute(),
        mapping_sha256=uncovered_digest,
        mapping_stay_column="stay_id",
        mapping_patient_column="patient_key",
        authority_coordinates={
            "schema_version": "easyicu.patient_grouping_runtime_authority/1",
            "authority_ref": "test/time-varying-input/v1",
            "mapping_sha256": uncovered_digest,
            "grouping_derivation": "prefix_before_:s",
            "provider_visible_values": False,
        },
    )
    with pytest.raises(TimeVaryingExecutionInputError, match="does not cover"):
        build_time_varying_execution_input(
            _panel(),
            _baseline(),
            uncovered,
            baseline_columns=("age",),
            missingness_policy="observed_state_indicator",
        )


def test_execution_input_refuses_implicit_baseline_missingness(tmp_path: Path) -> None:
    baseline = _baseline()
    baseline.loc[1, "age"] = np.nan

    with pytest.raises(TimeVaryingExecutionInputError, match="must be finite"):
        build_time_varying_execution_input(
            _panel(),
            baseline,
            _grouping(tmp_path),
            baseline_columns=("age",),
            missingness_policy="observed_state_indicator",
        )
