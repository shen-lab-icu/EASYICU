"""Seal a local-only input for a time-varying exposure survival runtime.

The early-exposure panel deliberately retains its ``unmeasured`` state, and
the patient grouping bridge deliberately retains raw patient keys.  Neither is
safe to hand directly to a generic model runner.  This owner is the narrow
boundary between those two facts and a counting-process implementation:

* it requires an explicit, typed missingness parameterization;
* it verifies and consumes the private stay-to-patient bridge locally;
* it replaces both source identities with per-run opaque integer indices; and
* it returns only an aggregate receipt plus an in-memory numeric model frame.

It does not select the missingness policy, covariates, exposure definition, or
survival method.  In particular, this module is not a registration of a
formal E2 runtime; a governed plan and runner still have to bind its input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .patient_grouping import (
    PatientGroupingBinding,
    PatientGroupingError,
    load_verified_patient_grouping,
)


_PANEL_COLUMNS = (
    "stay_id",
    "interval_start_hours",
    "interval_stop_hours",
    "hospital_death",
    "exposure_state",
    "exposure_running_max",
)
_OBSERVED_STATE = "observed_running_max"
_UNMEASURED_STATE = "unmeasured"
_MISSINGNESS_POLICY = "observed_state_indicator"
_MODEL_EXPOSURE_TERMS = (
    "exposure_running_max_when_observed",
    "exposure_unmeasured_indicator",
)
_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class TimeVaryingExecutionInputError(ValueError):
    """A counting-process input cannot be sealed safely."""


@dataclass(frozen=True, slots=True)
class _BinaryCategoricalEncoding:
    """One explicit binary source-level encoding for a baseline covariate."""

    source_column: str
    output_column: str
    positive_level: str
    negative_level: str

    def public_contract(self) -> dict[str, str]:
        return {
            "source_column": self.source_column,
            "output_column": self.output_column,
            "kind": "binary_indicator",
            "positive_level": self.positive_level,
            "negative_level": self.negative_level,
            "unknown_or_missing_policy": "reject",
        }


@dataclass(frozen=True, slots=True)
class TimeVaryingExecutionInput:
    """Opaque local model frame and path-free construction receipt."""

    frame: pd.DataFrame
    model_covariates: tuple[str, ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        forbidden = {"stay_id", "__private_patient_group"}
        if forbidden.intersection(self.frame.columns):
            raise TimeVaryingExecutionInputError(
                "local execution input retains a source identifier"
            )


def _require_columns(frame: pd.DataFrame, *, label: str, columns: Sequence[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TimeVaryingExecutionInputError(f"{label} must be a dataframe")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise TimeVaryingExecutionInputError(
            f"{label} lacks required columns: {', '.join(missing)}"
        )


def _exact_stay_ids(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float)
    if (
        numeric.isna().any()
        or not np.isfinite(array).all()
        or not np.equal(array, np.trunc(array)).all()
        or (np.abs(array) >= 2**53).any()
    ):
        raise TimeVaryingExecutionInputError(
            f"{label} must contain exactly representable integer stay identifiers"
        )
    return numeric.astype("int64")


def _finite_numeric(values: pd.Series, *, label: str) -> np.ndarray:
    converted = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(converted).all():
        raise TimeVaryingExecutionInputError(f"{label} must be finite and complete")
    return converted


def _validate_intervals(panel: pd.DataFrame) -> pd.DataFrame:
    _require_columns(panel, label="time-varying exposure panel", columns=_PANEL_COLUMNS)
    normalized = panel.loc[:, _PANEL_COLUMNS].copy()
    normalized["stay_id"] = _exact_stay_ids(
        normalized["stay_id"], label="time-varying exposure panel"
    )
    start = _finite_numeric(
        normalized["interval_start_hours"], label="interval starts"
    )
    stop = _finite_numeric(normalized["interval_stop_hours"], label="interval stops")
    if (start < 0.0).any() or (stop <= start).any():
        raise TimeVaryingExecutionInputError(
            "counting-process intervals require 0 <= start < stop"
        )
    event = _finite_numeric(normalized["hospital_death"], label="event indicator")
    if not np.isin(event, [0.0, 1.0]).all():
        raise TimeVaryingExecutionInputError("event indicator must be binary")
    normalized["interval_start_hours"] = start
    normalized["interval_stop_hours"] = stop
    normalized["hospital_death"] = event.astype("int8")

    state = normalized["exposure_state"].astype("string")
    allowed = {_OBSERVED_STATE, _UNMEASURED_STATE}
    if state.isna().any() or not set(state.dropna()).issubset(allowed):
        raise TimeVaryingExecutionInputError(
            "exposure state must be observed_running_max or unmeasured"
        )
    value = pd.to_numeric(normalized["exposure_running_max"], errors="coerce")
    observed = state.eq(_OBSERVED_STATE)
    if not np.isfinite(value.loc[observed].to_numpy(dtype=float)).all():
        raise TimeVaryingExecutionInputError(
            "observed exposure intervals require finite running-max values"
        )
    if normalized.loc[~observed, "exposure_running_max"].notna().any():
        raise TimeVaryingExecutionInputError(
            "unmeasured exposure intervals cannot carry a numeric exposure value"
        )
    normalized["exposure_state"] = state
    normalized["exposure_running_max"] = value.astype(float)

    ordered = normalized.sort_values(
        ["stay_id", "interval_start_hours", "interval_stop_hours"],
        kind="mergesort",
    )
    for _, intervals in ordered.groupby("stay_id", sort=False):
        starts = intervals["interval_start_hours"].to_numpy(dtype=float)
        stops = intervals["interval_stop_hours"].to_numpy(dtype=float)
        events = intervals["hospital_death"].to_numpy(dtype=int)
        if starts[0] != 0.0 or not np.allclose(
            starts[1:], stops[:-1], rtol=0.0, atol=1e-10
        ):
            raise TimeVaryingExecutionInputError(
                "each stay must provide contiguous intervals from time zero"
            )
        if events.sum() > 1 or (events.sum() and events[-1] != 1):
            raise TimeVaryingExecutionInputError(
                "a stay can record at most one final-interval event"
            )
    return normalized


def _baseline_columns(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TimeVaryingExecutionInputError("baseline covariates must be a sequence")
    columns = tuple(str(value) for value in values)
    if len(columns) != len(set(columns)):
        raise TimeVaryingExecutionInputError("baseline covariates must be unique")
    if any(_COLUMN.fullmatch(column) is None for column in columns):
        raise TimeVaryingExecutionInputError("baseline covariates must be canonical columns")
    forbidden = {
        "stay_id",
        "analysis_stay_index",
        "analysis_cluster_index",
        "interval_start_hours",
        "interval_stop_hours",
        "hospital_death",
        "exposure_state",
        "exposure_running_max",
        *_MODEL_EXPOSURE_TERMS,
    }
    if forbidden.intersection(columns):
        raise TimeVaryingExecutionInputError(
            "baseline covariates collide with runtime-owned columns"
        )
    return columns


def _binary_categorical_encodings(
    values: Mapping[str, Mapping[str, Any]] | None,
    *,
    source_columns: tuple[str, ...],
) -> tuple[_BinaryCategoricalEncoding, ...]:
    if values is None:
        return ()
    if not isinstance(values, Mapping):
        raise TimeVaryingExecutionInputError(
            "baseline categorical encodings must be a mapping"
        )
    unexpected = set(values) - set(source_columns)
    if unexpected:
        raise TimeVaryingExecutionInputError(
            "baseline categorical encoding names are not declared covariates"
        )
    encodings: list[_BinaryCategoricalEncoding] = []
    for source_column, raw in values.items():
        if not isinstance(raw, Mapping):
            raise TimeVaryingExecutionInputError(
                "baseline categorical encoding must be a mapping"
            )
        expected_keys = {
            "kind",
            "output_column",
            "positive_level",
            "negative_level",
            "unknown_or_missing_policy",
        }
        if set(raw) != expected_keys or raw.get("kind") != "binary_indicator":
            raise TimeVaryingExecutionInputError(
                "baseline categorical encoding is not a complete binary contract"
            )
        output_column = raw.get("output_column")
        positive_level = raw.get("positive_level")
        negative_level = raw.get("negative_level")
        if (
            not isinstance(output_column, str)
            or _COLUMN.fullmatch(output_column) is None
            or not isinstance(positive_level, str)
            or not positive_level
            or not isinstance(negative_level, str)
            or not negative_level
            or positive_level == negative_level
            or raw.get("unknown_or_missing_policy") != "reject"
        ):
            raise TimeVaryingExecutionInputError(
                "baseline categorical encoding has invalid levels or policy"
            )
        encodings.append(
            _BinaryCategoricalEncoding(
                source_column=str(source_column),
                output_column=output_column,
                positive_level=positive_level,
                negative_level=negative_level,
            )
        )
    output_columns = [encoding.output_column for encoding in encodings]
    if len(output_columns) != len(set(output_columns)):
        raise TimeVaryingExecutionInputError(
            "baseline categorical encodings have duplicate output columns"
        )
    forbidden = {
        "stay_id",
        "analysis_stay_index",
        "analysis_cluster_index",
        "interval_start_hours",
        "interval_stop_hours",
        "hospital_death",
        *_MODEL_EXPOSURE_TERMS,
    }
    if forbidden.intersection(output_columns):
        raise TimeVaryingExecutionInputError(
            "baseline categorical encoding collides with runtime-owned columns"
        )
    return tuple(encodings)


def _normalize_baseline(
    baseline: pd.DataFrame,
    *,
    source_columns: tuple[str, ...],
    encodings: tuple[_BinaryCategoricalEncoding, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    _require_columns(
        baseline,
        label="baseline covariate table",
        columns=("stay_id", *source_columns),
    )
    source = baseline.loc[:, ["stay_id", *source_columns]].copy()
    normalized = source.loc[:, ["stay_id"]].copy()
    normalized["stay_id"] = _exact_stay_ids(
        normalized["stay_id"], label="baseline covariate table"
    )
    if normalized["stay_id"].duplicated().any():
        raise TimeVaryingExecutionInputError(
            "baseline covariate table must have one row per stay"
        )
    encoding_by_source = {encoding.source_column: encoding for encoding in encodings}
    output_columns: list[str] = []
    for source_column in source_columns:
        encoding = encoding_by_source.get(source_column)
        if encoding is None:
            normalized[source_column] = _finite_numeric(
                source[source_column], label=f"baseline covariate {source_column!r}"
            )
            output_columns.append(source_column)
            continue
        values = source[source_column].astype("string")
        if values.isna().any() or not values.isin(
            [encoding.positive_level, encoding.negative_level]
        ).all():
            raise TimeVaryingExecutionInputError(
                f"baseline categorical covariate {source_column!r} has undeclared levels"
            )
        normalized[encoding.output_column] = values.eq(
            encoding.positive_level
        ).to_numpy(dtype="int8")
        output_columns.append(encoding.output_column)
    if len(output_columns) != len(set(output_columns)):
        raise TimeVaryingExecutionInputError(
            "baseline model covariates have duplicate output columns"
        )
    return normalized, tuple(output_columns)


def _safe_grouping_coordinates(binding: PatientGroupingBinding) -> dict[str, Any]:
    coordinates = dict(binding.authority_coordinates)
    if (
        coordinates.get("schema_version")
        != "easyicu.patient_grouping_runtime_authority/1"
        or not isinstance(coordinates.get("authority_ref"), str)
        or not str(coordinates["authority_ref"]).strip()
        or coordinates.get("mapping_sha256") != binding.mapping_sha256
        or coordinates.get("grouping_derivation") != "prefix_before_:s"
        or coordinates.get("provider_visible_values") is not False
    ):
        raise TimeVaryingExecutionInputError(
            "patient grouping authority is not valid for local clustered inference"
        )
    safe = {
        "authority_ref": str(coordinates["authority_ref"]),
        "mapping_sha256": binding.mapping_sha256,
        "grouping_derivation": "prefix_before_:s",
        "provider_visible_values": False,
    }
    for key in ("database", "export_manifest_sha256"):
        value = coordinates.get(key)
        if isinstance(value, str) and value:
            safe[key] = value
    return safe


def build_time_varying_execution_input(
    panel: pd.DataFrame,
    baseline_covariates: pd.DataFrame,
    patient_grouping: PatientGroupingBinding,
    *,
    baseline_columns: Sequence[str],
    missingness_policy: str,
    baseline_categorical_encodings: Mapping[str, Mapping[str, Any]] | None = None,
) -> TimeVaryingExecutionInput:
    """Create opaque local input for the registered Cox adapter.

    The only currently implemented policy, ``observed_state_indicator``, must
    be named explicitly by a higher scientific contract.  It creates two model
    terms: the observed running maximum (using ``0`` only as a coding
    reference while unmeasured) and an unmeasured-state indicator.  It is not a
    clinical-value imputation and it is never selected implicitly.  A textual
    baseline covariate is likewise rejected unless
    ``baseline_categorical_encodings`` declares an exhaustive binary coding
    and an explicit reference level.
    """

    if missingness_policy != _MISSINGNESS_POLICY:
        raise TimeVaryingExecutionInputError(
            "time-varying missingness policy is unsupported or undeclared"
        )
    if not isinstance(patient_grouping, PatientGroupingBinding):
        raise TimeVaryingExecutionInputError("patient grouping binding is required")
    source_covariates = _baseline_columns(baseline_columns)
    categorical_encodings = _binary_categorical_encodings(
        baseline_categorical_encodings,
        source_columns=source_covariates,
    )
    normalized_panel = _validate_intervals(panel)
    baseline, covariates = _normalize_baseline(
        baseline_covariates,
        source_columns=source_covariates,
        encodings=categorical_encodings,
    )
    grouping_coordinates = _safe_grouping_coordinates(patient_grouping)
    try:
        verified_grouping = load_verified_patient_grouping(patient_grouping)
    except PatientGroupingError as exc:
        raise TimeVaryingExecutionInputError(str(exc)) from exc

    joined = normalized_panel.merge(
        verified_grouping.frame,
        on="stay_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if joined["__private_patient_group"].isna().any():
        raise TimeVaryingExecutionInputError(
            "patient grouping mapping does not cover every panel stay"
        )
    if not normalized_panel["stay_id"].isin(baseline["stay_id"]).all():
        raise TimeVaryingExecutionInputError(
            "baseline covariate table does not cover every panel stay"
        )
    joined = joined.merge(
        baseline,
        on="stay_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if any(joined[column].isna().any() for column in covariates):
        raise TimeVaryingExecutionInputError(
            "baseline covariate table does not cover every panel stay"
        )

    stay_codes, _ = pd.factorize(joined["stay_id"], sort=False)
    cluster_codes, _ = pd.factorize(joined["__private_patient_group"], sort=False)
    if (stay_codes < 0).any() or (cluster_codes < 0).any():  # pragma: no cover
        raise TimeVaryingExecutionInputError("local identity encoding failed")
    observed = joined["exposure_state"].eq(_OBSERVED_STATE)
    model = pd.DataFrame(
        {
            "analysis_stay_index": stay_codes.astype("int64") + 1,
            "analysis_cluster_index": cluster_codes.astype("int64") + 1,
            "interval_start_hours": joined["interval_start_hours"].to_numpy(
                dtype=float
            ),
            "interval_stop_hours": joined["interval_stop_hours"].to_numpy(
                dtype=float
            ),
            "hospital_death": joined["hospital_death"].to_numpy(dtype="int8"),
            _MODEL_EXPOSURE_TERMS[0]: np.where(
                observed.to_numpy(),
                joined["exposure_running_max"].to_numpy(dtype=float),
                0.0,
            ),
            _MODEL_EXPOSURE_TERMS[1]: (~observed).to_numpy(dtype="int8"),
            **{
                column: joined[column].to_numpy(dtype=float)
                for column in covariates
            },
        }
    )
    model_covariates = (*_MODEL_EXPOSURE_TERMS, *covariates)
    # ``analysis_stay_index`` is intentionally not added to ``joined``: it
    # keeps raw identities out of every intermediary exposed by this owner.
    fully_unmeasured = int(
        normalized_panel.groupby("stay_id", sort=False)["exposure_state"]
        .apply(lambda states: bool(states.eq(_UNMEASURED_STATE).all()))
        .sum()
    )
    receipt = {
        "schema_version": "easyicu.time_varying_execution_input/1",
        "local_only": True,
        "missingness_policy": {
            "kind": _MISSINGNESS_POLICY,
            "unmeasured_state": "separate_indicator_term",
            "observed_value_reference_for_unmeasured": 0.0,
            "clinical_value_imputed": False,
            "model_terms": list(_MODEL_EXPOSURE_TERMS),
        },
        "model_covariates": list(model_covariates),
        "baseline_categorical_encodings": [
            encoding.public_contract() for encoding in categorical_encodings
        ],
        "counts": {
            "interval_rows": int(len(model)),
            "stay_count": int(model["analysis_stay_index"].nunique()),
            "event_count": int(model["hospital_death"].sum()),
            "cluster_count": int(model["analysis_cluster_index"].nunique()),
            "observed_exposure_interval_rows": int(observed.sum()),
            "unmeasured_exposure_interval_rows": int((~observed).sum()),
            "fully_unmeasured_stays": fully_unmeasured,
            "patient_grouping_mapping_rows": int(len(verified_grouping.frame)),
        },
        "patient_grouping": grouping_coordinates,
        "privacy": {
            "local_ephemeral_input": True,
            "identifier_values_returned": False,
            "source_paths_returned": False,
            "patient_rows_returned": False,
        },
    }
    return TimeVaryingExecutionInput(
        frame=model,
        model_covariates=model_covariates,
        receipt=receipt,
    )


__all__ = [
    "TimeVaryingExecutionInput",
    "TimeVaryingExecutionInputError",
    "build_time_varying_execution_input",
]
