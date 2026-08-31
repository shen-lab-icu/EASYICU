"""Compile fully typed Web study decisions into immutable runtime authority.

Owner
-----
This module owns the Web-to-Research-Agent projection for a closed landmark
restricted-cubic-spline association.  StudyContext owns the user's scientific
choices; the current-case runtime authority owns deterministic execution.  The
public contract below only joins those two typed boundaries after every
required coordinate is explicit.

Allowed dependencies are the dependency-neutral sensitivity contract, the
current-case authority builder, a parquet *schema* reader, and canonical
hashing.  It never reads patient rows, selects a variable, or infers a landmark
from prose.  Failures use ``WebScientificRuntimeProjectionError`` so the Web
runner can attribute the blocker to this owner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.dependence import PlannedDependenceRequirement
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)


class WebScientificRuntimeProjectionError(ValueError):
    """A user-reviewed design could not be bound to its deterministic owner."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)


@dataclass(frozen=True)
class WebScientificRuntimeProjection:
    authority: dict[str, Any]
    projection_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _one_spec(
    specs: Sequence[PrespecifiedSensitivitySpec], *, strategy: str
) -> PrespecifiedSensitivitySpec | None:
    matching = [spec for spec in specs if spec.strategy == strategy]
    if not matching:
        return None
    if len(matching) != 1:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_projection_ambiguous",
            f"The Web study declares more than one {strategy} sensitivity.",
            details={"strategy": strategy, "spec_ids": [spec.spec_id for spec in matching]},
        )
    return matching[0]


def _categorical_adjustments(
    universe_path: Path, *, covariates: Sequence[str]
) -> tuple[str, ...]:
    """Classify only physically typed string/dictionary/boolean columns."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.read_schema(universe_path)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    categorical: list[str] = []
    missing = [column for column in covariates if column not in schema.names]
    if missing:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_columns_missing",
            "The exact adjustment set is absent from the materialized universe.",
            details={"missing_columns": missing},
        )
    for column in covariates:
        column_type = schema.field(column).type
        if (
            pa.types.is_string(column_type)
            or pa.types.is_large_string(column_type)
            or pa.types.is_dictionary(column_type)
            or pa.types.is_boolean(column_type)
        ):
            categorical.append(column)
            continue
        if not (
            pa.types.is_integer(column_type)
            or pa.types.is_floating(column_type)
            or pa.types.is_decimal(column_type)
        ):
            raise WebScientificRuntimeProjectionError(
                "web_scientific_runtime_covariate_encoding_unsupported",
                "An exact adjustment column has no deterministic supported encoding.",
                details={"column": column, "parquet_type": str(column_type)},
            )
    return tuple(categorical)


def _operational_covariates(
    universe_path: Path,
    *,
    declared_covariates: Sequence[str],
    operationalizations: Mapping[str, str],
) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(universe_path).names)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    resolved = tuple(
        str(operationalizations.get(name) or name) for name in declared_covariates
    )
    missing = [
        declared
        for declared, operational in zip(declared_covariates, resolved)
        if operational not in schema_names
    ]
    if missing:
        raise WebScientificRuntimeProjectionError(
            "web_covariate_operationalization_required",
            "An exact covariate requires a user-reviewed materialized-column binding.",
            details={
                "missing_operationalizations": missing,
                "field": "covariate_operationalizations",
            },
        )
    if len(resolved) != len(set(resolved)):
        raise WebScientificRuntimeProjectionError(
            "web_covariate_operationalization_ambiguous",
            "Two exact covariates resolve to the same materialized column.",
            details={"resolved_covariates": list(resolved)},
        )
    return resolved


def compile_landmark_spline_runtime_projection(
    *,
    study: Mapping[str, Any],
    sensitivity_specs: Sequence[PrespecifiedSensitivitySpec],
    primary_exposure: str | None,
    primary_exposure_source: str | None,
    target_outcome: str | None,
    declared_covariates: Sequence[str],
    covariate_operationalizations: Mapping[str, str],
    target_is_event_status: bool,
    universe_path: Path,
    scientific_configuration_sha256: str,
    dependence: PlannedDependenceRequirement | None = None,
) -> WebScientificRuntimeProjection | None:
    """Return a signed runtime projection only for one fully closed design.

    Absence of either a landmark or RCS request means this owner is not
    applicable.  Once both are present, every remaining field is required and
    failures are explicit rather than silently falling back to agent-coded
    analysis.
    """

    landmark = _one_spec(sensitivity_specs, strategy="landmark")
    spline = _one_spec(sensitivity_specs, strategy="restricted_cubic_spline")
    if landmark is None or spline is None:
        return None

    missing_fields: list[str] = []
    if str(study.get("covariate_selection") or "") != "exact":
        missing_fields.append("covariate_selection=exact")
    if not primary_exposure:
        missing_fields.append("primary_exposure")
    if not primary_exposure_source:
        missing_fields.append("primary_exposure_source")
    if not target_outcome:
        missing_fields.append("target_outcome")
    if not target_is_event_status:
        missing_fields.append("binary_event_status_outcome")
    if not declared_covariates:
        missing_fields.append("exact_covariates")
    if landmark.event_time_variable is None:
        missing_fields.append("landmark.event_time_variable")
    if landmark.observation_duration_variable is None:
        missing_fields.append("landmark.observation_duration_variable")
    if landmark.observation_duration_unit is None:
        missing_fields.append("landmark.observation_duration_unit")
    if not landmark.require_alive_at_landmark:
        missing_fields.append("landmark.require_alive_at_landmark")
    if not landmark.exclude_negative_event_times:
        missing_fields.append("landmark.exclude_negative_event_times")
    if missing_fields:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_authority_incomplete",
            "The landmark spline design is user-selected but lacks executable coordinates.",
            details={"missing_fields": missing_fields},
        )

    if float(landmark.landmark_hours or 0.0) != 24.0:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_landmark_unsupported",
            "The verified Web landmark spline adapter currently supports a 24-hour landmark.",
            details={"landmark_hours": landmark.landmark_hours},
        )
    if landmark.observation_duration_unit != "days":
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_duration_unit_unsupported",
            "The verified Web landmark spline adapter currently expects ICU observation duration in days.",
            details={"observation_duration_unit": landmark.observation_duration_unit},
        )
    spline_sources = set(spline.execution_variables)
    if primary_exposure_source not in spline_sources and primary_exposure not in spline_sources:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_spline_exposure_binding_mismatch",
            "The RCS sensitivity is not bound to the configured primary exposure.",
            details={
                "primary_exposure": primary_exposure,
                "primary_exposure_source": primary_exposure_source,
                "spline_execution_variables": sorted(spline_sources),
            },
        )

    covariates = _operational_covariates(
        universe_path,
        declared_covariates=tuple(declared_covariates),
        operationalizations=covariate_operationalizations,
    )
    categorical = _categorical_adjustments(
        universe_path, covariates=covariates
    )
    required_columns = {
        str(primary_exposure),
        str(target_outcome),
        str(landmark.event_time_variable),
        str(landmark.observation_duration_variable),
        *map(str, covariates),
        *((dependence.group_source,) if dependence is not None else ()),
    }
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(universe_path).names)
    except Exception as exc:  # pragma: no cover - classified by helper above
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc
    absent = sorted(required_columns - schema_names)
    if absent:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_columns_missing",
            "The landmark spline runtime inputs are absent from the materialized universe.",
            details={"missing_columns": absent},
        )

    authority = build_current_case_scientific_runtime_authority(
        {
            "schema_version": (
                "easyicu.landmark_spline_runtime_authority/3"
                if dependence is not None
                else "easyicu.landmark_spline_runtime_authority/2"
            ),
            "authority_kind": "landmark_spline_association",
            "protocol_content_sha256": scientific_configuration_sha256,
            "plan_method": "signed_landmark_restricted_cubic_spline",
            "plan_intent": (
                "Execute the user-reviewed 24-hour landmark restricted-cubic-"
                "spline association and its prespecified linear sensitivity."
            ),
            "plan_outputs": [
                "table:landmark_rcs_curve",
                "table:landmark_rcs_contrasts",
                "table:landmark_linear_sensitivity",
                "table:landmark_adjusted_absolute_risk",
                "table:landmark_population_flow",
                "table:landmark_variable_opportunity_sensitivity",
                "log:landmark_scientific_runtime_receipt",
            ],
            "exposure_column": primary_exposure,
            "outcome_column": target_outcome,
            "outcome_time_column": landmark.event_time_variable,
            "observation_duration_column": landmark.observation_duration_variable,
            "observation_duration_unit": landmark.observation_duration_unit,
            "landmark_hours": 24,
            "required_adjustment_columns": list(covariates),
            "categorical_adjustment_columns": list(categorical),
            "alternative_exposure_columns": [],
            "dependence": (
                dependence.model_dump(mode="json")
                if dependence is not None
                else None
            ),
            "adjusted_absolute_risk_product": (
                "table:landmark_adjusted_absolute_risk"
            ),
            "population_flow_product": "table:landmark_population_flow",
            "variable_opportunity_sensitivity_product": (
                "table:landmark_variable_opportunity_sensitivity"
            ),
            "spline_knot_quantiles": [0.10, 0.50, 0.90],
            "spline_reference": "median_in_primary_population",
            "curve_quantile_range": [0.10, 0.90],
            "curve_points": 41,
            "linear_sensitivity_per_unit": 1.0,
            "interpretation": "descriptive_prognostic_association_not_causal",
        }
    ).model_dump(mode="json")
    projection_body = {
        "schema_version": "easyicu.web_scientific_runtime_projection/1",
        "study_scientific_configuration_sha256": scientific_configuration_sha256,
        "deterministic_execution_contract": authority,
    }
    return WebScientificRuntimeProjection(
        authority=authority,
        projection_sha256=hashlib.sha256(_canonical_bytes(projection_body)).hexdigest(),
    )


__all__ = [
    "WebScientificRuntimeProjection",
    "WebScientificRuntimeProjectionError",
    "compile_landmark_spline_runtime_projection",
]
