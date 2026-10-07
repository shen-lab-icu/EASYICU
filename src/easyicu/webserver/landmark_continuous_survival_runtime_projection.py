"""Web adapter for the sealed landmark survival suite of a continuous exposure.

Owner
-----
The binary survival adapter (``landmark_survival_runtime_projection``) owns a
survival-family landmark study's shared coordinates: the fixed-horizon
mortality endpoint with its paired follow-up, the landmark, the analysis unit
and the exact roster.  When the study's exposure is continuous it hands those
validated coordinates here, and this owner compiles the digest-bound
``LandmarkContinuousSurvivalRuntimeAuthority``: the exposure is one window
summary the materializer recorded from ICU admission to the landmark, modelled
per one readable step of its source's scale that the suite reads from the
modelled exposure, with the spline check that replaces the per-step estimate
when it rejects a linear term.

A value recorded after the landmark would let the future enter the exposure,
so the column must be the summary of a window that ends at the landmark.  The
verified materialized metadata proves it; a zero-row planning catalog, which
has none, binds the column by the ``<source>_<summary>`` name the materializer
gives it.  Anything the vocabularies cannot close fails with
``WebScientificRuntimeProjectionError`` instead of being inferred from prose or
from patient rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.concept.catalog import CONCEPT_DICTIONARY
from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.landmark_continuous_survival_runtime import (
    CONTINUOUS_EXPOSURE_WINDOW_SUMMARIES,
)
from easyicu.research_agent.concept_availability import concept_records_one_value_per_stay
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)

from .scientific_runtime_projection import (
    WebScientificRuntimeProjection,
    WebScientificRuntimeProjectionError,
    categorical_adjustments,
    operational_covariates,
    signed_projection,
)

#: The products every continuous suite declares, in its signed order.
CONTINUOUS_SURVIVAL_OUTPUTS = (
    "table:landmark_continuous_table_one",
    "table:landmark_continuous_risk_set_flow",
    "table:landmark_continuous_km_curve",
    "table:landmark_continuous_cox_summary",
    "table:landmark_continuous_ph_diagnostics",
    "table:landmark_continuous_time_varying_cox_summary",
    "table:landmark_continuous_spline_curve",
    "table:landmark_continuous_measurement_audit",
    "log:landmark_continuous_survival_receipt",
    "figure:landmark_continuous_survival_suite",
)
_SUMMARY_WORDS = {
    "max": "highest",
    "min": "lowest",
    "mean": "mean",
    "first": "first recorded",
}
_TIME_ORIGIN_LABELS = {"icu_admission": "ICU admission"}


def _hours_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}".replace(".", "p")


def _concept_entry(concept_id: str) -> tuple[str, str | None]:
    """The source concept's reader name and the unit its values are recorded in."""

    entry = CONCEPT_DICTIONARY.get(concept_id)
    if not isinstance(entry, tuple) or not entry or not str(entry[0]).strip():
        return concept_id, None
    unit = str(entry[2]).strip() if len(entry) > 2 and entry[2] is not None else ""
    return str(entry[0]).strip(), (unit or None)


def _window_summary(
    *,
    universe_path: Path,
    column: str,
    source: str,
    landmark_hours: float,
) -> str:
    """The summary the exposure column records over hours 0 to the landmark.

    Verified metadata must describe a numeric summary of the source over that
    exact window.  Without metadata (a zero-row planning catalog) the column
    must carry the materializer's ``<source>_<summary>`` name; the formal run's
    metadata then proves the window.
    """

    if concept_records_one_value_per_stay(source):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_continuous_survival_exposure_untimed",
            "The continuous survival suite models a value recorded by the "
            "landmark; this exposure is recorded once per stay, with no time.",
            details={"primary_exposure": column, "primary_exposure_source": source},
        )
    try:
        verified = load_verified_materialized_cohort_authority(Path(universe_path))
    except MaterializedMetadataError as exc:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_metadata_unverified",
            "The materialized universe's column metadata could not be verified "
            "for the survival exposure window.",
            details={"artifact": Path(universe_path).name, "reason": str(exc)[:500]},
        ) from exc
    if verified is None:
        summary = column[len(source) + 1:] if column.startswith(f"{source}_") else ""
        if summary not in CONTINUOUS_EXPOSURE_WINDOW_SUMMARIES:
            raise WebScientificRuntimeProjectionError(
                "web_landmark_continuous_survival_exposure_window_unverified",
                "The continuous survival exposure is not a window summary the "
                "materializer names for its source.",
                details={"primary_exposure": column, "primary_exposure_source": source},
            )
        return summary
    binding = next(
        (
            file_binding.columns.get(column)
            for file_binding in verified.sidecar.files
            if file_binding.columns.get(column) is not None
        ),
        None,
    )
    window = getattr(binding, "derivation_window", None)
    summary = str(getattr(getattr(binding, "metadata", None), "aggregation", "") or "")
    if (
        binding is None
        or binding.metadata.role is not ConceptColumnRole.NUMERIC_AGGREGATE
        or summary not in CONTINUOUS_EXPOSURE_WINDOW_SUMMARIES
        or binding.representation_transform != f"window_numeric_{summary}"
        or window is None
        or window.origin != "icu_admission"
        or float(window.start_hours) != 0.0
        or float(window.end_hours) != float(landmark_hours)
    ):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_continuous_survival_exposure_window_unverified",
            "The continuous survival exposure is not a numeric summary of its "
            "source recorded from ICU admission to the landmark.",
            details={
                "primary_exposure": column,
                "landmark_hours": landmark_hours,
                "derivation_window": (
                    None
                    if window is None
                    else [window.origin, float(window.start_hours), float(window.end_hours)]
                ),
                "representation_transform": getattr(binding, "representation_transform", None),
            },
        )
    return summary


def compile_landmark_continuous_survival_runtime_projection(
    *,
    study: Mapping[str, Any],
    endpoint: Any,
    landmark_hours: float,
    analysis_unit_label: str,
    primary_exposure: str,
    primary_exposure_source: str,
    declared_covariates: Sequence[str],
    covariate_operationalizations: Mapping[str, str],
    universe_path: Path,
    scientific_configuration_sha256: str,
) -> WebScientificRuntimeProjection:
    """Bind a validated survival landmark design with a continuous exposure."""

    summary = _window_summary(
        universe_path=universe_path,
        column=primary_exposure,
        source=primary_exposure_source,
        landmark_hours=landmark_hours,
    )
    tau = endpoint.horizon_days - landmark_hours / 24.0
    cutpoints = [
        float(value) for value in endpoint.time_varying_cutpoints_days if 0 < value < tau
    ]
    if not cutpoints:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_continuous_survival_followup_unsupported",
            "The continuous survival suite needs a follow-up interval cutpoint "
            "after the landmark for its prespecified interval model.",
            details={
                "landmark_hours": landmark_hours,
                "endpoint_horizon_days": endpoint.horizon_days,
            },
        )
    covariates = operational_covariates(
        universe_path,
        declared_covariates=tuple(declared_covariates),
        operationalizations=covariate_operationalizations,
    )
    categorical = categorical_adjustments(universe_path, covariates=covariates)
    required = [
        primary_exposure,
        endpoint.event_concept,
        endpoint.followup_concept,
        *covariates,
    ]
    if len(required) != len(set(required)):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_columns_ambiguous",
            "A survival design coordinate is bound to more than one role.",
            details={"columns": required},
        )
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.read_schema(universe_path).names)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": Path(universe_path).name, "reason": str(exc)[:500]},
        ) from exc
    absent = sorted(set(required) - schema_names)
    if absent:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_columns_missing",
            "The continuous landmark survival runtime inputs are absent from the "
            "materialized universe.",
            details={"missing_columns": absent},
        )

    token = _hours_token(landmark_hours)
    name, unit = _concept_entry(primary_exposure_source)
    label = f"{name}, {_SUMMARY_WORDS[summary]} in hours 0 to {landmark_hours:g}"
    outputs = list(CONTINUOUS_SURVIVAL_OUTPUTS)
    authority_body: dict[str, Any] = {
        "schema_version": "easyicu.landmark_continuous_survival_runtime_authority/2",
        "authority_kind": "landmark_continuous_survival_suite",
        "protocol_content_sha256": scientific_configuration_sha256,
        "plan_method": "signed_landmark_continuous_survival_suite",
        "plan_intent": (
            f"Execute the signed {token}-hour landmark survival suite of {label} "
            f"per exposure step for {endpoint.horizon_days}-day mortality with PH auditing."
        ),
        "plan_outputs": outputs,
        "development_execution_only_allowed": False,
        "exposure_column": primary_exposure,
        "exposure_label": label,
        "exposure_unit": unit,
        "exposure_window_summary": summary,
        "exposure_window_hours": [0.0, float(landmark_hours)],
        "exposure_increment_rule": "largest_round_step_within_interquartile_range",
        "event_column": endpoint.event_concept,
        "followup_time_column": endpoint.followup_concept,
        "endpoint_time_origin": _TIME_ORIGIN_LABELS[endpoint.time_origin],
        "endpoint_censoring_rule": endpoint.censoring_rule,
        "landmark_hours": float(landmark_hours),
        "endpoint_horizon_days": float(endpoint.horizon_days),
        "analysis_unit_label": analysis_unit_label,
        "derived_event_column": f"death_after_{token}h_by_day{endpoint.horizon_days}",
        "derived_time_column": f"followup_days_from_{token}h_landmark",
        "adjustment_columns": list(covariates),
        "categorical_adjustment_columns": list(categorical),
        "table_one_columns": list(covariates),
        "estimator": "cox_ph_lifelines_efron",
        "effect_measure": "hazard_ratio_per_exposure_step",
        "uncertainty_method": "wald_95_ci",
        "proportional_hazards_diagnostic": "schoenfeld_residual_test",
        "proportional_hazards_alpha": 0.05,
        "proportional_hazards_policy": "block_paper_authorization",
        "time_varying_effect_method": "piecewise_time_varying_cox",
        "time_varying_interval_cutpoints_days": cutpoints,
        "spline_knot_quantiles": [0.1, 0.5, 0.9],
        "spline_reference": "median_in_model_population",
        "curve_quantile_range": [0.1, 0.9],
        "curve_points": 41,
        "functional_form_alpha": 0.05,
        "functional_form_policy": "spline_contrasts_replace_linear_estimate",
        "descriptive_grouping": "value_tertiles",
        "interpretation": "descriptive_prognostic_association_not_causal",
        "table_one_product": outputs[0],
        "risk_set_product": outputs[1],
        "km_product": outputs[2],
        "cox_product": outputs[3],
        "ph_product": outputs[4],
        "time_varying_cox_product": outputs[5],
        "spline_product": outputs[6],
        "measurement_audit_product": outputs[7],
        "receipt_product": outputs[8],
        "figure_product": outputs[9],
    }
    authority = build_current_case_scientific_runtime_authority(
        authority_body
    ).model_dump(mode="json")
    return signed_projection(
        authority, scientific_configuration_sha256=scientific_configuration_sha256
    )


__all__ = [
    "CONTINUOUS_SURVIVAL_OUTPUTS",
    "compile_landmark_continuous_survival_runtime_projection",
]
