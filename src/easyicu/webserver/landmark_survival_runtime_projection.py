"""Web adapter for the sealed fixed-landmark survival suite.

Owner
-----
This sibling of ``scientific_runtime_projection`` compiles a StudyContext whose
``analysis_design.analysis_family`` is ``survival`` into the digest-bound
``LandmarkSurvivalRuntimeAuthority``.  The suite owns the risk set, Table 1,
Kaplan-Meier summary, adjusted Cox fit, PH audit, RMST/time-varying
alternatives and the composite figure; the Planner only labels the reader
display.  Every scientific coordinate comes from typed host vocabularies:

* the exposure status column is the user's event-status exposure and its onset
  is the producer-owned ``<concept>_first_time`` companion (hours from ICU
  admission, ``ConceptColumnRole.FIRST_OBSERVATION_TIME``);
* the endpoint is one closed fixed-horizon mortality concept whose paired
  ``followup_days_<h>d`` time and horizon come from
  ``easyicu.outcome_availability``;
* the landmark, its eligibility flags and the follow-up binding come from the
  study's single landmark sensitivity; the adjustment roster is the exact
  StudyContext roster.

It reads the materialized universe *schema* only.  Anything the vocabularies
cannot close fails with ``WebScientificRuntimeProjectionError`` instead of
being inferred from prose or from patient rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from easyicu.concept.catalog import CONCEPT_DICTIONARY
from easyicu.outcome_availability import (
    FIXED_HORIZON_MORTALITY_ENDPOINTS,
    fixed_horizon_mortality_endpoint,
)
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.planning.analysis_types import canonical_analysis_family
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)

from .scientific_runtime_projection import (
    WebScientificRuntimeProjection,
    WebScientificRuntimeProjectionError,
    categorical_adjustments,
    one_sensitivity_spec,
    operational_covariates,
    primary_exposure_kind,
    signed_projection,
)

#: The sealed suite fits one Efron Cox model per ICU stay with Wald intervals;
#: a study that declares another unit or variance estimator has no executable
#: owner here and must not be silently re-modelled.
_SUPPORTED_ANALYSIS_UNIT = "icu_stay"
_SUPPORTED_VARIANCE_ESTIMATOR = "model_based"
#: Producer-owned first-observation companion of a materialized concept.
_ONSET_SUFFIX = "_first_time"
_ANALYSIS_UNIT_LABELS = {"icu_stay": "ICU stays"}
_TIME_ORIGIN_LABELS = {"icu_admission": "ICU admission"}


def survival_family_declared(study: Mapping[str, Any]) -> bool:
    """Whether the typed analysis design names the survival family."""

    design = study.get("analysis_design")
    if not isinstance(design, Mapping):
        return False
    return canonical_analysis_family(design.get("analysis_family")) == "survival"


def survival_inference_supported(design: Mapping[str, Any]) -> bool:
    """Whether the sealed suite executes this analysis unit and variance estimator."""

    return (
        str(design.get("analysis_unit") or "") == _SUPPORTED_ANALYSIS_UNIT
        and str(design.get("variance_estimator") or "")
        == _SUPPORTED_VARIANCE_ESTIMATOR
    )


def _hours_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}".replace(".", "p")


def _concept_display_name(concept_id: str) -> str:
    entry = CONCEPT_DICTIONARY.get(concept_id)
    if isinstance(entry, tuple) and entry and str(entry[0]).strip():
        return str(entry[0]).strip()
    return concept_id


def _schema_names(universe_path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(universe_path).names)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for runtime binding.",
            details={"artifact": universe_path.name, "reason": str(exc)[:500]},
        ) from exc


def compile_landmark_survival_runtime_projection(
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
    literature_citation_keys: Sequence[str] = (),
    direct_comparator_literature_keys: Sequence[str] = (),
    dependence: Any = None,
) -> WebScientificRuntimeProjection | None:
    """Bind a survival-family landmark study to the sealed survival suite.

    Returns ``None`` when the study does not declare the survival family or
    declares no landmark, so the remaining projections keep their routes.
    """

    if not survival_family_declared(study):
        return None
    landmark = one_sensitivity_spec(sensitivity_specs, strategy="landmark")
    if landmark is None:
        return None

    endpoint = fixed_horizon_mortality_endpoint(str(target_outcome or ""))
    if endpoint is None:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_endpoint_unsupported",
            "The sealed survival suite requires one fixed-horizon mortality "
            "endpoint with its paired follow-up time concept.",
            details={
                "target_outcome": target_outcome,
                "supported_endpoints": sorted(FIXED_HORIZON_MORTALITY_ENDPOINTS),
                "field": "execution_concepts.outcome",
            },
        )

    missing_fields: list[str] = []
    if not primary_exposure:
        missing_fields.append("primary_exposure")
    if not primary_exposure_source:
        missing_fields.append("primary_exposure_source")
    if not target_is_event_status:
        missing_fields.append("binary_event_status_outcome")
    # The suite seals its roster before planning; a Planner-selectable roster
    # has no executable owner in this schema version.
    if str(study.get("covariate_selection") or "") != "exact":
        missing_fields.append("covariate_selection=exact")
    if not landmark.require_alive_at_landmark:
        missing_fields.append("landmark.require_alive_at_landmark")
    if not landmark.exclude_negative_event_times:
        missing_fields.append("landmark.exclude_negative_event_times")
    if landmark.observation_duration_variable is None:
        missing_fields.append("landmark.observation_duration_variable")
    if missing_fields:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_authority_incomplete",
            "The landmark survival design lacks executable typed coordinates.",
            details={"missing_fields": missing_fields},
        )
    if (
        landmark.observation_duration_variable != endpoint.followup_concept
        or landmark.observation_duration_unit != endpoint.followup_unit
    ):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_followup_binding_mismatch",
            "The landmark follow-up binding differs from the endpoint's paired "
            "event/censoring time concept.",
            details={
                "target_outcome": endpoint.event_concept,
                "required_observation_duration_variable": endpoint.followup_concept,
                "required_observation_duration_unit": endpoint.followup_unit,
                "declared_observation_duration_variable": (
                    landmark.observation_duration_variable
                ),
                "declared_observation_duration_unit": (
                    landmark.observation_duration_unit
                ),
                "field": "sensitivity_specs",
            },
        )
    landmark_hours = float(landmark.landmark_hours or 0.0)
    if landmark_hours <= 0 or landmark_hours / 24.0 >= endpoint.horizon_days:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_landmark_unsupported",
            "The landmark must fall inside the fixed endpoint horizon.",
            details={
                "landmark_hours": landmark.landmark_hours,
                "endpoint_horizon_days": endpoint.horizon_days,
            },
        )

    design = study.get("analysis_design")
    design = design if isinstance(design, Mapping) else {}
    analysis_unit = str(design.get("analysis_unit") or "")
    variance_estimator = str(design.get("variance_estimator") or "")
    if not survival_inference_supported(design):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_design_unsupported",
            "The sealed survival suite fits one model-based Cox model per ICU "
            "stay; the declared analysis unit or variance estimator has no "
            "executable owner.",
            details={
                "analysis_unit": analysis_unit,
                "variance_estimator": variance_estimator,
                "supported_analysis_unit": _SUPPORTED_ANALYSIS_UNIT,
                "supported_variance_estimator": _SUPPORTED_VARIANCE_ESTIMATOR,
                "field": "analysis_design",
            },
        )

    exposure_kind, _levels = primary_exposure_kind(
        universe_path=universe_path,
        primary_exposure=primary_exposure,
        primary_exposure_source=primary_exposure_source,
    )
    if exposure_kind != VariableKind.BINARY:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_exposure_incompatible",
            "The sealed survival suite contrasts one binary incident exposure "
            "against its comparator.",
            details={
                "primary_exposure": primary_exposure,
                "exposure_kind": exposure_kind.value,
            },
        )
    onset_column = f"{primary_exposure_source}{_ONSET_SUFFIX}"
    covariates = operational_covariates(
        universe_path,
        declared_covariates=tuple(declared_covariates),
        operationalizations=covariate_operationalizations,
    )
    categorical = categorical_adjustments(universe_path, covariates=covariates)
    required_columns = [
        str(primary_exposure),
        onset_column,
        endpoint.event_concept,
        endpoint.followup_concept,
        *covariates,
    ]
    if len(required_columns) != len(set(required_columns)):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_columns_ambiguous",
            "A survival design coordinate is bound to more than one role.",
            details={"columns": required_columns},
        )
    absent = sorted(set(required_columns) - _schema_names(universe_path))
    if absent:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_columns_missing",
            "The landmark survival runtime inputs are absent from the "
            "materialized universe.",
            details={"missing_columns": absent},
        )

    landmark_token = _hours_token(landmark_hours)
    exposure_name = _concept_display_name(str(primary_exposure_source))
    tau = endpoint.horizon_days - landmark_hours / 24.0
    cutpoints = [
        float(value)
        for value in endpoint.time_varying_cutpoints_days
        if 0 < value < tau
    ]
    outputs = [
        "table:landmark_table_one",
        "table:landmark_risk_set_flow",
        "table:landmark_km_curve",
        "table:landmark_cox_summary",
        "table:landmark_ph_diagnostics",
        "table:landmark_rmst_summary",
        *(["table:landmark_time_varying_cox_summary"] if cutpoints else []),
        "log:landmark_survival_receipt",
        "figure:landmark_survival_suite",
    ]
    authority_body: dict[str, Any] = {
        "schema_version": "easyicu.landmark_survival_runtime_authority/1",
        "authority_kind": "landmark_survival_suite",
        "protocol_content_sha256": scientific_configuration_sha256,
        "plan_method": "signed_landmark_survival_suite",
        "development_execution_only_allowed": False,
        "plan_intent": (
            f"Execute the signed {landmark_token}-hour landmark {exposure_name} "
            f"survival suite for {endpoint.horizon_days}-day mortality with "
            "explicit prevalent-exposure exclusion and PH auditing."
        ),
        "plan_outputs": outputs,
        "exposure_status_column": primary_exposure,
        "exposure_onset_column": onset_column,
        "event_column": endpoint.event_concept,
        "followup_time_column": endpoint.followup_concept,
        "endpoint_time_origin": _TIME_ORIGIN_LABELS[endpoint.time_origin],
        "endpoint_censoring_rule": endpoint.censoring_rule,
        "landmark_hours": landmark_hours,
        "endpoint_horizon_days": float(endpoint.horizon_days),
        "exposure_window_hours": [0.0, landmark_hours],
        "prevalent_exposure_cutoff_hours": 0.0,
        "prevalent_exposure_action": "exclude",
        "exposed_group_label": f"Incident {exposure_name} by {landmark_token} h",
        "comparator_group_label": (
            f"No incident {exposure_name} by {landmark_token} h"
        ),
        "analysis_unit_label": _ANALYSIS_UNIT_LABELS[analysis_unit],
        "derived_exposure_column": (
            f"incident_{primary_exposure_source}_by_{landmark_token}h"
        ),
        "derived_event_column": (
            f"death_after_{landmark_token}h_by_day{endpoint.horizon_days}"
        ),
        "derived_time_column": f"followup_days_from_{landmark_token}h_landmark",
        "adjustment_columns": list(covariates),
        "categorical_adjustment_columns": list(categorical),
        "table_one_columns": list(covariates),
        "estimator": "cox_ph_lifelines_efron",
        "effect_measure": "hazard_ratio",
        "uncertainty_method": "wald_95_ci",
        "proportional_hazards_diagnostic": "schoenfeld_residual_test",
        "proportional_hazards_alpha": 0.05,
        "proportional_hazards_policy": "block_paper_authorization",
        "non_ph_alternative": "unadjusted_rmst_difference",
        "time_varying_effect_method": (
            "piecewise_time_varying_cox" if cutpoints else None
        ),
        "time_varying_interval_cutpoints_days": cutpoints,
        "interpretation": "descriptive_prognostic_association_not_causal",
        "table_one_product": outputs[0],
        "risk_set_product": outputs[1],
        "km_product": outputs[2],
        "cox_product": outputs[3],
        "ph_product": outputs[4],
        "rmst_product": outputs[5],
        "time_varying_cox_product": (
            "table:landmark_time_varying_cox_summary" if cutpoints else None
        ),
        "receipt_product": "log:landmark_survival_receipt",
        "figure_product": "figure:landmark_survival_suite",
    }
    authority = build_current_case_scientific_runtime_authority(
        authority_body
    ).model_dump(mode="json")
    return signed_projection(
        authority, scientific_configuration_sha256=scientific_configuration_sha256
    )


__all__ = [
    "compile_landmark_survival_runtime_projection",
    "survival_family_declared",
    "survival_inference_supported",
]
