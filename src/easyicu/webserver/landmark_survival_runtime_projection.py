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
  is the producer-owned ``<concept>_onset_time`` companion: the first time the
  source recorded the exposure as present, in hours from ICU admission
  (``ConceptColumnRole.EVENT_TIME``, ``first_truthy_event_time``);
* the endpoint is one closed fixed-horizon mortality concept whose paired
  ``followup_days_<h>d`` time and horizon come from
  ``easyicu.outcome_availability``;
* the landmark, its eligibility flags and the follow-up binding come from the
  study's single landmark sensitivity; the adjustment roster is the exact
  StudyContext roster;
* the prevalence-definition sensitivity analysis re-fits at the hours its
  sealed rule gives for the exposure window (``sealed_suite_robustness``).

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
from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.research_agent.concept_availability import concept_records_one_value_per_stay
from easyicu.research_agent.contracts.sealed_suite_robustness import (
    EXPOSURE_ONSET_HOURS_PRODUCT,
    PREVALENCE_SENSITIVITY_PRODUCT,
    PREVALENCE_SENSITIVITY_RULE,
    prevalence_sensitivity_cutoffs_hours,
)
from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.planning.analysis_types import canonical_analysis_family
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
    normalize_prespecified_sensitivities,
)

from . import primary_cohort
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
#: Producer-owned onset companion of a typed event status: the first time the
#: materialization window recorded it present.  The first-observation
#: companion (``_first_time``) can be an absent record and is not an onset.
_ONSET_SUFFIX = "_onset_time"
_ONSET_REPRESENTATION = "first_truthy_event_time"
_ANALYSIS_UNIT_LABELS = {"icu_stay": "ICU stays"}
#: One row per patient once the host keeps each patient's first ICU stay. The
#: label is a reader noun phrase: claims and the cohort fact read it mid-sentence
#: ("the first ICU stays alive and under observation at the landmark").
_FIRST_STAY_UNIT_LABEL = "first ICU stays"
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


def _require_present_onset(universe_path: Path, column: str) -> None:
    """The onset column must be the materializer's first-present event time.

    The suite classifies prevalent and incident exposure by this column, so a
    column of another meaning under the onset name would misclassify both.  A
    universe without materialized metadata, such as a zero-row planning
    catalog, binds the column by its name: the materializer publishes it only
    for a typed event status.
    """

    try:
        verified = load_verified_materialized_cohort_authority(Path(universe_path))
    except MaterializedMetadataError as exc:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_metadata_unverified",
            "The materialized universe's column metadata could not be verified "
            "for the survival exposure onset.",
            details={"artifact": Path(universe_path).name, "reason": str(exc)[:500]},
        ) from exc
    if verified is None:
        return
    binding = next(
        (
            file_binding.columns.get(column)
            for file_binding in verified.sidecar.files
            if file_binding.columns.get(column) is not None
        ),
        None,
    )
    if (
        binding is not None
        and binding.metadata.role is ConceptColumnRole.EVENT_TIME
        and binding.representation_transform == _ONSET_REPRESENTATION
    ):
        return
    raise WebScientificRuntimeProjectionError(
        "web_landmark_survival_onset_unverified",
        "The survival exposure onset is not the materializer's first record of "
        "the exposure as present.",
        details={"onset_column": column},
    )


def _supported_endpoint(target_outcome: str | None) -> Any:
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
    return endpoint


def _declaration_missing_fields(
    study: Mapping[str, Any], landmark: PrespecifiedSensitivitySpec
) -> list[str]:
    missing_fields: list[str] = []
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
    return missing_fields


def _incomplete(missing_fields: list[str]) -> None:
    raise WebScientificRuntimeProjectionError(
        "web_landmark_survival_authority_incomplete",
        "The landmark survival design lacks executable typed coordinates.",
        details={"missing_fields": missing_fields},
    )


def _declared_landmark_hours(
    study: Mapping[str, Any], landmark: PrespecifiedSensitivitySpec, endpoint: Any
) -> float:
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
    if not survival_inference_supported(design):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_design_unsupported",
            "The sealed survival suite fits one model-based Cox model per ICU "
            "stay; the declared analysis unit or variance estimator has no "
            "executable owner.",
            details={
                "analysis_unit": str(design.get("analysis_unit") or ""),
                "variance_estimator": str(design.get("variance_estimator") or ""),
                "supported_analysis_unit": _SUPPORTED_ANALYSIS_UNIT,
                "supported_variance_estimator": _SUPPORTED_VARIANCE_ESTIMATOR,
                "field": "analysis_design",
            },
        )
    return landmark_hours


def _declared_landmark(
    study: Mapping[str, Any],
) -> PrespecifiedSensitivitySpec | None:
    try:
        specs = normalize_prespecified_sensitivities(study.get("sensitivity_specs"))
    except ValueError as exc:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_authority_incomplete",
            "The study's sensitivity specifications are not typed.",
            details={"field": "sensitivity_specs", "reason": str(exc)[:500]},
        ) from exc
    return one_sensitivity_spec(specs, strategy="landmark")


def validate_landmark_survival_declaration(
    study: Mapping[str, Any],
) -> float | None:
    """The source-independent half of this owner's contract.

    A caller that writes a survival design checks it here, before a launch
    spends anything, instead of restating the policy.  Returns the declared
    landmark in hours, or ``None`` for a study that declares no survival
    family or no landmark (the projection's ``None`` routes).  The exposure
    kind and the materialized columns are checked by the projection itself.
    """

    if not survival_family_declared(study):
        return None
    landmark = _declared_landmark(study)
    if landmark is None:
        return None
    execution = study.get("execution_concepts")
    execution = execution if isinstance(execution, Mapping) else {}
    endpoint = _supported_endpoint(execution.get("outcome"))
    missing_fields = [
        *(
            []
            if str(execution.get("primary_exposure") or "").strip()
            else ["execution_concepts.primary_exposure"]
        ),
        *_declaration_missing_fields(study, landmark),
    ]
    if missing_fields:
        _incomplete(missing_fields)
    return _declared_landmark_hours(study, landmark, endpoint)


def survival_exposure_onset_column(
    study: Mapping[str, Any],
    *,
    sensitivity_specs: Sequence[PrespecifiedSensitivitySpec],
    primary_exposure_source: str | None,
) -> str | None:
    """The onset column a declared landmark survival design binds, else ``None``.

    Formal materialization emits the producer-owned onset companion for every
    typed event-status exposure; a zero-row planning catalog lists only the operational columns
    the host binds.  Naming the onset here lets a candidate plan bind the
    suite without reading patient rows.
    """

    source = str(primary_exposure_source or "").strip()
    landmarks = [spec for spec in sensitivity_specs if spec.strategy == "landmark"]
    if not source or len(landmarks) != 1 or not survival_family_declared(study):
        return None
    return f"{source}{_ONSET_SUFFIX}"


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

    endpoint = _supported_endpoint(target_outcome)

    missing_fields: list[str] = []
    if not primary_exposure:
        missing_fields.append("primary_exposure")
    if not primary_exposure_source:
        missing_fields.append("primary_exposure_source")
    if not target_is_event_status:
        missing_fields.append("binary_event_status_outcome")
    missing_fields.extend(_declaration_missing_fields(study, landmark))
    if missing_fields:
        _incomplete(missing_fields)
    landmark_hours = _declared_landmark_hours(study, landmark, endpoint)

    exposure_kind, _levels = primary_exposure_kind(
        universe_path=universe_path,
        primary_exposure=primary_exposure,
        primary_exposure_source=primary_exposure_source,
    )
    analysis_unit_label = (
        _FIRST_STAY_UNIT_LABEL
        if primary_cohort.first_icu_stay_only(study.get("cohort"))
        else _ANALYSIS_UNIT_LABELS[_SUPPORTED_ANALYSIS_UNIT]
    )
    if exposure_kind == VariableKind.CONTINUOUS:
        # A continuous exposure is modelled per unit by its own sealed suite,
        # on the coordinates validated above.
        from .landmark_continuous_survival_runtime_projection import (
            compile_landmark_continuous_survival_runtime_projection,
        )

        return compile_landmark_continuous_survival_runtime_projection(
            study=study,
            endpoint=endpoint,
            landmark_hours=landmark_hours,
            analysis_unit_label=analysis_unit_label,
            primary_exposure=str(primary_exposure),
            primary_exposure_source=str(primary_exposure_source),
            declared_covariates=declared_covariates,
            covariate_operationalizations=covariate_operationalizations,
            universe_path=universe_path,
            scientific_configuration_sha256=scientific_configuration_sha256,
        )
    if exposure_kind != VariableKind.BINARY:
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_exposure_incompatible",
            "The sealed survival suites contrast one binary incident exposure "
            "against its comparator or model one continuous exposure per unit.",
            details={
                "primary_exposure": primary_exposure,
                "exposure_kind": exposure_kind.value,
            },
        )
    if concept_records_one_value_per_stay(str(primary_exposure_source)):
        raise WebScientificRuntimeProjectionError(
            "web_landmark_survival_exposure_incompatible",
            "The sealed survival suite times the exposure by its first record as "
            "present; this exposure is recorded once per stay.",
            details={
                "primary_exposure": primary_exposure,
                "primary_exposure_source": primary_exposure_source,
                "exposure_timing": "one_value_per_stay",
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
    _require_present_onset(universe_path, onset_column)

    landmark_token = _hours_token(landmark_hours)
    exposure_name = _concept_display_name(str(primary_exposure_source))
    tau = endpoint.horizon_days - landmark_hours / 24.0
    cutpoints = [
        float(value)
        for value in endpoint.time_varying_cutpoints_days
        if 0 < value < tau
    ]
    prevalence_cutoffs = prevalence_sensitivity_cutoffs_hours(landmark_hours)
    outputs = [
        "table:landmark_table_one",
        "table:landmark_risk_set_flow",
        "table:landmark_km_curve",
        "table:landmark_cox_summary",
        "table:landmark_ph_diagnostics",
        "table:landmark_rmst_summary",
        *(["table:landmark_time_varying_cox_summary"] if cutpoints else []),
        *(
            [PREVALENCE_SENSITIVITY_PRODUCT, EXPOSURE_ONSET_HOURS_PRODUCT]
            if prevalence_cutoffs
            else []
        ),
        "table:landmark_measurement_audit",
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
        "exposure_onset_representation": _ONSET_REPRESENTATION,
        "event_column": endpoint.event_concept,
        "followup_time_column": endpoint.followup_concept,
        "endpoint_time_origin": _TIME_ORIGIN_LABELS[endpoint.time_origin],
        "endpoint_censoring_rule": endpoint.censoring_rule,
        "landmark_hours": landmark_hours,
        "endpoint_horizon_days": float(endpoint.horizon_days),
        "exposure_window_hours": [0.0, landmark_hours],
        "prevalent_exposure_cutoff_hours": 0.0,
        "prevalent_exposure_action": "exclude",
        **(
            {
                "prevalence_sensitivity_rule": PREVALENCE_SENSITIVITY_RULE,
                "prevalence_sensitivity_cutoffs_hours": list(prevalence_cutoffs),
                "prevalence_sensitivity_product": PREVALENCE_SENSITIVITY_PRODUCT,
                "exposure_onset_hours_product": EXPOSURE_ONSET_HOURS_PRODUCT,
            }
            if prevalence_cutoffs
            else {}
        ),
        "exposed_group_label": f"Incident {exposure_name} by {landmark_token} h",
        "comparator_group_label": (
            f"No incident {exposure_name} by {landmark_token} h"
        ),
        "analysis_unit_label": analysis_unit_label,
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
        "measurement_audit_product": "table:landmark_measurement_audit",
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
    "survival_exposure_onset_column",
    "survival_family_declared",
    "survival_inference_supported",
    "validate_landmark_survival_declaration",
]
