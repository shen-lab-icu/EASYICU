"""Scientific configuration and materialization facts for pipeline launch."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.webserver import dataio, source_identity_authority
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.study_scientific_configuration import (
    ScientificConfiguration,
    ScientificConfigurationError,
)

_MATERIALIZED_FEATURE_SUFFIXES = tuple(
    sorted(
        (
            "_first_time",
            "_last_time",
            "_measured",
            "_first",
            "_mean",
            "_max",
            "_min",
            "_n",
        ),
        key=len,
        reverse=True,
    )
)


def _clean_text(value: Any, limit: int = 1_200) -> str:
    return re.sub(r"\\s+", " ", str(value or "")).strip()[:limit]


def _target_outcome(study: Mapping[str, Any]) -> Optional[str]:
    return ScientificConfiguration.inspect(study).target_outcome()


def _primary_exposure(study: Mapping[str, Any]) -> Optional[str]:
    return ScientificConfiguration.inspect(study).primary_exposure()


def _primary_exposure_aggregation(study: Mapping[str, Any]) -> Optional[str]:
    """Return the StudyContext-owned repeated-measure aggregation coordinate."""

    return ScientificConfiguration.inspect(study).primary_exposure_aggregation()


def _configured_covariates(study: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        return ScientificConfiguration.inspect(study).covariates()
    except ScientificConfigurationError as exc:
        raise ResearchPipelineRunError(exc.code, str(exc), details=exc.details) from exc


def _configured_covariate_selection(study: Mapping[str, Any]) -> str:
    """Return the one validated owner coordinate for adjustment authority."""

    try:
        return ScientificConfiguration.inspect(study).covariate_selection()
    except ScientificConfigurationError as exc:
        raise ResearchPipelineRunError(exc.code, str(exc), details=exc.details) from exc


def _configured_sensitivity_specs(study: Mapping[str, Any]) -> tuple[Any, ...]:
    """Load only the typed sensitivity authority owned by StudyContext."""

    try:
        specs = ScientificConfiguration.inspect(study).sensitivity_specs()
    except ScientificConfigurationError as exc:
        raise ResearchPipelineRunError(exc.code, str(exc), details=exc.details) from exc

    analysis_design = study.get("analysis_design")
    if not isinstance(analysis_design, Mapping):
        return specs
    primary_is_patient_clustered = (
        str(analysis_design.get("analysis_unit") or "").strip().lower()
        == "icu_stay"
        and str(analysis_design.get("variance_estimator") or "").strip().lower()
        == "cluster_robust"
        and str(analysis_design.get("cluster_unit") or "").strip().lower()
        == "patient"
    )
    if not primary_is_patient_clustered:
        return specs

    # Older Copilot builds recorded the primary dependence model a second
    # time as a requested sensitivity.  The Research Agent then correctly
    # asked for a separate execution that has no distinct estimand.  Normalize
    # that legacy state at the host boundary while preserving real repeated-
    # stay sensitivities such as first-stay or non-readmission restrictions.
    return tuple(
        spec
        for spec in specs
        if not (
            str(getattr(spec, "axis", "") or "") == "repeated_stays"
            and str(getattr(spec, "strategy", "") or "") == "cluster_robust"
        )
    )


def _runtime_projection_sensitivity_specs(
    sensitivity_specs: tuple[Any, ...],
    *,
    primary_exposure_source: str,
) -> tuple[Any, ...]:
    """Add only the deterministic runtime's automatic nonlinear safeguard.

    A user-selected landmark must close exposure opportunity in the primary
    estimator.  When the researcher has not requested a competing functional-
    form sensitivity, the signed landmark runtime supplies its standard RCS
    primary plus linear sensitivity as a plan-owned automatic remediation.  It
    is not written back to StudyContext or projected as a user request.
    """

    if not primary_exposure_source:
        return sensitivity_specs
    strategies = {
        str(getattr(item, "strategy", "") or "") for item in sensitivity_specs
    }
    axes = {str(getattr(item, "axis", "") or "") for item in sensitivity_specs}
    if "landmark" not in strategies or "functional_form" in axes:
        return sensitivity_specs
    from easyicu.research_agent.planning.sensitivity_authority import (
        PrespecifiedSensitivitySpec,
    )

    automatic = PrespecifiedSensitivitySpec(
        spec_id="easyicu_auto_primary_exposure_rcs",
        axis="functional_form",
        strategy="restricted_cubic_spline",
        execution_variables=(primary_exposure_source,),
    )
    return (*sensitivity_specs, automatic)


def _patient_grouping_for_analysis_design(
    study: Mapping[str, Any],
) -> Optional[PatientGroupingBinding]:
    raw = study.get("analysis_design")
    design = raw if isinstance(raw, Mapping) else {}
    if _clean_text(design.get("variance_estimator"), 80) != "cluster_robust":
        return None
    cluster_unit = _clean_text(design.get("cluster_unit"), 80)
    if cluster_unit != "patient":
        raise ResearchPipelineRunError(
            "research_pipeline_cluster_unit_unsupported",
            "The current Web runner supports cluster-robust inference only for a verified patient grouping.",
            details={
                "cluster_unit": cluster_unit or None,
                "supported_cluster_units": ["patient"],
            },
        )
    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    export_path = _clean_text(source.get("path"), 2_000)
    database = _clean_text(source.get("database"), 80)
    if not export_path or not database:
        return None
    try:
        return source_identity_authority.resolve_patient_grouping_authority(
            export_path=export_path,
            database=database,
        )
    except source_identity_authority.PatientGroupingAuthorityError as exc:
        raise ResearchPipelineRunError(
            exc.code,
            str(exc),
            details=exc.details,
        ) from exc


def _validate_analysis_design(study: Mapping[str, Any]) -> Dict[str, str]:
    """Fail closed on inference contracts the v1 Web runner cannot execute.

    This bridge must not translate an accepted robust/clustered request into an
    ordinary model-based fit.  StudyContext owns the semantic commitment; a
    future data-source adapter and association executor can add a digest-bound
    physical grouping coordinate without changing this case-neutral boundary.
    """

    raw = study.get("analysis_design")
    confirmations = study.get("confirmations")
    if (
        not raw
        and isinstance(confirmations, Mapping)
        and confirmations.get("plan_timing_descriptive_only") is True
    ):
        # Compatibility for a decision receipt written by hosts that saved the
        # descriptive ceiling before ``analysis_design`` became part of the
        # same atomic patch.  The confirmation is a typed, host-issued record
        # of the user's exact choice, not free-text inference.  Future clicks
        # persist this design directly in ``compile_plan_decision``.
        raw = {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        }
    if not raw:
        if _primary_exposure(study) and _target_outcome(study):
            raise ResearchPipelineRunError(
                "research_pipeline_analysis_design_required",
                (
                    "An exposure-outcome analysis requires a typed analysis "
                    "unit and variance estimator before pipeline launch."
                ),
                details={
                    "field": "analysis_design",
                    "required_fields": ["analysis_unit", "variance_estimator"],
                },
            )
        return {}
    if not isinstance(raw, Mapping):
        raise ResearchPipelineRunError(
            "research_pipeline_analysis_design_invalid",
            "The typed analysis design is invalid.",
            details={"field": "analysis_design"},
        )
    analysis_unit = _clean_text(raw.get("analysis_unit"), 80)
    variance_estimator = _clean_text(raw.get("variance_estimator"), 80)
    cluster_unit = _clean_text(raw.get("cluster_unit"), 80)
    if not analysis_unit or not variance_estimator:
        raise ResearchPipelineRunError(
            "research_pipeline_analysis_design_incomplete",
            "The typed analysis design is missing its analysis unit or variance estimator.",
            details={"field": "analysis_design"},
        )
    raw_cohort = study.get("cohort")
    cohort = raw_cohort if isinstance(raw_cohort, Mapping) else {}
    if cohort.get("exclude_readmissions") is True:
        raise ResearchPipelineRunError(
            "research_pipeline_first_stay_restriction_unverified",
            (
                "The selected export has an ICU-readmission indicator but no "
                "owner-verified first ICU stay per patient coordinate. The two "
                "are not interchangeable."
            ),
            details={
                "field": "cohort.exclude_readmissions",
                "first_stay_restriction_status": "unverified_in_selected_export",
                "icu_readmission_is_first_patient_stay_authority": False,
                "safe_alternatives": [
                    {
                        "id": "patient_clustered_all_stays",
                        "requires": "verified_patient_grouping",
                        "changes_scientific_question": False,
                    },
                    {
                        "id": "descriptive_only_without_independence_sensitive_inference",
                        "changes_scientific_question": True,
                    },
                ],
            },
        )
    dependence_finding = study_context_owner.analysis_dependence_finding(dict(study))
    if dependence_finding is not None:
        raise ResearchPipelineRunError(
            "research_pipeline_repeated_stay_dependence_unaddressed",
            (
                "Repeat ICU stays are retained, but the typed inference "
                "design does not address within-patient dependence."
            ),
            details={
                key: value
                for key, value in dependence_finding.items()
                if key != "error"
            },
        )
    if variance_estimator == "cluster_robust":
        grouping = _patient_grouping_for_analysis_design(study)
        if grouping is None:
            raise ResearchPipelineRunError(
                "research_pipeline_cluster_variance_unsupported",
                (
                    "This source and executor do not expose a verified grouping "
                    "coordinate for the requested cluster-robust inference."
                ),
                details={
                    "analysis_unit": analysis_unit,
                    "variance_estimator": variance_estimator,
                    "cluster_unit": cluster_unit or None,
                    "grouping_coordinate_status": "unavailable_or_unverified",
                    "first_stay_restriction_status": "unverified_in_selected_export",
                    "safe_alternatives": [
                        {
                            "id": "provide_verified_patient_grouping",
                            "executable_now": False,
                            "changes_scientific_question": False,
                        },
                        {
                            "id": "descriptive_only_without_independence_sensitive_inference",
                            "executable_now": True,
                            "changes_scientific_question": True,
                        },
                    ],
                },
            )
        return {
            "analysis_unit": analysis_unit,
            "variance_estimator": variance_estimator,
            "cluster_unit": "patient",
            "grouping_coordinate": grouping.output_identity_column,
        }
    if variance_estimator == "none_counts_only":
        return {
            "analysis_unit": analysis_unit,
            "variance_estimator": variance_estimator,
        }
    if variance_estimator != "model_based":
        raise ResearchPipelineRunError(
            "research_pipeline_variance_estimator_unsupported",
            "The current deterministic association executor does not implement the requested variance estimator.",
            details={
                "analysis_unit": analysis_unit,
                "variance_estimator": variance_estimator,
                "supported_variance_estimators": [
                    "model_based",
                    "none_counts_only",
                ],
            },
        )
    return {
        "analysis_unit": analysis_unit,
        "variance_estimator": variance_estimator,
    }


def validate_analysis_design_for_execution(
    study: Mapping[str, Any],
) -> Dict[str, str]:
    """Public, read-only capability gate for the current Web runner.

    Copilot uses this before spending a one-turn configuration grant so it can
    tell the user that a scientifically requested design is not executable by
    the selected source/runner.  The launch path calls the same owner logic
    again; this preview never weakens the authoritative launch gate.
    """

    return _validate_analysis_design(study)


def _analysis_requires_longitudinal_trajectory(
    study: Mapping[str, Any],
    *,
    validated_design: Mapping[str, str],
) -> bool:
    """Return whether the approved estimand needs row-level time trajectories."""

    for spec in _configured_sensitivity_specs(study):
        if spec.strategy == "landmark":
            return True
    return validated_design.get("variance_estimator") != "none_counts_only"


def _validate_primary_concept_selection(
    study: Mapping[str, Any],
    primary_exposure: Optional[str],
) -> None:
    """Enforce the concept owner's user-intent selection policy at launch."""

    if not primary_exposure:
        return
    from easyicu.concept.selection_policy import (
        concept_selection_confirmation_key,
        evaluate_concept_selection,
    )

    # Only the persisted scientific question can authorize an explicit-only
    # variant. Exposure labels and analysis prose are model-produced fields;
    # accepting them here would let a plan authorize its own semantic drift.
    intent = str(study.get("question") or "")
    confirmations = study.get("confirmations")
    confirmation_key = concept_selection_confirmation_key(primary_exposure)
    owner_confirmed = bool(
        isinstance(confirmations, Mapping) and confirmations.get(confirmation_key)
    )
    decision = evaluate_concept_selection(
        primary_exposure,
        user_intent=intent,
        owner_confirmed=owner_confirmed,
    )
    if decision.allowed:
        return
    raise ResearchPipelineRunError(
        decision.reason_code,
        (
            "The configured primary exposure is an explicit-only concept "
            "variant that the user did not request."
        ),
        details=decision.to_dict(),
    )


def _source_concept_for_operational_column(
    column: str,
    *,
    by_id: Mapping[str, Any],
) -> Optional[str]:
    """Resolve a wide materialized column back to its exported source concept."""

    if column in by_id:
        return column
    for suffix in _MATERIALIZED_FEATURE_SUFFIXES:
        if column.endswith(suffix):
            source_concept = column[: -len(suffix)]
            if source_concept in by_id:
                return source_concept
    return None


def _cohort_window(study: Mapping[str, Any]) -> tuple[float, float]:
    window_finding = study_context_owner.materialization_window_finding(dict(study))
    try:
        return ScientificConfiguration.inspect(study).materialization_window(
            window_finding=window_finding
        )
    except ScientificConfigurationError as exc:
        raise ResearchPipelineRunError(exc.code, str(exc), details=exc.details) from exc


def _configured_modules(study: Mapping[str, Any]) -> tuple[str, ...]:
    return ScientificConfiguration.inspect(study).modules()


# The cohort materializer interprets every outer-window offset from ICU
# admission (``study_contexts.materialization_window_finding``), so the anchor
# is the single executable value rather than a scientific choice.  The duration
# is EasyICU's standing outer feature window for an unplanned study.
_NEUTRAL_MATERIALIZATION_ANCHOR = "icu_admission"
_NEUTRAL_MATERIALIZATION_HOURS = 24.0


def _neutral_materialization_scope(
    study: Mapping[str, Any], *, export_path: str
) -> Dict[str, Any]:
    """Fill only the *materialization scope* a Planner-only run cannot infer.

    Owner note: this answers "what may EasyICU load when the user has not
    chosen a scope", never "what does this study analyse".  Exposure, outcome,
    cohort, covariates and the analytic window stay unset so the Planner --
    their owner -- proposes them for one human review.  Without this, a
    plan-only run demanded ``modules`` and ``time_window`` up front, which is
    exactly the slot-by-slot interrogation the plan exists to replace.

    The scope is deliberately hypothesis-free and as wide as the bound package
    allows: every module the package actually carries, and the standing outer
    window.  Values the caller already set are never overwritten.
    """

    patched = dict(study)
    applied: List[str] = []

    # Only a *wholly absent* scope is defaulted.  A scope the user or model
    # already committed to -- even a partial or non-executable one, such as a
    # prose window label carrying no hours -- must keep reaching its owner's
    # validation.  Completing it here would silently execute a different study
    # than the one the conversation agreed to.
    raw_modules = patched.get("modules")
    if raw_modules is None or (
        isinstance(raw_modules, (list, tuple)) and not raw_modules
    ):
        try:
            described = dataio.describe_export_source(export_path)
        except Exception:  # noqa: BLE001 - scope default must never mask the
            described = {}  # owner's own manifest/inventory error below
        available = [
            str(module).strip().lower()
            for module in (described.get("modules") or [])
            if str(module).strip()
        ]
        if available:
            patched["modules"] = sorted(dict.fromkeys(available))
            applied.append("modules")

    raw_window = patched.get("time_window")
    if raw_window is None or (isinstance(raw_window, Mapping) and not raw_window):
        patched["time_window"] = {
            "hours": _NEUTRAL_MATERIALIZATION_HOURS,
            "anchor": _NEUTRAL_MATERIALIZATION_ANCHOR,
        }
        applied.append("time_window")

    if applied:
        patched["materialization_scope_source"] = {
            "owner": "easyicu.webserver.agent_pipeline_runs",
            "kind": "easyicu_neutral_default",
            "applied_fields": applied,
        }
    return patched


def _normalized_metadata_planning_operationalized_columns(
    values: Sequence[str],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
            raise ResearchPipelineRunError(
                "research_pipeline_planning_operationalization_invalid",
                "The metadata-only planning schema contains an invalid "
                "operationalized column.",
            )
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _metadata_planning_operationalized_columns(
    *,
    primary_exposure_source: Optional[str],
    primary_exposure_aggregation: Optional[str],
    covariates: Sequence[str],
    covariate_selection: str,
    covariate_operationalizations: Mapping[str, Any],
    sensitivity_specs: Sequence[Any],
) -> tuple[str, ...]:
    """Project host-owned analysis columns into the zero-row plan schema.

    Planner canary runs deliberately read no patient rows, but their schema
    still has to expose every operational column that the host has already
    bound. Otherwise an exact adjustment set becomes impossible to satisfy:
    using its materialized column is rejected as unavailable, while omitting
    it is rejected as changing the user's adjustment decision.
    """

    values: list[str] = []
    if primary_exposure_source and primary_exposure_aggregation:
        values.append(f"{primary_exposure_source}_{primary_exposure_aggregation}")
    if covariate_selection == "exact":
        mapping = {
            str(key or "").strip(): str(value or "").strip()
            for key, value in covariate_operationalizations.items()
        }
        values.extend(mapping.get(name, name) for name in covariates)
    for spec in sensitivity_specs:
        values.extend(getattr(spec, "source_materialization_variables", ()) or ())
        # The outcome owner derives event time during real materialization, so
        # it is intentionally absent from ``source_materialization_variables``.
        # A zero-row planning catalog still needs that derived column in its
        # schema so the host can compile the already user-reviewed landmark
        # runtime without reading patient rows.
        event_time_variable = getattr(spec, "event_time_variable", None)
        if event_time_variable:
            values.append(event_time_variable)
    return _normalized_metadata_planning_operationalized_columns(values)


def _metadata_only_planning_coordinates(
    *, question: str, database: str
) -> Dict[str, Any]:
    """Project only concepts the researcher explicitly named into planning.

    This is proposal context, not executable StudyContext authority.  The
    deterministic intent reader supplies the exact user-text provenance and
    the database capability catalog proves that the named concepts exist.  A
    binary endpoint is emitted only when the concept owner declares
    ``event_status`` semantics; names and dtypes are never used to guess it.
    """

    from easyicu.research_agent.acquisition.catalog import (
        build_database_capability_catalog,
    )
    from easyicu.research_agent.contracts.endpoint import EndpointSpec
    from easyicu.webserver.study_intent import deterministic_intent

    intent = deterministic_intent(question)
    raw_slots = intent.get("slots")
    slots = raw_slots if isinstance(raw_slots, Mapping) else {}
    catalog = build_database_capability_catalog(database)
    catalog_by_id = {item.concept_id: item for item in catalog.concepts}

    def named_concept(slot_name: str) -> Optional[str]:
        raw = slots.get(slot_name)
        raw = raw if isinstance(raw, Mapping) else {}
        if str(raw.get("provenance") or "") != "user_text":
            return None
        value = _clean_text(raw.get("value"), 160)
        return value if value in catalog_by_id else None

    target_outcome = named_concept("outcome")
    primary_exposure = named_concept("exposure")
    endpoint = None
    outcome_type = slots.get("outcome_type")
    outcome_type = outcome_type if isinstance(outcome_type, Mapping) else {}
    target_catalog = catalog_by_id.get(str(target_outcome or ""))
    if (
        target_outcome
        and str(outcome_type.get("value") or "") == "binary"
        and str(outcome_type.get("provenance") or "") == "user_text"
        and target_catalog is not None
        and target_catalog.column_role == "event_status"
    ):
        endpoint = EndpointSpec(
            name=target_outcome,
            kind="binary",
            absence_semantics="no_absent_rows",
            levels=[0, 1],
        )
    return {
        "target_outcome": target_outcome,
        "primary_exposure": primary_exposure,
        "endpoint": endpoint,
        "source": "explicit_user_text_plus_database_capability",
        "execution_authorized": False,
    }


def _data_foundation_profile(
    *,
    export_path: str,
    study: Mapping[str, Any],
    target: Optional[str],
    primary_exposure: Optional[str] = None,
    require_target: bool = True,
    require_primary_exposure: bool = True,
    covariates: tuple[str, ...] = (),
    sensitivity_specs: tuple[Any, ...] = (),
) -> Dict[str, Any]:
    """Compile StudyContext modules into one typed materialization request."""

    from easyicu.research_agent.acquisition.catalog import build_available_catalog

    modules = _configured_modules(study)
    if not modules:
        raise ResearchPipelineRunError(
            "research_pipeline_modules_required",
            "A full Research Agent run requires configured feature modules.",
        )
    allowed = set(modules)
    catalog = build_available_catalog(Path(export_path).expanduser())
    concepts = [
        concept
        for concept in catalog.concepts
        if Path(concept.file_name).stem.lower() in allowed
    ]
    by_id = {concept.concept_id: concept for concept in concepts}
    demographic_values = [
        concept.concept_id
        for concept in concepts
        if Path(concept.file_name).stem.lower() == "demographics"
        and (not concept.typed_metadata or concept.column_role == "value")
    ]
    preferred_base = [
        concept for concept in ("age", "sex") if concept in demographic_values
    ]
    static_concepts = preferred_base or demographic_values[:1]
    if not static_concepts:
        raise ResearchPipelineRunError(
            "research_pipeline_stay_denominator_unavailable",
            "The configured modules do not provide a stay-level denominator concept.",
        )

    outcome_concepts: List[str] = []
    required_feature_concepts: List[str] = []
    require_outcome = False
    raw_cohort = study.get("cohort")
    cohort = raw_cohort if isinstance(raw_cohort, Mapping) else {}
    # Keep the owner-issued readmission indicator in a Planner-selectable
    # universe when available. It is a dependence-safety coordinate, not an
    # inferred exclusion: the plan may propose a first-stay analysis, while
    # human review still owns whether that restriction is adopted.
    readmission_meta = by_id.get("icu_readmission")
    if (
        _configured_covariate_selection(study) == "planner_selectable"
        and readmission_meta is not None
    ):
        readmission_module = Path(readmission_meta.file_name).stem.lower()
        if readmission_module in {"demographics", "outcome"} and (
            not readmission_meta.typed_metadata
            or readmission_meta.column_role == "value"
        ):
            static_concepts.append("icu_readmission")
        else:
            required_feature_concepts.append("icu_readmission")
    if cohort.get("exclude_readmissions") is True:
        if readmission_meta is None:
            raise ResearchPipelineRunError(
                "research_pipeline_readmission_indicator_unavailable",
                (
                    "The user-authorized first-stay restriction cannot run "
                    "because the selected modules expose no owner-issued "
                    "ICU-readmission indicator."
                ),
                details={
                    "field": "cohort.exclude_readmissions",
                    "required_concept": "icu_readmission",
                },
            )
        readmission_module = Path(readmission_meta.file_name).stem.lower()
        if readmission_module in {"demographics", "outcome"} and (
            not readmission_meta.typed_metadata
            or readmission_meta.column_role == "value"
        ):
            static_concepts.append("icu_readmission")
        else:
            required_feature_concepts.append("icu_readmission")
    if target:
        target_meta = by_id.get(target)
        if target_meta is None:
            if require_target:
                raise ResearchPipelineRunError(
                    "research_pipeline_target_outside_configured_modules",
                    (
                        "The configured outcome is not available in the "
                        "selected feature modules."
                    ),
                    details={
                        "field": "execution_concepts.outcome",
                        "concept_id": target,
                    },
                )
        else:
            target_module = Path(target_meta.file_name).stem.lower()
            if target_meta.column_role == "event_status":
                outcome_concepts.append(target)
                require_outcome = True
            elif target_module in {"demographics", "outcome"}:
                static_concepts.append(target)
            else:
                required_feature_concepts.append(target)

    sensitivity_variables = tuple(
        dict.fromkeys(
            variable
            for spec in sensitivity_specs
            for variable in spec.source_materialization_variables
        )
    )
    scientific_inputs = tuple(
        dict.fromkeys(
            value
            for value in (primary_exposure, *covariates, *sensitivity_variables)
            if value and value != target
        )
    )
    primary_exposure_source_concept: Optional[str] = None
    for concept_id in scientific_inputs:
        source_concept = _source_concept_for_operational_column(
            concept_id,
            by_id=by_id,
        )
        if source_concept is None:
            if concept_id == primary_exposure and not require_primary_exposure:
                continue
            role = (
                "primary_exposure"
                if concept_id == primary_exposure
                else (
                    "covariate" if concept_id in covariates else "sensitivity_variable"
                )
            )
            raise ResearchPipelineRunError(
                f"research_pipeline_{role}_outside_configured_modules",
                f"The configured {role.replace('_', ' ')} is not available in the selected feature modules.",
                details={
                    "field": f"execution_concepts.{role}",
                    "concept_id": concept_id,
                },
            )
        if concept_id == primary_exposure:
            primary_exposure_source_concept = source_concept
        concept_meta = by_id[source_concept]
        concept_module = Path(concept_meta.file_name).stem.lower()
        if (
            concept_id == source_concept
            and concept_module
            in {
                "demographics",
                "outcome",
            }
            and (not concept_meta.typed_metadata or concept_meta.column_role == "value")
        ):
            static_concepts.append(concept_id)
        else:
            required_feature_concepts.append(source_concept)

    return {
        "allowed_modules": modules,
        "static_concepts": tuple(dict.fromkeys(static_concepts)),
        "outcome_concepts": tuple(outcome_concepts),
        "required_feature_concepts": tuple(required_feature_concepts),
        "require_outcome": require_outcome,
        "primary_exposure_source_concept": primary_exposure_source_concept,
    }
