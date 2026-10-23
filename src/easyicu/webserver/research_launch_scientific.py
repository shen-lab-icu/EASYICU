"""Scientific configuration and materialization facts for pipeline launch."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from easyicu.research_agent.acquisition.first_icu_stay import FirstIcuStayBinding
from easyicu.research_agent.acquisition.patient_grouping import PatientGroupingBinding
from easyicu.research_agent.contracts.trajectory_design import FixedWindowTrajectoryDesign
from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.orchestration.exposure_grouping_phase import PlanningRunGroupings
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.target_trial_compile import (
    target_trial_acquisition,
)
from easyicu.research_agent.planning.target_trial_spec import TargetTrialSpec
from easyicu.research_agent.research_context.export_selection import SelectionBasis
from easyicu.webserver import dataio, primary_cohort, source_identity_authority
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.research_pipeline_run_errors import (
    ResearchPipelineRunError,
    WINDOWED_SUMMARY_STOP_MESSAGES,
)
from easyicu.webserver.study_intent import explicit_outcome_concepts
from easyicu.webserver.study_scientific_configuration import (
    ScientificConfiguration,
    ScientificConfigurationError,
)

_MATERIALIZED_FEATURE_SUFFIXES = tuple(
    sorted(
        (
            "_first_time",
            "_last_time",
            "_onset_time",
            "_measured",
            "_first",
            "_last",
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
    return ScientificConfiguration.inspect(study).executable_target_outcome()


def _primary_exposure(study: Mapping[str, Any]) -> Optional[str]:
    return ScientificConfiguration.inspect(study).executable_primary_exposure()


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
    primary_exposure: str = "",
    universe_path: Optional[Path] = None,
    exposure_groupings: PlanningRunGroupings | None = None,
) -> tuple[Any, ...]:
    """Add only the deterministic runtime's automatic nonlinear safeguard.

    A user-selected landmark must close exposure opportunity in the primary
    estimator.  When the researcher has not requested a competing functional-
    form sensitivity, the signed landmark runtime supplies its standard RCS
    primary plus linear sensitivity as a plan-owned automatic remediation.  It
    is not written back to StudyContext or projected as a user request.  The
    exposure's kind comes from the projection's own rule, read on the universe
    the projection signs (its schema and published column metadata), so an
    exposure the projection binds as binary never receives the continuous
    safeguard. A bound grouping supplies its declared kind before the source
    is classified. Without a universe only the names and grouping are read.
    The universe is read only for a landmark design that could receive the
    safeguard, and an unreadable universe keeps the projection's reason code.
    """

    if not primary_exposure_source:
        return sensitivity_specs
    strategies = {
        str(getattr(item, "strategy", "") or "") for item in sensitivity_specs
    }
    axes = {str(getattr(item, "axis", "") or "") for item in sensitivity_specs}
    if "landmark" not in strategies or "functional_form" in axes:
        return sensitivity_specs
    from easyicu.webserver.scientific_runtime_projection import (
        WebScientificRuntimeProjectionError,
        primary_exposure_kind,
    )

    try:
        exposure_kind, _levels = primary_exposure_kind(
            universe_path=Path(universe_path) if universe_path is not None else None,
            primary_exposure=primary_exposure,
            primary_exposure_source=primary_exposure_source,
            exposure_groupings=exposure_groupings,
        )
    except WebScientificRuntimeProjectionError as exc:
        raise ResearchPipelineRunError(exc.code, str(exc), details=exc.details) from exc
    if exposure_kind != VariableKind.CONTINUOUS:
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


def verified_patient_grouping(
    study: Mapping[str, Any],
) -> Optional[PatientGroupingBinding]:
    """The bound source's verified patient grouping, whatever the study's design.

    ``None`` when the study binds no source or the source has no grouping; an
    authority that fails its checks refuses with its typed reason.
    """

    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    export_path = _clean_text(source.get("path"), 2_000)
    database = _clean_text(source.get("database"), 80)
    if not export_path or not database:
        return None
    try:
        return source_identity_authority.resolve_study_patient_grouping(
            export_path=export_path,
            database=database,
        )
    except source_identity_authority.PatientGroupingAuthorityError as exc:
        raise ResearchPipelineRunError(
            exc.code,
            str(exc),
            details=exc.details,
        ) from exc


def declared_longitudinal_design(study: Mapping[str, Any]) -> Optional[str]:
    """The design a study declares that reads each stay's long trajectory, if any.

    A trajectory design reads the trajectory itself, and a landmark design
    cuts its at-risk set from it.  A grouped materialization emits no
    trajectory (``acquisition.foundation``), except beside a time-varying
    specification, whose owner applies the grouping to both row spaces.
    """

    if any(
        spec.strategy == "time_varying" for spec in _configured_sensitivity_specs(study)
    ):
        return None
    if _validate_trajectory_design(study) is not None:
        return "trajectory_design"
    if any(spec.strategy == "landmark" for spec in _configured_sensitivity_specs(study)):
        return "landmark"
    return None


def _patient_grouping_for_analysis_design(
    study: Mapping[str, Any],
) -> Optional[PatientGroupingBinding]:
    raw = study.get("analysis_design")
    design = raw if isinstance(raw, Mapping) else {}
    if not study_context_owner.analysis_design_reads_patient_grouping(design):
        return None
    cluster_unit = _clean_text(design.get("cluster_unit"), 80)
    if cluster_unit != "patient":
        raise ResearchPipelineRunError(
            "research_pipeline_cluster_unit_unsupported",
            "The current Web runner supports clustered inference only for a verified patient grouping.",
            details={
                "cluster_unit": cluster_unit or None,
                "supported_cluster_units": ["patient"],
            },
        )
    return verified_patient_grouping(study)


def _first_icu_stay_for_cohort(
    study: Mapping[str, Any],
) -> Optional[FirstIcuStayBinding]:
    """Resolve the verified coordinate a first-stay cohort requires, or refuse.

    An ICU-readmission indicator is not this authority: only a coordinate the
    host derived from the bound official stay table says which stay came
    first for each patient.
    """

    if not primary_cohort.first_icu_stay_only(study.get("cohort")):
        return None
    source = study.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    export_path = _clean_text(source.get("path"), 2_000)
    database = _clean_text(source.get("database"), 80)
    binding: Optional[FirstIcuStayBinding] = None
    reason_code = "first_icu_stay_authority_unavailable"
    cause: Dict[str, Any] = {}
    if export_path and database:
        try:
            binding = source_identity_authority.resolve_study_first_icu_stay(
                export_path=export_path,
                database=database,
            )
        except source_identity_authority.PatientGroupingAuthorityError as exc:
            reason_code = exc.code
            cause = {
                key: value
                for key, value in exc.details.items()
                if key in {"database", "cause_code"}
            }
    if binding is not None:
        return binding
    raise ResearchPipelineRunError(
        "research_pipeline_first_stay_restriction_unverified",
        (
            "The selected source cannot prove which ICU stay came first for each "
            "patient, so the first-stay restriction cannot be applied."
        ),
        details={
            "field": "cohort.exclude_readmissions",
            "first_stay_restriction_status": "unverified_in_selected_export",
            "first_icu_stay_reason_code": reason_code,
            **cause,
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


def resolve_study_analysis_design(study: Mapping[str, Any]) -> Dict[str, str]:
    """Resolve legacy descriptive scope identically for planning and launch.

    A confirmation may complete an absent legacy design, but cannot decide
    which of two contradictory persisted scientific commitments is newer.
    """
    raw = study.get("analysis_design")
    if raw is not None and not isinstance(raw, Mapping):
        raise ResearchPipelineRunError(
            "research_pipeline_analysis_design_invalid",
            "The typed analysis design is invalid.",
            details={"field": "analysis_design"},
        )
    design = dict(raw or {})
    confirmations = study.get("confirmations")
    if (
        isinstance(confirmations, Mapping)
        and confirmations.get("plan_timing_descriptive_only") is True
    ):
        descriptive = {
            "analysis_family": "descriptive_epidemiology",
            "analysis_unit": "icu_stay",
            "variance_estimator": "none_counts_only",
        }
        if any(design.get(key, value) != value for key, value in descriptive.items()) or design.get("cluster_unit"):
            raise ResearchPipelineRunError(
                "research_pipeline_descriptive_confirmation_conflict",
                "The descriptive-only confirmation conflicts with the explicit "
                "analysis design. Review a consistent study revision before planning or execution.",
                details={"field": "analysis_design", "remediation_route": "agent_plan_revision"},
            )
        design.update(descriptive)
    from easyicu.research_agent.contracts.analysis_design import AnalysisDesignConflict, validate_analysis_family_ceiling

    try:
        validate_analysis_family_ceiling(
            analysis_family=design.get("analysis_family"),
            variance_estimator=design.get("variance_estimator", ""),
        )
    except AnalysisDesignConflict as exc:
        raise ResearchPipelineRunError(
            exc.code, str(exc),
            details={"field": "analysis_design", "remediation_route": "agent_plan_revision", "requires_user_authorization": False},
        ) from exc
    return design


def _validate_trajectory_design(
    study: Mapping[str, Any],
) -> Optional[FixedWindowTrajectoryDesign]:
    """Surface a contradictory trajectory declaration before spending anything.

    The trajectory projection owns this policy; calling it here only moves the
    same refusal earlier, to the point where the user can still fix the study
    without having paid for a materialization.  Returns the typed design, or
    ``None`` for a study that declares no trajectory.
    """

    from easyicu.webserver.scientific_runtime_projection import (
        WebScientificRuntimeProjectionError,
    )
    from easyicu.webserver.trajectory_runtime_projection import (
        validate_trajectory_design_declaration,
    )

    try:
        return validate_trajectory_design_declaration(study)
    except WebScientificRuntimeProjectionError as exc:
        raise ResearchPipelineRunError(
            exc.code, str(exc), details=exc.details
        ) from exc


def _validate_analysis_design(study: Mapping[str, Any]) -> Dict[str, str]:
    """Fail closed without substituting a different inferential design."""

    _validate_trajectory_design(study)
    raw = resolve_study_analysis_design(study)
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
    analysis_unit = _clean_text(raw.get("analysis_unit"), 80)
    variance_estimator = _clean_text(raw.get("variance_estimator"), 80)
    cluster_unit = _clean_text(raw.get("cluster_unit"), 80)
    if not analysis_unit or not variance_estimator:
        raise ResearchPipelineRunError(
            "research_pipeline_analysis_design_incomplete",
            "The typed analysis design is missing its analysis unit or variance estimator.",
            details={"field": "analysis_design"},
        )
    _first_icu_stay_for_cohort(study)
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
    if study_context_owner.analysis_design_reads_patient_grouping(raw):
        grouping = _patient_grouping_for_analysis_design(study)
        if grouping is None:
            raise ResearchPipelineRunError(
                "research_pipeline_cluster_variance_unsupported",
                (
                    "This source and executor do not expose a verified grouping "
                    "coordinate for the requested patient-clustered inference."
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
    # A bootstrap of stays: the analysis design's ceiling keeps it to causal
    # inference, whose suite computes it.
    if variance_estimator in {"none_counts_only", "bootstrap"}:
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
        if spec.strategy in {"landmark", "time_varying"}:
            return True
    return validated_design.get("variance_estimator") != "none_counts_only"


def _validate_primary_concept_selection(
    study: Mapping[str, Any],
    primary_exposure: Optional[str],
) -> None:
    """Enforce the concept owner's user-intent selection policy at launch."""

    from easyicu.concept.selection_policy import (
        concept_id_for_module,
        concept_selection_authority_key,
        concept_selection_confirmation_key,
        evaluate_concept_selection,
    )

    # Only the persisted scientific question can authorize an explicit-only
    # variant. Exposure labels and analysis prose are model-produced fields;
    # accepting them here would let a plan authorize its own semantic drift.
    intent = str(study.get("question") or "")
    confirmations = study.get("confirmations")
    governed: list[tuple[str, str]] = []
    if primary_exposure:
        governed.append((str(primary_exposure), "execution_concepts.primary_exposure"))
    for module in study.get("modules") or ():
        concept_id = concept_id_for_module(module)
        if concept_id:
            governed.append((concept_id, f"modules.{module}"))

    seen: set[str] = set()
    for concept_id, field in governed:
        if concept_id in seen:
            continue
        seen.add(concept_id)
        confirmation_key = concept_selection_confirmation_key(concept_id)
        authority_key = concept_selection_authority_key(concept_id)
        owner_confirmed = bool(
            isinstance(confirmations, Mapping)
            and confirmations.get(confirmation_key)
            and confirmations.get(authority_key)
        )
        decision = evaluate_concept_selection(
            concept_id,
            user_intent=intent,
            owner_confirmed=owner_confirmed,
        )
        if decision.allowed:
            continue
        raise ResearchPipelineRunError(
            decision.reason_code,
            (
                "The configured study selects an explicit-only concept "
                "variant that the user did not request."
            ),
            details={**decision.to_dict(), "field": field},
        )


def _source_concept_for_operational_column(
    column: str,
    *,
    by_id: Mapping[str, Any],
) -> Optional[str]:
    """Resolve a wide materialized column back to its exported source concept."""

    # Some user-facing clinical concepts are published by a versioned
    # composite output whose loader/source name is intentionally different.
    # Materialization must follow that declarative owner mapping just as the
    # post-materialization resolver does; otherwise a reviewed ``sep3`` plan
    # silently omits its exposure because the package stores
    # ``sep3_sofa1``.  Ambiguous composite families remain fail-closed.
    from easyicu.concept_output_sources import resolve_composite_concept_output

    resolved = resolve_composite_concept_output(column, by_id.keys())
    if resolved is not None:
        return resolved
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


def _time_window_absent(raw: Any) -> bool:
    return raw is None or (isinstance(raw, Mapping) and not raw)


def launch_materialization_window(study: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the outer window a launch of this StudyContext materializes.

    A declared window is returned as declared, for its owner to validate at
    launch; a wholly absent one is the standing neutral window that
    ``_neutral_materialization_scope`` applies to exactly the same studies.
    ``None`` means a declared value that is not a window object, which the
    launch refuses.  A caller that must know the window a candidate plan was
    planned under reads it here instead of restating the default.
    """

    raw = study.get("time_window")
    if _time_window_absent(raw):
        return {
            "hours": _NEUTRAL_MATERIALIZATION_HOURS,
            "anchor": _NEUTRAL_MATERIALIZATION_ANCHOR,
        }
    return dict(raw) if isinstance(raw, Mapping) else None


#: A package that declares itself the study's complete prepared input
#: (``agent_pipeline_runs._metadata_only_planning_catalog`` reads the same mode).
_STUDY_LOCAL_PREPARED_COHORT = "study_local_prepared_cohort"


@dataclasses.dataclass(frozen=True)
class ExportCohortRefusal:
    """Why a new plan or run may not analyze the bound export's rows.

    ``code`` is the error the launch raises.  ``reason_codes`` is its cause:
    the mismatch codes, or the reason a cohort is not an executable
    extraction contract.
    """

    code: Literal[
        "research_pipeline_export_cohort_mismatch",
        "research_pipeline_export_cohort_invalid",
    ]
    reason_codes: tuple[str, ...]
    cause: Optional[BaseException] = dataclasses.field(
        default=None, compare=False, repr=False
    )

    def error(self) -> ResearchPipelineRunError:
        if self.code == "research_pipeline_export_cohort_invalid":
            return ResearchPipelineRunError(
                self.code,
                "The study's cohort or the bound export's recorded cohort is not an "
                "executable extraction contract.",
                details={"reason_code": self.reason_codes[0]},
            )
        return ResearchPipelineRunError(
            self.code,
            "The bound export does not hold the rows of the cohort this study "
            "states (details.mismatch_codes): it was extracted for another "
            "cohort or under an earlier rule, or it records no population for a "
            "cohort that only extraction applies. Retrying will fail "
            "identically; run easyicu_start_extraction to extract the study's "
            "cohort.",
            details={"mismatch_codes": list(self.reason_codes)},
        )


def bound_export_cohort_refusal(
    study: Mapping[str, Any], export_path: Optional[str]
) -> Optional[ExportCohortRefusal]:
    """The refusal a new plan or run on the bound export meets, or ``None``.

    Data Extraction records in each manifest the cohort contract the export
    was extracted for and the rule that executed it.  A study changed after
    extraction, or bound to an export prepared for another cohort, would
    otherwise analyze those rows under the cohort it states: a family
    template re-applies typed age and stay bounds and the host the first
    ICU stay, a progressive plan only the predicates its Planner writes,
    and nothing a concept-derived or diagnosis population, so a Sepsis-3
    study over an all-ICU export analyzes every stay.  A folder without a
    prepared manifest
    is left to the package validation that requires one.  A package that
    records no extraction contract (an export from before contracts were
    recorded) cannot be compared; it serves a study that states only typed
    age or stay bounds, which a family template re-applies (a progressive
    plan only through its Planner's predicates), but not a concept-derived
    or diagnosis population, which nothing after extraction applies.  A
    study-local prepared cohort
    declares itself the study's complete input.

    The launch raises this refusal (``_require_export_holds_study_cohort``).
    The Copilot workflow reads it before offering a plan, so it never offers
    a start that the launch refuses.
    """

    from easyicu.webserver.pi_copilot.extraction_handoff import (
        bound_export_mismatches,
        compile_study_cohort,
    )

    if not export_path:
        return None
    manifest = dataio.read_prepared_export_manifest(str(export_path))
    if manifest is None:
        return None
    try:
        if isinstance(manifest.get("cohort_contract"), Mapping):
            mismatches = bound_export_mismatches(study, manifest)
        elif str(manifest.get("entry_mode") or "") == _STUDY_LOCAL_PREPARED_COHORT:
            mismatches = ()
        else:
            stated = compile_study_cohort(study)
            unreapplied = bool(
                stated["preset"] in primary_cohort.CONCEPT_DERIVED_PRESETS
                or stated["icd_enabled"]
            )
            mismatches = ("registered_export_cohort_unrecorded",) if unreapplied else ()
    except dataio.ExportCohortError as exc:
        return ExportCohortRefusal(
            code="research_pipeline_export_cohort_invalid",
            reason_codes=(str(exc.error),),
            cause=exc,
        )
    if not mismatches:
        return None
    return ExportCohortRefusal(
        code="research_pipeline_export_cohort_mismatch",
        reason_codes=tuple(mismatches),
    )


def _require_export_holds_study_cohort(
    study: Mapping[str, Any], export_path: str
) -> None:
    """Refuse an export whose rows are not the population the study states."""

    refusal = bound_export_cohort_refusal(study, export_path)
    if refusal is not None:
        raise refusal.error() from refusal.cause


def bound_export_selection_basis(
    study: Mapping[str, Any], export_path: Optional[str]
) -> SelectionBasis:
    """How the host knows what selected the bound export's rows.

    ``export_contract`` only when the export's manifest records the cohort
    contract it was extracted for, and that contract selects this study's
    rows under the rule Data Extraction executes now
    (``bound_export_mismatches``): the study's executed criteria are then
    what selected the export's rows.  ``package_declaration`` when a
    study-local prepared cohort records no contract but declares itself the
    study's complete input, which the launch accepts
    (``_require_export_holds_study_cohort``): its rows are the study's
    population by criteria the host does not know.  Anything else (an export
    from before contracts were recorded, a folder without a manifest, a
    contract that does not hold the study's rows) is ``unrecorded``: what
    selected its rows is not known, so nothing may be declared as already
    applied to them, nor that nothing was.
    """

    from easyicu.webserver.pi_copilot.extraction_handoff import (
        bound_export_mismatches,
    )

    if not export_path:
        return "unrecorded"
    manifest = dataio.read_prepared_export_manifest(str(export_path))
    if manifest is None:
        return "unrecorded"
    if isinstance(manifest.get("cohort_contract"), Mapping):
        try:
            mismatches = bound_export_mismatches(study, manifest)
        except dataio.ExportCohortError:
            return "unrecorded"
        return "unrecorded" if mismatches else "export_contract"
    if str(manifest.get("entry_mode") or "") == _STUDY_LOCAL_PREPARED_COHORT:
        return "package_declaration"
    return "unrecorded"


def bound_export_selection_report(
    export_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    """The bound export's own count of the stays its selection kept, verbatim.

    Data Extraction writes ``cohort_report`` beside the cohort contract
    ``bound_export_selection_basis`` reads: the source's stays, each
    criterion with the stays it excluded, and a cap with the rule that chose
    the stays it kept.  The launch reads it from the same manifest, once, and
    records it for a recorded selection in
    ``data_constraints.source_selection.export_report``;
    ``research_context.export_selection`` alone reads it.  ``None`` without a
    manifest or a report.
    """

    if not export_path:
        return None
    manifest = dataio.read_prepared_export_manifest(str(export_path))
    if manifest is None:
        return None
    report = manifest.get("cohort_report")
    return dict(report) if isinstance(report, Mapping) else None


def bound_export_event_time_semantics(export_path: Optional[str]) -> Dict[str, str]:
    """What each event time the bound export issues is, as its producer labels it.

    The native export labels every event-time companion it issues in the
    outcome file's time-axis audit, ``death_time`` by the death time its source
    records (``easyicu.utils.death_time_semantics``).  The launch reads the
    label from the same manifest ``bound_export_selection_basis`` reads, once,
    and planning reads it from ``data_constraints.event_time_semantics``; no
    later stage opens the export again.  An export that labels none (written
    before the label, a folder without a manifest, a prepared package) gives
    an empty mapping, and so do two files that label one companion
    differently: the launch still records it, so that planning can tell a
    source whose death time nobody labelled from a context older than the
    record.
    """

    if not export_path:
        return {}
    manifest = dataio.read_prepared_export_manifest(str(export_path))
    if manifest is None:
        return {}
    labelled: Dict[str, str] = {}
    for record in manifest.get("files") or []:
        audit = record.get("time_axis_audit") if isinstance(record, Mapping) else None
        if not isinstance(audit, Mapping):
            continue
        companion = audit.get("event_time_companion")
        semantics = audit.get("event_time_semantics")
        if not (isinstance(companion, str) and companion.strip()):
            continue
        if not (isinstance(semantics, str) and semantics.strip()):
            continue
        if (
            labelled.setdefault(companion.strip(), semantics.strip())
            != semantics.strip()
        ):
            return {}
    return labelled


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

    if _time_window_absent(patched.get("time_window")):
        patched["time_window"] = launch_materialization_window(patched)
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
    exposure_onset_column: Optional[str] = None,
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
    if exposure_onset_column:
        # A declared survival design binds the exposure's producer-owned onset;
        # its owner names the column.
        values.append(exposure_onset_column)
    if covariate_selection == "exact":
        mapping = {
            str(key or "").strip(): str(value or "").strip()
            for key, value in covariate_operationalizations.items()
        }
        values.extend(mapping.get(name, name) for name in covariates)
    for spec in sensitivity_specs:
        values.extend(getattr(spec, "source_materialization_variables", ()) or ())
        # The outcome owner derives event time and may also derive follow-up
        # duration during real materialization.  They are intentionally absent
        # from source concept selection, but the zero-row catalog still needs
        # both owner-issued output columns so the host can bind the exact
        # landmark runtime without reading patient rows.
        for derived in (
            getattr(spec, "event_time_variable", None),
            getattr(spec, "observation_duration_variable", None),
        ):
            if derived:
                values.append(derived)
    return _normalized_metadata_planning_operationalized_columns(values)


#: Combined KDIGO stage bindings a question can name ("KDIGO AKI stage").  They
#: share the 0-3 stage space of the observability-preserving alternative, so
#: the concept owner's canonical reading changes no quantity.  Component stages
#: and binary AKI phenotypes are deliberately not here: substituting the
#: combined stage for them would change the question.
_COMBINED_KDIGO_STAGE_BINDINGS = frozenset(
    {"aki_stage", "aki_stage_reference", "aki_stage_source_native", "kdigo_stage"}
)


def _observability_preserving_exposure(
    concept: Optional[str], export_path: Optional[str]
) -> Optional[str]:
    """Read a named combined KDIGO stage the way its concept owner defines it.

    The concept owner (``concept.selection_policy``) declares that a stage
    whose zero absorbs never-assessed stays is the wrong exposure, and names
    the observability-preserving alternative.  The Web runtime refuses the
    collapsed binding as an exposure, so proposing it would guarantee a plan
    that cannot run.  When the bound source carries the evidence receipts the
    alternative is derived from, the named stage is read as that alternative;
    otherwise the reading is left unchanged and the runtime's own refusal
    (with its re-extraction remedy) still applies.
    """

    if concept not in _COMBINED_KDIGO_STAGE_BINDINGS or not export_path:
        return concept
    from easyicu.concept.selection_policy import concept_selection_policy
    from easyicu.webserver.scientific_runtime_projection import (
        export_kdigo_strict_derivation_available,
    )

    policy = concept_selection_policy(concept)
    alternative = policy.canonical_alternative if policy is not None else None
    if not alternative or not export_kdigo_strict_derivation_available(export_path):
        return concept
    return alternative


def _bound_source_column(export_path: Optional[str]):
    """Read a named concept as the bound source's owner-declared output column.

    Some public concepts are published by a versioned composite output under a
    different column name (``concept_output_sources``). Package materialization
    already follows that owner mapping. A planning coordinate must name the same
    column, or the zero-row planning schema never carries the requested concept
    and no plan can bind it. Absent or ambiguous outputs leave the name as is.
    """

    from easyicu.concept_output_sources import resolve_composite_concept_output
    from easyicu.research_agent.acquisition.catalog import build_available_catalog

    available: frozenset[str] = frozenset()
    if export_path:
        try:
            source = build_available_catalog(Path(export_path).expanduser())
        except (FileNotFoundError, OSError, ValueError):
            source = None
        if source is not None:
            available = frozenset(item.concept_id for item in source.concepts)

    def column(concept: Optional[str]) -> Optional[str]:
        if not concept:
            return concept
        return resolve_composite_concept_output(concept, available) or concept

    return column


def _stated_exposure_concept(
    text: Optional[str], catalog_by_id: Mapping[str, Any]
) -> Optional[str]:
    """The concept the study's exposure field names, read like the question."""

    from easyicu.webserver.study_intent import KEYWORD_PROVENANCE, deterministic_intent

    value = _clean_text(text, 160)
    if not value:
        return None
    if value in catalog_by_id:
        return value
    slots = deterministic_intent(value).get("slots")
    slot = slots.get("exposure") if isinstance(slots, Mapping) else None
    slot = slot if isinstance(slot, Mapping) else {}
    if str(slot.get("provenance") or "") != KEYWORD_PROVENANCE:
        return None
    concept = _clean_text(slot.get("value"), 160)
    return concept if concept in catalog_by_id else None


def _stated_catalog_concept(
    text: Optional[str], catalog_by_id: Mapping[str, Any]
) -> tuple[str, ...]:
    """The one source concept the exposure field names by the dictionary's names, if one."""

    from easyicu.webserver.study_intent import catalog_named_concepts

    value = _clean_text(text, 160)
    named = [concept for concept in catalog_named_concepts(value) if concept in catalog_by_id] if value else []
    return (named[0],) if len(named) == 1 else ()


def _stated_reading(item: Mapping[str, Any]) -> dict[str, Any]:
    """A saved reading as the conversation stated it, for a recheck."""

    reading = {"variable": item["variable"], "words": item["words"]}
    if item.get("concept"):
        reading["concept"] = item["concept"]
    return reading


def _checked_stated_variable(
    item: Any,
    *,
    question: str,
    question_concepts: set[str],
    available: set[str],
    source_bound: bool,
    recheck: bool = False,
    database_concepts: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Ground one structured variable reading, or say why it is not one.

    A save refuses a looked-up identifier the bound source does not offer, so
    the conversation looks it up again; a launch rechecking a saved reading
    keeps it as an unavailable requirement instead of dropping it.  Before a
    source is bound, ``database_concepts`` (the concepts of the database the
    question names) refuse an identifier that database does not define.
    """

    from easyicu.concept_output_sources import resolve_composite_concept_output
    from easyicu.webserver.study_intent import catalog_named_concepts

    if not isinstance(item, Mapping) or not {"variable", "words"} <= set(item) or not (
        set(item) <= {"variable", "words", "concept"}
    ):
        raise ValueError("a stated variable has variable, words and optionally concept")
    variable, words, stated = item["variable"], item["words"], item.get("concept")
    if (not isinstance(variable, str) or not isinstance(words, str)
            or not variable.strip() or not words.strip()
            or words not in question or variable not in words):
        raise ValueError("a stated variable must quote the question's own variable and words")
    if stated is not None and (not isinstance(stated, str) or not stated.strip()):
        raise ValueError("a stated variable's concept is an exact source identifier")
    named = catalog_named_concepts(variable)
    if len(named) > 1 or (named and named[0] not in question_concepts):
        raise ValueError("a stated variable must name one non-negated question concept")

    def output(concept: str) -> str:
        return resolve_composite_concept_output(concept, available) or concept

    # The dictionary's reading of the words, else the identifier the
    # conversation looked up for them (easyicu_list_source_concepts): many
    # scores and treatments have no dictionary name in every language.
    concept = output(named[0]) if named else None
    if stated is not None:
        stated = output(stated.strip())
        if concept is not None and stated != concept:
            raise ValueError(
                f"the dictionary reads {variable!r} as {concept!r}, not {stated!r}"
            )
        if source_bound and not recheck and stated not in available:
            raise ValueError(
                f"{stated!r} is not a concept the bound source offers; look it up "
                "with easyicu_list_source_concepts"
            )
        if not source_bound and not recheck and database_concepts and stated not in database_concepts:
            raise ValueError(
                f"{stated!r} is not a concept of the database the question names; "
                "look it up with easyicu_list_source_concepts"
            )
        concept = stated
    elif concept is None and variable in available:
        concept = output(variable)  # the question wrote the source's identifier
    if not source_bound:
        status = "unbound"
    elif concept is None:
        status = "unresolved"
    else:
        status = "available" if concept in available else "unavailable"
    return {"variable": variable, "words": words, "concept": concept, "source_status": status}


def checked_stated_variables(
    statements: Any,
    *,
    question: str,
    source: Mapping[str, Any],
    rejected: Optional[list[dict[str, str]]] = None,
    recheck: bool = False,
    database: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Ground structured variable readings in words and source metadata only.

    Missing source concepts remain visible requirements, and a reading whose
    concept nothing resolves stays ``unresolved``, never ``unavailable``. No
    keyword reader decides the scientific role of the quoted clause.  With
    ``rejected`` a reading that fails is recorded there with its reason and the
    others are kept; without it the first failure raises.
    """

    from easyicu.research_agent.acquisition.catalog import (
        build_available_catalog,
        build_database_capability_catalog,
    )
    from easyicu.research_agent.planning.adjustment_authority import normalize_stated_variables
    from easyicu.webserver.study_intent import catalog_named_concepts

    if not isinstance(statements, list) or len(statements) > 16:
        raise ValueError("stated variables must be a list of at most 16 readings")
    path = str(source.get("path") or "").strip()
    available = (
        {item.concept_id for item in build_available_catalog(Path(path)).concepts}
        if path else set()
    )
    database_concepts: frozenset[str] = frozenset()
    if not path and database and statements:
        try:
            database_concepts = frozenset(
                item.concept_id for item in build_database_capability_catalog(database).concepts
            )
        except (KeyError, ValueError):
            database_concepts = frozenset()
    question_concepts = set(catalog_named_concepts(question))
    records = []
    for item in statements:
        try:
            record = _checked_stated_variable(
                item, question=question, question_concepts=question_concepts,
                available=available, source_bound=bool(path), recheck=recheck,
                database_concepts=database_concepts,
            )
            records.extend(normalize_stated_variables([record]))
        except ValueError as exc:
            if rejected is None:
                raise
            variable = item.get("variable") if isinstance(item, Mapping) else None
            rejected.append({"variable": str(variable or "")[:160], "reason": str(exc)})
    unique = {(record["variable"], record["words"]): record for record in records}
    return normalize_stated_variables(list(unique.values()))


# Compatibility entry point for adjustment callers and saved studies.
checked_stated_adjustments = checked_stated_variables
checked_stated_subgroups = checked_stated_variables


def checked_exposure_concept(
    exposure: Any, concept: Any, *, question: str,
    source: Mapping[str, Any], database: Optional[str] = None,
) -> str:
    """Check an exposure's quoted words and looked-up identity against its catalog.

    Unlike a retained adjustment requirement, an exposure cannot acquire a
    planning coordinate from an unavailable or unbound, unchecked identifier.
    This check is repeated at launch, including after a source changes.
    """
    from easyicu.research_agent.acquisition.catalog import (
        build_available_catalog, build_database_capability_catalog,
    )
    from easyicu.webserver.study_intent import catalog_named_concepts

    path = str(source.get("path") or "").strip()
    if not path and not database:
        raise ValueError("尚未指定数据库，无法核验分组或暴露的概念；请先指定数据库并查找概念。")
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("请保存查找返回的分组或暴露概念标识。")
    catalog = (
        build_available_catalog(Path(path)) if path
        else build_database_capability_catalog(database)
    )
    available = {item.concept_id for item in catalog.concepts}
    record = _checked_stated_variable(
        {"variable": exposure, "words": exposure, "concept": concept},
        question=question, question_concepts=set(catalog_named_concepts(question)),
        available=available if path else set(), source_bound=bool(path),
        database_concepts=frozenset(available),
    )
    if record["concept"] not in available:
        raise ValueError("该数据库或数据源没有这个分组或暴露概念；请用 easyicu_list_source_concepts 重新查找。")
    return str(record["concept"])


def _study_exposure_reading(
    question_exposure: Optional[str],
    stated_exposure: Optional[str],
    *,
    excluded: set[Any],
    named: frozenset[str] = frozenset(),
    stated_names: tuple[str, ...] = (),
    demographic: frozenset[str] = frozenset(),
) -> Optional[str]:
    """The exposure both readings allow, or ``None`` when they disagree.

    The question's reading is the first concept it names; the study's
    exposure field says which named concept is the exposure.  When the
    question names the field's concept too, the field decides: on pilot1
    (10-22) a question stratified by sex before naming its measurement
    (「按性别分层后比较…」) projected sex, and the plan modelled it.
    ``stated_names`` is the one concept the field names by the dictionary's
    own names (「血钾」), which decides only among the question's concepts.
    A demographic the question names before a clinical measurement is
    projected only when the field says so: an adjustment named first
    (「考虑年龄影响后，…」) projected age.  Otherwise none is, and the planner
    reads the question.
    """

    for concept in (stated_exposure, *stated_names):
        if concept and concept not in excluded and concept in named:
            return concept
    readings = {
        concept
        for concept in (question_exposure, stated_exposure)
        if concept and concept not in excluded
    }
    if len(readings) != 1:
        return None
    reading = next(iter(readings))
    if (
        reading != stated_exposure
        and reading in demographic
        and any(concept not in demographic for concept in named - excluded - {reading})
    ):
        return None
    return reading


def _descriptive_window_summary(
    *, question: str, concept: Any, column: str, operation: Any,
) -> tuple[Any, Dict[str, Any]]:
    """Bind a named measurement reduction; a host default is never a request.

    Concept metadata establishes the quantity space, the intent reader owns
    the operation, the shared time parser owns the span, and the materializer
    owns the output name.
    The physical window is checked against this request at launch.
    """

    from easyicu.research_agent.acquisition.catalog import catalog_endpoint
    from easyicu.research_agent.cohort.materializer import summary_column_name
    from easyicu.research_agent.research_context.temporal_semantics import TimeWindowSemanticParser

    def stop(reason: str) -> None:
        raise ResearchPipelineRunError(
            reason, WINDOWED_SUMMARY_STOP_MESSAGES[reason], details={"concept": concept.concept_id},
        )

    if operation is None:
        stop("windowed_summary_statistic_unresolved")
    try:
        output = summary_column_name(column, operation.aggregation)
    except ValueError:
        stop("windowed_summary_statistic_unsupported")
    windows = {
        (constraint.anchor_event, constraint.start_hours, constraint.end_hours)
        for constraint in TimeWindowSemanticParser().parse(question)
        if constraint.relation in {"first_window", "within_after"}
    }
    if not windows:
        stop("windowed_summary_window_required")
    if len(windows) != 1:
        stop("windowed_summary_window_unresolved")
    anchor, start, hours = next(iter(windows))
    if (anchor != "icu_admission" or start != 0.0 or hours is None
            or not math.isfinite(hours) or hours <= 0):
        stop("windowed_summary_window_unsupported")
    endpoint = catalog_endpoint(concept, column=output, aggregation=operation.aggregation)
    if endpoint is None or endpoint.kind != "continuous":
        stop("windowed_summary_concept_not_continuous")
    return endpoint, {"anchor": "icu_admission", "start_hours": 0.0, "end_hours": float(hours)}


def _metadata_only_planning_coordinates(
    *,
    question: str,
    database: str,
    export_path: Optional[str] = None,
    configured_outcome: Optional[str] = None,
    stated_exposure: Optional[str] = None,
    stated_exposure_concept: Optional[str] = None,
    declared_covariates: Sequence[str] = (),
    stated_adjustments: Sequence[Mapping[str, Any]] = (),
    stated_subgroups: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Project only concepts the researcher explicitly named into planning.

    This is proposal context, not executable StudyContext authority.  The
    deterministic intent reader's keyword reading names them, with the words
    it matched, and the database capability catalog proves that the named
    concepts exist.  An
    endpoint is emitted only from the concept owner's event or continuous
    value definition; names and dtypes are never used to guess it.
    A named exposure is read through its concept owner's selection policy
    against the bound source (``export_path``), and both named coordinates as
    the column that source publishes for them; both callers -- the planning
    launch and the candidate-upgrade check -- must pass the same source so the
    proposal is reproducible.

    ``configured_outcome`` is the StudyContext's typed outcome concept, which
    the researcher confirmed; it takes precedence over a question-named one,
    and its endpoint follows the same concept-owner rule.

    The exposure is a concept neither the study's declared covariates
    (``declared_covariates``) nor its outcome: the keyword reader names the
    first concept a question mentions, and "以年龄、性别为协变量，分析…乳酸与
    院内死亡的关联" mentions a covariate first (pilot1 round 2, 10-22: age was
    sealed as the exposure, and the plan's model could not both adjust for
    it and estimate it).  The study's exposure field (``stated_exposure``,
    the conversation's reading of which words name it) is read through the
    same reader; when it and the question name different concepts, neither
    is projected and the planner reads the question.

    A prediction study names predictors without a primary exposure. A
    descriptive study can name a categorical grouping as its exposure; a lone
    continuous measurement does not become an exposure. A checked, stated
    grouping can disambiguate the descriptive reader's measurement slot;
    other families retain the conflict rule. The existing
    host family reading takes precedence over the keyword family hint.
    Every concept the question names besides its outcome and
    the exposure projected here is returned as ``question_named_concepts``:
    the source columns its name can denote and the words that name it, which
    the plan must account for.
    """

    from easyicu.research_agent.acquisition.catalog import (
        build_database_capability_catalog,
        catalog_endpoint,
    )
    from easyicu.webserver.study_intent import (
        KEYWORD_PROVENANCE,
        catalog_named_concepts,
        deterministic_intent,
        explicit_exposure_aggregation,
        named_study_concepts,
        substance_forms,
    )
    from easyicu.webserver.study_family_reading import study_family_reading

    intent = deterministic_intent(question)
    raw_slots = intent.get("slots")
    slots = raw_slots if isinstance(raw_slots, Mapping) else {}
    catalog = build_database_capability_catalog(database)
    catalog_by_id = {item.concept_id: item for item in catalog.concepts}
    required_adjustments = checked_stated_adjustments(
        [_stated_reading(item) for item in stated_adjustments],
        question=question, source={"path": export_path}, recheck=True,
    ) if stated_adjustments else []
    required_adjustment_concepts = [
        item["concept"] for item in required_adjustments if item["source_status"] == "available"
    ]
    required_subgroups = checked_stated_subgroups(
        [_stated_reading(item) for item in stated_subgroups],
        question=question, source={"path": export_path}, recheck=True,
    ) if stated_subgroups else []
    required_subgroup_concepts = [
        item["concept"] for item in required_subgroups if item["source_status"] == "available"
    ]
    declared_covariates = (*declared_covariates, *required_adjustment_concepts)

    def named_concept(slot_name: str) -> Optional[str]:
        raw = slots.get(slot_name)
        raw = raw if isinstance(raw, Mapping) else {}
        if str(raw.get("provenance") or "") != KEYWORD_PROVENANCE:
            return None
        value = _clean_text(raw.get("value"), 160)
        return value if value in catalog_by_id else None

    target_outcome = named_concept("outcome")
    family = slots.get("analysis_family")
    family = family if isinstance(family, Mapping) else {}
    prediction = (
        str(family.get("value") or "") == "prediction"
        and str(family.get("provenance") or "") == KEYWORD_PROVENANCE
    )
    host_family = study_family_reading(question)
    descriptive = (
        host_family.family == "descriptive_epidemiology"
        if host_family is not None
        else family.get("value") == "description"
        and family.get("provenance") == KEYWORD_PROVENANCE
    )
    question_concepts = frozenset(
        concept for concepts, _ in named_study_concepts(question) for concept in concepts
    ) | frozenset(catalog_named_concepts(question))
    question_concepts -= {named_concept("population")}
    from easyicu.concept.loader import load_dictionary
    from easyicu.webserver.pi_copilot.turn_authority import database_named_in

    def is_grouping(concept: str) -> bool:
        item = catalog_by_id.get(concept)
        definition = load_dictionary().get(concept)
        return item is not None and (
            item.column_role == "event_status"
            or (definition is not None and definition.class_name == "fct_cncpt"
                and definition.target == "id_tbl")
        )

    checked_grouping = None
    if descriptive and stated_exposure_concept:
        try:
            checked = checked_exposure_concept(
                stated_exposure, stated_exposure_concept, question=question,
                source={"path": export_path}, database=database_named_in(question),
            )
        except (KeyError, OSError, ValueError):
            checked = None
        if (checked and is_grouping(checked) and checked not in {
            *declared_covariates, *required_subgroup_concepts, target_outcome, configured_outcome,
        }):
            checked_grouping = checked
    named_exposure = (
        None
        if prediction
        else checked_grouping or _study_exposure_reading(
            named_concept("exposure"),
            _stated_exposure_concept(stated_exposure, catalog_by_id),
            excluded={*declared_covariates, *required_subgroup_concepts, target_outcome, configured_outcome},
            named=question_concepts,
            stated_names=_stated_catalog_concept(stated_exposure, catalog_by_id),
            demographic=frozenset(
                concept for concept, item in catalog_by_id.items()
                if getattr(item, "category", None) == "demographics"
            ),
        )
    )
    if descriptive and named_exposure:
        # Describing one factor's frequencies does not compare an outcome
        # across its levels. Keep that concept in the measurement reading.
        if not (target_outcome or configured_outcome or question_concepts - {
            named_exposure, *declared_covariates, *required_subgroup_concepts,
        }):
            named_exposure = None
        elif not is_grouping(named_exposure):
            named_exposure = None
    named_measurement = named_concept("exposure") if descriptive and not named_exposure else None
    primary_exposure = _observability_preserving_exposure(named_exposure, export_path)
    source_column = _bound_source_column(export_path)
    # A host-derived window reading is already one value per stay; an
    # aggregation phrase in the question applies only to a named source concept.
    exposure_operation = (
        explicit_exposure_aggregation(question, concept_id=primary_exposure)
        if primary_exposure and primary_exposure == named_exposure
        else None
    )
    endpoint = None
    configured = _clean_text(configured_outcome, 160)
    configured_catalog = catalog_by_id.get(configured) if configured else None
    if configured_catalog is not None:
        target_outcome = configured
    if descriptive and not target_outcome:
        # A grouped descriptive question may name a measurement instead of an
        # event in the intent reader. Only one owner-defined named quantity
        # can fill this otherwise empty slot; no convenient column is chosen.
        quantities = []
        for name in sorted(question_concepts - {named_exposure, *declared_covariates, *required_subgroup_concepts}):
            item = catalog_by_id.get(name)
            operation = explicit_exposure_aggregation(question, concept_id=name)
            if item is not None:
                proposed = catalog_endpoint(
                    item, column=source_column(name),
                    aggregation=operation.aggregation if operation is not None else None,
                )
                definition = load_dictionary().get(name)
                requested_measurement = (
                    definition is not None and definition.target != "id_tbl"
                    and (name == named_measurement or operation is not None
                         or (slots.get("time_window_hours") or {}).get("provenance") == "keyword")
                )
                if ((proposed is not None and (proposed.kind == "continuous" or checked_grouping))
                        or requested_measurement):
                    quantities.append(name)
        if len(quantities) == 1:
            target_outcome = quantities[0]
    target_catalog = catalog_by_id.get(str(target_outcome or ""))
    outcome_operation = (
        explicit_exposure_aggregation(question, concept_id=target_outcome)
        if target_outcome else None
    )
    outcome_window = None
    if (
        target_outcome
        and target_catalog is not None
    ):
        column = source_column(target_outcome)
        definition = load_dictionary().get(target_outcome)
        if descriptive and definition is not None and definition.target == "id_tbl":
            # A stay attribute is already one quantity per episode. "Mean"
            # there summarizes stays, not a second temporal reduction.
            outcome_operation = None
        if (descriptive and definition is not None and definition.target != "id_tbl"
                and target_catalog.column_role != "event_status"
                and (definition.units or outcome_operation is not None)):
            endpoint, outcome_window = _descriptive_window_summary(
                question=question, concept=target_catalog, column=column, operation=outcome_operation,
            )
            named_measurement = None
        else:
            endpoint = catalog_endpoint(
                target_catalog, column=column,
                aggregation=outcome_operation.aggregation if outcome_operation is not None else None,
            ) if column else None
    # The concept projected as the exposure is sealed as the exposure, which
    # accounts for it; a question that names nothing else keeps its identity.
    named_concepts = [
        {
            "concepts": list(
                dict.fromkeys(
                    source_column(concept)
                    for concept in concepts
                    if concept in catalog_by_id
                )
            ),
            "evidence": _clean_text(phrase, 120),
        }
        for concepts, phrase in named_study_concepts(question)
        if any(concept in catalog_by_id for concept in concepts)
        and _clean_text(phrase, 120)
        and named_exposure not in concepts
    ]
    coordinate_readings = []
    for role, concept, column, basis, slot_name in (
        ("outcome", target_outcome, endpoint.name if endpoint is not None else source_column(target_outcome),
         "configured" if configured_catalog is not None else "keyword",
         "exposure" if target_outcome == named_concept("exposure") else "outcome"),
        ("exposure", named_exposure, source_column(primary_exposure),
         "configured" if checked_grouping == named_exposure else "keyword", "exposure"),
        ("measurement", named_measurement, source_column(named_measurement), "keyword", "exposure"),
    ):
        capability = catalog_by_id.get(str(concept or ""))
        if capability is not None and column:
            slot = slots.get(slot_name) or {}
            # The slot's words quote the concept the keyword reader named; when
            # the stated exposure decided another one, they quote the wrong
            # concept, and the stated words are the reading's own.
            fragment = (
                _clean_text(slot.get("evidence"), 160)
                if named_concept(slot_name) == concept
                else _clean_text(stated_exposure, 160) if role == "exposure" else ""
            )
            if not fragment and role == "outcome" and checked_grouping:
                fragment = (
                    outcome_operation.evidence if outcome_operation is not None else
                    next((phrase for concepts, phrase in named_study_concepts(question)
                          if concept in concepts), "")
                )
            match = (
                re.search(r"\s+".join(re.escape(part) for part in fragment.split()), question, re.IGNORECASE)
                if fragment and basis == "keyword" else None
            )
            coordinate_readings.append({
                "role": role, "concept": concept, "column": column,
                "category": capability.category, "basis": basis,
                "evidence": match.group(0)[:160] if match is not None else None,
            })
    return {
        "target_outcome": source_column(target_outcome),
        "primary_exposure": source_column(primary_exposure),
        "coordinate_readings": coordinate_readings,
        "question_named_concepts": named_concepts,
        "required_adjustment_concepts": required_adjustment_concepts,
        "required_subgroup_concepts": required_subgroup_concepts,
        # A substance the question names as given or as measured: the plan
        # reads it in that form (planning.question_substance_forms).
        "question_substance_forms": [
            {
                **form,
                "concepts": [source_column(item) for item in form["concepts"]],
                "other": [source_column(item) for item in form["other"]],
                "evidence": _clean_text(form["evidence"], 120),
            }
            for form in substance_forms(question)
            if _clean_text(form["evidence"], 120)
        ],
        "primary_exposure_aggregation": (
            exposure_operation.aggregation if exposure_operation is not None else None
        ),
        "target_outcome_aggregation": (
            outcome_operation.aggregation if outcome_operation is not None else None
        ),
        "target_outcome_window": outcome_window,
        "endpoint": endpoint,
        "source": "question_keywords_plus_database_capability",
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
    additional_outcomes: tuple[str, ...] = (),
    analysis_inputs: tuple[str, ...] = (),
    trajectory_concepts: tuple[str, ...] = (),
) -> Dict[str, Any]:
    """Compile StudyContext modules into one typed materialization request.

    ``analysis_inputs`` are an accepted candidate's primary-analysis concepts
    (``planning.accepted_analysis_inputs``); they are materialized like any
    declared scientific input so the package-bound plan can keep them.
    ``trajectory_concepts`` are a declared trajectory design's concepts and
    the concepts its population reads; the long panel is cut from the
    materialized concepts, and the study cohort is selected on the same
    universe, so they must be among them.
    """

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
    stay_bound_concepts = primary_cohort.cohort_required_concepts(study.get("cohort"))
    for concept_id in primary_cohort.cohort_required_concepts(
        study.get("cohort"), study.get("analysis_design")
    ):
        cohort_meta = by_id.get(concept_id)
        if cohort_meta is None or Path(cohort_meta.file_name).stem.lower() not in {
            "demographics",
            "outcome",
        }:
            raise ResearchPipelineRunError(
                "research_pipeline_cohort_concept_unavailable",
                "The typed cohort reads a stay-level concept the selected modules do not provide.",
                details={
                    # A declared prediction study's population reads it too.
                    "field": (
                        "cohort.min_icu_los_hours"
                        if concept_id in stay_bound_concepts
                        else "analysis_design.analysis_family"
                    ),
                    "concept_id": concept_id,
                },
            )
        if concept_id not in static_concepts:
            static_concepts.append(concept_id)
    # A first-stay cohort needs no readmission column: the host restricts the
    # universe with its verified first-ICU-stay coordinate before planning.
    requested_outcomes = tuple(
        dict.fromkeys(
            value
            for value in (
                target,
                *additional_outcomes,
                *explicit_outcome_concepts(str(study.get("question") or "")),
            )
            if isinstance(value, str) and value.strip()
        )
    )
    for requested_outcome in requested_outcomes:
        target_meta = by_id.get(requested_outcome)
        if target_meta is None:
            if require_target or requested_outcome != target:
                raise ResearchPipelineRunError(
                    (
                        "research_pipeline_target_outside_configured_modules"
                        if requested_outcome == target
                        else "research_pipeline_outcome_outside_configured_modules"
                    ),
                    (
                        "A plan-required outcome is not available in the "
                        "selected feature modules."
                    ),
                    details={
                        "field": "execution_concepts.outcome",
                        "concept_id": requested_outcome,
                    },
                )
        else:
            target_module = Path(target_meta.file_name).stem.lower()
            if target_meta.column_role == "event_status":
                outcome_concepts.append(requested_outcome)
                require_outcome = True
            elif target_module in {"demographics", "outcome"}:
                static_concepts.append(requested_outcome)
            else:
                required_feature_concepts.append(requested_outcome)

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
            for value in (
                primary_exposure,
                *covariates,
                *sensitivity_variables,
                *analysis_inputs,
                *trajectory_concepts,
            )
            if value and value != target
        )
    )
    from easyicu.webserver.scientific_runtime_projection import (
        host_derived_design_sources,
    )

    primary_exposure_source_concept: Optional[str] = None
    for concept_id in scientific_inputs:
        derived_sources = host_derived_design_sources(concept_id)
        if concept_id == primary_exposure and derived_sources is not None:
            # A declared cross-concept exposure (the strict KDIGO stage) has no
            # single exported source.  The acquisition owner materializes it
            # from the concepts its declaration reads, so it is available
            # exactly when every one of them is in the selected modules.
            missing_sources = [
                source for source in derived_sources if source not in by_id
            ]
            if not missing_sources or not require_primary_exposure:
                continue
            raise ResearchPipelineRunError(
                "research_pipeline_primary_exposure_outside_configured_modules",
                "The configured primary exposure is derived from source concepts the selected feature modules do not provide.",
                details={
                    "field": "execution_concepts.primary_exposure",
                    "concept_id": concept_id,
                    "missing_source_concepts": missing_sources,
                },
            )
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
                else "covariate"
                if concept_id in covariates
                else "sensitivity_variable"
                if concept_id in sensitivity_variables
                else "analysis_input"
                if concept_id in analysis_inputs
                else "trajectory_concept"
            )
            raise ResearchPipelineRunError(
                f"research_pipeline_{role}_outside_configured_modules",
                f"The configured {role.replace('_', ' ')} is not available in the selected feature modules.",
                details={
                    "field": {
                        "analysis_input": "candidate_plan.steps.inputs",
                        "trajectory_concept": "trajectory_design",
                    }.get(role, f"execution_concepts.{role}"),
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

    first_icu_stay = _first_icu_stay_for_cohort(study)
    return {
        "allowed_modules": modules,
        "available_concepts": tuple(by_id),
        "static_concepts": tuple(dict.fromkeys(static_concepts)),
        "outcome_concepts": tuple(outcome_concepts),
        "required_feature_concepts": tuple(required_feature_concepts),
        "require_outcome": require_outcome,
        "primary_exposure_source_concept": primary_exposure_source_concept,
        # The host restricts the universe itself to each patient's first ICU
        # stay; a cohort that keeps every stay carries no such coordinate.
        **({"first_icu_stay": first_icu_stay} if first_icu_stay is not None else {}),
    }


@dataclasses.dataclass(frozen=True)
class TargetTrialLaunchInputs:
    """What a run of a study with a target trial acquires.

    The host's compile job for the trial's card acquires the same, so the
    record the researcher approves is the record the run compiles again on
    its own context (``bind_confirmed_target_trial``): covariates summarized
    over ``[0, T0)``, the treatment's onsets read over ``[0, T0 + G)``, the
    trial's concepts and the study's modules, every concept by the host.
    """

    cohort_window: tuple[float, float]
    event_onset_windows: Mapping[str, tuple[float, float]]
    target_outcome: str
    foundation_profile: Mapping[str, Any]

    @property
    def materialization_window(self) -> Dict[str, Any]:
        """The study's ``time_window`` as the run executes it: ``[0, T0)``.

        A run declares the window it summarized its columns over, so its
        scope carries the trial's covariate window in place of the study's.
        """

        start, end = self.cohort_window
        if start != 0.0:
            raise ValueError("a trial's covariate window opens at ICU admission")
        return {"hours": end, "anchor": "icu_admission"}

    def acquisition_arguments(
        self, *, database: str, patient_grouping: Any
    ) -> Dict[str, Any]:
        """The acquisition's arguments besides its export, question, model and output."""

        profile = self.foundation_profile
        return {
            "target_outcome": self.target_outcome,
            "primary_exposure_concept": None,
            # A concept the trial reads twice -- a confounder its population
            # also reads -- is acquired once.
            "outcome_concepts": list(dict.fromkeys(profile["outcome_concepts"])),
            "required_feature_concepts": list(
                dict.fromkeys(profile["required_feature_concepts"])
            ),
            "static_concepts": list(dict.fromkeys(profile["static_concepts"])),
            "allowed_modules": profile["allowed_modules"],
            "concept_selection_authority": "host_exact",
            "cohort_window": self.cohort_window,
            "trajectory_window": self.cohort_window,
            "database": database,
            "require_outcome": profile["require_outcome"],
            "emit_trajectory": False,
            "patient_grouping": patient_grouping,
            "host_derivations": (),
            "first_icu_stay": profile.get("first_icu_stay"),
            "event_onset_windows": dict(self.event_onset_windows),
        }


def target_trial_launch_inputs(
    *,
    export_path: str,
    materialization_study: Mapping[str, Any],
    spec: TargetTrialSpec,
    population_spec: PopulationSpec,
) -> TargetTrialLaunchInputs:
    """What a run of ``materialization_study`` acquires for ``spec``.

    The study's modules bound what can be read, as for any run; a concept the
    trial reads that they do not provide raises the launch's own typed error
    with the concept, so the compile job and the run stop alike.
    """

    acquisition = target_trial_acquisition(spec, population_spec)
    profile = _data_foundation_profile(
        export_path=export_path,
        study=materialization_study,
        target=acquisition.target_outcome,
        require_target=True,
        require_primary_exposure=False,
        covariates=acquisition.confounders,
        additional_outcomes=acquisition.other_outcomes,
        analysis_inputs=acquisition.design_concepts,
    )
    return TargetTrialLaunchInputs(
        cohort_window=acquisition.cohort_window,
        event_onset_windows=acquisition.event_onset_windows,
        target_outcome=acquisition.target_outcome,
        foundation_profile=profile,
    )
