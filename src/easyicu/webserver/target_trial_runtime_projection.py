"""Web adapter for the signed target trial emulation suite.

Owner
-----
This sibling of ``scientific_runtime_projection`` compiles a StudyContext
whose ``analysis_design.analysis_family`` is ``causal_inference`` and whose
``target_trial_design`` the researcher approved into the digest-bound
``TargetTrialRuntimeAuthority``.  The study setup states the trial, the host
compiled it for the approval card and the researcher's click approved that
record (``research_agent.planning.target_trial_configuration``); the host
keeps the record under the digest the study names (``target_trial_records``).
This adapter reads the approved record, never a description of it:

* the times, the strategies' labels, the onset columns and the endpoint come
  from the stated trial;
* the confounders are those the record carries at time zero, coded by the
  domain the materializer published for the column, else by the concept
  owner's declared domain; a measurement summarized over a window keeps its
  unmeasured state, a static demographic does not;
* the onset window is the one the materializer read the onset columns over,
  from ICU admission through the grace period;
* the bootstrap resamples patients when the run binds a patient grouping,
  else ICU stays.

It reads the materialized universe's schema and column metadata only.  The
run compiles the trial again on its own research context and requires the
approved record (``bind_confirmed_target_trial``), so this adapter checks only
what it reads.  A study whose trial is not approved routes nothing here: its
causal plan stops before the Planner is called (``tte_trial_not_confirmed``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from pydantic import ValidationError

from easyicu.concept.catalog import CONCEPT_DICTIONARY
from easyicu.outcome_availability import fixed_horizon_mortality_endpoint
from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.contracts.target_trial_design import (
    target_trial_host_policy_sha256,
)
from easyicu.research_agent.icu_rules import VariableKind
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
)
from easyicu.research_agent.planning.analysis_types import canonical_analysis_family
from easyicu.research_agent.planning.target_trial_configuration import (
    ConfirmedTargetTrial,
    TargetTrialApproval,
    TargetTrialCompileRecord,
    TargetTrialDesign,
    TargetTrialDesignError,
    load_target_trial_design,
)
from easyicu.research_agent.planning.target_trial_spec import TargetTrialSpec
from easyicu.research_agent.research_context.stay_events import (
    ICU_LENGTH_OF_STAY_CONCEPT,
)
from easyicu.research_agent.research_context.typed import declared_domain_for_variable
from easyicu.research_agent.schema import ConceptDescriptor
from easyicu.utils.death_time_semantics import DEATH_STATUS, DEATH_TIME_COMPANION

from . import primary_cohort
from .scientific_runtime_projection import (
    WebScientificRuntimeProjection,
    WebScientificRuntimeProjectionError,
    exposure_kind_for_dtype,
    signed_projection,
)
from .target_trial_records import TargetTrialRecordError, load_target_trial_record

_FAMILY = "causal_inference"
_ONSET_SUFFIX = "_onset_time"
_WINDOW_ORIGIN = "icu_admission"
_PRODUCTS = (
    "table:target_trial_protocol",
    "table:target_trial_eligibility_flow",
    "table:target_trial_table_one",
    "table:target_trial_risk_curves",
    "table:target_trial_effect_estimates",
    "table:target_trial_weight_models",
    "table:target_trial_weight_diagnostics",
    "table:target_trial_covariate_balance",
    "table:target_trial_positivity",
    "table:target_trial_adherence",
    "table:target_trial_bootstrap_replicates",
    "log:target_trial_runtime_receipt",
    "figure:target_trial_emulation",
)
_PRODUCT_FIELDS = (
    "protocol_product",
    "eligibility_product",
    "table_one_product",
    "risk_curve_product",
    "effect_product",
    "weight_model_product",
    "weight_product",
    "balance_product",
    "positivity_product",
    "adherence_product",
    "bootstrap_product",
    "receipt_product",
    "figure_product",
)
#: How a column summarizes its concept over the window before time zero.
_SUMMARIES = {
    "_max": "highest",
    "_min": "lowest",
    "_mean": "mean",
    "_first": "first",
    "_last": "last",
}
#: Characters a reader label of the authority never holds.
_UNREADABLE = set("{}[]<>`\\|*_#\n")


def target_trial_family_declared(study: Mapping[str, Any]) -> bool:
    """Whether the typed analysis design names the causal inference family."""

    design = study.get("analysis_design")
    if not isinstance(design, Mapping):
        return False
    return canonical_analysis_family(design.get("analysis_family")) == _FAMILY


@dataclass(frozen=True)
class _ApprovedTrial:
    """The study's approved section with the record the host keeps for it."""

    section: TargetTrialDesign
    kept: TargetTrialCompileRecord

    @property
    def spec(self) -> TargetTrialSpec:
        return self.kept.spec

    @property
    def compile_record(self) -> Mapping[str, Any]:
        return self.kept.record

    @property
    def compile_sha256(self) -> str:
        return self.section.compile_sha256

    @property
    def confirmation_lines(self) -> int:
        return self.section.confirmation_lines

    @property
    def approval(self) -> Optional[TargetTrialApproval]:
        return self.section.approval

    def confirmed(self) -> Optional[ConfirmedTargetTrial]:
        return self.section.confirmed(self.kept)


def _kept(study_id: str, section: TargetTrialDesign) -> _ApprovedTrial:
    """The record the approval names, as the host keeps it under that digest."""

    try:
        kept = load_target_trial_record(study_id, section.compile_sha256)
        section.check_record(kept)
    except TargetTrialRecordError as exc:
        raise WebScientificRuntimeProjectionError(
            "target_trial_configuration_invalid",
            "The host does not keep the record the study's approval names.",
            details={
                "field": "target_trial_design.compile_sha256",
                "reason_code": exc.code,
            },
        ) from exc
    except TargetTrialDesignError as exc:
        raise WebScientificRuntimeProjectionError(
            "target_trial_configuration_invalid",
            "The record kept is not the one the study's approval names.",
            details={"field": exc.field, "reason_code": exc.code},
        ) from exc
    return _ApprovedTrial(section=section, kept=kept)


def _design(study: Mapping[str, Any]) -> Optional[TargetTrialDesign]:
    raw = study.get("target_trial_design")
    if raw is None or (isinstance(raw, Mapping) and not raw):
        return None
    study_id = str(study.get("id") or "").strip() or None
    try:
        design = load_target_trial_design(raw, study_id=study_id)
    except TargetTrialDesignError as exc:
        raise WebScientificRuntimeProjectionError(
            "target_trial_configuration_invalid",
            "The study's target trial section breaks its contract.",
            details={"field": exc.field, "reason_code": exc.code},
        ) from exc
    if design is not None and design.approval is not None and study_id is None:
        raise WebScientificRuntimeProjectionError(
            "target_trial_configuration_invalid",
            "An approved target trial needs the study its approval was minted for.",
            details={
                "field": "id",
                "reason_code": "target_trial_approval_event_mismatch",
            },
        )
    return design


def _mismatch(design: _ApprovedTrial, message: str, **details: Any) -> None:
    """The extraction does not hold what the trial reads: name what it needs."""

    materialization = design.compile_record.get("materialization") or {}
    raise WebScientificRuntimeProjectionError(
        "target_trial_materialization_mismatch",
        message,
        details={
            **details,
            "covariate_window": materialization.get("covariate_window"),
            "treatment_onset_window": materialization.get("treatment_onset_window"),
        },
    )


def _schema_types(universe_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(universe_path)
    except Exception as exc:  # noqa: BLE001 - retyped at this owner boundary
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_schema_unavailable",
            "The materialized universe schema could not be read for the target trial.",
            details={"artifact": Path(universe_path).name, "reason": str(exc)[:500]},
        ) from exc
    return {field.name: field.type for field in schema}


def _verified(universe_path: Path, design: _ApprovedTrial) -> Any:
    try:
        verified = load_verified_materialized_cohort_authority(Path(universe_path))
    except MaterializedMetadataError as exc:
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_metadata_unverified",
            "The materialized universe's column metadata could not be verified "
            "for the target trial.",
            details={"artifact": Path(universe_path).name, "reason": str(exc)[:500]},
        ) from exc
    if verified is None:
        _mismatch(
            design,
            "The prepared data records no windows its columns were read over, "
            "which the target trial reads.",
            reason="no_column_metadata",
        )
    return verified


def _bindings(verified: Any) -> dict[str, Any]:
    return {
        name: binding
        for file_binding in verified.sidecar.files
        for name, binding in file_binding.columns.items()
    }


def _onset_window(
    design: _ApprovedTrial, bindings: Mapping[str, Any], onsets: Sequence[str]
) -> list[float]:
    """The one window the onset columns were read over, through the grace period."""

    spec = design.spec
    grace_end = spec.time_zero.hours_after_icu_admission + spec.grace_period.hours
    windows = set()
    for column in onsets:
        binding = bindings.get(column)
        window = getattr(binding, "derivation_window", None)
        if (
            window is None
            or window.origin != _WINDOW_ORIGIN
            or window.start_hours > 0
            or window.end_hours < grace_end
        ):
            _mismatch(
                design,
                "A treatment onset was not read from ICU admission through the "
                "trial's grace period.",
                onset_column=column,
                read_over=window.to_dict() if window is not None else None,
            )
        windows.add((float(window.start_hours), float(window.end_hours)))
    if len(windows) != 1:
        _mismatch(
            design,
            "The treatment's onset columns were read over different windows.",
            onset_columns=list(onsets),
        )
    start, end = windows.pop()
    return [start, end]


def _source_concept(column: str) -> str:
    for suffix in _SUMMARIES:
        stem = column[: -len(suffix)]
        if column.endswith(suffix) and stem in CONCEPT_DICTIONARY:
            return stem
    return column


def _readable(text: str) -> str:
    cleaned = "".join(" " if char in _UNREADABLE else char for char in text)
    words = " ".join(cleaned.split())
    return (words[:1].upper() + words[1:])[:160] if words else ""


def _covariate_label(column: str) -> str:
    source = _source_concept(column)
    entry = CONCEPT_DICTIONARY.get(source)
    name = (
        str(entry[0])
        if isinstance(entry, (tuple, list)) and entry and str(entry[0]).strip()
        else source
    )
    if source == column:
        return _readable(name)
    return _readable(f"{name}, {_SUMMARIES[column[len(source) :]]} before time zero")


def _covariate(
    column: str,
    *,
    role: str,
    column_type: Any,
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """One confounder as its column enters the weight models."""

    import pyarrow as pa

    source = _source_concept(column)
    binding = bindings.get(column)
    published = (
        tuple(binding.metadata.allowed_values or ()) if binding is not None else None
    )
    kind, levels = exposure_kind_for_dtype(
        primary_exposure=column,
        primary_exposure_source=source,
        dtype=str(column_type),
        published_levels=published,
    )
    numeric = (
        pa.types.is_integer(column_type)
        or pa.types.is_floating(column_type)
        or pa.types.is_decimal(column_type)
    )
    measured = kind in {
        VariableKind.CONTINUOUS,
        VariableKind.COUNT,
        VariableKind.ORDINAL,
    }
    if kind == VariableKind.BINARY and len(levels) == 2:
        coding, levels = "binary", list(levels)
    elif measured and numeric:
        # A score enters the weights linearly, as a measurement does.
        coding, levels = "continuous", []
    elif kind in {VariableKind.CATEGORICAL, VariableKind.ORDINAL}:
        declared, _basis = declared_domain_for_variable(
            ConceptDescriptor(
                name=column, dtype=str(column_type), source_concept=source
            )
        )
        levels = [str(value) for value in (levels or declared or ())]
        coding = "binary" if len(levels) == 2 else "categorical"
    else:
        coding = ""
    if coding != "continuous" and (len(levels) < 2 or len(set(levels)) != len(levels)):
        raise WebScientificRuntimeProjectionError(
            "web_scientific_runtime_covariate_encoding_unsupported",
            "A target trial confounder has no deterministic coding the weights "
            "can use.",
            details={"column": column, "parquet_type": str(column_type)},
        )
    return {
        "column": column,
        "label": _covariate_label(column),
        "coding": coding,
        "levels": levels,
        # The weights do not depend on which level is the reference.
        "reference_level": levels[0] if levels else None,
        "unmeasured_state": role == "at_or_before_time_zero",
    }


def _resampling(dependence: Any) -> dict[str, Any]:
    if dependence is None:
        return {
            "resampling_unit": "icu_stay",
            "patient_group_column": None,
            "patient_group_derivation": None,
            "patient_group_delimiter": None,
        }
    return {
        "resampling_unit": "patient",
        "patient_group_column": dependence.group_source,
        "patient_group_derivation": dependence.group_derivation,
        "patient_group_delimiter": dependence.delimiter,
    }


def _authority_body(
    *,
    study: Mapping[str, Any],
    design: _ApprovedTrial,
    columns: Mapping[str, Any],
    bindings: Mapping[str, Any],
    unit_id_column: str,
    onset_window: list[float],
    dependence: Any,
    scientific_configuration_sha256: str,
) -> dict[str, Any]:
    spec = design.spec
    approval = design.approval
    assert approval is not None  # routed only once approved
    record = design.compile_record
    endpoint = fixed_horizon_mortality_endpoint(spec.outcome.endpoint)
    if endpoint is None:  # the record is approvable only for one
        raise WebScientificRuntimeProjectionError(
            "target_trial_configuration_invalid",
            "The approved trial's outcome is not a closed fixed-horizon death.",
            details={"field": "target_trial_design.spec.outcome.endpoint"},
        )
    unit_label = (
        "first ICU stays"
        if primary_cohort.first_icu_stay_only(study.get("cohort"))
        else "ICU stays"
    )
    confounders = [
        item
        for item in record.get("confounders") or ()
        if item.get("disposition") == "applied"
    ]
    covariates = [
        _covariate(
            str(item["name"]),
            role=str(item.get("temporal_role") or ""),
            column_type=columns[str(item["name"])],
            bindings=bindings,
        )
        for item in confounders
    ]
    t0 = spec.time_zero.hours_after_icu_admission
    grace = spec.grace_period.hours
    treatment = f"{spec.treatment.treatment_class.replace('_', ' ')} treatment"
    return {
        "schema_version": "easyicu.target_trial_runtime_authority/1",
        "authority_kind": "target_trial_suite",
        "protocol_content_sha256": scientific_configuration_sha256,
        "target_trial_compile_sha256": design.compile_sha256,
        "target_trial_compile_confirmation_lines": design.confirmation_lines,
        "host_policy_sha256": target_trial_host_policy_sha256(),
        "confirmation": {
            "confirmed_by": "researcher",
            "approval_event_id": approval.approval_event_id,
            "confirmed_compile_sha256": approval.confirmed_compile_sha256,
            "n_lines_confirmed": approval.n_lines_confirmed,
        },
        "plan_method": "signed_target_trial_suite",
        "plan_intent": (
            f"Execute the signed emulation of a target trial of starting {treatment} "
            f"within {grace} hours of time zero, {t0} hours after ICU admission, "
            f"against not starting it then, for death by day {endpoint.horizon_days}, "
            "by clone, censor and weight."
        ),
        "plan_outputs": list(_PRODUCTS),
        "development_execution_only_allowed": False,
        "database": str(record.get("database") or ""),
        "analysis_unit_label": unit_label,
        "eligibility_label": f"{unit_label[:1].upper()}{unit_label[1:]} in the study population",
        "treatment_label": treatment,
        "initiate_label": spec.strategies.initiate_label,
        "defer_label": spec.strategies.defer_label,
        "outcome_label": "Death",
        "unit_id_column": unit_id_column,
        **_resampling(dependence),
        "treatment_onset_columns": [
            f"{c}{_ONSET_SUFFIX}" for c in spec.treatment.concepts
        ],
        "treatment_onset_window_hours": onset_window,
        "time_zero_hours": t0,
        "grace_period_hours": grace,
        "event_column": endpoint.event_concept,
        "followup_time_column": endpoint.followup_concept,
        "endpoint_horizon_days": endpoint.horizon_days,
        "endpoint_time_origin": endpoint.time_origin,
        "death_status_column": DEATH_STATUS,
        "death_time_column": DEATH_TIME_COMPANION,
        "icu_length_of_stay_column": ICU_LENGTH_OF_STAY_CONCEPT,
        "covariates": covariates,
        "estimator": "clone_censor_weight",
        "interpretation": "per_protocol_effect_under_emulation_assumptions",
        "evidence_ceiling": "analysis_only",
        **dict(zip(_PRODUCT_FIELDS, _PRODUCTS)),
    }


def compile_target_trial_runtime_projection(
    *,
    study: Mapping[str, Any],
    universe_path: Path,
    scientific_configuration_sha256: str,
    dependence: Any = None,
    **_coordinates: Any,
) -> WebScientificRuntimeProjection | None:
    """Bind a study's approved target trial to the signed emulation suite.

    Returns ``None`` for a study that states no trial or whose trial the
    researcher has not approved, so its causal plan stops before planning.
    The other coordinates every adapter receives name the study's own
    exposure and adjustment roster, which the trial does not read.
    """

    section = _design(study)
    if section is None:
        return None
    if not target_trial_family_declared(study):
        raise WebScientificRuntimeProjectionError(
            "target_trial_family_mismatch",
            "The study states a target trial but does not declare a causal design.",
            details={
                "field": "analysis_design.analysis_family",
                "required_family": _FAMILY,
            },
        )
    if section.approval is None:
        return None
    # ``_design`` refuses an approval without the study it was minted for.
    design = _kept(str(study.get("id")), section)
    confirmed = design.confirmed()
    if confirmed is None:  # pragma: no cover - the approval was checked above
        return None
    verified = _verified(Path(universe_path), design)
    bindings = _bindings(verified)
    columns = _schema_types(Path(universe_path))
    materialization = design.compile_record.get("materialization") or {}
    unit_id_column = str(verified.authority.identity_column)
    group = [dependence.group_source] if dependence is not None else []
    required = [
        unit_id_column,
        *group,
        *(str(item) for item in materialization.get("columns") or ()),
    ]
    absent = sorted({column for column in required if column not in columns})
    if absent:
        _mismatch(
            design,
            "The prepared data lacks columns the target trial reads.",
            missing_columns=absent,
        )
    onset_window = _onset_window(
        design,
        bindings,
        [f"{c}{_ONSET_SUFFIX}" for c in design.spec.treatment.concepts],
    )
    body = _authority_body(
        study=study,
        design=design,
        columns=columns,
        bindings=bindings,
        unit_id_column=unit_id_column,
        onset_window=onset_window,
        dependence=dependence,
        scientific_configuration_sha256=scientific_configuration_sha256,
    )
    try:
        authority = build_current_case_scientific_runtime_authority(body).model_dump(
            mode="json"
        )
    except ValidationError as exc:
        raise WebScientificRuntimeProjectionError(
            "target_trial_configuration_invalid",
            "The approved target trial cannot be signed for execution.",
            details={
                "fields": sorted(
                    {
                        ".".join(str(part) for part in error.get("loc") or ())
                        for error in exc.errors(include_url=False, include_input=False)
                    }
                )[:8],
            },
        ) from exc
    projection = signed_projection(
        authority, scientific_configuration_sha256=scientific_configuration_sha256
    )
    return replace(projection, bound_target_trial=confirmed.model_dump(mode="json"))


__all__ = [
    "compile_target_trial_runtime_projection",
    "target_trial_family_declared",
]
