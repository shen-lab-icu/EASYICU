"""Typed, read-only choices after a run stops; none is execution authority.

The UI sends plan-changing choices through its plan editor and the existing
fresh-candidate submission contract. A choice never changes the question,
substitutes an input, approves a plan, or starts a run while being projected.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from easyicu.concept.selection_policy import concept_selection_policy
from easyicu.research_agent.authority.baseline_association_runtime import BaselineAssociationAuthorityError
from easyicu.research_agent.concept_availability import explain_concept_availability
from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS
from easyicu.research_agent.providers.client_trust import PROVIDER_NOT_CLEARED
from easyicu.webserver.research_pipeline_run_errors import WINDOWED_SUMMARY_STOP_MESSAGES
from easyicu.webserver.study_family_scope import v1_scope_description

ActionKind = Literal[
    "rerun_without_element",
    "edit_plan",
    "use_alternative_concept",
    "drop_variable",
    "retry",
    "rephrase_in_scope",
    "open_provider_settings",
    "reexport_data",
]
_CODE = re.compile(r"[a-z][a-z0-9_]{0,79}")
Code = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")]
StepId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")]


class PlanEditParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["plan_edit"] = "plan_edit"
    intent: Literal["candidate_plan"] = "candidate_plan"
    planner_start_mode: Literal["fresh"] = "fresh"
    requires_explicit_choice: Literal[True] = True
    requires_plan_approval: Literal[True] = True
    element: Code | None = None
    concept: Code | None = None
    alternative_concept: Code | None = None
    database: Code | None = None
    step_id: StepId | None = None


class RetryParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["retry"] = "retry"


class ProviderSettingsParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["provider_settings"] = "provider_settings"


class DataExtractionParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["data_extraction"] = "data_extraction"
    requires_explicit_choice: Literal[True] = True


class ScopeParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: Literal["question_edit"] = "question_edit"
    scope_owner: Literal["easyicu.webserver.study_family_scope"] = (
        "easyicu.webserver.study_family_scope"
    )
    scope_label_key: Literal["v1_scope.target_trial", "v1_scope.study_family"]
    database: Code | None = None
    family: Code | None = None
    supported_databases: tuple[Code, ...] = Field(default=(), max_length=8)


class StopNextAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str = Field(pattern=r"^[a-z_]+_[a-f0-9]{16}$")
    kind: ActionKind
    label_key: str
    params: PlanEditParams | RetryParams | ScopeParams | ProviderSettingsParams | DataExtractionParams

    @model_validator(mode="after")
    def _matching_params(self) -> StopNextAction:
        expected = {
            "open_provider_settings": ProviderSettingsParams,
            "reexport_data": DataExtractionParams,
            "retry": RetryParams,
            "rephrase_in_scope": ScopeParams,
        }.get(self.kind, PlanEditParams)
        if not isinstance(self.params, expected):
            raise ValueError("action kind and params disagree")
        if isinstance(self.params, PlanEditParams):
            if (
                self.kind in {"drop_variable", "use_alternative_concept"}
                and self.params.concept is None
            ):
                raise ValueError("variable choice needs its source concept")
            if self.kind == "use_alternative_concept" and (
                self.params.alternative_concept is None or self.params.database is None
            ):
                raise ValueError(
                    "alternative choice needs the owner-issued alternative"
                )
            if self.kind == "rerun_without_element" and not (
                self.params.element or self.params.step_id
            ):
                raise ValueError("removing an element needs its typed coordinate")
        return self


# Explicit dispositions of the Web run owners' terminal codes. The owner-code
# enumeration test requires a decision here when an owner introduces a stop.
_EDIT_CODES = frozenset(
    {
        # A plan that refuses approval is edited like any other plan review.
        *PLAN_APPROVAL_STOPS,
        # Unbound measurement summaries need a new explicit design choice.
        *WINDOWED_SUMMARY_STOP_MESSAGES,
        # An unverified restriction cannot be silently dropped or retried.
        "research_pipeline_hospital_first_stay_restriction_unverified",
        # A revision whose source plan is gone or changed starts from a fresh plan.
        "plan_revision_source_plan_missing",
        # A population predicate emptied a declared group; an unchanged retry
        # repeats it, so the plan is generated again.
        "declared_exposure_level_empty",
        "web_baseline_association_incomplete",
        "family_spec_stated_adjustment_unavailable",
        "family_spec_subgroup_unavailable",
        "plan_revision_source_plan_mismatch",
        "accepted_baseline_resume_materialization_mismatch",
        "accepted_baseline_source_unresolved",
        "bound_literature_schema_invalid",
        "candidate_plan_materialization_authority_invalid",
        "candidate_plan_seed_too_large",
        "coverage_gate_failed",
        "human_plan_review_rejected",
        "human_plan_review_required",
        "numeric_evidence_gate_failed",
        "operator_plan_approval_required",
        "plan_change_required_concepts_unavailable",
        "plan_change_requirements_source_invalid",
        "plan_change_review_feedback_invalid",
        "plan_changes_require_fresh_candidate",
        "plan_revision_owner_resolution_required",
        "plan_revision_scientific_review_missing",
        "plan_revision_source_configuration_superseded",
        "plan_revision_source_not_changes_required",
        "plan_revision_source_not_found",
        "plan_scientific_changes_required",
        "prepared_plan_revision_resume_conflict",
        "prepared_plan_revision_source_invalid",
        "privacy_gate_failed",
        "research_agent_pipeline_failed_closed",
        "research_pipeline_analysis_design_conflict",
        "research_pipeline_analysis_design_incomplete",
        "research_pipeline_analysis_design_invalid",
        "research_pipeline_analysis_design_required",
        "research_pipeline_analysis_family_invalid",
        "research_pipeline_budget_mode_invalid",
        "research_pipeline_plan_step_needs_sandbox",
        "research_pipeline_cancelled",
        "research_pipeline_cluster_unit_unsupported",
        "research_pipeline_cluster_variance_unsupported",
        "research_pipeline_codex_user_auth_provider_required",
        "research_pipeline_codex_user_auth_required",
        "research_pipeline_cohort_concept_unavailable",
        "research_pipeline_conflicting_sealed_suites",
        "research_pipeline_credential_source_invalid",
        "research_pipeline_data_constraints_too_large",
        "research_pipeline_database_required",
        "research_pipeline_database_unknown",
        "research_pipeline_descriptive_confirmation_conflict",
        "research_pipeline_development_resume_acquisition_ambiguous",
        "research_pipeline_development_resume_acquisition_authority_mismatch",
        "research_pipeline_development_resume_acquisition_digest_mismatch",
        "research_pipeline_development_resume_acquisition_invalid",
        "research_pipeline_development_resume_acquisition_missing",
        "research_pipeline_development_resume_acquisition_unreadable",
        "research_pipeline_development_resume_budget_invalid",
        "research_pipeline_development_resume_checkpoint_invalid",
        "research_pipeline_development_resume_checkpoint_missing",
        "research_pipeline_development_resume_checkpoint_unreadable",
        "research_pipeline_development_resume_identity_authority_mismatch",
        "research_pipeline_development_resume_job_invalid",
        "research_pipeline_development_resume_literature_invalid",
        "research_pipeline_development_resume_literature_missing",
        "research_pipeline_development_resume_phase_not_replayed",
        "research_pipeline_development_resume_plan_contract_mismatch",
        "research_pipeline_development_resume_scope_invalid",
        "research_pipeline_development_resume_scope_mismatch",
        "research_pipeline_development_resume_sequence_invalid",
        "research_pipeline_development_resume_sequence_missing",
        "research_pipeline_development_resume_source_ambiguous",
        "research_pipeline_development_resume_source_invalid",
        "research_pipeline_development_resume_source_missing",
        "research_pipeline_execution_failed",
        "research_pipeline_execution_retry_checkpoint_ambiguous",
        "research_pipeline_execution_retry_checkpoint_missing",
        "research_pipeline_execution_retry_configuration_superseded",
        "research_pipeline_execution_retry_futile",
        "research_pipeline_execution_retry_input_invalid",
        "research_pipeline_execution_retry_path_invalid",
        "research_pipeline_execution_retry_recovery_seed_invalid",
        "research_pipeline_execution_retry_recovery_seed_missing",
        "research_pipeline_execution_retry_recovery_seed_superseded",
        "research_pipeline_execution_retry_source_not_failed_execution",
        "research_pipeline_execution_retry_source_not_found",
        "research_pipeline_execution_retry_unexpected_plan_review",
        "research_pipeline_execution_runtime_unavailable",
        "research_pipeline_first_stay_restriction_not_materialized",
        "research_pipeline_first_stay_restriction_unverified",
        "research_pipeline_kdigo_observability_authority_missing",
        "research_pipeline_landmark_authority_ambiguous",
        "research_pipeline_literature_path_invalid",
        "research_pipeline_literature_projection_privacy_failed",
        "research_pipeline_literature_run_id_missing",
        "research_pipeline_modules_required",
        "research_pipeline_package_binding_missing",
        "research_pipeline_pdf_document_digest_mismatch",
        "research_pipeline_pdf_document_missing",
        "research_pipeline_pdf_document_unsafe",
        "research_pipeline_pdf_receipt_binding_invalid",
        "research_pipeline_pdf_receipt_invalid",
        "research_pipeline_pdf_signature_invalid",
        "research_pipeline_plan_contract_exhausted",
        "research_pipeline_plan_outcome_catalog_unavailable",
        "research_pipeline_plan_outcome_materialization_unavailable",
        "research_pipeline_plan_primary_outcome_missing",
        "research_pipeline_planner_canary_execution_blocked",
        "research_pipeline_planner_efficiency_budget_exhausted",
        "research_pipeline_planner_exposure_proposal_unavailable",
        "research_pipeline_planner_provider_unavailable",
        "research_pipeline_planning_catalog_unavailable",
        "research_pipeline_planning_identity_unavailable",
        "research_pipeline_planning_operationalization_invalid",
        "research_pipeline_planning_patient_grouping_authority_invalid",
        "research_pipeline_prepared_source_catalog_unavailable",
        "research_pipeline_primary_exposure_aggregation_required",
        "research_pipeline_primary_exposure_outside_configured_modules",
        "research_pipeline_profile_dictionary_mismatch",
        "research_pipeline_progressive_compile_failed",
        "research_pipeline_project_workspace_required",
        "research_pipeline_provider_ledger_name_invalid",
        "research_pipeline_provider_recovery_probe_failed",
        "research_pipeline_provider_timeout",
        "research_pipeline_question_required",
        "research_pipeline_repeated_stay_dependence_unaddressed",
        "research_pipeline_required_concept_structurally_unavailable",
        "research_pipeline_required_concept_without_values",
        "research_pipeline_review_checkpoint_not_resumable",
        "research_pipeline_review_config_not_recoverable",
        "research_pipeline_review_configuration_superseded",
        "research_pipeline_review_decision_already_recorded",
        "research_pipeline_review_decision_invalid",
        "research_pipeline_review_not_resumable",
        "research_pipeline_review_recovery_capacity_full",
        "research_pipeline_review_recovery_failed",
        "research_pipeline_review_resume_in_progress",
        "research_pipeline_review_study_mismatch",
        "research_pipeline_reviewable_plan_required",
        "research_pipeline_runner_image_invalid",
        "research_pipeline_runner_image_mismatch",
        "research_pipeline_schema_validation_failed",
        "research_pipeline_scientific_authority_refused",
        "research_pipeline_stay_denominator_unavailable",
        "research_pipeline_target_outcome_materialization_unavailable",
        "research_pipeline_target_trial_plans_on_data",
        "research_pipeline_target_trial_window_unreadable",
        "research_pipeline_trajectory_authority_refused",
        "research_pipeline_variance_estimator_unsupported",
        "scientific_plan_review_changes_required",
        "scientific_plan_review_policy_stale",
        "strict_evidence_gate_failed",
        "study_family_reading_disagreement",
        "system_validation_authority_invalid",
        "system_validation_document_missing",
        "system_validation_document_unsafe",
        "system_validation_ledger_digest_mismatch",
        "system_validation_ledger_invalid",
        "system_validation_pdf_content_binding_mismatch",
        "system_validation_pdf_parse_invalid",
        "system_validation_pdf_signature_invalid",
        "system_validation_projection_privacy_blocked",
        "system_validation_receipt_digest_mismatch",
        "system_validation_receipt_invalid",
        "system_validation_report_invalid",
    }
)

_RETRY_CODES = frozenset(
    {
        # The run spent its model budget (failed transport attempts count): a
        # fresh run gets a fresh budget once the connection is stable.
        "research_pipeline_provider_budget_exhausted",
        "research_pipeline_planner_provider_unavailable",
        "research_pipeline_provider_timeout",
        "research_pipeline_provider_recovery_probe_failed",
        "research_pipeline_execution_runtime_unavailable",
        "research_pipeline_runner_image_mismatch",
        "research_pipeline_review_recovery_capacity_full",
        "research_pipeline_review_resume_in_progress",
        "research_pipeline_cancelled",
        "agent_run_cancelled_before_artifacts",
    }
)
_REPAIR_CODES = frozenset(
    {
        "progressive_capability_gap",
        "design_element_unsupported",
        "research_pipeline_progressive_compile_failed",
        "research_pipeline_scientific_authority_refused",
        "research_pipeline_trajectory_authority_refused",
        "agent_plan_revision_nonconvergent",
        "research_agent_pipeline_failed_closed",
        # A revision added an analysis neither the question nor a review
        # blocker asked for; the pre-revision plan is kept.
        "revision_added_unrequested_analysis",
    }
)
_DATA_CODES = frozenset(
    {
        "export_time_window_unbounded",
        "data_foundation_blocked",
        "concept_unavailable",
        "source_unavailable",
        "missing_data",
        "missing_required_concepts",
        "required_concepts_unavailable",
        "research_pipeline_required_concept_structurally_unavailable",
        "research_pipeline_required_concept_without_values",
        "research_pipeline_cohort_concept_unavailable",
        "research_pipeline_target_outcome_materialization_unavailable",
        "research_pipeline_plan_outcome_materialization_unavailable",
        "research_pipeline_plan_outcome_catalog_unavailable",
        "plan_change_required_concepts_unavailable",
    }
)
_SCOPE_CODES = frozenset(
    {
        "study_family_database_out_of_scope",
        "target_trial_database_out_of_scope",
    }
)
STOP_POLICIES = {
    PROVIDER_NOT_CLEARED: "provider_settings",
    **dict.fromkeys(_EDIT_CODES, "edit_plan"),
    **dict.fromkeys(_RETRY_CODES, "retry"),
    **dict.fromkeys(_REPAIR_CODES, "repair"),
    **dict.fromkeys(_DATA_CODES, "data"),
    **dict.fromkeys(_SCOPE_CODES, "scope"),
}


def _code(value: Any) -> str | None:
    return value if isinstance(value, str) and _CODE.fullmatch(value) else None


def _codes(value: Any, limit: int = 16) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(code for item in value[:limit] if (code := _code(item))))


def _step_id(value: Any) -> str | None:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[a-z0-9][a-z0-9_]{0,79}", value)
        else None
    )


def safe_stop_coordinates(payload: Any) -> dict[str, Any]:
    """Keep bounded coordinates, including a host-checked stated variable name.

    Exception messages and model candidate values stay out. A stated variable
    is the researcher's own wording from a checked question receipt.
    """

    if not isinstance(payload, Mapping):
        return {}
    raw_fields = payload.get("mismatch_fields")
    mismatch_fields = [field for field in BaselineAssociationAuthorityError.mismatch_field_names
                       if isinstance(raw_fields, list) and field in raw_fields]
    result: dict[str, Any] = {
        key: code
        for key in ("element", "concept", "database", "family")
        if (code := _code(payload.get(key))) is not None
    }
    if mismatch_fields:
        result["mismatch_fields"] = mismatch_fields
    if payload.get("runner_kind") == "not_started":
        result["runner_kind"] = "not_started"
    variable = payload.get("stated_variable")
    if (isinstance(variable, str) and 0 < len(variable.strip()) <= 160
            and not any(ord(char) < 32 or ord(char) == 127 for char in variable)):
        result["stated_variable"] = variable.strip()
    if (step := _step_id(payload.get("step_id"))) is not None:
        result["step_id"] = step
    if type(payload.get("question_required")) is bool:
        result["question_required"] = payload["question_required"]
    retries = payload.get("transport_retries")
    if type(retries) is int and 0 <= retries <= 32:
        result["transport_retries"] = retries
    for key in ("missing_concepts", "available_concepts"):
        if key in payload:
            result[key] = list(
                _codes(payload[key], 1024 if key == "available_concepts" else 16)
            )
    return result


def provider_timeout_coordinates(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count recorded transport retries without inferring unrecorded actions."""
    counts = [row.get("transport_attempts") for row in attempts]
    if not counts or any(type(count) is not int or not 1 <= count <= 33 for count in counts):
        return {}
    return safe_stop_coordinates({"transport_retries": sum(count - 1 for count in counts)})


def source_stop_coordinates(payload: Any) -> dict[str, Any]:
    """Normalize the data owners' existing typed error fields, without prose."""

    result = safe_stop_coordinates(payload)
    if not isinstance(payload, Mapping) or "missing_concepts" in result:
        return result
    for field in ("required_concepts", "missing_source_concepts", "concepts"):
        if missing := _codes(payload.get(field)):
            return {**result, "missing_concepts": list(missing)}
    for field in ("concept_id", "source_concept"):
        if concept := _code(payload.get(field)):
            return {**result, "missing_concepts": [concept]}
    return result


def _action(
    kind: ActionKind,
    params: PlanEditParams | RetryParams | ScopeParams | ProviderSettingsParams | DataExtractionParams,
) -> StopNextAction:
    content = json.dumps(
        [kind, params.model_dump(mode="json", exclude_none=True)], sort_keys=True
    )
    return StopNextAction(
        action_id=f"{kind}_{hashlib.sha256(content.encode()).hexdigest()[:16]}",
        kind=kind,
        label_key=f"stop_next_actions.{kind}",
        params=params,
    )


def _available_alternative(
    concept: str, database: str | None, detail: Mapping[str, Any]
) -> str | None:
    policy = concept_selection_policy(concept)
    alternative = _code(policy.canonical_alternative) if policy is not None else None
    if alternative is None or alternative == concept or database is None:
        return None
    if alternative in _codes(detail.get("missing_concepts")):
        return None
    available = detail.get("available_concepts")
    if available is not None and alternative not in _codes(available, 1024):
        return None
    availability = explain_concept_availability(concept=alternative, database=database)
    return alternative if availability.available else None


def next_actions_for_stop(
    reason_code: str,
    detail: Mapping[str, Any] | None = None,
    *,
    database: str | None = None,
) -> list[StopNextAction]:
    """Derive choices from typed coordinates, never from a stop's message.

    Missing optionality is not permission to remove a step. Even a proven
    optional step can only be removed by an explicit edit and fresh approval.
    Unknown historical codes retain a conservative plan-edit route; new owner
    codes still fail the registry's exhaustive governance test.
    """

    detail = detail if isinstance(detail, Mapping) else {}
    database = _code(detail.get("database")) or _code(database)
    reason = _code(reason_code)
    if reason is None:
        return []
    cause = _code(detail.get("cause_code"))
    inner = _code(detail.get("reason_code"))
    if "export_time_window_unbounded" in {
        reason, cause, inner, _code(detail.get("intake_error_code")),
    }:
        return [_action("reexport_data", DataExtractionParams())]
    policy = STOP_POLICIES.get(reason, "edit_plan")
    if policy == "provider_settings":
        return [_action("open_provider_settings", ProviderSettingsParams())]
    if reason not in _RETRY_CODES and inner in _SCOPE_CODES:
        policy = "scope"
    if reason not in _RETRY_CODES and (inner in _DATA_CODES or cause in _DATA_CODES):
        policy = "data"
    concepts = _codes(detail.get("missing_concepts"))
    element = _code(detail.get("element"))
    step = _step_id(detail.get("step_id"))
    concept = _code(detail.get("concept")) or next(iter(concepts), None)
    if policy == "data" and not concepts and concept is not None:
        concepts = (concept,)
    coordinates = {"element": element, "step_id": step, "concept": concept}
    edit = _action("edit_plan", PlanEditParams(**coordinates))
    if policy == "retry":
        return [_action("retry", RetryParams())]
    if policy == "scope":
        return [
            _action(
                "rephrase_in_scope",
                ScopeParams(
                    **v1_scope_description(
                        reason_code=inner or reason,
                        detail=detail,
                        database=database,
                    )
                ),
            )
        ]
    if policy == "data" and concepts:
        actions = []
        for missing in concepts:
            alternative = _available_alternative(missing, database, detail)
            actions.append(
                _action(
                    "use_alternative_concept" if alternative else "drop_variable",
                    PlanEditParams(
                        concept=missing,
                        alternative_concept=alternative,
                        database=database if alternative else None,
                    ),
                )
            )
        return [*actions, edit]
    if (
        policy == "repair"
        and (element is not None or step is not None)
        and detail.get("question_required") is False
        and (cause != "design_element_unsupported" or concept is not None)
    ):
        return [_action("rerun_without_element", PlanEditParams(**coordinates)), edit]
    return [edit]


def project_stop_next_actions(
    gate: Mapping[str, Any],
    *,
    pending_reason_codes: Sequence[str] = (),
    database: str | None = None,
    execution_retry_available: bool = False,
) -> list[dict[str, Any]]:
    """The same additive field for status, jobs and conversation run history."""

    if gate.get("status") != "blocked":
        return []
    recorded = gate.get("next_actions")
    if isinstance(recorded, (list, tuple)) and 0 < len(recorded) <= 17:
        try:
            actions = [StopNextAction.model_validate(action) for action in recorded]
            # Persist the source coordinate with an alternative so later
            # readers can check the same concept owners again. Always derive
            # choices afresh from the stop, including required-element guards.
            databases = {
                action.params.database
                for action in actions
                if isinstance(action.params, PlanEditParams)
                and action.params.database
                and action == _action(action.kind, action.params)
            }
            if database is None and len(databases) == 1:
                database = next(iter(databases))
        except (ValidationError, TypeError, ValueError):
            pass
    reason = str(gate.get("reason") or "")
    # Pending scientific revisions use the gate of an ordinary approval pause.
    # Only an actual blocker offers plan edits; normal approval keeps its UI.
    if reason == "human_plan_review_required":
        reason = next(
            (
                code
                for code in pending_reason_codes
                if code != "operator_plan_approval_required"
            ),
            "",
        )
    actions = next_actions_for_stop(reason, gate.get("detail"), database=database)
    # Only the workflow's existing retry assessment can add this offer; a
    # persisted stop record or model suggestion cannot authorize a retry.
    if execution_retry_available and reason in {
        "research_pipeline_execution_failed", "research_agent_pipeline_failed_closed",
    }:
        actions = [_action("retry", RetryParams()), *actions]
    return [
        action.model_dump(mode="json", exclude_none=True)
        for action in actions
    ]


__all__ = [
    "STOP_POLICIES",
    "StopNextAction",
    "next_actions_for_stop",
    "project_stop_next_actions",
]
