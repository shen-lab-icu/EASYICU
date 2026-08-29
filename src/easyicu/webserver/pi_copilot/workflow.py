"""Typed project-level research workflow projection for Pi Copilot.

This module owns only orchestration state.  Idea Mining, extraction, Research
Agent execution, evidence gates, and manuscript artefacts keep their existing
owners; the Copilot shell receives an immutable, path-free projection of those
owners' receipts.
"""

from __future__ import annotations

import hashlib
import re

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from easyicu.webserver import study_contexts as study_context_owner

WorkflowStatus = Literal[
    "blocked",
    "ready",
    "running",
    "complete",
    "optional",
    "review_required",
]


class ResearchWorkflowStage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal[
        "question",
        "idea",
        "setup",
        "extraction",
        "plan",
        "analysis",
        "interpretation",
        "manuscript",
    ]
    label: str
    status: WorkflowStatus
    owner: str
    reason_code: str


class StudySetupReceipt(BaseModel):
    """Path-free identity and configuration receipt for Copilot review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.pi-study-setup-receipt/1"] = (
        "easyicu.pi-study-setup-receipt/1"
    )
    study_context_id: str
    revision: int = Field(ge=0)
    configured_fields: List[str] = Field(default_factory=list, max_length=24)
    configuration: Mapping[str, Any]


class ResearchWorkflowSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.pi-research-workflow/1"] = (
        "easyicu.pi-research-workflow/1"
    )
    current_stage: str
    next_action_code: str
    missing_setup_fields: List[str] = Field(default_factory=list, max_length=16)
    planning_prerequisites_missing: List[str] = Field(
        default_factory=list, max_length=8
    )
    stages: List[ResearchWorkflowStage] = Field(min_length=8, max_length=8)
    completed_required_stages: int = Field(ge=0, le=7)
    required_stage_count: Literal[7] = 7
    scientific_authority: Literal["EasyICU"] = "EasyICU"
    pi_role: Literal["conversation_and_orchestration"] = (
        "conversation_and_orchestration"
    )
    study_setup_receipt: StudySetupReceipt
    plan_review_summary: Optional[Mapping[str, Any]] = None
    plan_execution_ready: bool = False


def _has_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def _owner_locked_clinical_definition(contract: Mapping[str, Any]) -> bool:
    """Return whether EasyICU already owns one executable standard profile."""

    return bool(
        contract.get("definition_locked") is True
        and str(contract.get("runtime_profile") or "").strip()
        and str(contract.get("implementation_profile") or "").strip()
        and _has_mapping(contract.get("locked_core"))
    )


def _missing_clinical_definition_confirmation(study: Mapping[str, Any]) -> str:
    """Return one unresolved phenotype choice that needs human scientific input."""

    cohort = study.get("cohort")
    cohort = cohort if isinstance(cohort, Mapping) else {}
    confirmations = study.get("confirmations")
    confirmations = confirmations if isinstance(confirmations, Mapping) else {}
    user_text = " ".join(
        str(study.get(field) or "")
        for field in ("question", "purpose", "primary_exposure")
    ).lower()
    normalized_text = re.sub(r"[^a-z0-9]+", " ", user_text)
    for field, contract in cohort.items():
        clean_field = str(field or "").strip().lower()
        if not clean_field.endswith("_definition") or not _has_mapping(contract):
            continue
        if _owner_locked_clinical_definition(contract):
            continue
        phenotype = clean_field.removesuffix("_definition").strip("_")
        normalized_phenotype = phenotype.replace("_", " ")
        confirmation_key = f"clinical_definition_{phenotype}"
        if (
            normalized_phenotype
            and normalized_phenotype in normalized_text
            and confirmations.get(confirmation_key) is not True
        ):
            return f"confirmations.{confirmation_key}"
    return ""


def _requires_primary_exposure(study: Mapping[str, Any]) -> bool:
    """Return whether the stated research intent includes an exposure relation."""

    if str(study.get("primary_exposure") or "").strip():
        return True
    intent = " ".join(
        str(study.get(field) or "") for field in ("question", "purpose")
    ).casefold()
    return bool(
        re.search(
            r"(?:关系|关联|相关|效应|影响|预测|危险因素|"
            r"association|associated|relationship|effect|predict|risk factor)",
            intent,
        )
    )


def active_export_matches_study(
    study: Optional[Mapping[str, Any]], active_export_path: Any
) -> bool:
    """Return whether the source owner's active export belongs to this study."""

    study_row: Mapping[str, Any] = study or {}
    source = study_row.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    expected = str(source.get("path") or "").strip()
    active = str(active_export_path or "").strip()
    return bool(expected and active and expected == active)


def registered_export_matches_study(
    study: Optional[Mapping[str, Any]], registry: Optional[Mapping[str, Any]]
) -> bool:
    """Return whether the project's exact bound export is still registered."""

    study_row: Mapping[str, Any] = study or {}
    confirmations = study_row.get("confirmations")
    confirmations = confirmations if isinstance(confirmations, Mapping) else {}
    if confirmations.get("extraction_completed") is not True:
        return False
    source = study_row.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    expected = str(source.get("path") or "").strip()
    if not expected:
        return False
    registry_row: Mapping[str, Any] = registry or {}
    return any(
        isinstance(row, Mapping)
        and bool(row.get("ok"))
        and str(row.get("path") or "").strip() == expected
        for row in (registry_row.get("sources") or [])
    )


# Plan generation needs only what the user owns: a bound question and a data
# source EasyICU has identified.  Every other setup field is a scientific design
# choice the Research Agent Planner proposes in its reviewable plan, so gating
# plan generation on them turned Copilot into a slot-by-slot questionnaire and
# forced the cohort to be frozen before the plan that should define it.  They
# still gate extraction and analysis, which execute the reviewed plan.
_PLANNING_PREREQUISITE_FIELDS = frozenset({"question", "data_source"})

# These are Planner proposal fields, not pre-plan setup questions.  The
# researcher reviews the complete candidate plan; they should not have to
# invent an endpoint contract or a sensitivity implementation before seeing
# that plan.  Full execution still requires the reviewed proposal to be
# promoted into typed StudyContext authority.
_PLANNER_PROPOSAL_FINDING_CODES = frozenset(
    {
        "OUTCOME_DEFINITION_UNRESOLVED",
        "ROBUSTNESS_AUTHORITY_NOT_PRESPECIFIED",
        "ADJUSTMENT_SET_NOT_USER_CONFIRMED",
        "POST_BASELINE_EXPOSURE_TIMING_NOT_CLOSED",
    }
)


def _identified_data_source(study: Mapping[str, Any]) -> bool:
    """Whether the study is bound to a data source EasyICU has identified."""

    source = study.get("data_source")
    if not isinstance(source, Mapping):
        return False
    return bool(str(source.get("database") or "").strip())


def _planning_prerequisites_missing(
    missing_setup_fields: Sequence[str],
) -> List[str]:
    """The subset of missing setup fields that genuinely blocks planning."""

    return [
        field
        for field in missing_setup_fields
        if str(field).split(".", 1)[0] in _PLANNING_PREREQUISITE_FIELDS
    ]


def _setup_missing(
    study: Mapping[str, Any], *, active_export_present: bool
) -> List[str]:
    raw_execution = study.get("execution_concepts")
    execution = raw_execution if isinstance(raw_execution, Mapping) else {}
    raw_window = study.get("time_window")
    time_window = raw_window if isinstance(raw_window, Mapping) else {}
    raw_confirmations = study.get("confirmations")
    confirmations = (
        raw_confirmations if isinstance(raw_confirmations, Mapping) else {}
    )
    human_outcome = bool(str(study.get("outcome") or "").strip())
    human_exposure = bool(str(study.get("primary_exposure") or "").strip())
    exposure_required = _requires_primary_exposure(study)
    executable_exposure = bool(
        str(execution.get("primary_exposure") or "").strip()
    )
    executable_outcome = bool(str(execution.get("outcome") or "").strip())
    analysis_design_present = _has_mapping(study.get("analysis_design"))
    dependence_finding = study_context_owner.analysis_dependence_finding(dict(study))
    window_finding = study_context_owner.materialization_window_finding(dict(study))
    clinical_definition_confirmation = _missing_clinical_definition_confirmation(
        study
    )
    feature_window_confirmed = confirmations.get("feature_time_window") is True
    window_hours = time_window.get("hours")
    if window_hours is None:
        window_hours = time_window.get("observation_hours")
    missing: List[str] = []
    checks = (
        ("question", bool(str(study.get("question") or "").strip())),
        (
            "data_source",
            active_export_present or _has_mapping(study.get("data_source")),
        ),
        ("cohort", _has_mapping(study.get("cohort"))),
        ("outcome", human_outcome),
        *(((("primary_exposure", human_exposure),)) if exposure_required else ()),
        (
            "analysis_goal",
            bool(str(study.get("analysis_goal") or "").strip()),
        ),
        *(
            (("time_window", False),)
            if not time_window
            else (
                ("time_window.hours", window_hours is not None),
                (
                    "time_window.anchor",
                    bool(str(time_window.get("anchor") or "").strip()),
                ),
                *(
                    (("time_window.anchor_supported", False),)
                    if window_finding is not None
                    else ()
                ),
                *(
                    (("confirmations.feature_time_window", False),)
                    if (
                        window_hours is not None
                        and bool(str(time_window.get("anchor") or "").strip())
                        and window_finding is None
                        and not feature_window_confirmed
                    )
                    else ()
                ),
            )
        ),
        *(
            (
                (
                    "covariates",
                    str(study.get("covariate_selection") or "").strip()
                    in {"exact", "planner_selectable"},
                ),
            )
            if exposure_required
            else ()
        ),
        (
            "export_format",
            bool(str(study.get("export_format") or "").strip())
            and confirmations.get("export_format") is True,
        ),
        # Finish user-owned scientific choices before EasyICU-owned
        # implementation readiness. Missing catalog ids must not create a
        # generic "continue" gate between two key user decisions.
        ("modules", bool(study.get("modules"))),
        *(
            (("execution_concepts.outcome", executable_outcome),)
            if human_outcome
            else ()
        ),
        *(
            (
                ("execution_concepts.primary_exposure", executable_exposure),
                ("analysis_design", analysis_design_present),
                *(
                    (("analysis_design.dependence", False),)
                    if analysis_design_present and dependence_finding is not None
                    else ()
                ),
            )
            if human_exposure or executable_exposure
            else ()
        ),
        *(
            ((clinical_definition_confirmation, False),)
            if clinical_definition_confirmation
            else ()
        ),
    )
    for name, present in checks:
        if not present:
            missing.append(name)
    return missing


def _safe_study_setup_receipt(study: Mapping[str, Any]) -> StudySetupReceipt:
    """Project only bounded setup fields; never return a local filesystem path."""

    raw_source = study.get("data_source")
    source = raw_source if isinstance(raw_source, Mapping) else {}
    source_path = str(source.get("path") or "").strip()
    safe_source: Dict[str, Any] = {
        key: str(source.get(key) or "").strip()
        for key in ("label", "database")
        if str(source.get(key) or "").strip()
    }
    if source_path:
        safe_source["path_hash"] = hashlib.sha256(
            source_path.encode("utf-8")
        ).hexdigest()[:16]

    raw_crossdb = study.get("crossdb_selection")
    crossdb = dict(raw_crossdb) if isinstance(raw_crossdb, Mapping) else {}
    raw_cohort = study.get("cohort")
    cohort = dict(raw_cohort) if isinstance(raw_cohort, Mapping) else {}
    raw_window = study.get("time_window")
    time_window = dict(raw_window) if isinstance(raw_window, Mapping) else {}
    raw_execution = study.get("execution_concepts")
    execution_concepts = (
        dict(raw_execution) if isinstance(raw_execution, Mapping) else {}
    )
    raw_analysis_design = study.get("analysis_design")
    analysis_design = (
        dict(raw_analysis_design)
        if isinstance(raw_analysis_design, Mapping)
        else {}
    )
    raw_rationales = study.get("covariate_rationales")
    covariate_rationales = (
        dict(raw_rationales) if isinstance(raw_rationales, Mapping) else {}
    )
    raw_temporal_roles = study.get("covariate_temporal_roles")
    covariate_temporal_roles = (
        dict(raw_temporal_roles)
        if isinstance(raw_temporal_roles, Mapping)
        else {}
    )
    raw_operationalizations = study.get("covariate_operationalizations")
    covariate_operationalizations = (
        dict(raw_operationalizations)
        if isinstance(raw_operationalizations, Mapping)
        else {}
    )
    raw_confirmations = study.get("confirmations")
    confirmations = (
        dict(raw_confirmations)
        if isinstance(raw_confirmations, Mapping)
        else {}
    )
    configuration: Dict[str, Any] = {
        "question": str(study.get("question") or "").strip(),
        "purpose": str(study.get("purpose") or "").strip(),
        "data_source": safe_source,
        "crossdb_selection": crossdb,
        "cohort": cohort,
        "modules": [
            str(value)
            for value in (study.get("modules") or [])
            if str(value).strip()
        ],
        "outcome": str(study.get("outcome") or "").strip(),
        "primary_exposure": str(study.get("primary_exposure") or "").strip(),
        "covariates": [
            str(value)
            for value in (study.get("covariates") or [])
            if str(value).strip()
        ],
        "covariate_selection": str(
            study.get("covariate_selection") or "planner_selectable"
        ).strip(),
        "covariate_rationales": covariate_rationales,
        "covariate_temporal_roles": covariate_temporal_roles,
        "covariate_operationalizations": covariate_operationalizations,
        "execution_concepts": execution_concepts,
        "analysis_design": analysis_design,
        "sensitivity_specs": [
            dict(value)
            for value in (study.get("sensitivity_specs") or [])
            if isinstance(value, Mapping)
        ],
        "time_window": time_window,
        "comparator": str(study.get("comparator") or "").strip(),
        "export_format": str(study.get("export_format") or "").strip(),
        "analysis_goal": str(study.get("analysis_goal") or "").strip(),
        "confirmations": confirmations,
    }
    configured_fields = [
        key for key, value in configuration.items() if bool(value)
    ]
    return StudySetupReceipt(
        study_context_id=str(study.get("id") or ""),
        revision=int(study.get("revision") or 0),
        configured_fields=configured_fields,
        configuration=configuration,
    )


def build_research_workflow_snapshot(
    *,
    study: Optional[Mapping[str, Any]],
    active_export_present: bool,
    active_job: Optional[Mapping[str, Any]],
    latest_run: Optional[Mapping[str, Any]],
    plan_review_authority: Optional[Mapping[str, Any]] = None,
) -> ResearchWorkflowSnapshot:
    """Compile owner receipts into one deterministic Copilot workflow state."""

    study_row: Mapping[str, Any] = study or {}
    job_row: Mapping[str, Any] = active_job or {}
    run_row: Mapping[str, Any] = latest_run or {}
    question_ready = bool(str(study_row.get("question") or "").strip())
    idea_handoff = study_row.get("idea_handoff")
    idea_handoff = idea_handoff if isinstance(idea_handoff, Mapping) else {}
    idea_accepted = bool(
        idea_handoff.get("status") == "accepted"
        and str(idea_handoff.get("run_id") or "").strip()
        and str(idea_handoff.get("idea_id") or "").strip()
        and len(str(idea_handoff.get("canonical_handoff_sha256") or "")) == 64
    )
    idea_recommendation = str(idea_handoff.get("go_no_go") or "").strip()
    idea_blocks_execution = bool(
        idea_accepted and idea_recommendation != "recommend"
    )
    missing = _setup_missing(
        study_row,
        active_export_present=bool(active_export_present),
    )
    setup_ready = not missing
    job_kind = str(job_row.get("kind") or "")
    job_status = str(job_row.get("status") or "")
    extraction_running = job_kind == "extract" and job_status == "running"
    analysis_running = job_kind == "agent-run" and job_status == "running"
    artifact_names = {
        str(item) for item in (run_row.get("artifact_names") or []) if item
    }
    run_type = str(run_row.get("run_type") or "")
    run_engine = str(run_row.get("engine") or "")
    has_plan = "agent_plan.json" in artifact_names
    has_evidence = "evidence_ledger.json" in artifact_names
    has_outputs = bool(
        artifact_names
        & {
            "result_tables.json",
            "table1_summary.json",
            "missingness_audit.json",
            "roc_curve.json",
            "calibration_curve.json",
            "figure_gallery.json",
        }
    )
    has_manuscript = "manuscript_draft.json" in artifact_names
    full_run = run_type == "full"
    pipeline_run = run_engine == "easyicu.research_agent.pipeline"
    pipeline_receipt = "source_run_manifest.json" in artifact_names
    gate_status = str(run_row.get("gate_status") or "")
    run_blocked = gate_status == "blocked"
    raw_gate_checks = run_row.get("gate_checks")
    gate_checks = (
        dict(raw_gate_checks) if isinstance(raw_gate_checks, Mapping) else {}
    )
    executed_analysis_validated = bool(
        gate_checks.get("execution_complete") is True
        and gate_checks.get("analysis_validated") is True
        and gate_checks.get("numeric_verified") is True
    )
    preflight_complete = bool(
        run_type == "preflight"
        and has_evidence
        and gate_status == "analysis_only"
        and str(run_row.get("readiness_status") or "")
        == "awaiting_human_signoff"
    )
    # A completed local preflight is an owner-issued receipt that the exact
    # bound registered export was resolved and reviewed.  Treat it as stronger
    # evidence than a legacy StudyContext extraction flag so prepared-export
    # reuse cannot fall backward to extraction after preflight succeeds.
    prepared_export_receipted = bool(active_export_present or preflight_complete)
    # Planning starts from the user's question plus a data source EasyICU has
    # identified -- a prepared export or a bound, identified local database.
    # The Planner proposes the unresolved design choices in its reviewable
    # plan; Copilot must not fabricate a shadow plan just to fill setup slots,
    # and the cohort must not be frozen before the plan that defines it.
    planning_data_ready = bool(
        prepared_export_receipted or _identified_data_source(study_row)
    )
    planning_prerequisites_missing = _planning_prerequisites_missing(missing)
    plan_generation_ready = bool(
        question_ready
        and planning_data_ready
        and not planning_prerequisites_missing
        and not has_plan
    )
    review_authority = (
        plan_review_authority
        if isinstance(plan_review_authority, Mapping)
        else {}
    )
    authority_requests = review_authority.get("requests")
    authority_reason_codes = {
        str(item.get("reason_code") or "").strip()
        for item in (
            authority_requests if isinstance(authority_requests, list) else []
        )
        if isinstance(item, Mapping) and str(item.get("reason_code") or "").strip()
    }
    pending_review_reason_codes = authority_reason_codes or {
        str(item).strip()
        for item in (run_row.get("pending_review_reason_codes") or [])
        if str(item).strip()
    }
    plan_review_codes = {
        "operator_plan_approval_required",
        "plan_scientific_changes_required",
    }
    active_plan_review_codes = sorted(
        pending_review_reason_codes & plan_review_codes
    )
    plan_review_declared = bool(
        has_plan
        and str(run_row.get("run_status") or "") == "human_review_pending"
        and active_plan_review_codes
    )
    raw_scientific_review = review_authority.get("scientific_plan_review")
    raw_scientific_review = (
        raw_scientific_review
        if isinstance(raw_scientific_review, Mapping)
        else {}
    )
    dimension_scores = raw_scientific_review.get("dimension_scores")
    dimension_scores = (
        dimension_scores if isinstance(dimension_scores, Mapping) else {}
    )
    review_findings = raw_scientific_review.get("findings")
    review_findings = review_findings if isinstance(review_findings, list) else []
    raw_facts = raw_scientific_review.get("facts")
    raw_facts = raw_facts if isinstance(raw_facts, Mapping) else {}
    raw_remediation_buckets = raw_facts.get("remediation_buckets")
    raw_remediation_buckets = (
        raw_remediation_buckets
        if isinstance(raw_remediation_buckets, Mapping)
        else {}
    )
    def projected_remediation_codes(route: str) -> List[str]:
        values = raw_remediation_buckets.get(route)
        rows = [
            str(code)[:120]
            for code in (values if isinstance(values, list) else [])[:40]
            if str(code).strip()
        ]
        if route == "study_authority_change":
            return [
                code for code in rows if code not in _PLANNER_PROPOSAL_FINDING_CODES
            ]
        if route == "agent_plan_revision":
            proposal_codes = [
                str(item.get("code") or "")[:120]
                for item in review_findings[:40]
                if isinstance(item, Mapping)
                and str(item.get("code") or "") in _PLANNER_PROPOSAL_FINDING_CODES
            ]
            return list(dict.fromkeys([*rows, *proposal_codes]))[:40]
        return rows

    plan_review_summary = (
        {
            "status": str(raw_scientific_review.get("status") or "")[:40],
            "score": raw_scientific_review.get("score"),
            "top_journal_candidate": bool(
                raw_scientific_review.get("top_journal_candidate")
            ),
            "review_scope": str(
                raw_scientific_review.get("review_scope") or "pre_execution_plan"
            )[:80],
            "rendered_outputs_assessed": bool(
                raw_scientific_review.get("rendered_outputs_assessed")
            ),
            "dimension_scores": {
                str(key)[:80]: int(value)
                for key, value in list(dimension_scores.items())[:12]
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            },
            "finding_codes": [
                str(item.get("code") or "")[:120]
                for item in review_findings[:40]
                if isinstance(item, Mapping) and str(item.get("code") or "").strip()
            ],
            "authorization_questions": [
                {
                    "code": str(item.get("code") or "")[:120],
                    "question": str(item.get("authorization_question") or "")[:1_200],
                }
                for item in review_findings[:40]
                if isinstance(item, Mapping)
                and bool(item.get("requires_user_authorization"))
                and str(item.get("code") or "")
                not in _PLANNER_PROPOSAL_FINDING_CODES
                and str(item.get("authorization_question") or "").strip()
            ],
            "remediation_buckets": {
                route: projected_remediation_codes(route)
                for route in (
                    "agent_plan_revision",
                    "study_authority_change",
                    "external_evidence",
                    "independent_review",
                )
            },
        }
        if raw_scientific_review
        else None
    )
    current_scientific_digest = (
        study_context_owner.scientific_configuration_sha256(study_row)
    )
    planned_scientific_digest = str(
        review_authority.get("scientific_configuration_sha256")
        or run_row.get("scientific_configuration_sha256")
        or ""
    ).strip()
    review_authority_available = bool(
        review_authority
        and str(review_authority.get("run_id") or "").strip()
        == str(run_row.get("run_id") or "").strip()
        and bool(review_authority.get("resumable_here"))
    )
    plan_configuration_matches = bool(
        len(planned_scientific_digest) == 64
        and planned_scientific_digest == current_scientific_digest
    )
    plan_review_pending = bool(
        plan_review_declared
        and review_authority_available
        and plan_configuration_matches
    )
    plan_execution_ready = bool(
        plan_review_pending
        and str(review_authority.get("budget_mode") or "full_reviewed")
        != "planner_canary"
    )
    live_plan_reason = (
        "plan_scientific_changes_required"
        if "plan_scientific_changes_required" in active_plan_review_codes
        else "operator_plan_approval_required"
        if plan_execution_ready
        else "plan_execution_upgrade_required"
    )
    plan_review_reason_code = (
        live_plan_reason
        if plan_review_pending
        else "plan_configuration_superseded"
        if plan_review_declared
        and planned_scientific_digest
        and not plan_configuration_matches
        else "plan_review_not_resumable"
        if plan_review_declared
        else ""
    )
    # A live, digest-matching review is an approval gate. A stale or
    # non-resumable plan remains historical evidence, but the next governed
    # action is a fresh planning run rather than approval or in-place editing.
    plan_attention_required = bool(plan_review_pending)
    plan_regeneration_required = bool(
        plan_review_declared and not plan_review_pending and not analysis_running
    )
    analysis_complete = bool(
        full_run
        and pipeline_run
        and pipeline_receipt
        and has_plan
        and has_evidence
        and has_outputs
        and (gate_status == "analysis_only" or executed_analysis_validated)
    )
    # A completed, validated full run is a stronger downstream receipt than
    # legacy blank setup slots. The approved Agent plan owns the exact study
    # design used for execution, so an older StudyContext must not pull the
    # visible workflow backward from result interpretation to setup.
    setup_receipted = bool(setup_ready or analysis_complete)
    pipeline_attempt_blocked = bool(
        full_run
        and pipeline_run
        and pipeline_receipt
        and run_blocked
        and not analysis_complete
    )
    # A terminal fail-closed run is historical evidence, not a completed
    # analysis stage.  The ordinary Copilot journey must return to a fresh Plan
    # confirmation instead of advancing to interpretation (where the user can
    # only be told that the old run failed).  The failed run stays immutable;
    # a newly authorized provider turn receives a new run id and Plan review.
    failed_pipeline_regeneration_required = bool(
        pipeline_attempt_blocked
        and not analysis_running
        and not plan_review_declared
    )
    failed_execution_retry_available = bool(
        failed_pipeline_regeneration_required
        and has_plan
        and has_evidence
        and str(run_row.get("gate_reason") or "")
        in {
            "research_agent_pipeline_failed_closed",
            "research_pipeline_execution_failed",
        }
        and len(planned_scientific_digest) == 64
        and planned_scientific_digest == current_scientific_digest
    )
    planner_checkpoint_resume_available = bool(
        failed_pipeline_regeneration_required
        and str(run_row.get("gate_reason") or "")
        in {
            "research_pipeline_planner_efficiency_budget_exhausted",
        }
        and bool(run_row.get("development_planner_checkpoint_available"))
        and len(planned_scientific_digest) == 64
        and planned_scientific_digest == current_scientific_digest
    )
    plan_regeneration_required = bool(
        plan_regeneration_required or failed_pipeline_regeneration_required
    )
    plan_regeneration_reason_code = (
        "planner_checkpoint_resume_available"
        if planner_checkpoint_resume_available
        else "failed_pipeline_execution_retry_available"
        if failed_execution_retry_available
        else "failed_pipeline_requires_fresh_plan"
        if failed_pipeline_regeneration_required
        else plan_review_reason_code
    )
    legacy_full_scaffold = bool(full_run and not pipeline_run and has_plan)

    stages = [
        ResearchWorkflowStage(
            id="question",
            label="Scientific question",
            status="complete" if question_ready else "ready",
            owner="easyicu.webserver.study_contexts",
            reason_code=("question_bound" if question_ready else "question_required"),
        ),
        ResearchWorkflowStage(
            id="idea",
            label="Idea mining",
            status=(
                "review_required"
                if idea_blocks_execution
                else "complete"
                if idea_accepted
                else "optional"
                if question_ready
                else "blocked"
            ),
            owner="easyicu.webserver.ideas.mining",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else "idea_handoff_accepted"
                if idea_accepted
                else "idea_mining_available"
                if question_ready
                else "question_required"
            ),
        ),
        ResearchWorkflowStage(
            id="setup",
            label="Study setup",
            status="complete"
            if setup_receipted
            else ("ready" if question_ready else "blocked"),
            owner="easyicu.webserver.study_contexts",
            reason_code=(
                "approved_plan_setup_receipt"
                if analysis_complete and not setup_ready
                else "study_setup_complete"
                if setup_ready
                else "study_setup_incomplete"
            ),
        ),
        ResearchWorkflowStage(
            id="extraction",
            label="Feature extraction",
            status=(
                "complete"
                if prepared_export_receipted
                else "running"
                if extraction_running
                else "ready"
                if setup_ready
                else "blocked"
            ),
            owner="easyicu.webserver.routes.jobs",
            reason_code=(
                "active_export_ready"
                if prepared_export_receipted
                else "extraction_running"
                if extraction_running
                else "extraction_ready"
                if setup_ready
                else "study_setup_incomplete"
            ),
        ),
        ResearchWorkflowStage(
            id="plan",
            label="Analysis plan",
            status=(
                "blocked"
                if idea_blocks_execution
                else "review_required"
                if plan_attention_required
                else "running"
                if analysis_running
                else "ready"
                if plan_regeneration_required
                else "complete"
                if has_plan
                else "ready"
                if plan_generation_ready
                else "blocked"
            ),
            owner="easyicu.research_agent.planning",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else plan_review_reason_code
                if plan_attention_required
                else "analysis_running"
                if analysis_running
                else plan_regeneration_reason_code
                if plan_regeneration_required
                else "agent_plan_ready"
                if has_plan
                else "provider_ready_to_generate_plan"
                if plan_generation_ready
                else "active_export_or_setup_required"
            ),
        ),
        ResearchWorkflowStage(
            id="analysis",
            label="Analysis and validation",
            status=(
                "blocked"
                if idea_blocks_execution
                else "blocked"
                if plan_attention_required or plan_regeneration_required
                else "running"
                if analysis_running
                else "complete"
                if analysis_complete
                else "review_required"
                if pipeline_attempt_blocked
                else "ready"
                if prepared_export_receipted and setup_ready
                else "blocked"
            ),
            owner="easyicu.research_agent.pipeline",
            reason_code=(
                "idea_feasibility_refresh_required"
                if idea_blocks_execution
                else (
                    plan_review_reason_code
                    if plan_attention_required
                    else plan_regeneration_reason_code
                )
                if plan_attention_required or plan_regeneration_required
                else "analysis_running"
                if analysis_running
                else "validated_analysis_ready"
                if analysis_complete
                else "analysis_gate_blocked"
                if pipeline_attempt_blocked
                else "research_pipeline_required"
                if legacy_full_scaffold
                else "analysis_ready"
                if prepared_export_receipted and setup_ready
                else "active_export_or_setup_required"
            ),
        ),
        ResearchWorkflowStage(
            id="interpretation",
            label="Result interpretation",
            status=("review_required" if analysis_complete else "blocked"),
            owner="easyicu.research_agent.reporting",
            reason_code=(
                "evidence_bound_interpretation_ready"
                if analysis_complete
                else "validated_analysis_required"
            ),
        ),
        ResearchWorkflowStage(
            id="manuscript",
            label="Manuscript",
            status=(
                "review_required" if analysis_complete and has_manuscript else "blocked"
            ),
            owner="easyicu.research_agent.reporting",
            reason_code=(
                "manuscript_draft_ready_for_review"
                if analysis_complete and has_manuscript
                else "full_agent_manuscript_required"
            ),
        ),
    ]

    required = [row for row in stages if row.id != "idea"]
    # ``review_required`` is an outstanding human action, never a completed
    # stage.  Counting it as done made analysis-only runs appear as 7/7 even
    # though their interpretation and manuscript were still awaiting review.
    completed = sum(1 for row in required if row.status == "complete")
    if (
        plan_attention_required
        or plan_regeneration_required
        or (plan_generation_ready and not idea_blocks_execution)
    ):
        next_stage = next(row for row in required if row.id == "plan")
    else:
        next_stage = next(
            (
                row
                for row in required
                if row.status != "complete"
            ),
            stages[-1],
        )
    next_action = next_stage.reason_code
    if all(row.status == "complete" for row in required):
        next_action = "human_review_and_reporting"
    return ResearchWorkflowSnapshot(
        current_stage=next_stage.id,
        next_action_code=next_action,
        planning_prerequisites_missing=planning_prerequisites_missing,
        missing_setup_fields=missing,
        stages=stages,
        completed_required_stages=completed,
        study_setup_receipt=_safe_study_setup_receipt(study_row),
        plan_review_summary=plan_review_summary,
        plan_execution_ready=plan_execution_ready,
    )


__all__ = [
    "ResearchWorkflowSnapshot",
    "ResearchWorkflowStage",
    "StudySetupReceipt",
    "WorkflowStatus",
    "active_export_matches_study",
    "registered_export_matches_study",
    "build_research_workflow_snapshot",
]
