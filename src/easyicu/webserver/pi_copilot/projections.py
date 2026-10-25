"""PHI-safe projections for model-visible Pi Copilot tools."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from easyicu.research_agent.planning.approval_stops import PLAN_APPROVAL_STOPS

from ..research_input_progress import project_research_input_state
from ..run_step_trace import TRACE_ACTION_KEYS, TRACE_FACT_KEYS
from ..run_record import MAX_ARTIFACT_SIZE_BYTES, RunGate
from ..stop_next_actions import STOP_POLICIES, project_stop_next_actions
from ..target_trial_setup import project_target_trial_compile
from . import cohort_eligibility
from .contracts import PiCopilotError, plan_approval_allowed
from .user_visible_text import project_tool_receipt, project_user_turn_text, sanitize_user_visible_text

MAX_PROJECTION_BYTES = 32_768
MAX_TEXT_CHARS = 2_000
MAX_LIST_ITEMS = 80
# Leave room for gate state and digest-bound artifact references in a job.
MAX_JOB_PROGRESS_BYTES = 22_000


class StudySetupReceipt(BaseModel):
    """Path-free StudyContext configuration projected for Copilot review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.pi-study-setup-receipt/1"] = (
        "easyicu.pi-study-setup-receipt/1"
    )
    study_context_id: str
    revision: int = Field(ge=0)
    configured_fields: List[str] = Field(default_factory=list, max_length=24)
    configuration: Mapping[str, Any]

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\b(?:subject|stay|hadm|patient|entity)[ _-]?ids?\b", re.I),
    re.compile(
        r"\b(?:medical record number|mrn)\s*[:=#]?\s*[A-Za-z0-9-]+", re.I
    ),
    re.compile(r"\b(?:dob|date of birth)\s*[:=]", re.I),
    re.compile(r"\b(?:note_text|free_text|clinical_note)\b", re.I),
    re.compile(
        r"(?:file://|(?<![A-Za-z0-9])/(?:Users|home|private|tmp|var|etc|opt|Volumes)/|\b[A-Za-z]:\\)",
        re.I,
    ),
    re.compile(
        r"(?:\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{8,}|"
        r"\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+|"
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
        re.I,
    ),
    re.compile(
        r"[\"']?(?:subject_id|stay_id|hadm_id|patient_id|mrn)[\"']?\s*[:,=]\s*[\"']?[A-Za-z0-9-]+",
        re.I,
    ),
)
_FORBIDDEN_KEYS = {
    "rows",
    "records",
    "observations",
    "series",
    "values",
    "patient",
    "patients",
    "patient_id",
    "patient_ids",
    "subject_id",
    "subject_ids",
    "stay_id",
    "stay_ids",
    "hadm_id",
    "hadm_ids",
    "entity_id",
    "entity_ids",
    "timestamps",
    "notes",
    "credentials",
    "api_key",
    "token",
    "source_path",
    "project_dir",
    "path",
}


def path_digest(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def reject_sensitive_message(text: str) -> None:
    """Reject obvious row/identifier material before it reaches the shell model."""

    for pattern in _SENSITIVE_TEXT_PATTERNS:
        if pattern.search(text):
            raise PiCopilotError(
                "pi_message_phi_risk",
                (
                    "The EasyICU Copilot message looks like row-level patient data or "
                    "an identifier. Use aggregate study metadata instead."
                ),
                status_code=400,
            )


def _bounded_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def project_study_context(
    context: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not context:
        return {"present": False}
    source = context.get("data_source")
    source = source if isinstance(source, Mapping) else {}
    cohort = context.get("cohort")
    cohort = cohort if isinstance(cohort, Mapping) else {}
    idea_handoff = context.get("idea_handoff")
    idea_handoff = idea_handoff if isinstance(idea_handoff, Mapping) else {}
    literature_authority = context.get("literature_authority")
    literature_authority = (
        literature_authority
        if isinstance(literature_authority, Mapping)
        else {}
    )
    execution_concepts = context.get("execution_concepts")
    execution_concepts = (
        execution_concepts if isinstance(execution_concepts, Mapping) else {}
    )
    analysis_design = context.get("analysis_design")
    analysis_design = (
        analysis_design if isinstance(analysis_design, Mapping) else {}
    )
    sensitivity_specs = context.get("sensitivity_specs")
    sensitivity_specs = (
        sensitivity_specs if isinstance(sensitivity_specs, list) else []
    )
    # Every cohort field the update tool accepts is projected back, so a
    # receipt shows what was saved of each one (study_context_update).
    safe_cohort_keys = (
        "preset",
        "label",
        "review",
        "review_scope",
        "comparison",
        "comparison_mode",
        "source_type",
        "age_min",
        "age_max",
        "min_icu_los_hours",
        "observation_window_hours",
        "max_patients",
        "entity_count",
        "full_entity_count",
        "cohort_size",
        "source_count",
        "module_count",
        "exclude_readmissions",
        # The removal half of the cohort statement. Without it the model writes
        # to a slot it can never read back, so it cannot tell an unstated
        # exclusion from one it already recorded.
        "exclusion_statement",
    )
    projected_cohort = {
        key: cohort[key]
        for key in safe_cohort_keys
        if key in cohort and not isinstance(cohort[key], (dict, list))
    }
    for key in ("include_diagnoses", "exclude_diagnoses"):
        if isinstance(cohort.get(key), list):
            projected_cohort[key] = [
                _bounded_text(item, 160) for item in cohort[key]
            ][:64]
    trajectory_design = context.get("trajectory_design")
    trajectory_design = (
        trajectory_design if isinstance(trajectory_design, Mapping) else {}
    )
    question = _bounded_text(context.get("question"), 1200)
    if question:
        reject_sensitive_message(question)
    from easyicu.webserver.causal_trial_reading import normalize_causal_trial_reading
    from easyicu.webserver.study_family_reading import normalize_study_family_reading
    from easyicu.webserver.design_variance_basis import normalize_design_variance_basis
    from easyicu.webserver.question_family_check import (
        QUESTION_FAMILY_CHECK_FIELD,
        normalize_question_family_check,
    )

    from easyicu.research_agent.planning.adjustment_authority import normalize_stated_variables

    readings = {}
    for field, normalize in (
        ("causal_trial_reading", normalize_causal_trial_reading),
        ("study_family_reading", normalize_study_family_reading),
        (QUESTION_FAMILY_CHECK_FIELD, normalize_question_family_check),
        ("design_variance_basis", normalize_design_variance_basis),
        ("stated_adjustments", normalize_stated_variables),
        ("stated_subgroups", normalize_stated_variables),
    ):
        try:
            value = normalize(context.get(field))
        except ValueError:
            value = None
        if value is not None:
            readings[field] = value
    return ensure_safe_projection(
        {
            "present": True,
            "id": context.get("id"),
            "revision": context.get("revision"),
            "title": _bounded_text(context.get("title"), 160),
            "question": question,
            **readings,
            "purpose": _bounded_text(context.get("purpose"), 800),
            "data_source": {
                "source_id": source.get("source_id") or source.get("id"),
                "label": _bounded_text(source.get("label"), 160),
                "source_type": source.get("source_type") or source.get("type"),
                "database": source.get("database") or source.get("source_id"),
                "path_digest": path_digest(source.get("path")),
                "status": source.get("status"),
                **({"stays": source["stays"]} if "stays" in source else {}),
            },
            "cohort": projected_cohort,
            "modules": [
                str(item)[:120] for item in (context.get("modules") or [])
            ][:MAX_LIST_ITEMS],
            "outcome": _bounded_text(context.get("outcome"), 500),
            "primary_exposure": _bounded_text(
                context.get("primary_exposure"), 160
            ),
            **({"primary_exposure_concept": _bounded_text(context["primary_exposure_concept"], 128)}
               if context.get("primary_exposure_concept") else {}),
            "covariates": [
                _bounded_text(item, 160)
                for item in (context.get("covariates") or [])
            ][:MAX_LIST_ITEMS],
            "covariate_selection": (
                _bounded_text(
                    context.get("covariate_selection") or "planner_selectable",
                    40,
                )
            ),
            "covariate_operationalizations": {
                _bounded_text(key, 80): _bounded_text(value, 80)
                for key, value in dict(
                    context.get("covariate_operationalizations") or {}
                ).items()
            },
            "covariate_rationales": {
                _bounded_text(key, 80): _bounded_text(value, 500)
                for key, value in list(
                    dict(context.get("covariate_rationales") or {}).items()
                )[:64]
            },
            "covariate_temporal_roles": {
                _bounded_text(key, 80): _bounded_text(value, 40)
                for key, value in list(
                    dict(context.get("covariate_temporal_roles") or {}).items()
                )[:64]
            },
            "execution_concepts": {
                **(
                    {"outcome": _bounded_text(execution_concepts.get("outcome"), 80)}
                    if execution_concepts.get("outcome")
                    else {}
                ),
                **(
                    {
                        "primary_exposure": _bounded_text(
                            execution_concepts.get("primary_exposure"), 80
                        )
                    }
                    if execution_concepts.get("primary_exposure")
                    else {}
                ),
                **(
                    {
                        "primary_exposure_aggregation": _bounded_text(
                            execution_concepts.get("primary_exposure_aggregation"),
                            16,
                        )
                    }
                    if execution_concepts.get("primary_exposure_aggregation")
                    else {}
                ),
                "covariates": [
                    _bounded_text(item, 80)
                    for item in (execution_concepts.get("covariates") or [])
                ][:MAX_LIST_ITEMS],
            },
            "analysis_design": {
                key: _bounded_text(analysis_design.get(key), 80)
                for key in (
                    "analysis_family",
                    "analysis_unit",
                    "variance_estimator",
                    "cluster_unit",
                )
                if analysis_design.get(key)
            },
            "trajectory_design": {
                key: (
                    [_bounded_text(item, 80) for item in trajectory_design[key]][:16]
                    if isinstance(trajectory_design[key], list)
                    else trajectory_design[key]
                )
                for key in (
                    "coordinate_concepts",
                    "descriptive_only_concepts",
                    "window_start_hours",
                    "window_end_hours",
                    "grid_width_hours",
                    "minimum_available_windows",
                    "candidate_cluster_min",
                    "candidate_cluster_max",
                    "stability_resamples",
                    "stability_sample_fraction",
                    "minimum_mean_stability",
                    "minimum_cluster_fraction",
                )
                if trajectory_design.get(key) is not None
            },
            "sensitivity_specs": [
                {
                    key: (
                        [
                            _bounded_text(item, 80)
                            for item in (spec.get(key) or [])
                        ][:16]
                        if key == "execution_variables"
                        else spec.get(key)
                    )
                    for key in (
                        "spec_id",
                        "axis",
                        "strategy",
                        "execution_variables",
                        "landmark_hours",
                        "require_alive_at_landmark",
                        "exclude_negative_event_times",
                        "event_time_variable",
                        "observation_duration_variable",
                        "observation_duration_unit",
                    )
                    if spec.get(key) is not None
                }
                for spec in sensitivity_specs[:16]
                if isinstance(spec, Mapping)
            ],
            "time_window": dict(context.get("time_window") or {}),
            "comparator": _bounded_text(context.get("comparator"), 500),
            "export_format": _bounded_text(context.get("export_format"), 40),
            "analysis_goal": _bounded_text(context.get("analysis_goal"), 1200),
            "current_stage": context.get("current_stage"),
            "last_route": context.get("last_route"),
            "active_job_id": context.get("active_job_id"),
            "confirmations": dict(context.get("confirmations") or {}),
            "idea_handoff": {
                key: idea_handoff.get(key)
                for key in (
                    "schema_version",
                    "run_id",
                    "idea_id",
                    "canonical_handoff_sha256",
                    "status",
                    "accepted_at",
                    "go_no_go",
                    "go_no_go_reason",
                )
                if idea_handoff.get(key) is not None
            },
            "literature_authority": {
                key: literature_authority.get(key)
                for key in (
                    "schema_version",
                    "receipt_id",
                    "receipt_sha256",
                    "status",
                    "result_count",
                    "searched_at",
                    "study_configuration_sha256",
                )
                if literature_authority.get(key) is not None
            },
        }
    )


def project_study_setup_receipt(study: Mapping[str, Any]) -> StudySetupReceipt:
    """Project the exact bounded StudyContext fields used by workflow review."""

    projected = project_study_context(study)
    if not projected.get("present"):
        return StudySetupReceipt(
            study_context_id="",
            revision=0,
            configured_fields=[],
            configuration={},
        )
    source = projected.get("data_source")
    source = dict(source) if isinstance(source, Mapping) else {}
    configuration: Dict[str, Any] = {
        key: projected.get(key)
        for key in (
            "question",
            "purpose",
            "cohort",
            "modules",
            "outcome",
            "primary_exposure",
            "primary_exposure_concept",
            "covariates",
            "covariate_selection",
            "covariate_operationalizations",
            "execution_concepts",
            "analysis_design",
            "study_family_reading",
            "question_family_check",
            "stated_adjustments",
            "stated_subgroups",
            "causal_trial_reading",
            "design_variance_basis",
            "sensitivity_specs",
            "time_window",
            "comparator",
            "export_format",
            "analysis_goal",
            "confirmations",
        )
    }
    for field in (
        "crossdb_selection",
        "covariate_rationales",
        "covariate_temporal_roles",
    ):
        raw = study.get(field)
        configuration[field] = dict(raw) if isinstance(raw, Mapping) else {}
    configuration["data_source"] = {
        key: value
        for key, value in source.items()
        if key
        in {
            "source_id",
            "label",
            "source_type",
            "database",
            "path_digest",
            "status",
            "stays",
        }
        and value not in (None, "")
    }
    configuration["cohort_eligibility_authority"] = (
        cohort_eligibility.validated_authority(study) or {}
    )
    configuration = ensure_safe_projection(configuration)
    return StudySetupReceipt(
        study_context_id=str(projected.get("id") or ""),
        revision=int(projected.get("revision") or 0),
        configured_fields=[key for key, value in configuration.items() if bool(value)],
        configuration=configuration,
    )


def project_job(snapshot: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return _project_job(snapshot, include_progress=True)


def project_job_for_replay(snapshot: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A size failure of one child must not take its conversation offline."""
    try:
        return project_job(snapshot)
    except PiCopilotError as exc:
        if exc.code != "pi_projection_too_large":
            raise
        return _project_job(snapshot, include_progress=False)


def _project_job(
    snapshot: Optional[Mapping[str, Any]], *, include_progress: bool
) -> Dict[str, Any]:
    if not snapshot:
        return {"present": False}
    events = snapshot.get("events")
    events = events if isinstance(events, list) else []
    result = snapshot.get("result")
    result = result if isinstance(result, Mapping) else {}
    # Completed jobs recover early timings from the audit projection even if
    # the transient Job history has rolled over. No diagnostic detail enters.
    timings = result.get("trace_timings")
    timings = timings if isinstance(timings, list) else []
    audit_run_id = stable_code(result.get("run_id"))
    audit_events = [
        {"type": "progress", "step": row.get("phase"), "label": row.get("event"),
         **({"run_id": audit_run_id} if audit_run_id else {}),
         **{key: row[key] for key in (
             "timestamp", "status", "step_id", "current", "total", "planning_unit",
             "started_at", "ended_at",
             "retry_phase", *TRACE_ACTION_KEYS, *TRACE_FACT_KEYS,
         ) if key in row}}
        for row in timings if isinstance(row, Mapping)
    ]
    progress = []
    for event in [*audit_events, *events]:
        if not isinstance(event, Mapping):
            continue
        if str(event.get("type") or "") == "end":
            # Terminal state is projected once below; a raw end event created
            # a second content-free "end" row in the conversation timeline.
            continue
        label = _bounded_text(event.get("label"), 240)
        if label:
            try:
                reject_sensitive_message(label)
            except PiCopilotError:
                label = ""
        progress.append(
            {
                key: event.get(key)
                for key in (
                    "seq",
                    "type",
                    "status",
                    "current",
                    "total",
                    "step",
                    "timestamp",
                    "started_at",
                    "ended_at",
                    "step_id",
                    "actor",
                    "planning_unit",
                    "retry_phase",
                    "current_step",
                    "total_steps",
                )
                if event.get(key) is not None
            }
            | {
                **({"label": label} if label else {}),
                **({"restart_stage": "metadata_planning"}
                   if event.get("step") == "provider_auto_resume"
                   and event.get("restart_stage") == "metadata_planning" else {}),
                **({"reason_code": code} if (code := stable_code(
                    event.get("reason") or event.get("reason_code")
                )) else {}),
            }
            | _trace_facts(event)
        )
    # Check before compaction can overwrite an older value or drop display
    # facts. A budget fallback must never hide a privacy refusal.
    _check_projection_privacy(progress)
    compacted = bool(len(progress) > MAX_LIST_ITEMS or audit_events)
    if compacted:
        progress = _compact_job_progress(progress)
    if not include_progress:
        progress = []
    run_id = stable_code(result.get("run_id"))
    gate = result.get("gate")
    gate = RunGate.from_payload(gate).to_dict() if isinstance(gate, Mapping) else {}
    stop_gate = gate
    error_code = _safe_error_code(snapshot.get("error"))
    if not gate.get("reason") and snapshot.get("status") == "failed" and error_code in STOP_POLICIES:
        # A launch may stop before it has a run receipt. The existing job
        # projection already owns this code; no message supplies coordinates.
        stop_gate = {"status": "blocked", "reason": error_code}
    gate_checks = {
        str(item.get("id") or "").strip(): bool(item.get("passed"))
        for item in (gate.get("checks") or [])
        if isinstance(item, Mapping) and str(item.get("id") or "").strip()
    }
    artifacts = result.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    # A target trial compile's own result: codes and numbers only.
    trial_compile = project_target_trial_compile(result.get("target_trial_compile"))
    revision = result.get("report_revision")
    revision = revision if isinstance(revision, Mapping) else {}
    own_revision = bool(revision.get("revision_id") == snapshot.get("id") and revision)
    report_only = own_revision or any(
        isinstance(event, Mapping) and event.get("step") == "report_repair"
        for event in events
    )
    revision_ready = bool(
        own_revision and snapshot.get("status") == "done"
        and revision.get("schema_version") == "easyicu.web-report-revision/1"
        and revision.get("source_run_id") == run_id
        and revision.get("status") == "pass"
        and revision.get("analysis_steps_executed") == 0
        and revision.get("claim_ceiling") == "analysis_only"
        and revision.get("publication_authorized") is False
        and re.fullmatch(r"[a-f0-9]{64}", str(revision.get("output_sha256") or ""))
    )
    pdf = revision.get("pdf_artifact")
    pdf = pdf if isinstance(pdf, Mapping) else {}
    revision_pdf_ready = bool(
        revision_ready and pdf.get("name") == "manuscript_revision.pdf"
        and pdf.get("revision_id") == snapshot.get("id")
        and pdf.get("manuscript_sha256") == revision.get("output_sha256")
        and re.fullmatch(r"[a-f0-9]{64}", str(pdf.get("sha256") or ""))
        and any(isinstance(row, Mapping) and row.get("name") == pdf.get("name")
                and row.get("sha256") == pdf.get("sha256") for row in artifacts)
    )
    artifact_names = {
        str(row.get("name") or "").strip()
        for row in artifacts
        if isinstance(row, Mapping)
    }
    analysis_artifacts_present = bool(
        "result_tables.json" in artifact_names
        or "figure_gallery.json" in artifact_names
    )
    # ``numeric_verified`` is the manuscript-to-table provenance gate.  It is
    # intentionally false for an analysis-only run that stops before a fully
    # evidence-bound manuscript, so it must not erase completed and
    # automatically validated result tables/figures from the Web workflow.
    # The separate numeric/evidence/reportable fields continue to keep
    # publication claims fail-closed.
    analysis_results_available = bool(
        gate_checks.get("execution_complete") is True
        and analysis_artifacts_present
        and (
            gate_checks.get("analysis_validated") is True
            or (
                gate_checks.get("numeric_verified") is True
                and gate_checks.get("evidence_complete") is True
            )
        )
    )
    diagnostic_only = (
        str(gate.get("status") or "") == "blocked"
        and not bool(result.get("human_review_pending"))
        and not analysis_results_available
    )
    diagnostic_artifacts = {
        "run_context.json",
        "quality_gate.json",
        "scientific_readiness.json",
        "source_run_manifest.json",
        "evidence_ledger.json",
    }
    artifact_refs = []
    for row in artifacts[:80]:
        if not isinstance(row, Mapping):
            continue
        name = _bounded_text(row.get("name"), 160)
        digest = _bounded_text(row.get("sha256"), 64).lower()
        if diagnostic_only and name not in diagnostic_artifacts and not (
            revision_pdf_ready and name == "manuscript_revision.pdf"
        ):
            continue
        if (
            not run_id
            or not name
            or Path(name).name != name
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            continue
        artifact_refs.append(
            {
                "kind": (
                    "system_validation_document"
                    if name in {
                        "system_validation_report.html",
                        "system_validation_report.pdf",
                    }
                    else "research_document"
                    if name in {
                        "manuscript_scaffold.pdf",
                        "manuscript_revision.pdf",
                        "manuscript_scaffold.tex",
                        "manuscript_scaffold.bib",
                    }
                    else "research_artifact"
                ),
                "run_id": run_id,
                "artifact": name,
                "label": name,
                "sha256": digest,
                **(
                    {"size": int(row.get("size") or row.get("bytes"))}
                    if isinstance(row.get("size") or row.get("bytes"), int)
                    else {}
                ),
            }
        )
    return _bound_job_projection(
        {
            "present": True,
            "job_id": snapshot.get("id"),
            "kind": snapshot.get("kind"),
            "status": snapshot.get("status"),
            "created_at_epoch": snapshot.get("created"),
            "finished_at_epoch": snapshot.get("finished"),
            "cancel_requested": bool(snapshot.get("cancel_requested")),
            "cancel_reason_code": stable_code(snapshot.get("cancel_reason")),
            "error_code": _safe_error_code(snapshot.get("error")),
            "progress": progress,
            **({"progress_compacted": True} if compacted else {}),
            **({"progress_withheld_reason": "projection_budget"} if not include_progress else {}),
            "run_id": run_id,
            "research_input_state": project_research_input_state(
                result.get("research_input_state")
            ),
            "artifact_refs": artifact_refs,
            **{key: gate[key] for key in ("analysis_outcome", "manuscript_outcome") if key in gate},
            "gate_status": stable_code(gate.get("status")),
            "gate_reason_code": stable_code(gate.get("reason")),
            **gate_detail_projection(gate.get("detail")),
            "next_actions": project_stop_next_actions(stop_gate),
            "execution_complete": gate_checks.get("execution_complete") is True,
            "analysis_validated": gate_checks.get("analysis_validated") is True,
            "numeric_verified": gate_checks.get("numeric_verified") is True,
            "evidence_complete": gate_checks.get("evidence_complete") is True,
            "manuscript_ready": gate_checks.get("manuscript_ready") is True,
            "analysis_results_available": analysis_results_available,
            "limitations": [
                {"step_id": stable_code(item.get("step_id")),
                 "message": _bounded_text(item.get("message"), 500)}
                for item in (gate.get("limitations") or [])[:20]
                if isinstance(item, Mapping)
            ],
            "report_only": report_only,
            "report_revision_ready": revision_ready,
            "report_revision_pdf_ready": revision_pdf_ready,
            "report_revision_id": stable_code(revision.get("revision_id")) if own_revision else "",
            "reportable": bool(gate.get("reportable")),
            "human_review_pending": bool(result.get("human_review_pending")),
            **(
                {"target_trial_compile": trial_compile}
                if trial_compile is not None
                else {}
            ),
        }
    )


def _projection_bytes(value: Any) -> int:
    # Match ensure_safe_projection's actual wire measurement, including spaces.
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _progress_table(progress: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Lossless column encoding; each nested list respects MAX_LIST_ITEMS.

    Constant fields (including run coordinates and bilingual reader labels)
    live once in defaults. The decoder restores timestamp from ended_at when
    they coincide. No operation is merged with a different action/attempt.
    """
    rows = []
    for event in progress:
        row = dict(event)
        if row.get("timestamp") == row.get("ended_at"):
            row.pop("timestamp", None)
        rows.append(row)
    defaults = {key: value for key, value in rows[0].items()
                if all(key in row and row[key] == value for row in rows)}
    columns = list(dict.fromkeys(key for row in rows for key in row if key not in defaults))
    dictionaries = {}
    indices = {}
    # A column's repeated values (step ids, actions, reader labels) are stored
    # once, without inheriting a label onto a row that never carried it.
    for key in columns:
        values = [row.get(key) for row in rows]
        unique = {json.dumps(value, ensure_ascii=False, sort_keys=True): value
                  for value in values if value is not None}
        if len(unique) > MAX_LIST_ITEMS:
            continue
        lookup = {encoded: index for index, encoded in enumerate(unique)}
        encoded = [lookup[json.dumps(value, ensure_ascii=False, sort_keys=True)]
                   if value is not None else None for value in values]
        if _projection_bytes(list(unique.values())) + _projection_bytes(encoded) + len(key) + 16 < _projection_bytes(values):
            dictionaries[key] = list(unique.values())
            indices[key] = encoded
    cells = []
    for index, row in enumerate(rows):
        values = [indices[key][index] if key in indices else row.get(key) for key in columns]
        while values and values[-1] is None:
            values.pop()
        cells.append(values)
    return {
        "schema_version": "easyicu.pi-job-progress/1",
        "defaults": defaults,
        "columns": columns,
        "dictionaries": dictionaries,
        "batches": [
            cells[start:start + MAX_LIST_ITEMS]
            for start in range(0, len(cells), MAX_LIST_ITEMS)
        ],
    }


def _bound_job_projection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic ladder: operations, lossless table, display-free table.

    An irreducible trace cannot fit an arbitrarily large number of identities
    into 32 KiB. Withhold it explicitly rather than silently taking a tail;
    the full trace remains in the run's audit artifacts.
    """
    _check_projection_privacy(payload)
    progress = payload["progress"]
    header = {key: value for key, value in payload.items() if key != "progress"}
    budget = min(MAX_JOB_PROGRESS_BYTES, MAX_PROJECTION_BYTES - _projection_bytes(header) - 256)
    if len(progress) > MAX_LIST_ITEMS or _projection_bytes(progress) > budget:
        progress = _compact_job_progress(progress)
        payload.update(progress=[], progress_compacted=True, progress_operation_count=len(progress))
        # No representation can fit this many nonempty operations in the byte
        # budget. Avoid allocating a table with unbounded batches.
        if progress and len(progress) <= MAX_LIST_ITEMS * MAX_LIST_ITEMS:
            for optional_fields in ((), ("label", "queries", "source_label", "source_database",
                                         "group_variable_label", "outcome_label")):
                table = _progress_table([
                    {key: value for key, value in row.items() if key not in optional_fields
                     # Legacy step/verdict/reuse semantics still live in these
                     # fixed labels; only typed action prose is redundant.
                     or (key == "label" and (not row.get("action")
                                             or row.get("step") in {"step", "resume", "planning"}))}
                    for row in progress
                ])
                if _projection_bytes(table) <= budget:
                    payload["progress_table"] = table
                    if optional_fields:
                        payload["progress_display_facts_omitted"] = True
                    break
        if "progress_table" not in payload:
            payload["progress_withheld_reason"] = "projection_budget"
    try:
        return ensure_safe_projection(payload)
    except PiCopilotError as exc:
        if exc.code != "pi_projection_too_large":
            raise
    # Even artifact lists or optional setup receipts can consume the entire
    # contract. Keep the typed outcome and as many complete references as fit.
    minimal = {key: value for key, value in payload.items() if key not in {
        "progress_table", "target_trial_compile", "research_input_state",
    }}
    minimal.update(progress=[], progress_compacted=True, progress_withheld_reason="projection_budget")
    refs = minimal.pop("artifact_refs")
    minimal["artifact_refs"] = []
    for ref in refs:
        candidate = {**minimal, "artifact_refs": [*minimal["artifact_refs"], ref]}
        if _projection_bytes(candidate) > MAX_PROJECTION_BYTES - 128:
            minimal["artifact_refs_withheld_reason"] = "projection_budget"
            break
        minimal["artifact_refs"].append(ref)
    return ensure_safe_projection(minimal)


def _visible_fact_text(value: Any, limit: int) -> str:
    text = _bounded_text(value, limit)
    if not text:
        return ""
    try:
        reject_sensitive_message(text)
    except PiCopilotError:
        return ""
    return text


def _trace_facts(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the closed trace-action facts of one job event.

    ``run_step_trace.trace_action`` produced them from fixed host envelopes;
    the projection re-checks each type so a transport cannot widen them.
    """

    facts: Dict[str, Any] = {}
    for key in ("action", "executor", "run_kind", "repair_kind", "run_id"):
        code = stable_code(event.get(key))
        if code:
            facts[key] = code
    for key in ("attempt", "group_variable_count", "plan_step_count", "query_count", "record_count"):
        value = event.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 100_000:
            facts[key] = value
    # The research context's own counts: a large database holds millions of stays.
    for key in ("stay_count", "variable_count"):
        value = event.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10_000_000:
            facts[key] = value
    for key in ("source_label", "source_database"):
        text = _visible_fact_text(event.get(key), 160)
        if text:
            facts[key] = text
    for key in ("group_variable_label", "outcome_label"):
        raw = event.get(key)
        if not isinstance(raw, Mapping):
            continue
        labels = {
            language: text
            for language in ("zh", "en")
            if isinstance(raw.get(language), str)
            and (text := _visible_fact_text(raw[language], 240))
        }
        if labels:
            facts[key] = labels
    queries = event.get("queries")
    if isinstance(queries, list):
        shown = [text for text in (_visible_fact_text(item, 200) for item in queries[:4]) if text]
        if shown:
            facts["queries"] = shown
    return facts


def _compact_job_progress(events: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Bound repeated updates by operation, preserving both timing endpoints.

    Keeping a tail of messages erased entire early steps. Compaction is by
    typed step identity (or the legacy fixed envelope), never by recency.
    """
    grouped: Dict[tuple, Dict[str, Any]] = {}
    for event in events:
        step = str(event.get("step") or event.get("type") or "")
        step_id = str(event.get("step_id") or "")
        if not step_id and step in {"step", "coder", "runner", "resume"}:
            match = re.search(r"(?:Step \d+/\d+ [^:]+: |for |completed[^:]*: )([A-Za-z0-9_.-]+)\.?$", str(event.get("label") or ""))
            step_id = match[1].rstrip(".") if match else ""
        # Typed actions stay distinct: each repair attempt and each planning
        # validation attempt is its own operation with its own times.
        key = (step, step_id, event.get("planning_unit"), event.get("retry_phase"),
               event.get("current") if step.startswith("provider_") or event.get("action") else None,
               event.get("action"), event.get("repair_kind"), event.get("attempt"))
        previous = grouped.get(key)
        timestamp = str(event.get("timestamp") or "")
        row = {**(previous or {}), **event}
        start = str(event.get("started_at") or timestamp)
        end = str(event.get("ended_at") or timestamp)
        if start or end:
            row.update(
                started_at=min(str((previous or {}).get("started_at") or start), start),
                ended_at=max(str((previous or {}).get("ended_at") or end), end),
            )
        if step_id:
            row["step_id"] = step_id
        grouped[key] = row
    return list(grouped.values())


def project_run_outcome(
    review: Optional[Mapping[str, Any]],
    *,
    review_evidence_refs: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Project a durable, path-free completed-run result for Guided Copilot."""

    if not isinstance(review, Mapping) or not review.get("ok"):
        return {"present": False}
    gate = review.get("gate")
    gate = RunGate.from_payload(gate).to_dict() if isinstance(gate, Mapping) else {}
    projection = project_job(
        {
            "id": review.get("run_id"),
            "kind": "agent-run",
            "status": "done",
            "result": {
                "run_id": review.get("run_id"),
                "gate": gate,
                "artifacts": review.get("artifacts") or [],
                "human_review_pending": False,
            },
        }
    )
    artifact_payloads = review.get("artifact_payloads")
    artifact_payloads = (
        artifact_payloads if isinstance(artifact_payloads, Mapping) else {}
    )
    provenance = artifact_payloads.get("manuscript_provenance.json") or {}
    revision = (provenance.get("report_revision") or {}) if isinstance(provenance, Mapping) else {}
    # This is draft availability only, not the immutable source manuscript
    # gate or permission to publish. Payloads have passed the run ledger check.
    projection["report_revision_ready"] = bool(
        isinstance(revision, Mapping)
        and revision.get("schema_version") == "easyicu.web-report-revision/1"
        and revision.get("status") == "pass"
        and revision.get("source_run_id") == review.get("run_id")
        and revision.get("analysis_steps_executed") == 0
        and revision.get("claim_ceiling") == "analysis_only"
        and revision.get("publication_authorized") is False
        and re.fullmatch(r"[a-f0-9]{64}", str(revision.get("output_sha256") or ""))
        and revision.get("output_sha256") == provenance.get("manuscript_sha256")
    )
    pdf = revision.get("pdf_artifact") if isinstance(revision, Mapping) else None
    projection["report_revision_pdf_ready"] = bool(
        projection["report_revision_ready"] and isinstance(pdf, Mapping)
        and pdf.get("name") == "manuscript_revision.pdf"
        and pdf.get("revision_id") == revision.get("revision_id")
        and pdf.get("manuscript_sha256") == revision.get("output_sha256")
        and any(row.get("artifact") == pdf.get("name") and row.get("sha256") == pdf.get("sha256")
                for row in projection.get("artifact_refs", []))
    )
    if not projection["report_revision_pdf_ready"]:
        projection["artifact_refs"] = [row for row in projection.get("artifact_refs", [])
                                       if row.get("artifact") != "manuscript_revision.pdf"]
    figure_gallery = artifact_payloads.get("figure_gallery.json")
    if isinstance(figure_gallery, Mapping):
        figures = figure_gallery.get("figures")
        figure_count = len(figures) if isinstance(figures, list) else 0
        projection["figure_count"] = figure_count
        if figure_count == 0:
            projection["artifact_refs"] = [
                row
                for row in list(projection.get("artifact_refs") or [])
                if row.get("artifact") != "figure_gallery.json"
            ]
    projected_review_refs = []
    for row in list(review_evidence_refs or [])[:40]:
        if not isinstance(row, Mapping):
            continue
        evidence_id = _bounded_text(row.get("evidence_id"), 160)
        digest = _bounded_text(row.get("sha256"), 64).lower()
        reference = _bounded_text(row.get("reference"), 240)
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", evidence_id)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or not reference
        ):
            continue
        projected_review_refs.append(
            {
                "reference": reference,
                "evidence_id": evidence_id,
                "sha256": digest,
                "kind": _bounded_text(row.get("kind") or "artifact", 80),
                "label": _bounded_text(row.get("label") or reference, 160),
                "description": _bounded_text(row.get("description"), 500),
                **(
                    {"pointer": _bounded_text(row.get("pointer"), 500)}
                    if row.get("pointer")
                    else {}
                ),
            }
        )
    projection["review_evidence_refs"] = projected_review_refs
    return ensure_safe_projection(
        {
            key: value
            for key, value in projection.items()
            if key
            in {
                "present",
                "run_id",
                "artifact_refs",
                "gate_status",
                "gate_reason_code",
                "gate_detail_code",
                "gate_detail_cause_code",
                "gate_detail_mismatch_fields",
                "gate_detail_step_id",
                "gate_detail_step_title",
                "gate_transport_retries",
                "gate_missing_concepts",
                "next_actions",
                "execution_complete",
                "analysis_validated",
                "numeric_verified",
                "evidence_complete",
                "manuscript_ready",
                "analysis_outcome",
                "manuscript_outcome",
                "report_revision_ready",
                "report_revision_pdf_ready",
                "analysis_results_available",
                "figure_count",
                "review_evidence_refs",
                "reportable",
            }
        }
    )


def _project_replay_resource(value: Any) -> Optional[Dict[str, Any]]:
    """Keep only reopenable, path-free references from a normalized Pi event."""

    if not isinstance(value, Mapping):
        return None
    kind = str(value.get("kind") or "").strip()
    if kind == "literature_source":
        url = _bounded_text(value.get("url"), 500)
        if not re.fullmatch(
            r"(?:https://pubmed\.ncbi\.nlm\.nih\.gov/[0-9]{1,12}/|"
            r"https://doi\.org/10\.[0-9]{4,9}/[A-Za-z0-9._;()/:+\-]+)",
            url,
        ):
            return None
        resource = {
            "kind": kind,
            "url": url,
            "label": _bounded_text(value.get("label"), 160),
            "title": _bounded_text(value.get("title"), 500),
            "year": _bounded_text(value.get("year"), 16),
            "venue": _bounded_text(value.get("venue"), 240),
            "doi": _bounded_text(value.get("doi"), 240),
            "pmid": _bounded_text(value.get("pmid"), 32),
            "authority_class": stable_code(value.get("authority_class")),
        }
        retrieval_fit = stable_code(value.get("retrieval_fit"))
        if retrieval_fit in {
            "direct_retrieval_fit",
            "adjacent_retrieval_fit",
            "unclassified",
        }:
            resource["retrieval_fit"] = retrieval_fit
            resource["retrieval_rationale"] = _bounded_text(
                value.get("retrieval_rationale"), 600
            )
        if value.get("relevance"):
            resource["relevance"] = _bounded_text(value.get("relevance"), 600)
        return resource
    if kind in {
        "research_artifact",
        "research_document",
        "system_validation_document",
    }:
        run_id = stable_code(value.get("run_id"))
        artifact = _bounded_text(value.get("artifact"), 160)
        digest = _bounded_text(value.get("sha256"), 64).lower()
        if (
            not run_id
            or not artifact
            or Path(artifact).name != artifact
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            return None
        return {
            "kind": kind,
            "run_id": run_id,
            "artifact": artifact,
            "label": _bounded_text(value.get("label") or artifact, 160),
            "media_type": _bounded_text(value.get("media_type"), 120),
            "sha256": digest,
        }
    if kind == "idea_plan":
        run_id = stable_code(value.get("run_id"))
        artifact = _bounded_text(value.get("artifact"), 160)
        if not run_id or artifact != "idea_plan.json":
            return None
        return {
            "kind": kind,
            "run_id": run_id,
            "artifact": artifact,
            "label": _bounded_text(
                value.get("label") or "Idea Mining plan preview", 160
            ),
            "media_type": "application/json",
            "authority_class": "idea_mining_planning_only",
        }
    if kind == "data_package_review":
        study_id = stable_code(value.get("study_context_id"))
        digest = _bounded_text(value.get("review_sha256"), 64).lower()
        revision = value.get("study_revision")
        if (
            not study_id
            or not isinstance(revision, int)
            or revision < 0
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            return None
        return {
            "kind": kind,
            "study_context_id": study_id,
            "study_revision": revision,
            "review_sha256": digest,
            "label": _bounded_text(value.get("label"), 160),
            "media_type": "application/json",
        }
    if kind == "data_workbench_snapshot":
        view = stable_code(value.get("view"))
        digest = _bounded_text(value.get("snapshot_sha256"), 64).lower()
        if (
            view
            not in {
                "cohort_summary",
                "feature_distribution",
                "icd_cohort_preview",
                "patient_timeline",
                "crossdb_comparison",
            }
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            return None
        return {
            "kind": kind,
            "view": view,
            "snapshot_sha256": digest,
            "label": _bounded_text(value.get("label") or "Data Workbench", 160),
            "media_type": "application/json",
        }
    if kind == "native_workspace":
        route = stable_code(value.get("route"))
        study_id = stable_code(value.get("study_context_id"))
        revision = value.get("study_revision")
        state = stable_code(value.get("state"))
        job_id = stable_code(value.get("job_id"))
        source_id = stable_code(value.get("source_id"))
        expected_database = stable_code(value.get("expected_database"))
        entry_mode = stable_code(value.get("entry_mode"))
        extraction_scope = stable_code(value.get("extraction_scope"))
        allowed_databases = {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}
        allowed_scopes = {"study_required", "all_supported", "reuse_prepared_full"}
        if (
            route != "extraction"
            or not study_id
            or not isinstance(revision, int)
            or revision < 0
            or state not in {"setup", "running", "review"}
            or (source_id and not re.fullmatch(r"src_[a-f0-9]{12}", source_id))
            or (expected_database and expected_database not in allowed_databases)
            or (entry_mode and entry_mode != "source_binding")
            or (extraction_scope and extraction_scope not in allowed_scopes)
        ):
            return None
        return {
            "kind": kind,
            "route": route,
            "state": state,
            "study_context_id": study_id,
            "study_revision": revision,
            "label": _bounded_text(value.get("label") or "Data Extraction", 160),
            "media_type": "application/vnd.easyicu.native-workspace",
            **({"job_id": job_id} if job_id else {}),
            **({"source_id": source_id} if source_id else {}),
            **({"entry_mode": entry_mode} if entry_mode else {}),
            **({"extraction_scope": extraction_scope} if extraction_scope else {}),
            **(
                {"expected_database": expected_database}
                if expected_database
                else {}
            ),
        }
    if kind in {"file", "webpage"}:
        file_name = _bounded_text(value.get("file"), 240).replace("\\", "/")
        parts = file_name.split("/")
        if (
            not file_name
            or file_name.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
        ):
            return None
        digest = _bounded_text(value.get("sha256"), 64).lower()
        checked_digest = _bounded_text(value.get("checked_sha256"), 64).lower()
        if kind == "webpage" and not re.fullmatch(r"[a-f0-9]{64}", checked_digest):
            return None
        return {
            "kind": kind,
            "file": file_name,
            "label": _bounded_text(value.get("label") or parts[-1], 160),
            "media_type": _bounded_text(value.get("media_type"), 120),
            **(
                {"sha256": digest}
                if re.fullmatch(r"[a-f0-9]{64}", digest)
                else {}
            ),
            **({"checked_sha256": checked_digest} if kind == "webpage" else {}),
        }
    return None


def project_transcript(
    rows: Any, *, limit: int = 200
) -> list[Dict[str, Any]]:
    """Bound the transcript and strip machine-facing identifiers from replies.

    Assistant turns have machine-facing identifiers removed from their text.
    Reasoning prose and model-directed tool summaries stay in the SDK session.
    User turns remain verbatim except for exact,
    known host-generated legacy action text that was previously persisted
    under the user role.
    """

    if not isinstance(rows, list):
        return []
    projected: list[Dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, Mapping):
            continue
        role = str(row.get("role") or "")
        if role not in {"assistant", "user", "tool"}:
            projected.append(dict(row))
            continue
        content = row.get("content")
        if not isinstance(content, list):
            projected.append(dict(row))
            continue
        blocks: list[Any] = []
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_result":
                blocks.append(project_tool_receipt(block))
                continue
            if (
                isinstance(block, Mapping)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                blocks.append(
                    {
                        **block,
                        "text": (
                            sanitize_user_visible_text(block["text"])
                            if role == "assistant"
                            else project_user_turn_text(block["text"])
                        ),
                    }
                )
                continue
            if (
                isinstance(block, Mapping)
                and block.get("type") == "thinking"
            ):
                if role != "assistant":
                    continue
                blocks.append({"type": "thinking"})
                continue
            blocks.append(block)
        projected.append({**row, "content": blocks})
    return projected


def project_pi_replay_event(event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return one bounded UX replay event without text deltas or tool arguments."""

    event_type = stable_code(event.get("type"))
    if event_type not in {
        "run_start",
        "turn_start",
        "assistant_start",
        "thinking_start",
        "thinking_end",
        "tool_start",
        "tool_progress",
        "tool_end",
        "message_end",
        "turn_end",
        "agent_cycle_end",
        "run_end",
        "compaction_start",
        "compaction_end",
        "retry",
    }:
        return None
    payload: Dict[str, Any] = {
        "type": event_type,
        "at": _bounded_text(event.get("at"), 64),
    }
    for key in (
        "turn_index",
        "phase",
        "attempt",
        "max_attempts",
        "tool_call_id",
        "tool_name",
        "status",
        "code",
        "owner",
        "stop_reason",
        "error_code",
        "job_id",
    ):
        value = event.get(key)
        if value is None or isinstance(value, (dict, list)):
            continue
        if key in {"turn_index", "phase", "attempt", "max_attempts"}:
            try:
                payload[key] = int(value)
            except (TypeError, ValueError):
                continue
        else:
            payload[key] = _bounded_text(value, 240)
    if event.get("reason") is not None:
        payload["reason_code"] = stable_code(event.get("reason"))
    if event.get("is_error") is not None:
        payload["is_error"] = bool(event.get("is_error"))
    if event.get("will_retry") is not None:
        payload["will_retry"] = bool(event.get("will_retry"))
    if event.get("aborted") is not None:
        payload["aborted"] = bool(event.get("aborted"))
    if event_type == "tool_end":
        payload.update({key: value for key, value in project_tool_receipt(event).items() if key == "receipt"})
    resource = _project_replay_resource(event.get("resource"))
    if resource is not None:
        payload["resource"] = resource
    resources = [
        projected
        for raw in (
            event.get("resources") if isinstance(event.get("resources"), list) else []
        )[:80]
        if (projected := _project_replay_resource(raw)) is not None
    ]
    if resources:
        payload["resources"] = resources
    idea_mining = event.get("idea_mining")
    if isinstance(idea_mining, Mapping):
        idea = idea_mining.get("idea")
        feasibility = idea_mining.get("feasibility")
        if isinstance(idea, Mapping):
            concepts = [
                {
                    key: _bounded_text(row.get(key), 180)
                    for key in (
                        "concept_id",
                        "label",
                        "module",
                        "role",
                        "status",
                    )
                    if row.get(key) is not None
                }
                | (
                    {"available": bool(row.get("available"))}
                    if isinstance(row.get("available"), bool)
                    else {}
                )
                for row in list(idea.get("mapped_concepts") or [])[:24]
                if isinstance(row, Mapping)
            ]
            payload["idea_mining"] = {
                "run_id": stable_code(idea_mining.get("run_id")),
                "selected_idea_id": stable_code(
                    idea_mining.get("selected_idea_id")
                ),
                "idea": {
                    key: _bounded_text(idea.get(key), 1600)
                    for key in (
                        "idea_title",
                        "population",
                        "exposure_or_predictor",
                        "outcome",
                        "analysis_family",
                        "rationale",
                        "go_no_go",
                        "go_no_go_reason",
                        "next_action",
                        "plan_status",
                    )
                    if idea.get(key) is not None
                }
                | {
                    "mapped_concepts": concepts,
                    "unresolved_slots": [
                        _bounded_text(value, 120)
                        for value in list(idea.get("unresolved_slots") or [])[:8]
                    ],
                    "design_support": (
                        {
                            key: idea["design_support"].get(key)
                            for key in (
                                "card_id",
                                "version",
                                "file_sha256",
                                "trust_level",
                                "review_status",
                                "summary",
                                "population",
                                "time_zero",
                                "outcome_family",
                                "requires_confirmation",
                                "stop_conditions",
                                "citations",
                                "authority",
                            )
                            if idea["design_support"].get(key) is not None
                        }
                        if isinstance(idea.get("design_support"), Mapping)
                        else {}
                    ),
                },
                "feasibility": (
                    {
                        "status": stable_code(feasibility.get("status")),
                        "reason": _bounded_text(feasibility.get("reason"), 1600),
                        "reportable": feasibility.get("reportable") is True,
                    }
                    if isinstance(feasibility, Mapping)
                    else {}
                ),
            }
    return ensure_safe_projection(payload)


def project_capabilities(payload: Mapping[str, Any]) -> Dict[str, Any]:
    settings = payload.get("settings")
    settings = settings if isinstance(settings, Mapping) else {}
    capabilities = payload.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, Mapping) else {}
    result: Dict[str, Any] = {"settings": dict(settings), "capabilities": {}}
    for name, raw in capabilities.items():
        row = raw if isinstance(raw, Mapping) else {}
        result["capabilities"][str(name)] = {
            key: row.get(key)
            for key in (
                "enabled",
                "available",
                "status",
                "reason",
                "behavior",
                "scope",
            )
            if row.get(key) is not None
        }
    return ensure_safe_projection(result)


def project_run_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    pending_review_reason_codes = [
        stable_code(item)
        for item in (row.get("pending_review_reason_codes") or [])
        if stable_code(item)
    ][:16]
    waiting_for_plan_review = (
        str(row.get("run_status") or "") == "human_review_pending"
        and bool(
            {
                "operator_plan_approval_required",
                "plan_scientific_changes_required",
                "scientific_plan_review_policy_stale",
                "agent_plan_revision_nonconvergent",
                # A plan that refuses its approval (its population, or an
                # analysis the question asks for) is still a plan under review:
                # nothing has been analysed.
                *PLAN_APPROVAL_STOPS,
            }
            & set(pending_review_reason_codes)
        )
    )
    projected = {
            key: row.get(key)
            for key in (
                "run_id",
                "study_id",
                "scientific_configuration_sha256",
                "mode",
                "run_type",
                "engine",
                "gate_status",
                "analysis_outcome",
                "manuscript_outcome",
                "run_status",
                "readiness_status",
                "signed",
                "signoff_stale",
                "integrity_status",
                "reportable",
                "draft_unlocked",
                "artifact_count",
                "artifact_names",
                "plan_available",
                "updated_at",
                "plan_approval_allowed",
                "scientific_plan_review_status",
                "scientific_plan_review_score",
                "plan_revision_source_run_id",
                "plan_revision_nonconvergent",
            )
            if row.get(key) is not None
        }
    if pending_review_reason_codes:
        projected["pending_review_reason_codes"] = pending_review_reason_codes
    projected["next_actions"] = project_stop_next_actions(
        {"status": row.get("gate_status"), "reason": row.get("gate_reason"),
         "detail": row.get("gate_detail"), "next_actions": row.get("next_actions")},
        pending_reason_codes=pending_review_reason_codes,
    )
    if waiting_for_plan_review:
        # Result/manuscript files are intentionally emitted as governed
        # placeholders at the plan stage.  Make the execution state explicit
        # so a conversational model cannot infer that analysis ran merely from
        # those filenames.
        projected.update(
            {
                "execution_phase": "plan_review",
                "human_plan_review_pending": True,
                "plan_approval_allowed": plan_approval_allowed(row),
                "analysis_executed": False,
                "scientific_results_available": False,
                "artifact_semantics": "plan_stage_placeholders_not_analysis_results",
            }
        )
    return ensure_safe_projection(projected)


def project_artifacts(rows: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    projected = []
    for row in list(rows)[:MAX_LIST_ITEMS]:
        size = row.get("size")
        if size is None:
            size = row.get("bytes")
        kind = row.get("kind")
        media_type = row.get("media_type")
        if media_type is None and kind == "json":
            media_type = "application/json"
        projected.append(
            {
                key: value
                for key, value in (
                    ("name", row.get("name")),
                    ("size", size),
                    ("size_bytes", row.get("size_bytes")
                     if type(row.get("size_bytes")) is int
                     and 0 <= row["size_bytes"] <= MAX_ARTIFACT_SIZE_BYTES else None),
                    ("sha256", row.get("sha256")),
                    ("media_type", media_type),
                    ("kind", kind),
                )
                if value is not None
            }
        )
    return ensure_safe_projection(projected)


def bounded_json_projection(
    value: Any, *, max_bytes: int = MAX_PROJECTION_BYTES
) -> Any:
    """Keep approved structured artefacts bounded without inventing a summary."""

    ensure_safe_projection(value, max_bytes=max_bytes)
    return value


def _check_projection_privacy(value: Any) -> None:
    """Privacy precedes size checks, including fields a size fallback omits."""
    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                normalized = str(key).strip().lower().replace("-", "_")
                if normalized in _FORBIDDEN_KEYS:
                    raise PiCopilotError(
                        "pi_projection_blocked",
                        (
                            "The host withheld a model-visible field owned by "
                            f"the patient-data boundary: {normalized}"
                        ),
                        status_code=500,
                        details={"field": normalized},
                    )
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
        elif isinstance(node, Path):
            raise PiCopilotError(
                "pi_projection_blocked",
                "Filesystem paths are not model-visible Pi tool results.",
                status_code=500,
            )
        elif isinstance(node, str):
            for pattern in _SENSITIVE_TEXT_PATTERNS:
                if pattern.search(node):
                    raise PiCopilotError(
                        "pi_projection_blocked",
                        "The host withheld a sensitive model-visible string value.",
                        status_code=500,
                    )

    visit(value)


def ensure_safe_projection(
    value: Any, *, max_bytes: int = MAX_PROJECTION_BYTES
) -> Any:
    _check_projection_privacy(value)

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            if len(node) > MAX_LIST_ITEMS:
                raise PiCopilotError(
                    "pi_projection_too_large",
                    "The host projection contains too many list items.",
                    status_code=500,
                )
            for child in node:
                visit(child)
        elif isinstance(node, str) and len(node) > MAX_TEXT_CHARS:
            raise PiCopilotError(
                "pi_projection_too_large",
                "A model-visible string exceeded the projection limit.",
                status_code=500,
                details={"max_chars": MAX_TEXT_CHARS},
            )

    visit(value)
    size = _projection_bytes(value)
    if size > max_bytes:
        raise PiCopilotError(
            "pi_projection_too_large",
            "The host projection exceeded the bounded Pi tool-result contract.",
            status_code=500,
            details={"bytes": size, "max_bytes": max_bytes},
        )
    return value


def _safe_error_code(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    # Legacy cohort filtering raises a localized exception. Project only this
    # known failure, never an arbitrary exception or its local data path.
    if text in {"不支持的数据库: eicu_demo", "不支持的数据库: mimic_demo"}:
        return "extraction_database_filter_unsupported"
    return stable_code(text.split(":", 1)[0])


def gate_detail_projection(detail: Any) -> Dict[str, Any]:
    """Project a blocked gate's cause and bounded plain step title, or nothing.

    ``detail`` is the gate owner's ``{reason_code, cause_code,
    missing_concepts}``: a data foundation's reason and the concept ids it
    could not source, or what stopped a failed-closed run and that stop's own
    cause. Only codes, concept ids and the approved step's plain title cross;
    a gate without a detail adds no keys, so ordinary rows keep their shape.
    """

    if not isinstance(detail, Mapping):
        return {}
    code = stable_code(detail.get("reason_code"))
    if not code:
        return {}
    missing = [
        concept
        for concept in (stable_code(item) for item in (detail.get("missing_concepts") or []))
        if concept
    ][:16]
    projected: Dict[str, Any] = {"gate_detail_code": code, "gate_missing_concepts": missing}
    from easyicu.webserver.stop_next_actions import safe_stop_coordinates

    counts = safe_stop_coordinates(detail)
    if "mismatch_fields" in counts:
        projected["gate_detail_mismatch_fields"] = counts["mismatch_fields"]
    if "transport_retries" in counts:
        projected["gate_transport_retries"] = counts["transport_retries"]
    cause = stable_code(detail.get("cause_code"))
    if cause:
        projected["gate_detail_cause_code"] = cause
    step_id = str(detail.get("step_id") or "")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", step_id):
        projected["gate_detail_step_id"] = step_id
    step_title = detail.get("step_title")
    if (
        isinstance(step_title, str) and step_title.strip()
        and re.fullmatch(r"[^<>\x00-\x1f\x7f]{1,160}", step_title)
    ):
        projected["gate_detail_step_title"] = step_title.strip()
    return projected


def stable_code(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,159}", text):
        return text
    return None


__all__ = [
    "StudySetupReceipt",
    "project_transcript",
    "bounded_json_projection",
    "ensure_safe_projection",
    "gate_detail_projection",
    "path_digest",
    "project_artifacts",
    "project_capabilities",
    "project_job",
    "project_job_for_replay",
    "project_run_row",
    "project_study_context",
    "project_study_setup_receipt",
    "reject_sensitive_message",
    "stable_code",
]
