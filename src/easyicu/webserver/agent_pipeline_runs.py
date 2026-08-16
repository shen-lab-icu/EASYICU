"""Native Web bridge to the real Research Agent scientific pipeline.

Pi Copilot and the Web route own conversation and job lifecycle only.  This
module delegates concept selection/materialisation to the data-foundation
owner and Plan -> Execute -> Validate -> Write to ``ResearchAgentPipeline``.
It then creates a bounded, path-free Web projection of the pipeline's own
artifacts; it never computes a scientific result itself.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

from easyicu.extensions import (
    ExtensionActivationSnapshot,
    ExtensionRegistry,
    ExtensionRegistryError,
)
from easyicu.databases import normalize_database_key

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
from easyicu.research_agent.orchestration.progress import ProgressControlSignal
from easyicu.research_agent.providers.structured_retry import (
    safe_provider_error_category,
    safe_structured_attempt_metadata,
)
from easyicu.research_agent.acquisition.patient_grouping import (
    PatientGroupingBinding,
)
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificReview,
    render_agent_plan_revision_contract,
)
from easyicu.research_agent.publication_skills import (
    publication_skill_flags_from_settings,
)
from easyicu.research_agent.reporting.system_validation_report import (
    SystemValidationReport,
    build_system_validation_receipt,
    build_system_validation_report,
    projection_payload_sha256,
    render_system_validation_html,
)
from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
    run_measurement_missingness_figure,
)
from easyicu.webserver import (
    agent_runs,
    capabilities as capability_policy,
    dataio,
    literature_authority,
    provider_adapter,
    source_identity_authority,
)
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.ideas import mining as idea_mining
from easyicu.webserver.literature_projection import (
    load_current_plan_authority,
    load_run_literature_projection,
)
from easyicu.webserver.scientific_readiness_projection import (
    build_scientific_readiness_projection,
)
from easyicu.webserver.agent_review_recovery import (
    WebReviewRecoveryError,
    WebReviewRecoveryRecord,
    WebReviewRecoverySeed,
    get_record as get_review_recovery_record,
    put_record as put_review_recovery_record,
    put_recovery_seed,
    register_pipeline_work_root,
    remove_record as remove_review_recovery_record,
    remove_recovery_seed,
    unregister_pipeline_work_root_if_unused,
)

_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_MANUSCRIPT_PREVIEW = 24_000
_MAX_FIGURE_EMBED_BYTES = 420_000
_MAX_FIGURE_EMBED_TOTAL = 1_400_000
_MAX_TABLE_ROWS = 30
_MAX_TABLE_COLUMNS = 12
# ``UserPreferences.data_constraints`` is transported as one JSON string that is
# read downstream both as prompt text and by token-scanning scientific gates
# (the repeated-stay dependence gate in ``planning.scientific_review`` is the
# load-bearing example).  Cutting the serialized text at a character offset
# silently deletes whole trailing keys -- ``sort_keys`` sorts ``confirmations``
# and ``materialization_window`` last -- and leaves an unparseable value behind.
# Bound the STRUCTURE instead, so every top-level constraint key survives, the
# value stays valid JSON, and anything actually dropped is dropped visibly.
_MAX_DATA_CONSTRAINTS_CHARS = 2_400
_DATA_CONSTRAINT_LIST_HEADS = (16, 8, 4, 2, 1, 0)
_MANUSCRIPT_DOCUMENT_SPECS = {
    "manuscript_scaffold.pdf": ("application/pdf", 16 * 1024 * 1024),
    "manuscript_scaffold.tex": ("text/x-tex; charset=utf-8", 2 * 1024 * 1024),
    "manuscript_scaffold.bib": (
        "application/x-bibtex; charset=utf-8",
        2 * 1024 * 1024,
    ),
}
_SYSTEM_VALIDATION_DOCUMENT_SPECS = {
    "system_validation_report.html": ("text/html; charset=utf-8", 8 * 1024 * 1024),
    "system_validation_report.pdf": ("application/pdf", 16 * 1024 * 1024),
}
_RUN_DOCUMENT_SPECS = {
    **_MANUSCRIPT_DOCUMENT_SPECS,
    **_SYSTEM_VALIDATION_DOCUMENT_SPECS,
}
_MAX_PENDING = 16
_PENDING_LOCK = threading.RLock()
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
_UNSAFE_PROJECTION_PATTERNS = (
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
        r"[\"']?(?:subject_id|stay_id|hadm_id|patient_id|mrn)[\"']?"
        r"\s*[:,=]\s*[\"']?[A-Za-z0-9-]+",
        re.I,
    ),
)


class ResearchPipelineRunError(ProgressControlSignal):
    """Stable Web-facing failure from the true Research Agent bridge."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


@dataclass(frozen=True)
class _WebHumanReviewGate:
    """Server-owned reviewer identity for the local durable review route."""

    def reviewer_identity_resolver(self) -> str:
        return server_reviewer_identity()


def server_reviewer_identity() -> str:
    """Return the host-authenticated identity used by the local Web service."""

    return "easyicu_local_web_operator"


@dataclass
class _PendingRun:
    pipeline: Any
    pending: Any
    wrapper_dir: Path
    study: Dict[str, Any]
    provider: Dict[str, Any]
    acquisition: Any
    created_at: float
    credential_source: str = "scientific_provider"
    budget_mode: str = "full_reviewed"
    prepared_package_binding: Optional[Dict[str, Any]] = None
    provider_hard_stop: Optional[Any] = None


_PENDING: Dict[str, _PendingRun] = {}


def _acquisition_recovery_projection(acquisition: Any) -> Dict[str, Any]:
    selection = getattr(acquisition, "selection", None)
    coverage = getattr(acquisition, "coverage", None)
    return {
        "selected_concepts": list(
            getattr(selection, "selected_concepts", ()) or ()
        )[:64],
        "materialized_concepts": list(
            getattr(acquisition, "materialized_concepts", ()) or ()
        )[:128],
        "coverage_sufficient": bool(getattr(coverage, "sufficient", False)),
    }


def _rehydrate_acquisition_projection(payload: Mapping[str, Any]) -> Any:
    return SimpleNamespace(
        selection=SimpleNamespace(
            selected_concepts=list(payload.get("selected_concepts") or ())
        ),
        materialized_concepts=list(payload.get("materialized_concepts") or ()),
        coverage=SimpleNamespace(
            sufficient=bool(payload.get("coverage_sufficient"))
        ),
    )


def _checkpoint_pending_from_record(
    record: WebReviewRecoveryRecord,
    *,
    recover_recorded_decisions: bool = False,
) -> Any:
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        checkpoint_path,
        load_checkpoint,
    )
    from easyicu.research_agent.orchestration.human_review_restore import (
        recover_checkpoint_decisions_from_evidence,
    )
    from easyicu.research_agent.orchestration.workflow import HumanReviewPending

    wrapper_dir = Path(record.wrapper_dir).resolve()
    run_dir = (wrapper_dir / "pipeline" / record.run_id).resolve()
    try:
        run_dir.relative_to((wrapper_dir / "pipeline").resolve())
    except ValueError as exc:
        raise WebReviewRecoveryError("Web review run path escaped its wrapper") from exc
    checkpoint_file = checkpoint_path(run_dir)
    checkpoint = (
        recover_checkpoint_decisions_from_evidence(
            checkpoint_file,
            run_dir=run_dir,
        )
        if recover_recorded_decisions
        else load_checkpoint(checkpoint_file, require_pending=False)
    )
    if checkpoint.run_id != record.run_id:
        raise WebReviewRecoveryError("Web review checkpoint belongs to another run")
    if checkpoint.state not in {
        "pending",
        "approved_pending_execution",
        "executing",
    }:
        raise WebReviewRecoveryError(
            f"Web review checkpoint state {checkpoint.state!r} is not resumable"
        )
    return HumanReviewPending(
        run_id=record.run_id,
        thread_id=record.run_id,
        run_dir=str(run_dir),
        requests=checkpoint.requests,
        resume_scope="durable_checkpoint",
        resume_pid=None,
    )


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "study")).strip("-.")
    return text[:96] or "study"


def _clean_text(value: Any, limit: int = 1_200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default


def _read_json_with_digest(path: Path) -> Dict[str, Any]:
    payload = _read_json(path, {})
    if not isinstance(payload, Mapping):
        return {}
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return dict(payload)
    return {**dict(payload), "_source_sha256": digest}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _pending_plan_authority(pending: Optional[Any]) -> Dict[str, Any]:
    """Return the one exact typed plan bound into every paused request."""

    if pending is None:
        return {}
    observed: Optional[PlanReviewAuthority] = None
    for request in pending.requests:
        payload = request.payload if isinstance(request.payload, Mapping) else {}
        raw = payload.get("plan_review_authority")
        if not isinstance(raw, Mapping):
            continue
        try:
            authority = PlanReviewAuthority.model_validate(raw)
        except ValueError:
            return {}
        if observed is not None and authority != observed:
            return {}
        observed = authority
    return dict(observed.plan_payload) if observed is not None else {}


def _pending_bound_evidence_sha256(
    pending: Optional[Any], evidence_id: str
) -> Optional[str]:
    if pending is None:
        return None
    expected: Optional[str] = None
    for request in pending.requests:
        payload = request.payload if isinstance(request.payload, Mapping) else {}
        raw = payload.get("plan_review_authority")
        if not isinstance(raw, Mapping):
            continue
        try:
            authority = PlanReviewAuthority.model_validate(raw)
        except ValueError:
            return None
        observed = authority.evidence_sha256.get(evidence_id)
        if not observed or (expected is not None and expected != observed):
            return None
        expected = observed
    return expected


def _load_pending_scientific_review(
    run_dir: Optional[Path], pending: Optional[Any]
) -> Dict[str, Any]:
    """Project only the review file included in the paused plan authority."""

    if run_dir is None or pending is None:
        return {}
    expected_sha = _pending_bound_evidence_sha256(pending, "scientific_plan_review")
    if expected_sha is None:
        return {}
    path = run_dir / "scientific_plan_review.json"
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_JSON_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
            return {}
        review = PlanScientificReview.model_validate_json(raw)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return review.model_dump(mode="json")


def _pipeline_failure_code(exc: BaseException) -> str:
    chain = _pipeline_exception_chain(exc)
    if any("timeout" in type(item).__name__.casefold() for item in chain):
        return "research_pipeline_provider_timeout"
    if any(type(item).__name__ == "StructuredResponseFailure" for item in chain):
        return "research_pipeline_plan_contract_exhausted"
    return "research_pipeline_execution_failed"


def _pipeline_exception_chain(exc: BaseException) -> List[BaseException]:
    """Return one bounded exception chain without following cycles."""

    chain: List[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in chain and len(chain) < 8:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _safe_pipeline_attempt_metadata(exc: BaseException) -> List[Dict[str, Any]]:
    """Extract only the approved, response-free structured-attempt fields."""

    for item in _pipeline_exception_chain(exc):
        attempts = getattr(item, "attempts", None)
        if isinstance(attempts, (list, tuple)):
            return safe_structured_attempt_metadata(attempts)
        projected = getattr(item, "easyicu_structured_attempt_metadata", None)
        if isinstance(projected, list):
            # Re-validate even producer-projected rows at the same closed-enum
            # owner boundary; exception attributes are mutable and untrusted.
            return safe_structured_attempt_metadata(projected)
    return []


def _pipeline_failure_category(exc: BaseException) -> str:
    """Return a closed diagnostic category for one bounded exception chain."""

    categories = [
        safe_provider_error_category(item) for item in _pipeline_exception_chain(exc)
    ]
    for preferred in (
        "provider_budget",
        "timeout",
        "rate_limit",
        "connection",
        "provider_http",
        "authorization",
        "structured_response",
        "validation",
        "parse",
    ):
        if preferred in categories:
            return preferred
    return "error"


def _write_pipeline_failure_diagnostic(
    *,
    wrapper_dir: Path,
    exc: BaseException,
    code: str,
) -> Optional[str]:
    """Persist bounded host diagnostics for a failed real pipeline run.

    Exception messages and notes are never public diagnostic channels: either
    can contain provider echoes, prompt fragments, validator inputs, private
    reasoning, paths, patient text, or credentials.
    """

    attempts = _safe_pipeline_attempt_metadata(exc)
    diagnostic_message = "The governed Research Agent operation failed."
    message_sha256 = hashlib.sha256(
        f"{type(exc).__name__}: {exc}".encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": "easyicu.web-research-pipeline-failure/3",
        "status": "failed",
        "code": code,
        "failure_type": _pipeline_failure_category(exc),
        "message": diagnostic_message,
        "message_sha256": message_sha256,
        "structured_retry_history": [],
        "structured_attempts": attempts,
        "raw_model_output_recorded": False,
        "prompt_recorded": False,
        "patient_rows_recorded": False,
        "secrets_recorded": False,
    }
    relative = "diagnostics/research_pipeline_failure.json"
    try:
        target = wrapper_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, payload)
    except OSError:
        return None
    return relative


def _safe_relative(root: Path, raw: Any) -> Optional[Path]:
    text = str(raw or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or "\0" in text:
        return None
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    candidate = root.joinpath(*parts)
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    if candidate.is_symlink() or not resolved.is_file():
        return None
    return resolved


def _target_outcome(study: Mapping[str, Any]) -> Optional[str]:
    execution = study.get("execution_concepts")
    execution = execution if isinstance(execution, Mapping) else {}
    value = _clean_text(execution.get("outcome") or study.get("outcome"), 160)
    if value.lower() in {
        "none",
        "n/a",
        "na",
        "not applicable",
        "descriptive only",
        "无",
        "不适用",
        "仅描述",
    }:
        return None
    return value or None


def _primary_exposure(study: Mapping[str, Any]) -> Optional[str]:
    execution = study.get("execution_concepts")
    execution = execution if isinstance(execution, Mapping) else {}
    return _clean_text(
        execution.get("primary_exposure") or study.get("primary_exposure"),
        160,
    ) or None


def _configured_covariates(study: Mapping[str, Any]) -> tuple[str, ...]:
    execution = study.get("execution_concepts")
    execution = execution if isinstance(execution, Mapping) else {}
    raw = (
        execution.get("covariates")
        if "covariates" in execution
        else study.get("covariates")
    )
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(
            _clean_text(value, 160)
            for value in raw
            if isinstance(value, str) and _clean_text(value, 160)
        )
    )


def _configured_sensitivity_specs(study: Mapping[str, Any]) -> tuple[Any, ...]:
    """Load only the typed sensitivity authority owned by StudyContext."""

    from easyicu.research_agent.planning.sensitivity_authority import (
        normalize_prespecified_sensitivities,
    )

    try:
        return normalize_prespecified_sensitivities(study.get("sensitivity_specs"))
    except (TypeError, ValueError) as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_sensitivity_specs_invalid",
            "The configured prespecified sensitivity contract is invalid.",
            details={"field": "sensitivity_specs", "reason": str(exc)[:500]},
        ) from exc


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


def _validate_primary_concept_selection(
    study: Mapping[str, Any],
    primary_exposure: Optional[str],
) -> None:
    """Enforce the concept owner's user-intent selection policy at launch."""

    if not primary_exposure:
        return
    from easyicu.concept.selection_policy import evaluate_concept_selection

    # Only the persisted scientific question can authorize an explicit-only
    # variant. Exposure labels and analysis prose are model-produced fields;
    # accepting them here would let a plan authorize its own semantic drift.
    intent = str(study.get("question") or "")
    decision = evaluate_concept_selection(primary_exposure, user_intent=intent)
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
    raw = study.get("time_window")
    window = raw if isinstance(raw, Mapping) else {}
    if not window:
        raise ResearchPipelineRunError(
            "research_pipeline_time_window_required",
            "A typed study time window is required before pipeline launch.",
            details={"field": "time_window"},
        )
    value = window.get("hours")
    if value is None:
        value = window.get("observation_hours")
    if value is None:
        raise ResearchPipelineRunError(
            "research_pipeline_time_window_hours_required",
            (
                "The time-window label or preset has no executable duration; "
                "hours or observation_hours must be explicitly bound."
            ),
            details={"field": "time_window.hours"},
        )
    if not _clean_text(window.get("anchor"), 160):
        raise ResearchPipelineRunError(
            "research_pipeline_time_window_anchor_required",
            "The typed study time window requires an explicit scientific anchor.",
            details={"field": "time_window.anchor"},
        )
    window_finding = study_context_owner.materialization_window_finding(dict(study))
    if window_finding is not None:
        raise ResearchPipelineRunError(
            "research_pipeline_materialization_window_anchor_unsupported",
            (
                "The configured time-window anchor is not an executable "
                "outer materialization coordinate for this pipeline."
            ),
            details={
                key: value
                for key, value in window_finding.items()
                if key != "error"
            },
        )
    if isinstance(value, bool):
        raise ResearchPipelineRunError(
            "research_pipeline_time_window_invalid",
            "The configured study time window must be a finite number of hours.",
        )
    try:
        hours = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_time_window_invalid",
            "The configured study time window must be a finite number of hours.",
        ) from exc
    if not math.isfinite(hours) or not 0 < hours <= 24 * 365:
        raise ResearchPipelineRunError(
            "research_pipeline_time_window_invalid",
            "The configured study time window is outside the supported range.",
        )
    return (0.0, hours)


def _configured_modules(study: Mapping[str, Any]) -> tuple[str, ...]:
    raw = study.get("modules")
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        dict.fromkeys(
            str(module).strip().lower()
            for module in raw
            if isinstance(module, str) and str(module).strip()
        )
    )


def _data_foundation_profile(
    *,
    export_path: str,
    study: Mapping[str, Any],
    target: Optional[str],
    primary_exposure: Optional[str] = None,
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
    if cohort.get("exclude_readmissions") is True:
        readmission_meta = by_id.get("icu_readmission")
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
            raise ResearchPipelineRunError(
                "research_pipeline_target_outside_configured_modules",
                "The configured outcome is not available in the selected feature modules.",
                details={"field": "execution_concepts.outcome", "concept_id": target},
            )
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
            for variable in spec.execution_variables
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
            role = (
                "primary_exposure"
                if concept_id == primary_exposure
                else (
                    "covariate"
                    if concept_id in covariates
                    else "sensitivity_variable"
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


def _resolve_materialized_primary_exposure(
    *,
    configured: Optional[str],
    source_concept: Optional[str],
    acquisition: Any,
) -> Optional[str]:
    """Resolve only an owner-issued materialized exposure coordinate."""

    if not configured:
        return None
    source = source_concept or configured
    if configured == source:
        return (getattr(acquisition, "analysis_columns", {}) or {}).get(source)
    materialized = set(getattr(acquisition, "materialized_columns", ()) or ())
    return configured if configured in materialized else None


def _elide_constraint_lists(
    constraints: Mapping[str, Any], *, head: int
) -> Dict[str, Any]:
    """Shorten list values in place of deleting whole constraint keys."""

    reduced: Dict[str, Any] = {}
    for key, value in constraints.items():
        if not isinstance(value, Mapping):
            reduced[key] = value
            continue
        row: Dict[str, Any] = {}
        for name, item in value.items():
            if isinstance(item, list) and len(item) > head:
                # The marker is deliberately token-free: it must not be able to
                # satisfy a downstream gate's text scan on its own.
                row[name] = [*item[:head], f"[{len(item) - head} omitted]"]
            else:
                row[name] = item
        reduced[key] = row
    return reduced


def _compile_data_constraints(constraints: Mapping[str, Any]) -> str:
    """Serialize the study's data constraints without silent key loss.

    Returns valid JSON containing every top-level constraint key, or fails
    closed.  A study whose constraints cannot fit even with every list fully
    elided is a configuration the user must shorten; it must never be answered
    by handing the Research Agent a truncated blob whose missing tail happens
    to be the user's confirmations.
    """

    payload = json.dumps(constraints, ensure_ascii=False, sort_keys=True)
    if len(payload) <= _MAX_DATA_CONSTRAINTS_CHARS:
        return payload
    for head in _DATA_CONSTRAINT_LIST_HEADS:
        candidate = json.dumps(
            _elide_constraint_lists(constraints, head=head),
            ensure_ascii=False,
            sort_keys=True,
        )
        if len(candidate) <= _MAX_DATA_CONSTRAINTS_CHARS:
            return candidate
    raise ResearchPipelineRunError(
        "research_pipeline_data_constraints_too_large",
        "The configured study constraints are too large to transport to the "
        "Research Agent without dropping part of them. Shorten the longest "
        "cohort text fields and retry.",
        details={
            "field": "data_constraints",
            "limit_chars": _MAX_DATA_CONSTRAINTS_CHARS,
            "serialized_chars": len(payload),
            "section_chars": {
                str(key): len(
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                )
                for key, value in constraints.items()
            },
        },
    )


def _research_user_preferences(
    study: Mapping[str, Any],
    *,
    patient_grouping: Optional[PatientGroupingBinding] = None,
) -> Dict[str, Any]:
    """Compile StudyContext into the existing strict preference contract."""

    preferences: Dict[str, Any] = {}
    purpose = _clean_text(study.get("purpose"), 1_200)
    analysis_goal = _clean_text(study.get("analysis_goal"), 1_200)
    comparator = _clean_text(study.get("comparator"), 800)
    if purpose:
        preferences["extra_notes"] = purpose
    if analysis_goal:
        preferences["must_have_outputs"] = analysis_goal
    if comparator:
        preferences["subgroup_sensitivity"] = comparator
    raw_analysis_design = study.get("analysis_design")
    analysis_design = (
        raw_analysis_design if isinstance(raw_analysis_design, Mapping) else {}
    )
    analysis_family = _clean_text(analysis_design.get("analysis_family"), 80)
    if analysis_family:
        from easyicu.research_agent.planning.analysis_types import (
            canonical_analysis_family,
        )

        if canonical_analysis_family(analysis_family) != analysis_family:
            raise ResearchPipelineRunError(
                "research_pipeline_analysis_family_invalid",
                "The typed StudyContext analysis family is not a canonical Research Agent family.",
                details={"field": "analysis_design.analysis_family"},
            )
        # ResearchContext already owns a typed family coordinate. Populate it
        # from the user-approved StudyContext instead of asking free-text
        # keyword inference to reinterpret a descriptive risk contrast.
        preferences["inferred_analysis_family"] = analysis_family
    covariates = _configured_covariates(study)
    selection = str(
        study.get("covariate_selection") or "planner_selectable"
    ).strip()
    if selection not in {"planner_selectable", "exact"}:
        raise ResearchPipelineRunError(
            "research_pipeline_covariate_selection_invalid",
            "StudyContext covariate_selection must be planner_selectable or exact.",
        )
    if selection == "exact":
        # An exact empty roster is a positive user decision to run unadjusted.
        # Merely serializing ``covariates=[]`` is not that authority.
        preferences["covariates"] = list(covariates)
        preferences["covariate_selection"] = "exact"
        preferences["covariate_rationales"] = dict(
            study.get("covariate_rationales") or {}
        )
        preferences["covariate_temporal_roles"] = dict(
            study.get("covariate_temporal_roles") or {}
        )
    elif covariates:
        preferences["covariates"] = list(covariates)

    constraints: Dict[str, Any] = {}
    cohort = study.get("cohort")
    confirmations = study.get("confirmations")
    if isinstance(cohort, Mapping) and cohort:
        constraints["cohort"] = dict(cohort)
    if isinstance(confirmations, Mapping) and confirmations:
        constraints["confirmations"] = dict(confirmations)
    if analysis_design:
        constraints["analysis_design"] = dict(analysis_design)
    if patient_grouping is not None:
        coordinates = dict(patient_grouping.authority_coordinates)
        constraints["verified_patient_grouping"] = {
            "coordinate": patient_grouping.output_identity_column,
            "group_derivation": "prefix_before_:s",
            "authority_ref": coordinates.get("authority_ref"),
            "mapping_sha256": patient_grouping.mapping_sha256,
            "provider_visible_values": False,
        }
    time_window = study.get("time_window")
    if isinstance(time_window, Mapping) and time_window:
        # StudyContext.time_window is owned by the Web materialization
        # contract.  It bounds the physical feature extraction coordinate; it
        # is not the phenotype's clinical definition anchor or the outcome
        # follow-up horizon.  Keep it in data constraints so the Research
        # Agent can audit the executed window without interpreting its ICU-
        # admission anchor as the study's clinical time zero.
        constraints["materialization_window"] = {
            "role": "outer_observation_window",
            **dict(time_window),
        }
    if constraints:
        preferences["data_constraints"] = _compile_data_constraints(constraints)
    sensitivity_specs = _configured_sensitivity_specs(study)
    if sensitivity_specs:
        preferences["sensitivity_specs"] = [
            spec.model_dump(mode="json") for spec in sensitivity_specs
        ]
        landmarks = {
            float(spec.landmark_hours)
            for spec in sensitivity_specs
            if spec.strategy == "landmark" and spec.landmark_hours is not None
        }
        if len(landmarks) > 1:
            raise ResearchPipelineRunError(
                "research_pipeline_landmark_authority_ambiguous",
                "The configured sensitivities declare more than one landmark origin.",
                details={"field": "sensitivity_specs", "landmark_hours": sorted(landmarks)},
            )
        if landmarks:
            preferences["landmark_hours"] = next(iter(landmarks))
    return preferences


def _inclusion_criteria(study: Mapping[str, Any]) -> List[str]:
    raw = study.get("cohort")
    cohort = raw if isinstance(raw, Mapping) else {}
    rows: List[str] = []
    review = _clean_text(cohort.get("review") or cohort.get("label"), 500)
    if review:
        rows.append(review)
    age_min = cohort.get("age_min")
    age_max = cohort.get("age_max")
    if age_min is not None or age_max is not None:
        rows.append(
            f"age range: {age_min if age_min is not None else '*'} to {age_max if age_max is not None else '*'}"
        )
    minimum_los = cohort.get("min_icu_los_hours")
    if minimum_los is not None:
        rows.append(f"minimum ICU length of stay: {minimum_los} hours")
    if cohort.get("exclude_readmissions") is True:
        rows.append("first eligible ICU stay per patient")
    for key, prefix in (
        ("include_diagnoses", "include diagnoses"),
        ("exclude_diagnoses", "exclude diagnoses"),
    ):
        values = cohort.get(key)
        if isinstance(values, list):
            clean = [_clean_text(item, 120) for item in values[:20]]
            clean = [item for item in clean if item]
            if clean:
                rows.append(f"{prefix}: {', '.join(clean)}")
    return rows[:32]


def _primary_cohort_selection_mode(study: Mapping[str, Any]) -> str:
    """Compile the user-owned Web cohort choice to the pipeline contract.

    The acquisition owner has already materialised the configured source,
    modules, outcome, and exposure.  Descriptive labels such as "has a
    Sepsis-3 determination" must not become a second, Planner-invented row
    filter.  Only explicit structured filtering fields in the StudyContext
    authorize a predicate-filtered primary cohort; otherwise every bound input
    row is the prespecified denominator.
    """

    raw = study.get("cohort")
    cohort = raw if isinstance(raw, Mapping) else {}
    explicit_filter_fields = (
        "age_min",
        "age_max",
        "min_icu_los_hours",
        "include_diagnoses",
        "exclude_diagnoses",
    )
    if cohort.get("exclude_readmissions") is True:
        return "predicate_filtered"
    if any(cohort.get(field) not in (None, "", []) for field in explicit_filter_fields):
        return "predicate_filtered"
    return "all_input_rows"


def _progress(job: Any, *, step: str, label: str, **extra: Any) -> None:
    event: Dict[str, Any] = {
        "type": "progress",
        "step": _clean_text(step, 80),
        "label": _clean_text(label, 240),
    }
    for key in ("current", "total", "status", "run_id"):
        if extra.get(key) is not None:
            event[key] = extra[key]
    job.emit(event)
    if getattr(job, "cancel_requested", False):
        raise ResearchPipelineRunError(
            "research_pipeline_cancelled",
            "The Research Agent run was cancelled by the user.",
        )


def _pipeline_progress(job: Any, event: Mapping[str, Any]) -> None:
    _progress(
        job,
        step=str(
            event.get("step")
            or event.get("stage")
            or event.get("phase")
            or "research_pipeline"
        ),
        label=str(
            event.get("label") or event.get("message") or "Research Agent working"
        ),
        current=event.get("current"),
        total=event.get("total"),
        status=event.get("status"),
        run_id=event.get("run_id"),
    )


def _safe_claims(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "claim_ledger.csv"
    try:
        if not path.is_file() or path.stat().st_size > 512_000:
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))[:40]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    claims: List[Dict[str, Any]] = []
    for row in rows:
        text = _clean_text(row.get("claim_text"), 2_000)
        if not text:
            continue
        refs = [
            _clean_text(item, 160)
            for item in re.split(r"[;,|]", str(row.get("evidence_refs") or ""))[:40]
            if _clean_text(item, 160)
        ]
        claims.append(
            {
                "id": _clean_text(row.get("claim_id"), 160),
                "text": text,
                "evidence_ids": refs,
                "status": _clean_text(row.get("status"), 120) or "evidence_bound_draft",
                "note": _clean_text(row.get("note"), 800),
            }
        )
    return claims


def _figure_projection(run_dir: Path) -> Dict[str, Any]:
    source = _read_json(run_dir / "figure_gallery.json", {})
    figures = source.get("figures") if isinstance(source, Mapping) else []
    public: List[Dict[str, Any]] = []
    embedded_total = 0
    for row in figures[:24] if isinstance(figures, list) else []:
        if not isinstance(row, Mapping):
            continue
        relative = _clean_text(row.get("relative_path"), 300)
        item: Dict[str, Any] = {
            "label": _clean_text(row.get("label") or row.get("figure_id"), 240),
            "name": Path(relative).name if relative else "figure",
            "relative_path": relative,
            "status": _clean_text(row.get("status"), 120) or "available",
            "tier": _clean_text(row.get("tier"), 120),
            "panel_roles": [
                _clean_text(value, 120)
                for value in list(row.get("panel_roles") or [])[:12]
                if _clean_text(value, 120)
            ],
            "chart_types": [
                _clean_text(value, 120)
                for value in list(row.get("chart_types") or [])[:12]
                if _clean_text(value, 120)
            ],
        }
        path = _safe_relative(run_dir, relative)
        if path is not None and path.suffix.lower() == ".png":
            try:
                size = path.stat().st_size
                if (
                    0 < size <= _MAX_FIGURE_EMBED_BYTES
                    and embedded_total + size <= _MAX_FIGURE_EMBED_TOTAL
                ):
                    item["data_url"] = "data:image/png;base64," + base64.b64encode(
                        path.read_bytes()
                    ).decode("ascii")
                    embedded_total += size
            except OSError:
                pass
        public.append(item)
    return {
        "kind": "figure_gallery",
        "schema_version": "easyicu.web-pipeline-figure-gallery/1",
        "status": _clean_text(
            source.get("status") if isinstance(source, Mapping) else "", 120
        )
        or ("available" if public else "no_figures"),
        "figures": public,
        "primary_count": int(source.get("primary_count") or 0)
        if isinstance(source, Mapping)
        else 0,
        "supporting_count": int(source.get("supporting_count") or 0)
        if isinstance(source, Mapping)
        else len(public),
        "embedded_count": sum(1 for row in public if row.get("data_url")),
    }


def _identifier_column(name: Any) -> bool:
    token = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    return token in {
        "stayid",
        "subjectid",
        "patientid",
        "hadmid",
        "icustayid",
        "patientunitstayid",
        "recordid",
    }


_TABLE_PREVIEW_PRIORITY_COLUMNS = (
    "row_role",
    "concept",
    "variable",
    "label",
    "exposure_level",
    "n_rows",
    "n_total",
    "n_before",
    "n_excluded",
    "n_remaining",
    "exposure_denominator",
    "exposure_pct",
    "outcome_events",
    "outcome_denominator",
    "outcome_rate_pct",
    "interval_method",
    "eligible_n",
    "not_applicable_n",
    "value_missing_n",
    "event_present_n",
    "event_absent_n",
    "before_origin_n",
    "indicator_semantics",
    "missingness_kind",
)


def _table_preview_indices(headers: List[str]) -> List[int]:
    """Keep review-critical aggregate columns when a wide table is bounded."""

    selected = [
        headers.index(name)
        for name in _TABLE_PREVIEW_PRIORITY_COLUMNS
        if name in headers
    ][:_MAX_TABLE_COLUMNS]
    for index in range(len(headers)):
        if len(selected) >= _MAX_TABLE_COLUMNS:
            break
        if index not in selected:
            selected.append(index)
    return selected


def _table_projection(run_dir: Path) -> Dict[str, Any]:
    evidence = _read_json(run_dir / "evidence" / "evidence_index.json", [])
    tables: List[Dict[str, Any]] = []
    skipped_sensitive = 0
    for record in evidence if isinstance(evidence, list) else []:
        if not isinstance(record, Mapping) or str(record.get("kind")) != "table":
            continue
        path = _safe_relative(run_dir, record.get("relative_path"))
        if path is None or path.suffix.lower() != ".csv":
            continue
        try:
            if path.stat().st_size > 1_500_000:
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                source_headers = next(reader, [])
                # Scan the complete header before bounding the preview. An
                # identifier beyond the visible column cap is still sensitive.
                if any(_identifier_column(value) for value in source_headers):
                    skipped_sensitive += 1
                    continue
                column_indices = _table_preview_indices(source_headers)
                headers = [source_headers[index] for index in column_indices]
                rows = [
                    [row[index] if index < len(row) else "" for index in column_indices]
                    for _, row in zip(range(_MAX_TABLE_ROWS), reader)
                ]
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        tables.append(
            {
                "evidence_id": _clean_text(record.get("evidence_id"), 160),
                "label": _clean_text(record.get("description"), 300) or path.stem,
                "name": path.name,
                "headers": [_clean_text(value, 160) for value in headers],
                "rows": [[_clean_text(value, 500) for value in row] for row in rows],
                "preview_truncated": len(rows) >= _MAX_TABLE_ROWS,
                "preview_columns_truncated": len(source_headers) > len(headers),
            }
        )
        if len(tables) >= 12:
            break
    return {
        "schema_version": "easyicu.web-pipeline-result-tables/1",
        "tables": tables,
        "table_count": len(tables),
        "skipped_identifier_tables": skipped_sensitive,
        "preview_policy": "aggregate_tables_only_no_identifier_columns",
    }


def _readiness_axes(run_dir: Path) -> Dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json", {})
    status = _read_json(run_dir / "run_status.json", {})
    readiness = manifest.get("readiness") if isinstance(manifest, Mapping) else {}
    if not isinstance(readiness, Mapping):
        readiness = {}
    gates = status.get("gates") if isinstance(status, Mapping) else {}
    if not isinstance(gates, Mapping):
        gates = {}
    merged = dict(gates)
    merged.update(readiness)
    return {
        key: merged.get(key)
        for key in (
            "execution_complete",
            "step_scientific_requirements_complete",
            "artifact_valid",
            "scientific_requirement_complete",
            "evidence_complete",
            "numeric_verified",
            "analysis_validated",
            "manuscript_ready",
            "manuscript_generated",
            "publication_ready",
            "paper_authorized",
            "display_suite_complete",
            "display_suite_errors",
            "missing_evidence_count",
            "numeric_error_count",
            "evidence_error_count",
            "analysis_error_count",
            "failed_steps",
            "missing_steps",
            "analysis_errors",
        )
        if key in merged
    }


def _provider_usage_projection(wrapper_dir: Path) -> Optional[Dict[str, Any]]:
    """Project aggregate Provider accounting without exposing request content."""

    ledger_path = wrapper_dir / ".runtime" / "provider_hard_stop_ledger.json"
    source = _read_json(ledger_path, {})
    tasks = source.get("tasks") if isinstance(source, Mapping) else None
    rows = [row for row in (tasks or []) if isinstance(row, Mapping)]
    if not rows:
        return None
    try:
        ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    except OSError:
        ledger_sha256 = None
    calls = [
        call
        for row in rows
        for call in list(row.get("calls") or [])
        if isinstance(call, Mapping)
    ]
    statuses = {_clean_text(row.get("status"), 80) for row in rows}
    return {
        "status": (
            "completed"
            if statuses == {"completed"}
            else sorted(statuses)[0]
            if len(statuses) == 1
            else "mixed"
        ),
        "calls": len(calls),
        "accounted_tokens": sum(
            max(0, int(call.get("accounted_tokens") or 0)) for call in calls
        ),
        "estimated_cost_usd": round(
            sum(
                max(0.0, float(call.get("accounted_estimated_cost_usd") or 0.0))
                for call in calls
            ),
            8,
        ),
        "ledger_sha256": ledger_sha256,
    }


def _gate_from_axes(axes: Mapping[str, Any], *, pending: bool) -> Dict[str, Any]:
    if pending:
        status = "blocked"
        reason = "human_plan_review_required"
    else:
        complete = bool(axes.get("execution_complete"))
        validated = bool(axes.get("analysis_validated"))
        evidence = bool(axes.get("evidence_complete"))
        numeric = bool(axes.get("numeric_verified"))
        status = (
            "analysis_only"
            if complete and validated and evidence and numeric
            else "blocked"
        )
        reason = (
            "research_agent_pipeline_complete_human_interpretation_required"
            if status == "analysis_only"
            else "research_agent_pipeline_failed_closed"
        )
    check_labels = {
        "execution_complete": "execution complete",
        "analysis_validated": "automated analysis validation",
        "evidence_complete": "evidence references complete",
        "numeric_verified": "reported numbers verified",
        # The Research Agent's manuscript_ready axis means that an
        # evidence-bound draft exists.  It is deliberately not the publication
        # axis; the old generic label made those two states look equivalent.
        "manuscript_ready": "evidence-bound draft generated",
        "publication_ready": "publication package ready",
        "paper_authorized": "exact-run paper authority granted",
    }
    checks = []
    for key, label in check_labels.items():
        passed = bool(axes.get(key))
        checks.append(
            {
                "id": key,
                "label": label,
                "passed": passed,
                "reason": None if passed else f"{key}_not_satisfied",
            }
        )
    return {
        "status": status,
        "reason": reason,
        "reportable": False,
        "draft_unlocked": False,
        "checks": checks,
    }


def _artifact_record(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    document_spec = _RUN_DOCUMENT_SPECS.get(path.name)
    return {
        "name": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "kind": "document" if document_spec is not None else "json",
        "media_type": (
            document_spec[0] if document_spec is not None else "application/json"
        ),
    }


def _validated_manuscript_documents(
    run_dir: Optional[Path],
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load only renderer-receipt-bound draft documents from a pipeline run."""

    if run_dir is None:
        return None, []
    receipt_path = run_dir / "manuscript_pdf_receipt.json"
    if not receipt_path.exists():
        return None, []
    receipt = _read_json(receipt_path, {})
    security = receipt.get("security") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "easyicu.manuscript_pdf_receipt.v1"
        or receipt.get("status") != "rendered"
        or receipt.get("draft_watermark") is not True
        or not isinstance(security, Mapping)
        or security.get("network_allowed") is not False
        or security.get("shell_escape_allowed") is not False
        or security.get("untrusted_input_mode") is not True
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_pdf_receipt_invalid",
            "The manuscript PDF receipt is missing its draft or sandbox authority.",
        )

    receipt_rows = {
        "manuscript_scaffold.pdf": receipt.get("pdf"),
        "manuscript_scaffold.tex": receipt.get("source"),
        "manuscript_scaffold.bib": receipt.get("bibliography"),
    }
    documents: List[Dict[str, Any]] = []
    root = run_dir.resolve(strict=True)
    for name, receipt_row in receipt_rows.items():
        if receipt_row is None and name == "manuscript_scaffold.bib":
            continue
        if (
            not isinstance(receipt_row, Mapping)
            or receipt_row.get("name") != name
            or not re.fullmatch(r"[0-9a-f]{64}", str(receipt_row.get("sha256") or ""))
        ):
            raise ResearchPipelineRunError(
                "research_pipeline_pdf_receipt_binding_invalid",
                f"The PDF receipt does not bind {name}.",
            )
        source = run_dir / name
        try:
            metadata = source.lstat()
            resolved = source.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ResearchPipelineRunError(
                "research_pipeline_pdf_document_missing",
                f"The receipt-bound manuscript document {name} is unavailable.",
            ) from exc
        _media_type, max_bytes = _MANUSCRIPT_DOCUMENT_SPECS[name]
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > max_bytes
        ):
            raise ResearchPipelineRunError(
                "research_pipeline_pdf_document_unsafe",
                f"The manuscript document {name} failed the file boundary.",
            )
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt_row.get("sha256"):
            raise ResearchPipelineRunError(
                "research_pipeline_pdf_document_digest_mismatch",
                f"The manuscript document {name} no longer matches its receipt.",
            )
        if name.endswith(".pdf") and not raw.startswith(b"%PDF"):
            raise ResearchPipelineRunError(
                "research_pipeline_pdf_signature_invalid",
                "The receipt-bound manuscript PDF has an invalid signature.",
            )
        documents.append(
            {
                "name": name,
                "content": raw,
                "media_type": _MANUSCRIPT_DOCUMENT_SPECS[name][0],
            }
        )
    return receipt, documents


def register_system_validation_pdf(wrapper_dir: Path) -> Dict[str, Any]:
    """Bind an externally rendered PDF to an existing system report projection."""

    root = wrapper_dir.resolve(strict=True)
    paths = {
        "report": root / "system_validation_report.json",
        "receipt": root / "system_validation_report_receipt.json",
        "html": root / "system_validation_report.html",
        "pdf": root / "system_validation_report.pdf",
    }
    raw: Dict[str, bytes] = {}
    limits = {
        "report": _MAX_JSON_BYTES,
        "receipt": _MAX_JSON_BYTES,
        "html": _SYSTEM_VALIDATION_DOCUMENT_SPECS[
            "system_validation_report.html"
        ][1],
        "pdf": _SYSTEM_VALIDATION_DOCUMENT_SPECS["system_validation_report.pdf"][1],
    }
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise ResearchPipelineRunError(
            "system_validation_document_missing",
            "The system validation document directory is unavailable.",
        ) from exc
    try:
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for key, path in paths.items():
            descriptor = -1
            try:
                descriptor = os.open(path.name, file_flags, dir_fd=root_descriptor)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limits[key]:
                    raise ResearchPipelineRunError(
                        "system_validation_document_unsafe",
                        f"The system validation {key} document failed its file boundary.",
                    )
                with os.fdopen(descriptor, "rb", closefd=True) as handle:
                    descriptor = -1
                    content = handle.read(limits[key] + 1)
                if len(content) > limits[key]:
                    raise ResearchPipelineRunError(
                        "system_validation_document_unsafe",
                        f"The system validation {key} document failed its file boundary.",
                    )
                raw[key] = content
            except FileNotFoundError as exc:
                raise ResearchPipelineRunError(
                    "system_validation_document_missing",
                    f"The system validation {key} document is unavailable.",
                ) from exc
            except OSError as exc:
                raise ResearchPipelineRunError(
                    "system_validation_document_unsafe",
                    f"The system validation {key} document failed its file boundary.",
                ) from exc
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    finally:
        os.close(root_descriptor)
    if not raw["pdf"].startswith(b"%PDF"):
        raise ResearchPipelineRunError(
            "system_validation_pdf_signature_invalid",
            "The system validation PDF has an invalid signature.",
        )
    try:
        import fitz  # type: ignore

        with fitz.open(stream=raw["pdf"], filetype="pdf") as document:
            pdf_privacy_text = "\n".join(
                [
                    *(page.get_text() for page in document),
                    *(
                        value
                        for value in document.metadata.values()
                        if isinstance(value, str)
                    ),
                ]
            )
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        raise ResearchPipelineRunError(
            "system_validation_pdf_parse_invalid",
            "The system validation PDF could not be parsed for privacy review.",
        ) from exc
    try:
        report_payload = json.loads(raw["report"].decode("utf-8"))
        report = SystemValidationReport.model_validate(report_payload)
        existing_receipt = json.loads(raw["receipt"].decode("utf-8"))
        html_text = raw["html"].decode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResearchPipelineRunError(
            "system_validation_report_invalid",
            "The system validation report does not match its typed schema.",
        ) from exc
    if report.authority_class != "engineering_validation_only":
        raise ResearchPipelineRunError(
            "system_validation_authority_invalid",
            "The system validation report has an invalid authority class.",
        )
    report_payload_digest = projection_payload_sha256(report_payload)
    html_binding = (
        f'<meta name="easyicu-report-payload-sha256" '
        f'content="{report_payload_digest}">'
    )
    compact_pdf_text = re.sub(r"\s+", "", pdf_privacy_text)
    normalized_pdf_text = re.sub(r"\s+", " ", pdf_privacy_text).strip()
    status_label = (
        "REVIEWER DEMONSTRATION COMPLETE"
        if report.status == "engineering_validation_complete"
        else "ENGINEERING VALIDATION INCOMPLETE"
    )
    required_pdf_text = (
        report.title,
        report.subtitle,
        report.executive_summary,
        report.thesis,
        f"Run {report.run_id}",
        "authority=engineering_validation_only",
        "publication_authorized=false",
        status_label,
        "ENGINEERING VALIDATION ONLY · NOT A CLINICAL MANUSCRIPT",
    )
    if (
        html_binding not in html_text
        or report_payload_digest not in compact_pdf_text
        or any(
            re.sub(r"\s+", " ", value).strip() not in normalized_pdf_text
            for value in required_pdf_text
        )
    ):
        raise ResearchPipelineRunError(
            "system_validation_pdf_content_binding_mismatch",
            "The system validation PDF does not bind the registered report content.",
        )
    if (
        not isinstance(existing_receipt, Mapping)
        or existing_receipt.get("schema_version")
        != "easyicu.system-validation-report-receipt/1"
    ):
        raise ResearchPipelineRunError(
            "system_validation_receipt_invalid",
            "The existing system validation receipt is unavailable or invalid.",
        )
    for key in ("report", "html"):
        binding = existing_receipt.get(key)
        expected_sha256 = (
            str(binding.get("sha256") or "").lower()
            if isinstance(binding, Mapping)
            else ""
        )
        expected_bytes = binding.get("bytes") if isinstance(binding, Mapping) else None
        if (
            not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
            or expected_sha256 != hashlib.sha256(raw[key]).hexdigest()
            or expected_bytes != len(raw[key])
        ):
            raise ResearchPipelineRunError(
                "system_validation_receipt_digest_mismatch",
                f"The existing system validation {key} bytes do not match their receipt.",
            )
    ledger_path = root / "evidence_ledger.json"
    ledger = _read_json(ledger_path, {})
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != "easyicu.web-research-pipeline-ledger/1"
        or not isinstance(ledger.get("artifacts"), list)
    ):
        raise ResearchPipelineRunError(
            "system_validation_ledger_invalid",
            "The run evidence ledger cannot register the system validation PDF.",
        )
    ledger_by_name = {
        str(row.get("name") or ""): row
        for row in ledger["artifacts"]
        if isinstance(row, Mapping)
    }
    for key, name in (
        ("report", "system_validation_report.json"),
        ("receipt", "system_validation_report_receipt.json"),
        ("html", "system_validation_report.html"),
    ):
        binding = ledger_by_name.get(name)
        expected_sha256 = (
            str(binding.get("sha256") or "").lower()
            if isinstance(binding, Mapping)
            else ""
        )
        if key == "receipt" and (
            not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
            or expected_sha256 != hashlib.sha256(raw[key]).hexdigest()
        ):
            # A crash after the receipt replacement but before the ledger
            # replacement is recoverable because the receipt's report/HTML
            # bindings were independently verified above.
            continue
        if (
            not re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
            or expected_sha256 != hashlib.sha256(raw[key]).hexdigest()
        ):
            raise ResearchPipelineRunError(
                "system_validation_ledger_digest_mismatch",
                f"The existing {name} bytes do not match the run evidence ledger.",
            )
    privacy = _projection_privacy_scan(
        {
            "system_validation_report.json": report_payload,
            "system_validation_report.html": html_text,
            "system_validation_report.pdf": pdf_privacy_text,
        }
    )
    if not privacy["passed"]:
        raise ResearchPipelineRunError(
            "system_validation_projection_privacy_blocked",
            "The system validation document failed the browser privacy boundary.",
        )
    receipt = build_system_validation_receipt(
        report_payload=report_payload,
        html_bytes=raw["html"],
        pdf_bytes=raw["pdf"],
        report_bytes=raw["report"],
    )
    receipt_path = root / "system_validation_report_receipt.json"
    _write_json(receipt_path, receipt)
    names = {
        "system_validation_report.json",
        "system_validation_report_receipt.json",
        "system_validation_report.html",
        "system_validation_report.pdf",
    }
    artifacts = [
        row
        for row in ledger["artifacts"]
        if isinstance(row, Mapping) and row.get("name") not in names
    ]
    registered = [
        _artifact_record(root / name)
        for name in (
            "system_validation_report.json",
            "system_validation_report_receipt.json",
            "system_validation_report.html",
            "system_validation_report.pdf",
        )
    ]
    ledger["artifacts"] = [*artifacts, *registered]
    _write_json(ledger_path, ledger)
    return {
        "ok": True,
        "authority_class": report.authority_class,
        "claim_ceiling": report.claim_ceiling,
        "publication_authorized": False,
        "artifacts": registered,
    }


def _projection_privacy_scan(payloads: Mapping[str, Any]) -> Dict[str, Any]:
    """Reject host paths, credentials, and row identifiers in Web artefacts.

    Aggregate result-table rows are intentionally permitted.  The scientific
    pipeline owns their disclosure checks, while this boundary independently
    rejects the three classes that must never cross into the browser.
    """

    hits: List[Dict[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, Path):
            hits.append({"path": path, "reason": "path_object"})
        elif isinstance(value, str):
            for index, pattern in enumerate(_UNSAFE_PROJECTION_PATTERNS):
                if pattern.search(value):
                    hits.append({"path": path, "reason": f"pattern_{index + 1}"})
                    break

    for name, payload in payloads.items():
        visit(payload, str(name))
    return {
        "passed": not hits,
        "scanned_artifacts": len(payloads),
        "unsafe_value_count": len(hits),
        "hits": hits[:40],
    }


def _system_validation_figure_gallery(
    *,
    wrapper_dir: Path,
    run_dir: Path,
    plan: Mapping[str, Any],
    figure_gallery: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Reproject legacy data-quality figures with explicit applicability.

    Finalized pipeline outputs remain immutable. This report-only projection
    reuses the exact resolved audit-table binding and current deterministic
    renderer so conditional event prevalence cannot appear as a measurement
    rate. New runs already emit the corrected source figure.
    """

    source_rows = [
        dict(row)
        for row in list(figure_gallery.get("figures") or [])
        if isinstance(row, Mapping)
    ]
    target_index = next(
        (
            index
            for index, row in enumerate(source_rows)
            if row.get("name") == "data_quality.png"
        ),
        None,
    )
    if target_index is None:
        return None
    figure_step = next(
        (
            row
            for row in list(plan.get("steps") or [])
            if isinstance(row, Mapping)
            and "figure:data_quality" in set(row.get("expected_outputs") or [])
        ),
        None,
    )
    if not isinstance(figure_step, Mapping):
        return None
    step_id = str(figure_step.get("step_id") or "").strip()
    table_inputs = [
        str(value).strip()
        for value in list(figure_step.get("inputs") or [])
        if str(value).strip().startswith("table:")
    ]
    if not step_id or len(table_inputs) != 1:
        return None
    resolved_inputs = run_dir / "resolved_inputs" / f"{step_id}.json"
    correction = {
        "code": "CONDITIONAL_EVENT_TIME_APPLICABILITY_SEPARATED",
        "source_figure": "data_quality.png",
        "renderer": "deterministic_measurement_missingness_figure",
    }
    try:
        out_dir = wrapper_dir / ".runtime" / "system_validation_figures"
        summary = run_measurement_missingness_figure(
            out_dir=out_dir,
            run_dir=run_dir,
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            figure_product="data_quality",
            input_key=table_inputs[0],
        )
        image_bytes = (out_dir / "data_quality.png").read_bytes()
        source_sha256 = str(
            (summary.get("source_sha256") or {}).get(table_inputs[0]) or ""
        )
        source_rows[target_index].update(
            {
                "data_url": "data:image/png;base64,"
                + base64.b64encode(image_bytes).decode("ascii"),
                "label": (
                    "figure:data quality · applicability-aware semantic correction"
                ),
                "status": "supporting_corrected_projection",
                "projection_note": (
                    "Conditional event times separate applicable events, true "
                    "missingness within applicable stays, and non-applicable stays."
                ),
                "source_input": table_inputs[0],
                "source_sha256": source_sha256,
            }
        )
        correction.update(
            {
                "status": "corrected",
                "source_input": table_inputs[0],
                "source_sha256": source_sha256,
            }
        )
    except Exception as exc:  # fail closed by withholding the ambiguous figure
        source_rows.pop(target_index)
        correction.update(
            {
                "status": "withheld",
                "reason_code": "semantic_correction_render_failed",
                "error_type": type(exc).__name__,
            }
        )
    return {
        **dict(figure_gallery),
        "schema_version": "easyicu.system-validation-figure-gallery/1",
        "figures": source_rows,
        "embedded_count": len(source_rows),
        "source_gallery_sha256": projection_payload_sha256(figure_gallery),
        "projection_corrections": [correction],
    }


def _privacy_blocked_payloads(
    *,
    run_context: Mapping[str, Any],
    scan: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return a fixed-schema package without echoing unsafe source values."""

    gate = {
        "status": "blocked",
        "reason": "research_pipeline_projection_privacy_blocked",
        "reportable": False,
        "draft_unlocked": False,
        "checks": [
            {
                "id": "browser_projection_privacy",
                "label": "Browser projection contains no host path, credential, or row identifier",
                "passed": False,
                "unsafe_value_count": int(scan.get("unsafe_value_count") or 0),
            }
        ],
    }
    return {
        "run_context.json": {
            "run_id": _clean_text(run_context.get("run_id"), 160),
            "study_id": _clean_text(run_context.get("study_id"), 160),
            "mode": "research_agent_pipeline",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "summary": {"projection_withheld": True},
            "local_first": {"uploads": 0},
        },
        "quality_gate.json": {
            "gate": gate,
            "quality": [],
            "privacy": {
                "passed": False,
                "unsafe_value_count": int(scan.get("unsafe_value_count") or 0),
                "payloads_withheld": True,
            },
        },
    }


def _write_projection(
    *,
    wrapper_dir: Path,
    study: Mapping[str, Any],
    provider: Mapping[str, Any],
    acquisition: Any,
    run_dir: Optional[Path],
    pending: Optional[Any] = None,
    blocked_reason: Optional[str] = None,
) -> Dict[str, Any]:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(
        getattr(pending, "run_id", "")
        or (run_dir.name if run_dir else wrapper_dir.name)
    )
    axes = _readiness_axes(run_dir) if run_dir is not None else {}
    gate = _gate_from_axes(axes, pending=pending is not None)
    if blocked_reason:
        gate["status"] = "blocked"
        gate["reason"] = blocked_reason
    selection = getattr(acquisition, "selection", None)
    selected_concepts = list(getattr(selection, "selected_concepts", []) or [])
    coverage = getattr(acquisition, "coverage", None)
    run_context = {
        "run_id": run_id,
        "study_id": _clean_text(study.get("id"), 160),
        "scientific_configuration_sha256": (
            study_context_owner.scientific_configuration_sha256(study)
        ),
        "mode": "research_agent_pipeline",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "question": _clean_text(study.get("question"), 1_200),
        "source": {
            "label": _clean_text((study.get("data_source") or {}).get("label"), 160)
            if isinstance(study.get("data_source"), Mapping)
            else "",
            "database": _clean_text(
                (study.get("data_source") or {}).get("database"), 64
            )
            if isinstance(study.get("data_source"), Mapping)
            else "",
        },
        "summary": {
            "selected_concepts": selected_concepts[:64],
            "materialized_concept_count": len(
                list(getattr(acquisition, "materialized_concepts", []) or [])
            ),
            "coverage_sufficient": bool(getattr(coverage, "sufficient", False)),
        },
        "local_first": {"uploads": 0},
    }
    cohort_summary = {
        "summary": run_context["summary"],
        "cohort": {
            "criteria": _inclusion_criteria(study),
            "target_outcome": _target_outcome(study),
            "time_window_hours": _cohort_window(study)[1],
        },
    }
    # At a plan-review pause the final manifest does not exist yet.  Recover
    # the plan only from the digest-bound review authority; never fall back to
    # the mutable initial file for this browser projection.
    plan = (
        _pending_plan_authority(pending)
        if pending is not None
        else (load_current_plan_authority(run_dir) if run_dir else {})
    )
    literature_evidence = (
        load_run_literature_projection(
            run_dir=run_dir,
            run_id=run_id,
            plan=plan,
        )
        if run_dir
        else {
            "schema_version": "easyicu.web-literature-evidence/5",
            "scope": "research_plan",
            "run_id": run_id,
            "status": "unavailable",
            "citations": [],
            "step_citation_map": [],
            "mapping_status": "not_applicable",
            "integrity": {
                "path_values_returned": False,
                "patient_rows_returned": False,
            },
        }
    )
    scientific_plan_review = _load_pending_scientific_review(run_dir, pending)
    scientific_readiness = build_scientific_readiness_projection(
        run_id=run_id,
        run_dir=run_dir,
        axes=axes,
        literature_evidence=literature_evidence,
        study=study,
    ).model_dump(mode="json")
    manuscript_path = run_dir / "manuscript_scaffold_bound.md" if run_dir else None
    manuscript_text = ""
    if manuscript_path is not None:
        try:
            manuscript_text = manuscript_path.read_text(encoding="utf-8")[
                :_MAX_MANUSCRIPT_PREVIEW
            ]
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            manuscript_text = ""
    claims = _safe_claims(run_dir) if run_dir else []
    figure_gallery = (
        _figure_projection(run_dir)
        if run_dir
        else {
            "kind": "figure_gallery",
            "schema_version": "easyicu.web-pipeline-figure-gallery/1",
            "status": "not_available",
            "figures": [],
            "primary_count": 0,
            "supporting_count": 0,
            "embedded_count": 0,
        }
    )
    result_tables = (
        _table_projection(run_dir)
        if run_dir
        else {
            "schema_version": "easyicu.web-pipeline-result-tables/1",
            "tables": [],
            "table_count": 0,
            "skipped_identifier_tables": 0,
            "preview_policy": "aggregate_tables_only_no_identifier_columns",
        }
    )
    evidence_index = (
        _read_json(run_dir / "evidence" / "evidence_index.json", []) if run_dir else []
    )
    evidence_count = len(evidence_index) if isinstance(evidence_index, list) else 0
    pdf_receipt, manuscript_documents = _validated_manuscript_documents(run_dir)
    pending_requests = []
    if pending is not None:
        pending_requests = [
            {
                "review_id": request.review_id,
                "kind": request.kind,
                "reason_code": _clean_text(
                    request.payload.get("reason")
                    if isinstance(request.payload, Mapping)
                    else None,
                    160,
                ),
                "summary": request.summary,
                "authority_sha256": request.authority_sha256,
                "approval_allowed": (
                    request.payload.get("approval_allowed", True)
                    if isinstance(request.payload, Mapping)
                    else True
                ),
                "review_score": (
                    request.payload.get("review_score")
                    if isinstance(request.payload, Mapping)
                    else None
                ),
                "finding_codes": (
                    list(request.payload.get("finding_codes") or ())[:40]
                    if isinstance(request.payload, Mapping)
                    else []
                ),
            }
            for request in pending.requests
        ]
    source_manifest = {
        "schema_version": "easyicu.web-research-pipeline-projection/1",
        "engine": "easyicu.research_agent.pipeline",
        "run_id": run_id,
        "status": "human_review_pending" if pending is not None else gate["status"],
        "resume_scope": getattr(pending, "resume_scope", None),
        "pending_reviews": pending_requests,
        "plan_approval_allowed": bool(
            pending_requests
            and all(item.get("approval_allowed", True) for item in pending_requests)
        ),
        "scientific_plan_review_status": scientific_plan_review.get("status"),
        "scientific_plan_review_score": scientific_plan_review.get("score"),
        "readiness": axes,
        "scientific_readiness_status": scientific_readiness["status"],
        "evidence_count": evidence_count,
        "result_table_count": result_tables.get("table_count", 0),
        "figure_count": len(figure_gallery.get("figures") or []),
        "manuscript_document_count": len(manuscript_documents),
        "draft_pdf_available": any(
            row.get("name") == "manuscript_scaffold.pdf"
            for row in manuscript_documents
        ),
        "system_validation_report_available": bool(
            run_dir is not None and axes.get("execution_complete")
        ),
        "system_validation_document_count": (
            1 if run_dir is not None and axes.get("execution_complete") else 0
        ),
        "provider": {
            key: provider.get(key)
            for key in (
                "provider",
                "model",
                "client",
                "provider_gate",
                "credential_fingerprint",
            )
            if provider.get(key) is not None
        },
        "path_values_returned": False,
    }
    payloads: Dict[str, Dict[str, Any]] = {
        "run_context.json": run_context,
        "cohort_summary.json": cohort_summary,
        "quality_gate.json": {"gate": gate, "quality": []},
        "agent_plan.json": dict(plan) if isinstance(plan, Mapping) else {},
        "literature_evidence.json": literature_evidence,
        **(
            {"scientific_plan_review.json": scientific_plan_review}
            if scientific_plan_review
            else {}
        ),
        "scientific_readiness.json": scientific_readiness,
        "manuscript_draft.json": {
            "run_id": run_id,
            "status": "locked_pending_human_review",
            "question": run_context["question"],
            "claims": claims,
            "sentences": [],
            "markdown_preview": manuscript_text,
            "source": "research_agent_manuscript_scaffold_bound",
        },
        "figure_gallery.json": figure_gallery,
        "result_tables.json": result_tables,
        "source_run_manifest.json": source_manifest,
    }
    system_figure_gallery = None
    if run_dir is not None and axes.get("execution_complete"):
        system_figure_gallery = _system_validation_figure_gallery(
            wrapper_dir=wrapper_dir,
            run_dir=run_dir,
            plan=plan if isinstance(plan, Mapping) else {},
            figure_gallery=figure_gallery,
        )
        if system_figure_gallery is not None:
            payloads["system_validation_figure_gallery.json"] = system_figure_gallery
    if pdf_receipt is not None:
        payloads["manuscript_pdf_receipt.json"] = pdf_receipt
    system_validation_html: Optional[str] = None
    privacy_scan = _projection_privacy_scan(payloads)
    if privacy_scan["passed"] and run_dir is not None and axes.get("execution_complete"):
        system_report = build_system_validation_report(
            run_id=run_id,
            projections=payloads,
            run_status=_read_json(run_dir / "run_status.json", {}),
            review_checkpoint=_read_json_with_digest(
                run_dir / "human_review_checkpoint.json"
            ),
            provider_usage=_provider_usage_projection(wrapper_dir),
            projection_privacy_passed=True,
        )
        system_report_payload = system_report.model_dump(mode="json")
        system_validation_html = render_system_validation_html(
            system_report,
            figure_gallery=system_figure_gallery or figure_gallery,
        )
        system_validation_receipt = build_system_validation_receipt(
            report_payload=system_report_payload,
            html_bytes=system_validation_html.encode("utf-8"),
        )
        payloads["system_validation_report.json"] = system_report_payload
        payloads["system_validation_report_receipt.json"] = (
            system_validation_receipt
        )
        privacy_scan = _projection_privacy_scan(
            {
                **payloads,
                "system_validation_report.html": system_validation_html,
            }
        )
    if not privacy_scan["passed"]:
        payloads = _privacy_blocked_payloads(
            run_context=run_context,
            scan=privacy_scan,
        )
        gate = dict(payloads["quality_gate.json"]["gate"])
        source_manifest = {
            "schema_version": "easyicu.web-research-pipeline-projection/1",
            "engine": "easyicu.research_agent.pipeline",
            "run_id": run_id,
            "status": "blocked",
            "provider": source_manifest["provider"],
            "path_values_returned": False,
            "projection_withheld": True,
        }
        payloads["source_run_manifest.json"] = source_manifest
        system_validation_html = None
    for name, payload in payloads.items():
        _write_json(wrapper_dir / name, payload)
    # A resumed/reprojected wrapper may predate this pass.  Fixed document names
    # are removed before the newly validated set is copied so a missing, stale,
    # or privacy-withheld receipt can never leave an older PDF reachable.
    for document_name in _RUN_DOCUMENT_SPECS:
        stale = wrapper_dir / document_name
        if stale.exists() or stale.is_symlink():
            stale.unlink()
    document_rows: List[Dict[str, Any]] = []
    if privacy_scan["passed"]:
        for document in manuscript_documents:
            target = wrapper_dir / str(document["name"])
            target.write_bytes(bytes(document["content"]))
            document_rows.append(_artifact_record(target))
        if system_validation_html is not None:
            target = wrapper_dir / "system_validation_report.html"
            target.write_text(system_validation_html, encoding="utf-8")
            document_rows.append(_artifact_record(target))
    artifact_rows = [
        *[_artifact_record(wrapper_dir / name) for name in payloads],
        *document_rows,
    ]
    ledger = {
        "schema_version": "easyicu.web-research-pipeline-ledger/1",
        "run_id": run_id,
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "status": gate["status"],
        "artifacts": artifact_rows,
        "provider": source_manifest["provider"],
        "pipeline_evidence_count": evidence_count,
        "privacy": {
            "patient_rows_in_projection": False,
            "identifier_columns_withheld": (
                result_tables.get("skipped_identifier_tables", 0)
                if privacy_scan["passed"]
                else True
            ),
            "path_values_returned": False,
            "projection_scan_passed": bool(privacy_scan["passed"]),
        },
    }
    _write_json(wrapper_dir / "evidence_ledger.json", ledger)
    artifacts = [*artifact_rows, _artifact_record(wrapper_dir / "evidence_ledger.json")]
    return {
        "run_id": run_id,
        "study_id": run_context["study_id"],
        "mode": "research_agent_pipeline",
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "project_dir": str(wrapper_dir),
        "gate": gate,
        "provider": source_manifest["provider"],
        "artifacts": artifacts,
        "human_review_pending": pending is not None,
        "pending_reviews": pending_requests,
        "uploads": 0,
    }


def _register_pending(entry: _PendingRun) -> None:
    with _PENDING_LOCK:
        _PENDING[str(entry.pending.run_id)] = entry
        while len(_PENDING) > _MAX_PENDING:
            oldest = min(_PENDING.items(), key=lambda item: item[1].created_at)[0]
            # Only evict the live object. Its signed recovery record and paused
            # cumulative Provider ledger remain available after restart/pressure.
            _PENDING.pop(oldest, None)


def _remove_local_recovery(wrapper_dir: Path) -> None:
    remove_recovery_seed(wrapper_dir)
    unregister_pipeline_work_root_if_unused(Path(wrapper_dir).parent.parent)


def _start_web_provider_hard_stop(
    *,
    wrapper_dir: Path,
    job_id: str,
    declaration_sha256: str,
    budget_mode: str = "full_reviewed",
) -> Any:
    """Start the one durable Provider task shared by acquisition and Pipeline."""

    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
    )

    runtime_dir = wrapper_dir / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        runtime_dir.chmod(0o700)
    except OSError:
        pass
    task_id = f"web-{_slug(job_id)}"
    ledger = ProviderHardStopLedger(
        path=(runtime_dir / "provider_hard_stop_ledger.json").resolve(),
        task_ids=(task_id,),
        limits=provider_adapter.web_research_agent_hard_stop_limits(budget_mode),
        batch_id=task_id,
        declaration_sha256=declaration_sha256,
    )
    return ledger.start_task(task_id)


def _finish_web_provider_hard_stop(
    task: Optional[Any],
    *,
    error: Optional[str] = None,
) -> None:
    """Idempotently close one Web task with a closed diagnostic category."""

    if task is None:
        return
    snapshot = task.ledger.snapshot()
    rows = snapshot.get("tasks")
    current = next(
        (
            row
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, Mapping) and row.get("task_id") == task.task_id
        ),
        None,
    )
    if isinstance(current, Mapping) and current.get("status") in {
        "completed",
        "failed",
        "batch_canary_blocked",
        "budget_exhausted",
    }:
        return
    task.finish(error=error)


def pending_review(run_id: Any) -> Optional[Dict[str, Any]]:
    key = _clean_text(run_id, 160)
    with _PENDING_LOCK:
        entry = _PENDING.get(key)
    if entry is None:
        try:
            record = get_review_recovery_record(key)
            if record is None:
                return None
            pending = _checkpoint_pending_from_record(record)
            study = record.study
            credential_source = record.credential_source
            budget_mode = record.budget_mode
            provider_name = _clean_text(record.provider_meta.get("provider"), 64)
        except WebReviewRecoveryError:
            return None
    else:
        pending = entry.pending
        study = entry.study
        credential_source = entry.credential_source
        budget_mode = entry.budget_mode
        provider_name = _clean_text(entry.provider.get("provider"), 64)
    return {
            "run_id": pending.run_id,
            "study_id": _clean_text(study.get("id"), 160),
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
            "resume_scope": pending.resume_scope,
            "credential_source": credential_source,
            "provider": provider_name,
            "budget_mode": budget_mode,
            "resumable_here": bool(pending.resumable_here),
            "requests": [
                {
                    "review_id": request.review_id,
                    "kind": request.kind,
                    "summary": request.summary,
                    "authority_sha256": request.authority_sha256,
                    "reason_code": _clean_text(
                        request.payload.get("reason")
                        if isinstance(request.payload, Mapping)
                        else None,
                        160,
                    ),
                    "approval_allowed": (
                        request.payload.get("approval_allowed", True)
                        if isinstance(request.payload, Mapping)
                        else True
                    ),
                    "review_score": (
                        request.payload.get("review_score")
                        if isinstance(request.payload, Mapping)
                        else None
                    ),
                    "finding_codes": (
                        list(request.payload.get("finding_codes") or ())[:40]
                        if isinstance(request.payload, Mapping)
                        else []
                    ),
                }
                for request in pending.requests
            ],
            "plan_approval_allowed": all(
                (
                    request.payload.get("approval_allowed", True)
                    if isinstance(request.payload, Mapping)
                    else True
                )
                for request in pending.requests
            ),
            "scientific_plan_review": _load_pending_scientific_review(
                Path(pending.run_dir), pending
            ),
        }


def _recover_pending_run(
    run_id: str,
    *,
    provider_environment: Optional[Mapping[str, str]],
    rejection_only: bool = False,
) -> Optional[_PendingRun]:
    """Reconstruct one paused Web run from private, digest-bound coordinates."""

    record = get_review_recovery_record(run_id)
    if record is None:
        return None
    from easyicu.research_agent import ResearchAgentPipeline
    from easyicu.research_agent.authority.provider_hard_stop import (
        ProviderHardStopLedger,
    )
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.services import PipelineServices

    pending = _checkpoint_pending_from_record(
        record,
        recover_recorded_decisions=True,
    )
    wrapper_dir = Path(record.wrapper_dir).resolve()
    if record.pipeline_config_sha256 is None:
        # Historical /1 records predate the exact recovery digest.  They remain
        # readable, but cannot be promoted to durable resume authority: the
        # Pipeline checkpoint only proves a mismatch after reconstruction.
        raise WebReviewRecoveryError(
            "Web review recovery record predates the exact pipeline config digest"
        )
    try:
        config = PipelineConfig.from_recovery_payload(
            record.pipeline_config,
            expected_digest=record.pipeline_config_sha256,
        )
    except ValueError as exc:
        raise WebReviewRecoveryError(
            "Web review pipeline config is inconsistent"
        ) from exc
    if Path(config.workdir).resolve() != (wrapper_dir / "pipeline").resolve():
        raise WebReviewRecoveryError("Web review pipeline workdir drifted")
    if not record.prepared_package_binding:
        raise WebReviewRecoveryError("Web review recovery lacks a package binding")
    if not rejection_only:
        source = record.study.get("data_source")
        source = source if isinstance(source, Mapping) else {}
        try:
            dataio.validate_research_pipeline_source(
                str(source.get("path") or ""),
                database=source.get("database"),
                expected_binding=record.prepared_package_binding,
            )
        except dataio.ExportCohortError as exc:
            raise WebReviewRecoveryError(
                "Web review prepared package changed after the pause"
            ) from exc
    client = None
    provider_public = dict(record.provider_public)
    if not rejection_only:
        client, provider_public = provider_adapter.build_research_agent_provider_client(
            dict(record.provider_meta),
            environ=provider_environment,
        )
        for key in (
            "provider",
            "model",
            "credential_fingerprint",
            "endpoint_fingerprint",
            "credential_header",
            "base_url_endpoint",
            "session_binding_sha256",
        ):
            if provider_public.get(key) != record.provider_public.get(key):
                raise WebReviewRecoveryError(
                    "Web review provider identity changed after the pause"
                )
    task = None
    if not rejection_only:
        task_id = record.hard_stop_task_id
        ledger = ProviderHardStopLedger(
            path=Path(record.hard_stop_ledger_path).resolve(),
            task_ids=(task_id,),
            limits=provider_adapter.web_research_agent_hard_stop_limits(
                record.budget_mode
            ),
            batch_id=task_id,
            declaration_sha256=record.hard_stop_declaration_sha256,
            resume_existing=True,
        )
        # Provider pause/reconcile state belongs to Pipeline.resume_human_review,
        # which performs it while holding the cross-process run execution lock.
        task = ledger.start_task(task_id)
    pipeline = ResearchAgentPipeline.from_config(
        config,
        services=PipelineServices(
            llm=client,
            human_review_gate=_WebHumanReviewGate(),
            provider_hard_stop=task,
        ),
    )
    return _PendingRun(
        pipeline=pipeline,
        pending=pending,
        wrapper_dir=wrapper_dir,
        study=dict(record.study),
        provider=dict(provider_public),
        acquisition=_rehydrate_acquisition_projection(
            record.acquisition_projection
        ),
        created_at=record.created_at,
        credential_source=record.credential_source,
        budget_mode=record.budget_mode,
        prepared_package_binding=dict(record.prepared_package_binding),
        provider_hard_stop=task,
    )


def _compile_plan_revision_contract(
    *,
    study: Mapping[str, Any],
    project_root: Optional[str],
    source_run_id: str,
) -> str:
    """Compile an exact prior review into an Agent-owned revision contract.

    An empty return value is intentional: it means the prior non-approvable
    review contains only study-authority, external-evidence, or independent-
    review findings, so the next attempt must be a fully fresh Plan rather than
    a constrained revision of the old one.
    """

    history = agent_runs.list_run_history(
        study_id=_clean_text(study.get("id"), 160),
        project_root=project_root,
        limit=100,
    )
    source_row = next(
        (
            row
            for row in history.get("runs", [])
            if _clean_text(row.get("run_id"), 160) == source_run_id
        ),
        None,
    )
    if not isinstance(source_row, Mapping):
        raise ResearchPipelineRunError(
            "plan_revision_source_not_found",
            "The exact prior scientific review could not be found.",
        )
    current_digest = study_context_owner.scientific_configuration_sha256(study)
    if _clean_text(source_row.get("scientific_configuration_sha256"), 80) != (
        current_digest
    ):
        raise ResearchPipelineRunError(
            "plan_revision_source_configuration_superseded",
            "The scientific setup changed after the reviewed plan; its repair contract cannot be reused.",
        )
    source_review = agent_runs.read_run_review(
        str(source_row.get("project_dir") or "")
    )
    review_payload = (
        (source_review.get("artifact_payloads") or {}).get(
            "scientific_plan_review.json"
        )
        if source_review.get("ok")
        else None
    )
    if not isinstance(review_payload, Mapping):
        raise ResearchPipelineRunError(
            "plan_revision_scientific_review_missing",
            "The prior run has no digest-verified scientific review to compile.",
        )
    parsed_review = PlanScientificReview.model_validate(review_payload)
    if parsed_review.approval_allowed:
        raise ResearchPipelineRunError(
            "plan_revision_source_not_changes_required",
            "Only a non-approvable scientific review may seed a fresh plan revision.",
        )
    return render_agent_plan_revision_contract(parsed_review)


def _validated_pipeline_credential_source(
    credential_source: str,
    *,
    provider: Mapping[str, Any],
) -> str:
    """Bind one Web credential source to the matching provider family."""

    selected = str(credential_source or "").strip().lower()
    if selected not in {"pi_verified", "codex_user_auth"}:
        raise ResearchPipelineRunError(
            "research_pipeline_credential_source_invalid",
            "Choose one server-verified Research Agent credential source.",
        )
    provider_name = _clean_text(provider.get("provider"), 64).lower()
    account_provider = provider_adapter.is_user_account_provider(provider_name)
    if selected == "codex_user_auth" and not account_provider:
        raise ResearchPipelineRunError(
            "research_pipeline_codex_user_auth_provider_required",
            "Codex user authentication requires the reviewed Codex account provider.",
        )
    if selected == "pi_verified" and account_provider:
        raise ResearchPipelineRunError(
            "research_pipeline_codex_user_auth_required",
            "The Codex account provider requires this browser user's ChatGPT login.",
        )
    return selected


def make_research_pipeline_run_runner(
    *,
    export_path: str,
    study_context: Mapping[str, Any],
    project_root: Optional[str],
    provider: Mapping[str, Any],
    provider_environment: Optional[Mapping[str, str]] = None,
    credential_source: str = "pi_verified",
    literature_search_authorized: bool = False,
    plan_revision_source_run_id: str = "",
    budget_mode: str = "planner_canary",
    runner_image: Optional[str] = None,
) -> Any:
    """Build the JobManager runner for a real, evidence-bound pipeline run."""

    study = dict(study_context)
    question = _clean_text(study.get("question"), 1_200)
    if not question:
        raise ResearchPipelineRunError(
            "research_pipeline_question_required",
            "A scientific question is required before starting the pipeline.",
        )
    source = study.get("data_source")
    if not isinstance(source, Mapping):
        raise ResearchPipelineRunError(
            "research_pipeline_database_required",
            "A typed ICU database is required before pipeline launch.",
        )
    database_raw = _clean_text(source.get("database"), 64)
    if not database_raw:
        raise ResearchPipelineRunError(
            "research_pipeline_database_required",
            "A typed ICU database is required before pipeline launch.",
        )
    try:
        database = normalize_database_key(database_raw)
    except KeyError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_database_unknown",
            "The configured ICU database is not supported.",
            details={"database": database_raw},
        ) from exc
    target = _target_outcome(study)
    primary_exposure = _primary_exposure(study)
    covariates = _configured_covariates(study)
    sensitivity_specs = _configured_sensitivity_specs(study)
    window = _cohort_window(study)
    _validate_primary_concept_selection(study, primary_exposure)
    validated_analysis_design = _validate_analysis_design(study)
    patient_grouping = (
        _patient_grouping_for_analysis_design(study)
        if validated_analysis_design.get("variance_estimator") == "cluster_robust"
        else None
    )
    # Compile and validate the display-to-execution boundary before JobManager
    # creates a background job. A prose label or stale concept therefore fails
    # at its owner instead of appearing later as an opaque pipeline crash.
    foundation_profile = _data_foundation_profile(
        export_path=export_path,
        study=study,
        target=target,
        primary_exposure=primary_exposure,
        covariates=covariates,
        sensitivity_specs=sensitivity_specs,
    )
    try:
        package_receipt = dataio.validate_research_pipeline_source(
            export_path,
            database=database,
        )
    except dataio.ExportCohortError as exc:
        raise ResearchPipelineRunError(
            str(exc.detail.get("error") or "research_pipeline_source_invalid"),
            "The Research Agent requires a manifest-backed prepared data package.",
            details={
                key: value for key, value in exc.detail.items() if key != "error"
            },
        ) from exc
    prepared_package_binding = dict(package_receipt["binding"])
    selected_budget_mode = str(budget_mode or "").strip().lower()
    try:
        provider_adapter.web_research_agent_hard_stop_limits(selected_budget_mode)
    except ValueError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_budget_mode_invalid",
            "Choose an explicit reviewed Research Agent budget mode.",
        ) from exc
    selected_credential_source = _validated_pipeline_credential_source(
        credential_source,
        provider=provider,
    )
    if provider_environment is None:
        raise ResearchPipelineRunError(
            (
                "research_pipeline_codex_user_auth_required"
                if selected_credential_source == "codex_user_auth"
                else "research_pipeline_pi_verified_credentials_required"
            ),
            "A verified provider credential is required for this pipeline.",
        )
    if not project_root:
        raise ResearchPipelineRunError(
            "research_pipeline_project_workspace_required",
            "A server-owned Pi project workspace is required for this pipeline.",
        )
    capability_settings = capability_policy.capability_settings()
    publication_skill_flags = publication_skill_flags_from_settings(
        capability_settings
    )
    try:
        extension_registry = ExtensionRegistry()
        extension_snapshot = extension_registry.snapshot()
        if not bool(capability_settings.get("mcp_tools_enabled", False)):
            extension_snapshot = ExtensionActivationSnapshot.build(
                revision=extension_snapshot.revision,
                skills=extension_snapshot.skills,
                mcp_servers=(),
            )
        user_extension_activation = extension_registry.pipeline_activation(
            extension_snapshot
        )
    except ExtensionRegistryError as exc:
        raise ResearchPipelineRunError(exc.code, exc.message, details=exc.details) from exc
    research_provider_environment = dict(provider_environment)
    source_run_id = _clean_text(plan_revision_source_run_id, 160)
    selected_runner_image = str(runner_image or "").strip()
    if runner_image is not None and not selected_runner_image:
        raise ResearchPipelineRunError(
            "research_pipeline_runner_image_invalid",
            "The server-owned runner image cannot be empty.",
        )

    def runner(job: Any) -> Dict[str, Any]:
        try:
            dataio.validate_research_pipeline_source(
                export_path,
                database=database,
                expected_binding=prepared_package_binding,
            )
        except dataio.ExportCohortError as exc:
            raise ResearchPipelineRunError(
                str(exc.detail.get("error") or "research_pipeline_source_invalid"),
                "The prepared data package changed after launch validation.",
            ) from exc
        root = Path(project_root).expanduser().resolve()
        wrapper_dir = root / _slug(study.get("id")) / f"run_{job.id}"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        bound_plan_revision_contract = ""
        if source_run_id:
            bound_plan_revision_contract = _compile_plan_revision_contract(
                study=study,
                project_root=project_root,
                source_run_id=source_run_id,
            )
        _progress(job, step="provider", label="Research Agent provider authorized")
        client, provider_public = provider_adapter.build_research_agent_provider_client(
            dict(provider),
            environ=research_provider_environment,
        )
        provider_hard_stop = None
        try:
            from easyicu.research_agent import ResearchAgentPipeline
            from easyicu.research_agent.acquisition.foundation import (
                acquire_universe_for_question,
            )
            from easyicu.research_agent.execution.runner import DockerRunner
            from easyicu.research_agent.orchestration.config import PipelineConfig
            from easyicu.research_agent.orchestration.profiles import (
                get_submission_profile,
            )
            from easyicu.research_agent.orchestration.services import PipelineServices
            from easyicu.research_agent.orchestration.workflow import HumanReviewPending
            from easyicu.research_agent.providers.hard_stop import HardStopClient

            provider_hard_stop = _start_web_provider_hard_stop(
                wrapper_dir=wrapper_dir,
                job_id=str(job.id),
                declaration_sha256=(
                    study_context_owner.scientific_configuration_sha256(study)
                ),
                budget_mode=selected_budget_mode,
            )
            provider_public["provider_hard_stop"] = {
                "schema_version": "easyicu.web-provider-hard-stop-policy/1",
                "required": True,
                "enforced": True,
                "scope": "acquisition_through_pipeline_terminal",
            }
            hard_stop_limits = provider_hard_stop.ledger.limits
            acquisition_client = HardStopClient(
                client,
                role="acquisition",
                task=provider_hard_stop,
            )

            _progress(
                job,
                step="data_foundation",
                label="Selecting concepts and materializing a typed analysis universe",
            )
            acquisition = acquire_universe_for_question(
                export_dir=Path(export_path).expanduser(),
                question=question,
                llm=acquisition_client,
                output_dir=wrapper_dir / "pipeline_input",
                stem="web_research_universe",
                target_outcome=target,
                primary_exposure_concept=(
                    foundation_profile.get("primary_exposure_source_concept")
                    or primary_exposure
                ),
                outcome_concepts=foundation_profile["outcome_concepts"],
                required_feature_concepts=foundation_profile[
                    "required_feature_concepts"
                ],
                static_concepts=foundation_profile["static_concepts"],
                allowed_modules=foundation_profile["allowed_modules"],
                cohort_window=window,
                database=database,
                require_outcome=foundation_profile["require_outcome"],
                # A verified composite patient/stay identity is needed by the
                # cluster-robust model.  The current materializer deliberately
                # refuses to attach a private mapping to a longitudinal table;
                # this fixed stay-level design therefore requests no unused
                # trajectory instead of silently publishing an ungrouped one.
                emit_trajectory=patient_grouping is None,
                patient_grouping=patient_grouping,
            )
            if acquisition.blocked or acquisition.universe_path is None:
                _finish_web_provider_hard_stop(
                    provider_hard_stop,
                    error="data_foundation_blocked",
                )
                return _write_projection(
                    wrapper_dir=wrapper_dir,
                    study=study,
                    provider=provider_public,
                    acquisition=acquisition,
                    run_dir=None,
                    blocked_reason="data_foundation_blocked",
                )
            resolved_primary_exposure = primary_exposure
            if primary_exposure:
                resolved_primary_exposure = _resolve_materialized_primary_exposure(
                    configured=primary_exposure,
                    source_concept=foundation_profile.get(
                        "primary_exposure_source_concept"
                    ),
                    acquisition=acquisition,
                )
                if not resolved_primary_exposure:
                    raise ResearchPipelineRunError(
                        "research_pipeline_primary_exposure_aggregation_required",
                        "The configured primary exposure requires an explicit "
                        "analysis aggregation before planning can start.",
                    )
            try:
                bound_preplan_literature = idea_mining.load_bound_prior_art_literature(
                    dict(study.get("idea_handoff") or {}),
                    research_question=question,
                )
            except idea_mining.IdeaMiningWebError as exc:
                detail = exc.detail
                raise ResearchPipelineRunError(
                    str(detail.get("error") or "prior_art_binding_invalid"),
                    str(
                        detail.get("reason")
                        or "The accepted Idea Mining literature receipt is invalid."
                    ),
                ) from exc
            if bound_preplan_literature is None:
                try:
                    bound_preplan_literature = (
                        literature_authority.load_bound_literature(
                            study=study,
                            research_question=question,
                        )
                    )
                except literature_authority.LiteratureAuthorityError as exc:
                    raise ResearchPipelineRunError(exc.code, exc.message) from exc
            submission_profile_ref = (
                "npj_dm_e1_demo_dev/20260815"
                if selected_budget_mode == "full_reviewed"
                else "npj_dm_e1_canary_dev/20260814"
            )
            submission_profile = get_submission_profile(submission_profile_ref)
            profile_options = submission_profile.pipeline_options()
            profile_options.update(
                {
                    "enable_memory": False,
                    "enable_experience_bank": False,
                    "enable_reviewed_memory": False,
                    "reviewed_memory_namespaces": (),
                }
            )
            config = PipelineConfig(
                workdir=wrapper_dir / "pipeline",
                enable_publication_figure_skill=publication_skill_flags[
                    "nature_figure_enabled"
                ],
                enable_nature_writing_skill=publication_skill_flags[
                    "nature_writing_enabled"
                ],
                extension_activation=user_extension_activation,
                # Strict manuscript enforcement can only bind the host's
                # typed ScientificClaim placeholders when the writer receives
                # the claim-aware v2 evidence digest.  The primary-only v1
                # digest omits that authority and would make an otherwise
                # valid Web analysis fail at the writing boundary.
                require_human_plan_review=True,
                require_reportable_scientific_capability=True,
                required_primary_cohort_selection_mode=(
                    _primary_cohort_selection_mode(study)
                ),
                enable_pdf_render=True,
                latex_draft_watermark=True,
                bound_preplan_literature=bound_preplan_literature,
                bound_plan_revision_contract=(
                    bound_plan_revision_contract or None
                ),
                # The Web host carries this from the user's turn grant.  When
                # an accepted Idea handoff already supplies a digest-bound
                # search receipt, reuse it and do not silently issue a second
                # search.  Otherwise a full Web study can now establish dated
                # direct prior art before the Planner runs.
                enable_pubmed=(
                    bool(literature_search_authorized)
                    and bound_preplan_literature is None
                ),
                runner_kind=submission_profile.requires_runner,
                runner_image=(selected_runner_image or DockerRunner.DEFAULT_IMAGE),
                runner_network="none",
                max_provider_attempts_per_run=(
                    hard_stop_limits.max_provider_attempts_per_run
                ),
                max_provider_attempts_per_batch=(
                    hard_stop_limits.max_provider_attempts_per_batch
                ),
                max_total_tokens_per_run=hard_stop_limits.max_total_tokens_per_run,
                max_total_tokens_per_batch=(
                    hard_stop_limits.max_total_tokens_per_batch
                ),
                max_estimated_cost_usd_per_batch=(
                    hard_stop_limits.max_estimated_cost_usd_per_batch
                ),
                max_wall_clock_seconds_per_task=(
                    hard_stop_limits.max_wall_clock_seconds_per_task
                ),
                provider_input_cost_usd_per_million_tokens=(
                    hard_stop_limits.input_cost_usd_per_million_tokens
                ),
                provider_output_cost_usd_per_million_tokens=(
                    hard_stop_limits.output_cost_usd_per_million_tokens
                ),
                **profile_options,
            )
            pipeline = ResearchAgentPipeline.from_config(
                config,
                services=PipelineServices(
                    llm=client,
                    human_review_gate=_WebHumanReviewGate(),
                    provider_hard_stop=provider_hard_stop,
                ),
            )
            try:
                config_payload = config.recovery_payload()
                PipelineConfig.from_recovery_payload(
                    config_payload,
                    expected_digest=config.canonical_digest(),
                )
            except ValueError as exc:
                raise ResearchPipelineRunError(
                    "research_pipeline_review_config_not_recoverable",
                    "The run configuration cannot be reconstructed safely.",
                ) from exc
            recovery_seed = WebReviewRecoverySeed.create(
                wrapper_dir=str(wrapper_dir.resolve()),
                study=study,
                scientific_configuration_sha256=(
                    study_context_owner.scientific_configuration_sha256(study)
                ),
                provider_meta=dict(provider),
                provider_public=dict(provider_public),
                credential_source=selected_credential_source,
                budget_mode=selected_budget_mode,
                prepared_package_binding=prepared_package_binding,
                pipeline_config=config_payload,
                pipeline_config_sha256=config.canonical_digest(),
                acquisition_projection=_acquisition_recovery_projection(acquisition),
                hard_stop_ledger_path=str(provider_hard_stop.ledger.path.resolve()),
                hard_stop_task_id=str(provider_hard_stop.task_id),
                hard_stop_declaration_sha256=(
                    study_context_owner.scientific_configuration_sha256(study)
                ),
                created_at=time.time(),
            )
            # This local seed precedes Planner execution. If the process dies
            # after the pipeline checkpoint but before the global index update,
            # bounded reconciliation can still discover the exact pause.
            register_pipeline_work_root(root)
            put_recovery_seed(recovery_seed)
            preferences = _research_user_preferences(
                study,
                patient_grouping=patient_grouping,
            )
            _progress(
                job,
                step="research_pipeline",
                label=(
                    "Research Agent planning started; execution remains blocked "
                    "pending human plan review"
                ),
            )
            outcome = pipeline.run(
                question=question,
                cohort=acquisition.universe_path,
                cohort_authority_path=acquisition.cohort_authority_path,
                cohort_authority_ref=(
                    acquisition.cohort_authority_ref.to_dict()
                    if acquisition.cohort_authority_ref is not None
                    else None
                ),
                trajectory_path=acquisition.trajectory_path,
                trajectory_authority_path=acquisition.trajectory_authority_path,
                trajectory_authority_ref=(
                    acquisition.trajectory_authority_ref.to_dict()
                    if acquisition.trajectory_authority_ref is not None
                    else None
                ),
                cohort_name=f"web_{_slug(study.get('id'))}",
                database=database,
                target_outcome=target,
                endpoint=acquisition.endpoint,
                primary_exposure=resolved_primary_exposure,
                inclusion_criteria=_inclusion_criteria(study),
                id_columns=(
                    [patient_grouping.output_identity_column]
                    if patient_grouping is not None
                    else None
                ),
                concept_descriptions=(
                    {
                        patient_grouping.output_identity_column: (
                            "Host-verified unique ICU-stay identity. Derive the "
                            "patient cluster only from the prefix before ':s'; "
                            "never report identifier values."
                        )
                    }
                    if patient_grouping is not None
                    else None
                ),
                user_preferences=preferences,
                notes=_clean_text(study.get("analysis_goal"), 1_200) or None,
                progress_callback=lambda event: _pipeline_progress(job, event),
            )
            if isinstance(outcome, HumanReviewPending):
                run_dir = Path(outcome.run_dir)
                put_review_recovery_record(
                    recovery_seed.record(str(outcome.run_id))
                )
                entry = _PendingRun(
                    pipeline=pipeline,
                    pending=outcome,
                    wrapper_dir=wrapper_dir,
                    study=study,
                    provider=provider_public,
                    acquisition=acquisition,
                    created_at=time.time(),
                    credential_source=selected_credential_source,
                    budget_mode=selected_budget_mode,
                    prepared_package_binding=prepared_package_binding,
                    provider_hard_stop=provider_hard_stop,
                )
                _register_pending(entry)
                return _write_projection(
                    wrapper_dir=wrapper_dir,
                    study=study,
                    provider=provider_public,
                    acquisition=acquisition,
                    run_dir=run_dir,
                    pending=outcome,
                )
            _finish_web_provider_hard_stop(provider_hard_stop)
            _remove_local_recovery(wrapper_dir)
            return _write_projection(
                wrapper_dir=wrapper_dir,
                study=study,
                provider=provider_public,
                acquisition=acquisition,
                run_dir=Path(outcome.manifest_path).parent,
            )
        except ResearchPipelineRunError:
            _finish_web_provider_hard_stop(
                provider_hard_stop,
                error="research_pipeline_error",
            )
            raise
        except Exception as exc:
            _finish_web_provider_hard_stop(
                provider_hard_stop,
                error=_pipeline_failure_category(exc),
            )
            code = _pipeline_failure_code(exc)
            diagnostic = _write_pipeline_failure_diagnostic(
                wrapper_dir=wrapper_dir,
                exc=exc,
                code=code,
            )
            if code == "research_pipeline_provider_timeout":
                _progress(
                    job,
                    step="planning",
                    label=(
                        "The model provider timed out; the run stopped without "
                        "approving a plan or starting analysis."
                    ),
                    status="error",
                )
                message = (
                    "The configured model provider timed out while the Research "
                    "Agent was generating a contract-valid plan. No analysis was run."
                )
            elif code == "research_pipeline_plan_contract_exhausted":
                message = (
                    "The model used every bounded planning attempt without producing "
                    "a contract-valid plan. No analysis was run."
                )
            else:
                message = (
                    "The Research Agent pipeline stopped before it could produce a "
                    f"governed result ({_pipeline_failure_category(exc)})."
                )
            raise ResearchPipelineRunError(
                code,
                message,
                details={
                    "failure_type": _pipeline_failure_category(exc),
                    "diagnostic": diagnostic,
                },
            ) from exc

    return runner


def resume_research_pipeline(
    *,
    run_id: str,
    study_context_id: str,
    decision: str,
    reviewer: str,
    note: str,
    job: Any,
    current_study_context: Optional[Mapping[str, Any]] = None,
    provider_environment: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Resume one digest-bound plan review, including after host restart."""

    key = _clean_text(run_id, 160)
    resolved = str(decision or "").strip().lower()
    if resolved not in {"approved", "rejected"}:
        raise ResearchPipelineRunError(
            "research_pipeline_review_decision_invalid",
            "Choose approved or rejected for the pending plan review.",
        )
    with _PENDING_LOCK:
        entry = _PENDING.get(key)
    if entry is None:
        try:
            entry = _recover_pending_run(
                key,
                provider_environment=provider_environment,
                rejection_only=resolved == "rejected",
            )
        except Exception as exc:
            raise ResearchPipelineRunError(
                "research_pipeline_review_recovery_failed",
                "The saved plan review could not be restored safely.",
                details={"failure_type": _pipeline_failure_category(exc)},
            ) from exc
        if entry is None:
            raise ResearchPipelineRunError(
                "research_pipeline_review_not_resumable",
                "No saved plan review exists for this run.",
            )
        with _PENDING_LOCK:
            incumbent = _PENDING.setdefault(key, entry)
            entry = incumbent
    if _clean_text(entry.study.get("id"), 160) != _clean_text(study_context_id, 160):
        raise ResearchPipelineRunError(
            "research_pipeline_review_study_mismatch",
            "The pending review belongs to a different research project.",
        )
    if resolved == "approved" and current_study_context is not None:
        planned_digest = study_context_owner.scientific_configuration_sha256(
            entry.study
        )
        current_digest = study_context_owner.scientific_configuration_sha256(
            dict(current_study_context)
        )
        if current_digest != planned_digest:
            with _PENDING_LOCK:
                _PENDING.pop(key, None)
            remove_review_recovery_record(key)
            _remove_local_recovery(entry.wrapper_dir)
            _finish_web_provider_hard_stop(
                entry.provider_hard_stop,
                error="configuration_superseded",
            )
            raise ResearchPipelineRunError(
                "research_pipeline_review_configuration_superseded",
                "The scientific setup changed after this plan was generated; the old plan cannot be approved.",
                details={
                    "planned_scientific_configuration_sha256": planned_digest,
                    "current_scientific_configuration_sha256": current_digest,
                },
            )
    if resolved == "approved" and entry.budget_mode == "planner_canary":
        raise ResearchPipelineRunError(
            "research_pipeline_planner_canary_execution_blocked",
            "A Planner-only canary cannot approve or execute its proposed plan.",
        )
    if resolved == "approved":
        source = entry.study.get("data_source")
        source = source if isinstance(source, Mapping) else {}
        if not entry.prepared_package_binding:
            raise ResearchPipelineRunError(
                "research_pipeline_package_binding_missing",
                "The paused run has no exact prepared-package authority.",
            )
        try:
            dataio.validate_research_pipeline_source(
                str(source.get("path") or ""),
                database=source.get("database"),
                expected_binding=entry.prepared_package_binding,
            )
        except dataio.ExportCohortError as exc:
            raise ResearchPipelineRunError(
                str(
                    exc.detail.get("error")
                    or "research_pipeline_package_binding_changed"
                ),
                "The prepared package changed after planning and cannot be approved.",
            ) from exc
    stored_decisions: List[Dict[str, Any]] = []
    checkpoint_file = Path(entry.pending.run_dir) / "human_review_checkpoint.json"
    if checkpoint_file.is_file():
        from easyicu.research_agent.orchestration.human_review_checkpoint import (
            load_checkpoint,
        )

        checkpoint = load_checkpoint(checkpoint_file, require_pending=False)
        if checkpoint.state not in {
            "pending",
            "approved_pending_execution",
            "executing",
        }:
            raise ResearchPipelineRunError(
                "research_pipeline_review_checkpoint_not_resumable",
                f"The saved review is in non-resumable phase {checkpoint.state!r}.",
            )
        stored_decisions = [dict(item) for item in checkpoint.approved_decisions]
    if stored_decisions:
        stored_kinds = {str(item.get("decision") or "") for item in stored_decisions}
        if stored_kinds != {resolved}:
            raise ResearchPipelineRunError(
                "research_pipeline_review_decision_already_recorded",
                "This review already has a different durable decision.",
            )
        decisions = stored_decisions
    else:
        decided_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        decisions = [
            {
                "review_id": request.review_id,
                "authority_sha256": request.authority_sha256,
                "decision": resolved,
                "reviewer": _clean_text(reviewer, 200) or "local_web_reviewer",
                "decided_at": decided_at,
                "note": _clean_text(note, 1_000),
            }
            for request in entry.pending.requests
        ]
    _progress(
        job,
        step="human_review",
        label=f"Applying {resolved} decision to the digest-bound Research Agent plan",
    )
    from easyicu.research_agent.orchestration.workflow import (
        HumanReviewPending,
        HumanReviewRejected,
    )

    # Lease the live pause before resuming it. Two approval jobs for the same
    # button click must not both drive one workflow or pause the Provider clock
    # underneath each other. Recoverable failures and a new pause explicitly
    # re-register the entry below.
    with _PENDING_LOCK:
        if _PENDING.get(key) is not entry:
            raise ResearchPipelineRunError(
                "research_pipeline_review_resume_in_progress",
                "This plan review is already being resumed by another job.",
            )
        _PENDING.pop(key, None)

    try:
        outcome = entry.pipeline.resume_human_review(
            decisions,
            run_id=key,
            progress_callback=lambda event: _pipeline_progress(job, event),
        )
    except HumanReviewRejected:
        remove_review_recovery_record(key)
        _remove_local_recovery(entry.wrapper_dir)
        _finish_web_provider_hard_stop(
            entry.provider_hard_stop,
            error="human_review_rejected",
        )
        return _write_projection(
            wrapper_dir=entry.wrapper_dir,
            study=entry.study,
            provider=entry.provider,
            acquisition=entry.acquisition,
            run_dir=Path(entry.pending.run_dir),
            blocked_reason="human_plan_review_rejected",
        )
    except Exception as exc:
        remains_resumable = bool(
            getattr(entry.pipeline, "has_resumable_human_review", False)
        )
        if remains_resumable:
            _register_pending(entry)
        if not remains_resumable:
            _finish_web_provider_hard_stop(
                entry.provider_hard_stop,
                error=_pipeline_failure_category(exc),
            )
            remove_review_recovery_record(key)
            _remove_local_recovery(entry.wrapper_dir)
        raise ResearchPipelineRunError(
            "research_pipeline_review_resume_failed",
            "The governed Research Agent run could not resume after plan review.",
            details={
                "failure_type": _pipeline_failure_category(exc),
                "review_resumable": remains_resumable,
            },
        ) from exc
    if isinstance(outcome, HumanReviewPending):
        entry.pending = outcome
        _register_pending(entry)
        return _write_projection(
            wrapper_dir=entry.wrapper_dir,
            study=entry.study,
            provider=entry.provider,
            acquisition=entry.acquisition,
            run_dir=Path(outcome.run_dir),
            pending=outcome,
        )
    _finish_web_provider_hard_stop(entry.provider_hard_stop)
    remove_review_recovery_record(key)
    _remove_local_recovery(entry.wrapper_dir)
    return _write_projection(
        wrapper_dir=entry.wrapper_dir,
        study=entry.study,
        provider=entry.provider,
        acquisition=entry.acquisition,
        run_dir=Path(outcome.manifest_path).parent,
    )


def refresh_literature_evidence_projection(wrapper_dir: Path) -> Dict[str, Any]:
    """Backfill the path-free literature projection for one existing Web run.

    This is a development migration for runs produced before the projection was
    added.  The pipeline's fixed run id selects the only source directory; no
    model- or browser-supplied path participates.
    """

    root = Path(wrapper_dir).expanduser().resolve()
    context = _read_json(root / "run_context.json", {})
    run_id = _clean_text(context.get("run_id"), 160)
    plan = _read_json(root / "agent_plan.json", {})
    if not run_id:
        raise ResearchPipelineRunError(
            "research_pipeline_literature_run_id_missing",
            "The Web projection has no pipeline run id.",
        )
    pipeline_run = (root / "pipeline" / run_id).resolve()
    try:
        pipeline_run.relative_to((root / "pipeline").resolve())
    except ValueError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_literature_path_invalid",
            "The pipeline literature source is outside the bound run.",
        ) from exc
    payload = load_run_literature_projection(
        run_dir=pipeline_run,
        run_id=run_id,
        plan=plan if isinstance(plan, Mapping) else {},
    )
    if not _projection_privacy_scan({"literature_evidence.json": payload})["passed"]:
        raise ResearchPipelineRunError(
            "research_pipeline_literature_projection_privacy_failed",
            "The literature projection failed the Web privacy boundary.",
        )
    target = root / "literature_evidence.json"
    _write_json(target, payload)
    ledger_path = root / "evidence_ledger.json"
    ledger = _read_json(ledger_path, {})
    if isinstance(ledger, dict):
        artifacts = [
            row
            for row in list(ledger.get("artifacts") or [])
            if isinstance(row, Mapping)
            and row.get("name") != "literature_evidence.json"
        ]
        artifacts.append(_artifact_record(target))
        ledger["artifacts"] = artifacts
        _write_json(ledger_path, ledger)
    return payload


__all__ = [
    "ResearchPipelineRunError",
    "make_research_pipeline_run_runner",
    "pending_review",
    "refresh_literature_evidence_projection",
    "resume_research_pipeline",
    "validate_analysis_design_for_execution",
]
