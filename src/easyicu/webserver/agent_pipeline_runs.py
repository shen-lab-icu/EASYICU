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
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import ValidationError

from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
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
from easyicu.research_agent.planning.progressive_artifacts import (
    ProgressivePlanningArtifactError,
    load_progressive_planner_checkpoint_chain,
)
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.schema import TimeWindow
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
    dataio,
    literature_authority,
    primary_cohort,
    provider_adapter,
    run_artifact_disclosure,
    source_identity_authority,
)
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.study_scientific_configuration import (
    ScientificConfiguration,
    ScientificConfigurationError,
)
from easyicu.webserver.ideas import mining as idea_mining
from easyicu.webserver.literature_projection import (
    load_current_plan_authority,
    load_run_literature_projection,
)
from easyicu.webserver.scientific_readiness_projection import (
    build_scientific_readiness_projection,
)
from easyicu.webserver.figure_presentation import verified_presentation_gallery
from easyicu.webserver.research_evidence_preview import is_identifier_column
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.research_pipeline_run_preparation import (
    ResearchPipelineLaunchRequest,
    prepare_research_pipeline_run,
)
from easyicu.webserver.pi_copilot.contracts import (
    EXECUTION_RETRY_REPLAYABLE_GATE_REASONS,
)
from easyicu.webserver.agent_review_recovery import (
    PendingReviewEntry as _PendingRun,
    PendingReviewRegistry,
    PendingReviewResumeFailure,
    PendingReviewResumeInProgress,
    WebReviewRecoveryError,
    WebReviewRecoverySeed,
    get_record as get_review_recovery_record,
    pending_from_record,
    put_record as put_review_recovery_record,
    put_recovery_seed,
    recover_pending_review,
    register_pipeline_work_root,
    remove_record as remove_review_recovery_record,
    remove_recovery_seed,
    resume_pending_review,
    unregister_pipeline_work_root_if_unused,
)

_MAX_JSON_BYTES = 2 * 1024 * 1024
_DEVELOPMENT_PROVIDER_REQUEST_TIMEOUT_SECONDS = 120.0
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

_SAFE_PIPELINE_EXCEPTION_TYPES = frozenset(
    {
        "CodexAppServerError",
        "ExecutionRuntimeUnavailableError",
        "PlannerEfficiencyBudgetExhausted",
        "ProgressivePlanCompileError",
        "ResearchPipelineRunError",
        "StructuredResponseFailure",
    }
)


@dataclass(frozen=True)
class _DevelopmentResumeAcquisition:
    """Verified server-owned acquisition authority for Planner replay."""

    kind: str
    feature_concepts: tuple[str, ...] = ()
    outcome_concepts: tuple[str, ...] = ()
    static_concepts: tuple[str, ...] = ()
    selected_concepts: tuple[str, ...] = ()
    universe_path: Optional[Path] = None
    provenance_path: Optional[Path] = None
    universe_sha256: str = ""
    provenance_sha256: str = ""


@dataclass(frozen=True)
class _WebHumanReviewGate:
    """Server-owned reviewer identity for the local durable review route."""

    def reviewer_identity_resolver(self) -> str:
        return server_reviewer_identity()


def server_reviewer_identity() -> str:
    """Return the host-authenticated identity used by the local Web service."""

    return "easyicu_local_web_operator"


_PENDING_REVIEWS = PendingReviewRegistry(max_entries=16)


def _acquisition_recovery_projection(acquisition: Any) -> Dict[str, Any]:
    selection = getattr(acquisition, "selection", None)
    coverage = getattr(acquisition, "coverage", None)
    return {
        "selected_concepts": list(getattr(selection, "selected_concepts", ()) or ())[
            :64
        ],
        "materialized_concepts": list(
            getattr(acquisition, "materialized_concepts", ()) or ()
        )[:128],
        "coverage_sufficient": bool(getattr(coverage, "sufficient", False)),
    }


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "study")).strip("-.")
    return text[:96] or "study"


def _development_progressive_resume_binding(
    *,
    project_root: str,
    study_id: str,
    source_job_id: str,
    budget_mode: str,
    checkpoint_sequence: str | int | None = None,
) -> tuple[Path, str]:
    """Resolve one server-owned Dev checkpoint without accepting client paths."""

    selected_job = str(source_job_id or "").strip()
    if budget_mode not in {"planner_canary", "full_reviewed"}:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_budget_invalid",
            "Development Planner resume requires a reviewed development budget.",
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", selected_job):
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_job_invalid",
            "Choose one valid prior canary job from this study workspace.",
        )
    root = Path(project_root).expanduser().resolve()
    study_root = root / _slug(study_id)
    wrapper = study_root / f"run_{selected_job}"
    pipeline_root = wrapper / "pipeline"
    if wrapper.is_symlink() or pipeline_root.is_symlink():
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_source_invalid",
            "The prior canary checkpoint source is not a regular owned workspace.",
        )
    try:
        resolved_pipeline = pipeline_root.resolve(strict=True)
        resolved_pipeline.relative_to(study_root.resolve())
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_source_missing",
            "The prior canary checkpoint is unavailable in this study workspace.",
        ) from exc
    run_dirs = sorted(
        path
        for path in resolved_pipeline.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.startswith("run_")
    )
    if len(run_dirs) != 1:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_source_ambiguous",
            "The prior canary does not identify exactly one pipeline run.",
        )
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dirs[0].iterdir():
        match = re.fullmatch(
            r"progressive_planner_checkpoint_([0-9]{3})\.json",
            path.name,
        )
        if match and path.is_file() and not path.is_symlink():
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_checkpoint_missing",
            "The prior canary has no validated Progressive Planner checkpoint.",
        )
    selected_sequence_text = (
        "" if checkpoint_sequence is None else str(checkpoint_sequence).strip()
    )
    if selected_sequence_text:
        if not re.fullmatch(r"(?:0|[1-9][0-9]{0,2})", selected_sequence_text):
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_sequence_invalid",
                "The server-selected development checkpoint sequence is invalid.",
            )
        selected_sequence = int(selected_sequence_text)
        selected = [
            path for sequence, path in checkpoints if sequence == selected_sequence
        ]
        if len(selected) != 1:
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_sequence_missing",
                "The selected development checkpoint is unavailable in the "
                "prior canary.",
            )
        terminal = selected[0]
    else:
        terminal = max(checkpoints, key=lambda item: item[0])[1]
    try:
        raw = terminal.read_bytes()
    except OSError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_checkpoint_unreadable",
            "The prior canary checkpoint cannot be read safely.",
        ) from exc
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        load_progressive_planner_checkpoint_chain(
            last_checkpoint_path=terminal,
            expected_artifact_sha256=artifact_sha256,
        )
    except ProgressivePlanningArtifactError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_checkpoint_invalid",
            "The prior canary checkpoint chain did not pass integrity validation.",
            details={"reason_code": exc.reason_code},
        ) from exc
    return terminal, artifact_sha256


def _development_resume_literature_bundle(*, checkpoint_path: Path) -> Dict[str, Any]:
    """Load the exact literature authority hashed into a Dev checkpoint."""

    path = checkpoint_path.parent / "preplan_literature_bundle.json"
    if path.is_symlink() or not path.is_file():
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_literature_missing",
            "The prior Planner checkpoint has no regular literature authority.",
        )
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_JSON_BYTES:
            raise ValueError("literature authority exceeds the bounded JSON size")
        bundle = LiteratureBundle.model_validate_json(raw)
    except (OSError, ValueError) as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_literature_invalid",
            "The prior Planner literature authority did not pass validation.",
        ) from exc
    return bundle.model_dump(mode="json")


def _inherited_metadata_planning_input(
    *, checkpoint_path: Path, wrapper_dir: Path
) -> tuple[Path, Path] | None:
    """Recover a legacy continuation's catalog from its shared checkpoint.

    Early development continuations copied the validated checkpoint prefix but
    did not restage the metadata-only acquisition receipt. Recovery stays
    bounded to regular sibling wrappers in the same study and accepts only one
    content-identical acquisition binding attached to an identical checkpoint
    file. New continuations restage their own receipt and do not need this path.
    """

    source_run_dir = checkpoint_path.parent.resolve()
    current_checkpoint_digests: set[str] = set()
    try:
        for path in source_run_dir.iterdir():
            if (
                path.is_file()
                and not path.is_symlink()
                and re.fullmatch(
                    r"progressive_planner_checkpoint_[0-9]{3}\.json",
                    path.name,
                )
            ):
                current_checkpoint_digests.add(
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
    except OSError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_unreadable",
            "The Planner checkpoint prefix cannot be inspected safely.",
        ) from exc
    if not current_checkpoint_digests:
        return None

    candidates: dict[tuple[str, str], tuple[Path, Path]] = {}
    study_root = wrapper_dir.parent.resolve()
    try:
        wrappers = list(study_root.iterdir())[:200]
    except OSError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_unreadable",
            "The Planner study workspace cannot be inspected safely.",
        ) from exc
    for candidate_wrapper in wrappers:
        if (
            candidate_wrapper == wrapper_dir
            or candidate_wrapper.is_symlink()
            or not candidate_wrapper.is_dir()
            or not candidate_wrapper.name.startswith("run_")
        ):
            continue
        pipeline_root = candidate_wrapper / "pipeline"
        candidate_input = candidate_wrapper / "pipeline_input"
        provenance_path = candidate_input / "planner_catalog_receipt.json"
        universe_path = candidate_input / "planner_catalog.parquet"
        if any(
            path.is_symlink() or not path.is_file()
            for path in (provenance_path, universe_path)
        ):
            continue
        try:
            pipeline_runs = [
                path
                for path in pipeline_root.iterdir()
                if path.is_dir() and not path.is_symlink()
            ]
        except OSError:
            continue
        shares_checkpoint = False
        for candidate_run in pipeline_runs[:2]:
            try:
                checkpoint_files = list(candidate_run.iterdir())[:200]
            except OSError:
                continue
            for candidate_checkpoint in checkpoint_files:
                if (
                    candidate_checkpoint.is_file()
                    and not candidate_checkpoint.is_symlink()
                    and re.fullmatch(
                        r"progressive_planner_checkpoint_[0-9]{3}\.json",
                        candidate_checkpoint.name,
                    )
                ):
                    try:
                        digest = hashlib.sha256(
                            candidate_checkpoint.read_bytes()
                        ).hexdigest()
                    except OSError:
                        continue
                    if digest in current_checkpoint_digests:
                        shares_checkpoint = True
                        break
            if shares_checkpoint:
                break
        if not shares_checkpoint:
            continue
        try:
            binding = (
                hashlib.sha256(universe_path.read_bytes()).hexdigest(),
                hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            )
        except OSError:
            continue
        candidates.setdefault(binding, (provenance_path, universe_path))
    if len(candidates) > 1:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_ambiguous",
            "The legacy Planner checkpoint maps to multiple acquisition receipts.",
        )
    return next(iter(candidates.values()), None)


def _development_resume_acquisition_profile(
    *,
    checkpoint_path: Path,
    database: str,
    cohort_window: tuple[float, float],
    outcome_concepts: Sequence[str],
    static_concepts: Sequence[str],
    required_feature_concepts: Sequence[str],
    planning_target_outcome: Optional[str] = None,
    planning_endpoint: Any = None,
    planning_operationalized_columns: Sequence[str] = (),
) -> _DevelopmentResumeAcquisition:
    """Restore the exact server-owned concept roster behind a Dev checkpoint.

    Progressive Planner replay binds the materialized cohort bytes and selected
    variable roster. Re-running the agent-selectable acquisition step can pick
    a different optional covariate and make an otherwise valid checkpoint
    unreplayable. The Web host therefore restores only the prior materializer's
    typed concept roster, then rematerializes it from the currently validated
    export. It never accepts a client path or reuses patient rows blindly.
    """

    source_run_dir = checkpoint_path.parent.resolve()
    wrapper_dir = source_run_dir.parent.parent.resolve()
    pipeline_input = (wrapper_dir / "pipeline_input").resolve()
    provenance_path = pipeline_input / "web_research_universe_provenance.json"
    universe_path = pipeline_input / "web_research_universe.parquet"
    metadata_provenance_path = pipeline_input / "planner_catalog_receipt.json"
    metadata_universe_path = pipeline_input / "planner_catalog.parquet"
    selected_owner_root = wrapper_dir
    if not any(
        path.is_file()
        for path in (
            provenance_path,
            universe_path,
            metadata_provenance_path,
            metadata_universe_path,
        )
    ):
        inherited = _inherited_metadata_planning_input(
            checkpoint_path=checkpoint_path,
            wrapper_dir=wrapper_dir,
        )
        if inherited is not None:
            metadata_provenance_path, metadata_universe_path = inherited
            selected_owner_root = wrapper_dir.parent.resolve()
    if provenance_path.is_file() or universe_path.is_file():
        selected_provenance_path = provenance_path
        selected_universe_path = universe_path
        acquisition_kind = "materialized_patient_universe"
    else:
        selected_provenance_path = metadata_provenance_path
        selected_universe_path = metadata_universe_path
        acquisition_kind = "metadata_only_planning_catalog"
    try:
        selected_provenance_path.relative_to(selected_owner_root)
        selected_universe_path.relative_to(selected_owner_root)
    except ValueError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_invalid",
            "The prior canary acquisition escaped its server-owned workspace.",
        ) from exc
    if any(
        path.is_symlink() or not path.is_file()
        for path in (selected_provenance_path, selected_universe_path)
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_missing",
            "The prior canary has no typed acquisition receipt.",
        )
    try:
        provenance = json.loads(selected_provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_unreadable",
            "The prior canary acquisition receipt cannot be verified.",
        ) from exc
    if acquisition_kind == "metadata_only_planning_catalog":
        try:
            provenance_raw = selected_provenance_path.read_bytes()
            universe_raw = selected_universe_path.read_bytes()
        except OSError as exc:
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_acquisition_unreadable",
                "The prior metadata-only planning catalog cannot be verified.",
            ) from exc
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("schema_version")
            != "easyicu.metadata-only-planning-catalog/1"
        ):
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_acquisition_invalid",
                "The prior metadata-only planning receipt has an unsupported schema.",
            )
        selected = provenance.get("selected_concepts")
        if (
            not isinstance(selected, list)
            or not selected
            or not all(isinstance(value, str) and value.strip() for value in selected)
        ):
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_acquisition_invalid",
                "The prior metadata-only planning receipt has an invalid concept roster.",
            )
        normalized_selected = tuple(dict.fromkeys(value.strip() for value in selected))
        normalized_operationalized = (
            _normalized_metadata_planning_operationalized_columns(
                planning_operationalized_columns
            )
        )
        selected_sha256 = hashlib.sha256(
            json.dumps(
                list(normalized_selected),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        expected_endpoint = (
            planning_endpoint.model_dump(mode="json")
            if planning_endpoint is not None
            and hasattr(planning_endpoint, "model_dump")
            else None
        )
        if (
            str(provenance.get("database") or "").strip().lower() != database
            or str(provenance.get("selected_concepts_sha256") or "") != selected_sha256
            or provenance.get("patient_rows_read") is not False
            or provenance.get("patient_rows_written") is not False
            or provenance.get("observed_feasibility_claims") is not False
            or provenance.get("execution_authorized") is not False
            or (provenance.get("planning_target_outcome") or None)
            != planning_target_outcome
            or (provenance.get("planning_endpoint") or None) != expected_endpoint
            or tuple(provenance.get("operationalized_columns") or ())
            != normalized_operationalized
        ):
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_acquisition_authority_mismatch",
                "The prior metadata-only catalog does not match the current study authority.",
            )
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(selected_universe_path)
            row_identity = str(provenance.get("row_identity_column") or "").strip()
            raw_patient_identity = provenance.get("patient_identity_column")
            patient_identity = (
                str(raw_patient_identity).strip()
                if raw_patient_identity is not None
                else ""
            )
            patient_identity_valid = bool(
                not patient_identity
                or (
                    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", patient_identity)
                    and patient_identity != row_identity
                    and patient_identity not in normalized_selected
                )
            )
            expected_columns = [
                row_identity,
                *([patient_identity] if patient_identity else []),
                *normalized_operationalized,
                *normalized_selected,
            ]
            expected_columns = list(dict.fromkeys(expected_columns))
            if (
                not row_identity
                or not patient_identity_valid
                or parquet.metadata.num_rows != 0
                or parquet.schema_arrow.names != expected_columns
            ):
                raise ResearchPipelineRunError(
                    "research_pipeline_development_resume_acquisition_invalid",
                    "The prior metadata-only catalog no longer matches its receipt.",
                )
        except ResearchPipelineRunError:
            raise
        except (OSError, ValueError) as exc:
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_acquisition_unreadable",
                "The prior metadata-only planning catalog cannot be verified.",
            ) from exc
        return _DevelopmentResumeAcquisition(
            kind=acquisition_kind,
            selected_concepts=normalized_selected,
            universe_path=selected_universe_path,
            provenance_path=selected_provenance_path,
            universe_sha256=hashlib.sha256(universe_raw).hexdigest(),
            provenance_sha256=hashlib.sha256(provenance_raw).hexdigest(),
        )

    try:
        universe_sha256 = hashlib.sha256(
            selected_universe_path.read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_unreadable",
            "The prior canary acquisition receipt cannot be verified.",
        ) from exc
    if not isinstance(provenance, Mapping) or provenance.get("schema_version") != (
        "easyicu.cohort_materializer/1"
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_invalid",
            "The prior canary acquisition receipt has an unsupported schema.",
        )
    if str(provenance.get("cohort_file_sha256") or "") != universe_sha256:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_digest_mismatch",
            "The prior canary acquisition no longer matches its typed receipt.",
        )

    def _concepts(key: str) -> tuple[str, ...]:
        values = provenance.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ResearchPipelineRunError(
                "research_pipeline_development_resume_acquisition_invalid",
                "The prior canary acquisition has an invalid concept roster.",
                details={"field": key},
            )
        return tuple(dict.fromkeys(value.strip() for value in values))

    restored = {
        "feature_concepts": _concepts("feature_concepts"),
        "outcome_concepts": _concepts("outcome_concepts"),
        "static_concepts": _concepts("static_concepts"),
    }
    recorded_window = provenance.get("cohort_window_hours")
    try:
        normalized_recorded_window = tuple(float(value) for value in recorded_window)
    except (TypeError, ValueError):
        normalized_recorded_window = ()
    expected_window = tuple(float(value) for value in cohort_window)
    if (
        str(provenance.get("database") or "").strip().lower() != database
        or normalized_recorded_window != expected_window
        or restored["outcome_concepts"]
        != tuple(dict.fromkeys(str(value) for value in outcome_concepts))
        or restored["static_concepts"]
        != tuple(dict.fromkeys(str(value) for value in static_concepts))
        or not set(required_feature_concepts).issubset(restored["feature_concepts"])
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_authority_mismatch",
            "The prior canary acquisition does not match the current study authority.",
        )
    return _DevelopmentResumeAcquisition(
        kind=acquisition_kind,
        feature_concepts=restored["feature_concepts"],
        outcome_concepts=restored["outcome_concepts"],
        static_concepts=restored["static_concepts"],
        universe_path=selected_universe_path,
        provenance_path=selected_provenance_path,
    )


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


def _plan_has_complete_reviewable_recommendation(plan: Mapping[str, Any]) -> bool:
    """Fail closed only for typed design selections that predate the proposal."""

    selection = plan.get("design_selection")
    if not isinstance(selection, Mapping):
        return True
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        return False
    selected = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and str(candidate.get("disposition") or "") == "selected"
        ),
        None,
    )
    if selected is None:
        return False
    recommendation = selected.get("reviewable_plan")
    return bool(
        isinstance(recommendation, list)
        and len(recommendation) == 6
        and all(len(_clean_text(value)) >= 8 for value in recommendation)
    )


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
        if (
            len(raw) > _MAX_JSON_BYTES
            or hashlib.sha256(raw).hexdigest() != expected_sha
        ):
            return {}
        review = PlanScientificReview.model_validate_json(raw)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return review.model_dump(mode="json")


def _pending_plan_approval_allowed(
    *,
    run_dir: Optional[Path],
    pending: Optional[Any],
    plan_recommendation_complete: bool,
) -> bool:
    """Fail closed unless the exact bound review satisfies current policy."""

    # No pending entry is no approval authority. Owning the guard here keeps
    # every call site from having to remember it before reading
    # ``pending.requests``.
    if pending is None or not plan_recommendation_complete:
        return False
    review = _load_pending_scientific_review(run_dir, pending)
    if not review or review.get("approval_allowed") is not True:
        return False
    return all(
        (
            request.payload.get("approval_allowed", True)
            if isinstance(request.payload, Mapping)
            else True
        )
        for request in pending.requests
    )


def _pending_review_reason_code(
    *,
    request: Any,
    plan_recommendation_complete: bool,
    scientific_plan_review: Mapping[str, Any],
) -> str:
    """Project one precise current-policy reason for the paused plan."""

    if not plan_recommendation_complete:
        return "plan_scientific_changes_required"
    if not scientific_plan_review:
        return "scientific_plan_review_policy_stale"
    if scientific_plan_review.get("approval_allowed") is not True:
        return "plan_scientific_changes_required"
    payload = request.payload if isinstance(request.payload, Mapping) else {}
    return _clean_text(payload.get("reason"), 160)


def _pipeline_failure_code(exc: BaseException) -> str:
    chain = _pipeline_exception_chain(exc)
    if any("timeout" in type(item).__name__.casefold() for item in chain):
        return "research_pipeline_provider_timeout"
    if any(type(item).__name__ == "StructuredResponseFailure" for item in chain):
        return "research_pipeline_plan_contract_exhausted"
    typed_failure = _safe_pipeline_typed_failure(exc)
    if typed_failure.get("owner") == ("easyicu.providers.planner_efficiency_budget_v1"):
        return "research_pipeline_planner_efficiency_budget_exhausted"
    if typed_failure.get(
        "owner"
    ) == "easyicu.providers.codex_app_server_v1" and typed_failure.get(
        "reason_code"
    ) in {
        "codex_auth_app_server_timeout",
        "codex_auth_notification_hard_timeout",
        "codex_auth_notification_timeout",
    }:
        return "research_pipeline_provider_timeout"
    if typed_failure.get("owner") == "easyicu.planning.progressive_compiler_v1":
        return "research_pipeline_progressive_compile_failed"
    if typed_failure.get("owner") == "easyicu.schema_validation_v1":
        return "research_pipeline_schema_validation_failed"
    if typed_failure.get("owner") == _EXECUTION_RUNTIME_DIAGNOSTIC_OWNER:
        # The same code the launch preflight uses, so a runtime that went down
        # mid-run is attributed to the host environment rather than reported as
        # a generic execution failure of the science.
        return "research_pipeline_execution_runtime_unavailable"
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


_SAFE_COMPILER_COORDINATE_RE = re.compile(r"^[A-Za-z0-9_.:\[\]-]{1,240}$")
_SAFE_SCHEMA_ERROR_TYPE_RE = re.compile(r"^[a-z][a-z0-9_.]{0,79}$")
_SAFE_PLANNER_BUDGET_REASONS = frozenset(
    {
        "call_limit",
        "provider_usage_unavailable",
        "reported_token_limit",
        "wall_clock_limit",
    }
)
_SAFE_CODEX_APP_SERVER_REASONS = frozenset(
    {
        "codex_auth_app_server_exited",
        "codex_auth_app_server_request_failed",
        "codex_auth_app_server_timeout",
        "codex_auth_notification_hard_timeout",
        "codex_auth_notification_timeout",
    }
)
# Mirrors the execution-runtime owner's published contract; the pairing is
# locked by test_web_execution_runtime_preflight.py so a new reason code cannot
# silently reach this boundary unprojected.
_EXECUTION_RUNTIME_DIAGNOSTIC_OWNER = "easyicu.execution.runtime_v1"
_SAFE_RUNNER_UNAVAILABLE_REASONS = frozenset(
    {
        "docker_daemon_unreachable",
        "docker_executable_missing",
        "docker_image_missing",
        "docker_probe_failed",
        "host_sandbox_missing",
    }
)


def _safe_pipeline_typed_failure(exc: BaseException) -> Dict[str, Any]:
    """Project one allowlisted owner diagnostic without exception text."""

    for item in _pipeline_exception_chain(exc):
        if isinstance(item, ValidationError):
            coordinates: List[Dict[str, Any]] = []
            for error in item.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )[:8]:
                error_type = error.get("type")
                location = error.get("loc")
                if (
                    not isinstance(error_type, str)
                    or _SAFE_SCHEMA_ERROR_TYPE_RE.fullmatch(error_type) is None
                    or not isinstance(location, tuple)
                ):
                    continue
                safe_location: List[Any] = []
                for part in location[:12]:
                    if isinstance(part, int) and not isinstance(part, bool):
                        if 0 <= part <= 10_000:
                            safe_location.append(part)
                        continue
                    if isinstance(part, str) and _SAFE_COMPILER_COORDINATE_RE.fullmatch(
                        part
                    ):
                        safe_location.append(part)
                coordinates.append(
                    {
                        "location": safe_location,
                        "error_type": error_type,
                    }
                )
            return {
                "owner": "easyicu.schema_validation_v1",
                "reason_code": "pydantic_contract_validation_failed",
                "error_count": min(item.error_count(), 10_000),
                "coordinates": coordinates,
            }
        raw = getattr(item, "easyicu_safe_diagnostic", None)
        if not isinstance(raw, Mapping):
            continue
        owner = raw.get("owner")
        if owner == "easyicu.planning.progressive_compiler_v1":
            reason_code = raw.get("reason_code")
            if not isinstance(reason_code, str) or not re.fullmatch(
                r"[a-z][a-z0-9_]{2,79}", reason_code
            ):
                continue
            projected: Dict[str, Any] = {
                "owner": owner,
                "reason_code": reason_code,
            }
            step_id = raw.get("step_id")
            if isinstance(step_id, str) and re.fullmatch(
                r"[a-z0-9][a-z0-9_]{0,79}", step_id
            ):
                projected["step_id"] = step_id
            step_index = raw.get("step_index")
            if (
                isinstance(step_index, int)
                and not isinstance(step_index, bool)
                and 0 <= step_index <= 10_000
            ):
                projected["step_index"] = step_index
            path = raw.get("path")
            if isinstance(path, str) and _SAFE_COMPILER_COORDINATE_RE.fullmatch(path):
                projected["path"] = path
            return projected
        if owner == "easyicu.providers.planner_efficiency_budget_v1":
            reason = raw.get("reason")
            reason_code = raw.get("reason_code")
            calls = raw.get("calls")
            reported_tokens = raw.get("reported_tokens")
            elapsed_seconds = raw.get("elapsed_seconds")
            limits = raw.get("limits")
            if (
                reason_code != "planner_efficiency_budget_exhausted"
                or reason not in _SAFE_PLANNER_BUDGET_REASONS
                or not isinstance(calls, int)
                or isinstance(calls, bool)
                or calls < 0
                or not isinstance(reported_tokens, int)
                or isinstance(reported_tokens, bool)
                or reported_tokens < 0
                or not isinstance(elapsed_seconds, (int, float))
                or isinstance(elapsed_seconds, bool)
                or not math.isfinite(float(elapsed_seconds))
                or float(elapsed_seconds) < 0
                or not isinstance(limits, Mapping)
            ):
                continue
            max_calls = limits.get("max_calls")
            max_reported_tokens = limits.get("max_reported_tokens")
            max_wall_seconds = limits.get("max_wall_seconds")
            if (
                not isinstance(max_calls, int)
                or isinstance(max_calls, bool)
                or max_calls <= 0
                or not isinstance(max_reported_tokens, int)
                or isinstance(max_reported_tokens, bool)
                or max_reported_tokens <= 0
                or not isinstance(max_wall_seconds, (int, float))
                or isinstance(max_wall_seconds, bool)
                or not math.isfinite(float(max_wall_seconds))
                or float(max_wall_seconds) <= 0
            ):
                continue
            return {
                "owner": owner,
                "reason_code": reason_code,
                "reason": reason,
                "calls": calls,
                "reported_tokens": reported_tokens,
                "elapsed_seconds": round(float(elapsed_seconds), 6),
                "limits": {
                    "max_calls": max_calls,
                    "max_reported_tokens": max_reported_tokens,
                    "max_wall_seconds": float(max_wall_seconds),
                },
            }
        if owner == "easyicu.providers.codex_app_server_v1":
            reason_code = raw.get("reason_code")
            if reason_code in _SAFE_CODEX_APP_SERVER_REASONS:
                return {
                    "owner": owner,
                    "reason_code": reason_code,
                }
        if owner == _EXECUTION_RUNTIME_DIAGNOSTIC_OWNER:
            reason_code = raw.get("reason_code")
            runner_kind = raw.get("runner_kind")
            # The backend's own wording names a host socket path, so only the
            # closed reason code and backend name cross this boundary. The
            # image reference is already carried by the run's config receipt.
            if (
                reason_code not in _SAFE_RUNNER_UNAVAILABLE_REASONS
                or not isinstance(runner_kind, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,31}", runner_kind) is None
            ):
                continue
            return {
                "owner": owner,
                "reason_code": reason_code,
                "runner_kind": runner_kind,
            }
    return {}


def _pipeline_failure_category(exc: BaseException) -> str:
    """Return a closed diagnostic category for one bounded exception chain."""

    typed_failure = _safe_pipeline_typed_failure(exc)
    if typed_failure.get(
        "owner"
    ) == "easyicu.providers.codex_app_server_v1" and "timeout" in str(
        typed_failure.get("reason_code") or ""
    ):
        return "timeout"
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
    traceback_frames = []
    for item in _pipeline_exception_chain(exc):
        for frame in traceback.extract_tb(item.__traceback__):
            filename = Path(frame.filename).name
            function = str(frame.name or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", filename):
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_<>]{0,159}", function):
                continue
            traceback_frames.append(
                {
                    "file": filename,
                    "function": function,
                    "line": int(frame.lineno),
                }
            )
    payload = {
        "schema_version": "easyicu.web-research-pipeline-failure/4",
        "status": "failed",
        "code": code,
        "failure_type": _pipeline_failure_category(exc),
        "typed_failure": _safe_pipeline_typed_failure(exc),
        "exception_types": [
            type(item).__name__
            for item in _pipeline_exception_chain(exc)
            if type(item).__name__ in _SAFE_PIPELINE_EXCEPTION_TYPES
        ],
        # Bounded code coordinates make an otherwise generic RuntimeError
        # attributable without persisting exception messages, prompts, paths,
        # provider output, or patient data.
        "traceback_frames": traceback_frames[-16:],
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


def _write_pipeline_failure_projection(
    *,
    wrapper_dir: Path,
    study: Mapping[str, Any],
    provider: Mapping[str, Any],
    code: str,
    failure_type: str,
    diagnostic: Optional[str],
) -> bool:
    """Write a fail-closed terminal receipt that Project Monitor can index."""

    run_id = wrapper_dir.name
    provider_public = {
        key: provider.get(key)
        for key in (
            "provider",
            "model",
            "client",
            "provider_gate",
            "credential_fingerprint",
        )
        if provider.get(key) is not None
    }
    gate = {
        "status": "blocked",
        "reason": code,
        "reportable": False,
        "draft_unlocked": False,
        "checks": [
            {
                "id": "research_pipeline_execution",
                "label": "Research Agent pipeline reached a governed result",
                "passed": False,
                "reason_code": code,
            }
        ],
    }
    payloads: Dict[str, Dict[str, Any]] = {
        "run_context.json": {
            "run_id": run_id,
            "study_id": _clean_text(study.get("id"), 160),
            "scientific_configuration_sha256": (
                study_context_owner.scientific_configuration_sha256(study)
            ),
            "mode": "research_agent_pipeline",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "summary": {
                "execution_complete": False,
                "analysis_started": False,
            },
            "local_first": {"uploads": 0},
        },
        "quality_gate.json": {"gate": gate, "quality": []},
        "source_run_manifest.json": {
            "schema_version": "easyicu.web-research-pipeline-projection/1",
            "engine": "easyicu.research_agent.pipeline",
            "run_id": run_id,
            "status": "failed",
            "failure_code": code,
            "failure_type": failure_type,
            "diagnostic_available": bool(diagnostic),
            "provider": provider_public,
            "path_values_returned": False,
            "analysis_started": False,
            "publication_authorized": False,
        },
    }
    privacy_scan = run_artifact_disclosure.scan_browser_projection(payloads)
    if not privacy_scan["passed"]:
        provider_public = {}
        payloads["source_run_manifest.json"]["provider"] = {}
        privacy_scan = run_artifact_disclosure.scan_browser_projection(payloads)
    if not privacy_scan["passed"]:
        return False
    try:
        for name, payload in payloads.items():
            _write_json(wrapper_dir / name, payload)
        artifacts = [_artifact_record(wrapper_dir / name) for name in payloads]
        _write_json(
            wrapper_dir / "evidence_ledger.json",
            {
                "schema_version": "easyicu.web-research-pipeline-ledger/1",
                "run_id": run_id,
                "run_type": "full",
                "engine": "easyicu.research_agent.pipeline",
                "status": "blocked",
                "artifacts": artifacts,
                "provider": provider_public,
                "pipeline_evidence_count": 0,
                "privacy": {
                    "patient_rows_in_projection": False,
                    "path_values_returned": False,
                    "projection_scan_passed": bool(privacy_scan["passed"]),
                },
            },
        )
    except OSError:
        return False
    return True


def _write_review_resume_failure_diagnostic(
    *,
    wrapper_dir: Path,
    exc: BaseException,
    review_resumable: bool,
) -> Optional[str]:
    """Persist a private, bounded diagnostic for one failed review resume."""

    module = type(exc).__module__
    name = type(exc).__name__
    exception_type = (
        f"{module}.{name}"
        if module == "builtins"
        or module == "pydantic_core"
        or module.startswith("easyicu.")
        else "unclassified"
    )
    payload = {
        "schema_version": "easyicu.web-research-review-resume-failure/1",
        "status": "failed",
        "code": "research_pipeline_review_resume_failed",
        "failure_type": _pipeline_failure_category(exc),
        "exception_type": exception_type,
        "typed_failure": _safe_pipeline_typed_failure(exc),
        "review_resumable": bool(review_resumable),
        "message_sha256": hashlib.sha256(
            f"{type(exc).__name__}: {exc}".encode("utf-8")
        ).hexdigest(),
        "raw_exception_recorded": False,
        "provider_output_recorded": False,
        "patient_rows_recorded": False,
        "secrets_recorded": False,
    }
    relative = "diagnostics/research_pipeline_review_resume_failure.json"
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
        return ScientificConfiguration.inspect(study).sensitivity_specs()
    except ScientificConfigurationError as exc:
        raise ResearchPipelineRunError(exc.code, str(exc), details=exc.details) from exc


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


def _metadata_only_patient_grouping_authority(
    patient_grouping: Optional[PatientGroupingBinding],
) -> Optional[Dict[str, Any]]:
    if patient_grouping is None:
        return None
    coordinates = dict(patient_grouping.authority_coordinates)
    if (
        coordinates.get("schema_version")
        != "easyicu.patient_grouping_runtime_authority/1"
        or not isinstance(coordinates.get("authority_ref"), str)
        or not coordinates.get("authority_ref")
        or coordinates.get("mapping_sha256") != patient_grouping.mapping_sha256
        or coordinates.get("grouping_derivation") != "prefix_before_:s"
        or coordinates.get("provider_visible_values") is not False
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_planning_patient_grouping_authority_invalid",
            "The verified patient-grouping authority cannot be projected into "
            "the metadata-only planning schema.",
        )
    safe_coordinates = {
        key: coordinates[key]
        for key in (
            "schema_version",
            "authority_ref",
            "database",
            "export_manifest_file",
            "export_manifest_sha256",
            "mapping_sha256",
            "grouping_derivation",
            "provider_visible_values",
        )
        if key in coordinates
    }
    return {
        "output_identity_column": patient_grouping.output_identity_column,
        "mapping_file_sha256": patient_grouping.mapping_sha256,
        "mapped_cohort_rows": 0,
        "patient_group_derivation": {
            "algorithm": "prefix_before_:s",
            "delimiter": ":s",
        },
        "authority_coordinates": safe_coordinates,
    }


def _metadata_only_planning_acquisition(
    *,
    database: str,
    question: str,
    llm: Any,
    output_dir: Path,
    target_outcome: Optional[str] = None,
    endpoint: Any = None,
    required_concepts: Sequence[str] = (),
    patient_grouping: Optional[PatientGroupingBinding] = None,
    operationalized_columns: Sequence[str] = (),
) -> Any:
    """Select a planning catalog without reading patient data.

    This is the Planner-only counterpart of ``acquire_universe_for_question``:
    the same data-foundation model chooses a parsimonious concept set, but its
    menu comes from EasyICU's database capability registry rather than files in
    an export package. The resulting parquet has schema only and zero rows. It
    can inform a reviewable plan, but it cannot authorize execution or support
    denominator, missingness, event-rate, or effect claims.
    """

    import pandas as pd

    from easyicu.research_agent.acquisition.catalog import (
        assess_coverage,
        build_database_capability_catalog,
    )
    from easyicu.research_agent.acquisition.foundation import (
        AcquisitionResult,
        DataFoundationAgent,
    )
    from easyicu.database_config import ID_COLUMNS
    from easyicu.research_agent.concept_availability import normalize_database_name

    catalog = build_database_capability_catalog(database)
    if not catalog.concepts:
        raise ResearchPipelineRunError(
            "research_pipeline_planning_catalog_unavailable",
            "EasyICU has no metadata-only concept catalog for this database.",
            details={"database": database},
        )
    selection = DataFoundationAgent(llm).select_concepts(
        question=question,
        catalog=catalog,
        target_outcome=target_outcome,
    )
    resolvable_required = [
        value.strip()
        for value in required_concepts
        if isinstance(value, str)
        and value.strip()
        and assess_coverage([value.strip()], catalog).sufficient
    ]
    selection.selected_concepts = list(
        dict.fromkeys(
            [
                *selection.selected_concepts,
                *resolvable_required,
            ]
        )
    )
    selection.coverage = assess_coverage(selection.selected_concepts, catalog)
    coverage = selection.coverage or assess_coverage(
        selection.selected_concepts,
        catalog,
    )
    selected = list(dict.fromkeys(coverage.available))
    blocked = bool(
        not selection.selection_succeeded or not selected or not coverage.sufficient
    )
    if blocked:
        return AcquisitionResult(
            universe_path=None,
            provenance_path=None,
            selection=selection,
            materialized_concepts=[],
            coverage=coverage,
            blocked=True,
            note=(
                "Metadata-only concept selection failed before Planner launch; "
                "no patient data were read."
            ),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = output_dir / "planner_catalog.parquet"
    normalized_database = normalize_database_name(database)
    database_id_columns = ID_COLUMNS.get(normalized_database, [])
    if not database_id_columns:
        raise ResearchPipelineRunError(
            "research_pipeline_planning_identity_unavailable",
            "EasyICU has no canonical ICU-stay identity for metadata-only "
            "planning on this database.",
            details={"database": normalized_database},
        )
    row_identity_column = database_id_columns[0]
    normalized_operationalized = _normalized_metadata_planning_operationalized_columns(
        operationalized_columns
    )
    replacement_row_identity = _metadata_only_patient_grouping_authority(
        patient_grouping
    )
    projected_patient_identity = (
        patient_grouping.output_identity_column
        if patient_grouping is not None
        and patient_grouping.output_identity_column != row_identity_column
        else None
    )
    planning_columns: Dict[str, Any] = {row_identity_column: pd.Series(dtype="int64")}
    if projected_patient_identity:
        # This is schema authority only. The private stay-to-patient mapping is
        # applied later by the governed materializer and no identifier values
        # cross the Planner boundary.
        planning_columns[projected_patient_identity] = pd.Series(dtype="string")
    planning_columns.update(
        {
            column: pd.Series(dtype="float64")
            for column in normalized_operationalized
            if column not in planning_columns
        }
    )
    planning_columns.update(
        {
            concept: pd.Series(dtype="float64")
            for concept in selected
            if concept not in planning_columns
        }
    )
    planning_catalog = pd.DataFrame(planning_columns)
    planning_catalog.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog",
        "patient_rows_read": False,
        **(
            {"replacement_row_identity": replacement_row_identity}
            if replacement_row_identity is not None
            else {}
        ),
    }
    planning_catalog.to_parquet(universe_path, index=False)
    provenance_path = output_dir / "planner_catalog_receipt.json"
    _write_json(
        provenance_path,
        {
            "schema_version": "easyicu.metadata-only-planning-catalog/1",
            "database": database,
            "catalog_source": catalog.source,
            "row_identity_column": row_identity_column,
            "patient_identity_column": projected_patient_identity,
            "operationalized_columns": list(normalized_operationalized),
            "replacement_row_identity": replacement_row_identity,
            "selected_concepts": selected,
            "selected_concepts_sha256": hashlib.sha256(
                json.dumps(
                    selected,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "patient_rows_read": False,
            "patient_rows_written": False,
            "observed_feasibility_claims": False,
            "execution_authorized": False,
            "planning_target_outcome": target_outcome,
            "planning_endpoint": (
                endpoint.model_dump(mode="json")
                if endpoint is not None and hasattr(endpoint, "model_dump")
                else None
            ),
        },
    )
    return AcquisitionResult(
        universe_path=universe_path,
        provenance_path=provenance_path,
        selection=selection,
        materialized_concepts=[],
        coverage=coverage,
        blocked=False,
        endpoint=endpoint,
        note=(
            "Metadata-only planning catalog; no patient rows were read and "
            "execution requires a separately prepared package."
        ),
    )


def _restore_metadata_only_planning_acquisition(
    *,
    database: str,
    profile: _DevelopmentResumeAcquisition,
    output_dir: Path,
    endpoint: Any = None,
    patient_grouping: Optional[PatientGroupingBinding] = None,
    operationalized_columns: Sequence[str] = (),
) -> Any:
    """Replay and restage one verified zero-row catalog without an LLM call.

    Each continuation owns a fresh immutable wrapper. Restaging the verified
    catalog and receipt into that wrapper lets a later budget-bounded
    continuation resume from this run as well, instead of losing acquisition
    authority after exactly one hop.
    """

    from easyicu.research_agent.acquisition.catalog import (
        assess_coverage,
        build_database_capability_catalog,
    )
    from easyicu.research_agent.acquisition.foundation import (
        AcquisitionResult,
        ConceptSelection,
    )

    if (
        profile.kind != "metadata_only_planning_catalog"
        or profile.universe_path is None
        or profile.provenance_path is None
        or not profile.selected_concepts
        or not profile.universe_sha256
        or not profile.provenance_sha256
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_invalid",
            "The Planner checkpoint has no restorable metadata-only catalog.",
        )
    catalog = build_database_capability_catalog(database)
    coverage = assess_coverage(profile.selected_concepts, catalog)
    if not coverage.sufficient:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_authority_mismatch",
            "The prior Planner catalog is no longer executable in the current "
            "database capability registry.",
        )
    try:
        universe_raw = profile.universe_path.read_bytes()
        provenance_raw = profile.provenance_path.read_bytes()
    except OSError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_unreadable",
            "The verified Planner catalog could not be restaged.",
        ) from exc
    if (
        hashlib.sha256(universe_raw).hexdigest() != profile.universe_sha256
        or hashlib.sha256(provenance_raw).hexdigest() != profile.provenance_sha256
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_acquisition_digest_mismatch",
            "The verified Planner catalog changed before it could be restaged.",
        )
    import pyarrow.parquet as pq

    receipt = _read_json(profile.provenance_path, {})
    schema_columns = set(pq.read_schema(profile.universe_path).names)
    expected_patient_identity = (
        patient_grouping.output_identity_column
        if patient_grouping is not None
        else None
    )
    expected_replacement = _metadata_only_patient_grouping_authority(patient_grouping)
    expected_operationalized = _normalized_metadata_planning_operationalized_columns(
        operationalized_columns
    )
    if (
        receipt.get("patient_identity_column") != expected_patient_identity
        or receipt.get("replacement_row_identity") != expected_replacement
        or tuple(receipt.get("operationalized_columns") or ())
        != expected_operationalized
        or (
            expected_patient_identity is not None
            and expected_patient_identity not in schema_columns
        )
        or any(column not in schema_columns for column in expected_operationalized)
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_development_resume_identity_authority_mismatch",
            "The prior Planner catalog does not match the current patient "
            "grouping and operationalized planning schema.",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = output_dir / "planner_catalog.parquet"
    provenance_path = output_dir / "planner_catalog_receipt.json"
    universe_path.write_bytes(universe_raw)
    provenance_path.write_bytes(provenance_raw)
    selection = ConceptSelection(
        selected_concepts=list(profile.selected_concepts),
        rationale=(
            "Restored from the verified metadata-only Planner checkpoint catalog."
        ),
        coverage=coverage,
        selection_authority="host_exact",
    )
    return AcquisitionResult(
        universe_path=universe_path,
        provenance_path=provenance_path,
        selection=selection,
        materialized_concepts=[],
        coverage=coverage,
        blocked=False,
        endpoint=endpoint,
        note=(
            "Verified metadata-only Planner replay; no patient rows were read "
            "and execution remains unauthorized."
        ),
    )


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


def _resolve_materialized_primary_exposure(
    *,
    configured: Optional[str],
    source_concept: Optional[str],
    aggregation: Optional[str] = None,
    acquisition: Any,
) -> Optional[str]:
    """Resolve only an owner-issued materialized exposure coordinate."""

    if not configured:
        return None
    source = source_concept or configured
    if aggregation:
        operational = f"{source}_{aggregation}"
        materialized = set(getattr(acquisition, "materialized_columns", ()) or ())
        return operational if operational in materialized else None
    if configured == source:
        return (getattr(acquisition, "analysis_columns", {}) or {}).get(source)
    materialized = set(getattr(acquisition, "materialized_columns", ()) or ())
    return configured if configured in materialized else None


def _resolve_planner_proposed_primary_exposure(
    *, source_concept: str, acquisition: Any
) -> Optional[str]:
    """Choose one reviewable, executable representation for a Planner proposal.

    This is deliberately separate from ``_resolve_materialized_primary_exposure``:
    that function enforces an already user-approved aggregation. Here the user
    named the scientific concept in the question but asked EasyICU to propose
    the study design first. We therefore consult the case-neutral ICU
    aggregation policy, select the first representation the sealed materialized
    universe actually provides, and leave the choice inside the candidate Plan
    for human review. The StudyContext is not mutated.
    """

    analysis_columns = dict(getattr(acquisition, "analysis_columns", {}) or {})
    direct = str(analysis_columns.get(source_concept) or "").strip()
    if direct:
        return direct

    from easyicu.research_agent.icu_rules import (
        aggregation_rule_for,
        classify_variable,
    )
    from easyicu.research_agent.schema import AggregationRule

    hint = classify_variable(source_concept, "float64")
    ordered_rules = list(
        dict.fromkeys(
            [hint.aggregation_default, *aggregation_rule_for(hint.role, hint.kind)]
        )
    )
    suffixes = {
        AggregationRule.MEAN_MEDIAN: ("mean", "median"),
        AggregationRule.MEDIAN_ONLY: ("median",),
        AggregationRule.MAX_LAST: ("max", "last"),
        AggregationRule.FIRST_VALUE: ("first",),
        AggregationRule.SUM: ("sum",),
        AggregationRule.NONE: ("",),
        AggregationRule.ANY: ("mean", "median", "max", "last", "first"),
    }
    materialized = set(getattr(acquisition, "materialized_columns", ()) or ())
    for rule in ordered_rules:
        for suffix in suffixes.get(rule, ()):
            candidate = source_concept if not suffix else f"{source_concept}_{suffix}"
            if candidate in materialized:
                return candidate
    return None


def _resolve_materialized_target_outcome(
    *, source_concept: str, acquisition: Any
) -> Optional[str]:
    """Resolve a concept-level outcome to its owner-issued cohort column.

    Event-status concepts such as ``death`` are normalized by the data
    foundation to a stable stay-level coordinate such as ``death_max``.  The
    Research Agent must receive that materialized coordinate rather than the
    pre-materialization concept id; otherwise cohort validation incorrectly
    reports that a present outcome is missing.
    """

    analysis_columns = dict(getattr(acquisition, "analysis_columns", {}) or {})
    materialized = set(getattr(acquisition, "materialized_columns", ()) or ())
    resolved = str(analysis_columns.get(source_concept) or "").strip()
    if resolved:
        return resolved if resolved in materialized else None
    return source_concept if source_concept in materialized else None


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
                str(key): len(json.dumps(value, ensure_ascii=False, sort_keys=True))
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
        # A comparator is the estimand's contrast, not a request for subgroup
        # analyses. Filed as `subgroup_sensitivity` it reached the Planner as
        # "Include subgroup/sensitivity requests: <comparator>"
        # (`skills.py::_preference_rationale_note`), turning a stated reference
        # group into extra analyses nobody asked for. The inbound contract has
        # no comparator field -- the contrast belongs to the Planner's own
        # `PlannedModelRequirement.exposure_reference_level` -- so it travels as
        # a declared statement the Planner may honour, not as an instruction.
        preferences["extra_notes"] = "\n".join(
            part
            for part in (
                preferences.get("extra_notes"),
                f"Comparator stated by the researcher: {comparator}",
            )
            if part
        )
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
    selection = _configured_covariate_selection(study)
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
        preferences["covariate_operationalizations"] = dict(
            study.get("covariate_operationalizations") or {}
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
                details={
                    "field": "sensitivity_specs",
                    "landmark_hours": sorted(landmarks),
                },
            )
        if landmarks:
            preferences["landmark_hours"] = next(iter(landmarks))
    return preferences


def _declared_time_windows(
    window: tuple[float, float], study: Mapping[str, Any]
) -> List[TimeWindow]:
    """Declare the materialized window as the study's analysis window.

    ``pipeline.run`` accepts ``time_windows`` and the outbound Planner contract
    publishes it as a first-class list, but the Web caller never sent one. With
    none supplied the context builder falls back to
    ``inferred_windows or default_time_windows()`` -- a generic roster whose
    third entry is ``full_stay`` 0-720h. So a study materialized to the opening
    24 hours offered the Planner a whole-stay window the data cannot support,
    while the researcher's own confirmed window was dropped.

    The tuple passed here is the same one that bounded materialization, so the
    plan cannot name a window the cohort does not carry. ``_cohort_window`` has
    already fail-closed on a missing duration and on any anchor the materializer
    cannot honour, which is what lets the anchor be stated as ICU admission.
    """

    start, end = float(window[0]), float(window[1])
    raw = study.get("time_window")
    stated = raw if isinstance(raw, Mapping) else {}
    label = _clean_text(stated.get("label"), 120)
    return [
        TimeWindow(
            name=label or f"icu_admission_0_{end:g}h",
            anchor="icu_admission",
            start_hours=start,
            end_hours=end,
            rationale=(
                "Outer feature-materialization window bound by the host before "
                "this run; the cohort carries no observation outside it."
            ),
        )
    ]


def _diagnosis_filter_applies(cohort: Mapping[str, Any]) -> bool:
    """Will the export actually run a diagnosis filter for this cohort?

    Ask the primary-cohort owner instead of restating what the conversation
    happened to type. ``primary_cohort.normalize_execution_cohort`` decides what
    counts as an executable diagnosis predicate -- it treats a structured
    include/exclude list as one in its own right, and it refuses a cohort it
    cannot execute at all (an ``icd`` preset carrying no tokens, an unsupported
    preset). Reading the owner keeps this declaration true whichever way that
    rule moves.

    What it prevents: declaring a criterion no row was ever measured against.
    The cohort ledger draws its exclusion stage from this declaration, so an
    unexecutable filter would surface in the write-up as a stage that ran --
    the same unearned claim the idea handoff used to make about an adult cohort
    nobody had chosen.
    """

    try:
        contract = dataio.normalize_export_cohort_contract(cohort)
    except Exception:  # noqa: BLE001 - an unverifiable filter is declared as none
        return False
    return bool(contract.get("icd_enabled"))


def _diagnosis_criteria(
    cohort: Mapping[str, Any], keys: tuple[str, ...], prefix: str
) -> List[str]:
    """Declare one diagnosis criterion, in the same precedence the export uses.

    ``dataio`` reads ``icd_include or include_diagnoses`` (and the exclusion
    mirror); the researcher's own wording is kept rather than the expanded token
    roster, because a stated range is what they would recognise in the write-up.

    The declaration is bounded by the same roster the export executes rather
    than by a separate literal. A private cap of 20 declared 20 of 35 stated
    criteria while 39 codes actually ran, and the cohort ledger and the
    manuscript's inclusion criteria both read this declaration -- so the
    write-up stated a narrower cohort than the one that was analysed. If a
    roster is ever still trimmed here, the declaration says so instead of
    ending silently.
    """

    if not _diagnosis_filter_applies(cohort):
        return []
    for key in keys:
        values = cohort.get(key)
        if isinstance(values, str):
            values = [part for part in values.replace(";", ",").split(",")]
        if not isinstance(values, list):
            continue
        clean = [_clean_text(item, 120) for item in values]
        clean = [item for item in clean if item]
        if not clean:
            continue
        shown = clean[: primary_cohort.MAX_DIAGNOSIS_TOKENS]
        undeclared = len(clean) - len(shown)
        stated = ", ".join(shown)
        if undeclared:
            stated += f" (+{undeclared} further stated criteria not declared)"
        return [f"{prefix}: {stated}"]
    return []


def _inclusion_criteria(study: Mapping[str, Any]) -> List[str]:
    """Compile only the criteria that say who ENTERS the cohort.

    Everything the researcher set used to arrive here, exclusions included:
    ``exclude_diagnoses`` was sent as "exclude diagnoses: ..." inside the
    inclusion list, and a readmission exclusion as "first eligible ICU stay per
    patient". The ResearchContext has carried an ``exclusion_criteria`` field
    the whole time -- the CLI fills it from ``--exclusion`` and the outbound
    Planner context publishes it as ``exclusion_contract`` -- so the Web caller
    was the only consumer that had no exclusion channel. Across 52 recorded
    plans not one carried a single exclusion predicate, and a cohort-flow
    figure cannot draw an exclusion stage that was never declared as one.
    """

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
    rows.extend(
        _diagnosis_criteria(
            cohort, ("icd_include", "include_diagnoses"), "include diagnoses"
        )
    )
    return rows[:32]


def _exclusion_criteria(study: Mapping[str, Any]) -> List[str]:
    """Compile the criteria that say who is REMOVED from the cohort.

    ``exclusion_statement`` is the prose half. Splitting the structured filter
    fields was not enough: in practice the conversation writes the cohort as one
    free-text ``review`` blob, and a removal stated there ("excluding stays that
    ended before the landmark") arrived as an inclusion criterion because
    ``review`` is the inclusion channel. Parsing that prose back apart would
    just be another renderer guessing at study semantics, so the removal half
    gets its own slot and the Copilot entrypoint is told to use it.
    """

    raw = study.get("cohort")
    cohort = raw if isinstance(raw, Mapping) else {}
    rows: List[str] = []
    stated = _clean_text(cohort.get("exclusion_statement"), 500)
    if stated:
        rows.append(stated)
    if cohort.get("exclude_readmissions") is True:
        rows.append("readmissions after the first eligible ICU stay per patient")
    rows.extend(
        _diagnosis_criteria(
            cohort, ("icd_exclude", "exclude_diagnoses"), "exclude diagnoses"
        )
    )
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

    return study_context_owner.primary_cohort_selection_mode(study)


#: Public name for the same policy. Copilot's eligibility question has to say
#: what the study would run as, and re-deriving that from the cohort fields is
#: how two layers drift apart; it reads this instead.
primary_cohort_selection_mode = _primary_cohort_selection_mode


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


def _manuscript_provenance_projection(run_dir: Path) -> Dict[str, Any]:
    """Load only the host-generated, digest-bound reader projection."""

    path = run_dir / "manuscript_provenance.json"
    manuscript_path = run_dir / "manuscript_scaffold_bound.md"
    try:
        raw = path.read_bytes()
        manuscript_raw = manuscript_path.read_bytes()
        if len(raw) > _MAX_JSON_BYTES or len(manuscript_raw) > _MAX_JSON_BYTES:
            return {}
        payload = json.loads(raw.decode("utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if payload.get("schema_version") != "easyicu.manuscript-provenance/1":
        return {}
    if payload.get("manuscript_sha256") != hashlib.sha256(manuscript_raw).hexdigest():
        return {}
    integrity = payload.get("integrity")
    if not isinstance(integrity, Mapping):
        return {}
    if any(
        integrity.get(key) is not False
        for key in (
            "path_values_returned",
            "patient_rows_returned",
            "raw_data_returned",
        )
    ):
        return {}
    return payload


def _figure_projection(run_dir: Path) -> Dict[str, Any]:
    canonical_source = _read_json(run_dir / "figure_gallery.json", {})
    source = (
        verified_presentation_gallery(run_dir, canonical_source)
        if isinstance(canonical_source, Mapping)
        else None
    ) or canonical_source
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
    presentation_variant = (
        isinstance(source, Mapping)
        and source.get("schema_version") == "easyicu.presentation-figure-gallery/1"
    )
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
        "presentation_variant": presentation_variant,
        "authority_ceiling": "analysis_only" if presentation_variant else "",
        "original_run_figures_preserved": presentation_variant,
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
                if any(is_identifier_column(value) for value in source_headers):
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
        "html": _SYSTEM_VALIDATION_DOCUMENT_SPECS["system_validation_report.html"][1],
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
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
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
        f'<meta name="easyicu-report-payload-sha256" content="{report_payload_digest}">'
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
    privacy = run_artifact_disclosure.scan_browser_projection(
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


def _projection_time_window_hours(study: Mapping[str, Any]) -> Optional[float]:
    """Project an explicit study window without inventing Planner authority."""

    if not isinstance(study.get("time_window"), Mapping) or not study.get(
        "time_window"
    ):
        return None
    return _cohort_window(study)[1]


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
            "exclusion_criteria": _exclusion_criteria(study),
            "target_outcome": _target_outcome(study),
            "time_window_hours": _projection_time_window_hours(study),
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
    plan_recommendation_complete = _plan_has_complete_reviewable_recommendation(plan)
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
    current_review_approval_allowed = _pending_plan_approval_allowed(
        run_dir=run_dir,
        pending=pending,
        plan_recommendation_complete=plan_recommendation_complete,
    )
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
    manuscript_provenance = (
        _manuscript_provenance_projection(run_dir) if run_dir else {}
    )
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
                    _pending_review_reason_code(
                        request=request,
                        plan_recommendation_complete=plan_recommendation_complete,
                        scientific_plan_review=scientific_plan_review,
                    ),
                    160,
                ),
                "summary": request.summary,
                "authority_sha256": request.authority_sha256,
                "approval_allowed": (
                    request.payload.get("approval_allowed", True)
                    and plan_recommendation_complete
                    and current_review_approval_allowed
                    if isinstance(request.payload, Mapping)
                    else plan_recommendation_complete
                    and current_review_approval_allowed
                ),
                "review_score": (
                    request.payload.get("review_score")
                    if isinstance(request.payload, Mapping)
                    else None
                ),
                "finding_codes": (
                    list(
                        dict.fromkeys(
                            [
                                *list(request.payload.get("finding_codes") or ()),
                                *(
                                    ["REVIEWABLE_PLAN_SPECIFICATION_MISSING"]
                                    if not plan_recommendation_complete
                                    else []
                                ),
                            ]
                        )
                    )[:40]
                    if isinstance(request.payload, Mapping)
                    else []
                ),
            }
            for request in pending.requests
        ]
    system_validation_applicable = bool(
        run_dir is not None
        and axes.get("execution_complete")
        and not axes.get("manuscript_ready")
        and not axes.get("paper_authorized")
    )
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
            and plan_recommendation_complete
            and current_review_approval_allowed
        ),
        "scientific_plan_review_status": (
            scientific_plan_review.get("status")
            if plan_recommendation_complete
            else "changes_required"
        ),
        "scientific_plan_review_score": scientific_plan_review.get("score"),
        "readiness": axes,
        "scientific_readiness_status": scientific_readiness["status"],
        "evidence_count": evidence_count,
        "result_table_count": result_tables.get("table_count", 0),
        "figure_count": len(figure_gallery.get("figures") or []),
        "manuscript_document_count": len(manuscript_documents),
        "draft_pdf_available": any(
            row.get("name") == "manuscript_scaffold.pdf" for row in manuscript_documents
        ),
        "system_validation_report_available": system_validation_applicable,
        "system_validation_document_count": (1 if system_validation_applicable else 0),
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
        **(
            {"manuscript_provenance.json": manuscript_provenance}
            if manuscript_provenance
            else {}
        ),
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
    privacy_scan = run_artifact_disclosure.scan_browser_projection(payloads)
    if privacy_scan["passed"] and system_validation_applicable:
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
        payloads["system_validation_report_receipt.json"] = system_validation_receipt
        privacy_scan = run_artifact_disclosure.scan_browser_projection(
            {
                **payloads,
                "system_validation_report.html": system_validation_html,
            }
        )
    if not privacy_scan["passed"]:
        payloads = run_artifact_disclosure.privacy_blocked_projection(
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


def _remove_local_recovery(wrapper_dir: Path) -> None:
    remove_recovery_seed(wrapper_dir)
    unregister_pipeline_work_root_if_unused(Path(wrapper_dir).parent.parent)


def _start_web_provider_hard_stop(
    *,
    wrapper_dir: Path,
    job_id: str,
    declaration_sha256: str,
    budget_mode: str = "full_reviewed",
    ledger_name: str = "provider_hard_stop_ledger.json",
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
    safe_ledger_name = Path(str(ledger_name or "")).name
    if not safe_ledger_name.endswith(".json"):
        raise ResearchPipelineRunError(
            "research_pipeline_provider_ledger_name_invalid",
            "The server-owned Provider ledger name is invalid.",
        )
    ledger = ProviderHardStopLedger(
        path=(runtime_dir / safe_ledger_name).resolve(),
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
    entry = _PENDING_REVIEWS.get(key)
    if entry is None:
        try:
            record = get_review_recovery_record(key)
            if record is None:
                return None
            pending = pending_from_record(record)
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
    plan_recommendation_complete = _plan_has_complete_reviewable_recommendation(
        _pending_plan_authority(pending)
    )
    run_dir = Path(pending.run_dir)
    scientific_plan_review = _load_pending_scientific_review(run_dir, pending)
    current_review_approval_allowed = _pending_plan_approval_allowed(
        run_dir=run_dir,
        pending=pending,
        plan_recommendation_complete=plan_recommendation_complete,
    )
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
                    _pending_review_reason_code(
                        request=request,
                        plan_recommendation_complete=plan_recommendation_complete,
                        scientific_plan_review=scientific_plan_review,
                    ),
                    160,
                ),
                "approval_allowed": (
                    request.payload.get("approval_allowed", True)
                    and plan_recommendation_complete
                    and current_review_approval_allowed
                    if isinstance(request.payload, Mapping)
                    else plan_recommendation_complete
                    and current_review_approval_allowed
                ),
                "review_score": (
                    request.payload.get("review_score")
                    if isinstance(request.payload, Mapping)
                    else None
                ),
                "finding_codes": (
                    list(
                        dict.fromkeys(
                            [
                                *list(request.payload.get("finding_codes") or ()),
                                *(
                                    ["REVIEWABLE_PLAN_SPECIFICATION_MISSING"]
                                    if not plan_recommendation_complete
                                    else []
                                ),
                            ]
                        )
                    )[:40]
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
        )
        and plan_recommendation_complete
        and current_review_approval_allowed,
        "scientific_plan_review": scientific_plan_review,
    }


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
    source_review = agent_runs.read_run_review(str(source_row.get("project_dir") or ""))
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


@dataclass(frozen=True)
class _ExecutionResumeTarget:
    """Exact Web wrapper and inner pipeline checkpoint selected for retry."""

    wrapper_dir: Path
    pipeline_run_id: str


def _resolve_execution_resume_wrapper(
    *,
    study: Mapping[str, Any],
    project_root: Optional[str],
    source_run_id: str,
) -> _ExecutionResumeTarget:
    """Resolve one failed, approved execution checkpoint for exact-plan retry."""

    root = Path(str(project_root or "")).expanduser().resolve()
    history = agent_runs.list_run_history(
        study_id=_clean_text(study.get("id"), 160),
        project_root=str(root),
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
            "research_pipeline_execution_retry_source_not_found",
            "The failed approved run could not be found.",
        )
    current_digest = study_context_owner.scientific_configuration_sha256(study)
    if _clean_text(source_row.get("scientific_configuration_sha256"), 80) != (
        current_digest
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_execution_retry_configuration_superseded",
            "The study configuration changed after approval; generate a new plan.",
        )
    if _clean_text(
        source_row.get("gate_reason"), 160
    ) not in EXECUTION_RETRY_REPLAYABLE_GATE_REASONS or _clean_text(
        source_row.get("run_status"), 80
    ) not in {"blocked", "failed"}:
        raise ResearchPipelineRunError(
            "research_pipeline_execution_retry_source_not_failed_execution",
            "Only a failed-closed approved execution can resume without replanning.",
        )
    wrapper_dir = Path(str(source_row.get("project_dir") or "")).resolve()
    try:
        wrapper_dir.relative_to(root)
    except ValueError as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_execution_retry_path_invalid",
            "The failed run is outside the governed project workspace.",
        ) from exc
    pipeline_root = (wrapper_dir / "pipeline").resolve()
    candidate_dirs = []
    direct = pipeline_root / source_run_id
    if direct.is_dir():
        candidate_dirs.append(direct)
    candidate_dirs.extend(
        path
        for path in sorted(pipeline_root.glob("run_*"), reverse=True)
        if path.is_dir() and path not in candidate_dirs
    )
    resumable: List[tuple[Path, Any, Mapping[str, Any]]] = []
    from easyicu.research_agent.orchestration.human_review_checkpoint import (
        load_checkpoint,
    )

    for run_dir in candidate_dirs:
        checkpoint_path = run_dir / "human_review_checkpoint.json"
        status_path = run_dir / "run_status.json"
        if not checkpoint_path.is_file() or not status_path.is_file():
            continue
        try:
            checkpoint = load_checkpoint(checkpoint_path, require_pending=False)
        except Exception:
            continue
        approved = list(checkpoint.approved_decisions or ())
        status = _read_json(status_path, {}) or {}
        gates = status.get("gates") if isinstance(status, Mapping) else {}
        execution_failed = bool(
            isinstance(gates, Mapping) and list(gates.get("failed_steps") or ())
        )
        validation_repair = bool(
            isinstance(gates, Mapping)
            and gates.get("execution_complete") is True
            and gates.get("evidence_complete") is True
            and gates.get("numeric_verified") is True
            and gates.get("analysis_validated") is False
        )
        if (
            checkpoint.state == "completed"
            and approved
            and all(str(item.get("decision") or "") == "approved" for item in approved)
            and (execution_failed or validation_repair)
        ):
            resumable.append((run_dir, checkpoint, status))
    if not resumable:
        raise ResearchPipelineRunError(
            "research_pipeline_execution_retry_checkpoint_missing",
            "The failed run has no complete approved execution checkpoint.",
        )
    if len(resumable) != 1:
        raise ResearchPipelineRunError(
            "research_pipeline_execution_retry_checkpoint_ambiguous",
            "The failed run contains more than one resumable execution checkpoint.",
        )
    run_dir, _, _ = resumable[0]
    return _ExecutionResumeTarget(
        wrapper_dir=wrapper_dir,
        pipeline_run_id=run_dir.name,
    )


def _submission_profile_ref(*, budget_mode: str, live_pubmed: bool) -> str:
    """Resolve the one submission profile a budget mode selects.

    The launch-time runtime preflight and the run itself have to agree on which
    profile will be used. Compiling that mapping twice is how a preflight ends
    up guarding a profile the run never selects, so both read it from here.
    """

    from easyicu.research_agent.orchestration.profiles import (
        CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF,
        CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF,
        CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF,
    )

    if str(budget_mode or "").strip().lower() == "full_reviewed":
        return (
            CURRENT_E1_REVIEWED_DEMO_LIVE_PUBMED_DEV_PROFILE_REF
            if live_pubmed
            else CURRENT_E1_REVIEWED_DEMO_DEV_PROFILE_REF
        )
    return (
        CURRENT_E1_PLANNER_CANARY_LIVE_PUBMED_DEV_PROFILE_REF
        if live_pubmed
        else CURRENT_E1_PLANNER_CANARY_DEV_PROFILE_REF
    )


def _require_execution_runtime(*, budget_mode: str, runner_image: str) -> None:
    """Refuse a launch whose execution backend is already known to be down.

    Whether the mandated container runtime can run is a static fact a bounded
    probe settles in seconds. Left unasked at launch, it is discovered only
    inside the pipeline -- after the provider is authorized, the data
    foundation is built and the cohort is materialized -- so a stopped daemon
    costs a minute and a half of real provider spend before saying so.
    """

    from easyicu.research_agent.execution import runner as runner_module
    from easyicu.research_agent.orchestration.profiles import get_submission_profile

    required: set[str] = set()
    for live_pubmed in (False, True):
        # Which of the two variants a run picks depends on a literature
        # binding resolved inside the run, so guard both.
        profile = get_submission_profile(
            _submission_profile_ref(budget_mode=budget_mode, live_pubmed=live_pubmed)
        )
        # A planner-only profile never launches generated code, and its runs
        # must stay startable on a host with no container runtime at all.
        if profile.planner_only:
            continue
        kind = str(profile.requires_runner or "").strip().lower()
        if kind:
            required.add(kind)
    for kind in sorted(required):
        availability = runner_module.probe_runner_availability(
            kind,
            image=(runner_image or runner_module.DockerRunner.DEFAULT_IMAGE),
        )
        if availability.available:
            continue
        raise ResearchPipelineRunError(
            "research_pipeline_execution_runtime_unavailable",
            "The governed execution runtime is not ready, so this run would "
            "fail after the provider and cohort work it is about to spend. "
            + runner_module.runner_unavailable_remediation(availability.reason_code),
            details={
                "owner": "easyicu.research_agent.execution.runner",
                "reason_code": availability.reason_code,
                "runner_kind": availability.kind,
                "runner_image": availability.image,
            },
        )


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
    execution_resume_source_run_id: str = "",
    development_resume_source_job_id: str = "",
    budget_mode: str = "planner_canary",
    runner_image: Optional[str] = None,
) -> Any:
    """Build the JobManager runner for a real, evidence-bound pipeline run."""

    prepared = prepare_research_pipeline_run(
        ResearchPipelineLaunchRequest(
            export_path=export_path,
            study_context=study_context,
            project_root=project_root,
            provider=provider,
            provider_environment=provider_environment,
            credential_source=credential_source,
            literature_search_authorized=literature_search_authorized,
            plan_revision_source_run_id=plan_revision_source_run_id,
            execution_resume_source_run_id=execution_resume_source_run_id,
            development_resume_source_job_id=development_resume_source_job_id,
            budget_mode=budget_mode,
            runner_image=runner_image,
        )
    )

    def runner(job: Any) -> Dict[str, Any]:
        scientific = prepared.scientific
        authority = prepared.authority
        execution = prepared.execution
        export_path = execution.export_path
        study = scientific.study
        project_root = execution.project_root
        provider = authority.provider
        literature_search_authorized = authority.literature_search_authorized
        question = scientific.question
        database = scientific.database
        selected_budget_mode = execution.budget_mode
        configured_primary_exposure = scientific.configured_primary_exposure
        target = scientific.target
        primary_exposure = scientific.primary_exposure
        covariates = scientific.covariates
        covariate_selection = scientific.covariate_selection
        sensitivity_specs = scientific.sensitivity_specs
        window = scientific.cohort_window
        validated_analysis_design = scientific.validated_analysis_design
        patient_grouping = scientific.patient_grouping
        metadata_only_planning = scientific.metadata_only_planning
        candidate_planning_study = scientific.materialization_study
        metadata_planning_coordinates = scientific.metadata_planning_coordinates
        execution_concepts = scientific.execution_concepts
        planning_exposure_source = scientific.planning_exposure_source
        metadata_operationalized_columns = scientific.metadata_operationalized_columns
        prepared_package_binding = scientific.prepared_package_binding
        foundation_profile = scientific.foundation_profile
        selected_credential_source = authority.credential_source
        development_resume_binding = execution.development_resume_binding
        development_resume_acquisition = execution.development_resume_acquisition
        development_resume_literature = execution.development_resume_literature
        publication_skill_flags = authority.publication_skill_flags
        user_extension_activation = authority.user_extension_activation
        research_provider_environment = authority.provider_environment
        source_run_id = execution.plan_revision_source_run_id
        execution_resume_run_id = execution.execution_resume_source_run_id
        selected_runner_image = execution.runner_image
        if prepared_package_binding is not None:
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
        execution_resume_target = (
            _resolve_execution_resume_wrapper(
                study=study,
                project_root=project_root,
                source_run_id=execution_resume_run_id,
            )
            if execution_resume_run_id
            else None
        )
        wrapper_dir = (
            execution_resume_target.wrapper_dir
            if execution_resume_target is not None
            else root / _slug(study.get("id")) / f"run_{job.id}"
        )
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
            request_timeout=(
                _DEVELOPMENT_PROVIDER_REQUEST_TIMEOUT_SECONDS
                if selected_budget_mode != "full_reviewed"
                else None
            ),
            request_hard_timeout=(
                _DEVELOPMENT_PROVIDER_REQUEST_TIMEOUT_SECONDS
                if selected_budget_mode != "full_reviewed"
                else None
            ),
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
                ledger_name=(
                    f"provider_hard_stop_retry_{_slug(job.id)}.json"
                    if execution_resume_target is not None
                    else "provider_hard_stop_ledger.json"
                ),
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
                label=(
                    "Selecting concepts from database metadata; no patient data "
                    "will be read"
                    if metadata_only_planning
                    else "Selecting concepts and materializing a typed analysis universe"
                ),
            )
            if (
                metadata_only_planning
                and development_resume_acquisition is not None
                and development_resume_acquisition.kind
                == "metadata_only_planning_catalog"
            ):
                acquisition = _restore_metadata_only_planning_acquisition(
                    database=database,
                    profile=development_resume_acquisition,
                    output_dir=wrapper_dir / "pipeline_input",
                    endpoint=metadata_planning_coordinates.get("endpoint"),
                    patient_grouping=patient_grouping,
                    operationalized_columns=metadata_operationalized_columns,
                )
            elif metadata_only_planning:
                acquisition = _metadata_only_planning_acquisition(
                    database=database,
                    question=question,
                    llm=acquisition_client,
                    output_dir=wrapper_dir / "pipeline_input",
                    target_outcome=metadata_planning_coordinates.get("target_outcome"),
                    endpoint=metadata_planning_coordinates.get("endpoint"),
                    required_concepts=(
                        target,
                        primary_exposure,
                        metadata_planning_coordinates.get("target_outcome"),
                        metadata_planning_coordinates.get("primary_exposure"),
                        *covariates,
                        *(
                            variable
                            for spec in sensitivity_specs
                            for variable in spec.source_materialization_variables
                        ),
                    ),
                    patient_grouping=patient_grouping,
                    operationalized_columns=metadata_operationalized_columns,
                )
            else:
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
                    outcome_concepts=(
                        development_resume_acquisition.outcome_concepts
                        if development_resume_acquisition is not None
                        else foundation_profile["outcome_concepts"]
                    ),
                    required_feature_concepts=(
                        development_resume_acquisition.feature_concepts
                        if development_resume_acquisition is not None
                        else foundation_profile["required_feature_concepts"]
                    ),
                    static_concepts=(
                        development_resume_acquisition.static_concepts
                        if development_resume_acquisition is not None
                        else foundation_profile["static_concepts"]
                    ),
                    allowed_modules=foundation_profile["allowed_modules"],
                    concept_selection_authority=(
                        "host_exact"
                        if (
                            covariate_selection == "exact"
                            or development_resume_acquisition is not None
                        )
                        else "agent_selectable"
                    ),
                    cohort_window=window,
                    database=database,
                    require_outcome=foundation_profile["require_outcome"],
                    # A verified composite patient/stay identity is needed by the
                    # cluster-robust model. The current materializer deliberately
                    # refuses to attach a private mapping to a longitudinal table;
                    # this fixed stay-level design therefore requests no unused
                    # trajectory instead of silently publishing an ungrouped one.
                    emit_trajectory=(
                        patient_grouping is None
                        and _analysis_requires_longitudinal_trajectory(
                            candidate_planning_study,
                            validated_design=validated_analysis_design,
                        )
                    ),
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
            if configured_primary_exposure and not metadata_only_planning:
                resolved_primary_exposure = _resolve_materialized_primary_exposure(
                    configured=configured_primary_exposure,
                    source_concept=foundation_profile.get(
                        "primary_exposure_source_concept"
                    ),
                    aggregation=_primary_exposure_aggregation(study),
                    acquisition=acquisition,
                )
                if not resolved_primary_exposure:
                    raise ResearchPipelineRunError(
                        "research_pipeline_primary_exposure_aggregation_required",
                        "The configured primary exposure requires an explicit "
                        "analysis aggregation before planning can start.",
                    )
            elif primary_exposure and not metadata_only_planning:
                resolved_primary_exposure = _resolve_planner_proposed_primary_exposure(
                    source_concept=str(
                        foundation_profile.get("primary_exposure_source_concept")
                        or primary_exposure
                    ),
                    acquisition=acquisition,
                )
                if not resolved_primary_exposure:
                    raise ResearchPipelineRunError(
                        "research_pipeline_planner_exposure_proposal_unavailable",
                        "The Planner could not bind a reviewable materialized representation for the exposure named in the research question.",
                    )
            pipeline_target = target
            if target and not metadata_only_planning:
                pipeline_target = _resolve_materialized_target_outcome(
                    source_concept=str(target),
                    acquisition=acquisition,
                )
                if not pipeline_target:
                    raise ResearchPipelineRunError(
                        "research_pipeline_target_outcome_materialization_unavailable",
                        "The selected outcome is not available as a verified analysis column in the materialized cohort.",
                        details={
                            "source_concept": str(target),
                            "available_analysis_columns": dict(
                                getattr(acquisition, "analysis_columns", {}) or {}
                            ),
                        },
                    )
            if metadata_only_planning:
                execution_concepts = study.get("execution_concepts")
                execution_concepts = (
                    execution_concepts
                    if isinstance(execution_concepts, Mapping)
                    else {}
                )
                pipeline_target = (
                    _clean_text(execution_concepts.get("outcome"), 160)
                    or metadata_planning_coordinates.get("target_outcome")
                    or None
                )
                resolved_primary_exposure = (
                    _clean_text(
                        execution_concepts.get("primary_exposure"),
                        160,
                    )
                    or metadata_planning_coordinates.get("primary_exposure")
                    or None
                )
                aggregation = _primary_exposure_aggregation(study)
                if resolved_primary_exposure and aggregation:
                    resolved_primary_exposure = (
                        f"{resolved_primary_exposure}_{aggregation}"
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
            if development_resume_literature is not None:
                bound_preplan_literature = development_resume_literature
            live_pubmed_requested = (
                bool(literature_search_authorized) and bound_preplan_literature is None
            )
            submission_profile_ref = _submission_profile_ref(
                budget_mode=selected_budget_mode,
                live_pubmed=live_pubmed_requested,
            )
            submission_profile = get_submission_profile(submission_profile_ref)
            profile_options = submission_profile.pipeline_options()
            profile_options.update(
                {
                    "enable_memory": False,
                    "enable_experience_bank": False,
                    "enable_reviewed_memory": False,
                    "reviewed_memory_namespaces": (),
                    # Copilot presents one complete digest-bound plan for the
                    # operator to approve. Runtime replanning would replace
                    # that approved plan without a second durable review.
                    "enable_replanning": False,
                }
            )
            from easyicu.webserver.scientific_runtime_projection import (
                WebScientificRuntimeProjectionError,
                compile_landmark_spline_runtime_projection,
            )
            from easyicu.research_agent.contracts.dependence import (
                PlannedDependenceRequirement,
            )

            planning_endpoint = metadata_planning_coordinates.get("endpoint")
            runtime_primary_exposure_source = str(
                foundation_profile.get("primary_exposure_source_concept")
                or planning_exposure_source
                or primary_exposure
                or ""
            ).strip()
            runtime_projection_specs = _runtime_projection_sensitivity_specs(
                sensitivity_specs,
                primary_exposure_source=runtime_primary_exposure_source,
            )
            try:
                runtime_projection = compile_landmark_spline_runtime_projection(
                    study=candidate_planning_study,
                    sensitivity_specs=runtime_projection_specs,
                    primary_exposure=resolved_primary_exposure,
                    primary_exposure_source=runtime_primary_exposure_source,
                    target_outcome=pipeline_target,
                    declared_covariates=covariates,
                    covariate_operationalizations=dict(
                        study.get("covariate_operationalizations") or {}
                    ),
                    target_is_event_status=(
                        bool(foundation_profile.get("require_outcome"))
                        or (
                            metadata_only_planning
                            and str(getattr(planning_endpoint, "kind", ""))
                            .strip()
                            .casefold()
                            == "binary"
                        )
                    ),
                    dependence=(
                        PlannedDependenceRequirement(
                            group_source=patient_grouping.output_identity_column,
                            group_derivation="prefix_before_delimiter",
                            delimiter=":s",
                        )
                        if patient_grouping is not None
                        else None
                    ),
                    universe_path=Path(acquisition.universe_path),
                    scientific_configuration_sha256=(
                        study_context_owner.scientific_configuration_sha256(study)
                    ),
                )
            except WebScientificRuntimeProjectionError as exc:
                raise ResearchPipelineRunError(
                    exc.code,
                    str(exc),
                    details=exc.details,
                ) from exc
            if runtime_projection is not None:
                profile_options.update(
                    {
                        "current_case_scientific_runtime_authority": (
                            runtime_projection.authority
                        ),
                        "scientific_runtime_projection_sha256": (
                            runtime_projection.projection_sha256
                        ),
                    }
                )
            if development_resume_binding is not None:
                profile_options.update(
                    {
                        "development_progressive_resume_checkpoint_path": (
                            development_resume_binding[0]
                        ),
                        "development_progressive_resume_checkpoint_sha256": (
                            development_resume_binding[1]
                        ),
                        "development_progressive_resume_reuse_bound_literature": True,
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
                bound_plan_revision_contract=(bound_plan_revision_contract or None),
                # Live PubMed is frozen by the selected additive profile, not
                # passed as an ad-hoc override. When an accepted Idea handoff
                # already supplies a digest-bound receipt, the no-search
                # profile above prevents a silent second retrieval.
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
                # User-requested Web planning is already bounded by the durable
                # provider hard stop above.  The smaller routine-E1 iteration
                # envelope is intentionally not applied here: progressive plans
                # can require more than six valid, checkpointed calls even when
                # every remaining step succeeds on its first attempt.
                development_planner_efficiency_max_calls=None,
                development_planner_efficiency_max_reported_tokens=None,
                development_planner_efficiency_max_wall_seconds=None,
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
                candidate_planning_study,
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
                target_outcome=pipeline_target,
                endpoint=acquisition.endpoint,
                primary_exposure=resolved_primary_exposure,
                inclusion_criteria=_inclusion_criteria(study),
                exclusion_criteria=_exclusion_criteria(study),
                time_windows=_declared_time_windows(window, study),
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
                resume_run_id=(
                    execution_resume_target.pipeline_run_id
                    if execution_resume_target is not None
                    else None
                ),
                progress_callback=lambda event: _pipeline_progress(job, event),
            )
            if execution_resume_target is not None and isinstance(
                outcome, HumanReviewPending
            ):
                raise ResearchPipelineRunError(
                    "research_pipeline_execution_retry_unexpected_plan_review",
                    "The exact-plan execution retry unexpectedly requested a new "
                    "plan review; no approval was created or overwritten.",
                )
            if isinstance(outcome, HumanReviewPending):
                run_dir = Path(outcome.run_dir)
                put_review_recovery_record(recovery_seed.record(str(outcome.run_id)))
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
                _PENDING_REVIEWS.register(entry)
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
        except ResearchPipelineRunError as exc:
            _finish_web_provider_hard_stop(
                provider_hard_stop,
                error="research_pipeline_error",
            )
            diagnostic = _write_pipeline_failure_diagnostic(
                wrapper_dir=wrapper_dir,
                exc=exc,
                code=exc.code,
            )
            _write_pipeline_failure_projection(
                wrapper_dir=wrapper_dir,
                study=study,
                provider=provider_public,
                code=exc.code,
                failure_type=_pipeline_failure_category(exc),
                diagnostic=diagnostic,
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
            _write_pipeline_failure_projection(
                wrapper_dir=wrapper_dir,
                study=study,
                provider=provider_public,
                code=code,
                failure_type=_pipeline_failure_category(exc),
                diagnostic=diagnostic,
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
            elif code == "research_pipeline_planner_efficiency_budget_exhausted":
                message = (
                    "The development Planner reached its call, token, or time budget. "
                    "Its validated checkpoint prefix was preserved; no analysis was run."
                )
            elif code == "research_pipeline_progressive_compile_failed":
                message = (
                    "The deterministic host compiler rejected the bounded Planner "
                    "repairs. A local replay artifact was preserved; no analysis was run."
                )
            elif code == "research_pipeline_execution_runtime_unavailable":
                # A host-environment failure, not a scientific one. Saying so
                # is the whole point: the generic wording sent the researcher
                # looking for a problem in their study. Reported through the
                # raised code and message only -- ``_progress`` raises on a
                # pending cancellation, which would replace this cause.
                message = (
                    "The container runtime that executes analysis code was not "
                    "available, so no analysis was run. Start it and run again."
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
    entry = _PENDING_REVIEWS.get(key)
    if entry is None:
        try:
            entry = recover_pending_review(
                key,
                provider_environment=provider_environment,
                reviewer_identity=server_reviewer_identity(),
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
        entry = _PENDING_REVIEWS.install_recovered(key, entry)
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
            _PENDING_REVIEWS.discard(key, expected=entry)
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
    if resolved == "approved" and not _plan_has_complete_reviewable_recommendation(
        _pending_plan_authority(entry.pending)
    ):
        raise ResearchPipelineRunError(
            "research_pipeline_reviewable_plan_required",
            "This candidate predates the complete reviewable-plan contract and "
            "must be regenerated before it can be approved.",
        )
    if resolved == "approved":
        current_review = _load_pending_scientific_review(
            Path(entry.pending.run_dir), entry.pending
        )
        if not current_review:
            raise ResearchPipelineRunError(
                "scientific_plan_review_policy_stale",
                "This plan was reviewed under an older scientific policy and must be regenerated before approval.",
            )
        if current_review.get("approval_allowed") is not True:
            raise ResearchPipelineRunError(
                "scientific_plan_review_changes_required",
                "The current scientific review requires plan changes before approval.",
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
    try:
        resume_result = resume_pending_review(
            _PENDING_REVIEWS,
            entry,
            decisions,
            run_id=key,
            progress_callback=lambda event: _pipeline_progress(job, event),
        )
    except PendingReviewResumeInProgress as exc:
        raise ResearchPipelineRunError(
            "research_pipeline_review_resume_in_progress",
            "This plan review is already being resumed by another job.",
        ) from exc
    except PendingReviewResumeFailure as failure:
        exc = failure.cause
        remains_resumable = failure.resumable
        diagnostic = _write_review_resume_failure_diagnostic(
            wrapper_dir=entry.wrapper_dir,
            exc=exc,
            review_resumable=remains_resumable,
        )
        if not remains_resumable:
            _finish_web_provider_hard_stop(
                entry.provider_hard_stop,
                error=_pipeline_failure_category(exc),
            )
            remove_review_recovery_record(key)
            _remove_local_recovery(entry.wrapper_dir)
        runtime_unavailable = (
            _safe_pipeline_typed_failure(exc).get("owner")
            == _EXECUTION_RUNTIME_DIAGNOSTIC_OWNER
        )
        raise ResearchPipelineRunError(
            (
                "research_pipeline_execution_runtime_unavailable"
                if runtime_unavailable
                else "research_pipeline_review_resume_failed"
            ),
            (
                "The container runtime that executes analysis code was not "
                "available, so the approved plan did not run. Start it and "
                "resume again."
                if runtime_unavailable
                else "The governed Research Agent run could not resume after "
                "plan review."
            ),
            details={
                "failure_type": _pipeline_failure_category(exc),
                "review_resumable": remains_resumable,
                "diagnostic": diagnostic,
            },
        ) from exc
    if resume_result.state == "rejected":
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
    if resume_result.state == "pending":
        outcome = resume_result.outcome
        return _write_projection(
            wrapper_dir=entry.wrapper_dir,
            study=entry.study,
            provider=entry.provider,
            acquisition=entry.acquisition,
            run_dir=Path(outcome.run_dir),
            pending=outcome,
        )
    outcome = resume_result.outcome
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
    if not run_artifact_disclosure.scan_browser_projection(
        {"literature_evidence.json": payload}
    )["passed"]:
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
