"""Server-owned development checkpoint and acquisition restoration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.planning.progressive_artifacts import (
    ProgressivePlanningArtifactError,
    load_progressive_planner_checkpoint_chain,
)
from easyicu.webserver.research_launch_scientific import (
    _normalized_metadata_planning_operationalized_columns,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

_MAX_JSON_BYTES = 2 * 1024 * 1024


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
