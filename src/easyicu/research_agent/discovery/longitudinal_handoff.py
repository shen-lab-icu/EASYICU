"""Build analysis task packs from longitudinal Idea Mining evidence.

The pack is intentionally protocol-pending.  Idea Mining can prove that a
repeated concept is available across databases, but it cannot choose time zero,
window, trajectory representation, or clustering/stability thresholds.  Those
scientific choices remain explicit confirmations before any child handoff can
be promoted from ``hold`` to ``recommend``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .discovery_handoff import build_handoff_from_row, write_handoff_packet

LONGITUDINAL_TASK_PACK_SCHEMA_VERSION = "easyicu.longitudinal_analysis_task_pack/1"

DEFAULT_LONGITUDINAL_PROTOCOL_CONFIRMATIONS = [
    "time_zero",
    "observation_window",
    "minimum_measurement_support",
    "time_grid_and_aggregation",
    "trajectory_representation",
    "class_or_feature_selection_method",
    "within_database_stability_threshold",
    "cross_database_matching_and_transportability_metric",
    "outcome_blind_class_discovery",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LongitudinalDatabaseTask(BaseModel):
    """One database child of a cross-database longitudinal task pack."""

    model_config = ConfigDict(extra="forbid")

    database: str
    artifact_path: str
    artifact_sha256: str
    row_count: int = Field(ge=1)
    id_column: str
    time_column: str
    value_column: str
    handoff_path: Optional[str] = None

    @field_validator(
        "database",
        "artifact_path",
        "artifact_sha256",
        "id_column",
        "time_column",
        "value_column",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


class LongitudinalAnalysisTaskPack(BaseModel):
    """Frozen parent task plus database-specific child handoffs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = LONGITUDINAL_TASK_PACK_SCHEMA_VERSION
    created_at: str = Field(default_factory=_utc_now_iso)
    source_manifest_path: str
    source_manifest_sha256: str
    candidate_id: str
    concept: str
    analysis_family: str = "trajectory_clustering"
    design_archetype: str = "cross_database_trajectory_transportability"
    research_question: str
    go_no_go: str = "hold"
    go_no_go_reason: str
    protocol_status: str = "awaiting_human_confirmation"
    required_protocol_confirmations: List[str] = Field(
        default_factory=lambda: list(DEFAULT_LONGITUDINAL_PROTOCOL_CONFIRMATIONS)
    )
    database_tasks: List[LongitudinalDatabaseTask]
    novelty_claimed: bool = False
    scientific_result_claimed: bool = False
    paper_authorized: bool = False

    @model_validator(mode="after")
    def _validate_pack(self) -> "LongitudinalAnalysisTaskPack":
        if self.analysis_family != "trajectory_clustering":
            raise ValueError("longitudinal task pack must use trajectory_clustering")
        if len({task.database for task in self.database_tasks}) != len(
            self.database_tasks
        ):
            raise ValueError("database tasks must be unique by database")
        if not self.database_tasks:
            raise ValueError("task pack requires at least one database task")
        if self.go_no_go != "hold" or self.paper_authorized:
            raise ValueError("unreviewed task pack must remain hold and unauthorized")
        return self


def build_longitudinal_analysis_task_pack(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    concept: Optional[str] = None,
) -> LongitudinalAnalysisTaskPack:
    """Convert one readiness candidate into frozen per-database handoffs."""

    source = Path(manifest_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("longitudinal discovery manifest has no candidates")
    selected = _select_candidate(raw_candidates, concept=concept)
    selected_concept = str(selected.get("concept") or "").strip()
    family = str(selected.get("analysis_family") or "").strip()
    if family != "trajectory_clustering" or not selected_concept:
        raise ValueError("selected longitudinal candidate has no trajectory concept")
    profiles = selected.get("artifact_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("selected longitudinal candidate has no artifact profiles")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    question = (
        f"Are prespecified longitudinal {selected_concept} trajectory features "
        "or classes reproducible within databases and transportable across the "
        "prepared ICU databases?"
    )
    row: Dict[str, Any] = {
        "literature_idea_id": f"longitudinal_{selected_concept}_transportability",
        "executable_candidate_id": f"trajectory_{selected_concept}_six_database",
        "candidate_topic": (
            f"Cross-database transportability of longitudinal {selected_concept} "
            "trajectories"
        ),
        "analysis_family": family,
        "resolved_analysis_concepts": [selected_concept],
        "go_no_go": "hold",
        "go_no_go_reason": (
            "Repeated measurements are data-ready, but time zero, observation "
            "window, trajectory representation, stability thresholds, and "
            "cross-database matching require explicit protocol confirmation."
        ),
        "feasibility_route": "cross_database_protocol_review",
        "feasibility_next_action": (
            "Confirm the frozen longitudinal protocol, then promote the child "
            "handoffs through the standard human-confirmed analysis gate."
        ),
        "target_databases": [
            str(profile.get("database") or "") for profile in profiles
        ],
        "source_longitudinal_manifest_sha256": _sha256(source),
        "required_protocol_confirmations": list(
            DEFAULT_LONGITUDINAL_PROTOCOL_CONFIRMATIONS
        ),
    }
    ledger_path = out / "candidate_triage_report.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.longitudinal_candidate_triage/1",
                "source_manifest_path": str(source),
                "source_manifest_sha256": _sha256(source),
                "discovery_ledger": [row],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    database_tasks: List[LongitudinalDatabaseTask] = []
    for profile in profiles:
        database = str(profile.get("database") or "").strip()
        child_dir = out / "databases" / database
        handoff = build_handoff_from_row(
            row,
            triage_report_path=ledger_path,
            database=database,
            research_question=question,
            human_confirmed=False,
        )
        handoff_path = write_handoff_packet(
            handoff,
            child_dir / "discovery_handoff.json",
        )
        database_tasks.append(
            LongitudinalDatabaseTask(
                database=database,
                artifact_path=str(profile.get("artifact_path") or ""),
                artifact_sha256=str(profile.get("artifact_sha256") or ""),
                row_count=int(profile.get("row_count") or 0),
                id_column=str(profile.get("id_column") or ""),
                time_column=str(profile.get("time_column") or ""),
                value_column=str(profile.get("value_column") or ""),
                handoff_path=str(handoff_path),
            )
        )

    pack = LongitudinalAnalysisTaskPack(
        source_manifest_path=str(source),
        source_manifest_sha256=_sha256(source),
        candidate_id=str(row["executable_candidate_id"]),
        concept=selected_concept,
        research_question=question,
        go_no_go_reason=str(row["go_no_go_reason"]),
        database_tasks=database_tasks,
    )
    (out / "longitudinal_analysis_task_pack.json").write_text(
        pack.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return pack


def _select_candidate(
    candidates: List[Mapping[str, Any]],
    *,
    concept: Optional[str],
) -> Mapping[str, Any]:
    if concept is None:
        if len(candidates) != 1:
            raise ValueError(
                "--concept is required when the manifest has multiple candidates"
            )
        return candidates[0]
    wanted = str(concept).strip().lower()
    matches = [
        candidate
        for candidate in candidates
        if str(candidate.get("concept") or "").strip().lower() == wanted
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one longitudinal candidate for concept {concept!r}")
    return matches[0]


__all__ = [
    "DEFAULT_LONGITUDINAL_PROTOCOL_CONFIRMATIONS",
    "LONGITUDINAL_TASK_PACK_SCHEMA_VERSION",
    "LongitudinalAnalysisTaskPack",
    "LongitudinalDatabaseTask",
    "build_longitudinal_analysis_task_pack",
]
