"""Discovery-to-analysis handoff contracts.

This module is the hard boundary between idea mining and the downstream
research-agent workflow. It mirrors the shape used by end-to-end scientist
systems: a structured idea record is selected, frozen, and passed forward as
the research task's provenance rather than being rewritten by hand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


DISCOVERY_HANDOFF_SCHEMA_VERSION = "easyicu.discovery_handoff/1"

DiscoverySelectionMode = Literal[
    "agent_selected",
    "human_curated",
    "manual_scaffold",
]


DEFAULT_DISCOVERY_FIGURE_ROLES = [
    "discovery_provenance",
    "cohort_evaluability",
    "primary_result",
    "audit_reproducibility",
]


class DiscoveryHandoffPacket(BaseModel):
    """Frozen packet passed from idea mining into analysis/writing."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = DISCOVERY_HANDOFF_SCHEMA_VERSION
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_triage_report_path: str
    selection_mode: DiscoverySelectionMode = "agent_selected"
    selection_rationale: str
    literature_idea_id: str
    executable_candidate_id: Optional[str] = None
    candidate_topic: str
    literature_source: Optional[str] = None
    gap_evidence_quote: Optional[str] = None
    go_no_go: str
    go_no_go_reason: str
    novelty_label: Optional[str] = None
    feasibility_route: Optional[str] = None
    feasibility_next_action: Optional[str] = None
    resolved_predictor_concept: Optional[str] = None
    resolved_outcome_concept: Optional[str] = None
    target_outcome: str = "death"
    database: str = "miiv"
    research_question: str
    inclusion_criteria: List[str] = Field(default_factory=list)
    selected_ledger_row: Dict[str, Any] = Field(default_factory=dict)
    required_manuscript_figure_roles: List[str] = Field(
        default_factory=lambda: list(DEFAULT_DISCOVERY_FIGURE_ROLES)
    )

    @field_validator(
        "source_triage_report_path",
        "selection_rationale",
        "literature_idea_id",
        "candidate_topic",
        "go_no_go",
        "go_no_go_reason",
        "target_outcome",
        "database",
        "research_question",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text


def load_discovery_ledger(triage_report_path: str | Path) -> List[Dict[str, Any]]:
    """Load flat discovery rows from a candidate triage report."""

    path = Path(triage_report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("discovery_ledger")
    if rows is None and isinstance(payload.get("discovery_records"), list):
        rows = payload["discovery_records"]
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a discovery_ledger list")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def select_discovery_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    index: Optional[int] = None,
) -> Dict[str, Any]:
    """Select the highest-priority idea row for downstream execution.

    The ranking is deterministic so the handoff is reproducible. The selected
    row is still marked ``agent_selected`` by callers only when the source rows
    came from an agentic idea-mining run; human overrides should use
    ``human_curated``.
    """

    if index is not None:
        if index < 0 or index >= len(rows):
            raise IndexError(f"idea index {index} out of range for {len(rows)} rows")
        return dict(rows[index])
    if not rows:
        raise ValueError("cannot select from an empty discovery ledger")
    return dict(max(rows, key=_row_priority))


def build_handoff_from_row(
    row: Mapping[str, Any],
    *,
    triage_report_path: str | Path,
    selection_mode: DiscoverySelectionMode = "agent_selected",
    selection_rationale: Optional[str] = None,
    target_outcome: str = "death",
    database: str = "miiv",
    research_question: Optional[str] = None,
    inclusion_criteria: Optional[Sequence[str]] = None,
) -> DiscoveryHandoffPacket:
    """Create a frozen handoff packet from one discovery-ledger row."""

    topic = str(row.get("candidate_topic") or "").strip()
    if not topic:
        raise ValueError("selected discovery row has no candidate_topic")
    question = research_question or _default_research_question(topic)
    rationale = selection_rationale or _default_selection_rationale(row)
    return DiscoveryHandoffPacket(
        source_triage_report_path=str(Path(triage_report_path)),
        selection_mode=selection_mode,
        selection_rationale=rationale,
        literature_idea_id=str(row.get("literature_idea_id") or ""),
        executable_candidate_id=(
            str(row.get("executable_candidate_id"))
            if row.get("executable_candidate_id")
            else None
        ),
        candidate_topic=topic,
        literature_source=(
            str(row.get("literature_source"))
            if row.get("literature_source")
            else None
        ),
        gap_evidence_quote=(
            str(row.get("gap_evidence_quote"))
            if row.get("gap_evidence_quote")
            else None
        ),
        go_no_go=str(row.get("go_no_go") or ""),
        go_no_go_reason=str(row.get("go_no_go_reason") or ""),
        novelty_label=(
            str(row.get("novelty_label")) if row.get("novelty_label") else None
        ),
        feasibility_route=(
            str(row.get("feasibility_route"))
            if row.get("feasibility_route")
            else None
        ),
        feasibility_next_action=(
            str(row.get("feasibility_next_action"))
            if row.get("feasibility_next_action")
            else None
        ),
        resolved_predictor_concept=(
            str(row.get("resolved_predictor_concept"))
            if row.get("resolved_predictor_concept")
            else None
        ),
        resolved_outcome_concept=(
            str(row.get("resolved_outcome_concept"))
            if row.get("resolved_outcome_concept")
            else None
        ),
        target_outcome=target_outcome,
        database=database,
        research_question=question,
        inclusion_criteria=list(inclusion_criteria or _default_inclusion_criteria()),
        selected_ledger_row=dict(row),
    )


def write_handoff_packet(packet: DiscoveryHandoffPacket, path: str | Path) -> Path:
    """Write a handoff packet as JSON and return the output path."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        packet.model_dump_json(indent=2), encoding="utf-8"
    )
    return out


def _row_priority(row: Mapping[str, Any]) -> tuple:
    go = str(row.get("go_no_go") or "").lower()
    novelty = str(row.get("novelty_label") or "").lower()
    direct_same_topic = len(row.get("direct_same_topic_pmids") or [])
    differentiators = len(row.get("differentiators") or [])
    risks = len(row.get("risks") or [])
    go_score = {"recommend": 3, "hold": 2, "db-cannot-do": 0}.get(go, 1)
    novelty_score = {
        "apparently_gap": 3,
        "sparse": 2,
        "crowded_but_differentiable": 1,
        "already_done": 0,
    }.get(novelty, 1)
    return (
        go_score,
        novelty_score,
        differentiators,
        -direct_same_topic,
        -risks,
    )


def _default_research_question(topic: str) -> str:
    return (
        "Starting from the agent-mined ICU literature idea, evaluate the "
        f"following candidate in an adult ICU cohort: {topic}. The analysis "
        "must first document the literature provenance, data feasibility, "
        "cohort construction, component evaluability, and missingness. Any "
        "outcome association is exploratory only and must be blocked rather "
        "than inferred when explicit row-level grouping/status fields are not "
        "certified."
    )


def _default_inclusion_criteria() -> List[str]:
    return [
        "Universe = all ICU stays in the prepared EasyICU export; the research "
        "agent must define and justify the adult analytic cohort in-sandbox.",
    ]


def _default_selection_rationale(row: Mapping[str, Any]) -> str:
    return (
        "Selected from the flat discovery ledger using reproducible priority "
        f"ranking: go_no_go={row.get('go_no_go')}, "
        f"novelty_label={row.get('novelty_label')}, "
        f"differentiators={len(row.get('differentiators') or [])}, "
        f"direct_same_topic_pmids={len(row.get('direct_same_topic_pmids') or [])}."
    )


__all__ = [
    "DISCOVERY_HANDOFF_SCHEMA_VERSION",
    "DEFAULT_DISCOVERY_FIGURE_ROLES",
    "DiscoveryHandoffPacket",
    "DiscoverySelectionMode",
    "build_handoff_from_row",
    "load_discovery_ledger",
    "select_discovery_row",
    "write_handoff_packet",
]
