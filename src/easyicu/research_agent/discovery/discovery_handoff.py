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

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from ..canonical_json import canonical_sha256, sha256_file
from ..contracts.frozen_payload import freeze_payload, thaw_payload
from ..authority.filesystem import publish_write_once_bytes

from ..planning.analysis_types import (
    is_concept_set_family,
    normalize_analysis_family,
)

DISCOVERY_HANDOFF_SCHEMA_VERSION = "easyicu.discovery_handoff/4"

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

ANALYSIS_READY_DECISIONS = frozenset({"go", "recommend"})


class DiscoveryHandoffPacket(BaseModel):
    """Sealed proposal provenance; never a substitute for plan/execution approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.discovery_handoff/4"] = DISCOVERY_HANDOFF_SCHEMA_VERSION
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_triage_report_path: str
    source_triage_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    handoff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_mode: DiscoverySelectionMode = "agent_selected"
    selection_rationale: str
    literature_idea_id: str
    executable_candidate_id: Optional[str] = None
    candidate_topic: str
    literature_source: Optional[str] = None
    gap_evidence_quote: Optional[str] = None
    go_no_go: str
    go_no_go_reason: str
    human_confirmed: bool = False
    human_confirmed_at: Optional[str] = None
    human_confirmation_note: Optional[str] = None
    novelty_label: Optional[str] = None
    feasibility_route: Optional[str] = None
    feasibility_next_action: Optional[str] = None
    analysis_family: str = "association_study"
    resolved_analysis_concepts: tuple[str, ...] = ()
    resolved_predictor_concept: Optional[str] = None
    resolved_outcome_concept: Optional[str] = None
    target_outcome: Optional[str] = None
    database: str = "miiv"
    research_question: str
    inclusion_criteria: tuple[str, ...] = ()
    selected_ledger_row: Mapping[str, Any]
    required_manuscript_figure_roles: tuple[str, ...] = Field(
        default_factory=lambda: tuple(DEFAULT_DISCOVERY_FIGURE_ROLES)
    )

    @field_validator("analysis_family", mode="before")
    @classmethod
    def _canonical_family(cls, value: Any) -> str:
        return normalize_analysis_family(value)

    @field_validator("resolved_analysis_concepts", mode="before")
    @classmethod
    def _canonical_concepts(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("resolved_analysis_concepts must be a sequence")
        return tuple(dict.fromkeys(str(item).strip() for item in (value or ()) if str(item).strip()))

    @field_validator("selected_ledger_row")
    @classmethod
    def _freeze_row(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_payload(value)

    @field_serializer("selected_ledger_row")
    def _row_projection(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_payload(value)

    @field_validator(
        "source_triage_report_path",
        "selection_rationale",
        "literature_idea_id",
        "candidate_topic",
        "go_no_go",
        "go_no_go_reason",
        "analysis_family",
        "database",
        "research_question",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("field must be non-empty")
        return text

    @model_validator(mode="after")
    def _validate_execution_gate(self) -> "DiscoveryHandoffPacket":
        if self.human_confirmed and (
            not self.human_confirmed_at or not self.human_confirmation_note
        ):
            raise ValueError(
                "human confirmation requires confirmed_at and a confirmation note"
            )
        concept_set = is_concept_set_family(self.analysis_family)
        if concept_set and not self.resolved_analysis_concepts:
            raise ValueError(
                "concept-set analysis family requires resolved_analysis_concepts"
            )
        if not concept_set and not str(self.target_outcome or "").strip():
            raise ValueError(
                "target_outcome is required for predictor/outcome analysis families"
            )
        if self.resolved_outcome_concept and not _same_endpoint(
            self.resolved_outcome_concept, self.target_outcome
        ):
            raise ValueError(
                "resolved_outcome_concept and target_outcome must identify the "
                "same endpoint"
            )
        self.verify_seal()
        return self

    def verify_seal(self) -> None:
        """Recheck also at use time to reject unvalidated model_copy updates."""
        payload = self.model_dump(mode="json", exclude={"handoff_sha256"})
        if canonical_sha256(payload["selected_ledger_row"]) != self.selected_candidate_sha256:
            raise ValueError("discovery_candidate_digest_mismatch")
        if canonical_sha256(payload) != self.handoff_sha256:
            raise ValueError("discovery_handoff_digest_mismatch")

    def verify_source(self) -> None:
        self.verify_seal()
        try:
            actual = sha256_file(self.source_triage_report_path)
        except OSError as exc:
            raise ValueError("discovery_source_evidence_unavailable") from exc
        if actual != self.source_triage_report_sha256:
            raise ValueError("discovery_source_evidence_changed")

    @property
    def analysis_ready(self) -> bool:
        try:
            self.verify_source()
        except ValueError:
            return False
        return (
            _normalise_decision(self.go_no_go) in ANALYSIS_READY_DECISIONS
            and self.human_confirmed
            and _handoff_shape_is_valid(self)
        )


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
    require_analysis_ready: bool = False,
    require_resolved_outcome: bool = False,
    require_executable_shape: bool = False,
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
        selected = dict(rows[index])
        if require_analysis_ready and not _row_recommended(selected):
            raise ValueError(
                "selected discovery row is not go/recommend and cannot enter analysis"
            )
        if require_resolved_outcome and not selected.get("resolved_outcome_concept"):
            raise ValueError("selected discovery row has no resolved outcome concept")
        if require_executable_shape and not _row_has_executable_shape(selected):
            raise ValueError("selected discovery row has no executable analysis shape")
        return selected
    if not rows:
        raise ValueError("cannot select from an empty discovery ledger")
    selectable = [
        row
        for row in rows
        if not require_resolved_outcome or row.get("resolved_outcome_concept")
    ]
    if not selectable:
        raise ValueError("discovery ledger has no row with a resolved outcome concept")
    if require_executable_shape:
        selectable = [row for row in selectable if _row_has_executable_shape(row)]
        if not selectable:
            raise ValueError("discovery ledger has no executable analysis shape")
    eligible = [row for row in selectable if _row_recommended(row)]
    if require_analysis_ready and not eligible:
        raise ValueError("discovery ledger has no go/recommend row for analysis")
    return dict(max(eligible or selectable, key=_row_priority))


def build_handoff_from_row(
    row: Mapping[str, Any],
    *,
    triage_report_path: str | Path,
    selection_mode: DiscoverySelectionMode = "agent_selected",
    selection_rationale: Optional[str] = None,
    target_outcome: Optional[str] = None,
    database: str = "miiv",
    research_question: Optional[str] = None,
    inclusion_criteria: Optional[Sequence[str]] = None,
    human_confirmed: bool = False,
    human_confirmation_note: Optional[str] = None,
) -> DiscoveryHandoffPacket:
    """Create a frozen handoff packet from one discovery-ledger row."""

    topic = str(row.get("candidate_topic") or "").strip()
    if not topic:
        raise ValueError("selected discovery row has no candidate_topic")
    analysis_family = normalize_analysis_family(row.get("analysis_family"))
    resolved_analysis_concepts = [
        str(value).strip()
        for value in row.get("resolved_analysis_concepts") or []
        if str(value).strip()
    ]
    concept_set = is_concept_set_family(analysis_family)
    question = research_question or _default_research_question(
        topic,
        analysis_family=analysis_family,
    )
    rationale = selection_rationale or _default_selection_rationale(row)
    resolved_outcome = str(row.get("resolved_outcome_concept") or "").strip()
    requested_outcome = str(target_outcome or "").strip()
    if (
        resolved_outcome
        and requested_outcome
        and not _same_endpoint(resolved_outcome, requested_outcome)
    ):
        raise ValueError(
            "target_outcome conflicts with the selected row's "
            f"resolved_outcome_concept ({requested_outcome!r} != "
            f"{resolved_outcome!r})"
        )
    effective_outcome = resolved_outcome or requested_outcome
    if not effective_outcome and not concept_set:
        raise ValueError(
            "target_outcome is required when the selected row has no "
            "resolved_outcome_concept"
        )
    if concept_set and not resolved_analysis_concepts:
        raise ValueError(
            "resolved_analysis_concepts is required for a concept-set analysis"
        )
    confirmed_at = datetime.now(timezone.utc).isoformat() if human_confirmed else None
    confirmation_note = None
    if human_confirmed:
        confirmation_note = str(
            human_confirmation_note
            or "Explicitly confirmed by the discovery launcher operator."
        ).strip()
    source = Path(triage_report_path).resolve()
    try:
        source_digest = sha256_file(source)
    except OSError as exc:
        raise ValueError("discovery_source_evidence_unavailable") from exc
    payload = dict(
        schema_version=DISCOVERY_HANDOFF_SCHEMA_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_triage_report_path=str(source),
        source_triage_report_sha256=source_digest,
        selected_candidate_sha256=canonical_sha256(thaw_payload(row)),
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
            str(row.get("literature_source")) if row.get("literature_source") else None
        ),
        gap_evidence_quote=(
            str(row.get("gap_evidence_quote"))
            if row.get("gap_evidence_quote")
            else None
        ),
        go_no_go=str(row.get("go_no_go") or ""),
        go_no_go_reason=str(row.get("go_no_go_reason") or ""),
        human_confirmed=bool(human_confirmed),
        human_confirmed_at=confirmed_at,
        human_confirmation_note=confirmation_note,
        novelty_label=(
            str(row.get("novelty_label")) if row.get("novelty_label") else None
        ),
        feasibility_route=(
            str(row.get("feasibility_route")) if row.get("feasibility_route") else None
        ),
        feasibility_next_action=(
            str(row.get("feasibility_next_action"))
            if row.get("feasibility_next_action")
            else None
        ),
        analysis_family=analysis_family,
        resolved_analysis_concepts=list(dict.fromkeys(resolved_analysis_concepts)),
        resolved_predictor_concept=(
            str(row.get("resolved_predictor_concept"))
            if row.get("resolved_predictor_concept")
            else None
        ),
        resolved_outcome_concept=resolved_outcome or None,
        target_outcome=effective_outcome or None,
        database=database,
        research_question=question,
        inclusion_criteria=list(inclusion_criteria or _default_inclusion_criteria()),
        selected_ledger_row=thaw_payload(row),
        required_manuscript_figure_roles=list(DEFAULT_DISCOVERY_FIGURE_ROLES),
    )
    for key in (
        "source_triage_report_path", "selection_rationale", "literature_idea_id",
        "candidate_topic", "go_no_go", "go_no_go_reason", "analysis_family",
        "database", "research_question",
    ):
        payload[key] = str(payload[key]).strip()
    return DiscoveryHandoffPacket(**payload, handoff_sha256=canonical_sha256(payload))


def write_handoff_packet(packet: DiscoveryHandoffPacket, path: str | Path) -> Path:
    """Write a handoff packet as JSON and return the output path."""

    out = Path(path)
    packet.verify_source()
    publish_write_once_bytes(
        out, packet.model_dump_json(indent=2).encode("utf-8"),
        temp_prefix=".discovery-handoff-", conflict_error=ValueError,
        conflict_message="discovery_handoff_immutable_conflict: create a new version",
    )
    return out


def _row_priority(row: Mapping[str, Any]) -> tuple:
    go = str(row.get("go_no_go") or "").lower()
    novelty = str(row.get("novelty_label") or "").lower()
    direct_same_topic = len(row.get("direct_same_topic_pmids") or [])
    differentiators = len(row.get("differentiators") or [])
    risks = len(row.get("risks") or [])
    go_score = {"go": 3, "recommend": 3, "hold": 2, "db-cannot-do": 0}.get(go, 1)
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


def _normalise_decision(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _row_recommended(row: Mapping[str, Any]) -> bool:
    return _normalise_decision(row.get("go_no_go")) in ANALYSIS_READY_DECISIONS


def _row_has_executable_shape(row: Mapping[str, Any]) -> bool:
    family = normalize_analysis_family(row.get("analysis_family"))
    if is_concept_set_family(family):
        return any(
            str(value).strip() for value in row.get("resolved_analysis_concepts") or []
        )
    return bool(str(row.get("resolved_outcome_concept") or "").strip())


def _normalise_endpoint(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _same_endpoint(left: Any, right: Any) -> bool:
    return bool(_normalise_endpoint(left)) and _normalise_endpoint(
        left
    ) == _normalise_endpoint(right)


def assert_discovery_analysis_ready(packet: DiscoveryHandoffPacket) -> bool:
    """Validate proposal intake; reviewed plan and execution authority remain required."""

    decision = _normalise_decision(packet.go_no_go)
    if decision not in ANALYSIS_READY_DECISIONS:
        raise ValueError(f"discovery decision {packet.go_no_go!r} is not go/recommend")
    if not packet.human_confirmed:
        raise ValueError("explicit human confirmation is required before analysis")
    packet.verify_source()
    if not _handoff_shape_is_valid(packet):
        raise ValueError("discovery handoff has no valid executable analysis shape")
    return True


def _handoff_shape_is_valid(packet: DiscoveryHandoffPacket) -> bool:
    if is_concept_set_family(packet.analysis_family):
        return bool(packet.resolved_analysis_concepts)
    return bool(str(packet.target_outcome or "").strip()) and (
        not packet.resolved_outcome_concept
        or _same_endpoint(packet.resolved_outcome_concept, packet.target_outcome)
    )


def _default_research_question(topic: str, *, analysis_family: str) -> str:
    if is_concept_set_family(analysis_family):
        return (
            "Starting from the agent-mined ICU literature idea, evaluate the "
            f"following concept-set candidate in an adult ICU cohort: {topic}. "
            "The analysis must prespecify time zero, observation window, "
            "trajectory representation, stability criteria, and validation "
            "across databases before fitting. Outcomes must not be used to "
            "select trajectory classes."
        )
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
    "ANALYSIS_READY_DECISIONS",
    "assert_discovery_analysis_ready",
    "build_handoff_from_row",
    "load_discovery_ledger",
    "select_discovery_row",
    "write_handoff_packet",
]
