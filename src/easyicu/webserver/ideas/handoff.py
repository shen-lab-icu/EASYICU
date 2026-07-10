"""Canonical discovery-handoff adapter for the native Web Idea Mining flow.

The Web UI keeps its metadata-rich response envelope for compatibility, while
this module owns the narrow conversion into the research-agent's canonical
``DiscoveryHandoffPacket``.  The packet is persisted separately and verified
again before an Agent Project seed can be created.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pydantic import ValidationError

from easyicu.research_agent.discovery_handoff import (
    DiscoveryHandoffPacket,
    DiscoverySelectionMode,
    build_handoff_from_row,
    write_handoff_packet,
)

CANONICAL_HANDOFF_FILENAME = "discovery_handoff.json"
CANONICAL_ENVELOPE_FIELDS = frozenset(
    {
        "canonical_handoff",
        "canonical_handoff_path",
        "canonical_handoff_sha256",
    }
)


class CanonicalHandoffIntegrityError(ValueError):
    """Raised when the frozen canonical handoff cannot be trusted."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


def map_web_ledger_row(
    *,
    idea: Mapping[str, Any],
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    pre_experiment: Mapping[str, Any],
    prior_art_check: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Map the metadata-only Web ledger row to canonical discovery semantics."""

    mapped_concepts = [
        dict(row)
        for row in idea.get("mapped_concepts") or []
        if isinstance(row, Mapping)
    ]
    outcome_concept = _role_concept(mapped_concepts, "outcome")
    predictor_concept = _predictor_concept(mapped_concepts)
    prior_art = _prior_art_payload(idea, prior_art_check)
    feasibility = idea.get("feasibility") or {}
    source_title = str(source.get("title") or "").strip()
    source_quote = str(source.get("evidence_quote") or "").strip()

    row = dict(idea)
    row.update(
        {
            "literature_idea_id": str(idea.get("idea_id") or "").strip(),
            "candidate_topic": str(idea.get("idea_title") or "").strip(),
            "literature_source": source_title or None,
            "gap_evidence_quote": source_quote or None,
            "novelty_label": str(
                prior_art.get("novelty_label")
                or prior_art.get("status")
                or "unknown_until_search"
            ),
            "feasibility_route": str(
                feasibility.get("tier")
                or pre_experiment.get("status")
                or "blocked"
            ),
            "feasibility_next_action": str(
                idea.get("next_action")
                or pre_experiment.get("reason")
                or "Review feasibility before analysis."
            ),
            "resolved_predictor_concept": predictor_concept,
            "resolved_outcome_concept": outcome_concept,
            "web_handoff_context": {
                "plan_status": plan.get("plan_status"),
                "analysis_family": plan.get("analysis_family")
                or idea.get("analysis_family"),
                "pre_experiment_status": pre_experiment.get("status"),
                "source_id": source.get("source_id"),
                "source_type": source.get("source_type"),
                "source_text_sha256": source.get("source_text_sha256"),
                "prior_art_status": prior_art.get("status"),
            },
        }
    )
    return row


def build_web_handoff_packet(
    *,
    idea: Mapping[str, Any],
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    pre_experiment: Mapping[str, Any],
    prior_art_check: Optional[Mapping[str, Any]],
    run_dir: str | Path,
) -> DiscoveryHandoffPacket:
    """Build an unconfirmed canonical packet from the current Web artifacts."""

    run_path = Path(run_dir)
    row = map_web_ledger_row(
        idea=idea,
        source=source,
        plan=plan,
        pre_experiment=pre_experiment,
        prior_art_check=prior_art_check,
    )
    target_outcome = str(
        row.get("resolved_outcome_concept") or _target_outcome(idea)
    ).strip()
    database = str(
        (pre_experiment.get("source") or {}).get("database")
        or (plan.get("active_export_contract") or {}).get("database")
        or "unspecified"
    ).strip()
    rationale = str(idea.get("rationale") or "").strip() or None
    question = str(plan.get("research_question") or "").strip() or None

    return build_handoff_from_row(
        row,
        triage_report_path=run_path / "idea_mining_run.json",
        selection_mode=_selection_mode(plan),
        selection_rationale=rationale,
        target_outcome=target_outcome,
        database=database,
        research_question=question,
        inclusion_criteria=_inclusion_criteria(plan),
        # Creating or editing a Web handoff is not analysis confirmation.
        human_confirmed=False,
    )


def persist_canonical_handoff(
    packet: DiscoveryHandoffPacket,
    *,
    run_dir: str | Path,
) -> Dict[str, Any]:
    """Persist the fixed canonical artifact and return envelope reference fields."""

    run_path = Path(run_dir)
    path = write_handoff_packet(packet, run_path / CANONICAL_HANDOFF_FILENAME)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "canonical_handoff": packet.model_dump(mode="json"),
        "canonical_handoff_path": CANONICAL_HANDOFF_FILENAME,
        "canonical_handoff_sha256": digest,
    }


def is_legacy_handoff_envelope(envelope: Mapping[str, Any]) -> bool:
    """Return True only for envelopes predating every canonical field."""

    return not any(field in envelope for field in CANONICAL_ENVELOPE_FIELDS)


def load_validated_canonical_handoff(
    envelope: Mapping[str, Any],
    *,
    run_dir: str | Path,
) -> DiscoveryHandoffPacket:
    """Re-read, hash-check, and validate the canonical packet fail-closed."""

    relative_path = str(envelope.get("canonical_handoff_path") or "").strip()
    if relative_path != CANONICAL_HANDOFF_FILENAME:
        raise CanonicalHandoffIntegrityError(
            "canonical handoff path must be discovery_handoff.json"
        )
    expected_hash = str(envelope.get("canonical_handoff_sha256") or "").strip()
    if len(expected_hash) != 64:
        raise CanonicalHandoffIntegrityError(
            "canonical handoff envelope is missing a valid sha256"
        )

    path = Path(run_dir) / CANONICAL_HANDOFF_FILENAME
    if not path.is_file():
        raise CanonicalHandoffIntegrityError("canonical handoff artifact is missing")
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_hash, expected_hash):
        raise CanonicalHandoffIntegrityError("canonical handoff sha256 mismatch")

    try:
        file_payload = json.loads(raw.decode("utf-8"))
        file_packet = DiscoveryHandoffPacket.model_validate(file_payload)
        envelope_packet = DiscoveryHandoffPacket.model_validate(
            envelope.get("canonical_handoff")
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise CanonicalHandoffIntegrityError(
            f"canonical handoff validation failed: {exc}"
        ) from exc
    if file_packet.model_dump(mode="json") != envelope_packet.model_dump(mode="json"):
        raise CanonicalHandoffIntegrityError(
            "canonical handoff envelope does not match the frozen artifact"
        )
    _validate_legacy_identity(envelope, file_packet)
    return file_packet


def _validate_legacy_identity(
    envelope: Mapping[str, Any],
    packet: DiscoveryHandoffPacket,
) -> None:
    envelope_idea_id = str(envelope.get("idea_id") or "").strip()
    selected_idea_id = str(
        packet.selected_ledger_row.get("idea_id") or ""
    ).strip()
    if not envelope_idea_id or not selected_idea_id or not (
        envelope_idea_id == packet.literature_idea_id == selected_idea_id
    ):
        raise CanonicalHandoffIntegrityError(
            "legacy envelope idea_id does not match the canonical handoff"
        )
    envelope_topic = str(envelope.get("candidate_topic") or "").strip()
    if not envelope_topic or envelope_topic != packet.candidate_topic:
        raise CanonicalHandoffIntegrityError(
            "legacy envelope candidate_topic does not match the canonical handoff"
        )


def _selection_mode(plan: Mapping[str, Any]) -> DiscoverySelectionMode:
    mode = str(plan.get("selection_mode") or "").lower()
    if "human" in mode:
        return "human_curated"
    return "agent_selected"


def _inclusion_criteria(plan: Mapping[str, Any]) -> list[str]:
    cohort = plan.get("cohort") or {}
    default = str(cohort.get("default") or "").strip()
    if default:
        return [default]
    return ["Adult ICU cohort from the active prepared EasyICU export."]


def _prior_art_payload(
    idea: Mapping[str, Any],
    prior_art_check: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    checked = (prior_art_check or {}).get("prior_art") or {}
    if isinstance(checked, Mapping) and checked:
        return checked
    fallback = idea.get("prior_art") or {}
    return fallback if isinstance(fallback, Mapping) else {}


def _role_concept(rows: list[Dict[str, Any]], role: str) -> Optional[str]:
    for row in rows:
        if str(row.get("role") or "") == role and row.get("concept_id"):
            return str(row["concept_id"])
    return None


def _predictor_concept(rows: list[Dict[str, Any]]) -> Optional[str]:
    for role in ("exposure", "predictor", "covariate_or_subgroup", "feature"):
        concept = _role_concept(rows, role)
        if concept:
            return concept
    return None


def _target_outcome(idea: Mapping[str, Any]) -> str:
    label = str(idea.get("outcome") or "").strip().lower()
    if "length" in label or "stay" in label or "los" in label:
        return "los_icu"
    return "death"


__all__ = [
    "CANONICAL_HANDOFF_FILENAME",
    "CANONICAL_ENVELOPE_FIELDS",
    "CanonicalHandoffIntegrityError",
    "build_web_handoff_packet",
    "is_legacy_handoff_envelope",
    "load_validated_canonical_handoff",
    "map_web_ledger_row",
    "persist_canonical_handoff",
]
