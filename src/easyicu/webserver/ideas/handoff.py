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
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pydantic import ValidationError

from easyicu.research_agent.discovery.discovery_handoff import (
    DiscoveryHandoffPacket,
    DiscoverySelectionMode,
    build_handoff_from_row,
    write_handoff_packet,
)
from easyicu.webserver import study_contexts

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


class CanonicalHandoffAcceptanceError(ValueError):
    """Fail-closed reason from the Idea handoff-to-StudyContext transaction."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        owner: str = "easyicu.webserver.ideas.handoff",
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.code = str(code)
        self.message = str(message)
        self.owner = str(owner)
        self.details = dict(details or {})
        super().__init__(self.message)


def accept_canonical_handoff(
    *,
    current: Mapping[str, Any],
    body: Mapping[str, Any],
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
    prior_art_binding: Optional[Mapping[str, Any]],
    readiness_binding: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and atomically project one canonical idea into StudyContext."""

    digest = str(handoff.get("canonical_handoff_sha256") or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CanonicalHandoffAcceptanceError(
            "canonical_idea_handoff_digest_required",
            "The selected Idea Mining handoff has no valid canonical digest.",
        )
    readiness = (
        readiness_binding if isinstance(readiness_binding, Mapping) else {}
    )
    if not readiness.get("execution_ready_for_confirmation"):
        raise CanonicalHandoffAcceptanceError(
            "canonical_idea_execution_readiness_required",
            (
                "The Idea Mining handoff requires current differentiated prior art "
                "and ready source-bound feasibility before acceptance."
            ),
        )
    canonical = (
        handoff.get("canonical_handoff")
        if isinstance(handoff.get("canonical_handoff"), Mapping)
        else {}
    )
    selected_row = (
        canonical.get("selected_ledger_row")
        if isinstance(canonical.get("selected_ledger_row"), Mapping)
        else {}
    )
    mapped_concepts = [
        row
        for row in selected_row.get("mapped_concepts") or []
        if isinstance(row, Mapping)
    ]
    module_by_concept = {
        str(row.get("concept_id") or "").strip(): str(row.get("module") or "").strip()
        for row in mapped_concepts
        if str(row.get("concept_id") or "").strip()
        and str(row.get("module") or "").strip()
    }
    predictor_concept = str(canonical.get("resolved_predictor_concept") or "").strip()
    outcome_concept = str(
        canonical.get("resolved_outcome_concept")
        or canonical.get("target_outcome")
        or ""
    ).strip()
    analysis_concepts = list(
        dict.fromkeys(
            str(value).strip()
            for value in canonical.get("resolved_analysis_concepts") or []
            if str(value).strip()
        )
    )
    execution_concepts = list(
        dict.fromkeys(
            value
            for value in (predictor_concept, outcome_concept, *analysis_concepts)
            if value
        )
    )
    missing_modules = [
        concept_id
        for concept_id in execution_concepts
        if concept_id not in module_by_concept
    ]
    if not canonical or not execution_concepts or missing_modules:
        raise CanonicalHandoffAcceptanceError(
            "canonical_idea_execution_contract_required",
            (
                "The canonical Idea Mining handoff does not contain a complete "
                "digest-bound concept-to-module execution contract."
            ),
            details={"missing_concept_modules": missing_modules},
        )

    plan_body = plan.get("plan") if isinstance(plan.get("plan"), Mapping) else {}
    agent_seed = (
        handoff.get("agent_seed")
        if isinstance(handoff.get("agent_seed"), Mapping)
        else {}
    )
    patch: Dict[str, Any] = {
        "id": current["id"],
        "idea_handoff": {
            "schema_version": "easyicu.pi-idea-selection/1",
            "run_id": str(body.get("run_id") or "").strip(),
            "idea_id": str(
                handoff.get("idea_id") or body.get("idea_id") or ""
            ).strip(),
            "canonical_handoff_sha256": digest,
            "status": "accepted",
            "accepted_at": str(handoff.get("created_at") or ""),
            "go_no_go": str(handoff.get("go_no_go") or ""),
            "go_no_go_reason": str(handoff.get("go_no_go_reason") or "")[:500],
            **dict(prior_art_binding or {}),
            "prior_art_adjudication_schema_version": str(
                readiness.get("prior_art_adjudication_schema_version") or ""
            ),
            "prior_art_adjudication_sha256": str(
                readiness.get("prior_art_adjudication_sha256") or ""
            ),
            "prior_art_decision": str(readiness.get("prior_art_decision") or ""),
            "source_feasibility_schema_version": str(
                readiness.get("source_feasibility_schema_version") or ""
            ),
            "source_feasibility_sha256": str(
                readiness.get("source_feasibility_sha256") or ""
            ),
            "source_feasibility_status": str(
                readiness.get("source_feasibility_status") or ""
            ),
            "idea_definition_sha256": str(
                readiness.get("idea_definition_sha256") or ""
            ),
            "source_path_hash": str(readiness.get("source_path_hash") or ""),
        },
        "current_stage": "study_setup",
        "last_route": "guided",
    }
    derived_fields = {
        "title": handoff.get("candidate_topic"),
        "question": plan_body.get("research_question") or agent_seed.get("question"),
        "outcome": outcome_concept or plan_body.get("outcome"),
        "primary_exposure": predictor_concept or plan_body.get("exposure"),
        "comparator": plan_body.get("comparator"),
        "analysis_goal": canonical.get("analysis_family")
        or plan_body.get("analysis_family"),
    }
    limits = {
        "title": 160,
        "question": 1200,
        "outcome": 500,
        "primary_exposure": 160,
        "comparator": 500,
        "analysis_goal": 1200,
    }
    patch.update(
        {
            key: str(value).strip()[: limits[key]]
            for key, value in derived_fields.items()
            if str(value or "").strip()
        }
    )
    previous_exposure = str(current.get("primary_exposure") or "").strip()
    if (
        predictor_concept != previous_exposure
        and not str(plan_body.get("comparator") or "").strip()
    ):
        patch["comparator"] = ""
    patch["modules"] = list(
        dict.fromkeys(
            module_by_concept[concept_id] for concept_id in execution_concepts
        )
    )
    requested_adjustment_concepts = list(
        dict.fromkeys(
            str(value).strip()
            for value in selected_row.get("requested_adjustment_concepts") or []
            if str(value).strip()
        )
    )
    invalid_adjustment_concepts = [
        concept_id
        for concept_id in requested_adjustment_concepts
        if concept_id not in analysis_concepts
        or concept_id in {predictor_concept, outcome_concept}
    ]
    if invalid_adjustment_concepts:
        raise CanonicalHandoffAcceptanceError(
            "canonical_idea_adjustment_contract_invalid",
            (
                "The canonical Idea Mining handoff contains adjustment variables "
                "outside its digest-bound analysis concept set."
            ),
            details={"invalid_adjustment_concepts": invalid_adjustment_concepts},
        )
    patch["covariates"] = requested_adjustment_concepts
    patch["execution_concepts"] = {
        **({"outcome": outcome_concept} if outcome_concept else {}),
        **({"primary_exposure": predictor_concept} if predictor_concept else {}),
        "covariates": requested_adjustment_concepts,
    }
    try:
        return study_contexts.upsert_context(
            patch,
            active=True,
            expected_revision=int(current.get("revision") or 0),
            require_revision=True,
            lifecycle_write=False,
        )
    except study_contexts.StudyContextError as exc:
        raise CanonicalHandoffAcceptanceError(
            str(exc.detail.get("error") or "idea_handoff_binding_blocked"),
            "The StudyContext owner rejected the selected Idea Mining handoff.",
            owner="easyicu.webserver.study_contexts",
        ) from exc


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
    resolved_analysis_concepts = [
        str(value).strip()
        for value in idea.get("resolved_analysis_concepts") or []
        if str(value).strip()
    ]
    if not resolved_analysis_concepts:
        resolved_analysis_concepts = list(
            dict.fromkeys(
                str(item.get("concept_id") or "").strip()
                for item in mapped_concepts
                if str(item.get("role") or "").strip().lower() != "outcome"
                and str(item.get("concept_id") or "").strip()
            )
        )
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
                feasibility.get("tier") or pre_experiment.get("status") or "blocked"
            ),
            "feasibility_next_action": str(
                idea.get("next_action")
                or pre_experiment.get("reason")
                or "Review feasibility before analysis."
            ),
            "resolved_predictor_concept": predictor_concept,
            "resolved_outcome_concept": outcome_concept,
            "resolved_analysis_concepts": resolved_analysis_concepts,
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
    relative_path = f"handoffs/{packet.handoff_sha256}.json"
    path = write_handoff_packet(packet, run_path / relative_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "canonical_handoff": packet.model_dump(mode="json"),
        "canonical_handoff_path": relative_path,
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
    if not re.fullmatch(r"handoffs/[0-9a-f]{64}\.json", relative_path):
        raise CanonicalHandoffIntegrityError(
            "canonical handoff path must identify a sealed handoff version"
        )
    expected_hash = str(envelope.get("canonical_handoff_sha256") or "").strip()
    if len(expected_hash) != 64:
        raise CanonicalHandoffIntegrityError(
            "canonical handoff envelope is missing a valid sha256"
        )

    path = Path(run_dir) / relative_path
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
    if relative_path != f"handoffs/{file_packet.handoff_sha256}.json":
        raise CanonicalHandoffIntegrityError("canonical handoff version mismatch")
    try:
        file_packet.verify_source()
    except ValueError as exc:
        raise CanonicalHandoffIntegrityError(str(exc)) from exc
    _validate_legacy_identity(envelope, file_packet)
    return file_packet


def _validate_legacy_identity(
    envelope: Mapping[str, Any],
    packet: DiscoveryHandoffPacket,
) -> None:
    envelope_idea_id = str(envelope.get("idea_id") or "").strip()
    selected_idea_id = str(packet.selected_ledger_row.get("idea_id") or "").strip()
    if (
        not envelope_idea_id
        or not selected_idea_id
        or not (envelope_idea_id == packet.literature_idea_id == selected_idea_id)
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
    """State the idea's own cohort wording, and claim nothing when it has none.

    The fallback used to assert "Adult ICU cohort", which is an eligibility
    criterion no one chose: the handoff carries it to the Planner as a bound
    inclusion contract, so an idea that never mentioned age arrived claiming an
    adult restriction the export may not apply. Eligibility is the
    researcher's decision -- Copilot asks for it in study setup -- so an idea
    with no cohort wording declares no criterion here.
    """

    cohort = plan.get("cohort") or {}
    default = str(cohort.get("default") or "").strip()
    return [default] if default else []


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
    "CanonicalHandoffAcceptanceError",
    "CanonicalHandoffIntegrityError",
    "accept_canonical_handoff",
    "build_web_handoff_packet",
    "is_legacy_handoff_envelope",
    "load_validated_canonical_handoff",
    "map_web_ledger_row",
    "persist_canonical_handoff",
]
