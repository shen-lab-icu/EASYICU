"""Candidate membership and deterministic Web proposal transformations.

This proves provenance, not execution readiness or scientific approval. The Web
readiness owner still validates clinical feasibility and independent review.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from easyicu.concept import catalog as concept_catalog
from ..canonical_json import canonical_sha256
from ..contracts.frozen_payload import freeze_payload, thaw_payload


class DiscoveryCandidateSource(BaseModel):
    """Identity of a member of a recognised source, plus a replayable mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_kind: Literal["discovery_ledger", "discovery_records", "web_idea_ledger"]
    candidate_id: str = Field(min_length=1)
    executable_candidate_id: str | None = None
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transformation: Mapping[str, Any] | None = None

    @field_validator("transformation")
    @classmethod
    def _freeze(cls, value):
        return freeze_payload(value) if value is not None else None

    @field_serializer("transformation")
    def _project(self, value):
        return thaw_payload(value) if value is not None else None


def _bound_document(path: Path, expected_sha: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError("discovery_transform_evidence_unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha or not isinstance(
        value, Mapping
    ):
        raise ValueError("discovery_transform_evidence_changed")
    return value


def bind_source_candidate(
    *,
    payload: Mapping[str, Any],
    source_path: Path,
    row: Mapping[str, Any],
    transformation: Mapping[str, Any] | None = None,
) -> DiscoveryCandidateSource:
    """Prove exact membership first, then replay the declared host mapping."""
    if not isinstance(payload, Mapping):
        raise ValueError("discovery_source_type_invalid")
    if payload.get("schema_version") not in {
        None,
        "easyicu.idea_mining_dry_run/1",
        "easyicu.longitudinal_candidate_triage/1",
        "easyicu.web_idea_mining/1",
    }:
        raise ValueError("discovery_source_type_invalid")
    if payload.get("schema_version") == "easyicu.web_idea_mining/1":
        kind, key, identity = "web_idea_ledger", "idea_ledger", "idea_id"
    elif isinstance(payload.get("discovery_ledger"), list):
        kind, key, identity = (
            "discovery_ledger",
            "discovery_ledger",
            "literature_idea_id",
        )
    elif isinstance(payload.get("discovery_records"), list):
        kind, key, identity = (
            "discovery_records",
            "discovery_records",
            "literature_idea_id",
        )
    else:
        raise ValueError("discovery_source_type_invalid")
    rows = payload.get(key)
    if not isinstance(rows, list) or any(
        not isinstance(item, Mapping) for item in rows
    ):
        raise ValueError("discovery_source_type_invalid")
    candidate_id = str(row.get(identity) or "").strip()
    members = [
        item
        for item in rows
        if str(item.get(identity) or "").strip() == candidate_id
        and item.get("executable_candidate_id") == row.get("executable_candidate_id")
    ]
    if not candidate_id or len(members) != 1:
        raise ValueError("discovery_source_candidate_identity_invalid")
    original = members[0]
    if kind == "web_idea_ledger":
        transform = thaw_payload(transformation) if transformation is not None else {}
        if transform.get("schema_version") != "easyicu.web_candidate_mapping/1" or set(
            transform
        ) != {
            "schema_version",
            "readiness",
            "source",
            "plan",
            "pre_experiment",
            "prior_art_check",
            "prior_art_sha256",
        }:
            raise ValueError("discovery_candidate_transformation_required")
        if any(
            not isinstance(transform[field], Mapping)
            for field in ("readiness", "source", "plan", "pre_experiment")
        ) or (
            transform["prior_art_check"] is not None
            and not isinstance(transform["prior_art_check"], Mapping)
        ):
            raise ValueError("discovery_candidate_transformation_invalid")
        sources = payload.get("source_evidence") or [{}]
        if not isinstance(sources, list) or any(
            not isinstance(item, Mapping) for item in sources
        ):
            raise ValueError("discovery_source_type_invalid")
        if transform["source"] != (
            (payload.get("source_evidence") or [{}])[0] or {}
        ) or transform["pre_experiment"] != (payload.get("pre_experiment") or {}):
            raise ValueError("discovery_transform_source_mismatch")
        readiness = transform["readiness"] or {}
        if readiness.get("execution_ready_for_confirmation"):
            feasibility = _bound_document(
                source_path.parent / "bounded_sample_feasibility.json",
                readiness.get("source_feasibility_sha256", ""),
            )
            modules = {
                str(item.get("concept_id")): str(item.get("module"))
                for item in feasibility.get("feature_statistics", [])
                if isinstance(item, Mapping)
                and item.get("concept_id")
                and item.get("module")
            }
            if (
                feasibility.get("schema_version")
                != "easyicu.web_idea_bounded_sample_feasibility/2"
                or feasibility.get("idea_id") != candidate_id
                or feasibility.get("run_id") != payload.get("run_id")
                or feasibility.get("concept_bindings")
                != readiness.get("concept_bindings")
                or modules != readiness.get("concept_modules")
                or feasibility.get("status") != "ready"
            ):
                raise ValueError("discovery_transform_readiness_mismatch")
        prior = transform["prior_art_check"]
        if prior:
            if (
                _bound_document(
                    source_path.parent / "prior_art_check.json",
                    transform["prior_art_sha256"],
                )
                != prior
            ):
                raise ValueError("discovery_transform_prior_art_mismatch")
        mapped = map_web_ledger_row(
            idea=idea_with_readiness_overlay(original, readiness),
            source=transform["source"],
            plan=transform["plan"],
            pre_experiment=transform["pre_experiment"],
            prior_art_check=prior,
        )
    else:
        if transformation is not None:
            raise ValueError("discovery_candidate_transformation_not_supported")
        mapped = original
    if canonical_sha256(mapped) != canonical_sha256(thaw_payload(row)):
        raise ValueError("discovery_source_candidate_mismatch")
    return DiscoveryCandidateSource(
        source_kind=kind,
        candidate_id=candidate_id,
        executable_candidate_id=original.get("executable_candidate_id"),
        candidate_sha256=canonical_sha256(original),
        transformation=transformation,
    )


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


def idea_with_readiness_overlay(
    idea: Mapping[str, Any], readiness: Mapping[str, Any]
) -> Dict[str, Any]:
    selected = dict(idea)
    if not readiness.get("execution_ready_for_confirmation"):
        return selected
    bindings = readiness.get("concept_bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    modules = readiness.get("concept_modules")
    modules = modules if isinstance(modules, Mapping) else {}
    role_by_concept = {
        str(bindings.get("primary_exposure") or ""): "predictor",
        str(bindings.get("outcome") or ""): "outcome",
        str(bindings.get("time_zero") or ""): "time_zero",
        **{
            str(value): "adjustment"
            for value in bindings.get("covariates") or []
            if str(value).strip()
        },
    }
    mapped = [
        {
            "concept_id": concept_id,
            "label": str(
                concept_catalog.CONCEPT_DICTIONARY.get(concept_id, (concept_id,))[0]
            ),
            "module": str(modules.get(concept_id) or ""),
            "role": role,
            "status": "source_bound_feasibility_ready",
            "available": True,
        }
        for concept_id, role in role_by_concept.items()
        if concept_id and modules.get(concept_id)
    ]
    selected.update(
        {
            "mapped_concepts": mapped,
            "requested_adjustment_concepts": list(bindings.get("covariates") or []),
            "resolved_analysis_concepts": [
                row["concept_id"] for row in mapped if row["role"] != "outcome"
            ],
            "feasibility": {
                "tier": "executable",
                "label": "Source-bound feasibility ready",
                "reason": "Exact concepts passed bounded source feasibility.",
            },
            "go_no_go": "recommend",
            "go_no_go_reason": (
                "Differentiated prior art and source-bound feasibility are current; "
                "the plan is ready for researcher confirmation."
            ),
            "next_action": "Review and accept the execution-ready Idea Plan.",
        }
    )
    return selected
