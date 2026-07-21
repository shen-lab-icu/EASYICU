"""Prepare the bounded, deterministic know-how context used before planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..know_how import KnowHowHit, KnowHowIntegrityError, KnowHowRegistry
from ..schema import ResearchContext
from .analysis_types import infer_analysis_type

_RETRIEVAL_FAMILY = {
    "descriptive_epidemiology": "descriptive",
    "association_study": "association",
    "prediction_model": "prediction",
    "dynamic_prediction": "prediction",
    "validation": "prediction",
    "cross_database_replication": "association",
    "survival": "time_to_event",
    "trajectory_clustering": "phenotyping",
    "treatment_response": "causal_emulation",
    "causal_inference": "causal_emulation",
}


@dataclass(frozen=True)
class PreplanKnowHow:
    """Persisted know-how selection and the exact planner projection."""

    registry: KnowHowRegistry
    hits: tuple[KnowHowHit, ...]
    prompt: str

    @property
    def selected_ids(self) -> tuple[str, ...]:
        return tuple(hit.card_id for hit in self.hits)


def _write_or_verify(
    path: Path, content: bytes, *, evidence: Any, evidence_id: str
) -> None:
    """Make resume idempotent and fail closed when an artifact was changed."""
    existing = evidence.get(evidence_id)
    if existing is not None:
        if not path.exists() or path.read_bytes() != content:
            raise KnowHowIntegrityError(
                f"persisted {evidence_id} does not match deterministic reconstruction"
            )
        return
    path.write_bytes(content)


def prepare_preplan_know_how(
    *,
    context: ResearchContext,
    run_dir: Path,
    evidence: Any,
    database: str,
    paths: Sequence[str | Path] = (),
    top_k: int = 3,
    min_score: float = 0.15,
) -> PreplanKnowHow:
    """Retrieve cards, persist exact inputs, and register both evidence files."""
    registry = KnowHowRegistry.load(paths)
    inferred = infer_analysis_type(context)
    retrieval_family = _RETRIEVAL_FAMILY.get(inferred.key, inferred.key)
    available_concepts = sorted(
        {
            concept
            for variable in context.variables
            for concept in (
                variable.name,
                variable.source_concept,
                *variable.derived_from_concepts,
            )
            if concept
        }
    )
    hits = tuple(
        registry.retrieve(
            query=context.research_question,
            study_family=retrieval_family,
            database=database,
            available_concepts=available_concepts,
            top_k=top_k,
            min_score=min_score,
        )
    )
    prompt = registry.render_prompt(hits)
    receipt = registry.retrieval_receipt(
        query=context.research_question,
        study_family=retrieval_family,
        database=database,
        available_concepts=available_concepts,
        hits=hits,
        top_k=top_k,
        min_score=min_score,
    )
    receipt_bytes = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    prompt_bytes = prompt.encode("utf-8")
    receipt_path = run_dir / "know_how_retrieval.json"
    prompt_path = run_dir / "know_how_prompt.md"

    _write_or_verify(
        receipt_path,
        receipt_bytes,
        evidence=evidence,
        evidence_id="know_how_retrieval",
    )
    _write_or_verify(
        prompt_path,
        prompt_bytes,
        evidence=evidence,
        evidence_id="know_how_prompt",
    )
    if evidence.get("know_how_retrieval") is None:
        evidence.register_file(
            kind="log",
            description="Deterministic research know-how retrieval receipt.",
            source_path=receipt_path,
            evidence_id="know_how_retrieval",
            producer="research_know_how",
            generation_mode="deterministic_skill",
        )
    if evidence.get("know_how_prompt") is None:
        evidence.register_file(
            kind="log",
            description="Exact bounded research know-how projection supplied to Planner.",
            source_path=prompt_path,
            evidence_id="know_how_prompt",
            producer="research_know_how",
            generation_mode="deterministic_skill",
            inputs=["know_how_retrieval"],
        )
    return PreplanKnowHow(registry=registry, hits=hits, prompt=prompt)


__all__ = ["PreplanKnowHow", "prepare_preplan_know_how"]
