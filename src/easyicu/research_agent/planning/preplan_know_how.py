"""Prepare the bounded, deterministic know-how context used before planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..know_how import KnowHowHit, KnowHowIntegrityError, KnowHowRegistry
from ..resources import ResourceScheduler, ResourceSelectionQuery
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

    @property
    def decision_authority(self) -> Mapping[str, Mapping[str, Any]]:
        """Exact card/claim coordinates the Planner may cite for this run."""
        authority: dict[str, Mapping[str, Any]] = {}
        for hit in self.hits:
            card = self.registry.get(hit.card_id)
            authority[hit.card_id] = {
                "version": hit.version,
                "file_sha256": hit.file_sha256,
                "claims": {
                    claim.claim_id: tuple(claim.citation_ids) for claim in card.claims
                },
            }
        return authority


@dataclass(frozen=True)
class PlannerKnowHowBinding:
    """All Planner-facing runtime coordinates for one retrieved card set."""

    prompt: str = ""
    selected_ids: tuple[str, ...] = ()
    decision_authority: Mapping[str, Mapping[str, Any]] | None = None
    enabled: bool = False

    @classmethod
    def from_prepared(cls, prepared: PreplanKnowHow) -> "PlannerKnowHowBinding":
        return cls(
            prompt=(prepared.prompt if prepared.hits else ""),
            selected_ids=prepared.selected_ids,
            decision_authority=prepared.decision_authority,
            enabled=True,
        )

    @property
    def planner_kwargs(self) -> dict[str, Any]:
        return {
            "allowed_know_how_decisions": self.decision_authority,
            "know_how_context": self.prompt,
        }

    def verify_resume(self, decisions: Sequence[Any], *, enabled: bool) -> None:
        if decisions and not enabled:
            raise KnowHowIntegrityError(
                "resume plan contains know_how_decisions but know-how retrieval "
                "is disabled"
            )
        if enabled:
            verify_know_how_decisions(decisions, self.decision_authority or {})

    def prompt_metrics(
        self,
        planner: Any,
        context: ResearchContext,
        *,
        planning_contract_context: str = "",
    ) -> dict[str, Any]:
        strict_transport_schema = bool(
            planner.last_prompt_metrics.get("structured_output_payload_bytes")
        )
        baseline = planner.request_metrics(
            context,
            planning_contract_context=planning_contract_context,
            strict_transport_schema=strict_transport_schema,
        )
        metrics = dict(planner.last_prompt_metrics)
        return {
            **metrics,
            "without_know_how_total_bytes": baseline["total_bytes"],
            "know_how_added_bytes": metrics["total_bytes"] - baseline["total_bytes"],
            "know_how_selected_count": len(self.selected_ids),
            "know_how_enabled": self.enabled,
        }

    def persist_prompt_metrics(
        self,
        metrics: Mapping[str, Any] | None,
        *,
        run_dir: Path,
        evidence: Any,
    ) -> None:
        if metrics is None:
            return
        path = run_dir / "planner_prompt_metrics.json"
        path.write_text(
            json.dumps(
                {"schema_version": "easyicu.planner_prompt_metrics/1", **metrics},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if evidence.get("planner_prompt_metrics") is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Exact initial Planner request byte metrics and bounded "
                    "know-how contribution."
                ),
                source_path=path,
                evidence_id="planner_prompt_metrics",
                producer="planner",
                generation_mode="deterministic_skill",
                inputs=(["know_how_prompt"] if self.prompt else []),
            )


def verify_know_how_decisions(
    decisions: Sequence[Any],
    authority: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail closed unless decisions exactly match this run's retrieval authority."""
    decided_cards: set[str] = set()
    for decision in decisions:
        card_id = str(decision.card_id)
        card = authority.get(card_id)
        if card is None:
            raise KnowHowIntegrityError(
                f"know-how decision references unretrieved card {card_id!r}"
            )
        if str(decision.card_version) != str(card.get("version")) or str(
            decision.card_sha256
        ) != str(card.get("file_sha256")):
            raise KnowHowIntegrityError(
                f"know-how decision changed version/SHA for {card_id!r}"
            )
        expected_citations = (card.get("claims") or {}).get(decision.claim_id)
        if expected_citations is None:
            raise KnowHowIntegrityError(
                "know-how decision references unknown claim "
                f"{card_id}.{decision.claim_id}"
            )
        if tuple(decision.citation_ids) != tuple(expected_citations):
            raise KnowHowIntegrityError(
                "know-how decision changed citation binding for "
                f"{card_id}.{decision.claim_id}"
            )
        decided_cards.add(card_id)
    missing = sorted(set(authority) - decided_cards)
    if missing:
        raise KnowHowIntegrityError(
            "Planner omitted claim-level dispositions for retrieved cards: "
            f"{missing!r}"
        )


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
    allow_curated_mvp: bool = False,
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
    selection = ResourceScheduler.select_protocols(
        registry=registry,
        query=ResourceSelectionQuery(
            purpose="planner",
            query=context.research_question,
            analysis_family=retrieval_family,
            database=database,
            available_input_roles=tuple(available_concepts),
        ),
        available_concepts=tuple(available_concepts),
        top_k=top_k,
        min_score=min_score,
        allowed_review_statuses=(
            ("curated_mvp", "clinical_reviewed")
            if allow_curated_mvp
            else ("clinical_reviewed",)
        ),
    )
    hits = selection.hits
    prompt = selection.prompt
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
    resource_receipt_bytes = (
        selection.receipt.model_dump_json(indent=2).encode("utf-8") + b"\n"
    )
    receipt_path = run_dir / "know_how_retrieval.json"
    prompt_path = run_dir / "know_how_prompt.md"
    resource_receipt_path = run_dir / "resource_selection_receipt.json"

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
    _write_or_verify(
        resource_receipt_path,
        resource_receipt_bytes,
        evidence=evidence,
        evidence_id="resource_selection_receipt",
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
    if evidence.get("resource_selection_receipt") is None:
        evidence.register_file(
            kind="log",
            description=("Host-allowlisted deterministic resource-selection receipt."),
            source_path=resource_receipt_path,
            evidence_id="resource_selection_receipt",
            producer="resource_scheduler",
            generation_mode="deterministic_skill",
            inputs=["know_how_retrieval", "know_how_prompt"],
        )
    return PreplanKnowHow(registry=registry, hits=hits, prompt=prompt)


__all__ = ["PreplanKnowHow", "prepare_preplan_know_how"]
