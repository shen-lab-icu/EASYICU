"""Deterministic resource scheduling inside a host-owned allowlist."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from ..know_how import KnowHowHit, KnowHowIntegrityError, KnowHowRegistry
from .catalog import ResourceCatalog, protocol_catalog_from_know_how
from .schema import (
    ResourceSelectionPolicy,
    ResourceSelectionQuery,
    ResourceSelectionReceipt,
    SelectedResource,
)


@dataclass(frozen=True)
class ProtocolResourceSelection:
    """Planner protocol selection plus exact prompt projection and receipt."""

    registry: KnowHowRegistry
    hits: tuple[KnowHowHit, ...]
    prompt: str
    receipt: ResourceSelectionReceipt


class ResourceScheduler:
    """Select resources without an LLM and without expanding host authority."""

    @staticmethod
    def select_protocols(
        *,
        registry: KnowHowRegistry,
        query: ResourceSelectionQuery,
        available_concepts: tuple[str, ...],
        top_k: int = 3,
        min_score: float = 0.15,
    ) -> ProtocolResourceSelection:
        if query.purpose != "planner":
            raise ValueError("protocol resources may only be selected for Planner")
        policy = ResourceSelectionPolicy(
            allowed_kinds=("protocol",),
            allowed_review_statuses=("curated_mvp", "clinical_reviewed"),
            allowed_permissions=("planner_context",),
            max_protocols=top_k,
        )
        catalog = protocol_catalog_from_know_how(registry)
        allowlist = catalog.allowlist(query=query, policy=policy, kind="protocol")
        allowed_card_ids = {
            descriptor.resource_id.removeprefix("protocol:") for descriptor in allowlist
        }
        hits = tuple(
            registry.retrieve(
                query=query.query,
                study_family=query.analysis_family,
                database=query.database,
                available_concepts=available_concepts,
                top_k=min(top_k, policy.max_protocols),
                min_score=min_score,
            )
        )
        escaped = sorted(
            hit.card_id for hit in hits if hit.card_id not in allowed_card_ids
        )
        if escaped:
            raise KnowHowIntegrityError(
                "resource ranker escaped the host protocol allowlist: " f"{escaped!r}"
            )
        prompt = registry.render_prompt(hits)
        prompt_bytes = prompt.encode("utf-8")
        descriptors = {
            item.resource_id.removeprefix("protocol:"): item for item in allowlist
        }
        selected = tuple(
            SelectedResource(
                resource_id=f"protocol:{hit.card_id}",
                version=hit.version,
                sha256=hit.file_sha256,
                kind="protocol",
                score=hit.score,
                reasons=tuple(hit.match_reasons),
            )
            for hit in hits
        )
        for item in selected:
            descriptor = descriptors[item.resource_id.removeprefix("protocol:")]
            if item.version != descriptor.version or item.sha256 != descriptor.sha256:
                raise KnowHowIntegrityError(
                    f"selected resource coordinates changed for {item.resource_id}"
                )
        receipt = ResourceSelectionReceipt(
            query=query,
            policy=policy,
            catalog_sha256=catalog.sha256,
            allowlist_sha256=ResourceCatalog.digest(allowlist),
            candidate_count=len(allowlist),
            selected=selected,
            projection_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
            projection_bytes=len(prompt_bytes),
        )
        return ProtocolResourceSelection(
            registry=registry,
            hits=hits,
            prompt=prompt,
            receipt=receipt,
        )


__all__ = ["ProtocolResourceSelection", "ResourceScheduler"]
