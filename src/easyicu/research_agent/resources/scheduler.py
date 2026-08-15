"""Deterministic resource scheduling inside a host-owned allowlist."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..know_how import KnowHowHit, KnowHowIntegrityError, KnowHowRegistry
from .catalog import ResourceCatalog, protocol_catalog_from_know_how
from .schema import (
    ResourceReviewStatus,
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


@dataclass(frozen=True)
class ResourceSelection:
    """Generic non-protocol resource selection and its prompt projection."""

    resources: tuple
    prompt: str
    receipt: ResourceSelectionReceipt


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", value.lower()))


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
        allowed_review_statuses: tuple[ResourceReviewStatus, ...] = (
            "clinical_reviewed",
        ),
    ) -> ProtocolResourceSelection:
        if query.purpose != "planner":
            raise ValueError("protocol resources may only be selected for Planner")
        policy = ResourceSelectionPolicy(
            allowed_kinds=("protocol",),
            allowed_review_statuses=allowed_review_statuses,
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
                allowed_review_statuses=allowed_review_statuses,
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

    @staticmethod
    def select_resources(
        *,
        catalog: ResourceCatalog,
        query: ResourceSelectionQuery,
        policy: ResourceSelectionPolicy,
        kind: str,
        required_resource_ids: tuple[str, ...] = (),
    ) -> ResourceSelection:
        """Rank action/software/data resources inside an exact host allowlist."""

        if kind == "protocol":
            raise ValueError("protocol resources require the Know-How scheduler")
        allowlist = catalog.allowlist(query=query, policy=policy, kind=kind)
        required = tuple(dict.fromkeys(str(value) for value in required_resource_ids))
        if len(required) > policy.limit_for(kind):
            raise ValueError(
                f"required {kind} resources exceed the bounded selection limit: "
                f"{len(required)}>{policy.limit_for(kind)}"
            )
        allowed_by_id = {resource.resource_id: resource for resource in allowlist}
        missing_required = sorted(set(required) - set(allowed_by_id))
        if missing_required:
            raise ValueError(
                f"required {kind} resources are unavailable in the verified "
                f"catalog: {missing_required!r}"
            )
        query_tokens = _tokens(
            " ".join((query.query, query.analysis_family, query.step_role or ""))
        )
        ranked: list[tuple[float, object, tuple[str, ...]]] = []
        for resource in allowlist:
            if resource.resource_id in required:
                continue
            resource_tokens = _tokens(" ".join(resource.search_terms))
            overlap = len(query_tokens & resource_tokens)
            family_match = query.analysis_family in resource.analysis_families
            if overlap == 0 and not family_match:
                continue
            denominator = max(1, len(query_tokens | resource_tokens))
            score = min(1.0, overlap / denominator + (0.25 if family_match else 0.0))
            reasons = []
            if family_match:
                reasons.append("analysis_family")
            if overlap:
                reasons.append("lexical_overlap")
            ranked.append((score, resource, tuple(reasons)))
        ranked.sort(key=lambda item: (-item[0], item[1].resource_id))
        limit = policy.limit_for(kind)
        selected_rows = [
            (1.0, allowed_by_id[resource_id], ("host_required",))
            for resource_id in required
        ]
        selected_rows.extend(ranked[: limit - len(selected_rows)])
        selected_resources = tuple(item[1] for item in selected_rows)
        selected = tuple(
            SelectedResource(
                resource_id=resource.resource_id,
                version=resource.version,
                sha256=resource.sha256,
                kind=resource.kind,
                score=score,
                reasons=reasons,
            )
            for score, resource, reasons in selected_rows
        )
        prompt = "\n\n".join(
            resource.prompt_projection for resource in selected_resources
        )
        prompt_bytes = prompt.encode("utf-8")
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
        return ResourceSelection(
            resources=selected_resources,
            prompt=prompt,
            receipt=receipt,
        )


__all__ = ["ProtocolResourceSelection", "ResourceScheduler", "ResourceSelection"]
