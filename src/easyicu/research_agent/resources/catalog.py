"""Immutable resource catalog and Know-How protocol adapter."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from ..know_how import KnowHowRegistry
from .schema import ResourceDescriptor, ResourceSelectionPolicy, ResourceSelectionQuery


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ResourceCatalog:
    """Sorted, duplicate-free set of immutable resource descriptors."""

    def __init__(self, resources: Iterable[ResourceDescriptor]) -> None:
        by_id: dict[str, ResourceDescriptor] = {}
        for resource in resources:
            if resource.resource_id in by_id:
                raise ValueError(f"duplicate resource_id: {resource.resource_id}")
            by_id[resource.resource_id] = resource
        self._resources = tuple(by_id[key] for key in sorted(by_id))

    @property
    def resources(self) -> tuple[ResourceDescriptor, ...]:
        return self._resources

    @property
    def sha256(self) -> str:
        payload = [item.model_dump(mode="json") for item in self._resources]
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    def allowlist(
        self,
        *,
        query: ResourceSelectionQuery,
        policy: ResourceSelectionPolicy,
        kind: str,
    ) -> tuple[ResourceDescriptor, ...]:
        """Apply hard host permissions before any relevance ranking."""
        allowed_kinds = set(policy.allowed_kinds)
        allowed_statuses = set(policy.allowed_review_statuses)
        allowed_permissions = set(policy.allowed_permissions)
        required_permission = (
            "planner_context" if query.purpose == "planner" else "coder_context"
        )
        selected: list[ResourceDescriptor] = []
        for resource in self._resources:
            if resource.kind != kind or resource.kind not in allowed_kinds:
                continue
            if resource.review_status not in allowed_statuses:
                continue
            if required_permission not in resource.permissions:
                continue
            if not set(resource.permissions) <= allowed_permissions:
                continue
            if (
                resource.analysis_families
                and query.analysis_family not in resource.analysis_families
            ):
                continue
            if resource.kind != "protocol" and not set(
                resource.required_input_roles
            ) <= set(query.available_input_roles):
                continue
            selected.append(resource)
        return tuple(selected)

    @staticmethod
    def digest(resources: Iterable[ResourceDescriptor]) -> str:
        payload = [
            item.model_dump(mode="json")
            for item in sorted(resources, key=lambda value: value.resource_id)
        ]
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


def protocol_catalog_from_know_how(registry: KnowHowRegistry) -> ResourceCatalog:
    """Adapt reviewed Know-How cards into the shared protocol resource schema."""
    descriptors: list[ResourceDescriptor] = []
    for card in registry.cards:
        source_sha = registry.source_sha256(card.card_id)
        search_terms = tuple(
            dict.fromkeys([card.title, *card.clinical_topics, *card.topic_aliases])
        )
        projection = json.dumps(
            {
                "card_id": card.card_id,
                "version": card.version,
                "claims": [claim.model_dump(mode="json") for claim in card.claims],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        descriptors.append(
            ResourceDescriptor(
                resource_id=f"protocol:{card.card_id}",
                version=card.version,
                sha256=source_sha,
                kind="protocol",
                analysis_families=tuple(card.study_families),
                required_input_roles=tuple(card.required_concepts),
                permissions=("planner_context",),
                review_status=card.review_status,
                search_terms=search_terms,
                prompt_projection=projection,
            )
        )
    return ResourceCatalog(descriptors)


__all__ = ["ResourceCatalog", "protocol_catalog_from_know_how"]
