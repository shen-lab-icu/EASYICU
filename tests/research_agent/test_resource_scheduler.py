"""Host-allowlist and deterministic resource-scheduler contracts."""

from __future__ import annotations

import hashlib

from easyicu.research_agent.know_how import KnowHowRegistry
from easyicu.research_agent.resources import (
    ResourceCatalog,
    ResourceDescriptor,
    ResourceScheduler,
    ResourceSelectionPolicy,
    ResourceSelectionQuery,
    protocol_catalog_from_know_how,
)


def _planner_query(*, family: str = "association") -> ResourceSelectionQuery:
    return ResourceSelectionQuery(
        purpose="planner",
        query="Estimate the association of peak lactate with mortality.",
        analysis_family=family,
        database="miiv",
        available_input_roles=("lactate", "death"),
    )


def test_protocol_selection_is_deterministic_digest_bound_and_llm_free() -> None:
    registry = KnowHowRegistry.load()

    first = ResourceScheduler.select_protocols(
        registry=registry,
        query=_planner_query(),
        available_concepts=("lactate", "death"),
    )
    second = ResourceScheduler.select_protocols(
        registry=registry,
        query=_planner_query(),
        available_concepts=("lactate", "death"),
    )

    assert first.receipt == second.receipt
    assert [hit.card_id for hit in first.hits] == ["early_peak_lactate_association"]
    assert first.receipt.provider_calls == 0
    assert (
        first.receipt.projection_sha256
        == hashlib.sha256(first.prompt.encode("utf-8")).hexdigest()
    )
    assert first.receipt.projection_bytes == len(first.prompt.encode("utf-8"))


def test_protocol_scheduler_allows_an_honest_zero_match() -> None:
    selection = ResourceScheduler.select_protocols(
        registry=KnowHowRegistry.load(),
        query=ResourceSelectionQuery(
            purpose="planner",
            query="Estimate the KDIGO stage gradient for mortality.",
            analysis_family="association",
            database="miiv",
        ),
        available_concepts=("kdigo", "death"),
    )

    assert selection.hits == ()
    assert selection.receipt.selected == ()
    assert selection.receipt.candidate_count >= 0


def test_host_allowlist_rejects_wrong_family_and_unreviewed_resource() -> None:
    reviewed = ResourceDescriptor(
        resource_id="protocol:reviewed",
        version="1.0.0",
        sha256="1" * 64,
        kind="protocol",
        analysis_families=("prediction",),
        permissions=("planner_context",),
        review_status="clinical_reviewed",
    )
    unreviewed = ResourceDescriptor(
        resource_id="protocol:unreviewed",
        version="1.0.0",
        sha256="2" * 64,
        kind="protocol",
        analysis_families=("association",),
        permissions=("planner_context",),
        review_status="unreviewed",
    )
    catalog = ResourceCatalog((reviewed, unreviewed))
    policy = ResourceSelectionPolicy(
        allowed_kinds=("protocol",),
        allowed_review_statuses=("clinical_reviewed",),
        allowed_permissions=("planner_context",),
    )

    assert (
        catalog.allowlist(query=_planner_query(), policy=policy, kind="protocol") == ()
    )


def test_action_resource_requires_exact_typed_input_roles() -> None:
    action = ResourceDescriptor(
        resource_id="action:table_one",
        version="1.0.0",
        sha256="3" * 64,
        kind="action",
        analysis_families=("association",),
        required_input_roles=("analysis_cohort", "table_one_spec"),
        produced_output_roles=("table:table_one",),
        permissions=("coder_context",),
        review_status="validated",
    )
    catalog = ResourceCatalog((action,))
    policy = ResourceSelectionPolicy(
        allowed_kinds=("action",),
        allowed_review_statuses=("validated",),
        allowed_permissions=("coder_context",),
    )
    incomplete = ResourceSelectionQuery(
        purpose="coder",
        query="Build Table 1",
        analysis_family="association",
        available_input_roles=("analysis_cohort",),
    )
    complete = incomplete.model_copy(
        update={
            "available_input_roles": ("analysis_cohort", "table_one_spec"),
        }
    )

    assert catalog.allowlist(query=incomplete, policy=policy, kind="action") == ()
    assert catalog.allowlist(query=complete, policy=policy, kind="action") == (action,)


def test_know_how_adapter_catalog_is_stable_and_source_bound() -> None:
    registry = KnowHowRegistry.load()
    catalog = protocol_catalog_from_know_how(registry)

    assert catalog.sha256 == protocol_catalog_from_know_how(registry).sha256
    assert len(catalog.resources) == len(registry.cards)
    assert all(
        resource.sha256
        == registry.source_sha256(resource.resource_id.removeprefix("protocol:"))
        for resource in catalog.resources
    )
