"""Dependency-neutral citation authority for scientific plans.

Planning owns the rule that a proposed plan must cite only the sealed
pre-plan literature bundle and explain how each source governs the design.
Agent adapters may re-export this contract for compatibility, but planning
must never import those adapters to validate a replan candidate.
"""

from __future__ import annotations

from typing import Sequence

from ..schema import AnalysisPlan, ResearchContext
from .method_literature import (
    METHOD_CARDS,
    method_binding_support,
)
from .scientific_review import required_method_layers_for_plan


def normalize_literature_citation_keys(
    values: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return the ordered, unique run-bound citation-key authority."""

    return tuple(
        dict.fromkeys(
            str(value or "").strip()
            for value in (values or [])
            if str(value or "").strip()
        )
    )


def allowed_method_source_keys(allowed_keys: Sequence[str]) -> tuple[str, ...]:
    """Return method-card sources actually present in this run's bundle."""

    allowed = set(normalize_literature_citation_keys(allowed_keys))
    return tuple(
        dict.fromkeys(
            card.source_key for card in METHOD_CARDS if card.source_key in allowed
        )
    )


def _method_layers_by_source_key() -> dict[str, set[str]]:
    layers: dict[str, set[str]] = {}
    for card in METHOD_CARDS:
        layers.setdefault(card.source_key, set()).add(card.layer)
    return layers


def _method_elements_by_source_key() -> dict[str, set[str]]:
    elements: dict[str, set[str]] = {}
    for card in METHOD_CARDS:
        elements.setdefault(card.source_key, set()).update(card.design_elements)
    return elements


def _available_method_sources_by_element(
    method_source_keys: set[str],
    allowed_keys: set[str],
) -> dict[str, list[str]]:
    sources: dict[str, set[str]] = {}
    for card in METHOD_CARDS:
        if (
            card.source_key not in method_source_keys
            or card.source_key not in allowed_keys
        ):
            continue
        for element in card.design_elements:
            sources.setdefault(element, set()).add(card.source_key)
    return {element: sorted(keys) for element, keys in sources.items()}


def validate_literature_citation_bindings(
    plan: AnalysisPlan,
    allowed_keys: Sequence[str],
    *,
    context: ResearchContext | None = None,
    direct_comparator_keys: Sequence[str] = (),
) -> None:
    """Reject invented keys and unbound scientific steps."""

    allowed = set(allowed_keys)
    declared = {key for step in plan.steps for key in step.literature_citation_keys}
    unknown = sorted(declared - allowed)
    if unknown:
        raise ValueError(
            "Planner cited keys outside this run's pre-plan LiteratureBundle: "
            + ", ".join(unknown)
        )
    unbound = [
        step.step_id
        for step in plan.steps
        if step.planned_analysis_role in {"primary", "secondary", "sensitivity"}
        and not step.literature_citation_keys
    ]
    if allowed and unbound:
        raise ValueError(
            "Each primary/secondary/sensitivity plan step must bind an exact key "
            "from the pre-plan LiteratureBundle; unbound steps: " + ", ".join(unbound)
        )
    design_unbound = []
    for step in plan.steps:
        if step.planned_analysis_role not in {
            "primary",
            "secondary",
            "sensitivity",
        }:
            continue
        cited = set(step.literature_citation_keys)
        explained = {
            binding.citation_key for binding in step.literature_design_bindings
        }
        missing = sorted(cited - explained)
        if not step.literature_design_bindings or missing:
            design_unbound.append(
                step.step_id
                + (f" (unexplained: {', '.join(missing)})" if missing else "")
            )
    if allowed and design_unbound:
        raise ValueError(
            "Each primary/secondary/sensitivity plan step must explain how every "
            "cited source governs an exact design decision; missing or incomplete "
            "literature_design_bindings: "
            + ", ".join(design_unbound)
        )
    method_source_keys = set(allowed_method_source_keys(allowed_keys))
    method_unbound = [
        step.step_id
        for step in plan.steps
        if step.planned_analysis_role in {"primary", "secondary", "sensitivity"}
        and method_source_keys.isdisjoint(step.literature_citation_keys)
    ]
    if method_source_keys and method_unbound:
        raise ValueError(
            "Each primary/secondary/sensitivity plan step must bind at least one "
            "method-source key from the run's pre-plan LiteratureBundle; steps "
            "citing only topic/data sources: "
            + ", ".join(method_unbound)
        )
    unsupported_method_bindings: list[str] = []
    method_elements_by_source = _method_elements_by_source_key()
    available_sources_by_element = _available_method_sources_by_element(
        method_source_keys,
        allowed,
    )
    matched_layers_by_step: dict[str, set[str]] = {}
    for step in plan.steps:
        if step.planned_analysis_role not in {
            "primary",
            "secondary",
            "sensitivity",
        }:
            continue
        matched_layers: set[str] = set()
        for binding in step.literature_design_bindings:
            support = method_binding_support(
                binding.citation_key,
                binding.design_elements,
            )
            matched_layers.update(support["matched_layers"])
            unsupported = support["unsupported_design_elements"]
            if support["method_source"] and unsupported:
                alternatives = ", ".join(
                    f"{element}={available_sources_by_element.get(element, [])!r}"
                    for element in unsupported
                )
                unsupported_method_bindings.append(
                    f"{step.step_id}:{binding.citation_key}="
                    + ",".join(unsupported)
                    + " (source supports: "
                    + ",".join(
                        sorted(
                            method_elements_by_source.get(binding.citation_key, set())
                        )
                    )
                    + "; available run sources by requested element: "
                    + alternatives
                    + ")"
                )
        matched_layers_by_step[step.step_id] = matched_layers
    if unsupported_method_bindings:
        raise ValueError(
            "Planner bound method sources to design elements their curated "
            "method cards do not support: "
            + "; ".join(unsupported_method_bindings)
        )
    method_card_unbound = sorted(
        step_id
        for step_id, layers in matched_layers_by_step.items()
        if not layers
    )
    if method_source_keys and method_card_unbound:
        raise ValueError(
            "Each scientific step must bind at least one exact method card "
            "through a supported design element; citation presence alone is "
            "insufficient: "
            + ", ".join(method_card_unbound)
        )
    if context is not None and method_source_keys:
        layer_by_key = _method_layers_by_source_key()
        required_layers = set(required_method_layers_for_plan(plan, context))
        available_layers = {
            layer
            for key in method_source_keys
            for layer in layer_by_key.get(key, set())
        }
        enforceable_layers = required_layers & available_layers
        cited_layers = {
            layer
            for layers in matched_layers_by_step.values()
            for layer in layers
        }
        missing_layers = sorted(enforceable_layers - cited_layers)
        if missing_layers:
            raise ValueError(
                "The scientific plan does not bind method sources for all "
                "case-applicable design decisions; missing method layers: "
                + ", ".join(missing_layers)
            )
    direct_keys = {
        key
        for key in normalize_literature_citation_keys(direct_comparator_keys)
        if key in allowed
    }
    if direct_keys:
        primary_bound = {
            key
            for step in plan.steps
            if step.planned_analysis_role == "primary"
            for key in step.literature_citation_keys
        }
        if direct_keys.isdisjoint(primary_bound):
            raise ValueError(
                "The primary scientific plan does not bind any screened direct "
                "comparator from the run's pre-plan LiteratureBundle: "
                + ", ".join(sorted(direct_keys))
            )


__all__ = [
    "allowed_method_source_keys",
    "normalize_literature_citation_keys",
    "validate_literature_citation_bindings",
]
