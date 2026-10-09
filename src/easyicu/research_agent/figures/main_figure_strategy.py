"""Which step figure bundle leads the article, by its family's main-figure strategy.

Owner
-----
A study family's main figure has a hero role and a pool of roles it should
cover.  This module reads a figure contract's panel roles and chart types,
says whether a contract or a step bundle already satisfies its family's
strategy, and orders step bundles for promotion to the article's main figure.
``figures.skill`` selects and promotes the bundle; it asks this module how the
bundles rank.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..planning.study_design import infer_study_design_family
from ..schema import ResearchContext


_PRIMARY_RESULT_PANEL_ROLES = {
    "descriptive_result",
    "primary_estimand",
    "temporal_absolute_risk",
    "survival_effect",
    "clinical_utility",
}
_VALIDATION_PANEL_ROLES = {
    "model_performance",
    "calibration",
    "validation",
    "explainability",
    "transportability",
}
_CONTEXT_PANEL_ROLES = {
    "cohort_accounting",
    "baseline_context",
    "data_quality",
    "overview",
    "relationship",
    "heterogeneity",
    "distribution",
}
_SUPPLEMENTAL_PANEL_ROLES = {
    "robustness",
    "audit",
    "diagnostics",
    "stability",
    "supplementary_provenance",
}
_PRIMARY_CONTEXT_CHART_TOKENS = (
    "absolute_risk",
    "event_rate",
    "prevalence",
    "incidence",
    "survival",
)
_PRIMARY_PUBLICATION_ROLE_POOLS: Dict[str, set[str]] = {
    "association": {
        "descriptive_result",
        "primary_estimand",
        "robustness",
        "data_quality",
    },
    "prediction": {
        "model_performance",
        "calibration",
        "validation",
        "data_quality",
    },
    "time_to_event": {
        "temporal_absolute_risk",
        "survival_effect",
        "diagnostics",
    },
    "phenotyping": {
        "phenotype_structure",
        "phenotype_profile",
        "stability",
        "data_quality",
    },
    "causal_emulation": {
        "causal_protocol",
        "balance_positivity",
        "causal_contrast",
        "robustness",
    },
    "descriptive": {
        "distribution",
        "descriptive_result",
        "cohort_accounting",
        "data_quality",
    },
}
_PRIMARY_PUBLICATION_HERO_ROLES: Dict[str, str] = {
    "association": "descriptive_result",
    "prediction": "calibration",
    "time_to_event": "temporal_absolute_risk",
    "phenotyping": "phenotype_structure",
    "causal_emulation": "causal_protocol",
    "descriptive": "distribution",
}
_PRIMARY_PUBLICATION_MIN_ROLE_COUNTS: Dict[str, int] = {
    "association": 3,
    "prediction": 3,
    "time_to_event": 2,
    "phenotyping": 3,
    "causal_emulation": 3,
    "descriptive": 2,
}


def bundle_primary_strategy_ready(
    context: ResearchContext,
    bundle: Dict[str, Any],
) -> bool:
    """Return True when a step-level bundle is rich enough for the main figure."""

    payload = bundle.get("contract_payload")
    if not isinstance(payload, dict):
        return False
    return contract_primary_strategy_ready(context, payload)


def contract_primary_strategy_ready(
    context: ResearchContext,
    payload: Dict[str, Any],
) -> bool:
    """Return True when a figure contract is rich enough for the main figure."""

    family = str(infer_study_design_family(context))
    role_pool = _PRIMARY_PUBLICATION_ROLE_POOLS.get(family)
    if not role_pool:
        return True
    roles = _contract_payload_roles(payload)
    hero_role = _PRIMARY_PUBLICATION_HERO_ROLES.get(family)
    if hero_role and hero_role not in roles:
        return False
    minimum = min(
        len(role_pool),
        _PRIMARY_PUBLICATION_MIN_ROLE_COUNTS.get(family, min(3, len(role_pool))),
    )
    return len(roles & role_pool) >= minimum


def _bundle_is_supplementary_surface(bundle: Dict[str, Any]) -> bool:
    """Whether the plan placed every panel of this surface in the supplement."""

    payload = bundle.get("contract_payload")
    panels = payload.get("panels") if isinstance(payload, dict) else None
    if not isinstance(panels, list) or not panels:
        return False
    placements = []
    for panel in panels:
        metadata = panel.get("metadata") if isinstance(panel, dict) else None
        placements.append(
            str((metadata or {}).get("placement") or "").strip().lower()
            if isinstance(metadata, dict)
            else ""
        )
    return all(placement == "supplementary" for placement in placements)


def step_publication_bundle_rank(
    bundle: Dict[str, Any],
    *,
    context: Optional[ResearchContext] = None,
) -> Tuple[int, int, int, int, int, int, int, int, str]:
    """Order step bundles for promotion to the article's main figure.

    A surface the plan placed in the supplement never leads, and a bundle that
    already satisfies the study family's main-figure strategy (its hero role
    and enough of its roles) comes first.  The remaining heuristics order the
    rest; among equals the bundle carrying the family's hero role, then the one
    covering more of its roles, is preferred.
    """

    roles = _bundle_contract_roles(bundle)
    chart_types = bundle_contract_chart_types(bundle)
    step_text = str(bundle.get("step_id") or "").lower()
    stem_text = str(bundle.get("stem") or "").lower()
    text = f"{step_text} {stem_text}"
    generic_penalty = 1 if stem_text in {"publication_figure", "figure"} else 0
    primary_role_count = len(roles & _PRIMARY_RESULT_PANEL_ROLES)
    has_absolute_context = any(
        any(token in chart_type for token in _PRIMARY_CONTEXT_CHART_TOKENS)
        for chart_type in chart_types
    )
    supplemental_only = bool(roles) and roles <= _SUPPLEMENTAL_PANEL_ROLES
    sensitivity_or_robustness = "sensitivity" in text or "robust" in text

    if {"descriptive_result", "primary_estimand"} <= roles:
        family_rank = 0
    elif primary_role_count and has_absolute_context:
        family_rank = 1
    elif primary_role_count:
        family_rank = 2
    elif "prediction" in text or "calibration" in text or "discrimination" in text:
        family_rank = 3
    elif roles & _VALIDATION_PANEL_ROLES:
        family_rank = 4
    elif "overlap" in text or "eligibility" in text or "definition" in text:
        family_rank = 5
    elif roles & _CONTEXT_PANEL_ROLES:
        family_rank = 6
    elif sensitivity_or_robustness or supplemental_only:
        family_rank = 7
    elif "primary" in text or "association" in text:
        family_rank = 8
    else:
        family_rank = 9
    family = str(infer_study_design_family(context)) if context is not None else ""
    role_pool = _PRIMARY_PUBLICATION_ROLE_POOLS.get(family, set())
    hero_role = _PRIMARY_PUBLICATION_HERO_ROLES.get(family)
    payload = bundle.get("contract_payload")
    strategy_ready = bool(
        role_pool
        and isinstance(payload, dict)
        and contract_primary_strategy_ready(context, payload)
    )
    return (
        1 if _bundle_is_supplementary_surface(bundle) else 0,
        0 if strategy_ready else 1,
        family_rank,
        1 if supplemental_only else 0,
        0 if hero_role and hero_role in roles else 1,
        -len(roles & role_pool),
        generic_penalty,
        -int(bundle.get("order", 0)),
        str(bundle.get("stem") or ""),
    )


def _bundle_contract_roles(bundle: Dict[str, Any]) -> set[str]:
    payload = bundle.get("contract_payload")
    if not isinstance(payload, dict):
        return set()
    return _contract_payload_roles(payload)


def _contract_payload_roles(payload: Dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for panel in payload.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        role = str(panel.get("role") or "").strip().lower()
        if role:
            roles.add(role)
    for key in ("hero_role", "primary_role", "figure_role"):
        role = str(payload.get(key) or "").strip().lower()
        if role and role != "publication_figure":
            roles.add(role)
    return roles


def bundle_contract_chart_types(bundle: Dict[str, Any]) -> set[str]:
    payload = bundle.get("contract_payload")
    if not isinstance(payload, dict):
        return set()
    return contract_payload_chart_types(payload)


def contract_payload_chart_types(payload: Dict[str, Any]) -> set[str]:
    chart_types: set[str] = set()
    for panel in payload.get("panels") or []:
        if not isinstance(panel, dict):
            continue
        candidates = [panel.get("chart_type")]
        metadata = panel.get("metadata")
        if isinstance(metadata, dict):
            candidates.append(metadata.get("chart_type"))
        for value in candidates:
            chart_type = str(value or "").strip().lower()
            if chart_type:
                chart_types.add(chart_type)
    return chart_types


__all__ = [
    "bundle_contract_chart_types",
    "bundle_primary_strategy_ready",
    "contract_payload_chart_types",
    "contract_primary_strategy_ready",
    "step_publication_bundle_rank",
]
