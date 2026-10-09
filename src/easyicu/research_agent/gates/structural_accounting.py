"""Which tables a rendering step's structural-accounting guards protect.

Cohort flows, attrition, denominator reconciliations and source-availability
audits account for every row of a population.  A rendering step that draws one
may filter its rows only after failing closed on an incomplete validation mask,
and may print its counts as whole numbers only after failing closed on
fractional values (``preflight``'s structural filter and integer findings).
This owner decides, from a step's declared roles, which of its typed table
inputs are such tables.
"""

from __future__ import annotations

import re

from ..schema import AnalysisStep

STRUCTURAL_ACCOUNTING_PRODUCTS = frozenset(
    {
        "attrition",
        "cohort_accounting",
        "cohort_flow",
        "denominator_reconciliation",
        "source_availability",
        "source_availability_audit",
        "universe_count_reconciliation",
    }
)


def is_structural_accounting_name(value: object) -> bool:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if not token:
        return False
    if token in STRUCTURAL_ACCOUNTING_PRODUCTS:
        return True
    parts = {part for part in token.split("_") if part}
    if "attrition" in parts or "consort" in parts:
        return True
    population = {"cohort", "population", "participant", "eligibility", "denominator"}
    accounting = {"account", "accounting", "flow", "funnel", "reconciliation"}
    return bool(
        (parts & population and parts & accounting)
        or ("source" in parts and "availability" in parts)
        or ("universe" in parts and parts & {"count", "counts", "reconciliation"})
    )


def typed_table_products(tokens: object) -> set[str]:
    """The product names of the ``table:<name>`` tokens among ``tokens``."""

    products: set[str] = set()
    for raw in tokens or ():
        kind, separator, name = str(raw or "").strip().lower().partition(":")
        if separator and kind == "table" and name:
            products.add(name)
    return products


def structural_accounting_products(step: AnalysisStep) -> set[str]:
    """Resolve accounting inputs by semantic role, not an exact product name.

    Resolution order keeps the guard bound to the tables it protects: an
    accounting-named product wins; otherwise an accounting-role panel binds
    only its own declared ``source_products``; only a step whose own
    intent/outputs are accounting-shaped, with no panel-level binding to
    consult, treats every table input as an accounting table.
    """

    table_products = typed_table_products(step.inputs)
    matched = {
        product for product in table_products if is_structural_accounting_name(product)
    }
    if matched:
        return matched

    panels = list(step.figure_panels or [])
    panel_matched: set[str] = set()
    for panel in panels:
        panel_roles = (
            getattr(panel, "panel_id", ""),
            getattr(panel, "article_role", ""),
            getattr(panel, "figure_output", ""),
        )
        if not any(is_structural_accounting_name(role) for role in panel_roles):
            continue
        sources = typed_table_products(getattr(panel, "source_products", ()))
        panel_matched |= (sources & table_products) if sources else table_products
    if panel_matched:
        return panel_matched
    if panels and all(
        typed_table_products(getattr(panel, "source_products", ())) for panel in panels
    ):
        # Every panel binds explicit sources and none of them is an
        # accounting panel; the step-level label cannot widen that binding.
        return set()

    step_roles: list[object] = [step.intent, *(step.expected_outputs or [])]
    if any(is_structural_accounting_name(role) for role in step_roles):
        return table_products
    return set()


__all__ = [
    "STRUCTURAL_ACCOUNTING_PRODUCTS",
    "is_structural_accounting_name",
    "structural_accounting_products",
    "typed_table_products",
]
