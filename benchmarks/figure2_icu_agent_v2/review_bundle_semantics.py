"""Shared structural semantics for both Figure 2 review-bundle producers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CANONICAL_FILES = (
    "01_plan.json",
    "02_cohort.json",
    "03_results.json",
    "04_diagnostics.json",
    "05_evidence_manifest.json",
    "06_report.md",
    "07_run_receipt.json",
)
ARTIFACT_REFERENCE_FILES = frozenset(CANONICAL_FILES[:4] + ("06_report.md",))
SUBSTANTIVE_OUTPUT_FILES = (
    "02_cohort.json",
    "03_results.json",
    "04_diagnostics.json",
    "06_report.md",
)


def normalize_artifact_inventory(
    inventory: Mapping[str, Any],
    mandatory_artifacts: Sequence[str],
) -> dict[str, list[str]]:
    """Validate only references to canonical bundle files, without judging science."""

    labels = tuple(mandatory_artifacts)
    if set(inventory) != set(labels):
        raise ValueError("artifact_inventory must map every frozen mandatory artifact")
    normalized: dict[str, list[str]] = {}
    for label in labels:
        references = inventory[label]
        if (
            not isinstance(references, list)
            or not references
            or not all(
                isinstance(reference, str)
                and reference in ARTIFACT_REFERENCE_FILES
                for reference in references
            )
        ):
            raise ValueError(f"artifact_inventory has invalid references for {label!r}")
        normalized[label] = list(dict.fromkeys(references))
    return normalized


def substantive_file_flags(
    *,
    plan: Mapping[str, Any],
    cohort: Mapping[str, Any],
    results: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    report: str,
) -> dict[str, bool]:
    """Return file-level non-emptiness flags; these are not scientific validation."""

    values = {
        "01_plan.json": bool(plan),
        "02_cohort.json": bool(cohort),
        "03_results.json": bool(results),
        "04_diagnostics.json": bool(diagnostics),
        "06_report.md": bool(report.strip()),
    }
    return {name: values[name] for name in SUBSTANTIVE_OUTPUT_FILES}


def asserted_artifact_presence(
    inventory: Mapping[str, Sequence[str]],
    *,
    plan: Mapping[str, Any],
    cohort: Mapping[str, Any],
    results: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    report: str,
) -> dict[str, bool]:
    """Project producer assertions; human gates still determine adequacy."""

    substantive = {
        "01_plan.json": bool(plan),
        **substantive_file_flags(
            plan=plan,
            cohort=cohort,
            results=results,
            diagnostics=diagnostics,
            report=report,
        ),
    }
    return {
        label: all(substantive[reference] for reference in references)
        for label, references in inventory.items()
    }


__all__ = [
    "ARTIFACT_REFERENCE_FILES",
    "CANONICAL_FILES",
    "SUBSTANTIVE_OUTPUT_FILES",
    "asserted_artifact_presence",
    "normalize_artifact_inventory",
    "substantive_file_flags",
]
