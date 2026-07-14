"""Step-scoped prompt and metadata projection for the analysis coder.

The planner owns the scientific contract.  This module only reduces transport:
it selects guidance from exact method/output structure and projects the already
registered variable metadata needed by the current step.  It never chooses an
exposure, outcome, cohort, method, or estimand.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .schema import AnalysisStep, ResearchContext


_COMPANION_SUFFIXES = (
    "_measured",
    "_first_time",
    "_last_time",
    "_first",
    "_max",
    "_min",
    "_mean",
    "_n",
)

_FIGURE_METHODS = frozenset(
    {
        "figure",
        "publication_figure",
        "visualization",
        "descriptive_visualization",
        "forest_plot",
        "kaplan_meier_plot",
    }
)
_TABLE_METHODS = frozenset(
    {
        "cohort_description",
        "data_quality",
        "descriptive_statistics",
        "descriptive_summary",
        "missingness_audit",
        "table_one",
    }
)
_TRAJECTORY_METHODS = frozenset(
    {
        "kmeans",
        "kmeans_clustering",
        "trajectory_clustering",
        "trajectory_phenotyping",
        "trajectory_stability",
    }
)
_ROBUSTNESS_METHODS = frozenset(
    {
        "cohort_definition_sensitivity",
        "prespecified_robustness_analysis",
        "robustness_analysis",
        "sensitivity_analysis",
    }
)


def normalised_method_head(method: object) -> str:
    """Return the exact scientific method head before an optional rider."""

    normalised = re.sub(
        r"[^a-z0-9]+", "_", str(method or "").strip().lower()
    ).strip("_")
    return normalised.split("_with_", 1)[0]


def _typed_products(values: Iterable[object]) -> tuple[tuple[str, str], ...]:
    products = []
    for raw in values:
        kind, separator, name = str(raw or "").strip().lower().partition(":")
        if separator and kind and name:
            products.append((kind, name))
    return tuple(products)


def _guide_segments(full_guide: str) -> dict[str, str]:
    """Split the versioned guide at stable semantic headings.

    Failing closed here is deliberate: a prompt-pack edit must preserve these
    headings or update this selector and its tests instead of silently sending
    the entire guide again.
    """

    anchors = {
        "figure": "- For rendering-only figure steps,",
        "trajectory": "- OPTIONAL trajectory:",
        "visual": '- Use matplotlib\'s "Agg" backend;',
        "source": "- When reporting a source-status count map,",
        "table": "TABLE-ONE / DESCRIPTIVE SUMMARIES:",
        "clinical": "CLINICAL SCORE AND MISSINGNESS SEMANTICS:",
        "hygiene": "PYTHON HYGIENE:",
        "robustness": "ROBUSTNESS:",
    }
    positions = {name: full_guide.find(anchor) for name, anchor in anchors.items()}
    missing = [name for name, position in positions.items() if position < 0]
    if missing:
        raise ValueError(f"Coder prompt is missing scoped section anchors: {missing}")
    order = [
        "figure",
        "trajectory",
        "visual",
        "source",
        "table",
        "clinical",
        "hygiene",
        "robustness",
    ]
    if [positions[name] for name in order] != sorted(positions.values()):
        raise ValueError("Coder prompt scoped section anchors are out of order")
    segments = {"core": full_guide[: positions["figure"]].strip()}
    for index, name in enumerate(order):
        end = positions[order[index + 1]] if index + 1 < len(order) else len(full_guide)
        segments[name] = full_guide[positions[name] : end].strip()
    return segments


def coder_guide_for_step(full_guide: str, step: AnalysisStep) -> str:
    """Select prompt sections from exact method and typed-product evidence."""

    sections = _guide_segments(full_guide)
    method = normalised_method_head(step.method)
    inputs = _typed_products(step.inputs or [])
    outputs = _typed_products(step.expected_outputs or [])
    output_kinds = {kind for kind, _ in outputs}
    input_names = {name for _, name in inputs}
    output_names = {name for _, name in outputs}

    selected = ["core"]
    is_figure = method in _FIGURE_METHODS or bool(
        output_kinds & {"figure", "plot", "chart", "heatmap"}
    )
    if is_figure:
        selected.extend(("figure", "visual"))
    if method in _TRAJECTORY_METHODS or step.trajectory_stability_spec is not None:
        selected.append("trajectory")
    if method in _TABLE_METHODS or bool(
        output_names & {"table_one", "missingness_audit", "source_status"}
    ):
        selected.extend(("source", "table"))
    if step.model_requirements or not (
        is_figure or method in _TABLE_METHODS or method in _TRAJECTORY_METHODS
    ):
        selected.extend(("clinical", "hygiene"))
    if method in _ROBUSTNESS_METHODS:
        selected.append("robustness")
    if any(name.endswith(("source_status", "missingness_audit")) for name in input_names):
        selected.append("source")

    unique = []
    for name in selected:
        if name not in unique:
            unique.append(name)
    return "\n\n".join(sections[name] for name in unique if sections[name]).strip()


def _variable_family(name: object) -> str:
    lowered = str(name or "").strip().lower()
    for suffix in _COMPANION_SUFFIXES:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def scoped_coder_context(
    context: ResearchContext,
    step: AnalysisStep,
    *,
    code: str = "",
    max_variables: int = 36,
) -> ResearchContext:
    """Project context variables to the current step without changing science."""

    declared = {
        str(value or "").strip().lower()
        for value in (step.inputs or [])
        if ":" not in str(value or "") and str(value or "").strip()
    }
    families = {_variable_family(value) for value in declared}
    direct = {
        str(value).strip().lower()
        for value in (context.target_outcome, context.primary_exposure)
        if value
    }
    seed_names = declared | direct
    if code:
        seed_names.update(
            variable.name.lower()
            for variable in context.variables
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(variable.name)}(?![A-Za-z0-9_])",
                code,
            )
        )
    source_concepts = {
        str(variable.source_concept).strip().lower()
        for variable in context.variables
        if variable.name.lower() in seed_names and variable.source_concept
    }
    priority = []
    referenced = []
    remaining = []
    for variable in context.variables:
        name = variable.name.lower()
        source_concept = str(variable.source_concept or "").strip().lower()
        if (
            name in declared
            or _variable_family(name) in families
            or name in direct
            or (source_concept and source_concept in source_concepts)
        ):
            priority.append(variable)
        elif code and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(variable.name)}(?![A-Za-z0-9_])",
            code,
        ):
            referenced.append(variable)
        else:
            remaining.append(variable)
    selected = (priority + referenced + remaining)[: max(1, int(max_variables))]
    return context.model_copy(update={"variables": selected})


__all__ = [
    "coder_guide_for_step",
    "normalised_method_head",
    "scoped_coder_context",
]
