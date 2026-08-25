"""Dependency-neutral Planner contract for manuscript figure panels.

The declared input products establish lineage, but they do not establish what
reader-facing article role a panel serves or which chart grammar it will use.
This contract keeps those semantics explicit without importing plan, figure,
or reporting owners.
"""

from __future__ import annotations

from typing import Any, List, Literal, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .product_identity import is_canonical_typed_product_token


class PlannedFigurePanelSpec(BaseModel):
    """Planner-owned article role and chart grammar for one figure panel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.planned_figure_panel/1"] = (
        "easyicu.planned_figure_panel/1"
    )
    panel_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    figure_output: str = Field(pattern=r"^figure:[a-z][a-z0-9_]{0,79}$")
    article_role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    chart_type: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    source_products: List[str] = Field(min_length=1, max_length=16)

    @field_validator("source_products")
    @classmethod
    def _source_products_are_unique_typed_inputs(cls, values: List[str]) -> List[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not is_canonical_typed_product_token(value) for value in cleaned):
            raise ValueError(
                "source_products must contain canonical typed kind:product inputs"
            )
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("source_products must be unique")
        return cleaned


class DeterministicFigurePanelTemplate(BaseModel):
    """Panel contract shared by a deterministic renderer and plan shaping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    panel_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    article_role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    chart_type: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    source_products: Tuple[str, ...] = Field(min_length=1, max_length=16)

    @field_validator("source_products")
    @classmethod
    def _template_sources_are_unique_typed_inputs(
        cls, values: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        cleaned = tuple(str(value or "").strip() for value in values)
        if any(not is_canonical_typed_product_token(value) for value in cleaned):
            raise ValueError(
                "source_products must contain canonical typed kind:product inputs"
            )
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("source_products must be unique")
        return cleaned

    def bind(self, *, figure_output: str) -> PlannedFigurePanelSpec:
        return PlannedFigurePanelSpec(
            panel_id=self.panel_id,
            figure_output=figure_output,
            article_role=self.article_role,
            chart_type=self.chart_type,
            source_products=list(self.source_products),
        )


EXPOSURE_OUTCOME_DISTRIBUTION_INPUT = "table:exposure_outcome_distribution"
GROUPED_DESCRIPTIVE_DISTRIBUTION_INPUT = "table:distribution_prevalence"
MISSINGNESS_MEASUREMENT_AUDIT_INPUT = "table:missingness_measurement_audit"
MEASUREMENT_PROCESS_AUDIT_INPUT = "table:measurement_process_audit"
COHORT_FLOW_INPUT = "table:cohort_flow"
ROBUSTNESS_FIGURE_INPUT = "table:robustness_matrix"
ROBUSTNESS_PRIMARY_ESTIMATE_INPUT = "statistic:primary_or"
ROBUSTNESS_PRIMARY_EFFECT_INPUT = "statistic:primary_effect"
ROBUSTNESS_COMPLETE_CASE_INPUT = "statistic:complete_case_n"
ROBUSTNESS_FIGURE_KNOWN_INPUTS = frozenset(
    {
        ROBUSTNESS_FIGURE_INPUT,
        "table:robustness_summary",
        "statistic:robustness_summary",
        ROBUSTNESS_PRIMARY_EFFECT_INPUT,
        ROBUSTNESS_PRIMARY_ESTIMATE_INPUT,
        ROBUSTNESS_COMPLETE_CASE_INPUT,
    }
)
LANDMARK_ASSOCIATION_COMPOSITE_INPUTS = frozenset(
    {
        "table:absolute_risk_context",
        "table:robustness_summary",
    }
)
ASSOCIATION_SENSITIVITY_COMPOSITE_FIXED_INPUTS = frozenset(
    {
        "table:exposure_outcome_distribution",
        "table:adjusted_association_estimates",
        "table:exposure_component_completeness_audit",
    }
)
COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS = (
    "table:cohort_flow",
    "table:table_one",
    "table:adjusted_association_estimates",
    "table:robustness_matrix",
)
ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS = (
    "table:absolute_risk_context",
    "table:adjusted_association_estimates",
    "table:robustness_matrix",
    "table:robustness_summary",
)


def absolute_risk_association_composite_panels(
    source_products: Sequence[str],
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind absolute risk, adjusted association, robustness, and quality."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    if cleaned != ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS:
        raise ValueError(
            "absolute-risk association composite requires its four exact tables"
        )
    return (
        DeterministicFigurePanelTemplate(
            panel_id="absolute_risk_context",
            article_role="descriptive_result",
            chart_type="dot_interval_absolute_risk",
            source_products=("table:absolute_risk_context",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="primary_adjusted_association",
            article_role="primary_estimand",
            chart_type="forest",
            source_products=("table:adjusted_association_estimates",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="robustness_estimates",
            article_role="robustness",
            chart_type="sensitivity_forest",
            source_products=("table:robustness_matrix",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="robustness_ranges",
            article_role="robustness",
            chart_type="specification_grid",
            source_products=("table:robustness_summary",),
        ),
    )


def cohort_balance_association_composite_panels(
    source_products: Sequence[str],
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind cohort, balance, primary-association, and robustness panels."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    if cleaned != COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS:
        raise ValueError(
            "cohort-balance association composite requires its four exact tables"
        )
    return (
        DeterministicFigurePanelTemplate(
            panel_id="cohort_accounting",
            article_role="cohort_accounting",
            chart_type="cohort_flow",
            source_products=("table:cohort_flow",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="baseline_balance",
            article_role="descriptive_result",
            chart_type="standardized_difference",
            source_products=("table:table_one",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="primary_adjusted_association",
            article_role="primary_estimand",
            chart_type="forest_plot",
            source_products=("table:adjusted_association_estimates",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="robustness_estimates",
            article_role="robustness",
            chart_type="forest_plot",
            source_products=("table:robustness_matrix",),
        ),
    )


def association_sensitivity_composite_panels(
    source_products: Sequence[str],
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind a scientific-sensitivity association display to four typed tables."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    extra = [
        value
        for value in cleaned
        if value not in ASSOCIATION_SENSITIVITY_COMPOSITE_FIXED_INPUTS
    ]
    if (
        len(cleaned) != 4
        or len(cleaned) != len(set(cleaned))
        or not ASSOCIATION_SENSITIVITY_COMPOSITE_FIXED_INPUTS <= set(cleaned)
        or len(extra) != 1
        or not extra[0].startswith("table:")
    ):
        raise ValueError(
            "association sensitivity composite requires three fixed tables "
            "and one scientific-sensitivity table"
        )
    sensitivity = extra[0]
    return (
        DeterministicFigurePanelTemplate(
            panel_id="absolute_risk_context",
            article_role="descriptive_result",
            chart_type="grouped_absolute_risk",
            source_products=("table:exposure_outcome_distribution",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="primary_adjusted_association",
            article_role="primary_estimand",
            chart_type="forest_plot",
            source_products=("table:adjusted_association_estimates",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="scientific_sensitivity",
            article_role="robustness",
            chart_type="sensitivity_forest_plot",
            source_products=(sensitivity,),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="component_completeness",
            article_role="data_quality",
            chart_type="availability_heatmap",
            source_products=("table:exposure_component_completeness_audit",),
        ),
    )


def _landmark_curve_product(source_products: Sequence[str]) -> str | None:
    reserved = {
        "table:absolute_risk_context",
        "table:robustness_summary",
    }
    matches = [
        value
        for value in source_products
        if value.startswith("table:")
        and value not in reserved
        and value.partition(":")[2]
        not in {"measurement_process", "measurement_process_audit"}
    ]
    return matches[0] if len(matches) == 1 else None


def _measurement_process_product(source_products: Sequence[str]) -> str | None:
    matches = [
        value
        for value in source_products
        if value.startswith("table:")
        and value.partition(":")[2] in {"measurement_process", "measurement_process_audit"}
    ]
    return matches[0] if len(matches) == 1 else None


def landmark_association_composite_panels(
    source_products: Sequence[str],
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind a four-panel landmark-association display to typed parents."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    curve = _landmark_curve_product(cleaned)
    measurement = _measurement_process_product(cleaned)
    if (
        curve is None
        or measurement is None
        or len(cleaned) != 4
        or len(cleaned) != len(set(cleaned))
        or not LANDMARK_ASSOCIATION_COMPOSITE_INPUTS <= set(cleaned)
    ):
        raise ValueError("landmark composite requires its four exact typed tables")
    return (
        DeterministicFigurePanelTemplate(
            panel_id="association_curve",
            article_role="primary_estimand",
            chart_type="marginal_effect_panel",
            source_products=(curve,),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="absolute_risk_context",
            article_role="descriptive_result",
            chart_type="dot_interval_absolute_risk",
            source_products=("table:absolute_risk_context",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="robustness_summary",
            article_role="robustness",
            chart_type="specification_grid",
            source_products=("table:robustness_summary",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="measurement_process",
            article_role="data_quality",
            chart_type="availability_panel",
            source_products=(measurement,),
        ),
    )


DATA_QUALITY_AUDIT_ROLES = (
    "measurement_missingness",
    "measurement_process",
)
_CANONICAL_DATA_QUALITY_ROLE_BY_INPUT = {
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT: "measurement_missingness",
    MEASUREMENT_PROCESS_AUDIT_INPUT: "measurement_process",
}


def data_quality_audit_source_candidates(
    steps: Sequence[Any],
) -> dict[str, list[tuple[str, str]]]:
    """Return plan-declared table sources grouped by typed audit meaning."""

    candidates: dict[str, list[tuple[str, str]]] = {
        role: [] for role in DATA_QUALITY_AUDIT_ROLES
    }
    for step in steps:
        audit_spec = getattr(step, "measurement_audit_spec", None)
        for raw_output in getattr(step, "expected_outputs", ()) or ():
            output = str(raw_output or "").strip()
            kind, separator, product = output.partition(":")
            if kind != "table" or not separator:
                continue
            role = audit_spec.audit_for(product) if audit_spec is not None else None
            if role is None:
                role = _CANONICAL_DATA_QUALITY_ROLE_BY_INPUT.get(output)
            if role in candidates:
                candidates[role].append(
                    (output, str(getattr(step, "step_id", "") or ""))
                )
    return candidates


def resolve_data_quality_figure_inputs(
    inputs: Sequence[Any],
    *,
    steps: Sequence[Any] = (),
) -> dict[str, str] | None:
    """Resolve one exact missingness/process pair without spelling inference."""

    input_keys = [str(value or "").strip() for value in inputs]
    if len(input_keys) != 2 or len(set(input_keys)) != 2:
        return None
    roles_by_input: dict[str, set[str]] = {}
    for role, values in data_quality_audit_source_candidates(steps).items():
        for source, _step_id in values:
            roles_by_input.setdefault(source, set()).add(role)
    resolved: dict[str, str] = {}
    for input_key in input_keys:
        kind, separator, _product = input_key.partition(":")
        if kind != "table" or not separator:
            return None
        roles = roles_by_input.get(input_key, set())
        if not roles:
            canonical_role = _CANONICAL_DATA_QUALITY_ROLE_BY_INPUT.get(input_key)
            roles = {canonical_role} if canonical_role is not None else set()
        if len(roles) != 1:
            return None
        role = next(iter(roles))
        if role in resolved:
            return None
        resolved[role] = input_key
    if set(resolved) != set(DATA_QUALITY_AUDIT_ROLES):
        return None
    return resolved


EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="exposure_prevalence",
        article_role="distribution",
        chart_type="prevalence_panel",
        source_products=(EXPOSURE_OUTCOME_DISTRIBUTION_INPUT,),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="outcome_absolute_risk",
        article_role="descriptive_result",
        chart_type="dot_interval_absolute_risk",
        source_products=(EXPOSURE_OUTCOME_DISTRIBUTION_INPUT,),
    ),
)
EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="exposure_prevalence",
        article_role="distribution",
        chart_type="prevalence_panel",
        source_products=(EXPOSURE_OUTCOME_DISTRIBUTION_INPUT,),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="outcome_absolute_risk",
        article_role="descriptive_result",
        chart_type="point_absolute_risk",
        source_products=(EXPOSURE_OUTCOME_DISTRIBUTION_INPUT,),
    ),
)
GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="grouped_distribution",
        article_role="distribution",
        chart_type="point_range",
        source_products=(GROUPED_DESCRIPTIVE_DISTRIBUTION_INPUT,),
    ),
)
COHORT_FLOW_FIGURE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="cohort_accounting",
        article_role="cohort_accounting",
        chart_type="cohort_flow",
        source_products=(COHORT_FLOW_INPUT,),
    ),
)
DATA_QUALITY_FIGURE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="source_availability",
        article_role="data_quality",
        chart_type="availability_panel",
        source_products=(MISSINGNESS_MEASUREMENT_AUDIT_INPUT,),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="measurement_process_coverage",
        article_role="data_quality",
        chart_type="coverage_heatmap",
        source_products=(MEASUREMENT_PROCESS_AUDIT_INPUT,),
    ),
)


def measurement_availability_figure_panels(
    source_product: str,
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Return the single-panel audit renderer contract for one typed alias.

    The measurement-audit producer may preserve a Planner-selected product id.
    Its typed ``MeasurementAuditSpec`` -- not this leaf contract -- proves that
    the alias means ``measurement_missingness``.  Once that authority is
    established, the renderer and plan shaper share this exact visual
    projection instead of maintaining a second spelling table.
    """

    return (
        DeterministicFigurePanelTemplate(
            panel_id="source_availability",
            article_role="data_quality",
            chart_type="availability_panel",
            source_products=(source_product,),
        ),
    )


def robustness_figure_panels(
    source_products: Sequence[str],
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind the deterministic sensitivity forest to its exact typed parents."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    if (
        ROBUSTNESS_FIGURE_INPUT not in cleaned
        or len(cleaned) != len(set(cleaned))
        or any(not is_canonical_typed_product_token(value) for value in cleaned)
    ):
        raise ValueError(
            "robustness figure sources must be unique typed inputs and "
            "include the robustness matrix"
        )
    return (
        DeterministicFigurePanelTemplate(
            panel_id="robustness_grid",
            article_role="robustness",
            chart_type="sensitivity_forest",
            source_products=cleaned,
        ),
    )


__all__ = [
    "ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS",
    "COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS",
    "COHORT_FLOW_FIGURE_PANELS",
    "COHORT_FLOW_INPUT",
    "DATA_QUALITY_AUDIT_ROLES",
    "DATA_QUALITY_FIGURE_PANELS",
    "DeterministicFigurePanelTemplate",
    "EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_PANELS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_COUNTS_ONLY_FIGURE_PANELS",
    "EXPOSURE_OUTCOME_DISTRIBUTION_INPUT",
    "GROUPED_DESCRIPTIVE_DISTRIBUTION_FIGURE_PANELS",
    "GROUPED_DESCRIPTIVE_DISTRIBUTION_INPUT",
    "MEASUREMENT_PROCESS_AUDIT_INPUT",
    "MISSINGNESS_MEASUREMENT_AUDIT_INPUT",
    "ROBUSTNESS_COMPLETE_CASE_INPUT",
    "ROBUSTNESS_FIGURE_INPUT",
    "ROBUSTNESS_FIGURE_KNOWN_INPUTS",
    "ROBUSTNESS_PRIMARY_EFFECT_INPUT",
    "ROBUSTNESS_PRIMARY_ESTIMATE_INPUT",
    "PlannedFigurePanelSpec",
    "absolute_risk_association_composite_panels",
    "data_quality_audit_source_candidates",
    "cohort_balance_association_composite_panels",
    "measurement_availability_figure_panels",
    "LANDMARK_ASSOCIATION_COMPOSITE_INPUTS",
    "landmark_association_composite_panels",
    "robustness_figure_panels",
    "resolve_data_quality_figure_inputs",
]
