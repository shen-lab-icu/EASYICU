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
    def _source_products_are_unique_typed_inputs(
        cls, values: List[str]
    ) -> List[str]:
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


__all__ = [
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
    "PlannedFigurePanelSpec",
    "data_quality_audit_source_candidates",
    "measurement_availability_figure_panels",
    "resolve_data_quality_figure_inputs",
]
