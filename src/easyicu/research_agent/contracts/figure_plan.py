"""Dependency-neutral Planner contract for manuscript figure panels.

The declared input products establish lineage, but they do not establish what
reader-facing article role a panel serves or which chart grammar it will use.
This contract keeps those semantics explicit without importing plan, figure,
or reporting owners.
"""

from __future__ import annotations

import re
from typing import Any, Collection, List, Literal, NamedTuple, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .product_identity import is_canonical_typed_product_token


class FigurePresentationSpec(BaseModel):
    """Bounded display choices; no statistical or source transformations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    layout: Literal["row", "column", "grid"] = "row"
    width_mm: float = Field(default=183, ge=80, le=500, allow_inf_nan=False)
    height_mm: float = Field(default=85, ge=50, le=500, allow_inf_nan=False)
    font_size: float = Field(default=7.5, ge=5, le=28, allow_inf_nan=False)
    font_family: Literal["sans-serif", "serif", "monospace"] = "sans-serif"
    palette: Literal["clinical", "colorblind", "grayscale"] = "clinical"
    legend_location: Literal[
        "best",
        "upper right",
        "upper left",
        "lower right",
        "lower left",
        "outside bottom",
    ] = "best"


class PlannedFigurePanelSpec(BaseModel):
    """Planner-owned article role and chart grammar for one figure panel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.planned_figure_panel/1"] = (
        "easyicu.planned_figure_panel/1"
    )
    panel_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
    figure_output: str = Field(pattern=r"^figure:[a-z][a-z0-9_]{0,79}$")
    article_role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    chart_type: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    placement: Literal["main", "supplementary"] = "main"
    source_products: List[str] = Field(min_length=1, max_length=16)
    policy_alternative_chart_types: List[str] = Field(
        default_factory=list,
        max_length=4,
        # Absent from dumps when empty so every existing plan keeps its exact
        # bytes and digest; only a sealed policy-dependent panel carries it.
        exclude_if=lambda value: not value,
        description=(
            "Chart grammars a sealed runtime policy may substitute for chart_type "
            "at execution time (for example a PH-free contrast when the "
            "proportional-hazards policy withholds the constant hazard ratio). "
            "Plan-time role coverage still reads chart_type; the end-of-execute "
            "join accepts exactly chart_type or one listed alternative."
        ),
    )
    presentation: FigurePresentationSpec | None = Field(
        default=None,
        description="Optional display parameters shared by all panels of this output. Only renderers declaring support may consume these settings; they never change data or scientific coordinates.",
    )

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

    @field_validator("policy_alternative_chart_types")
    @classmethod
    def _alternatives_are_distinct_chart_grammars(cls, values: List[str]) -> List[str]:
        return _validate_alternative_chart_types(values)

    @model_validator(mode="after")
    def _alternatives_differ_from_chart_type(self) -> "PlannedFigurePanelSpec":
        if self.chart_type in self.policy_alternative_chart_types:
            raise ValueError(
                "policy_alternative_chart_types must not repeat chart_type"
            )
        return self


_CHART_TYPE_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


def _validate_alternative_chart_types(values: Sequence[str]) -> List[str]:
    cleaned = [str(value or "").strip() for value in values]
    if any(_CHART_TYPE_TOKEN.fullmatch(value) is None for value in cleaned):
        raise ValueError(
            "policy_alternative_chart_types must be canonical chart-type tokens"
        )
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("policy_alternative_chart_types must be unique")
    return cleaned


class DeterministicFigurePanelTemplate(BaseModel):
    """Panel contract shared by a deterministic renderer and plan shaping.

    ``separable_display`` states that the renderer exports this panel on its own
    physical surface, so a sibling panel of the same product slot can stay in
    the main article while this one goes to the supplement. One exported image
    is one surface: a template that does not declare the flag cannot be moved
    away from its siblings, and the plan shaper must not promise it there.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    panel_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
    article_role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    chart_type: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    placement: Literal["main", "supplementary"] = "main"
    separable_display: bool = False
    source_products: Tuple[str, ...] = Field(min_length=1, max_length=16)
    policy_alternative_chart_types: Tuple[str, ...] = Field(
        default=(), max_length=4, exclude_if=lambda value: not value
    )

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

    @field_validator("policy_alternative_chart_types")
    @classmethod
    def _template_alternatives_are_distinct(
        cls, values: Tuple[str, ...]
    ) -> Tuple[str, ...]:
        return tuple(_validate_alternative_chart_types(values))

    @model_validator(mode="after")
    def _template_alternatives_differ_from_chart_type(
        self,
    ) -> "DeterministicFigurePanelTemplate":
        if self.chart_type in self.policy_alternative_chart_types:
            raise ValueError(
                "policy_alternative_chart_types must not repeat chart_type"
            )
        return self

    def bind(self, *, figure_output: str) -> PlannedFigurePanelSpec:
        return PlannedFigurePanelSpec(
            panel_id=self.panel_id,
            figure_output=figure_output,
            article_role=self.article_role,
            chart_type=self.chart_type,
            placement=self.placement,
            source_products=list(self.source_products),
            policy_alternative_chart_types=list(self.policy_alternative_chart_types),
        )


EXPOSURE_OUTCOME_DISTRIBUTION_INPUT = "table:exposure_outcome_distribution"
CROSS_SECTIONAL_PHENOTYPING_FIGURE_INPUTS = (
    "table:phenotype_profiles",
    "table:phenotype_assignments",
    "table:cluster_stability",
)
CROSS_SECTIONAL_PHENOTYPING_FIGURE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="a", article_role="phenotype_structure", chart_type="embedding_plot",
        source_products=("table:phenotype_assignments",),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="b", article_role="phenotype_profile", chart_type="profile_heatmap",
        source_products=("table:phenotype_profiles",),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="c", article_role="stability", chart_type="subsampling_ari",
        source_products=("table:cluster_stability",),
    ),
)
STATIC_PREDICTION_FIGURE_INPUTS = (
    "table:prediction_scores",
    "table:model_performance",
    "table:validation",
    "table:calibration",
    "table:clinical_utility",
)
# Exact main-surface panels of the host static-prediction composite renderer
# (``execution/runners/prediction_figure_executor.py``).  The composite is the
# figure a manuscript leads with, so it carries calibration (the prediction
# hero), discrimination and repeated patient-level split validation together:
# the prediction article strategy asks the main figure for at least three of
# its roles.  The decision-curve surface stays a supplementary export with no
# plan-time panel promise.
_STATIC_PREDICTION_PERFORMANCE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="a", article_role="calibration", chart_type="calibration_curve",
        source_products=("table:calibration",),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="b", article_role="model_performance", chart_type="roc_curve",
        source_products=("table:prediction_scores", "table:model_performance"),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="c", article_role="model_performance",
        chart_type="precision_recall_curve",
        source_products=("table:prediction_scores", "table:model_performance"),
    ),
)
STATIC_PREDICTION_FIGURE_PANELS = (
    *_STATIC_PREDICTION_PERFORMANCE_PANELS,
    DeterministicFigurePanelTemplate(
        panel_id="d", article_role="validation", chart_type="metric_dot_interval",
        source_products=("table:model_performance", "table:validation"),
    ),
)
#: A step may instead declare the renderer's separate repeated-split surface
#: as its own product slot (plans reviewed before the composite carried
#: validation do).  Its composite then keeps the three performance panels and
#: the surface carries the validation roles, so no panel is drawn twice.
STATIC_PREDICTION_SPLIT_SURFACE_FIGURE_PANELS = _STATIC_PREDICTION_PERFORMANCE_PANELS
#: The separate surface is a product slot of its own rather than more panels
#: on the composite, because one exported image is one surface and the
#: end-of-execute join resolves a runtime contract per declared figure output.
STATIC_PREDICTION_VALIDATION_FIGURE_SUFFIX = "_validation_stability"
STATIC_PREDICTION_VALIDATION_FIGURE_PANELS = (
    DeterministicFigurePanelTemplate(
        panel_id="a", article_role="validation_design",
        chart_type="cohort_split_diagram",
        source_products=("table:model_performance", "table:validation"),
    ),
    DeterministicFigurePanelTemplate(
        panel_id="b", article_role="validation",
        chart_type="metric_dot_interval",
        source_products=("table:model_performance", "table:validation"),
    ),
)
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
BALANCE_ASSOCIATION_COMPOSITE_INPUTS = (
    "table:balance_positivity_context",
    "table:adjusted_association_estimates",
    "table:robustness_matrix",
    "table:robustness_summary",
)
ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS = (
    "table:absolute_risk_context",
    "table:adjusted_association_estimates",
    "table:robustness_matrix",
    "table:robustness_summary",
)
ASSOCIATION_SUMMARY_COMPOSITE_INPUTS = (
    "table:exposure_outcome_distribution",
    "table:adjusted_association_estimates",
    "table:robustness_summary",
    "table:measurement_missingness",
)


def association_summary_composite_panels(
    source_products: Sequence[str],
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind the standard association summary and routine quality panel."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    if set(cleaned) != set(ASSOCIATION_SUMMARY_COMPOSITE_INPUTS) or len(cleaned) != 4:
        raise ValueError("association summary composite requires its four exact tables")
    return (
        DeterministicFigurePanelTemplate(
            panel_id="A",
            article_role="descriptive_result",
            chart_type="event_rate_panel",
            source_products=("table:exposure_outcome_distribution",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="B",
            article_role="primary_estimand",
            chart_type="forest",
            source_products=("table:adjusted_association_estimates",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="C",
            article_role="robustness",
            chart_type="sensitivity_coverage_matrix",
            source_products=("table:robustness_summary",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="D",
            article_role="data_quality",
            chart_type="availability_panel",
            source_products=("table:measurement_missingness",),
        ),
    )


def balance_association_composite_panels(
    source_products: Sequence[str],
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind balance, primary association, and two robustness summaries."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    if cleaned != BALANCE_ASSOCIATION_COMPOSITE_INPUTS:
        raise ValueError("balance association composite requires its four exact tables")
    return (
        DeterministicFigurePanelTemplate(
            panel_id="baseline_balance",
            article_role="descriptive_result",
            chart_type="standardized_difference",
            source_products=("table:balance_positivity_context",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="primary_adjusted_association",
            article_role="primary_estimand",
            chart_type="forest",
            source_products=("table:adjusted_association_estimates",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="robustness_specification_status",
            article_role="robustness",
            chart_type="sensitivity_specification_status",
            source_products=("table:robustness_matrix",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="robustness_coverage",
            article_role="robustness",
            chart_type="sensitivity_coverage_matrix",
            source_products=("table:robustness_summary",),
        ),
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
            panel_id="robustness_specification_status",
            article_role="robustness",
            chart_type="sensitivity_specification_status",
            source_products=("table:robustness_matrix",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="robustness_coverage",
            article_role="robustness",
            chart_type="sensitivity_coverage_matrix",
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
            panel_id="robustness_specification_status",
            article_role="robustness",
            chart_type="sensitivity_specification_status",
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


#: Product ids that name the measurement-process audit without a typed
#: declaration.  A ``MeasurementAuditSpec`` may give the same table any id.
CANONICAL_MEASUREMENT_PROCESS_PRODUCT_IDS = frozenset(
    {"measurement_process", "measurement_process_audit"}
)


def _measurement_process_products(
    source_products: Sequence[str],
    typed_products: Collection[str] = (),
) -> list[str]:
    typed = {str(value or "").strip() for value in typed_products}
    return [
        value
        for value in source_products
        if value.startswith("table:")
        and (
            value in typed
            or value.partition(":")[2] in CANONICAL_MEASUREMENT_PROCESS_PRODUCT_IDS
        )
    ]


def _landmark_curve_product(
    source_products: Sequence[str],
    typed_measurement_products: Collection[str] = (),
) -> str | None:
    sensitivity = _landmark_sensitivity_contrast_product(source_products)
    reserved = {
        "table:robustness_summary",
        sensitivity,
        *_measurement_process_products(
            source_products, typed_measurement_products
        ),
    }
    adjusted_risk = _landmark_adjusted_risk_product(source_products)
    matches = [
        value
        for value in source_products
        if value.startswith("table:")
        and value not in reserved
        and value != adjusted_risk
        and not (
            (
                "robustness" in value.partition(":")[2]
                or "sensitivity" in value.partition(":")[2]
            )
            and value.partition(":")[2].endswith("_exposure_curve")
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _landmark_adjusted_risk_product(
    source_products: Sequence[str],
) -> str | None:
    """Return the unique model-standardised absolute-risk curve product.

    The landmark runtime publishes this curve on the same exposure grid as the
    ratio-scale primary result.  It is distinct from the generic
    ``absolute_risk_context`` table, whose continuous-exposure rows describe
    measurement availability rather than the scientific dose-response claim.
    """

    accepted_tokens = (
        "adjusted_absolute_risk",
        "standardized_absolute_risk",
        "standardised_absolute_risk",
        "absolute_risk_curve",
    )
    matches = [
        value
        for value in source_products
        if value.startswith("table:")
        and any(token in value.partition(":")[2] for token in accepted_tokens)
    ]
    return matches[0] if len(matches) == 1 else None


def _measurement_process_product(
    source_products: Sequence[str],
    typed_products: Collection[str] = (),
) -> str | None:
    matches = _measurement_process_products(source_products, typed_products)
    return matches[0] if len(matches) == 1 else None


def _landmark_sensitivity_contrast_product(
    source_products: Sequence[str],
) -> str | None:
    matches = [
        value
        for value in source_products
        if value.startswith("table:")
        and value.partition(":")[2].endswith("_exposure_contrasts")
        and (
            "robustness" in value.partition(":")[2]
            or "sensitivity" in value.partition(":")[2]
        )
    ]
    return matches[0] if len(matches) == 1 else None


class LandmarkCompositeRoles(NamedTuple):
    """Which typed input fills each role of the landmark association display."""

    curve: str | None
    adjusted_risk: str | None
    sensitivity: str | None
    measurement: str | None


def landmark_association_composite_roles(
    source_products: Sequence[str],
    *,
    measurement_process_products: Collection[str] = (),
) -> LandmarkCompositeRoles:
    """Resolve the display roles of a landmark association profile.

    ``measurement_process_products`` are the tables a plan's typed measurement
    audits declare as the process view (see
    :func:`typed_measurement_process_products`); a canonical spelling keeps
    its meaning without one.  The renderer, the runtime that binds it, and the
    plan shaper all read roles from here.
    """

    cleaned = tuple(str(value or "").strip() for value in source_products)
    return LandmarkCompositeRoles(
        curve=_landmark_curve_product(cleaned, measurement_process_products),
        adjusted_risk=_landmark_adjusted_risk_product(cleaned),
        sensitivity=_landmark_sensitivity_contrast_product(cleaned),
        measurement=_measurement_process_product(
            cleaned, measurement_process_products
        ),
    )


def landmark_association_composite_panels(
    source_products: Sequence[str],
    *,
    measurement_process_products: Collection[str] = (),
) -> Tuple[DeterministicFigurePanelTemplate, ...]:
    """Bind a claim-led landmark-association display to typed parents."""

    cleaned = tuple(str(value or "").strip() for value in source_products)
    curve, adjusted_risk, sensitivity, measurement = (
        landmark_association_composite_roles(
            cleaned, measurement_process_products=measurement_process_products
        )
    )
    if (
        curve is None
        or adjusted_risk is None
        or len(cleaned) not in {2, 3, 4, 5}
        or len(cleaned) != len(set(cleaned))
        or (
            len(cleaned) == 3
            and (
                sensitivity is None
                or measurement is not None
                or "table:robustness_summary" in cleaned
            )
        )
        or (
            len(cleaned) in {4, 5}
            and (
                measurement is None
                or not LANDMARK_ASSOCIATION_COMPOSITE_INPUTS <= set(cleaned)
            )
        )
        or (len(cleaned) == 4 and sensitivity is not None)
        or (len(cleaned) == 5 and sensitivity is None)
    ):
        raise ValueError(
            "landmark composite requires two curves, an optional comparable "
            "sensitivity table, or the complete audit profile"
        )
    panels = (
        DeterministicFigurePanelTemplate(
            panel_id="association_curve",
            article_role="primary_estimand",
            chart_type="marginal_effect_panel",
            source_products=(curve,),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="absolute_risk_curve",
            article_role="descriptive_result",
            chart_type="absolute_risk_curve",
            source_products=(adjusted_risk,),
        ),
    )
    if sensitivity is not None:
        panels = (
            *panels,
            DeterministicFigurePanelTemplate(
                panel_id="sensitivity_contrasts",
                article_role="robustness",
                chart_type="sensitivity_forest",
                source_products=(sensitivity,),
            ),
        )
    if len(cleaned) in {2, 3}:
        return panels
    assert measurement is not None
    return (
        *panels,
        DeterministicFigurePanelTemplate(
            panel_id="robustness_summary",
            article_role="robustness",
            chart_type="sensitivity_coverage_matrix",
            placement="supplementary",
            separable_display=True,
            source_products=("table:robustness_summary",),
        ),
        DeterministicFigurePanelTemplate(
            panel_id="measurement_process",
            article_role="data_quality",
            chart_type="availability_panel",
            placement="supplementary",
            separable_display=True,
            source_products=(measurement,),
        ),
    )


def separable_display_panel_ids(
    *,
    source_products: Sequence[str],
    panel_ids: Sequence[str],
    measurement_process_products: Collection[str] = (),
) -> frozenset[str]:
    """Which of a step's planned panels its bound renderer can export apart.

    The shaper asks this against the shared contract of the renderer that the
    step's exact typed inputs select, so a placement split is honored only when
    some artifact can carry it. The panel ids must be the whole contract: a
    hand-written subset no longer proves which renderer was chosen, and the
    conservative answer to an unproven group is that nothing is separable.
    """

    wanted = {str(value or "").strip() for value in panel_ids}
    if not wanted:
        return frozenset()
    try:
        templates = landmark_association_composite_panels(
            source_products,
            measurement_process_products=measurement_process_products,
        )
    except ValueError:
        return frozenset()
    declared = {str(template.panel_id) for template in templates}
    if declared != wanted:
        return frozenset()
    return frozenset(
        str(template.panel_id)
        for template in templates
        if template.separable_display
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


def typed_measurement_process_products(steps: Sequence[Any]) -> frozenset[str]:
    """Tables a plan's typed measurement audits declare as the process view.

    The producing step's ``MeasurementAuditSpec`` says what its table means,
    whatever id the Planner or a family template chose, so a renderer never
    has to recognize the table by its spelling.
    """

    return frozenset(
        output
        for output, _step_id in data_quality_audit_source_candidates(steps)[
            "measurement_process"
        ]
    )


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
    """Bind a specification table without presuming effect comparability."""

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
            chart_type="specification_grid",
            source_products=cleaned,
        ),
    )


__all__ = [
    "CROSS_SECTIONAL_PHENOTYPING_FIGURE_INPUTS",
    "CROSS_SECTIONAL_PHENOTYPING_FIGURE_PANELS",
    "ASSOCIATION_SUMMARY_COMPOSITE_INPUTS",
    "association_summary_composite_panels",
    "ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS",
    "BALANCE_ASSOCIATION_COMPOSITE_INPUTS",
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
    "STATIC_PREDICTION_FIGURE_INPUTS",
    "STATIC_PREDICTION_FIGURE_PANELS",
    "STATIC_PREDICTION_SPLIT_SURFACE_FIGURE_PANELS",
    "STATIC_PREDICTION_VALIDATION_FIGURE_PANELS",
    "STATIC_PREDICTION_VALIDATION_FIGURE_SUFFIX",
    "PlannedFigurePanelSpec",
    "absolute_risk_association_composite_panels",
    "balance_association_composite_panels",
    "data_quality_audit_source_candidates",
    "cohort_balance_association_composite_panels",
    "measurement_availability_figure_panels",
    "CANONICAL_MEASUREMENT_PROCESS_PRODUCT_IDS",
    "LANDMARK_ASSOCIATION_COMPOSITE_INPUTS",
    "LandmarkCompositeRoles",
    "landmark_association_composite_panels",
    "landmark_association_composite_roles",
    "separable_display_panel_ids",
    "robustness_figure_panels",
    "resolve_data_quality_figure_inputs",
    "typed_measurement_process_products",
]
