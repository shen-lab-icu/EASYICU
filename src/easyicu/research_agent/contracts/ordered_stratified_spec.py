"""Typed design contract for the deterministic ordered-stratified adapter."""

from __future__ import annotations

import math
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .association_execution import association_estimator_support
from .closed_levels import validate_closed_scalar_levels
from .dependence import PlannedDependenceRequirement
from .model_terms import ModelTermSpec, validate_model_term_roster

#: The fit kind each binary primary estimator refits with, and the ratio it
#: reports per level increment.
TREND_EFFECT_MEASURE_BY_KIND = {"logistic": "odds_ratio", "glm_poisson": "risk_ratio"}


class OrderedTrendModel(BaseModel):
    """The primary model refitted with the ordered exposure as one per-level term.

    Every coordinate is the primary model requirement's own -- outcome,
    estimator, adjustment roster and coding, dependence contract, missing-data
    policy -- except the exposure term, which is ``ordinal_linear`` over the
    primary's declared level order (one unit per level).  The adapter does not
    choose any of them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str = Field(min_length=1)
    #: The primary model whose rows, events and fit coordinates the refit
    #: must reproduce (its ``model_id`` in the primary estimates table).
    primary_requirement_id: str = Field(min_length=1)
    estimator_kind: Literal["logistic", "glm_poisson"]
    method_family: str = Field(min_length=1)
    analysis_set: str = Field(min_length=1)
    covariates: List[str]
    model_terms: List[ModelTermSpec]
    dependence: Optional[PlannedDependenceRequirement] = None
    missing_category_covariates: List[str] = Field(default_factory=list)

    @property
    def effect_measure(self) -> str:
        return TREND_EFFECT_MEASURE_BY_KIND[self.estimator_kind]

    @property
    def exposure_term(self) -> ModelTermSpec:
        return next(term for term in self.model_terms if term.role == "exposure")

    @model_validator(mode="after")
    def _one_per_level_exposure(self) -> "OrderedTrendModel":
        if self.requirement_id != f"{self.primary_requirement_id}__ordinal_trend":
            raise ValueError("the trend model is named after its primary model")
        exposures = [term for term in self.model_terms if term.role == "exposure"]
        if len(exposures) != 1:
            raise ValueError("the trend model declares exactly one exposure term")
        exposure = exposures[0]
        if exposure.coding != "ordinal_linear" or len(exposure.levels or ()) < 3:
            raise ValueError(
                "the trend model codes its exposure ordinal_linear over at least "
                "three declared levels"
            )
        validate_model_term_roster(
            terms=self.model_terms, exposure=exposure.name, covariates=self.covariates,
        )
        unknown = sorted(set(self.missing_category_covariates) - set(self.covariates))
        if unknown:
            raise ValueError(
                "missing-category covariates must be declared covariates: "
                + ", ".join(unknown)
            )
        return self


class OrderedTrendUnsupported(ValueError):
    """A primary declaration whose per-level trend refit this owner cannot run."""


def ordered_trend_exposure_term(requirement: Any) -> ModelTermSpec:
    """The primary's one exposure term over at least three ordered levels."""

    terms = [
        term
        for term in getattr(requirement, "model_terms", None) or []
        if term.role == "exposure"
        and term.name == requirement.exposure_source
        and (
            (term.coding == "ordinal_linear" and term.transform == "declared_level_index")
            or (
                term.coding == "categorical"
                and term.transform == "treatment_contrast"
                and term.reference_level is not None
            )
        )
        and len(term.levels or []) >= 3
    ]
    if len(terms) != 1:
        raise OrderedTrendUnsupported(
            "the primary model declares no single exposure term over at least "
            "three ordered levels (ordinal_linear, or treatment contrasts with a "
            "reference level)"
        )
    return terms[0]


def ordered_trend_model_for(requirement: Any) -> OrderedTrendModel:
    """The primary model requirement refitted with its exposure as one per-level term.

    The one predicate the progressive compiler and the ordered-trend owner
    both apply: a declaration it refuses is a typed plan stop at compile
    time, never a step that silently loses its adjusted trend.  Only an
    estimator the adjusted-association owner refits on a ratio scale
    (logistic or modified Poisson) with a declared adjustment set and no
    effect modifiers has a refit.
    """

    exposure = ordered_trend_exposure_term(requirement)
    support = association_estimator_support(requirement)
    if not support.supported or support.runtime_kind not in TREND_EFFECT_MEASURE_BY_KIND:
        raise OrderedTrendUnsupported(
            f"the primary estimator {getattr(requirement, 'method_family', None)!r} "
            "is neither the logistic nor the modified-Poisson model"
        )
    if requirement.covariates is None:
        raise OrderedTrendUnsupported("the primary model declares no adjustment set")
    if getattr(requirement, "effect_modifiers", None):
        raise OrderedTrendUnsupported(
            "the primary model declares effect modifiers, so one per-level term "
            "would be a different model"
        )
    per_level = ModelTermSpec(
        name=exposure.name,
        role="exposure",
        coding="ordinal_linear",
        levels=list(exposure.levels or []),
        reference_level=None,
        transform="declared_level_index",
    )
    try:
        return OrderedTrendModel(
            requirement_id=f"{requirement.requirement_id}__ordinal_trend",
            primary_requirement_id=requirement.requirement_id,
            estimator_kind=support.runtime_kind,
            method_family=requirement.method_family,
            analysis_set=requirement.analysis_set,
            covariates=list(requirement.covariates),
            model_terms=[
                per_level if term.role == "exposure" else term
                for term in requirement.model_terms or ()
            ],
            dependence=requirement.dependence,
            missing_category_covariates=list(requirement.missing_category_covariates()),
        )
    except ValueError as exc:
        raise OrderedTrendUnsupported(
            f"the per-level refit of the primary model does not validate: {exc}"
        ) from exc


class OrderedStratifiedSpec(BaseModel):
    """Planner-selected coordinates executed by the ordered-trend owner.

    The adapter does not choose variables or an event definition.  The
    progressive compiler binds them from the preceding typed primary model and
    the current step's explicit raw-input roster.  The continuous outcome is
    optional: a binary-only trend reports the adjusted per-level refit and the
    Cochran-Armitage test.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.ordered_stratified/1"] = (
        "easyicu.ordered_stratified/1"
    )
    ordered_exposure: str
    ordered_levels: List[Any] = Field(min_length=3)
    cochran_armitage_scores: List[float] = Field(min_length=3)
    score_scheme: Literal["consecutive_ordinal_ranks"] = (
        "consecutive_ordinal_ranks"
    )
    binary_outcome: str
    binary_event_value: Literal[1] = 1
    continuous_outcome: Optional[str] = None
    trend_model: OrderedTrendModel
    confidence_level: Literal[0.95] = 0.95
    stratified_product: Literal["table:ordered_stratified_outcomes"] = (
        "table:ordered_stratified_outcomes"
    )
    trend_product: str
    test_product: Literal["test:ordinal_trend"] = "test:ordinal_trend"

    @model_validator(mode="after")
    def _closed_design(self) -> "OrderedStratifiedSpec":
        names = [
            str(self.ordered_exposure or "").strip(),
            str(self.binary_outcome or "").strip(),
        ]
        if self.continuous_outcome is not None:
            names.append(str(self.continuous_outcome).strip())
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(
                "ordered exposure, binary outcome, and any continuous outcome must "
                "be distinct non-empty columns"
            )
        levels = validate_closed_scalar_levels(
            self.ordered_levels, label="ordered-stratified exposure levels"
        )
        if len(levels) < 3:
            raise ValueError("ordered-stratified analysis requires at least three levels")
        scores = [float(value) for value in self.cochran_armitage_scores]
        if len(scores) != len(levels) or any(not math.isfinite(value) for value in scores):
            raise ValueError("one finite trend score is required per ordered level")
        if scores != [float(index) for index in range(len(levels))]:
            raise ValueError(
                "consecutive_ordinal_ranks requires scores 0..K-1 in level order"
            )
        if not str(self.trend_product or "").startswith("table:"):
            raise ValueError("trend_product must be a typed table product")
        exposure = self.trend_model.exposure_term
        if exposure.name != self.ordered_exposure or len(exposure.levels or ()) != len(levels):
            raise ValueError(
                "the trend model's exposure must be the ordered exposure over the "
                "same number of declared levels"
            )
        return self


__all__ = [
    "OrderedStratifiedSpec",
    "OrderedTrendModel",
    "OrderedTrendUnsupported",
    "TREND_EFFECT_MEASURE_BY_KIND",
    "ordered_trend_exposure_term",
    "ordered_trend_model_for",
]
