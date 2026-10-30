"""Typed article-to-design bindings emitted by the scientific Planner."""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


LiteratureDesignElement = Literal[
    "population",
    "time_zero",
    "exposure",
    "outcome",
    "estimand",
    "adjustment",
    "dependence",
    "missing_data",
    "robustness",
    "reporting",
]


class LiteratureDesignBinding(BaseModel):
    """Explain which exact design decisions one sealed source supports."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$",
    )
    design_elements: list[LiteratureDesignElement] = Field(min_length=1)
    application: str = Field(min_length=8, max_length=1200)
    divergence: str | None = Field(default=None, max_length=1200)

    @field_validator("design_elements")
    @classmethod
    def _unique_design_elements(
        cls, values: list[LiteratureDesignElement]
    ) -> list[LiteratureDesignElement]:
        if len(values) != len(set(values)):
            raise ValueError("literature design_elements must be unique")
        return values


__all__ = ["LiteratureDesignBinding", "LiteratureDesignElement"]


ReviewRole = Literal[
    "topic_gap", "analysis_method", "definition", "adjustment", "data_source", "reporting_guideline",
]
REVIEW_ROLES: tuple[ReviewRole, ...] = (
    "topic_gap", "analysis_method", "definition", "adjustment", "data_source", "reporting_guideline",
)


class ReviewPolicy(BaseModel):
    """Targets are search effort and coverage goals, never inclusion overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    screening_target: int = Field(default=75, ge=1, le=400)
    candidates_per_source: int = Field(default=100, ge=1, le=200)
    candidates_per_query: int = Field(default=25, ge=1, le=100)
    recency_years: int = Field(default=10, ge=1, le=50)
    role_targets: Dict[ReviewRole, Annotated[int, Field(ge=0, le=100)]] = Field(default_factory=lambda: {
        "topic_gap": 8, "analysis_method": 5, "definition": 3,
        "adjustment": 3, "data_source": 3, "reporting_guideline": 1,
    })


class ImpactProxies(BaseModel):
    """Transparent proxies; none is an official journal impact factor."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: Literal["openalex", "curated_venue_list", "unavailable"] = "unavailable"
    citation_count: int | None = Field(default=None, ge=0)
    citations_per_year: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    venue_two_year_mean_citedness: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    venue_h_index: int | None = Field(default=None, ge=0)
    curated_top_venue: bool = False
    proxies_used: List[Literal[
        "citations_per_year", "venue_two_year_mean_citedness", "venue_h_index", "curated_top_venue",
    ]] = Field(default_factory=list)


class SourceResourceIdentity(BaseModel):
    """Match to an explicitly curated official resource identity, not its content."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    database: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: Literal["source_publication", "dataset_release"]
    pmid: str | None = None
    doi: str
    dataset_version: str | None = None
    registry_reference: str
    registry_version: Literal["easyicu.official_source_identity/1", "easyicu.official_source_identity/2"] = "easyicu.official_source_identity/2"
    content_status: Literal["not_verified"] = "not_verified"
    registered_dataset_version_status: Literal["not_assessed"] = "not_assessed"


class RecordReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    roles: List[ReviewRole] = Field(default_factory=list)
    recency: Literal["recent", "outside_window", "method_classic", "unknown", "future"]
    age_years: int | None = Field(default=None, ge=0)
    recency_years: int = Field(ge=1)
    assessed_year: int
    source_resource_identity: SourceResourceIdentity | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )


class ReviewShortfall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["screening", "topic_gap", "analysis_method", "definition", "adjustment", "data_source", "reporting_guideline"]
    target: int = Field(ge=0)
    actual: int = Field(ge=0)
    missing: int = Field(ge=1)


class ReviewCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role_counts: Dict[ReviewRole, int]
    shortfalls: List[ReviewShortfall]
    targets_met: bool
    screened_count: int = Field(ge=0)
    assessment_year: int




def analysis_terms(question: str) -> tuple[str, ...]:
    text = question.casefold()
    if any(term in text for term in ("predict", "prognos", "validation")):
        return ("prediction", "validation", "calibration", "discrimination")
    if any(term in text for term in ("survival", "hazard", "time-to-event")):
        return ("survival", "Cox", "censoring", "competing risks")
    if any(term in text for term in ("trajectory", "longitudinal")):
        return ("longitudinal", "trajectory", "mixed effects")
    if any(term in text for term in ("cluster", "phenotyp")):
        return ("clustering", "phenotyping", "latent class")
    return ("regression", "propensity", "confounding", "sensitivity analysis")
