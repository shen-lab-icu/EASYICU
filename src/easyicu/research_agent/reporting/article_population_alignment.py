"""Case-neutral article contract for Table 1 versus primary-analysis scope."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ARTICLE_POPULATION_SCOPE_UNLABELED = "ARTICLE_POPULATION_SCOPE_UNLABELED"


class ArticlePopulationAlignmentError(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.validator = "article_population_alignment"


class ArticlePopulationAlignmentReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.article_population_alignment/1"] = (
        "easyicu.article_population_alignment/1"
    )
    table_population_n: int | None = Field(default=None, ge=0)
    primary_analysis_population_n: int | None = Field(default=None, ge=0)
    status: Literal["aligned", "explicitly_different", "unresolved"]
    reason_code: str


def assess_article_population_alignment(
    *,
    table_population_n: int | None,
    primary_analysis_population_n: int | None,
    table_scope_explicit: bool,
) -> ArticlePopulationAlignmentReceipt:
    """Require a visible scope label whenever Table 1 and the model differ."""

    if table_population_n is None or primary_analysis_population_n is None:
        return ArticlePopulationAlignmentReceipt(
            table_population_n=table_population_n,
            primary_analysis_population_n=primary_analysis_population_n,
            status="unresolved",
            reason_code="ARTICLE_POPULATION_DENOMINATOR_UNAVAILABLE",
        )
    if table_population_n == primary_analysis_population_n:
        return ArticlePopulationAlignmentReceipt(
            table_population_n=table_population_n,
            primary_analysis_population_n=primary_analysis_population_n,
            status="aligned",
            reason_code="ARTICLE_POPULATION_DENOMINATOR_ALIGNED",
        )
    if not table_scope_explicit:
        raise ArticlePopulationAlignmentError(
            ARTICLE_POPULATION_SCOPE_UNLABELED,
            "Table 1 and the primary analysis use different populations, but the table scope is not explicit.",
        )
    return ArticlePopulationAlignmentReceipt(
        table_population_n=table_population_n,
        primary_analysis_population_n=primary_analysis_population_n,
        status="explicitly_different",
        reason_code="ARTICLE_POPULATION_DIFFERENCE_EXPLICITLY_LABELED",
    )


__all__ = [
    "ARTICLE_POPULATION_SCOPE_UNLABELED",
    "ArticlePopulationAlignmentError",
    "ArticlePopulationAlignmentReceipt",
    "assess_article_population_alignment",
]
