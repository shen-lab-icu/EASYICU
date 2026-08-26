from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.article_population_alignment import (
    ARTICLE_POPULATION_SCOPE_UNLABELED,
    ArticlePopulationAlignmentError,
    assess_article_population_alignment,
)


def test_population_alignment_accepts_equal_denominators() -> None:
    receipt = assess_article_population_alignment(
        table_population_n=44095,
        primary_analysis_population_n=44095,
        table_scope_explicit=False,
    )
    assert receipt.status == "aligned"


def test_population_alignment_requires_label_for_different_denominators() -> None:
    with pytest.raises(ArticlePopulationAlignmentError) as caught:
        assess_article_population_alignment(
            table_population_n=94458,
            primary_analysis_population_n=44095,
            table_scope_explicit=False,
        )
    assert caught.value.reason_code == ARTICLE_POPULATION_SCOPE_UNLABELED


def test_population_alignment_accepts_an_explicit_source_cohort_label() -> None:
    receipt = assess_article_population_alignment(
        table_population_n=94458,
        primary_analysis_population_n=44095,
        table_scope_explicit=True,
    )
    assert receipt.status == "explicitly_different"
