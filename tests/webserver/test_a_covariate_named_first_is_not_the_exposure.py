"""A covariate a question names first is not projected as its exposure.

pilot1 round 2 (10-22), A1: 「以年龄、性别为协变量，分析 eICU 中入 ICU 后 24
小时内最高乳酸与院内死亡的关联。」  The study declared age and sex as exact
covariates and its exposure field read 「入 ICU 后 24 小时内最高乳酸」, but
the planning launch projected the keyword reader's exposure -- the first
concept the question mentions, age -- into the run's context.  The plan's
model could not both adjust for age and estimate it, every repair met the
same adjustment authority, and planning stopped.

The launch's coordinates now exclude the study's declared covariates and
outcome from the exposure, and read the study's exposure field through the
same reader.  When the field and the question name different concepts,
neither is projected and the planner reads the question.  A question whose
exposure comes first is unchanged.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.research_launch_scientific import (
    _metadata_only_planning_coordinates,
)

_A1 = "以年龄、性别为协变量，分析 eICU 中入 ICU 后 24 小时内最高乳酸与院内死亡的关联。"
_ADJUSTED_FIRST = "考虑年龄和性别的影响后，eICU 患者入 ICU 首日最高血肌酐与院内死亡有关吗？"
_EXPOSURE_FIRST = "在 eICU 中，入 ICU 后 24 小时内最高乳酸与院内死亡的关联有多强？请调整年龄和性别。"


def _exposure(question: str, **study: object) -> object:
    return _metadata_only_planning_coordinates(
        question=question, database="eicu", **study,
    ).get("primary_exposure")


def test_the_stated_exposure_is_projected_not_the_covariate_named_first() -> None:
    assert _exposure(
        _A1,
        stated_exposure="入 ICU 后 24 小时内最高乳酸",
        declared_covariates=("age", "sex"),
    ) == "lact"
    assert _exposure(
        _ADJUSTED_FIRST,
        stated_exposure="入 ICU 首日最高血肌酐",
        declared_covariates=("age", "sex"),
    ) == "crea"


def test_a_declared_covariate_is_never_projected_as_the_exposure() -> None:
    # The study's exposure field names nothing the reader knows: the planner
    # reads the question rather than receive a covariate as its exposure.
    assert _exposure(_A1, declared_covariates=("age", "sex")) is None
    assert _exposure(_A1, stated_exposure="最高血糖", declared_covariates=("age", "sex")) is None


def test_readings_that_disagree_project_no_exposure() -> None:
    assert _exposure(
        _EXPOSURE_FIRST,
        stated_exposure="入 ICU 首日最高血肌酐",
        declared_covariates=("age", "sex"),
    ) is None


@pytest.mark.parametrize(
    "study",
    [
        {},
        {"declared_covariates": ("age", "sex")},
        {"stated_exposure": "入 ICU 后 24 小时内最高乳酸", "declared_covariates": ("age", "sex")},
        {"stated_exposure": "lact"},
    ],
)
def test_a_question_whose_exposure_comes_first_is_unchanged(study: dict) -> None:
    assert _exposure(_EXPOSURE_FIRST, **study) == "lact"
