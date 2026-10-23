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


# pilot1 round 3 (10-22), H2: the question named its stratifier first, the
# launch projected it as the exposure and the plan modelled sex.
_STRATIFIED_FIRST = "在 eICU 中按性别分层，比较入 ICU 24 小时内最低白细胞计数与院内死亡的关系。"


def test_the_stated_exposure_decides_among_the_concepts_the_question_names() -> None:
    assert _exposure(_STRATIFIED_FIRST) is None
    assert _exposure(_STRATIFIED_FIRST, stated_exposure="入 ICU 24 小时内最低白细胞计数") == "wbc"
    # A field naming a concept the question does not name decides nothing.
    assert _exposure(_STRATIFIED_FIRST, stated_exposure="入 ICU 首日最高血肌酐") is None


def test_the_conversation_states_the_exposure_in_the_questions_words() -> None:
    from pathlib import Path

    main = (
        Path(__file__).resolve().parents[2]
        / "src" / "easyicu" / "webserver" / "pi_copilot" / "node_app" / "src" / "main.mjs"
    ).read_text(encoding="utf-8")

    assert "also pass primary_exposure and outcome in the question's own words" in main
    assert "only stratifies by or adjusts for is not its exposure" in main


def test_an_analyte_named_by_the_dictionary_decides_too() -> None:
    # 「血钾」 is named only by the dictionary ("钾"); a stratifier named first
    # is still not the exposure once the field names potassium.
    question = "在 eICU 中按性别分层，比较入 ICU 首日最高血钾与院内死亡的关系。"

    assert _exposure(question, stated_exposure="入 ICU 首日最高血钾") == "k"
    assert _exposure(_A1, stated_exposure="最高血糖", declared_covariates=("age", "sex")) is None


@pytest.mark.parametrize(
    "question",
    [
        _STRATIFIED_FIRST,
        # round 3 (10-22), H3: an adjustment named first was projected.
        "考虑年龄影响后，eICU 患者入 ICU 首日最高白细胞计数与院内死亡有没有关系？",
    ],
)
def test_a_demographic_named_before_a_measurement_is_not_projected_by_order(question: str) -> None:
    assert _exposure(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "在 eICU 中，入 ICU 后 24 小时内最高乳酸与院内死亡的关联有多强？",
        "在 eICU 中，年龄与院内死亡有关吗？",
    ],
)
def test_a_question_naming_one_concept_projects_it(question: str) -> None:
    assert _exposure(question) is not None
