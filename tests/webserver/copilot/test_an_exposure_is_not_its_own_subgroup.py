"""A study's exposure is the groups it compares, never a subgroup of itself.

MIMIC-IV walk 11-03: 「院内死亡率是多少？请按性别分组，报告各组的比例和 95%
置信区间。」 was saved with sex as the exposure and, from the same words, as a
stated subgroup.  A subgroup is excluded as the exposure, so the launch
projected none, no descriptive template fit, and the plan stopped with
``progressive_article_required_roles_missing``.  The setup save now keeps the
exposure and does not record its own levels as a subgroup; any other stated
subgroup is kept.  Synthetic catalogs only; no rows are read.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.research_launch_scientific import (
    _metadata_only_planning_coordinates,
    stated_exposure_concepts,
)
from easyicu.webserver.unread_stated_variables import UNREAD_STATED_VARIABLES_FIELD

from .stated_variable_fixtures import _update

QUESTION = "在 MIMIC-IV 演示数据中，ICU 患者的院内死亡率是多少？请按性别分组，报告各组的比例和 95% 置信区间。"
STRATIFIED = "按年龄和性别分层后，指标与结局的关系如何？"


def _current(question: str, **fields) -> dict:
    return {
        "id": "adjustment-study", "revision": 1, "question": question,
        "data_source": {"path": "/synthetic/source", "database": "miiv"}, **fields,
    }


def test_the_walk_save_keeps_sex_as_the_exposure_and_not_as_its_subgroup(monkeypatch):
    result, writes = _update(
        monkeypatch, field="stated_subgroups", question=QUESTION, concepts=("sex", "death"),
        current=_current(QUESTION, primary_exposure="性别"),
        statements=[{"variable": "性别", "words": "请按性别分组"}],
    )
    assert result["status"] == "ok"
    assert writes[0]["stated_subgroups"] == []
    # Not an unread requirement: nothing stops the plan for it.
    assert UNREAD_STATED_VARIABLES_FIELD not in writes[0]
    assert "stated_subgroups not recorded: 性别 is the study's exposure" in result["summary"]


@pytest.mark.parametrize("exposure,kept", [("性别", ["年龄"]), ("乳酸", ["年龄", "性别"])])
def test_only_the_exposures_own_levels_leave_the_subgroups(monkeypatch, exposure, kept):
    result, writes = _update(
        monkeypatch, field="stated_subgroups", question=STRATIFIED, concepts=("age", "sex"),
        current=_current(STRATIFIED, primary_exposure=exposure),
        statements=[{"variable": name, "words": "按年龄和性别分层"} for name in ("年龄", "性别")],
    )
    assert result["status"] == "ok"
    assert [item["variable"] for item in writes[0]["stated_subgroups"]] == kept


def test_a_saved_conflict_is_cleared_by_the_next_save(monkeypatch):
    saved = [{"variable": "性别", "words": "请按性别分组", "concept": "sex", "source_status": "available"}]
    _result, writes = _update(
        monkeypatch, field="stated_subgroups", question=QUESTION, concepts=("sex", "death"),
        current=_current(QUESTION, primary_exposure="性别", stated_subgroups=saved), statements=[],
    )
    assert writes[0]["stated_subgroups"] == []


def test_the_exposure_names_its_checked_concept_or_its_words():
    assert "sex" in stated_exposure_concepts("性别")
    assert "sex" in stated_exposure_concepts("gender")
    assert stated_exposure_concepts("Male patients", "sex") >= {"sex"}
    assert stated_exposure_concepts(None) == frozenset()


def test_without_the_subgroup_the_launch_projects_sex_as_the_grouping():
    coordinates = _metadata_only_planning_coordinates(
        question=QUESTION, database="miiv", stated_exposure="性别",
    )
    assert coordinates["primary_exposure"] == "sex"
    assert coordinates["target_outcome"] == "death"
