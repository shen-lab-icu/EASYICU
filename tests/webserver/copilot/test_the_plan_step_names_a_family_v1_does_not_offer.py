"""The plan step names a study family v1 does not offer on the bound database.

pilot1 round 2 (10-22): an eICU Cox question was confirmed on eICU, the plan
card offered 「开始生成研究计划」, and only that click met the launch's refusal
(``study_family_scope``), as a banner beside a card that still offered the
plan.  The workflow now asks the owner the launch asks, on the study's own
question and bound source: the plan step is blocked with the gap's code and
the snapshot carries the gap, so the card states the refusal instead of
offering a plan.  MIMIC-IV, an eICU question of a family v1 offers there, and
a study with no identified source are offered their plan as before.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.study_family_scope import (
    STUDY_FAMILY_DATABASE_OUT_OF_SCOPE,
    study_family_database_gap,
)

_COX = "在 eICU 中，用 Cox 回归分析入 ICU 首日最高乳酸与 28 天生存的关系。"
_SUBTYPES = "在 eICU 中，用入 ICU 首日的化验和生命体征把脓毒症患者聚类成不同亚型。"
_ASSOCIATION = "以年龄、性别为协变量，分析 eICU 中入 ICU 后 24 小时内最高乳酸与院内死亡的关联。"


def _study(question: str, database: str) -> dict[str, Any]:
    return {
        "id": "study-family-scope",
        "revision": 2,
        "question": question,
        "data_source": {"path": "/prepared/export", "database": database},
    }


def _snapshot(study: dict[str, Any]):
    return build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None, latest_run=None,
    )


def _plan_stage(snapshot: Any) -> Any:
    return next(stage for stage in snapshot.stages if stage.id == "plan")


@pytest.mark.parametrize(
    "question, family", [(_COX, "survival"), (_SUBTYPES, "trajectory_clustering")],
)
def test_an_eicu_study_of_a_family_v1_does_not_offer_there_is_not_offered_a_plan(
    question: str, family: str
) -> None:
    study = _study(question, "eicu")
    snapshot = _snapshot(study)

    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == STUDY_FAMILY_DATABASE_OUT_OF_SCOPE
    assert _plan_stage(snapshot).status == "blocked"
    assert _plan_stage(snapshot).reason_code == STUDY_FAMILY_DATABASE_OUT_OF_SCOPE
    # The gap the launch would raise, codes only.
    assert snapshot.study_family_gap == study_family_database_gap(
        study, question=question, database="eicu"
    ) == {
        "code": STUDY_FAMILY_DATABASE_OUT_OF_SCOPE,
        "family": family,
        "database": "eicu",
        "supported_databases": ["miiv"],
    }


@pytest.mark.parametrize(
    "question, database",
    [(_COX, "miiv"), (_SUBTYPES, "miiv"), (_ASSOCIATION, "eicu")],
)
def test_a_family_offered_on_its_database_is_offered_its_plan(
    question: str, database: str
) -> None:
    snapshot = _snapshot(_study(question, database))

    assert snapshot.next_action_code == "provider_ready_to_generate_plan"
    assert _plan_stage(snapshot).status == "ready"
    assert snapshot.study_family_gap is None


def test_the_recorded_design_is_the_family_the_plan_step_checks() -> None:
    study = {**_study(_COX, "eicu"), "analysis_design": {
        "analysis_family": "association_study",
        "analysis_unit": "icu_stay",
        "variance_estimator": "model_based",
    }}

    assert _snapshot(study).next_action_code == "provider_ready_to_generate_plan"


def test_a_study_with_no_identified_source_is_not_judged() -> None:
    study = {**_study(_COX, "eicu"), "data_source": {"path": "/prepared/export"}}

    assert _snapshot(study).study_family_gap is None
