"""A study family v1 does not offer on its database is refused before planning.

Product scope U1 (2026-10-09): on eICU-CRD the survival, causal and
subtyping/trajectory cells are out of scope and must be refused with the
reason.  The causal cell stops at its trial owner; the other two now stop at
the launch, before any model is called, on the family the plan would use: the
study's recorded design, else the planner's own routing of the question.
pilot1's fixed set (10-22) planned 「把脓毒症患者聚类成不同亚型」 on eICU as a
description, and a Cox question was recorded as an association study, so
neither was refused for its scope.  MIMIC-IV offers both families, and eICU's
descriptive, association and prediction questions are untouched.
"""

from __future__ import annotations

import pytest

from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.research_pipeline_run_preparation import (
    ResearchPipelineLaunchRequest,
    prepare_research_pipeline_run,
)
from easyicu.webserver.study_family_scope import (
    STUDY_FAMILY_DATABASE_OUT_OF_SCOPE,
    study_family_database_gap,
)

_SUBTYPES = "在 eICU 中，用入 ICU 首日的化验和生命体征把脓毒症患者聚类成不同亚型。"
_COX = "在 eICU 中，用 Cox 回归分析入 ICU 首日最高乳酸与 28 天生存的关系。"
_IN_SCOPE = [
    "在 eICU 全部 ICU 入住中，男性和女性的院内死亡率分别是多少？",
    "以年龄、性别为协变量，分析 eICU 中入 ICU 后 24 小时内最高乳酸与院内死亡的关联。",
    "eICU 中，入 ICU 首日最高乳酸与 ICU 住院时长的关系如何？",
    "能不能用 eICU 患者入 ICU 首日的生命体征预测院内死亡？",
]


def _launch(tmp_path, question: str, database: str = "eicu"):
    request = ResearchPipelineLaunchRequest(
        export_path=str(tmp_path / "export"),
        study_context={
            "id": "study-scope",
            "revision": 3,
            "question": question,
            "data_source": {"path": str(tmp_path / "export"), "database": database},
            "analysis_design": {},
        },
        project_root=str(tmp_path / "workspace"),
        provider={"provider": "openai"},
        provider_environment={"OPENAI_API_KEY": "test"},
        credential_source="pi_verified",
        literature_search_authorized=False,
        plan_revision_source_run_id="",
        execution_resume_source_run_id="",
        development_resume_source_job_id="",
        budget_mode="planner_canary",
        runner_image=None,
    )
    try:
        prepare_research_pipeline_run(request)
    except ResearchPipelineRunError as stopped:
        return stopped
    return None


@pytest.mark.parametrize(
    "question, family", [(_COX, "survival")],
)
def test_an_eicu_launch_stops_before_any_model_is_called(tmp_path, question, family):
    stopped = _launch(tmp_path, question)

    assert stopped.code == STUDY_FAMILY_DATABASE_OUT_OF_SCOPE
    gap = stopped.details["study_family_gap"]
    assert (gap["code"], gap["family"], gap["database"]) == (STUDY_FAMILY_DATABASE_OUT_OF_SCOPE, family, "eicu")
    assert "event_time" in {item["capability"] for item in gap["missing_capabilities"]}


def test_eicu_trajectory_launch_is_no_longer_blocked_by_product_scope(tmp_path):
    assert _launch(tmp_path, _SUBTYPES) is None


@pytest.mark.parametrize("question", [_SUBTYPES, _COX])
def test_mimic_iv_offers_both_families(question):
    assert study_family_database_gap({}, question=question, database="miiv") is None


@pytest.mark.parametrize("question", _IN_SCOPE)
def test_eicu_descriptions_associations_and_predictions_go_on(question):
    assert study_family_database_gap({}, question=question, database="eicu") is None


def test_the_recorded_design_is_the_family_checked():
    association = {"analysis_design": {"analysis_family": "association_study"}}
    survival = {"analysis_design": {"analysis_family": "survival"}}

    # The recorded family remains authoritative; an unavailable 28-day endpoint
    # is a source gap even for an association, so use the available status here.
    assert study_family_database_gap(association, question="Estimate the association with hospital mortality.", database="eicu") is None
    assert study_family_database_gap(
        survival, question="入 ICU 首日最高乳酸与院内死亡有关吗？", database="eicu",
    )["family"] == "survival"


def test_a_demo_source_has_the_scope_of_the_database_it_samples():
    # The public demos are separate sources, not separate v1 scopes.
    survival = {"analysis_design": {"analysis_family": "survival"}}
    gap = study_family_database_gap(survival, question=_COX, database="eicu_demo")

    assert gap["code"] == STUDY_FAMILY_DATABASE_OUT_OF_SCOPE
    assert gap["database"] == "eicu_demo"
    for question in _IN_SCOPE:
        assert study_family_database_gap({}, question=question, database="eicu_demo") is None
