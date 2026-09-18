"""Ordinary research wording reaches a family without technical answer prompts."""

import pytest

from easyicu.research_agent.agents.progressive_planner import candidate_analysis_types
from easyicu.research_agent.planning.analysis_types import (
    infer_analysis_type,
    strong_trajectory_clustering_framing,
)
from easyicu.research_agent.planning.study_design import infer_study_design_family
from easyicu.research_agent.schema import UserPreferences
from tests.research_agent.planning.progressive_planner_fixtures import _context


@pytest.mark.parametrize(
    "question",
    [
        "在成人 ICU 患者中，入科后前24小时最高乳酸与院内死亡之间的关系是什么？",
        "请研究血小板计数和住院时长的关系，并描述人群特征。",
        "体温与院内死亡有关系吗？",
        "请评估呼吸频率和死亡之间有何关系。",
    ],
)
def test_plain_relationship_question_is_not_replaced_by_descriptive_context(question):
    context = _context().model_copy(update={
        "research_question": question,
        "user_preferences": UserPreferences(
            extra_notes="描述数据覆盖、研究人群和基线特征。",
            data_constraints='{"cohort": {"age_min": 18}}',
        ),
    })
    before = context.model_dump(mode="json")
    assert infer_analysis_type(context).key == "association_study"
    assert candidate_analysis_types(context)[0] == "association_study"
    assert infer_study_design_family(context) == "association"
    assert context.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "question",
    [
        "根据生命体征和化验指标识别不同临床亚型，并比较各亚型的死亡情况。",
        "根据多变量指标发现不同的患者表型。",
        "根据观测特征把患者划分为不同亚型，并描述各组特征。",
        "识别不稳定的器官功能轨迹，并描述不同群体。",
    ],
)
def test_different_or_unstable_is_not_a_discovery_negation(question):
    context = _context().model_copy(update={"research_question": question})
    assert strong_trajectory_clustering_framing(question)
    assert infer_analysis_type(context).key == "trajectory_clustering"
    assert candidate_analysis_types(context)[0] == "trajectory_clustering"
    assert infer_study_design_family(context) == "phenotyping"


@pytest.mark.parametrize(
    "question",
    [
        "不进行患者表型分群，只描述各项指标的分布。",
        "不要重新识别临床亚型，只描述已有分组。",
        "无需发现新亚型，只描述既有临床表型。",
        "不做患者表型发现，描述现有分组特征。",
        "比较既有亚型与死亡的关联，使用患者层面聚类稳健标准误。",
        "描述已有亚型的分布，不重新划分患者群。",
    ],
)
def test_declined_discovery_and_existing_groups_do_not_authorize_clustering(question):
    assert not strong_trajectory_clustering_framing(question)


@pytest.mark.parametrize(
    "question",
    [
        "仅描述暴露人数和死亡人数，不分析二者的关系。",
        "描述临床特征，不研究指标与死亡的关系。",
        "描述患病率；关联研究与本次任务无关。",
    ],
)
def test_declined_relationship_does_not_displace_a_descriptive_question(question):
    context = _context().model_copy(update={"research_question": question})
    assert infer_analysis_type(context).key == "descriptive_epidemiology"


def test_relationship_in_a_method_note_does_not_replace_the_primary_question():
    context = _context().model_copy(update={
        "research_question": "描述患病率和死亡人数。",
        "user_preferences": UserPreferences(
            extra_notes="其他研究曾探讨暴露与死亡的关系。",
        ),
    })
    assert infer_analysis_type(context).key == "descriptive_epidemiology"
