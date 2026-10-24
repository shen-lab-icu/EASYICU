"""Explicit exclusions do not become the study's requested analysis family."""

import pytest

from easyicu.research_agent.planning.analysis_types import (
    infer_analysis_type_with_basis,
    mask_declined_analysis_phrases,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
from easyicu.webserver.question_family_check import check_question_family, host_question_family
from easyicu.webserver.study_family_reading import study_family_reading

TRIAL_QUESTION = (
    "做描述性分析：住院患者的低钠血症比例，按性别报告人数和百分比。"
    "不做统计推断、置信区间、显著性检验、回归调整、预测或因果分析。"
)



def planner(question):
    spec, basis = infer_analysis_type_with_basis(ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="synthetic", database="eicu", n_stays=0),
        variables=[],
    ))
    return spec.key, basis


def test_declined_methods_do_not_fabricate_a_prediction_disagreement():
    assert planner(TRIAL_QUESTION) == ("descriptive_epidemiology", "keyword")
    receipt = check_question_family(
        {"family": "descriptive_epidemiology", "words": "描述性分析"}, TRIAL_QUESTION,
    )
    assert receipt["host_family"] in (None, "descriptive_epidemiology")
    assert receipt["status"] == "model_only"


@pytest.mark.parametrize("question", [
    "低钠血症的比例是多少？不做统计推断、置信区间、显著性检验、回归调整、预测或因果分析。",
    "低钠血症的比例是多少？不做回归调整、预测和因果分析。",
    "What proportion died? No adjustment, regression, prediction or causal inference.",
    "What proportion died? Do not build a predictive model or perform causal inference.",
    "What proportion died, without regression, prediction, or causal inference?",
    "What proportion died? Avoid prediction and do not use propensity weighting.",
    "不做预测，但低钠血症的比例是多少？",
    "Do not predict mortality, but what proportion died?",
])
def test_declined_methods_remain_a_description_with_original_word_evidence(question):
    assert planner(question) == ("descriptive_epidemiology", "keyword")
    reading = study_family_reading(question)
    assert reading is not None and reading.family == "descriptive_epidemiology"
    assert all(question[e.start:e.end] == e.evidence for e in reading.elements)
    masked = mask_declined_analysis_phrases(question)
    assert len(masked) == len(question)


@pytest.mark.parametrize("question,family", [
    ("建立预测院内死亡的模型。", "prediction_model"),
    ("Build a model to predict in-hospital mortality.", "prediction_model"),
    ("Without adjusting for age, predict hospital mortality.", "prediction_model"),
    ("Do not fit regression, predict mortality with a score.", "prediction_model"),
    ("Do not use regression, prediction should use a neural network.", "prediction_model"),
    ("Do not exclude patients with missing measurements when building a mortality prediction model.", "prediction_model"),
    ("Only include patients with no prior hospitalization and build a model to predict mortality.", "prediction_model"),
    ("不做回归，预测院内死亡。", "prediction_model"),
    ("不做回归,预测院内死亡。", "prediction_model"),
    ("估计治疗的因果效应。", "causal_inference"),
    ("Estimate the causal effect of treatment.", "causal_inference"),
    ("不做因果分析，但建立预测院内死亡的模型。", "prediction_model"),
    ("不做预测，但估计治疗的因果效应。", "causal_inference"),
    ("不做预测，进行因果分析。", "causal_inference"),
    ("Do not predict mortality; instead estimate the causal effect of treatment.", "causal_inference"),
    ("No causal inference, but build a mortality prediction model.", "prediction_model"),
    ("Do not estimate causal effects for drug A. Estimate the causal effect of drug B.", "causal_inference"),
    ("Not only predict mortality but also report calibration.", "prediction_model"),
    ("调整性别后分析乳酸与死亡的关系，不做预测或因果分析。", "association_study"),
])
def test_positive_requests_and_later_contrasts_keep_their_family(question, family):
    assert planner(question)[0] == family
    assert host_question_family(question) == family


def test_population_negation_is_not_an_analysis_exclusion():
    question = "不要求患者跨住院唯一，建立预测院内死亡的模型。"
    assert mask_declined_analysis_phrases(question) == question
    assert planner(question)[0] == "prediction_model"


def test_host_adjustment_reading_ignores_declined_other_families():
    question = "请调整性别比较乳酸，不做预测或因果分析。"
    reading = study_family_reading(question)
    assert reading is not None and reading.family == "association_study"
    assert reading.elements[0].element == "adjustment"
    assert reading.elements[0].evidence == "调整性别比较乳酸"
