"""The host's family reading and the planner's routing agree on a question's words.

Two layers read a question's analysis family: the host's study setup records
a design from it (``webserver.study_family_reading``), and the planner's
keyword routing plans a study that records none
(``research_agent.planning.analysis_types``).  On every question the host
reads, the planner, given no design and no fields, reads the same family; on
a question about the data themselves, which the host leaves unread, the
planner audits the data.  Synthetic questions only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.planning.analysis_types import (
    infer_analysis_type_with_basis,
)
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
from easyicu.webserver.study_family_reading import study_family_reading

_READ = [
    "Is admission lactate associated with 28-day mortality among adult ICU stays?",
    "Examine the association between body mass index and in-hospital mortality.",
    "What is the relationship between early fluid balance and ICU mortality?",
    "What are the risk factors for delirium in adult ICU stays?",
    "在成人 ICU 住院中，入 ICU 时的乳酸是否与 28 天死亡相关？",
    "在成人 ICU 住院中，入 ICU 时的乳酸与 28 天死亡是否有关？",
    "在成人 ICU 住院中，乳酸跟院内死亡是否有关？",
    "成人 ICU 住院中谵妄的危险因素有哪些？",
    "Among adult ICU stays, what proportion had a sodium below 130 mmol/L in the first 24 hours?",
    "How common is hypoglycemia in the first 24 hours of an ICU stay?",
    "在成人 ICU 住院中，前 24 小时低钠血症的比例是多少？",
    "在成人 ICU 住院中，前 24 小时有多大比例发生低血糖？",
    "估计成人 ICU 住院中低钠血症的患病率，并描述其与院内死亡的关系。",
    "What proportion of stays have abnormal sodium values in the first 24 hours?",
    "What proportion of patients have elevated lactate measurements during the first day?",
    "What percentage of stays had hypoglycemic glucose values within 24 hours?",
    "What fraction of lactate values are measured above 4 mmol/L during the first day?",
    "What percentage of sodium values were recorded below 130 mmol/L within 12 hours?",
    "What proportion of stays with low measurement completeness have hypoglycemia?",
    "What fraction of stays have sodium values in the normal range during the first day?",
    "What proportion of stays have lactate measurements within the elevated range in the first day?",
    "What percentage of stays have glucose values in the range below 60 mg/dL?",
    "What fraction of stays have sodium values in mmol/L within the first day?",
    "记录完整性较低的患者中，高乳酸血症的比例是多少？",
]
_DATA = [
    "What is the missingness of lactate in the first 24 hours of adult ICU stays?",
    "在成人 ICU 住院中，前 24 小时乳酸的缺失率和数据完整性如何？",
    "列出各变量的缺失情况。",
    "列出是否与本研究相关的变量的缺失情况。",
    "Show missingness of the study variables.",
    "What proportion of stays have a lactate value measured in the first 24 hours?",
    "What proportion of stays have a lactate value documented within 24 hours?",
]


def _planner(question: str) -> tuple[str, str]:
    spec, basis = infer_analysis_type_with_basis(
        ResearchContext(
            research_question=question,
            cohort=CohortDescriptor(cohort_name="study", database="miiv", n_stays=0),
            variables=[],
        )
    )
    return spec.key, basis


@pytest.mark.parametrize("question", _READ)
def test_the_planner_reads_the_family_the_host_reads(question: str) -> None:
    reading = study_family_reading(question)

    assert reading is not None
    assert _planner(question) == (reading.family, "keyword")


@pytest.mark.parametrize("question", _DATA)
def test_a_question_about_the_data_is_unread_by_the_host_and_audited_by_the_planner(
    question: str,
) -> None:
    assert study_family_reading(question) is None
    assert _planner(question) == ("data_quality_audit", "keyword")


@pytest.mark.parametrize(
    ("question", "family", "host_family", "basis"),
    [
        ("排除乳酸缺失的患者", "descriptive_epidemiology", None, "fallback"),
        ("缺失值不插补", "descriptive_epidemiology", None, "fallback"),
        ("排除乳酸缺失的患者，只描述乳酸分布", "descriptive_epidemiology", None, "keyword"),
        ("描述乳酸分布并排除缺失率较高的变量", "descriptive_epidemiology", None, "keyword"),
        ("Describe lactate after excluding records with low completeness.", "descriptive_epidemiology", None, "keyword"),
        ("不评估各变量的完整性，只描述乳酸分布", "descriptive_epidemiology", None, "keyword"),
        ("Do not audit missingness; describe lactate.", "descriptive_epidemiology", None, "keyword"),
        (
            "缺失值不插补，低钠血症的比例是多少？",
            "descriptive_epidemiology", "descriptive_epidemiology", "keyword",
        ),
        (
            "Describe lactate; exclude stays with missing lactate values. "
            "What proportion have hypoglycemia?",
            "descriptive_epidemiology", "descriptive_epidemiology", "keyword",
        ),
        (
            "Do not impute missing values; what proportion have hypoglycemia?",
            "descriptive_epidemiology", "descriptive_epidemiology", "keyword",
        ),
        ("乳酸的缺失率是多少？", "data_quality_audit", None, "keyword"),
        ("评估各变量的完整性", "data_quality_audit", None, "keyword"),
        ("What proportion of stays have a lactate value?", "data_quality_audit", None, "keyword"),
        ("Audit measurement coverage across stays.", "data_quality_audit", None, "keyword"),
        ("What proportion of lactate values are missing?", "data_quality_audit", None, "keyword"),
        ("What proportion had a lactate value below 2 mmol/L?", "descriptive_epidemiology", "descriptive_epidemiology", "keyword"),
        ("What proportion of stays with missing lactate died?", "descriptive_epidemiology", "descriptive_epidemiology", "keyword"),
        ("What proportion died among stays with recorded lactate?", "descriptive_epidemiology", "descriptive_epidemiology", "keyword"),
        ("乳酸是否与死亡相关", "association_study", "association_study", "keyword"),
        ("Is lactate associated with mortality?", "association_study", "association_study", "keyword"),
        ("Is lactate related to mortality?", "association_study", "association_study", "keyword"),
        ("乳酸与死亡有关系吗？", "association_study", "association_study", "keyword"),
        ("乳酸和死亡之间有何关系？", "association_study", "association_study", "keyword"),
        ("描述乳酸分布并估计乳酸与死亡的关联", "association_study", "association_study", "keyword"),
        ("Describe lactate and estimate the association between lactate and mortality.", "association_study", "association_study", "keyword"),
        ("描述乳酸与死亡的关系", "descriptive_epidemiology", None, "keyword"),
        ("Describe the relationship between lactate and mortality.", "descriptive_epidemiology", None, "keyword"),
        ("不分析乳酸是否与死亡相关，只描述分布", "descriptive_epidemiology", None, "keyword"),
        ("不研究乳酸与死亡是否有关，只描述分布", "descriptive_epidemiology", None, "keyword"),
        ("乳酸是否与死亡相关不在本次研究范围，只描述分布", "descriptive_epidemiology", None, "keyword"),
        (
            "乳酸与死亡是否有关不在本次研究范围；低钠血症的比例是多少？",
            "descriptive_epidemiology", "descriptive_epidemiology", "keyword",
        ),
        (
            "Whether lactate is associated with mortality is outside the scope of this study; describe lactate.",
            "descriptive_epidemiology", None, "keyword",
        ),
        ("Do not test whether lactate is related to mortality; describe lactate.", "descriptive_epidemiology", None, "keyword"),
        ("是否与本研究相关的变量", "descriptive_epidemiology", None, "fallback"),
        ("是否与本研究有关的记录", "descriptive_epidemiology", None, "fallback"),
        ("是否与本研究关联的变量", "descriptive_epidemiology", None, "fallback"),
        ("乳酸与本研究是否有关的变量", "descriptive_epidemiology", None, "fallback"),
        ("是否有关于液体平衡的记录", "descriptive_epidemiology", None, "fallback"),
        ("Are these variables related to the study?", "descriptive_epidemiology", None, "fallback"),
        ("Are these records associated with this study?", "descriptive_epidemiology", None, "fallback"),
        ("Is there a record about fluid balance?", "descriptive_epidemiology", None, "fallback"),
        ("Are there records about the association between lactate and mortality?", "descriptive_epidemiology", None, "fallback"),
        (
            "不分析乳酸是否与死亡相关；血钠是否与死亡相关？",
            "association_study", "association_study", "keyword",
        ),
        (
            "不研究乳酸与死亡是否有关，血钠与死亡是否有关？",
            "association_study", "association_study", "keyword",
        ),
        (
            "乳酸与死亡是否有关不在本次研究范围；血钠是否与死亡相关？",
            "association_study", "association_study", "keyword",
        ),
        (
            "Do not test whether lactate is associated with mortality, but is sodium associated with mortality?",
            "association_study", "association_study", "keyword",
        ),
    ],
)
def test_incidental_declined_and_attributive_words_do_not_state_a_family(
    question: str, family: str, host_family: str | None, basis: str,
) -> None:
    # The host deliberately leaves descriptions without proportion evidence,
    # and data-quality questions, to the conversation rather than recording a design.
    reading = study_family_reading(question)
    assert (reading.family if reading else None) == host_family
    assert _planner(question) == (family, basis)


def test_the_host_evidence_is_from_the_requested_relationship_clause() -> None:
    question = "乳酸与死亡是否有关不在本次研究范围；血钠是否与死亡相关？"
    reading = study_family_reading(question)

    assert reading is not None
    assert [item.evidence for item in reading.elements] == ["是否与死亡相关"]
    assert reading.elements[0].start > question.index("；")


@pytest.mark.parametrize(
    ("question", "family", "host_family"),
    [
        ("What is the proportion of patients with missing potassium records?", "data_quality_audit", None),
        ("How many patients with missing potassium records are there?", "data_quality_audit", None),
        ("What proportion of patients with lactate values recorded have sepsis?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("How many patients have missing limb measurements?", "data_quality_audit", None),
        ("Does the dataset have missing tooth data?", "data_quality_audit", None),
        ("乳酸数据缺失的患者占比是多少？", "data_quality_audit", None),
        ("指标缺失的比例是多少？", "data_quality_audit", None),
        ("缺失指标的比例是多少？", "data_quality_audit", None),
        ("指缺失的患者比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("趾缺失的患者比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Missing glucose values will not be imputed; what proportion have hypotension?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Missing sodium values are excluded; what proportion have hypotension?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("乳酸缺失的比例是多少？", "data_quality_audit", None),
        ("血糖缺失的住院比例是多少？", "data_quality_audit", None),
        ("有多大比例的住院乳酸缺失？", "data_quality_audit", None),
        ("前 24 小时乳酸缺失的患者占比是多少？", "data_quality_audit", None),
        ("各变量缺失的比例是多少？", "data_quality_audit", None),
        ("What proportion of stays are missing lactate?", "data_quality_audit", None),
        ("What percentage of patients lack a glucose measurement in the first 24 hours?", "data_quality_audit", None),
        ("What proportion of stays have no recorded creatinine?", "data_quality_audit", None),
        ("How many patients are missing a sodium value?", "data_quality_audit", None),
        ("缺失的比例是多少？", "data_quality_audit", None),
        ("缺失的模式如何？", "data_quality_audit", None),
        ("What proportion is missing?", "data_quality_audit", None),
        ("What proportion of stays lack potassium?", "data_quality_audit", None),
        ("What proportion of stays have absent chloride measurements?", "data_quality_audit", None),
        ("What proportion of stays have lactate not recorded?", "data_quality_audit", None),
        ("What is the pattern of absent values?", "data_quality_audit", None),
        ("肢体数据缺失的比例是多少？", "data_quality_audit", None),
        ("What proportion of patients have missing limb measurements?", "data_quality_audit", None),
        ("What proportion of patients have lactate missing and missing teeth?", "data_quality_audit", None),
        ("截肢患者的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("肢体缺失的患者占比是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients have a missing limb?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("手指缺失的患者占比是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients lack a digit?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients have an absent toe?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients have teeth missing at admission?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("排除乳酸缺失的住院后，低血压的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Among stays without missing vital signs, what proportion have hypotension?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients with absent glucose records have hypotension?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Among patients who lack sodium measurements, what proportion have hypotension?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("缺失乳酸的患者中，低血压的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("乳酸缺失的患者中低血压的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Exclude stays with missing potassium; what proportion have hypotension?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("不统计乳酸缺失的比例，只描述血压分布。", "descriptive_epidemiology", None),
        ("What proportion of patients have missing heart rate data?", "data_quality_audit", None),
        ("How complete are the blood pressure recordings in the first 24 hours?", "data_quality_audit", None),
        ("What percentage of stays have missing respiratory rate values?", "data_quality_audit", None),
        ("What fraction of patients had missing temperature measurements?", "data_quality_audit", None),
        ("How many admissions have missing urine output records?", "data_quality_audit", None),
        ("What proportion of stays have missing oxygen saturation recordings?", "data_quality_audit", None),
        ("What share of patients has missing fluid balance documentation?", "data_quality_audit", None),
        ("How complete is the respiratory rate data during admission?", "data_quality_audit", None),
        ("How available are the potassium measurements in the first day?", "data_quality_audit", None),
        ("How well recorded are the urine output measurements?", "data_quality_audit", None),
        ("How often is heart rate recorded in the first 24 hours?", "data_quality_audit", None),
        ("How often is urine output documented during the stay?", "data_quality_audit", None),
        ("Is temperature data available for all stays?", "data_quality_audit", None),
        ("Is respiratory rate recorded for all stays?", "data_quality_audit", None),
        ("首日血压数据缺失的比例是多少？", "data_quality_audit", None),
        ("检查呼吸频率的数据缺失情况。", "data_quality_audit", None),
        ("体温记录是否完整？", "data_quality_audit", None),
        ("尿量的记录完整吗？", "data_quality_audit", None),
        ("前一天血氧饱和度的数据是否完整？", "data_quality_audit", None),
        ("有多少住院缺少心率记录？", "data_quality_audit", None),
        ("有多少患者缺少血压测量记录？", "data_quality_audit", None),
        ("有多少住院缺少尿量数据？", "data_quality_audit", None),
        ("血压多久记录一次？", "data_quality_audit", None),
        ("尿量多久记载一次？", "data_quality_audit", None),
        ("是否所有住院都有体温记录？", "data_quality_audit", None),
        ("血氧饱和度是否在所有住院中都有记录？", "data_quality_audit", None),
        ("What proportion of patients have a complete response?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What percentage of stays have complete heart block?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What fraction of patients have a recorded diagnosis of sepsis?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of stays have documented delirium?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What percentage of beds are available during the night?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of beds are available for all patients?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients have missing low blood pressure measurements?", "data_quality_audit", None),
        ("Does the dataset have missing blood pressure documentation?", "data_quality_audit", None),
        ("Which dataset has missing temperature recordings?", "data_quality_audit", None),
        ("What proportion of patients have missing teeth at admission?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What fraction of stays have high blood pressure recorded in the first day?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients have low oxygen saturation documented during admission?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("How common is complete remission among adult patients?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Describe the distribution of available beds across wards.", "descriptive_epidemiology", None),
        ("What proportion of patients with missing heart rate data have sepsis?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of patients have documented acute kidney injury?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("完全缓解的患者比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("完全性房室传导阻滞的住院占比是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("记录为脓毒症的患者比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("有谵妄记载的住院比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("夜间可用床位的占比是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("入院时存在牙齿缺失的患者比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("记录为高血压的患者比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("有低血氧记载的住院比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("描述完全缓解患者的年龄分布。", "descriptive_epidemiology", None),
        ("描述各病区可用床位的分布。", "descriptive_epidemiology", None),
        ("缺少心率记录的患者中，脓毒症的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("有急性肾损伤记载的患者比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Exclude stays that have missing heart rate data; what proportion have sepsis?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion have sepsis after excluding stays that have missing temperature recordings?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Handle missing urine output documentation; what proportion have sepsis?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("Do not assess how complete the blood pressure recordings are; what proportion have sepsis?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("排除缺少心率记录的患者；脓毒症的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("处理数据缺失后，脓毒症的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("不检查体温记录是否完整；脓毒症的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("不要统计有多少住院缺少尿量数据；脓毒症的比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("What proportion of stays had at least one lactate measured in the first 6 hours?", "descriptive_epidemiology", "descriptive_epidemiology"),
        ("前 24 小时有血糖记录的住院比例是多少？", "descriptive_epidemiology", "descriptive_epidemiology"),
    ],
)
def test_data_predicates_and_clinical_lookalikes_agree(
    question: str, family: str, host_family: str | None,
) -> None:
    reading = study_family_reading(question)
    assert (reading.family if reading else None) == host_family
    assert _planner(question) == (family, "keyword")


# A question that states its design in its own words (pilot1 10-22: a Cox
# question with 「关系」 was recorded as an association study) is left to the
# planner, which reads the stated family; the host records nothing.
_STATED_DESIGNS = [
    ("在 eICU 中，用 Cox 回归分析入 ICU 首日最高乳酸与 28 天生存的关系。", "survival"),
    ("高乳酸与低乳酸患者的生存时间有差别吗？请调整年龄。", "survival"),
    ("用首日生命体征预测院内死亡，并分析各指标与死亡的关系。", "prediction_model"),
    ("早期使用去甲肾上腺素与院内死亡的因果关系。", "causal_inference"),
    ("把脓毒症患者聚类成不同亚型，比较各亚型的院内死亡比例。", "trajectory_clustering"),
]


@pytest.mark.parametrize(("question", "family"), _STATED_DESIGNS)
def test_a_design_the_question_states_is_the_planners_to_read(question: str, family: str) -> None:
    assert study_family_reading(question) is None
    assert _planner(question)[0] == family
