"""A question is planned on the family its words state, and the run log says how.

The planner's keyword routing (``analysis_types.infer_analysis_type``) reads a
relationship question in the orders researchers write it ("X 是否与 Y 相关",
"X 与 Y 是否有关"), a proportion question as descriptive, and a question about
the data themselves as a data-quality audit, whichever fields the context
holds.  A question no family's words read is described; only an empty
question is a data-quality audit.  How the family was chosen -- the study's
typed design, the question's words, or that fallback -- is written to the run
log (``analysis_family_trace``).  Synthetic questions only.
"""

from __future__ import annotations

from typing import Any

import pytest

from easyicu.research_agent.planning.analysis_family_trace import (
    traced_analysis_type_key,
)
from easyicu.research_agent.planning.analysis_types import (
    FAMILY_BASIS_FALLBACK,
    FAMILY_BASIS_KEYWORD,
    FAMILY_BASIS_STUDY_DESIGN,
    infer_analysis_type_with_basis,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)


def _context(question: str, **fields: Any) -> ResearchContext:
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="study", database="miiv", n_stays=0),
        variables=[],
        **fields,
    )


_FIELDS = (
    {},
    {"target_outcome": "death"},
    {"target_outcome": "death", "primary_exposure": "lactate_max"},
)


@pytest.mark.parametrize("fields", _FIELDS)
@pytest.mark.parametrize(
    ("question", "family"),
    [
        (
            "在成人 ICU 住院中，入 ICU 时的乳酸是否与 28 天死亡相关？",
            "association_study",
        ),
        (
            "在成人 ICU 住院中，入 ICU 时的乳酸与 28 天死亡是否有关？",
            "association_study",
        ),
        ("在成人 ICU 住院中，乳酸和院内死亡是否存在关联？", "association_study"),
        ("在成人 ICU 住院中，乳酸跟院内死亡是否有关？", "association_study"),
        (
            "Among adult ICU stays, what proportion had a sodium below 130 mmol/L?",
            "descriptive_epidemiology",
        ),
        (
            "在成人 ICU 住院中，前 24 小时低钠血症的比例是多少？",
            "descriptive_epidemiology",
        ),
        (
            "在成人 ICU 住院中，前 24 小时有多大比例发生低血糖？",
            "descriptive_epidemiology",
        ),
        (
            "在成人 ICU 住院中，前 24 小时乳酸的缺失率和数据完整性如何？",
            "data_quality_audit",
        ),
    ],
)
def test_the_question_states_its_family_whatever_the_context_holds(
    question, family, fields
) -> None:
    spec, basis = infer_analysis_type_with_basis(_context(question, **fields))

    assert (spec.key, basis) == (family, FAMILY_BASIS_KEYWORD)


@pytest.mark.parametrize(
    "question",
    [
        "列出与本研究相关的变量，并报告它们的缺失率。",
        "Among adult ICU stays, what proportion died after adjustment for age?",
    ],
)
def test_an_attributive_or_inferential_wording_is_not_read_as_those_families(
    question,
) -> None:
    spec, basis = infer_analysis_type_with_basis(_context(question))

    if "相关的变量" in question:
        # A noun qualified as relevant asks no relationship; the data are asked about.
        assert spec.key == "data_quality_audit"
    else:
        # An adjusted proportion is not a description by its words.
        assert basis == FAMILY_BASIS_FALLBACK


def test_a_question_no_family_reads_is_described_and_an_empty_one_audited() -> None:
    question = "成人 ICU 住院患者前 24 小时的血糖情况。"

    spec, basis = infer_analysis_type_with_basis(_context(question))
    assert (spec.key, basis) == ("descriptive_epidemiology", FAMILY_BASIS_FALLBACK)
    paired = infer_analysis_type_with_basis(
        _context(question, target_outcome="death", primary_exposure="glu_min")
    )
    assert (paired[0].key, paired[1]) == ("association_study", FAMILY_BASIS_FALLBACK)
    empty = infer_analysis_type_with_basis(_context(""))
    assert (empty[0].key, empty[1]) == ("data_quality_audit", FAMILY_BASIS_FALLBACK)


def test_a_stated_design_decides_and_the_run_log_says_how() -> None:
    context = _context(
        "成人 ICU 住院患者前 24 小时的血糖情况。",
        user_preferences=UserPreferences(inferred_analysis_family="association_study"),
    )
    events: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    key = traced_analysis_type_key(
        context, lambda *args, **kwargs: events.append((args, kwargs)), "run-1"
    )

    assert key == "association_study"
    ((args, kwargs),) = events
    assert args[0] == "plan"
    assert kwargs == {
        "run_id": "run-1",
        "analysis_family": "association_study",
        "family_basis": FAMILY_BASIS_STUDY_DESIGN,
    }


@pytest.mark.parametrize(
    ("question", "family", "basis"),
    [
        (
            "前 24 小时低钠血症的比例是多少？",
            "descriptive_epidemiology",
            FAMILY_BASIS_KEYWORD,
        ),
        (
            "成人 ICU 入住患者的血糖情况。",
            "descriptive_epidemiology",
            FAMILY_BASIS_FALLBACK,
        ),
        ("", "data_quality_audit", FAMILY_BASIS_FALLBACK),
    ],
)
def test_the_live_progress_channel_keeps_the_family_basis(
    question, family, basis
) -> None:
    from types import SimpleNamespace
    from easyicu.research_agent.orchestration.progress import ResumableProgressChannel

    events = []
    audit = []
    channel = ResumableProgressChannel(events.append)
    channel.bind_audit_logger(SimpleNamespace(emit=lambda **row: audit.append(row)))
    assert traced_analysis_type_key(_context(question), channel.emit, "run-1") == family
    assert len(events) == 1
    assert events[0]["stage"] == "plan"
    assert events[0]["analysis_family"] == family
    assert events[0]["family_basis"] == basis
    assert "research_question" not in events[0]
    assert audit[0]["phase"] == "plan"
    assert audit[0]["detail"]["analysis_family"] == family
    assert audit[0]["detail"]["family_basis"] == basis


def test_the_current_design_wins_over_conflicting_question_words() -> None:
    context = _context(
        "前 24 小时低钠血症的比例是多少？",
        user_preferences=UserPreferences(inferred_analysis_family="association_study"),
    )
    assert infer_analysis_type_with_basis(context)[0].key == "association_study"
    assert infer_analysis_type_with_basis(context)[1] == FAMILY_BASIS_STUDY_DESIGN


@pytest.mark.parametrize(
    ("question", "fields", "family", "basis"),
    [
        ("血糖情况", {"target_outcome": "death"}, "descriptive_epidemiology", "context"),
        (
            "Describe lactate.", {"target_outcome": "death"},
            "descriptive_epidemiology", FAMILY_BASIS_KEYWORD,
        ),
        (
            "Describe lactate.", {"cross_database_validation": ["synthetic_external"]},
            "cross_database_replication", "context",
        ),
        (
            "Describe lactate.",
            {"user_preferences": UserPreferences(preferred_methods="predict")},
            "prediction_model", "context",
        ),
        (
            "Predict mortality.", {"target_outcome": "death"},
            "prediction_model", FAMILY_BASIS_KEYWORD,
        ),
    ],
)
def test_the_basis_names_the_source_that_decided_the_family(
    question, fields, family, basis,
) -> None:
    spec, actual_basis = infer_analysis_type_with_basis(_context(question, **fields))
    assert (spec.key, actual_basis) == (family, basis)


def test_a_family_selected_by_variable_metadata_has_a_context_basis() -> None:
    context = _context("Describe lactate.").model_copy(update={
        "variables": [ConceptDescriptor(name="notes", role=VariableRole.OTHER, dtype="str")],
    })
    spec, basis = infer_analysis_type_with_basis(context)
    assert (spec.key, basis) == ("multimodal", "context")


@pytest.mark.parametrize(
    "field",
    [
        "preferred_methods", "evaluation_focus", "subgroup_sensitivity",
        "timing_and_design", "data_constraints", "must_have_outputs", "extra_notes",
        "covariates",
    ],
)
def test_data_quality_is_asked_only_by_the_question_not_context(field: str) -> None:
    words = "Assess missingness, completeness, coverage and 数据质量"
    preferences = UserPreferences(**{field: [words] if field == "covariates" else words})
    spec, basis = infer_analysis_type_with_basis(_context("血糖情况", user_preferences=preferences))
    assert (spec.key, basis) == ("descriptive_epidemiology", FAMILY_BASIS_FALLBACK)


@pytest.mark.parametrize(
    ("question", "family"),
    [
        ("What is the proportion of patients with missing potassium records?", "data_quality_audit"),
        ("How many patients with missing potassium records are there?", "data_quality_audit"),
        ("What proportion of patients with lactate values recorded have sepsis?", "descriptive_epidemiology"),
        ("How many patients have missing limb measurements?", "data_quality_audit"),
        ("Does the dataset have missing tooth data?", "data_quality_audit"),
        ("乳酸数据缺失的患者占比是多少？", "data_quality_audit"),
        ("指标缺失的比例是多少？", "data_quality_audit"),
        ("缺失指标的比例是多少？", "data_quality_audit"),
        ("指缺失的患者比例是多少？", "descriptive_epidemiology"),
        ("趾缺失的患者比例是多少？", "descriptive_epidemiology"),
        ("Missing glucose values will not be imputed; what proportion have hypotension?", "descriptive_epidemiology"),
        ("Missing sodium values are excluded; what proportion have hypotension?", "descriptive_epidemiology"),
        ("乳酸缺失的比例是多少？", "data_quality_audit"),
        ("血糖缺失的住院比例是多少？", "data_quality_audit"),
        ("有多大比例的住院乳酸缺失？", "data_quality_audit"),
        ("前 24 小时乳酸缺失的患者占比是多少？", "data_quality_audit"),
        ("各变量缺失的比例是多少？", "data_quality_audit"),
        ("What proportion of stays are missing lactate?", "data_quality_audit"),
        ("What percentage of patients lack a glucose measurement in the first 24 hours?", "data_quality_audit"),
        ("What proportion of stays have no recorded creatinine?", "data_quality_audit"),
        ("How many patients are missing a sodium value?", "data_quality_audit"),
        ("缺失的比例是多少？", "data_quality_audit"),
        ("缺失的模式如何？", "data_quality_audit"),
        ("What proportion is missing?", "data_quality_audit"),
        ("What proportion of stays lack potassium?", "data_quality_audit"),
        ("What proportion of stays have absent chloride measurements?", "data_quality_audit"),
        ("What proportion of stays have lactate not recorded?", "data_quality_audit"),
        ("What is the pattern of absent values?", "data_quality_audit"),
        ("肢体数据缺失的比例是多少？", "data_quality_audit"),
        ("What proportion of patients have missing limb measurements?", "data_quality_audit"),
        ("What proportion of patients have lactate missing and missing teeth?", "data_quality_audit"),
        ("截肢患者的比例是多少？", "descriptive_epidemiology"),
        ("肢体缺失的患者占比是多少？", "descriptive_epidemiology"),
        ("What proportion of patients have a missing limb?", "descriptive_epidemiology"),
        ("手指缺失的患者占比是多少？", "descriptive_epidemiology"),
        ("What proportion of patients lack a digit?", "descriptive_epidemiology"),
        ("What proportion of patients have an absent toe?", "descriptive_epidemiology"),
        ("What proportion of patients have teeth missing at admission?", "descriptive_epidemiology"),
        ("排除乳酸缺失的住院后，低血压的比例是多少？", "descriptive_epidemiology"),
        ("Among stays without missing vital signs, what proportion have hypotension?", "descriptive_epidemiology"),
        ("What proportion of patients with absent glucose records have hypotension?", "descriptive_epidemiology"),
        ("Among patients who lack sodium measurements, what proportion have hypotension?", "descriptive_epidemiology"),
        ("缺失乳酸的患者中，低血压的比例是多少？", "descriptive_epidemiology"),
        ("乳酸缺失的患者中低血压的比例是多少？", "descriptive_epidemiology"),
        ("Exclude stays with missing potassium; what proportion have hypotension?", "descriptive_epidemiology"),
        ("不统计乳酸缺失的比例，只描述血压分布。", "descriptive_epidemiology"),
        (
            "What proportion of stays have abnormal sodium values in the first 24 hours?",
            "descriptive_epidemiology",
        ),
        (
            "What proportion of patients have elevated lactate measurements during the first day?",
            "descriptive_epidemiology",
        ),
        (
            "What percentage of stays had hypoglycemic glucose values within 24 hours?",
            "descriptive_epidemiology",
        ),
        (
            "What proportion of stays have a glucose value below 70 mg/dL in the first 24 hours?",
            "descriptive_epidemiology",
        ),
        (
            "What fraction of ICU stays had creatinine values above 2 mg/dL during the stay?",
            "descriptive_epidemiology",
        ),
        (
            "Among adult ICU stays, how common is a lactate above 4 mmol/L in the first 24 hours?",
            "descriptive_epidemiology",
        ),
        ("前 24 小时血糖低于 70 mg/dL 的住院比例是多少？", "descriptive_epidemiology"),
        ("排除乳酸缺失的患者后，描述乳酸的分布。", "descriptive_epidemiology"),
        ("描述乳酸分布，缺失值不插补。", "descriptive_epidemiology"),
        (
            "Describe lactate distribution after excluding stays with missing lactate.",
            "descriptive_epidemiology",
        ),
        ("描述乳酸与死亡的关系。", "descriptive_epidemiology"),
        ("不分析乳酸是否与死亡相关，只描述分布。", "descriptive_epidemiology"),
        ("乳酸与死亡的关联不在本次研究范围，只描述乳酸分布。", "descriptive_epidemiology"),
        (
            "Do not analyse whether lactate is associated with death; only describe its distribution.",
            "descriptive_epidemiology",
        ),
        (
            "估计成人 ICU 住院中低钠血症的患病率，并描述其与院内死亡的关系。",
            "descriptive_epidemiology",
        ),
        ("列出各变量的缺失情况。", "data_quality_audit"),
        ("列出是否与本研究相关的变量的缺失情况。", "data_quality_audit"),
        ("How many stays are missing a lactate measurement?", "data_quality_audit"),
        (
            "What proportion of stays have a lactate value recorded in the first 24 hours?",
            "data_quality_audit",
        ),
        (
            "What proportion of stays have a lactate value measured in the first 24 hours?",
            "data_quality_audit",
        ),
        (
            "What proportion of stays have a lactate value documented within 24 hours?",
            "data_quality_audit",
        ),
        ("How many lactate values were documented during admission?", "data_quality_audit"),
        (
            "What is the missingness of lactate in the first 24 hours of adult ICU stays?",
            "data_quality_audit",
        ),
        ("乳酸的缺失率是多少？", "data_quality_audit"),
        ("评估各变量的完整性。", "data_quality_audit"),
        ("检查数据质量。", "data_quality_audit"),
        ("Audit the availability of creatinine measurements across ICU units.", "data_quality_audit"),
        ("乳酸与死亡有关系吗？", "association_study"),
        ("不分析乳酸与死亡的关系；评估血糖是否与死亡相关。", "association_study"),
        (
            "Is hyperglycemia linked to ICU mortality after adjustment for age and SOFA?",
            "association_study",
        ),
        ("在成人 ICU 住院中，入 ICU 时的乳酸是否与 28 天死亡相关？", "association_study"),
        ("是否有关于液体平衡的记录？", None),
        ("What proportion of patients have missing heart rate data?", "data_quality_audit"),
        ("How complete are the blood pressure recordings in the first 24 hours?", "data_quality_audit"),
        ("What percentage of stays have missing respiratory rate values?", "data_quality_audit"),
        ("What fraction of patients had missing temperature measurements?", "data_quality_audit"),
        ("How many admissions have missing urine output records?", "data_quality_audit"),
        ("What proportion of stays have missing oxygen saturation recordings?", "data_quality_audit"),
        ("What share of patients has missing fluid balance documentation?", "data_quality_audit"),
        ("How complete is the respiratory rate data during admission?", "data_quality_audit"),
        ("How available are the potassium measurements in the first day?", "data_quality_audit"),
        ("How well recorded are the urine output measurements?", "data_quality_audit"),
        ("How often is heart rate recorded in the first 24 hours?", "data_quality_audit"),
        ("How often is urine output documented during the stay?", "data_quality_audit"),
        ("Is temperature data available for all stays?", "data_quality_audit"),
        ("Is respiratory rate recorded for all stays?", "data_quality_audit"),
        ("首日血压数据缺失的比例是多少？", "data_quality_audit"),
        ("检查呼吸频率的数据缺失情况。", "data_quality_audit"),
        ("体温记录是否完整？", "data_quality_audit"),
        ("尿量的记录完整吗？", "data_quality_audit"),
        ("前一天血氧饱和度的数据是否完整？", "data_quality_audit"),
        ("有多少住院缺少心率记录？", "data_quality_audit"),
        ("有多少患者缺少血压测量记录？", "data_quality_audit"),
        ("有多少住院缺少尿量数据？", "data_quality_audit"),
        ("血压多久记录一次？", "data_quality_audit"),
        ("尿量多久记载一次？", "data_quality_audit"),
        ("是否所有住院都有体温记录？", "data_quality_audit"),
        ("血氧饱和度是否在所有住院中都有记录？", "data_quality_audit"),
        ("What proportion of patients have a complete response?", "descriptive_epidemiology"),
        ("What percentage of stays have complete heart block?", "descriptive_epidemiology"),
        ("What fraction of patients have a recorded diagnosis of sepsis?", "descriptive_epidemiology"),
        ("What proportion of stays have documented delirium?", "descriptive_epidemiology"),
        ("What percentage of beds are available during the night?", "descriptive_epidemiology"),
        ("What proportion of beds are available for all patients?", "descriptive_epidemiology"),
        ("What proportion of patients have missing low blood pressure measurements?", "data_quality_audit"),
        ("Does the dataset have missing blood pressure documentation?", "data_quality_audit"),
        ("Which dataset has missing temperature recordings?", "data_quality_audit"),
        ("What proportion of patients have missing teeth at admission?", "descriptive_epidemiology"),
        ("What fraction of stays have high blood pressure recorded in the first day?", "descriptive_epidemiology"),
        ("What proportion of patients have low oxygen saturation documented during admission?", "descriptive_epidemiology"),
        ("How common is complete remission among adult patients?", "descriptive_epidemiology"),
        ("Describe the distribution of available beds across wards.", "descriptive_epidemiology"),
        ("What proportion of patients with missing heart rate data have sepsis?", "descriptive_epidemiology"),
        ("What proportion of patients have documented acute kidney injury?", "descriptive_epidemiology"),
        ("完全缓解的患者比例是多少？", "descriptive_epidemiology"),
        ("完全性房室传导阻滞的住院占比是多少？", "descriptive_epidemiology"),
        ("记录为脓毒症的患者比例是多少？", "descriptive_epidemiology"),
        ("有谵妄记载的住院比例是多少？", "descriptive_epidemiology"),
        ("夜间可用床位的占比是多少？", "descriptive_epidemiology"),
        ("入院时存在牙齿缺失的患者比例是多少？", "descriptive_epidemiology"),
        ("记录为高血压的患者比例是多少？", "descriptive_epidemiology"),
        ("有低血氧记载的住院比例是多少？", "descriptive_epidemiology"),
        ("描述完全缓解患者的年龄分布。", "descriptive_epidemiology"),
        ("描述各病区可用床位的分布。", "descriptive_epidemiology"),
        ("缺少心率记录的患者中，脓毒症的比例是多少？", "descriptive_epidemiology"),
        ("有急性肾损伤记载的患者比例是多少？", "descriptive_epidemiology"),
        ("Exclude stays that have missing heart rate data; what proportion have sepsis?", "descriptive_epidemiology"),
        ("What proportion have sepsis after excluding stays that have missing temperature recordings?", "descriptive_epidemiology"),
        ("Handle missing urine output documentation; what proportion have sepsis?", "descriptive_epidemiology"),
        ("Do not assess how complete the blood pressure recordings are; what proportion have sepsis?", "descriptive_epidemiology"),
        ("排除缺少心率记录的患者；脓毒症的比例是多少？", "descriptive_epidemiology"),
        ("处理数据缺失后，脓毒症的比例是多少？", "descriptive_epidemiology"),
        ("不检查体温记录是否完整；脓毒症的比例是多少？", "descriptive_epidemiology"),
        ("不要统计有多少住院缺少尿量数据；脓毒症的比例是多少？", "descriptive_epidemiology"),
        ("What proportion of stays had at least one lactate measured in the first 6 hours?", "descriptive_epidemiology"),
        ("前 24 小时有血糖记录的住院比例是多少？", "descriptive_epidemiology"),
    ],
)
def test_clinical_states_and_variable_audits_keep_the_requested_family(
    question: str, family: str | None,
) -> None:
    spec, _ = infer_analysis_type_with_basis(_context(question))
    if family is None:
        assert spec.key != "association_study"
        return
    assert spec.key == family
    paired, _ = infer_analysis_type_with_basis(
        _context(question, primary_exposure="lactate", target_outcome="death")
    )
    assert paired.key == family


@pytest.mark.parametrize(
    "question",
    [
        'What proportion of stays have raised lactate values in the first day?',
        'What fraction of patients had high potassium measurements during admission?',
        'What percentage of stays have low hemoglobin values within 12 hours?',
        'What proportion of patients had reduced calcium measurements in the first day?',
        'What proportion of stays have normal sodium values during admission?',
        'What proportion of patients have hypokalemic potassium values in the first day?',
        'What fraction of stays had hypernatremic sodium measurements within 48 hours?',
        'What proportion of stays have glucose values over 180 mg/dL during admission?',
        'What proportion of stays have glucose values under 60 mg/dL during admission?',
        'What proportion of stays have creatinine values greater than 3 mg/dL within 24 hours?',
        'What fraction of stays have calcium values less than 8 mg/dL in the first day?',
        'What percentage of stays have > 4 mmol/L lactate values during admission?',
        'What proportion of stays have 60 mg/dL glucose values in the first day?',
        'What proportion of stays have glucose values = 60 mg/dL in the first day?',
        'What proportion of sodium values are recorded as 135 during admission?',
        'What fraction of stays have sodium values in the normal range during the first day?',
        'What proportion of stays have lactate measurements within the elevated range in the first day?',
        'What percentage of stays have glucose values in the range below 60 mg/dL?',
        'What fraction of stays have sodium values in mmol/L within the first day?',
        'What proportion of stays have mmol/L lactate values during admission?',
        'What proportion of stays have sodium values recorded as >ULN within 24 hours?',
        'What fraction of lactate values are measured above 4 mmol/L during the first day?',
        'What percentage of sodium values were recorded below 130 mmol/L within 12 hours?',
        'What is the rate of hypokalemia among stays with recorded potassium values?',
        'Describe the distribution of low sodium values among adult ICU stays.',
        'What proportion of stays have sodium values recorded as abnormal within 24 hours?',
        'Describe the distribution of elevated lactate among stays with low measurement coverage.',
        'What is the rate of hypoglycemia in stays with low measurement coverage?',
        'What proportion of stays with low measurement completeness have hypoglycemia?',
        '前一天出现高乳酸血症的患者比例是多少？',
        '统计住院患者中血钾偏低的比例。',
        '前 12 小时血红蛋白降低的发生率是多少？',
        '描述成人 ICU 患者中低钙血症的分布。',
        '入院时血钠正常的住院占比是多少？',
        '前两天存在低钾血症的患者比例是多少？',
        '统计高钠血症在不同病区的发生率。',
        '描述血糖超过 180 mg/dL 的患者比例。',
        '血糖低于 60 mg/dL 的住院比例是多少？',
        '首日肌酐大于 3 mg/dL 的患者有多大比例？',
        '血钙小于 8 mg/dL 的住院占比是多少？',
        '前一天乳酸 > 4 mmol/L 的患者比例是多少？',
        '描述前一天乳酸升高的分布，不插补缺失值。',
        '排除缺失血钠的记录后，低钠血症的发生率是多少？',
        '描述测量覆盖率较低的患者中低钠血症的分布。',
        '记录完整性较低的患者中，高乳酸血症的比例是多少？',
    ],
)
def test_clinical_state_proportions_rates_and_distributions_are_descriptive(question: str) -> None:
    spec, _ = infer_analysis_type_with_basis(_context(question))
    assert spec.key == "descriptive_epidemiology"


@pytest.mark.parametrize(
    ("audit_request", "decline"),
    [
        ("列出各变量的缺失情况。", "不列出各变量的缺失情况，只描述乳酸分布。"),
        ("汇总各变量的完整性。", "不汇总各变量的完整性，只描述乳酸分布。"),
        ("展示各变量的缺失情况。", "不展示各变量的缺失情况，只描述乳酸分布。"),
        ("呈现各变量的完整性。", "不呈现各变量的完整性，只描述乳酸分布。"),
        ("查看各变量的缺失情况。", "不查看各变量的缺失情况，只描述乳酸分布。"),
        ("List missingness of the study variables.", "Do not list missingness; describe lactate."),
        ("Tabulate completeness of the study variables.", "Do not tabulate completeness; describe lactate."),
        ("Show missingness of the study variables.", "Do not show missingness; describe lactate."),
    ],
)
def test_listing_requests_a_variable_audit_unless_declined(audit_request: str, decline: str) -> None:
    spec, _ = infer_analysis_type_with_basis(_context(audit_request))
    assert spec.key == "data_quality_audit"
    spec, _ = infer_analysis_type_with_basis(_context(decline))
    assert spec.key == "descriptive_epidemiology"
