"""A question about mean survival over a horizon reads its horizon, its groups and its estimand.

「在 SICdb 中，入 ICU 第一天接受机械通气的患者，28 天内平均存活多少天？与未通气的
患者相比，调整年龄和性别后差多少？」 read no endpoint: the shared horizon reader
(``outcome_availability``) read a horizon only right beside its death or
survival word, and 「平均」 or "on average" stood between.  With no main outcome
the keyword slot fell back to the first clinical event, ventilation.  The
shortened negative half 「未通气」 named no comparison group, and in the English
wording "age" inside "average" became the exposure.

The question asks how long patients lived within 28 days, a restricted mean
survival time: a time-to-event estimand, never the share who died by day 28.
Every family reader reads it as survival, the context builder requests a
time-to-event endpoint over the horizon, and because no plan reports a
restricted mean survival time as a study's main result the launch stops
(``question_mean_survival``) instead of answering with 28-day mortality (lead
review 10-27: it was planned as a binary 28-day death in every family).
Synthetic questions only.
"""

from __future__ import annotations

import pytest

from easyicu.outcome_availability import (
    stated_mean_survival_mentions,
    stated_mortality_horizon_mentions,
)
from easyicu.research_agent.planning.analysis_types import infer_analysis_type
from easyicu.research_agent.research_context.builder import _infer_outcome_semantics
from easyicu.research_agent.schema import CohortDescriptor, ResearchContext
from easyicu.webserver import research_launch_scientific as launch
from easyicu.webserver import stop_next_actions
from easyicu.webserver.question_family_check import host_question_family
from easyicu.webserver.question_mean_survival import (
    MEAN_SURVIVAL_UNPLANNED,
    asked_mean_survival,
    stop_unplanned_mean_survival,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.study_intent import (
    deterministic_intent,
    explicit_outcome_concepts,
    stated_comparison_groupings,
)

SICDB_RMST = (
    "在 SICdb 中，入 ICU 第一天接受机械通气的患者，28 天内平均存活多少天？与未通气的患者相比，"
    "调整年龄和性别后差多少？"
)
SICDB_RMST_STATED = (
    "在 SICdb 中，入 ICU 第一天接受机械通气的患者，以 28 天死亡为终点，28 天内平均存活多少天？"
    "与未通气的患者相比，调整年龄和性别后差多少？"
)
SICDB_RMST_EN = (
    "In SICdb, how many days do patients ventilated on the first ICU day survive on average "
    "within 28 days, compared with non-ventilated patients, adjusted for age and sex?"
)


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("28 天内平均存活多少天？", "28 天内平均存活"),
        ("28 天限制性平均生存时间", "28 天限制性平均生存"),
        ("二十八天内的平均生存时间", "二十八天内的平均生存"),
        ("How long do patients survive on average within 28 days?", "survive on average within 28 days"),
        ("restricted mean survival time (RMST) at 28 days", "survival time (RMST) at 28 days"),
        ("the 28-day restricted mean survival time", "28-day restricted mean survival"),
        ("mean survival time within 28 days", "survival time within 28 days"),
        ("days alive within 28 days", "alive within 28 days"),
    ],
)
def test_mean_survival_over_a_horizon_states_that_horizon(text: str, words: str) -> None:
    mentions = stated_mortality_horizon_mentions(text)
    assert [(item.horizon.count, item.horizon.unit, text[item.start:item.end]) for item in mentions] == [
        (28, "day", words),
    ]
    assert explicit_outcome_concepts(text) == ("mort_28d",)


@pytest.mark.parametrize(
    "text",
    [
        # Alive at a time is who is included, not an endpoint.
        "以入 ICU 后第 6 小时为时间零点，纳入当时仍存活的患者",
        "patients alive at 24 hours",
        # A mean over a horizon of something else.
        "28 天内平均乳酸",
        "平均住院 28 天的患者存活率",
        # Survival with no horizon names no fixed-horizon endpoint.
        "平均存活多少天？",
    ],
)
def test_survival_words_without_their_horizon_state_none(text: str) -> None:
    assert stated_mortality_horizon_mentions(text) == ()
    assert "mort_28d" not in explicit_outcome_concepts(text)


def _planner_family(question: str) -> str:
    return infer_analysis_type(ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="study", database="sic", n_stays=0),
        variables=[],
    )).key


@pytest.mark.parametrize("question", [SICDB_RMST, SICDB_RMST_STATED, SICDB_RMST_EN])
def test_the_sicdb_question_reads_a_time_to_event_estimand_by_ventilation(question: str) -> None:
    slots = deterministic_intent(question)["slots"]
    # The horizon's endpoint is 28-day death, read by ventilation ...
    assert (slots["exposure"]["value"], slots["outcome"]["value"]) == ("vent_ind", "mort_28d")
    # ... and the estimand is a time-to-event one in every family reader.
    assert slots["analysis_family"]["value"] == "survival"
    assert host_question_family(question) == "survival"
    assert _planner_family(question) == "survival"
    request = _infer_outcome_semantics(research_question=question, outcome_name="mort_28d")
    assert request["source_concept"] == "time_to_event_endpoint"
    assert [(item.count, item.unit) for item in request["horizons"]] == [(28, "day")]


@pytest.mark.parametrize("question", [SICDB_RMST, SICDB_RMST_STATED, SICDB_RMST_EN])
@pytest.mark.parametrize("family", [None, "association_study", "survival", "descriptive_epidemiology"])
def test_the_sicdb_question_stops_instead_of_answering_with_28_day_death(question: str, family) -> None:
    coordinates = launch._metadata_only_planning_coordinates(
        question=question, database="sic", analysis_family=family,
    )
    assert coordinates["primary_exposure"] == "vent_ind"
    with pytest.raises(ResearchPipelineRunError) as stopped:
        stop_unplanned_mean_survival(question)
    assert stopped.value.code == MEAN_SURVIVAL_UNPLANNED
    assert stopped.value.details["mean_survival"]


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("28 天内平均存活多少天？", "平均存活"),
        ("28 天限制性平均生存时间", "限制性平均生存时间"),
        ("存活天数的平均值", "存活天数的平均值"),
        ("What is the restricted mean survival time at 28 days?", "restricted mean survival time"),
        ("Report the RMST at day 28.", "RMST"),
        ("the mean survival time within 28 days", "mean survival time"),
        ("How long do patients survive on average within 28 days?", "survive on average"),
        ("How many days were patients alive on average?", "alive on average"),
        ("How many days alive within 28 days did ventilated patients have?", "days alive within 28 days"),
        ("What were the mean days alive within 28 days by group?", "mean days alive"),
        (
            "How long did ventilated patients live on average within 28 days compared with "
            "non-ventilated patients?",
            "live on average",
        ),
        ("How many days did ventilated patients survive, on average, within 28 days?", "survive, on average"),
    ],
)
def test_a_mean_survival_time_is_asked_in_these_words(text: str, words: str) -> None:
    assert [item.evidence for item in asked_mean_survival(text)] == [words]
    assert _planner_family(text) == "survival"


@pytest.mark.parametrize(
    "text",
    [
        # A share alive, a mean of something else, or a count.
        "平均存活率是多少？",
        "What is the average survival rate at 28 days?",
        "存活患者的平均乳酸是多少？",
        "mean arterial pressure among survivors",
        "days alive and free of ventilation at day 28",
        "days alive and out of hospital at day 90",
        "Is the average survival probability at 28 days different between ventilated and "
        "non-ventilated patients?",
        # Survivors described, or other days counted (delta review 10-27).
        "In MIMIC-IV, were ICU survivors on average older than non-survivors?",
        "Survivors were on average 5 years younger; is age associated with 28-day mortality?",
        "What is the number of days alive outside the hospital within 90 days by sepsis status?",
        "What is the number of days alive at home within 90 days by sepsis status?",
        "What is the mean days alive outside the hospital within 90 days by sepsis status?",
        "Survivors spent on average 3 fewer days in the ICU; is sepsis associated with ICU death?",
        # A copula between: the living described, not how long they lived
        # (fix2 review 10-27).
        "Patients still alive were on average older and spent more days in the ICU; is sepsis "
        "associated with ICU death?",
        # "On average" beside no duration, with and without a relative pronoun
        # (fix3 review 10-27).
        "Do patients who live on average 10 km away have higher mortality?",
        "Ventilated patients live on average 10 km from the hospital; is ventilation associated "
        "with 28-day mortality?",
        # Declined.
        "Use logistic regression rather than RMST for 28-day mortality.",
        "不用 RMST，报告 28 天死亡率",
    ],
)
def test_words_that_ask_no_mean_survival_time_state_none(text: str) -> None:
    assert asked_mean_survival(text) == ()
    stop_unplanned_mean_survival(text)


@pytest.mark.parametrize(
    "text",
    [
        # A relative clause describes survivors, whatever other days the
        # sentence counts (fix2 review 10-27).
        "Patients who survived, on average, spent fewer days on mechanical ventilation; is "
        "ventilation associated with 28-day mortality?",
        "Patients who survived on average had 3 more ventilator-free days; is ventilation "
        "associated with 28-day mortality?",
        "Patients that were alive on average had 3 more ventilator-free days; is ventilation "
        "associated with 28-day mortality?",
    ],
)
def test_survivors_described_in_a_relative_clause_ask_no_mean_survival(text: str) -> None:
    assert stated_mean_survival_mentions(text) == ()
    stop_unplanned_mean_survival(text)
    assert deterministic_intent(text)["slots"]["analysis_family"]["value"] == "association"
    assert host_question_family(text) == "association_study"
    assert _planner_family(text) == "association_study"


@pytest.mark.parametrize(
    "text",
    [
        "Cox regression of 28-day survival by ventilation, with RMST as a sensitivity analysis.",
        "用 Cox 回归分析 28 天生存，以 RMST 作为敏感性分析",
        "Cox regression as the main analysis, with RMST as a sensitivity analysis.",
        "Report a Cox model and a sensitivity analysis using RMST at 28 days.",
        # The SICdb rewording (delta review 10-27).
        "在 SICdb 中，入 ICU 第一天接受机械通气的患者与未通气的患者相比，调整年龄和性别后 28 天生存是否不同？"
        "并以 28 天限制性平均生存时间差作为敏感性分析。",
    ],
)
def test_a_mean_survival_asked_as_a_sensitivity_analysis_is_not_stopped(text: str) -> None:
    assert stated_mean_survival_mentions(text)
    assert asked_mean_survival(text) == ()
    stop_unplanned_mean_survival(text)


@pytest.mark.parametrize(
    "text",
    [
        "报告 28 天 RMST，并做排除早期死亡的敏感性分析",
        # Another method is the sensitivity, or the mean survival is the
        # primary result (delta review 10-27).
        "Estimate the 28-day RMST difference between ventilated and non-ventilated patients and "
        "use a Cox model as a sensitivity analysis.",
        "以 28 天限制性平均生存时间差为主要结局并以 Cox 模型作为敏感性分析，比较通气与未通气患者。",
        # The sensitivity words name another method just before the mean
        # survival (fix2 review 10-27).
        "Sensitivity analysis using Cox; estimate the RMST difference by sepsis status.",
        # The words bind, but the clause makes the mean survival the primary
        # analysis as well (fix2 review 10-27).
        "Primary and sensitivity analyses using the 28-day RMST, by sepsis status.",
        "主分析和敏感性分析采用 28 天 RMST，按脓毒症分组比较。",
    ],
)
def test_a_sensitivity_analysis_that_does_not_bind_to_the_mean_survival_does_not_exempt_it(
    text: str,
) -> None:
    with pytest.raises(ResearchPipelineRunError):
        stop_unplanned_mean_survival(text)


def test_the_launch_stops_before_anything_is_written(tmp_path, monkeypatch) -> None:
    from easyicu.webserver import study_contexts as context_store
    from easyicu.webserver.research_pipeline_run_preparation import (
        ResearchPipelineLaunchRequest,
        prepare_research_pipeline_run,
    )

    writes: list = []
    monkeypatch.setattr(
        context_store, "upsert_context", lambda raw, **kwargs: writes.append(dict(raw)) or dict(raw),
    )
    request = ResearchPipelineLaunchRequest(
        export_path=str(tmp_path / "export"),
        study_context={
            "id": "study-launch", "revision": 5, "question": SICDB_RMST,
            "data_source": {"path": str(tmp_path / "export"), "database": "sic"},
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
    with pytest.raises(ResearchPipelineRunError) as stopped:
        prepare_research_pipeline_run(request)
    assert stopped.value.code == MEAN_SURVIVAL_UNPLANNED
    assert writes == []
    assert not (tmp_path / "workspace").exists()


def test_the_stop_offers_an_explicit_plan_edit() -> None:
    assert stop_next_actions.STOP_POLICIES[MEAN_SURVIVAL_UNPLANNED] == "edit_plan"
    actions = stop_next_actions.next_actions_for_stop(MEAN_SURVIVAL_UNPLANNED, detail={})
    assert [action.kind for action in actions] == ["edit_plan"]


@pytest.mark.parametrize(("language", "expected"), [("en", "mean survival"), ("zh", "平均生存时间")])
def test_the_conversation_explains_the_stop_in_a_sentence(language: str, expected: str) -> None:
    import json
    from pathlib import Path

    from tests.support.node import run_node

    script = r"""
const modules = {};
global.window = { EU_HTML: {esc: x => x}, EasyICU: {guidedPi: {declare: (name, api) => {modules[name] = api;}}}};
require(process.argv[1]);
const pick = process.argv[2] === 'zh' ? (en, zh) => zh : (en, zh) => en;
const owner = modules.errorText.create({tr: pick, staticPreview: false});
console.log(JSON.stringify(owner.runFailureText(process.argv[3], {})));
"""
    path = (
        Path(__file__).resolve().parents[2]
        / "src/easyicu/webserver/static/js/screens-guided-pi-error-text.js"
    )
    result = run_node("node", script, str(path), language, MEAN_SURVIVAL_UNPLANNED, check=True)
    text = json.loads(result.stdout)
    assert expected in text
    assert MEAN_SURVIVAL_UNPLANNED not in text


@pytest.mark.parametrize(
    ("question", "evidence"),
    [
        (SICDB_RMST, "与未通气的患者相比"),
        ("接受机械通气患者的乳酸，与未接受机械通气的患者相比有何不同？", "与未接受机械通气的患者相比"),
        ("机械通气患者的乳酸较未通气患者高吗？", "较未通气患者"),
        ("机械通气组与未通气组相比的乳酸", "与未通气组相比"),
        (SICDB_RMST_EN, "compared with non-ventilated"),
        (
            "Is lactate higher with mechanical ventilation, compared with patients not receiving "
            "mechanical ventilation?",
            "compared with patients not receiving mechanical ventilation",
        ),
        ("Mortality in AKI, compared with patients without AKI", "compared with patients without AKI"),
    ],
)
def test_a_comparison_with_the_group_without_it_states_the_grouping(question: str, evidence: str) -> None:
    assert [(item.concept_id, item.evidence) for item in stated_comparison_groupings(question)] == [
        (
            "aki" if "AKI" in question else "vent_ind",
            evidence,
        ),
    ]


@pytest.mark.parametrize(
    "question",
    [
        # No mention names what the short word stands for.
        "与未通气的患者相比，乳酸更高吗？",
        # Joined in a list, not compared.
        "接受机械通气的患者和未通气的患者都纳入，描述乳酸",
        # No group noun: another thing without ventilation.
        "机械通气患者与未通气时间相比的乳酸",
        # 「无创」 is a kind of ventilation, not its absence.
        "与无创机械通气患者相比，有创机械通气患者的乳酸",
    ],
)
def test_a_negated_word_that_names_no_group_states_none(question: str) -> None:
    assert stated_comparison_groupings(question) == ()


@pytest.mark.parametrize(
    ("question", "exposure"),
    [
        ("Is the average lactate associated with in-hospital mortality, adjusted for age?", "lact"),
        ("Does the average heart rate differ by sex, adjusted for age?", "hr"),
        # A whole word, and a Chinese name, still read.
        ("Is age associated with in-hospital mortality?", "age"),
        ("年龄与院内死亡的关系", "age"),
    ],
)
def test_an_english_name_inside_another_word_is_not_read(question: str, exposure: str) -> None:
    assert deterministic_intent(question)["slots"]["exposure"]["value"] == exposure
