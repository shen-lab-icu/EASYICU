"""A treatment the question names is read as its use or as its dose, by its words.

Some treatments are recorded twice: whether they were given (a use status the
concept owner declares an event status, such as ``vaso_ind``) and how much was
given (a dose or rate, such as the norepinephrine-equivalent dose).  The Web
reader (``webserver.study_intent``) read 「血管活性药」 and "vasopressor" as the
dose, so a question about how many patients received vasopressors, or when
they were first started, was planned on a dose: a silent definitional error.
Use words (使用、用了、接受、received、administration、exposure、on) now
read the use status, dose words (剂量、速率、最大、dose、rate) the dose, and a
question about the first use the status's onset.  A dose-only word of the
clause (剂量、强度、dose、intensity) says the dose whatever use word stands
nearer (「血管活性药使用剂量」, "the dose of vasopressor used"), "based on" or
"depending on" is no use, and "patients receiving X" is the population of a
question that asks X's dose (lead review 10-27: these read the use).  A
treatment named with neither word is its use, as an exposure ordinarily is.
A use of a treatment no concept records the use of (norepinephrine: a rate
only) stops the launch (``question_treatment_use``), unless it only
illustrates a class whose use is recorded (「血管活性药（如去甲肾上腺素）」).
Synthetic questions and zero-row schemas only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.concept.catalog import CONCEPT_DICTIONARY, CONCEPT_GROUPS_INTERNAL
from easyicu.concept.export_metadata import concept_declares_event_status
from easyicu.concept.loader import load_dictionary
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import TimeWindow
from easyicu.webserver import research_launch_scientific as launch
from easyicu.webserver import stop_next_actions, study_intent
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.question_treatment_use import (
    TREATMENT_USE_UNRECORDED,
    stop_unrecorded_treatment_uses,
    unrecorded_treatment_uses,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.research_pipeline_run_preparation import (
    ResearchPipelineLaunchRequest,
    prepare_research_pipeline_run,
)
from easyicu.webserver.study_intent import (
    catalog_named_concepts,
    deterministic_intent,
    named_study_concepts,
    stated_comparison_groupings,
    stated_treatment_uses,
)
from tests.support.node import run_node

DESCRIPTIVE = "descriptive_epidemiology"
FIRST_USE = "在 AmsterdamUMCdb 中，入 ICU 第一天用了血管活性药的患者，首次使用多在入 ICU 后几小时？"


def _readings(question: str) -> list[tuple[str, str, bool]]:
    return [
        (item.concept_id, item.form, item.first_use) for item in stated_treatment_uses(question)
    ]


def test_the_table_pairs_a_declared_use_status_with_dose_quantities_of_the_same_treatment() -> None:
    dictionary = load_dictionary()
    treatments = set(CONCEPT_GROUPS_INTERNAL["medications"]) | set(
        CONCEPT_GROUPS_INTERNAL["vasopressors"]
    )
    for status, doses in study_intent._USE_OR_DOSE:
        assert doses
        if status is not None:
            assert status in treatments, status
            assert concept_declares_event_status(status, dictionary.get(status)), status
        for dose in doses:
            assert dose in treatments, dose
            assert not concept_declares_event_status(dose, dictionary.get(dose)), dose
            assert CONCEPT_DICTIONARY[dose][2] not in {"boolean", "category"}, dose
    named = [concept for entry in study_intent._USE_OR_DOSE for concept in (entry[0], *entry[1]) if concept]
    assert len(named) == len(set(named))


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # How many used it: its use status.
        ("在 AmsterdamUMCdb 中，入 ICU 第一天使用血管活性药的患者占多少？", [("vaso_ind", "use", False)]),
        ("入 ICU 后接受升压药治疗的患者比例", [("vaso_ind", "use", False)]),
        ("用血管活性药的患者院内死亡率", [("vaso_ind", "use", False)]),
        ("What share of patients received a vasopressor on the first ICU day?", [("vaso_ind", "use", False)]),
        ("Mortality of patients on vasopressors", [("vaso_ind", "use", False)]),
        ("Is any vasopressor use associated with death?", [("vaso_ind", "use", False)]),
        # When it was first used: the use status, its first use.
        (FIRST_USE, [("vaso_ind", "use", True)]),
        ("开始使用血管活性药的时间距入 ICU 多久？", [("vaso_ind", "use", True)]),
        (
            "Among patients who received vasopressors, how many hours after ICU admission "
            "were they first started?",
            [("vaso_ind", "use", True)],
        ),
        ("What is the time to first vasopressor use after ICU admission?", [("vaso_ind", "use", True)]),
        # How much was given: the dose.
        ("入 ICU 第一天血管活性药最大剂量与院内死亡的关联", [("norepi_equiv", "dose", False)]),
        ("Is the vasopressor dose on the first ICU day associated with death?", [("norepi_equiv", "dose", False)]),
        ("Peak vasopressor rate in mcg/kg/min and mortality", [("norepi_equiv", "dose", False)]),
        ("去甲肾上腺素当量与死亡的关系", [("norepi_equiv", "dose", False)]),
        ("年龄 18 岁以上的患者，去甲肾上腺素剂量与 28 天死亡的关系", [("norepi_rate", "dose", False)]),
        # A treatment no concept records the use of keeps its rate, read as a use.
        ("使用去甲肾上腺素的患者占多少？", [("norepi_rate", "use", False)]),
        # A use status read by its own name, or its dose.
        ("使用胰岛素的患者比例", [("insulin", "use", False)]),
        ("胰岛素剂量与低血糖的关系", [("ins", "dose", False)]),
        ("丙泊酚剂量与谵妄", [("propofol_rate", "dose", False)]),
        # Administration or exposure is its use.
        ("Is vasopressor administration associated with in-hospital death?", [("vaso_ind", "use", False)]),
        ("Is exposure to vasopressors associated with in-hospital death?", [("vaso_ind", "use", False)]),
        ("血管活性药暴露与院内死亡的关系", [("vaso_ind", "use", False)]),
        # A rate or a maximum of something else says no dose of the treatment.
        ("使用血管活性药患者的最高乳酸", [("vaso_ind", "use", False)]),
        ("What is the mortality rate of patients who received vasopressors?", [("vaso_ind", "use", False)]),
        # Another treatment's dose is its own.
        ("患者使用血管活性药时的胰岛素剂量", [("vaso_ind", "use", False), ("ins", "dose", False)]),
        (
            "Among patients receiving vasopressors, is the insulin dose associated with death?",
            [("vaso_ind", "use", False), ("ins", "dose", False)],
        ),
        # A dose the question sets aside says no dose (delta review 10-27).
        ("Is vasopressor use rather than dose associated with in-hospital mortality?", [("vaso_ind", "use", False)]),
        (
            "Is any vasopressor use irrespective of dose associated with in-hospital mortality?",
            [("vaso_ind", "use", False)],
        ),
        ("Is vasopressor use (regardless of dose) associated with in-hospital mortality?", [("vaso_ind", "use", False)]),
        ("Is receiving vasopressors at any dose associated with in-hospital mortality?", [("vaso_ind", "use", False)]),
        ("使用血管活性药（无论剂量）是否与院内死亡相关？", [("vaso_ind", "use", False)]),
        ("是否使用血管活性药（不论剂量）与院内死亡相关？", [("vaso_ind", "use", False)]),
        ("使用血管活性药而非其剂量与院内死亡的关系", [("vaso_ind", "use", False)]),
        ("Is vasopressor use, regardless of dose, associated with in-hospital mortality?", [("vaso_ind", "use", False)]),
        ("Is any vasopressor use, not the dose, associated with in-hospital death?", [("vaso_ind", "use", False)]),
        # A count or a grade of something else is no dose (delta review 10-27).
        ("使用血管活性药的患者总量是多少？", [("vaso_ind", "use", False)]),
        ("What amount of patients received vasopressors?", [("vaso_ind", "use", False)]),
        ("Is the amount of time on vasopressors associated with in-hospital mortality?", [("vaso_ind", "use", False)]),
        ("Is vasopressor use associated with mortality in high-intensity units?", [("vaso_ind", "use", False)]),
        (
            "Is vasopressor use associated with in-hospital death at a lower nurse staffing intensity?",
            [("vaso_ind", "use", False)],
        ),
        # The same words of the treatment itself are its dose.
        ("血管活性药总量与院内死亡的关系", [("norepi_equiv", "dose", False)]),
        ("Is the amount of vasopressor given associated with in-hospital mortality?", [("norepi_equiv", "dose", False)]),
        ("Is high-dose vasopressor therapy associated with in-hospital mortality?", [("norepi_equiv", "dose", False)]),
        # A treatment that illustrates a class is not asked of itself.
        ("使用血管活性药（如去甲肾上腺素）的患者占多少？", [("vaso_ind", "use", False)]),
        ("What share of patients received vasopressors (e.g., norepinephrine)?", [("vaso_ind", "use", False)]),
    ],
)
def test_a_treatment_is_read_as_its_use_or_its_dose(question: str, expected: list) -> None:
    assert _readings(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # The lead review's inputs (10-27): each asks the dose, and read the use.
        ("血管活性药使用剂量与院内死亡的关系", [("norepi_equiv", "dose", False)]),
        ("使用血管活性药的患者的最大剂量与院内死亡的关系", [("norepi_equiv", "dose", False)]),
        ("Is in-hospital mortality different based on vasopressor dose?", [("norepi_equiv", "dose", False)]),
        ("Is in-hospital mortality associated with the dose of vasopressor used?", [("norepi_equiv", "dose", False)]),
        (
            "Among patients receiving vasopressors, is the maximum vasopressor dose associated "
            "with in-hospital death?",
            [("norepi_equiv", "dose", False)],
        ),
        ("血管活性药治疗强度与院内死亡的关系", [("norepi_equiv", "dose", False)]),
        ("Is vasopressor therapy intensity associated with in-hospital death?", [("norepi_equiv", "dose", False)]),
        ("Does in-hospital mortality differ depending on norepinephrine dose?", [("norepi_rate", "dose", False)]),
        (
            "Among patients who received norepinephrine, what was the maximum norepinephrine dose?",
            [("norepi_rate", "dose", False)],
        ),
        # The yes/no form 「是不是」 asks the dose; its 不是 sets nothing aside
        # (fix2 review 10-27).
        ("血管活性药是不是剂量越大死亡风险越高？", [("norepi_equiv", "dose", False)]),
        ("去甲肾上腺素是不是剂量越高，28 天死亡率越高？", [("norepi_rate", "dose", False)]),
    ],
)
def test_a_dose_question_is_read_as_the_dose_whatever_use_word_it_also_says(
    question: str, expected: list,
) -> None:
    assert _readings(question) == expected
    assert unrecorded_treatment_uses(question) == ()
    stop_unrecorded_treatment_uses(question)


@pytest.mark.parametrize(
    "question",
    [
        "Is in-hospital mortality associated with vasopressors at any rate?",
        "血管活性药（无论峰值）与院内死亡的关系",
    ],
)
def test_a_set_aside_dose_word_that_decides_only_as_the_nearest_word_says_no_dose(question: str) -> None:
    # Rate and peak also qualify a mortality or a lactate, so they say the dose
    # only as the nearest word; set aside, they say none, and the bare
    # treatment reads its use (fix2 review 10-27).
    assert _readings(question) == []
    assert deterministic_intent(question)["slots"]["exposure"]["value"] == "vaso_ind"


@pytest.mark.parametrize(
    "question",
    [
        # Neither use nor dose: no form stated (the slot reads its use).
        "血管活性药与院内死亡的关联",
        "Vasopressors and ICU mortality",
        "胰岛素与低血糖的关系",
        # A study the question declines.
        "不研究血管活性药，只看乳酸与死亡",
        # Nothing of a treatment.
        "乳酸与院内死亡的关联",
    ],
)
def test_words_that_say_neither_use_nor_dose_read_no_treatment_form(question: str) -> None:
    assert _readings(question) == []


@pytest.mark.parametrize(
    "question",
    [
        # A use within a window is no question of when it began.
        "入 ICU 后 6 小时内开始使用血管活性药的患者比例",
        # A time that is another concept's (住院时间 is the hospital stay).
        "使用血管活性药的患者住院时间是否更长？",
        "Patients who received vasopressors within the first 24 hours: mortality?",
    ],
)
def test_a_use_with_no_question_of_its_start_is_no_first_use(question: str) -> None:
    assert all(not first_use for _concept, _form, first_use in _readings(question))


def test_the_slot_reader_reads_the_use_status_the_dose_or_a_bare_treatment_as_its_use() -> None:
    use = deterministic_intent("在 AmsterdamUMCdb 中，入 ICU 第一天使用血管活性药的患者占多少？")
    assert use["slots"]["exposure"]["value"] == "vaso_ind"
    dose = deterministic_intent("入 ICU 第一天血管活性药最大剂量与院内死亡的关联")
    assert dose["slots"]["exposure"]["value"] == "norepi_equiv"
    # A treatment named as an exposure with neither word is its use (lead
    # review 10-27: the exposure was left unread).
    for bare in ("血管活性药与院内死亡的关联", "血管活性药与院内死亡的关系", "Vasopressors and ICU mortality"):
        assert deterministic_intent(bare)["slots"]["exposure"]["value"] == "vaso_ind"
    assert named_study_concepts("血管活性药与院内死亡的关联") == ((("vaso_ind",), "血管活性药"),)
    # A later mention that says the form decides the one that did not.
    decided = "血管活性药与院内死亡：比较使用与未使用血管活性药的患者"
    assert deterministic_intent(decided)["slots"]["exposure"]["value"] == "vaso_ind"


@pytest.mark.parametrize(
    ("question", "evidence"),
    [
        ("入 ICU 第一天使用与未使用血管活性药的患者，院内死亡率有何不同？", "使用与未使用血管活性药"),
        ("Mortality in patients who did and did not receive vasopressors", "did and did not receive vasopressors"),
    ],
)
def test_used_and_not_used_groups_are_the_use_status(question: str, evidence: str) -> None:
    assert [(item.concept_id, item.evidence) for item in stated_comparison_groupings(question)] == [
        ("vaso_ind", evidence),
    ]


def test_a_dictionary_name_read_with_use_or_dose_words_names_that_concept() -> None:
    used = catalog_named_concepts("使用胰岛素的患者比例")
    assert "insulin" in used and "ins" not in used
    dosed = catalog_named_concepts("胰岛素剂量的中位数")
    assert "ins" in dosed and "insulin" not in dosed
    # The name alone is the treatment's use, as the slot reader reads it.
    named = catalog_named_concepts("胰岛素与死亡")
    assert "insulin" in named and "ins" not in named


def _coordinates(question: str, family: Any = None, database: str = "aumc") -> dict[str, Any]:
    return launch._metadata_only_planning_coordinates(
        question=question, database=database, analysis_family=family,
    )


def test_the_aumc_first_use_question_reads_the_use_status_and_its_onset() -> None:
    coordinates = _coordinates(FIRST_USE, DESCRIPTIVE)
    endpoint = coordinates["endpoint"]
    assert coordinates["target_outcome"] == "vaso_ind"
    assert (endpoint.name, endpoint.kind) == ("vaso_ind_onset_time", "continuous")
    assert coordinates["target_outcome_window"] == {
        "anchor": "icu_admission", "start_hours": 0.0, "end_hours": 24.0,
    }
    assert [(item["role"], item["concept"], item["column"]) for item in coordinates["coordinate_readings"]] == [
        ("outcome", "vaso_ind", "vaso_ind_onset_time"),
    ]
    # Without a descriptive design the treatment is the exposure, its use.
    assert _coordinates(FIRST_USE)["primary_exposure"] == "vaso_ind"


def test_a_share_question_reads_the_use_status_and_a_dose_question_the_dose() -> None:
    share = _coordinates("在 AmsterdamUMCdb 中，入 ICU 第一天使用血管活性药的患者占多少？", DESCRIPTIVE)
    assert share["target_outcome"] == "vaso_ind"
    assert (share["endpoint"].name, share["endpoint"].kind) == ("vaso_ind", "binary")
    grouped = _coordinates("入 ICU 第一天使用与未使用血管活性药的患者，院内死亡率有何不同？", DESCRIPTIVE)
    assert (grouped["primary_exposure"], grouped["target_outcome"]) == ("vaso_ind", "death")
    dose = _coordinates("入 ICU 第一天血管活性药最大剂量与院内死亡的关联？")
    assert (dose["primary_exposure"], dose["primary_exposure_aggregation"]) == ("norepi_equiv", "max")


def test_the_onset_of_first_use_is_a_reportable_descriptive_summary() -> None:
    from easyicu.research_agent.planning.capability_registry import assess_scientific_capability
    from easyicu.research_agent.planning.progressive_compiler import compile_progressive_plan
    from easyicu.research_agent.planning.progressive_contract import ProgressivePlanSkeleton
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    coordinates = _coordinates(FIRST_USE, DESCRIPTIVE)
    endpoint = coordinates["endpoint"]
    frame = pd.DataFrame({
        "stay_id": pd.Series(dtype="int64"),
        "vaso_ind": pd.Series(dtype="float64"),
        endpoint.name: pd.Series(dtype="float64"),
    })
    frame.attrs["easyicu_planning_authority"] = {
        "kind": "metadata_only_planning_catalog", "patient_rows_read": False,
    }
    built = build_research_context(
        research_question=FIRST_USE, cohort=frame, cohort_name="synthetic", database="aumc",
        id_columns=["stay_id"], target_outcome=endpoint.name, endpoint=endpoint,
        primary_exposure=None,
        time_windows=[TimeWindow(name="bound", anchor="icu_admission", start_hours=0, end_hours=24)],
        user_preferences={"inferred_analysis_family": DESCRIPTIVE, "data_constraints": json.dumps({
            "materialization_window": {"role": "outer_observation_window", "anchor": "icu_admission", "hours": 24},
        })},
    )
    context = getattr(built, "context", built)
    assert context.target_outcome == "vaso_ind_onset_time"
    skeleton = ProgressivePlanSkeleton.model_validate({
        "analysis_type": DESCRIPTIVE, "rationale": "Describe when the treatment was first used.",
        "cohort": {"name": "synthetic", "selection_mode": "all_input_rows"},
        "display_labels": [{"key": name, "value": name} for name in ("vaso_ind", endpoint.name)],
        "steps": [
            {"step_id": "cohort", "module_id": "cohort_definition", "planned_analysis_role": "auxiliary",
             "objective": "Account for the synthetic stay denominator.", "raw_inputs": ["stay_id"]},
            {"step_id": "onset", "module_id": "custom_analysis", "planned_analysis_role": "primary",
             "objective": "Describe the hours from ICU admission to the first use.",
             "depends_on": ["cohort"], "raw_inputs": [endpoint.name],
             "product_inputs": [{"producer_step_id": "cohort", "product_id": "artifact:analysis_cohort"}],
             "scientific_action_id": "descriptive.descriptive_summary", "custom_method": "empirical_summary",
             "outputs": [{"product_id": product, "semantic_role": "custom"}
                         for product in ("table:cohort_summary", "table:distribution_summary")]},
        ],
    })
    plan, _ = compile_progressive_plan(skeleton=skeleton, context=context)
    review = build_plan_scientific_review(context=context, plan=plan, require_reportable_capability=True)
    assert assess_scientific_capability(
        analysis_type=plan.analysis_type, plan=plan, context=context,
    ).claim_ceiling == "reportable"
    codes = {finding.code for finding in review.findings}
    assert not codes & {
        "SCIENTIFIC_CAPABILITY_NOT_REPORTABLE", "REQUESTED_OUTCOME_COVERAGE_INCOMPLETE",
        "OUTCOME_DEFINITION_UNRESOLVED",
    }


# --------------------------------------------- a use the data do not record ---
def test_a_use_of_a_treatment_recorded_only_as_a_rate_stops() -> None:
    question = "在 AmsterdamUMCdb 中，入 ICU 第一天使用去甲肾上腺素的患者占多少？"
    assert [item.concept_id for item in unrecorded_treatment_uses(question)] == ["norepi_rate"]
    with pytest.raises(ResearchPipelineRunError) as stopped:
        stop_unrecorded_treatment_uses(question)
    error = stopped.value
    assert error.code == TREATMENT_USE_UNRECORDED
    assert error.details["concept"] == "norepi_rate"
    assert error.details["treatment_uses"][0]["form"] == "use"
    assert "dose" in str(error)


@pytest.mark.parametrize(
    "question",
    [
        # Its dose is recorded.
        "去甲肾上腺素剂量与 28 天死亡的关系",
        # A use status records it.
        "在 AmsterdamUMCdb 中，入 ICU 第一天使用血管活性药的患者占多少？",
        # A rate-only treatment that illustrates a recorded class.
        "使用血管活性药（如去甲肾上腺素）的患者占多少？",
        # A target trial reads its treatment from the capture registry.
        "以入 ICU 后第 6 小时为时间零点，比较 6 小时内开始去甲肾上腺素与这 6 小时内不开始的 28 天死亡。",
        "",
    ],
)
def test_a_dose_a_recorded_use_or_a_trial_is_not_stopped(question: str) -> None:
    stop_unrecorded_treatment_uses(question)


def test_a_study_with_a_causal_design_is_left_to_its_trial() -> None:
    stop_unrecorded_treatment_uses(
        "使用去甲肾上腺素是否降低死亡？",
        study={"analysis_design": {"analysis_family": "causal_inference"}},
    )


def _request(tmp_path: Path, question: str) -> ResearchPipelineLaunchRequest:
    return ResearchPipelineLaunchRequest(
        export_path=str(tmp_path / "export"),
        study_context={
            "id": "study-launch", "revision": 5, "question": question,
            "data_source": {"path": str(tmp_path / "export"), "database": "aumc"},
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


def test_the_launch_stops_before_anything_is_written(tmp_path, monkeypatch) -> None:
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        context_store, "upsert_context", lambda raw, **kwargs: writes.append(dict(raw)) or dict(raw),
    )
    with pytest.raises(ResearchPipelineRunError) as stopped:
        prepare_research_pipeline_run(_request(tmp_path, "What share of patients received norepinephrine?"))
    assert stopped.value.code == TREATMENT_USE_UNRECORDED
    assert writes == []
    assert not (tmp_path / "workspace").exists()


def test_the_stop_offers_an_explicit_plan_edit() -> None:
    assert stop_next_actions.STOP_POLICIES[TREATMENT_USE_UNRECORDED] == "edit_plan"
    actions = stop_next_actions.next_actions_for_stop(TREATMENT_USE_UNRECORDED, detail={})
    assert [action.kind for action in actions] == ["edit_plan"]
    assert actions[0].params.requires_explicit_choice is True


@pytest.mark.parametrize(("language", "expected"), [("en", "dose"), ("zh", "剂量")])
def test_the_conversation_explains_the_stop_in_a_sentence(language: str, expected: str) -> None:
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
    result = run_node("node", script, str(path), language, TREATMENT_USE_UNRECORDED, check=True)
    text = json.loads(result.stdout)
    assert expected in text
    assert TREATMENT_USE_UNRECORDED not in text
