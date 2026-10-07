"""An event that qualifies the study, rather than timing it, is no time zero.

A question's event phrase is its time zero when the study counts time from it
("the first 24 hours after sepsis onset", "within 6 hours of intubation").
The reader took every phrase in its event vocabulary as one, so a study whose
windows count from ICU admission was stopped before planning when a bare
mention of the event only qualified it:

- the population: "patients admitted to the ICU after cardiac arrest",
  "in patients after ROSC", "心脏骤停后入ICU的患者";
- the time elapsed since an event, a variable rather than a window:
  "adjusting for hours since sepsis onset", "the time from intubation to
  extubation", "调整自疑似感染起的时间";
- a negation or an exclusion: "not after intubation", "excluding values
  measured after intubation", "排除插管后测得的值".

A duration counted from an event is a window wherever it stands, and a phrase
that may state the exposure ("patients who received steroids after septic
shock onset") or a covariate measured from an event is still read, so such a
study still stops before planning.  An hour count inside any event phrase is
never a window from ICU admission, and an outcome counted from an event (a
death within 28 days of ICU discharge) still states the study's time zero:
every EasyICU follow-up counts from an admission.  Synthetic contexts only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.gates.preplan import (
    clinical_time_authority_findings,
    preplan_data_failure_reason,
)
from easyicu.research_agent.research_context.temporal_semantics import (
    TemporalAlignmentEngine,
    TimeWindowSemanticParser,
    event_anchored_spans,
    stated_event_time_zeros,
)
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _study(question: str) -> ResearchContext:
    """A first-day exposure whose window counts from ICU admission."""

    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="synthetic ICU stays",
            database="synthetic",
            n_stays=0,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            ConceptDescriptor(
                name="lab_max",
                role=VariableRole.LAB,
                dtype="float64",
                unit="mmol/L",
                source_concept="lab",
                analysis_window="icu_admission[0,24]h",
                analysis_window_role="outer_observation_window",
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="float64"),
        ],
        target_outcome="death",
        primary_exposure="lab_max",
    )


def _kinds(question: str) -> list[str]:
    return [
        item.detail["kind"]
        for item in clinical_time_authority_findings(_study(question))
    ]


_QUALIFYING = [
    # The population.
    "Among patients admitted to the ICU after cardiac arrest, is the highest lab "
    "value in the first 24 hours associated with in-hospital mortality?",
    "Among patients admitted to the intensive care unit after cardiac arrest, is "
    "the first-day lab value associated with death?",
    "Among ICU admissions after cardiac arrest, is the first-day lab value "
    "associated with death?",
    "In patients after ROSC, is the lowest lab value in the first 24 hours "
    "associated with death?",
    "Among post-cardiac arrest patients, is the first-day lab value associated "
    "with death?",
    "心脏骤停后入ICU的患者中，首24小时最高检验值与院内死亡是否相关？",
    "在自主循环恢复后的患者中，首日检验值与死亡是否相关？",
    # The time elapsed since an event.
    "Is the first-day lab value associated with mortality, adjusting for hours "
    "since sepsis onset?",
    "Is the first-day lab value associated with death after controlling for the "
    "time from suspected infection onset?",
    "Is the first-day lab value associated with the time from intubation to "
    "extubation?",
    "调整自疑似感染起的时间后，首日检验值与死亡是否相关？",
    # A negation or an exclusion.
    "Is the lab value (not after intubation) in the first 24 hours associated "
    "with death?",
    "Is the first-day lab value, excluding values measured after intubation, "
    "associated with death?",
    "排除插管后测得的检验值后，首日检验值与死亡是否相关？",
]


@pytest.mark.parametrize("question", _QUALIFYING)
def test_an_event_that_qualifies_the_study_states_no_time_zero(question) -> None:
    assert stated_event_time_zeros(question) == ()
    # The phrase still counts from its event.
    assert event_anchored_spans(question)
    assert _kinds(question) == []


@pytest.mark.parametrize(
    ("question", "anchor", "hours"),
    [
        (
            "Is the lab value within 6 hours of intubation associated with death?",
            "intubation",
            6.0,
        ),
        (
            "Is the lab value after cardiac arrest associated with death?",
            "cardiac_arrest",
            None,
        ),
        ("Is the post-intubation lab value associated with death?", "intubation", None),
        (
            "Is the highest lab value in the first 24 hours after sepsis onset "
            "associated with 28-day mortality?",
            "sepsis_onset",
            24.0,
        ),
        (
            "Among patients admitted to the ICU, is the lab value within 6 hours "
            "after ROSC associated with death?",
            "rosc",
            6.0,
        ),
        ("自主循环恢复后的检验值与死亡是否相关？", "rosc", None),
        ("插管后第一天的检验值与死亡是否相关？", "intubation", 24.0),
        ("脓毒症发生后首日的检验值与死亡是否相关？", "sepsis_onset", 24.0),
    ],
)
def test_an_event_the_study_counts_from_is_still_its_time_zero(
    question, anchor, hours
) -> None:
    [statement] = stated_event_time_zeros(question)

    assert (statement.anchor, statement.hours) == (anchor, hours)
    findings = clinical_time_authority_findings(_study(question))
    assert [item.detail["kind"] for item in findings] == [
        "primary_exposure_time_anchor_unverified"
    ]
    assert preplan_data_failure_reason(findings) == "clinical_time_authority_failed"


@pytest.mark.parametrize(
    ("question", "anchor", "hours"),
    [
        # A relative clause may state the exposure, not the population.
        (
            "Do patients who received steroids after septic shock onset have "
            "lower mortality?",
            "septic_shock_onset",
            None,
        ),
        (
            "Among patients who were intubated after cardiac arrest, is the "
            "first-day lab value associated with death?",
            "cardiac_arrest",
            None,
        ),
        # An admission reaches the event across a short place only.
        (
            "Among patients admitted to the ICU and treated with steroids after "
            "septic shock onset, is mortality lower?",
            "septic_shock_onset",
            None,
        ),
        # "Admission" names a time, not a population.
        (
            "Is the lab value measured at ICU admission and after intubation "
            "associated with death?",
            "intubation",
            None,
        ),
        # A duration counted from the event is a window wherever it stands.
        (
            "Is the lab value measured in patients within 6 hours after ROSC "
            "associated with death?",
            "rosc",
            6.0,
        ),
        (
            "Is the first-day lab value, excluding values measured within 6 hours "
            "after intubation, associated with death?",
            "intubation",
            6.0,
        ),
        # The first hours since an event are a window, not an elapsed time.
        (
            "Is the lab value in the first hours since sepsis onset associated with "
            "death?",
            "sepsis_onset",
            None,
        ),
        (
            "Is the lab value from intubation to ICU discharge associated with death?",
            "intubation",
            None,
        ),
        # A covariate measured from an event is not the time elapsed since it.
        (
            "Is the first-day lab value associated with mortality, adjusting for "
            "SOFA after sepsis onset?",
            "sepsis_onset",
            None,
        ),
        # Words outside the event's own clause neither adjust nor negate it.
        (
            "After adjusting for age is the lab value after intubation associated "
            "with death?",
            "intubation",
            None,
        ),
        (
            "Is the lab value not associated with death after intubation?",
            "intubation",
            None,
        ),
        ("插管后的时间窗内检验值与死亡是否相关？", "intubation", None),
        ("自插管起的时间窗内检验值与死亡是否相关？", "intubation", None),
        ("插管后入量与死亡是否相关？", "intubation", None),
        ("不同插管后检验值水平的患者死亡率是否不同？", "intubation", None),
    ],
)
def test_a_phrase_that_may_time_the_study_is_still_read(
    question, anchor, hours
) -> None:
    [statement] = stated_event_time_zeros(question)

    assert (statement.anchor, statement.hours) == (anchor, hours)
    assert _kinds(question) == ["primary_exposure_time_anchor_unverified"]


@pytest.mark.parametrize(
    ("question", "anchor", "hours"),
    [
        # An outcome counted from discharge: no owner follows a study from
        # discharge, and the horizon reader would count "28 days" from ICU
        # admission.
        (
            "Is the first-day lab value associated with death within 28 days of "
            "ICU discharge?",
            "icu_discharge",
            672.0,
        ),
        ("首日检验值与出ICU后28天内死亡是否相关？", "icu_discharge", 672.0),
        (
            "Is the first-day lab value associated with 30-day readmission after "
            "hospital discharge?",
            "hospital_discharge",
            None,
        ),
        (
            "Is the first-day lab value associated with mortality after ICU discharge?",
            "icu_discharge",
            None,
        ),
        ("首日检验值与出院后再入院是否相关？", "hospital_discharge", None),
        # An exclusion timed from an event: no window from ICU admission
        # carries it.
        (
            "Is the first-day lab value associated with death, excluding patients "
            "who died within 24 hours of intubation?",
            "intubation",
            24.0,
        ),
        ("排除插管后24小时内死亡者后，首日检验值与死亡是否相关？", "intubation", 24.0),
    ],
)
def test_a_study_no_owner_can_time_still_stops_before_planning(
    question, anchor, hours
) -> None:
    [statement] = stated_event_time_zeros(question)

    assert (statement.anchor, statement.hours) == (anchor, hours)
    assert _kinds(question) == ["primary_exposure_time_anchor_unverified"]


def test_a_population_event_leaves_the_studys_own_time_zero_in_place() -> None:
    question = (
        "Among patients admitted after cardiac arrest, is the lab value within 6 "
        "hours of intubation associated with death?"
    )

    [statement] = stated_event_time_zeros(question)

    assert statement.anchor == "intubation"
    assert len(event_anchored_spans(question)) == 2


@pytest.mark.parametrize(
    "question",
    [
        "Is the first-day lab value, excluding values measured within 6 hours "
        "after intubation, associated with death?",
        "Is the highest lab value in the first 24 hours associated with death, "
        "excluding the first 6 hours after intubation?",
    ],
)
def test_hours_counted_from_an_event_are_no_window_from_icu_admission(question) -> None:
    constraints = TimeWindowSemanticParser().parse(question)
    windows, _ = TemporalAlignmentEngine().infer(research_question=question)

    assert not any(
        item.relation == "first_window" and item.end_hours == 6.0
        for item in constraints
    )
    # Only a duration that counts from ICU admission is an ICU window.
    assert all(window.end_hours != 6.0 for window in windows)


def test_an_event_anchored_span_is_where_the_phrase_is() -> None:
    question = (
        "Among patients admitted after cardiac arrest, is the lab value within 6 "
        "hours of intubation associated with death?"
    )

    spans = event_anchored_spans(question)

    assert [question[start:end] for start, end in spans] == [
        "after cardiac arrest",
        "within 6 hours of intubation",
    ]
    assert event_anchored_spans("") == ()
    assert event_anchored_spans("Is the lab value associated with death?") == ()
