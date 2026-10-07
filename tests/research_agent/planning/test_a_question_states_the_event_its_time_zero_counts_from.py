"""A question states the clinical event its time zero counts from.

Every materialized window and follow-up counts from an admission.  A question
that counts from another event ("the first 24 hours after sepsis onset",
"within 6 hours of intubation", "脓毒症发生后24小时内") declared no time
zero, so the study was planned silently from ICU admission, and "the first 24
hours after sepsis onset" even became a 0-24 h window from ICU admission.  The
question's event is now its declared time zero, read from a closed event
vocabulary, so the pre-Provider gate stops a study its windows cannot honour.

The opposite error is fixed too: a question that counted "from ICU admission"
was stopped as unverified although its exposure's window counts from that
admission.  An admission is no clinical definition; a disease or event anchor
still needs the owner-issued definition.

Synthetic, case-neutral contexts only.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.gates.preplan import (
    clinical_time_authority_findings,
    preplan_data_failure_reason,
)
from easyicu.research_agent.planning.adjustment_authority import (
    host_outer_feature_window_end_hours,
    primary_landmark_hours,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    plan_time_zero_hours,
)
from easyicu.research_agent.reporting.scientific_maturity import (
    build_scientific_maturity_audit,
)
from easyicu.research_agent.research_context.temporal_semantics import (
    TemporalAlignmentEngine,
    TimeWindowSemanticParser,
    primary_exposure_time_anchor_alignment,
    stated_event_time_zeros,
    study_time_origin_alignment,
)
from easyicu.research_agent.schema import (
    ClinicalDefinitionReference,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

from .scientific_review_fixtures import _context, _literature, _plan

_ICU_WINDOW = "icu_admission[0,24]h"


def _study(
    question: str,
    *,
    window: str = _ICU_WINDOW,
    exposure: bool = True,
    definition_anchor: str | None = None,
    constraints: bool = False,
) -> ResearchContext:
    """A window-summary exposure (or none) whose window counts from ``window``."""

    lab = ConceptDescriptor(
        name="lab_max",
        role=VariableRole.LAB,
        dtype="float64",
        unit="mmol/L",
        source_concept="lab",
        analysis_window=window,
        analysis_window_role="outer_observation_window",
        clinical_definition=(
            ClinicalDefinitionReference(
                contract_id="synthetic_contract",
                definition="Synthetic phenotype",
                version="1",
                source_id="PMID:1",
                definition_time_anchor=definition_anchor,
                status="source_bound_golden",
                validation_status="automated_golden",
                canonical_definition=True,
            )
            if definition_anchor is not None
            else None
        ),
    )
    _windows, parsed = TemporalAlignmentEngine().infer(research_question=question)
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="synthetic ICU stays", database="synthetic", n_stays=0,
            id_columns=["stay_id"], outcome_columns=["death_28d"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="int64"),
            lab,
            ConceptDescriptor(name="death_28d", role=VariableRole.OUTCOME, dtype="float64"),
        ],
        temporal_constraints=parsed if constraints else [],
        target_outcome="death_28d" if exposure else None,
        primary_exposure="lab_max" if exposure else None,
    )


def _kinds(context: ResearchContext) -> list[str]:
    return [item.detail["kind"] for item in clinical_time_authority_findings(context)]


@pytest.mark.parametrize(
    ("question", "anchor", "hours"),
    [
        ("Is the highest lab value in the first 24 hours after sepsis onset associated "
         "with 28-day mortality?", "sepsis_onset", 24.0),
        ("Is the lowest lab value within 6 hours of intubation associated with death?",
         "intubation", 6.0),
        ("Does the lab value on the first day of mechanical ventilation predict death?",
         "mechanical_ventilation_start", 24.0),
        ("Is the lab value within 6 h of the onset of septic shock associated with death?",
         "septic_shock_onset", 6.0),
        ("Is the lab value 48 h after AKI onset associated with death?", "aki_onset", 48.0),
        ("Is the lab value after vasopressor initiation associated with death?",
         "vasopressor_start", None),
        ("Is the lab value since the start of renal replacement therapy associated with "
         "death?", "rrt_start", None),
        ("Is the post-intubation lab value associated with death?", "intubation", None),
        ("Is the lab value at ROSC associated with death?", "rosc", None),
        ("Is the lab value after cardiac arrest associated with death?", "cardiac_arrest", None),
        ("Is the lab value associated with death within 30 days after hospital discharge?",
         "hospital_discharge", 720.0),
        ("脓毒症发生后24小时内的最高检验值与28天死亡率是否相关？", "sepsis_onset", 24.0),
        ("插管后6小时内的检验值与死亡是否相关？", "intubation", 6.0),
        ("开始机械通气后的检验值与死亡是否相关？", "mechanical_ventilation_start", None),
        ("感染性休克发生后最初24小时的检验值与死亡是否相关？", "septic_shock_onset", 24.0),
        ("启动肾脏替代治疗后的检验值与死亡是否相关？", "rrt_start", None),
        ("自主循环恢复后的检验值与死亡是否相关？", "rosc", None),
        ("出院后30天内的死亡与检验值是否相关？", "hospital_discharge", 720.0),
    ],
)
def test_a_question_counting_from_an_event_declares_it_and_stops_before_planning(
    question, anchor, hours
) -> None:
    [statement] = stated_event_time_zeros(question)
    assert (statement.anchor, statement.hours) == (anchor, hours)

    context = _study(question)
    alignment = primary_exposure_time_anchor_alignment(context)
    assert (alignment.status, alignment.declared_anchor, alignment.declared_source) == (
        "declared_only", anchor, "research_question.stated_event_time_zero",
    )
    findings = clinical_time_authority_findings(context)
    assert [item.detail["kind"] for item in findings] == [
        "primary_exposure_time_anchor_unverified"
    ]
    assert preplan_data_failure_reason(findings) == "clinical_time_authority_failed"


@pytest.mark.parametrize(
    "question",
    [
        # A population, a status, or a therapy's duration is no time zero.
        "Among patients with septic shock, is the lab value associated with death?",
        "Is the lab value associated with mortality after sepsis?",
        "Is the lab value associated with mortality at discharge?",
        "Among patients who received at least 6 h of vasopressor therapy, is the lab "
        "value associated with death?",
        "Is the worst lab value before intubation associated with death?",
        "Is the lab value associated with AKI after cardiac surgery?",
        "Is the highest lab value in the first 24 hours after ICU admission associated "
        "with death?",
        "感染性休克患者的检验值与死亡是否相关？",
        "脓毒症后遗症患者的检验值与死亡是否相关？",
        "插管前的检验值与死亡是否相关？",
        "入ICU后24小时内的检验值与死亡是否相关？",
    ],
)
def test_a_population_status_or_duration_states_no_time_zero(question) -> None:
    assert stated_event_time_zeros(question) == ()
    assert _kinds(_study(question)) == []


def test_the_first_hours_after_an_event_are_not_a_window_from_icu_admission() -> None:
    question = "Is the highest lab value in the first 24 hours after sepsis onset associated with death?"

    constraints = TimeWindowSemanticParser().parse(question)
    windows, _ = TemporalAlignmentEngine().infer(research_question=question)

    assert [(item.relation, item.anchor_event, item.start_hours, item.end_hours)
            for item in constraints] == [("after_event", "sepsis_onset", 0.0, 24.0)]
    assert windows == []
    # A bare duration still counts from ICU admission.
    bare, _ = TemporalAlignmentEngine().infer(
        research_question="Is the highest lab value in the first 24 hours associated with death?"
    )
    assert [(window.name, window.anchor) for window in bare] == [("first_24h", "icu_admission")]


def test_an_exposure_counted_from_the_declared_admission_is_aligned() -> None:
    question = (
        "Is the highest lab value in the first 24 hours from ICU admission associated "
        "with death?"
    )

    alignment = primary_exposure_time_anchor_alignment(_study(question))

    assert (alignment.status, alignment.declared_anchor, alignment.definition_anchor) == (
        "aligned", "icu_admission", "icu_admission",
    )
    assert alignment.definition_source == "variables.lab_max.analysis_window"
    assert _kinds(_study(question)) == []
    # A window from another admission does not carry it.
    assert _kinds(_study(question, window="hospital_admission[0,24]h")) == [
        "primary_exposure_time_anchor_unverified"
    ]


def test_an_outer_window_from_the_event_is_no_definition_of_it() -> None:
    # Only an admission is carried by the window that counts from it.
    context = _study(
        "Is the lab value from suspected infection onset associated with death?",
        window="suspected_infection_onset[0,24]h",
    )

    alignment = primary_exposure_time_anchor_alignment(context)

    assert (alignment.status, alignment.declared_anchor) == (
        "declared_only", "suspected_infection_onset",
    )
    assert _kinds(context) == ["primary_exposure_time_anchor_unverified"]


def test_an_owner_issued_definition_still_decides_a_disease_anchor() -> None:
    question = (
        "Is the highest lab value in the first 24 hours from ICU admission associated "
        "with death?"
    )
    context = _study(question, definition_anchor="suspected_infection_onset")

    alignment = primary_exposure_time_anchor_alignment(context)

    assert (alignment.status, alignment.definition_anchor) == (
        "mismatch", "suspected_infection_onset",
    )
    assert _kinds(context) == ["primary_exposure_time_anchor_mismatch"]


@pytest.mark.parametrize(
    "question",
    [
        "Is sepsis within 24 h after suspected infection onset associated with 28-day "
        "mortality?",
        "Is sepsis in the first 24 hours after suspected infection onset associated with "
        "28-day mortality?",
    ],
)
def test_an_aligned_event_duration_is_no_landmark_or_window_on_the_icu_axis(question) -> None:
    # The exposure's definition counts from the declared event, so the study
    # passes; the event's 24 hours never become hours after ICU admission.
    windows, constraints = TemporalAlignmentEngine().infer(research_question=question)
    context = _study(question, definition_anchor="suspected_infection_onset").model_copy(
        update={"time_windows": windows, "temporal_constraints": constraints}
    )

    alignment = primary_exposure_time_anchor_alignment(context)

    assert (alignment.status, alignment.declared_anchor) == (
        "aligned", "suspected_infection_onset",
    )
    assert _kinds(context) == []
    assert windows == []
    assert [(item.relation, item.anchor_event, item.end_hours) for item in constraints] == [
        ("after_event", "suspected_infection_onset", 24.0)
    ]
    assert primary_landmark_hours(context) is None
    assert host_outer_feature_window_end_hours(context) is None
    assert plan_time_zero_hours(context, None, None) is None


def test_an_event_beside_an_admission_is_the_time_zero_and_the_first_event_wins() -> None:
    beside = _study(
        "Is the lab value within 6 hours of intubation associated with 28-day mortality "
        "from ICU admission?"
    )
    two = _study(
        "Is the lab value after vasopressor initiation associated with death within "
        "48 h of AKI onset?"
    )
    admissions = _study(
        "Is the lab value from ICU admission associated with death counted from "
        "hospital admission?"
    )

    assert primary_exposure_time_anchor_alignment(beside).declared_anchor == "intubation"
    assert _kinds(beside) == ["primary_exposure_time_anchor_unverified"]
    assert primary_exposure_time_anchor_alignment(two).declared_anchor == "vasopressor_start"
    # Two admissions alone stay unresolved, as before.
    assert primary_exposure_time_anchor_alignment(admissions).status == "unspecified"


def test_the_typed_constraints_carry_the_event() -> None:
    context = _study(
        "Is the lab value within 6 hours of intubation associated with death?",
        constraints=True,
    )

    alignment = primary_exposure_time_anchor_alignment(context)

    assert (alignment.declared_anchor, alignment.declared_source) == (
        "intubation", "temporal_constraints.after_event",
    )


def test_a_study_without_an_exposure_compares_the_event_with_its_windows() -> None:
    context = _study(
        "Describe the lab value over the first 6 hours after intubation.", exposure=False
    )

    origin = study_time_origin_alignment(context)

    assert (origin.status, origin.declared_anchor, origin.window_anchors) == (
        "mismatch", "intubation", ("icu_admission",),
    )
    assert _kinds(context) == ["study_time_zero_mismatch"]


def _codes(context: ResearchContext, tmp_path) -> set[str]:
    plan = _plan()
    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    audit = build_scientific_maturity_audit(context=context, plan=plan, run_dir=tmp_path)
    return {item.code for item in review.findings} | {item.code for item in audit.findings}


@pytest.mark.parametrize(
    ("question", "code"),
    [
        ("Among adult ICU stays, is an exposure within 6 hours after sepsis onset "
         "associated with in-hospital mortality?", "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH"),
        ("在成人 ICU 住院中，插管后6小时内的暴露与院内死亡是否相关？",
         "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH"),
    ],
)
def test_an_event_time_zero_never_reaches_planning_on_the_icu_axis(
    question, code, tmp_path
) -> None:
    # The plan's time zero, cohort eligibility and event-time rules count hours
    # from ICU admission: the gate stops first, and the review and the audit
    # name the same blocker.
    context = _context().model_copy(update={"research_question": question})

    findings = clinical_time_authority_findings(context)

    assert preplan_data_failure_reason(findings) == "clinical_time_authority_failed"
    assert code in _codes(context, tmp_path)


def test_an_admission_time_zero_has_no_time_anchor_finding(tmp_path) -> None:
    context = _context().model_copy(
        update={
            "research_question": (
                "Among adult ICU stays, is an exposure in the first 24 hours from ICU "
                "admission associated with in-hospital mortality?"
            )
        }
    )

    assert clinical_time_authority_findings(context) == []
    assert not {
        "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH",
        "PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED",
        "STUDY_TIME_ZERO_MISMATCH",
    } & _codes(context, tmp_path)
