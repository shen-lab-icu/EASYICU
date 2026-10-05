"""A trajectory study keeps the window its question states.

A question can state the window its trajectories run over ("first-24h
trajectories", "trajectories over the first 72 hours", "首24小时…轨迹").  The
review publishes that window beside the coordinates, and the Host compiles
both into the study's fixed-window design.  Before, the design always took the
owner's default 0-72 h on a 12 h grid, so a 24-hour question was planned and
run over 72 hours and nothing said so.  A window counted from another event,
several windows, or one no grid divides are not a fixed-window design: the
review states the limitation instead of falling back to the default.  A window
elsewhere in the question (a cohort criterion, an outcome horizon, a
measurement) is not the trajectories'.  Fixtures are generic.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.contracts.trajectory_design import (
    FIXED_WINDOW_TRAJECTORY_DEFAULTS,
    TRAJECTORY_PRIMARY_ACTION,
    normalize_trajectory_design,
    trajectory_window_design,
)
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    remediation_route_for_finding,
)
from easyicu.research_agent.research_context.temporal_semantics import (
    TrajectoryWindowStatement,
    trajectory_window_statements,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)
from easyicu.webserver.pi_copilot.plan_decisions import (
    PlanDecisionError,
    agent_plan_configuration_available,
    compile_agent_plan_configuration,
)
from easyicu.webserver.trajectory_runtime_projection import (
    validate_trajectory_design_declaration,
)

_ICU = "icu_admission"
_OWNER = "TRAJECTORY_LONGITUDINAL_OWNER_NOT_SEALED"
_WINDOW = ("window_start_hours", "window_end_hours", "grid_width_hours")
_STUDY = {
    "cohort": {"preset": "all_icu"},
    "confirmations": {"feature_time_window": True},
    "sensitivity_specs": [],
}
_ROLES = {
    "stay_id": VariableRole.ID,
    "sofa2_resp": VariableRole.ORDINAL_SCORE,
    "sofa2_cardio": VariableRole.ORDINAL_SCORE,
    "lact": VariableRole.LAB,
    "death": VariableRole.OUTCOME,
}


@pytest.mark.parametrize(
    ("question", "stated"),
    [
        (
            "Among adults receiving vasopressors, do first-24h physiologic "
            "trajectories form distinct classes?",
            [(24.0, _ICU)],
        ),
        ("Do first 48 hours SOFA-2 trajectories cluster into classes?", [(48.0, _ICU)]),
        (
            "Which classes do trajectories of SOFA-2 components and lactate over "
            "the first 72 h of the ICU stay form?",
            [(72.0, _ICU)],
        ),
        (
            "Do organ-dysfunction trajectories during the first 3 days identify subgroups?",
            [(72.0, _ICU)],
        ),
        ("Do SOFA-2 trajectories over the first day cluster?", [(24.0, _ICU)]),
        (
            "Do SOFA-2 trajectories over the first 24 h after intubation form classes?",
            [(24.0, "intubation")],
        ),
        ("Do first-24h trajectories post-intubation cluster?", [(24.0, "intubation")]),
        (
            "Do vital-sign trajectories in the first 48 h following septic shock "
            "onset cluster?",
            [(48.0, "septic_shock_onset")],
        ),
        (
            "Do trajectories of heart rate, MAP, and lactate within the first 12 h "
            "of hospital admission cluster?",
            [(12.0, "hospital_admission")],
        ),
        (
            "Do first-24h and first-72h SOFA-2 trajectories differ?",
            [(24.0, _ICU), (72.0, _ICU)],
        ),
        ("入ICU后首24小时的SOFA轨迹能否聚类？", [(24.0, _ICU)]),
        ("前72小时器官功能轨迹可分为几类？", [(72.0, _ICU)]),
        ("首日生命体征轨迹能否聚类？", [(24.0, _ICU)]),
        ("入ICU后前3天的SOFA轨迹能否聚类？", [(72.0, _ICU)]),
        ("机械通气开始后前48小时的生理轨迹是否存在亚型？", [(48.0, "机械通气开始")]),
    ],
)
def test_the_window_stated_for_the_trajectories_is_read(question, stated) -> None:
    assert [
        (item.hours, item.anchor) for item in trajectory_window_statements(question)
    ] == stated


@pytest.mark.parametrize(
    "question",
    [
        # A window that bounds the cohort, an outcome or a measurement.
        "Among patients intubated within the first 24 h, do SOFA-2 trajectories cluster?",
        "Among patients intubated within the first 24 h do SOFA-2 trajectories cluster?",
        "Do SOFA-2 trajectories over the ICU stay predict mortality within the first 28 days?",
        "Do 24-hour urine output trajectories cluster?",
        "Is first 24 h lactate associated with mortality?",
        "Which organ-dysfunction phenotypes emerge from first-day values in the ICU?",
        "首24小时内插管的患者SOFA轨迹能否聚类？",
        "24小时尿量轨迹能否聚类？",
        # "前" after an event is "before" it.
        "入ICU前72小时的乳酸轨迹能否聚类？",
    ],
)
def test_a_window_that_is_not_the_trajectories_is_not_read(question) -> None:
    assert trajectory_window_statements(question) == ()


def _stated(*windows: tuple[float, str]) -> list[TrajectoryWindowStatement]:
    return [
        TrajectoryWindowStatement(hours, anchor, f"first {hours:g} h")
        for hours, anchor in windows
    ]


def test_a_question_without_a_window_keeps_the_default_design() -> None:
    window = trajectory_window_design([])

    assert window["source"] == "design_default"
    assert window["executable"] is True
    assert {name: window[name] for name in _WINDOW} == {
        name: FIXED_WINDOW_TRAJECTORY_DEFAULTS[name] for name in _WINDOW
    }


@pytest.mark.parametrize(
    ("hours", "grid"), [(24, 4), (48, 8), (72, 12), (168, 24), (6, 1)]
)
def test_a_stated_window_keeps_the_default_number_of_windows(hours, grid) -> None:
    window = trajectory_window_design(_stated((float(hours), _ICU)))

    assert window["source"] == "question"
    assert window["executable"] is True
    assert tuple(window[name] for name in _WINDOW) == (0, hours, grid)
    # The design owner accepts what it proposes.
    normalize_trajectory_design(
        {
            "coordinate_concepts": ["sofa2_resp", "sofa2_cardio"],
            **{name: window[name] for name in _WINDOW},
        }
    )


@pytest.mark.parametrize(
    ("windows", "reason"),
    [
        ([(24.0, "intubation")], "counts its trajectory window from intubation"),
        ([(24.0, _ICU), (72.0, _ICU)], "several trajectory windows (24 h, 72 h)"),
        ([(6.5, _ICU)], "not a whole number of hours"),
        ([(1.0, _ICU)], "divides the stated 1-hour trajectory window"),
    ],
)
def test_a_window_the_owner_cannot_count_is_refused_not_replaced(windows, reason) -> None:
    window = trajectory_window_design(_stated(*windows))

    assert window["executable"] is False
    assert reason in window["reason"]
    assert not set(_WINDOW) & set(window)


def _review(question: str):
    context = ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(
            cohort_name="trajectory_window_fixture", database="synthetic", n_stays=0,
            id_columns=["stay_id"], outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name=name, role=role, dtype="float64")
            for name, role in _ROLES.items()
        ],
        target_outcome="death",
    )
    step = AnalysisStep(
        step_id="trajectory_solution",
        planned_analysis_role="primary",
        intent="Cluster the prespecified coordinates into trajectory classes.",
        inputs=["stay_id", "sofa2_resp", "sofa2_cardio", "lact", "artifact:analysis_cohort"],
        expected_outputs=["table:phenotype_assignments"],
        method="prespecified_trajectory_feature_clustering",
        scientific_action_id=TRAJECTORY_PRIMARY_ACTION,
    )
    plan = AnalysisPlan(
        research_question=question, analysis_type="trajectory_clustering", steps=[step]
    )
    review = build_plan_scientific_review(context=context, plan=plan, literature=None)
    [finding] = [item for item in review.findings if item.code.startswith("TRAJECTORY_")]
    return review, finding


def _compile(facts):
    return compile_agent_plan_configuration(
        study=dict(_STUDY),
        agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER,),
        patient_cluster_available=False,
        review_facts=facts,
    )


def test_the_host_compiles_the_window_the_question_states() -> None:
    review, finding = _review(
        "Among adults receiving vasopressors, do first-24h physiologic trajectories "
        "form distinct classes?"
    )

    assert finding.code == _OWNER
    assert "over 0–24 h after ICU admission on a 4 h grid, the window the question states" in (
        finding.message
    )
    assert "research_context.json.research_question" in finding.evidence_refs
    compiled = _compile(review.facts)
    design = compiled.patch["trajectory_design"]
    assert tuple(design[name] for name in _WINDOW) == (0, 24, 4)
    declared = validate_trajectory_design_declaration({**_STUDY, **compiled.patch})
    assert declared is not None and declared.window_count == 6


def test_a_question_without_a_window_compiles_the_default_and_says_so() -> None:
    review, finding = _review(
        "Which organ-dysfunction trajectory classes emerge early in the ICU stay?"
    )

    assert finding.code == _OWNER
    assert "over its default 0–72 h after ICU admission on a 12 h grid" in finding.message
    design = _compile(review.facts).patch["trajectory_design"]
    assert {name: design[name] for name in _WINDOW} == {
        name: FIXED_WINDOW_TRAJECTORY_DEFAULTS[name] for name in _WINDOW
    }


def test_a_window_from_another_event_is_a_stated_limitation_not_the_default() -> None:
    review, finding = _review(
        "Do SOFA-2 trajectories over the first 24 h after intubation form classes?"
    )

    assert finding.code == "TRAJECTORY_REPRESENTATION_NOT_LONGITUDINAL"
    assert finding.severity == "major"
    assert remediation_route_for_finding(finding) == "agent_plan_revision"
    assert "counts its trajectory window from intubation" in finding.message
    assert "Keep the window the question states" in finding.remediation
    facts = review.facts["trajectory_representation"]
    assert facts["coordinates_executable"] is True
    assert facts["executable"] is False
    assert not agent_plan_configuration_available(
        study=dict(_STUDY), agent_plan={"steps": []},
        runtime_finding_codes=(_OWNER,), review_facts=review.facts,
    )


def _facts(window=None) -> dict:
    representation = {
        "longitudinal_owner": None,
        "proposed_coordinates": ["sofa2_resp", "sofa2_cardio"],
        "executable": True,
    }
    if window is not None:
        representation["trajectory_window"] = window
    return {"trajectory_representation": representation}


def test_the_compiled_window_is_the_published_fact_and_its_rules_still_hold() -> None:
    published = {
        "source": "question", "executable": True,
        "window_start_hours": 0, "window_end_hours": 48, "grid_width_hours": 8,
    }
    design = _compile(_facts(published)).patch["trajectory_design"]
    assert tuple(design[name] for name in _WINDOW) == (0, 48, 8)

    # Facts published before the review read a window keep the default.
    design = _compile(_facts()).patch["trajectory_design"]
    assert design["window_end_hours"] == FIXED_WINDOW_TRAJECTORY_DEFAULTS["window_end_hours"]

    # A tampered or stale window still has to satisfy the design owner.
    with pytest.raises(PlanDecisionError) as raised:
        _compile(_facts({**published, "grid_width_hours": 5}))
    assert raised.value.code == "agent_plan_trajectory_design_invalid"
