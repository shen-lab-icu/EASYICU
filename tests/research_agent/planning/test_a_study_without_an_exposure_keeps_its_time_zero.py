"""A study without a primary exposure keeps its own time zero.

A trajectory, descriptive or audit study has no exposure definition to carry a
declared time zero.  The exposure rule still judged it: a question that counted
"from ICU admission" was stopped before planning because "the primary exposure
carries no verifiable anchor", and one anchored at another event was stopped
for the same misattributed reason.  The declaration is now compared with the
event the study's materialized windows count from.
Synthetic, case-neutral contexts only.
"""

from __future__ import annotations

import json

from easyicu.research_agent.gates.preplan import (
    clinical_time_authority_findings,
    preplan_data_failure_reason,
)
from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.reporting.scientific_maturity import (
    build_scientific_maturity_audit,
)
from easyicu.research_agent.research_context.temporal_semantics import (
    primary_exposure_time_anchor_alignment,
    study_time_origin_alignment,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

from .scientific_review_fixtures import _context, _literature, _plan

_STUDY = "STUDY_TIME_ZERO_MISMATCH"
_EXPOSURE = {"PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH", "PRIMARY_EXPOSURE_TIME_ANCHOR_UNVERIFIED"}


def _exposure_free(
    question: str,
    *,
    anchor: str | None = None,
    windows: tuple[str, ...] = ("icu_admission[0,72]h", "icu_admission[0,72]h"),
) -> ResearchContext:
    coordinates = [
        ConceptDescriptor(
            name=f"coordinate_{index}",
            role=VariableRole.ORDINAL_SCORE,
            dtype="float64",
            analysis_window=window,
        )
        for index, window in enumerate(windows)
    ]
    return _context().model_copy(
        update={
            "research_question": question,
            "variables": [
                *coordinates,
                ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
            ],
            "primary_exposure": None,
            "target_outcome": None,
            "endpoint": None,
            "user_preferences": UserPreferences(
                timing_and_design=json.dumps({"anchor": anchor}) if anchor else None
            ),
        }
    )


def _codes(context: ResearchContext, tmp_path) -> tuple[set[str], set[str]]:
    plan = _plan()
    review = build_plan_scientific_review(
        context=context,
        plan=plan,
        literature=_literature(),
        figure_strategy=build_article_figure_strategy(context),
    )
    audit = build_scientific_maturity_audit(context=context, plan=plan, run_dir=tmp_path)
    return {item.code for item in review.findings}, {item.code for item in audit.findings}


def test_a_question_counting_from_icu_admission_is_not_stopped() -> None:
    context = _exposure_free(
        "Do organ-dysfunction trajectories over the first 72 h from ICU admission "
        "cluster into distinct subgroups?"
    )

    assert clinical_time_authority_findings(context) == []
    exposure = primary_exposure_time_anchor_alignment(context)
    assert (exposure.status, exposure.declared_anchor) == ("not_applicable", "icu_admission")
    origin = study_time_origin_alignment(context)
    assert (origin.status, origin.window_anchors) == ("aligned", ("icu_admission",))


def test_a_study_counting_from_another_event_stops_before_planning() -> None:
    context = _exposure_free(
        "Do organ-dysfunction trajectories cluster into distinct subgroups?",
        anchor="septic shock onset",
    )

    findings = clinical_time_authority_findings(context)

    assert len(findings) == 1
    finding = findings[0]
    assert (finding.validator, finding.severity) == ("clinical_time_authority_gate", "error")
    assert "`septic_shock_onset`" in finding.message and "`icu_admission`" in finding.message
    assert "primary exposure" in finding.message
    assert finding.detail == {
        "kind": "study_time_zero_mismatch",
        "status": "mismatch",
        "declared_anchor": "septic_shock_onset",
        "declared_source": "user_preferences.timing_and_design.anchor",
        "window_anchors": ["icu_admission"],
        "window_sources": [
            "variables.coordinate_0.analysis_window",
            "variables.coordinate_1.analysis_window",
        ],
        "required_action": (
            "create_new_study_or_materialization_authority_with_matching_anchor"
        ),
        "provider_called": False,
    }
    assert preplan_data_failure_reason(findings) == "clinical_time_authority_failed"


def test_the_question_alone_can_name_the_other_event() -> None:
    context = _exposure_free(
        "Do lactate trajectories from suspected infection onset cluster into subgroups?"
    )

    origin = study_time_origin_alignment(context)

    assert (origin.status, origin.declared_anchor, origin.declared_source) == (
        "mismatch",
        "suspected_infection_onset",
        "research_question.explicit_relative_anchor",
    )
    assert [item.detail["kind"] for item in clinical_time_authority_findings(context)] == [
        "study_time_zero_mismatch"
    ]


def test_windows_counting_from_two_events_do_not_align() -> None:
    context = _exposure_free(
        "Describe the coordinates from ICU admission.",
        windows=("icu_admission[0,24]h", "hospital_admission[0,24]h"),
    )

    origin = study_time_origin_alignment(context)

    assert (origin.status, origin.window_anchors) == (
        "mismatch",
        ("hospital_admission", "icu_admission"),
    )


def test_the_review_and_the_audit_name_the_study_time_zero(tmp_path) -> None:
    context = _exposure_free(
        "Do organ-dysfunction trajectories cluster into distinct subgroups?",
        anchor="septic shock onset",
    )

    review_codes, audit_codes = _codes(context, tmp_path)

    assert _STUDY in review_codes and _STUDY in audit_codes
    assert not (_EXPOSURE & (review_codes | audit_codes))


def test_an_aligned_study_has_no_time_zero_finding(tmp_path) -> None:
    context = _exposure_free(
        "Do organ-dysfunction trajectories over the first 72 h from ICU admission "
        "cluster into distinct subgroups?"
    )

    review_codes, audit_codes = _codes(context, tmp_path)

    assert not ({_STUDY} | _EXPOSURE) & (review_codes | audit_codes)


def test_a_declaration_without_windows_asks_nothing() -> None:
    context = _exposure_free(
        "Describe the cohort.", anchor="septic shock onset", windows=()
    )

    assert study_time_origin_alignment(context).status == "declared_only"
    assert clinical_time_authority_findings(context) == []


def test_a_study_with_a_primary_exposure_keeps_the_exposure_rule(tmp_path) -> None:
    context = _context().model_copy(
        update={
            "user_preferences": UserPreferences(
                covariates=["age"], timing_and_design='{"anchor":"event onset"}'
            )
        }
    )

    assert study_time_origin_alignment(context).status == "not_applicable"
    assert [item.detail["kind"] for item in clinical_time_authority_findings(context)] == [
        "primary_exposure_time_anchor_mismatch"
    ]
    review_codes, audit_codes = _codes(context, tmp_path)
    assert "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH" in review_codes
    assert "PRIMARY_EXPOSURE_TIME_ANCHOR_MISMATCH" in audit_codes
    assert _STUDY not in review_codes | audit_codes
