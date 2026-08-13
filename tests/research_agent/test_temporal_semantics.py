from __future__ import annotations

from easyicu.research_agent.research_context.temporal_semantics import (
    primary_exposure_time_anchor_alignment,
    window_extends_after_anchor,
)
from easyicu.research_agent.schema import (
    ClinicalDefinitionReference,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
)


def test_window_extends_after_anchor_distinguishes_range_dash_from_negative_time():
    assert window_extends_after_anchor("0-24h") is True
    assert window_extends_after_anchor("0–24 hours after ICU admission") is True
    assert window_extends_after_anchor("-24 to 0h") is False
    assert window_extends_after_anchor("at ICU admission") is False


def test_time_window_semantic_parser_extracts_common_icu_phrases(ra):
    parser = ra.TimeWindowSemanticParser()
    constraints = parser.parse(
        "Use first 24h SOFA, worst lactate before vasopressor, and AKI within 48h after ICU admission."
    )
    relations = {c.relation for c in constraints}
    assert "first_window" in relations
    assert "worst_before_event" in relations
    assert "within_after" in relations


def test_temporal_alignment_engine_derives_time_window(ra):
    engine = ra.TemporalAlignmentEngine()
    windows, constraints = engine.infer(
        research_question="Assess AKI within 48h after ICU admission.",
        timing_and_design=None,
        explicit_windows=None,
    )
    assert constraints
    assert any(w.name.startswith("within_48h_after_icu_admission") for w in windows)


def test_parser_records_explicit_relative_anchor_phrases(ra):
    parser = ra.TimeWindowSemanticParser()
    phrases = (
        ("Bin measurements into six-hour windows from ICU admission.", "icu_admission"),
        (
            "Construct trajectories anchored at hospital admission.",
            "hospital_admission",
        ),
        ("Express every window relative to event-onset.", "event_onset"),
    )

    for phrase, expected_anchor in phrases:
        constraints = parser.parse(phrase)
        matches = [
            item for item in constraints if item.relation == "relative_to_anchor"
        ]
        assert len(matches) == 1
        assert matches[0].anchor_event == expected_anchor
        assert matches[0].start_hours is None
        assert matches[0].end_hours is None


def _anchor_context(
    *,
    declared: str,
    materialized: str,
    definition_anchor: str | None = None,
    window_role: str | None = "exposure_definition",
) -> ResearchContext:
    return ResearchContext(
        research_question=(
            "Classify an exposure anchored to an explicitly declared clinical event."
        ),
        cohort=CohortDescriptor(
            cohort_name="adult ICU stays",
            database="synthetic",
            n_stays=20,
        ),
        variables=[
            ConceptDescriptor(
                name="exposure",
                role="other",
                dtype="int64",
                analysis_window=materialized,
                analysis_window_role=window_role,
                clinical_definition=(
                    ClinicalDefinitionReference(
                        contract_id="test_contract",
                        definition="Test phenotype",
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
        ],
        primary_exposure="exposure",
        user_preferences=UserPreferences(
            timing_and_design=(
                '{"anchor": "' + declared + '", "preset": "clinical_event"}'
            )
        ),
    )


def test_primary_exposure_anchor_alignment_never_equates_two_clinical_events():
    decision = primary_exposure_time_anchor_alignment(
        _anchor_context(
            declared="suspected-infection onset",
            materialized="icu_admission[0,24]h",
        )
    )

    assert decision.status == "mismatch"
    assert decision.declared_anchor == "suspected_infection_onset"
    assert decision.definition_anchor == "icu_admission"
    assert decision.observation_window_anchor == "icu_admission"
    assert decision.declared_source == "user_preferences.timing_and_design.anchor"


def test_primary_exposure_anchor_alignment_accepts_spelling_variants_only():
    decision = primary_exposure_time_anchor_alignment(
        _anchor_context(
            declared="ICU admission",
            materialized="icu-admission[0,24]h",
        )
    )

    assert decision.status == "aligned"
    assert decision.declared_anchor == decision.definition_anchor


def test_derived_phenotype_definition_anchor_is_distinct_from_outer_window():
    decision = primary_exposure_time_anchor_alignment(
        _anchor_context(
            declared="suspected-infection onset",
            definition_anchor="suspected_infection_onset",
            materialized="icu_admission[0,24]h",
            window_role="outer_observation_window",
        )
    )

    assert decision.status == "aligned"
    assert decision.definition_anchor == "suspected_infection_onset"
    assert decision.observation_window_anchor == "icu_admission"
    assert decision.observation_window_role == "outer_observation_window"
