from __future__ import annotations


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
