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
