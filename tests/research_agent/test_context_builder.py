"""Context-builder tests.

The builder is the bridge between cohort dataframe and agent prompts.
A regression here means the agent sees a wrong picture of the cohort,
which is essentially uncatchable downstream — so we pin the behaviour
tightly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_wide_companion_columns_inherit_exact_base_concept_metadata(
    ra, monkeypatch
):
    from easyicu.research_agent import context as context_module

    def fake_info(name):
        if name == "event_ind":
            return {
                "description": "Registered binary intervention indicator.",
                "clinical_caveats": ["Treat structural absence as event-negative."],
            }
        return None

    monkeypatch.setattr(context_module, "_safe_get_concept_info", fake_info)
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "event_ind_n": [0, 2, 1],
            "event_ind_measured": [0, 1, 1],
            "event_ind_max": [None, 1, 1],
        }
    )

    context = ra.build_research_context(
        research_question="Evaluate a registered intervention indicator.",
        cohort=frame,
        cohort_name="synthetic",
        database="synthetic",
    )

    for column in ("event_ind_n", "event_ind_measured", "event_ind_max"):
        descriptor = context.variable(column)
        assert descriptor is not None
        assert descriptor.source_concept == "event_ind"
        assert descriptor.description == "Registered binary intervention indicator."
        assert "Treat structural absence as event-negative." in (
            descriptor.clinical_caveats
        )


def test_wide_concept_resolution_does_not_fuzzy_match_unknown_columns(
    monkeypatch,
):
    from easyicu.research_agent import context as context_module

    monkeypatch.setattr(
        context_module,
        "_safe_get_concept_info",
        lambda name: {"description": "Known."} if name == "known" else None,
    )

    info, source = context_module._concept_info_for_wide_column("unknown_max")

    assert info is None
    assert source is None


def test_composite_wide_output_inherits_catalog_source_concept(ra, monkeypatch):
    from easyicu.concept import catalog
    from easyicu.research_agent import context as context_module

    monkeypatch.setitem(
        catalog.COMPOSITE_CONCEPT_OUTPUT_SOURCES,
        "derived_signal",
        "canonical_signal",
    )
    monkeypatch.setattr(
        context_module,
        "_safe_get_concept_info",
        lambda name: (
            {
                "name": "canonical_signal",
                "description": "Catalog-owned composite source.",
            }
            if name == "canonical_signal"
            else None
        ),
    )

    context = ra.build_research_context(
        research_question="Evaluate a catalog-owned composite signal.",
        cohort=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "derived_signal_max": [0.0, 1.0, 2.0],
                "derived_signal_measured": [1, 1, 1],
            }
        ),
        cohort_name="synthetic",
        database="synthetic",
    )

    assert context.variable("derived_signal_max").source_concept == (
        "canonical_signal"
    )
    assert context.variable("derived_signal_measured").source_concept == (
        "canonical_signal"
    )


def test_build_context_basic(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60.0, 72.5, 55.1, 80.0],
        "sex": ["M", "F", "F", "M"],
        "sofa2": [0, 1, 5, 8],
        "lact": [1.2, 3.4, 2.1, 8.0],
        "death": [0, 0, 1, 1],
    })
    ctx = ra.build_research_context(
        research_question="Does sofa2 predict death?",
        cohort=df,
        cohort_name="hand_written",
        database="synthetic",
        target_outcome="death",
    )
    assert ctx.research_question.startswith("Does sofa2")
    assert ctx.cohort.n_stays == 4
    assert ctx.cohort.n_patients == 4  # one row per id
    assert "stay_id" in ctx.cohort.id_columns
    assert "death" in ctx.cohort.outcome_columns
    assert ctx.target_outcome == "death"
    # SOFA-2 picked up as a composite score with pitfalls.
    sofa = ctx.variable("sofa2")
    assert sofa is not None
    assert sofa.role.value == "composite_score"
    assert sofa.is_ordinal is True
    assert sofa.pitfalls, "sofa2 must carry the missingness pitfall into context"


def test_id_time_outcome_overrides(ra):
    df = pd.DataFrame({
        "custom_id": [1, 2, 3],
        "ts_event": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "weird_outcome": [0, 1, 0],
    })
    ctx = ra.build_research_context(
        research_question="x",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        id_columns=["custom_id"],
        time_columns=["ts_event"],
        outcome_columns=["weird_outcome"],
        target_outcome="weird_outcome",
    )
    roles = {v.name: v.role.value for v in ctx.variables}
    assert roles["custom_id"] == "id"
    assert roles["ts_event"] == "time"
    assert roles["weird_outcome"] == "outcome"


def test_missingness_profile_is_populated(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "age": [60, 70, 50, 80, 65, 75, 90, 40, 55, 60],
        "lact": [1.0, np.nan, np.nan, 2.0, np.nan, 3.5, np.nan, np.nan, np.nan, 4.0],
    })
    ctx = ra.build_research_context(
        research_question="x", cohort=df,
        cohort_name="c", database="synthetic",
    )
    lact = ctx.variable("lact")
    assert lact is not None
    assert lact.missingness is not None
    assert lact.missingness.n_missing == 6
    assert lact.missingness.n_total == 10
    assert abs(lact.missingness.fraction_missing - 0.6) < 1e-6
    assert lact.missingness.missingness_severity == "high"
    assert lact.missingness.missingness_test in {"little_mcar_em", "not_run"}


def test_missingness_test_is_not_run_when_panel_is_too_small(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "lact": [1.0, np.nan, 2.0],
        "creat": [0.8, 1.1, 1.0],
    })
    ctx = ra.build_research_context(
        research_question="x",
        cohort=df,
        cohort_name="c",
        database="synthetic",
    )
    lact = ctx.variable("lact")
    assert lact is not None
    assert lact.missingness is not None
    assert lact.missingness.missingness_test == "not_run"


def test_context_marks_generic_count_and_measurement_companions_as_audit_metadata(ra):
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "creat_max": [0.8, 1.4, np.nan, 2.1],
            "creat_n": [2, 1, 0, 3],
            "hr_measured": [1, 1, 0, 1],
            "sofa2_measurement_flag": [1, 1, 0, 1],
        }
    )
    ctx = ra.build_research_context(
        research_question="Audit availability for several generic ICU variables.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
    )

    count = ctx.variable("creat_n")
    measured = ctx.variable("hr_measured")
    ordinal_flag = ctx.variable("sofa2_measurement_flag")
    value = ctx.variable("creat_max")

    assert count is not None and count.role.value == "meta"
    assert count.unit is None and count.valid_range is None
    assert count.is_ordinal is False and count.ordinal_levels is None
    assert measured is not None and measured.role.value == "meta"
    assert measured.unit is None and measured.valid_range == [0.0, 1.0]
    assert ordinal_flag is not None and ordinal_flag.role.value == "meta"
    assert ordinal_flag.is_ordinal is False and ordinal_flag.ordinal_levels is None
    assert value is not None and value.role.value == "lab"
    assert value.unit == "mg/dL"


def test_default_time_windows_attached(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2],
        "age": [60.0, 70.0],
        "death": [0, 1],
    })
    ctx = ra.build_research_context(
        research_question="x", cohort=df,
        cohort_name="c", database="synthetic",
        target_outcome="death",
    )
    names = {w.name for w in ctx.time_windows}
    assert {"first_24h", "first_6h", "full_stay"} <= names


def test_target_outcome_semantics_are_enriched_from_question(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60.0, 70.0, 80.0, 55.0],
        "death": [0, 1, 0, 1],
    })
    ctx = ra.build_research_context(
        research_question="Is age associated with ICU mortality?",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    death = ctx.variable("death")
    assert death is not None
    assert death.description is not None
    assert "ICU mortality" in death.description
    assert death.source_concept == "icu_mortality"
    assert any("explicitly treated as ICU mortality" in note for note in death.clinical_caveats)


def test_non_mortality_target_outcomes_receive_explicit_semantics(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60.0, 70.0, 80.0, 55.0],
        "los_icu": [2.5, 7.0, 4.5, 11.0],
    })
    ctx = ra.build_research_context(
        research_question="Model ICU length of stay from admission age.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="los_icu",
    )
    los = ctx.variable("los_icu")
    assert los is not None
    assert los.role.value == "outcome"
    assert los.description is not None
    assert "Length-of-stay" in los.description or "length-of-stay" in los.description
    assert los.source_concept == "length_of_stay"
    assert any("Do not convert length of stay" in note for note in los.cross_database_notes)


def test_unknown_declared_target_outcome_is_not_left_semantically_blank(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "age": [60.0, 70.0, 80.0, 55.0],
        "custom_endpoint": [0.2, 0.5, 0.4, 0.9],
    })
    ctx = ra.build_research_context(
        research_question="Estimate the relation between age and the custom endpoint.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="custom_endpoint",
    )
    endpoint = ctx.variable("custom_endpoint")
    assert endpoint is not None
    assert endpoint.role.value == "outcome"
    assert endpoint.description is not None
    assert "Primary outcome column declared by the caller" in endpoint.description
    assert endpoint.source_concept == "declared_primary_outcome"
    assert any("Do not replace this declared outcome" in note for note in endpoint.cross_database_notes)


def test_survival_question_marks_target_as_time_to_event_endpoint(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "followup_days": [7, 14, 21, 28],
        "death": [0, 1, 0, 1],
    })
    ctx = ra.build_research_context(
        research_question="Evaluate 28-day survival after ICU admission with a Cox model.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    death = ctx.variable("death")
    assert death is not None
    assert death.source_concept == "time_to_event_endpoint"
    assert any("time-to-event endpoint" in note for note in death.clinical_caveats)


def test_naive_context_strips_icu_metadata_and_preferences(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "lact": [1.2, 3.4, np.nan, 2.0],
        "vaso_any_24h": [0, 1, 0, 1],
        "death": [0, 1, 0, 1],
    })
    ctx = ra.build_naive_research_context(
        research_question="Does lactate predict death?",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
        concept_descriptions={"lact": "Lactate with a clinical description."},
        user_preferences={"preferred_methods": "Use logistic regression."},
    )

    assert ctx.time_windows == []
    assert ctx.temporal_constraints == []
    assert ctx.user_preferences is None
    lact = ctx.variable("lact")
    death = ctx.variable("death")
    vaso = ctx.variable("vaso_any_24h")
    assert lact is not None and death is not None and vaso is not None
    assert lact.description is None
    assert lact.role.value == "other"
    assert lact.allowed_aggregations == []
    assert lact.missingness is None
    assert lact.pitfalls == []
    assert lact.clinical_caveats == []
    assert lact.source_concept is None
    assert vaso.role.value == "other"
    assert death.role.value == "outcome"


def test_temporal_constraints_and_provenance_are_attached(ra):
    df = pd.DataFrame({
        "stay_id": [1, 2],
        "age": [60.0, 70.0],
        "death": [0, 1],
        "lact": [2.0, 4.5],
    })
    ctx = ra.build_research_context(
        research_question="Assess AKI within 48h after ICU admission and worst lactate before vasopressor.",
        cohort=df,
        cohort_name="c",
        database="synthetic",
        target_outcome="death",
    )
    assert ctx.temporal_constraints, "temporal semantics should be parsed into deterministic constraints"
    assert any(c.relation == "within_after" for c in ctx.temporal_constraints)
    assert "resolver" in ctx.cohort.provenance


def test_n_patients_distinct_from_n_stays(ra):
    """If two rows share the same stay_id, n_patients < n_stays."""
    df = pd.DataFrame({
        "stay_id": [1, 1, 2, 3, 3, 3],
        "age": [60.0, 60.0, 72.0, 55.0, 55.0, 55.0],
        "death": [0, 0, 1, 1, 1, 1],
    })
    ctx = ra.build_research_context(
        research_question="x", cohort=df,
        cohort_name="c", database="synthetic",
    )
    assert ctx.cohort.n_stays == 6
    assert ctx.cohort.n_patients == 3


def test_retrieved_context_keeps_relevant_and_outcome_variables(ra):
    schema = ra.schema
    variables = [
        schema.ConceptDescriptor(name="stay_id", role="id", dtype="int64"),
        schema.ConceptDescriptor(name="age", role="demographic", dtype="float64"),
        schema.ConceptDescriptor(
            name="sofa2", role="composite_score", dtype="int64",
            pitfalls=["SOFA2 zero can indicate missingness"],
        ),
        schema.ConceptDescriptor(name="lact", role="lab", dtype="float64"),
        schema.ConceptDescriptor(name="death", role="outcome", dtype="int64"),
        schema.ConceptDescriptor(name="map", role="vital", dtype="float64"),
    ]
    ctx = schema.ResearchContext(
        research_question="Is SOFA-2 associated with ICU mortality?",
        cohort=schema.CohortDescriptor(
            cohort_name="c", database="d", n_patients=1, n_stays=1,
            id_columns=["stay_id"], outcome_columns=["death"],
        ),
        variables=variables,
        target_outcome="death",
    )

    from easyicu.research_agent.context import build_retrieved_research_context

    compact = build_retrieved_research_context(ctx, top_k=2)
    names = [v.name for v in compact.variables]
    assert "sofa2" in names
    assert "death" in names
    assert "stay_id" in names
    assert len(names) < len(variables)
    assert "Context retrieval active" in compact.notes
