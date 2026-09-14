"""Exposure-horizon query coverage for the PubMed protocol."""

from easyicu.research_agent.literature import (
    build_pubmed_protocol_queries_for_context,
)


def test_protocol_queries_include_database_and_exposure_horizon_stratum(ra):
    schema = ra.schema
    context = schema.ResearchContext(
        research_question=(
            "Estimate the association between first-24-hour maximum lactate "
            "and in-hospital mortality."
        ),
        cohort=schema.CohortDescriptor(
            cohort_name="adult ICU", database="miiv", n_patients=10, n_stays=10
        ),
        variables=[
            schema.ConceptDescriptor(
                name="lact_max",
                description="lactate",
                source_concept="lact",
                role="lab",
                dtype="float64",
                analysis_window="icu_admission[0,24]h",
            ),
            schema.ConceptDescriptor(
                name="death",
                description="in hospital mortality",
                source_concept="death",
                role="outcome",
                dtype="int64",
            ),
        ],
        time_windows=[
            schema.TimeWindow(
                name="icu_admission_0_24h",
                anchor="icu_admission",
                start_hours=0.0,
                end_hours=24.0,
            )
        ],
        primary_exposure="lact_max",
        target_outcome="death",
    )

    queries = build_pubmed_protocol_queries_for_context(context)

    database_horizon = next(
        query
        for query in queries
        if '"MIMIC-III"[Title/Abstract]' in query
        and '"24-hour"[Title/Abstract]' in query
    )
    assert '"lactate"[Title/Abstract]' in database_horizon
    assert '"in-hospital mortality"[Title/Abstract]' in database_horizon
    assert '"inhospital mortality"[Title/Abstract]' in database_horizon
    assert "mortality[Title/Abstract]" in database_horizon
    assert "death[Title/Abstract]" in database_horizon
    assert '"24 h"[Title/Abstract]' in database_horizon
    assert "MIMIC-IV" in database_horizon
