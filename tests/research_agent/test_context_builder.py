"""Context-builder tests.

The builder is the bridge between cohort dataframe and agent prompts.
A regression here means the agent sees a wrong picture of the cohort,
which is essentially uncatchable downstream — so we pin the behaviour
tightly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    # 60% missing should bucket into MNAR_likely per builder heuristic.
    assert lact.missingness.missingness_kind == "MNAR_likely"


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
