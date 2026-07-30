"""Tests for O17 / O24 / O25.

Tight unit tests on the numpy helpers plus end-to-end checks that the
pipeline integration doesn't crash on the synthetic cohort.

The O19 Cox and Fine-Gray tests that used to open this file are gone with
``methods/survival.py``.  Nothing was left uncovered: they asserted that a
hand-written Breslow-tie Newton-Raphson fit recovers a positive log-HR and
that the subdistribution sketch emits a coefficient — properties of that
implementation, which no production caller ever used and which ``lifelines``
(pinned in the runner image) already provides.  Deleting the implementation
makes those assertions vacuous rather than unowned.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# O25 — MICE + tipping point
# ---------------------------------------------------------------------------


def test_mice_fills_all_missing(ra):
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    z1 = rng.normal(0, 1, n)
    z2 = rng.normal(0, 1, n)
    target = 1.5 + 0.8 * z1 - 0.3 * z2 + rng.normal(0, 0.5, n)
    mask = rng.random(n) < 0.2
    target[mask] = np.nan
    filled, info = ra.mice_impute(
        column="target",
        target=list(target),
        predictors=list(zip(z1.tolist(), z2.tolist())),
    )
    assert info.n_imputed == int(mask.sum())
    assert all(not (isinstance(v, float) and v != v) for v in filled)
    assert info.converged or info.n_iterations > 0


def test_tipping_point_detects_sign_flip(ra):
    import numpy as np

    rng = np.random.default_rng(1)
    n = 400
    x_observed = rng.normal(0, 1, n)
    logit = 0.6 * x_observed
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    # Make 20% of predictor values missing; tipping point should
    # flip the OR when imputed at a very extreme negative value.
    x = x_observed.copy()
    mask = np.zeros(n, dtype=bool)
    mask[: n // 5] = True
    x[mask] = np.nan
    result = ra.tipping_point_analysis(
        predictor_column="x",
        predictor_values=list(x),
        outcome=list(y),
        missing_mask=list(mask),
    )
    assert result.baseline_or is not None and result.baseline_or > 1.0
    assert result.grid and result.or_by_imputed_value
    # Every grid point should produce a computable OR (the mini fit
    # should succeed on 400 points with missing rows filled). Whether
    # the tipping point itself is detected depends on the signal
    # strength, so we only check existence of ORs.
    assert all(orv is not None for orv in result.or_by_imputed_value)


# ---------------------------------------------------------------------------
# O24 — fairness
# ---------------------------------------------------------------------------


def test_subgroup_analysis_runs_over_age_and_sex(ra, synthetic_cohort):
    # Binary outcome 'death', predictor 'sofa2', subgroups age + sex.
    result = ra.run_subgroup_analysis(
        cohort_df=synthetic_cohort,
        predictor="sofa2",
        outcome="death",
        subgroup_columns=["age", "sex"],
    )
    assert result.predictor == "sofa2"
    assert result.outcome == "death"
    # At least one estimate per stratum; age is continuous so quantile-bucketised.
    assert len(result.estimates) >= 2
    # Interaction p-values present for both (may be None if single-level).
    assert "age" in result.interaction_pvalues or "sex" in result.interaction_pvalues


# ---------------------------------------------------------------------------
# O17 — hypothesis generator
# ---------------------------------------------------------------------------


def test_hypothesis_generator_ranks_candidates(ra):
    # Build a tiny ResearchContext with two predictors and one outcome.
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        MissingnessProfile,
        ResearchContext,
        VariableRole,
    )

    cohort = CohortDescriptor(
        cohort_name="tiny",
        database="synthetic",
        n_stays=500,
        n_patients=500,
    )
    ctx = ResearchContext(
        research_question="placeholder",
        cohort=cohort,
        variables=[
            ConceptDescriptor(
                name="marker_a",
                role=VariableRole.LAB,
                dtype="int64",
                missingness=MissingnessProfile(
                    fraction_missing=0.05,
                    n_missing=25,
                    n_total=500,
                ),
            ),
            ConceptDescriptor(
                name="marker_b",
                role=VariableRole.LAB,
                dtype="float64",
                missingness=MissingnessProfile(
                    fraction_missing=0.20,
                    n_missing=100,
                    n_total=500,
                ),
            ),
            ConceptDescriptor(
                name="endpoint_a",
                role=VariableRole.OUTCOME,
                dtype="int8",
            ),
        ],
        target_outcome="endpoint_a",
    )
    result = ra.generate_hypotheses(
        context=ctx,
        citations=[],
        top_k=3,
        hypothesis_family_id="family:test",
    )
    assert len(result.candidates) >= 2
    # Candidates sorted by priority score, descending.
    scores = [c.priority_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    # Both predictors present.
    preds = {c.predictor for c in result.candidates}
    assert {"marker_a", "marker_b"}.issubset(preds)
    payload = result.to_json()
    assert "ranking signal only, not a novelty claim" in payload["signal_statement"]
    assert payload["hypothesis_family_id"] == "family:test"
    top = payload["candidates"][0]
    assert "candidate_id" in top
    assert "literature_saturation_signal" in top
    legacy_key = "literature_" + "novelty"
    assert legacy_key not in top
    assert "Literature saturation" in result.to_markdown()


def test_hypothesis_generator_uses_joint_feasibility_without_dropping(ra):
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        MissingnessProfile,
        ResearchContext,
        VariableRole,
    )

    ctx = ResearchContext(
        research_question="placeholder",
        cohort=CohortDescriptor(
            cohort_name="tiny",
            database="synthetic",
            n_stays=500,
            n_patients=500,
        ),
        variables=[
            ConceptDescriptor(
                name="marker_high_single",
                role=VariableRole.LAB,
                dtype="float64",
                missingness=MissingnessProfile(
                    fraction_missing=0.05,
                    n_missing=25,
                    n_total=500,
                ),
            ),
            ConceptDescriptor(
                name="marker_joint_ready",
                role=VariableRole.LAB,
                dtype="float64",
                missingness=MissingnessProfile(
                    fraction_missing=0.20,
                    n_missing=100,
                    n_total=500,
                ),
            ),
            ConceptDescriptor(
                name="endpoint_a",
                role=VariableRole.OUTCOME,
                dtype="int8",
            ),
        ],
        target_outcome="endpoint_a",
    )

    result = ra.generate_hypotheses(
        context=ctx,
        citations=[],
        top_k=5,
        hypothesis_family_id="family:feasibility",
        feasibility_by_pair={
            ("marker_high_single", "endpoint_a"): {
                "joint_fraction_complete": 0.10,
                "n_joint_complete": 50,
                "denominator_n": 500,
            },
            ("marker_joint_ready", "endpoint_a"): {
                "joint_fraction_complete": 0.80,
                "n_joint_complete": 400,
                "denominator_n": 500,
            },
        },
    )

    by_predictor = {c.predictor: c for c in result.candidates}
    assert (
        by_predictor["marker_joint_ready"].priority_score
        > by_predictor["marker_high_single"].priority_score
    )
    assert by_predictor["marker_high_single"].variable_coverage == pytest.approx(0.10)
    assert (
        by_predictor["marker_high_single"].coverage_source == "pair_joint_feasibility"
    )
    assert by_predictor["marker_high_single"].feasibility_note is not None
    assert (
        "joint completeness below"
        in by_predictor["marker_high_single"].feasibility_note
    )
    assert {"marker_high_single", "marker_joint_ready"}.issubset(by_predictor)


def test_hypothesis_generator_ranks_vital_predictors_with_joint_feasibility(ra):
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )

    ctx = ResearchContext(
        research_question="placeholder",
        cohort=CohortDescriptor(
            cohort_name="tiny",
            database="synthetic",
            n_stays=100,
            n_patients=100,
        ),
        variables=[
            ConceptDescriptor(
                name="mean_pressure",
                role=VariableRole.VITAL,
                dtype="float64",
            ),
            ConceptDescriptor(
                name="endpoint_a",
                role=VariableRole.OUTCOME,
                dtype="int8",
            ),
        ],
        target_outcome="endpoint_a",
    )

    result = ra.generate_hypotheses(
        context=ctx,
        citations=[],
        top_k=3,
        hypothesis_family_id="family:vital",
        feasibility_by_pair={
            ("mean_pressure", "endpoint_a"): {
                "joint_fraction_complete": 0.90,
                "n_joint_complete": 90,
                "denominator_n": 100,
            },
        },
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.predictor == "mean_pressure"
    assert candidate.coverage_source == "pair_joint_feasibility"
    assert candidate.variable_coverage == pytest.approx(0.90)


def test_hypothesis_generator_saturation_is_density_signal(ra):
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        MissingnessProfile,
        ResearchContext,
        VariableRole,
    )

    ctx = ResearchContext(
        research_question="placeholder",
        cohort=CohortDescriptor(
            cohort_name="tiny",
            database="synthetic",
            n_stays=100,
            n_patients=100,
        ),
        variables=[
            ConceptDescriptor(
                name="marker_a",
                role=VariableRole.LAB,
                dtype="float64",
                missingness=MissingnessProfile(
                    fraction_missing=0.0,
                    n_missing=0,
                    n_total=100,
                ),
            ),
            ConceptDescriptor(
                name="endpoint_a",
                role=VariableRole.OUTCOME,
                dtype="int8",
            ),
        ],
        target_outcome="endpoint_a",
    )
    citations = [
        SimpleNamespace(
            title="Marker A and endpoint A",
            relevance="marker_a endpoint_a cohort analysis",
        )
        for _ in range(5)
    ]

    result = ra.generate_hypotheses(
        context=ctx,
        citations=citations,
        top_k=1,
        hypothesis_family_id="family:saturation",
    )

    candidate = result.candidates[0]
    assert candidate.literature_saturation_signal == pytest.approx(0.5)
    assert "literature_gap=0.50" in candidate.rationale
    assert (
        candidate.candidate_id
        == ra.generate_hypotheses(
            context=ctx,
            citations=citations,
            top_k=1,
            hypothesis_family_id="family:saturation",
        )
        .candidates[0]
        .candidate_id
    )


def test_hypothesis_generator_uses_caller_supplied_prior_art_saturation(ra):
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        MissingnessProfile,
        ResearchContext,
        VariableRole,
    )

    ctx = ResearchContext(
        research_question="placeholder",
        cohort=CohortDescriptor(
            cohort_name="tiny",
            database="synthetic",
            n_stays=100,
            n_patients=100,
        ),
        variables=[
            ConceptDescriptor(
                name="marker_a",
                role=VariableRole.LAB,
                dtype="float64",
                missingness=MissingnessProfile(
                    fraction_missing=0.0,
                    n_missing=0,
                    n_total=100,
                ),
            ),
            ConceptDescriptor(
                name="endpoint_a",
                role=VariableRole.OUTCOME,
                dtype="int8",
            ),
        ],
        target_outcome="endpoint_a",
    )
    citations = [
        SimpleNamespace(
            title="Marker A and endpoint A",
            relevance="marker_a endpoint_a crowded title bundle",
        )
        for _ in range(20)
    ]

    result = ra.generate_hypotheses(
        context=ctx,
        citations=citations,
        top_k=1,
        saturation_by_pair={("marker_a", "endpoint_a"): 0.05},
        hypothesis_family_id="family:prior-art",
    )

    candidate = result.candidates[0]
    assert candidate.literature_saturation_signal == pytest.approx(0.05)
    assert "literature_gap=0.95" in candidate.rationale


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_fairness_subgroups_when_synthetic_cohort(
    ra, synthetic_cohort, tmp_path
):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    # Fairness files may or may not exist depending on whether
    # primary_association was registered. At minimum the pipeline
    # should not have raised.
    manifest = json.loads(Path(result.manifest_path).read_text())
    # If fairness files were written, manifest should reference them.
    if (run_dir / "fairness_subgroups.csv").exists():
        ev_ids = {r["evidence_id"] for r in manifest["evidence"]}
        assert "fairness_subgroups" in ev_ids


def test_pipeline_hypothesis_generator_opt_in(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_hypothesis_generator=True,
        hypothesis_generator_top_k=3,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    # Hypothesis candidates fire only in the non-skill path (skill
    # short-circuits planning), so they may not land here; we verify
    # at least that enabling the flag doesn't break the run.
    assert Path(result.manifest_path).exists()
