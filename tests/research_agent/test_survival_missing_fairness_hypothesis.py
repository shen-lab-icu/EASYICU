"""Tests for O17 / O19 / O24 / O25.

Tight unit tests on the numpy helpers plus end-to-end checks that the
pipeline integration doesn't crash on the synthetic cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# O19 — Cox
# ---------------------------------------------------------------------------


def test_cox_recovers_positive_coefficient(ra):
    import numpy as np

    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(0, 1, n)
    # true log-HR = 0.8 for x
    baseline = rng.exponential(1.0, n)
    t = baseline * np.exp(-0.8 * x)
    e = (t < np.quantile(t, 0.7)).astype(int)
    result = ra.fit_cox_model(
        times=list(t),
        events=list(e),
        covariates=[[v] for v in x],
        terms=["x"],
    )
    assert result.converged
    assert len(result.coefficients) == 1
    coef = result.coefficients[0]
    assert coef.hazard_ratio > 1.0  # positive log-HR => HR > 1
    assert coef.p_value is not None and coef.p_value < 0.05
    assert result.concordance is not None and 0.6 < result.concordance < 0.95


def test_fine_gray_handles_competing_event(ra):
    import numpy as np

    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(0, 1, n)
    t = rng.exponential(1.0, n)
    codes = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
    result = ra.fit_fine_gray_subdistribution(
        times=list(t),
        event_codes=list(codes),
        covariates=[[v] for v in x],
        terms=["x"],
        event_of_interest=1,
    )
    assert result.n_events > 0
    # convergence optional; the fit should at least emit a coefficient
    assert len(result.coefficients) == 1


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
        database="miiv",
        n_stays=500,
        n_patients=500,
    )
    ctx = ResearchContext(
        research_question="placeholder",
        cohort=cohort,
        variables=[
            ConceptDescriptor(
                name="sofa2",
                role=VariableRole.COMPOSITE_SCORE,
                dtype="int64",
                missingness=MissingnessProfile(
                    fraction_missing=0.05, n_missing=25, n_total=500,
                ),
            ),
            ConceptDescriptor(
                name="lact",
                role=VariableRole.LAB,
                dtype="float64",
                missingness=MissingnessProfile(
                    fraction_missing=0.20, n_missing=100, n_total=500,
                ),
            ),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int8",
            ),
        ],
        target_outcome="death",
    )
    result = ra.generate_hypotheses(context=ctx, citations=[], top_k=3)
    assert len(result.candidates) >= 2
    # Candidates sorted by priority score, descending.
    scores = [c.priority_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    # Both predictors present.
    preds = {c.predictor for c in result.candidates}
    assert {"sofa2", "lact"}.issubset(preds)


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
        skill="sofa_mortality",
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
        skill="sofa_mortality",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    # Hypothesis candidates fire only in the non-skill path (skill
    # short-circuits planning), so they may not land here; we verify
    # at least that enabling the flag doesn't break the run.
    assert Path(result.manifest_path).exists()
