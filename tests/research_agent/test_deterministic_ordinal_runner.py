"""Regression: the deterministic ordinal dose-response runner produces a bound
primary trend odds ratio without an LLM coder call.

Built 2026-07-07 for E3 (KDIGO AKI-stage dose-response vs in-hospital
mortality). The single reproducible headline for a graded ordinal EXPOSURE is
the covariate-adjusted odds ratio per +1 stage (the trend); the per-stage ORs
(vs the lowest stage) are the forest rows. An LLM coder left to invent this
drifts between per-stage contrasts, a continuous slope, and -- wrongly -- an
ordinal-outcome model (here the ordinal variable is the exposure).

The tests exec the runner's code string against synthetic cohorts (no real
data), asserting: it resolves a ``kdigo`` alias to the composite
``aki_stage_max`` grade (not a sub-component), recovers a > 1 trend OR with a
monotone per-stage forest, binds the scale, computes a secondary continuous
gradient, and blocks gracefully when the exposure is not an ordinal grade.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.execution.runners.deterministic_ordinal import (
    ordinal_dose_response_analysis_code,
)


def _exec_runner(run_dir: Path, cohort: pd.DataFrame, context: dict) -> dict:
    out_dir = run_dir / "steps" / "02_dose_response" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "research_context.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path, index=False)

    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(cohort_path)
    try:
        code = ordinal_dose_response_analysis_code()
        try:
            exec(compile(code, "<det_ordinal>", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return (
        json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8")),
        out_dir,
    )


def _dose_response_cohort(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Graded exposure 0-3 with a monotone true dose-response on mortality and a
    monotone LOS gradient, mildly confounded by age/severity.

    The stage is given independent variation (a large idiosyncratic latent term)
    on top of a modest severity link, so the true stage->outcome effect stays
    identifiable after adjustment -- i.e. the test checks that a real gradient
    survives confounder control, not that adjustment collinearly wipes it out.
    """
    rng = np.random.default_rng(seed)
    age = rng.normal(62, 15, n)
    sofa = rng.poisson(4, n).astype(float)
    # graded exposure (0..3): modest severity link + a dominant independent term,
    # cut into quartiles so the four levels are balanced and not collinear.
    latent = 0.4 * (sofa - 4) / 2.0 + 0.2 * (age - 62) / 15.0 + rng.normal(0, 1, n)
    qs = np.quantile(latent, [0.25, 0.5, 0.75])
    stage = np.digitize(latent, qs).astype(float)  # 0,1,2,3 balanced
    # outcome: strong true positive dose-response (logOR ~0.7 per stage) + mild
    # confounding through age/severity.
    lin_y = -3.0 + 0.7 * stage + 0.02 * (age - 62) + 0.05 * (sofa - 4)
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-lin_y))).astype(float)
    los_icu = np.exp(0.6 + 0.25 * stage + rng.normal(0, 0.4, n))  # monotone in stage
    return pd.DataFrame(
        {
            # composite grade the runner must prefer:
            "aki_stage_max": stage,
            # sub-component grades that must NOT be chosen as the exposure:
            "aki_stage_creat_max": np.clip(stage - rng.integers(0, 2, n), 0, 3).astype(
                float
            ),
            "aki_stage_uo_max": np.clip(stage - rng.integers(0, 2, n), 0, 3).astype(
                float
            ),
            # non-grade aggregates that must be ignored:
            "aki_stage_mean": stage * rng.uniform(0.3, 0.9, n),
            "aki_stage_measured": np.ones(n),
            "age": age,
            "sex": rng.choice(["Male", "Female"], n),
            "sofa_cardio_max": sofa,
            "los_icu": los_icu,
            "death": death,
        }
    )


def test_recovers_positive_trend_or_with_monotone_forest(tmp_path: Path):
    ctx = {"primary_exposure": "kdigo", "target_outcome": "death"}
    summary, out_dir = _exec_runner(tmp_path / "run", _dose_response_cohort(), ctx)
    assert summary["status"] == "ok", summary
    assert summary["analysis_family"] == "association"
    assert summary["adjusted_effect_scale"] == "odds_ratio"
    # true logOR per stage is +0.5 -> OR ~ 1.65; adjusted estimate must be > 1
    assert summary["adjusted_effect"] > 1.15, summary["adjusted_effect"]
    assert (
        summary["adjusted_effect_ci_low"]
        < summary["adjusted_effect"]
        < summary["adjusted_effect_ci_high"]
    )
    assert summary["grade_levels"] == [0, 1, 2, 3]
    assert summary["reference_level"] == 0
    # per-stage forest: a row per level, reference OR pinned to 1.0
    rows = summary["per_stage_effects"]
    assert [r["stage"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["is_reference"] and rows[0]["odds_ratio"] == 1.0
    assert rows[-1]["odds_ratio"] > 1.0
    assert summary["per_stage_monotonic"] is True
    # tables written for the association forest renderer
    assert (out_dir / "dose_response.csv").exists()
    assert (out_dir / "dose_response_trend.csv").exists()
    assert (out_dir / "los_gradient.csv").exists()


def test_resolves_kdigo_alias_to_composite_grade_not_subcomponent(tmp_path: Path):
    ctx = {"primary_exposure": "kdigo", "target_outcome": "death"}
    summary, out_dir = _exec_runner(tmp_path / "run", _dose_response_cohort(), ctx)
    deriv = pd.read_csv(out_dir / "exposure_derivation.csv")
    assert deriv.loc[0, "source_column"] == "aki_stage_max"


def test_secondary_continuous_gradient_is_reported(tmp_path: Path):
    ctx = {"primary_exposure": "kdigo", "target_outcome": "death"}
    summary, _ = _exec_runner(tmp_path / "run", _dose_response_cohort(), ctx)
    grad = summary["secondary_outcome_gradient"]
    assert grad is not None
    assert grad["outcome"] == "los_icu"
    assert grad["spearman_rho"] > 0.2  # LOS rises with stage


def test_blocks_when_exposure_is_not_ordinal(tmp_path: Path):
    # a binary indicator (2 levels) is not a >=3-level graded exposure
    df = _dose_response_cohort()
    df["aki_stage_max"] = (df["aki_stage_max"] >= 1).astype(float)
    ctx = {"primary_exposure": "kdigo", "target_outcome": "death"}
    summary, _ = _exec_runner(tmp_path / "run", df, ctx)
    assert summary["status"] == "blocked"
    assert summary["adjusted_effect"] is None
    assert "ordinal" in summary["blocking_reason"].lower()


def test_blocks_when_outcome_absent(tmp_path: Path):
    df = _dose_response_cohort().drop(columns=["death"])
    ctx = {"primary_exposure": "kdigo", "target_outcome": "death"}
    summary, _ = _exec_runner(tmp_path / "run", df, ctx)
    assert summary["status"] == "blocked"
    assert "outcome" in summary["blocking_reason"].lower()


def test_config_covariates_override_default_adjustment(tmp_path: Path):
    ctx = {
        "primary_exposure": "kdigo",
        "target_outcome": "death",
        "user_preferences": {"covariates": ["age"]},
    }
    summary, _ = _exec_runner(tmp_path / "run", _dose_response_cohort(), ctx)
    assert summary["adjustment_source"] == "config"
    # only the configured covariate is adjusted for (sofa_cardio not pulled in)
    assert summary["adjustment_covariates"] == ["age"]
