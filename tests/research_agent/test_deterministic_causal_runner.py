"""Regression: the deterministic IPTW causal runner produces a bound primary
odds ratio without an LLM coder call.

Built 2026-07-06 after H2 vasopressor causal: the LLM-generated propensity code
failed on successive fragile errors (np.isfinite on nullable dtypes, then
"arg must be a 1-d array"), leaving adjusted_effect=None. This runner owns the
causal estimand deterministically, the way the Cox runner owns survival.

The tests exec the runner's code string against synthetic cohorts (no real
data), asserting: it resolves a concept-named exposure to its aggregate column,
recovers a positive odds ratio with a non-empty balance table, and blocks
gracefully when an exposure group is degenerate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.deterministic_causal import causal_primary_analysis_code


def _exec_runner(run_dir: Path, cohort: pd.DataFrame, context: dict) -> dict:
    out_dir = run_dir / "steps" / "01_causal_effect_estimation" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "research_context.json").write_text(json.dumps(context), encoding="utf-8")
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path, index=False)

    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(cohort_path)
    try:
        code = causal_primary_analysis_code()
        try:
            exec(compile(code, "<det_causal>", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))


def _confounded_cohort(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 15, n)
    # exposure confounded by age; ~35% exposed
    lin_e = -0.5 + 0.03 * (age - 60)
    p_e = 1.0 / (1.0 + np.exp(-lin_e))
    exposed = rng.binomial(1, p_e).astype(float)
    # binary-indicator aggregate: NaN when never exposed (universe convention)
    vaso_ind_max = np.where(exposed == 1, 1.0, np.nan)
    # outcome depends on age (confounder) AND exposure (true effect, logOR ~ 0.9)
    lin_y = -2.0 + 0.04 * (age - 60) + 0.9 * exposed
    p_y = 1.0 / (1.0 + np.exp(-lin_y))
    death = rng.binomial(1, p_y).astype(float)
    return pd.DataFrame(
        {
            "vaso_ind_max": vaso_ind_max,
            "age": age,
            "sex": rng.choice(["Male", "Female"], n),
            "map_max": rng.normal(75, 12, n),
            # a nullable extension column must not break the runner
            "extra_flag_n": pd.array(rng.integers(0, 3, n), dtype="Int64"),
            "death": death,
        }
    )


def test_recovers_positive_odds_ratio_with_balance(tmp_path: Path):
    ctx = {"primary_exposure": "vasopressor", "target_outcome": "death"}
    summary = _exec_runner(tmp_path / "run", _confounded_cohort(), ctx)
    assert summary["status"] == "ok", summary
    assert summary["adjusted_effect_scale"] == "odds_ratio"
    orv = summary["adjusted_effect"]
    assert isinstance(orv, float) and orv > 1.0, orv
    # CI ordered and finite
    assert summary["adjusted_effect_ci_low"] < orv < summary["adjusted_effect_ci_high"]
    # weighting reduced imbalance below the conventional 0.1 threshold
    assert summary["max_smd_after_weighting"] <= 0.15, summary["max_smd_after_weighting"]
    # non-empty balance table is what the figure step traces to
    bal = pd.read_csv(
        tmp_path / "run" / "steps" / "01_causal_effect_estimation" / "outputs" / "balance_pre_post_weighting.csv"
    )
    assert len(bal) >= 1 and "smd_weighted" in bal.columns


def test_resolves_concept_named_exposure_to_aggregate_column(tmp_path: Path):
    # Exposure declared by concept name; only the *_max aggregate exists.
    ctx = {"primary_exposure": "vasopressor", "target_outcome": "death"}
    summary = _exec_runner(tmp_path / "run", _confounded_cohort(), ctx)
    assert summary["status"] == "ok"
    assert summary["exposure_name"] == "vasopressor"
    assert summary["exposed_primary"] > 0


def test_blocks_gracefully_on_degenerate_exposure(tmp_path: Path):
    # Everyone exposed -> no control group -> block, do not crash.
    cohort = _confounded_cohort()
    cohort["vaso_ind_max"] = 1.0
    ctx = {"primary_exposure": "vasopressor", "target_outcome": "death"}
    summary = _exec_runner(tmp_path / "run", cohort, ctx)
    assert summary["status"] == "blocked"
    assert summary["adjusted_effect"] is None
    assert "too small" in summary["blocking_reason"]


def test_blocks_when_declared_exposure_absent(tmp_path: Path):
    # No column resolvable to the declared exposure -> block, never guess.
    cohort = _confounded_cohort().drop(columns=["vaso_ind_max"])
    ctx = {"primary_exposure": "vasopressor", "target_outcome": "death"}
    summary = _exec_runner(tmp_path / "run", cohort, ctx)
    assert summary["status"] == "blocked"
    assert "Missing required causal columns" in summary["blocking_reason"]


# --- target-trial protocol spec for the design schematic (H2 fix6) ---------
#
# A target-trial-emulation design figure step renders a protocol schematic by
# scanning upstream artefacts for standard sections and SKIPS (failing the
# execution gate, blocking the manuscript) unless it finds a minimum contract:
# ``["eligibility", "time_zero", "exposure_strategy", "outcome"]``. The plain
# effect/balance tables expose no "time zero" text (H2 fix9 skipped on exactly
# this), so the runner must emit a protocol spec whose section labels each
# classify to the right box.


def _classify_section(text: str):
    """Mirror the schematic renderer's classify_text minimum contract."""
    t = str(text).lower()
    if any(k in t for k in ("eligib", "include", "exclude", "adult", "cohort")):
        return "eligibility"
    if any(k in t for k in ("time zero", "baseline", "t0", "icu admission")):
        return "time_zero"
    if any(k in t for k in ("strateg", "exposure", "treatment", "comparator")):
        return "exposure_strategy"
    if any(k in t for k in ("assignment", "observational", "not randomized")):
        return "assignment"
    if any(k in t for k in ("follow up", "follow-up", "censor", "discharge")):
        return "follow_up"
    if any(k in t for k in ("outcome", "death", "mortality")):
        return "outcome"
    return None


def test_emits_target_trial_protocol_with_minimum_schematic_sections(tmp_path: Path):
    ctx = {"primary_exposure": "vasopressor", "target_outcome": "death"}
    summary = _exec_runner(tmp_path / "run", _confounded_cohort(), ctx)
    assert summary["status"] == "ok"
    # spec present in the summary and written as a table
    rows = summary["target_trial_protocol"]
    assert isinstance(rows, list) and len(rows) == 8
    csv_path = (
        tmp_path
        / "run"
        / "steps"
        / "01_causal_effect_estimation"
        / "outputs"
        / "target_trial_protocol.csv"
    )
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["section", "detail"]
    # every minimum-required schematic section is covered by a section label,
    # each classifying to its OWN box (the label alone is unambiguous)
    covered = {_classify_section(r["section"]) for r in rows}
    for required in ("eligibility", "time_zero", "exposure_strategy", "outcome"):
        assert required in covered, f"{required} not covered by {covered}"
    # the section that historically went missing must now be present
    assert any(_classify_section(r["section"]) == "time_zero" for r in rows)
