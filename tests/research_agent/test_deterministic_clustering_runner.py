"""Historical reproducibility tests for the unrouted clustering fixture.

The tests exec the legacy code string against a synthetic SOFA-shaped cohort,
asserting: it clusters on OBSERVED-window features (never zero-imputes), selects k
by silhouette, emits the certified tables the phenotype figure renderer consumes,
reports a DESCRIPTIVE outcome-by-cluster contrast (adjusted_effect=None, no OR),
and blocks gracefully when no trajectory columns resolve.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.execution.runners.deterministic_clustering import (
    trajectory_clustering_analysis_code,
)


def _exec_runner(run_dir: Path, cohort: pd.DataFrame, context: dict):
    out_dir = run_dir / "steps" / "05_trajectory_clustering" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "research_context.json").write_text(json.dumps(context), encoding="utf-8")
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path, index=False)
    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(cohort_path)
    try:
        code = trajectory_clustering_analysis_code()
        try:
            exec(compile(code, "<det_clustering>", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return (
        json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8")),
        out_dir,
    )


def _h3_cohort(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    windows = list(range(0, 72, 6))  # 12 fixed 6h windows: h0_6 .. h66_72
    # three planted archetypes: stable-low / rising / resolving
    arche = rng.integers(0, 3, n)
    data = {
        "stay_id": np.arange(n),
        "age": rng.integers(40, 90, n),
        "sex": rng.choice(["Male", "Female"], n),
    }
    total = np.zeros((n, len(windows)))
    for i in range(n):
        base = {0: 2.0, 1: 3.0, 2: 9.0}[arche[i]]
        slope = {0: 0.0, 1: 0.6, 2: -0.5}[arche[i]]
        traj = base + slope * np.arange(len(windows)) + rng.normal(0, 0.4, len(windows))
        traj = np.clip(traj, 0, 24)
        # real trailing NA: shorter stays lose late windows
        keep = rng.integers(6, len(windows) + 1)
        traj[keep:] = np.nan
        total[i] = traj
    for j, w in enumerate(windows):
        data[f"sofa2_h{w}_{w+6}"] = total[:, j]
        # one organ family (cardio), correlated
        data[f"sofa2_cardio_h{w}_{w+6}"] = np.clip(total[:, j] * 0.4 + rng.normal(0, 0.2, n), 0, 6)
    # mortality rises with archetype severity (descriptive association only)
    p = np.array([0.06, 0.15, 0.40])[arche]
    data["death"] = (rng.random(n) < p).astype(int)
    return pd.DataFrame(data)


def test_clustering_runs_and_emits_certified_tables(tmp_path: Path):
    summary, out_dir = _exec_runner(tmp_path, _h3_cohort(seed=1), {})
    assert summary["status"] == "ok", summary
    assert summary["analysis_family"] == "phenotyping"
    assert summary["interpretation_class"] == "phenotyping_descriptive"
    assert summary["adjusted_effect"] is None  # descriptive, never an effect estimate
    assert "primary_or" not in summary and "hazard_ratio" not in summary
    assert summary["n_clusters"] >= 2

    for fname in (
        "cluster_assignments.csv",
        "cluster_sizes.csv",
        "cluster_characteristics.csv",
        "clustering_metrics.csv",
        "cluster_stability.csv",
        "outcome_by_cluster.csv",
        "trajectory_features.csv",
        "na_handling.csv",
        "cohort_flow.csv",
    ):
        assert (out_dir / fname).exists(), f"missing {fname}"

    sizes = pd.read_csv(out_dir / "cluster_sizes.csv")
    assert int(sizes["n"].sum()) == summary["n_analysis"]

    metrics = pd.read_csv(out_dir / "clustering_metrics.csv")
    assert "silhouette" in metrics.columns
    assert bool(metrics["chosen"].any())

    chars = pd.read_csv(out_dir / "cluster_characteristics.csv")
    assert set(["cluster", "feature", "mean", "median"]) <= set(chars.columns)  # LONG

    oc = pd.read_csv(out_dir / "outcome_by_cluster.csv")
    assert set(["cluster", "n", "n_deaths", "mortality_rate", "ci_low", "ci_high"]) <= set(oc.columns)


def test_never_zero_imputes_trailing_na(tmp_path: Path):
    # A stay observed only for the first 2 windows must have total-SOFA2 coverage
    # 2/12; its mean feature must equal the mean of the 2 OBSERVED windows, not be
    # diluted by zeros for the 10 unobserved windows.
    windows = list(range(0, 72, 6))
    row = {"stay_id": 0, "age": 70, "sex": "Male", "death": 0}
    for j, w in enumerate(windows):
        row[f"sofa2_h{w}_{w+6}"] = (4.0 if j == 0 else 6.0 if j == 1 else np.nan)
        row[f"sofa2_cardio_h{w}_{w+6}"] = (1.0 if j < 2 else np.nan)
    # pad with a normal cohort so clustering can run
    base = _h3_cohort(n=200, seed=3)
    cohort = pd.concat([pd.DataFrame([row]), base], ignore_index=True)
    _summary, out_dir = _exec_runner(tmp_path, cohort, {"user_preferences": {"min_observed_windows": 2}})
    feats = pd.read_csv(out_dir / "trajectory_features.csv")
    r0 = feats.iloc[0]
    assert abs(r0["sofa2_coverage"] - 2 / 12) < 1e-6
    assert abs(r0["sofa2_mean"] - 5.0) < 1e-6  # mean(4,6) = 5, not diluted by zeros


def test_blocks_without_trajectory_columns(tmp_path: Path):
    cohort = pd.DataFrame(
        {"stay_id": range(50), "age": [70] * 50, "sex": ["Male"] * 50, "death": [0, 1] * 25}
    )
    summary, _out = _exec_runner(tmp_path, cohort, {})
    assert summary["status"] == "blocked"
    assert summary["adjusted_effect"] is None
    assert "sofa" in summary["blocking_reason"].lower() or "trajectory" in summary["blocking_reason"].lower()
