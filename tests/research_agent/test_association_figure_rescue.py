"""The deterministic association forest-plot rescue must tolerate the OR/CI
column-name variants that free-model code emits (e.g. ``ci_lower``/``ci_upper``
rather than ``or_ci_low``/``or_ci_high``). Without this, a figure-only step
fails the whole run even though the parent step computed a valid odds ratio.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.pipeline import (
    _render_association_publication_bundle_from_prior_outputs as rescue,
)


def _make_parent_step(run_dir: Path, csv_name: str, columns: dict) -> None:
    out = run_dir / "steps" / "03_association_model" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(out / csv_name, index=False)


def test_rescue_handles_ci_lower_upper_variant(tmp_path: Path):
    # free-model style column names: odds_ratio + ci_lower/ci_upper
    _make_parent_step(
        tmp_path,
        "adjusted_odds_ratios.csv",
        {
            "variable": ["const", "sepsis3", "age"],
            "odds_ratio": [0.01, 0.80, 1.03],
            "ci_lower": [0.0, 0.74, 1.01],
            "ci_upper": [0.1, 0.86, 1.05],
        },
    )
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(
        run_dir=tmp_path, current_step_id="03_association_model_figure", out_dir=out
    )
    assert rid is not None
    figs = {p.suffix for p in out.iterdir()}
    assert ".png" in figs and ".svg" in figs


def test_rescue_handles_canonical_or_ci_columns(tmp_path: Path):
    # our deterministic fallback style: or_ci_low/or_ci_high
    _make_parent_step(
        tmp_path,
        "association_results.csv",
        {
            "variable": ["sepsis3"],
            "odds_ratio": [0.80],
            "or_ci_low": [0.74],
            "or_ci_high": [0.86],
        },
    )
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out)
    assert rid is not None


def test_rescue_returns_none_without_or_ci_table(tmp_path: Path):
    _make_parent_step(
        tmp_path, "prevalence.csv", {"group": ["a"], "rate": [0.3]}
    )
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    assert rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out) is None
