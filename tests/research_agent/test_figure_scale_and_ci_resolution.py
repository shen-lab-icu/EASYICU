"""Effect-scale and confidence-interval resolution in the causal / survival
renderers must match what the DETERMINISTIC runners actually emit.

Two false-pass bugs this locks down (found by the 2026-07-07 audit, verified on
real H2/H1 runs):

* causal: deterministic_causal writes scale="odds_ratio" (canonical full name) +
  ci_low/ci_high. The renderer classified "odds_ratio" as a difference measure
  (null line at 0, linear axis) and missed ci_low/ci_high (CI collapsed to the
  point estimate). H2's hero panel shipped mislabeled with no interval.
* survival: deterministic_survival writes THREE cox tables; cox_summary.csv is
  metadata-only (no HR) and is matched first, so _parse_cox returned None and the
  HR forest never rendered. The HR + ci_low/ci_high live in cox_model.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.figures.base import load_table
from easyicu.research_agent.figures.causal import _effect_scale_info, _load_effect
from easyicu.research_agent.figures.survival import _COX_TABLE_NAMES, _parse_cox

_COX_REQUIRE = [
    [
        "hazard_ratio",
        "hr",
        "exp(coef)",
        "exp_coef",
        "coef",
        "log_hr",
        "estimate",
        "beta",
    ]
]


def _register(evidence: EvidenceStore, run_dir: Path, name: str, df: pd.DataFrame):
    path = run_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    evidence.register_file(
        kind="table",
        description=f"{name} table.",
        source_path=path,
        evidence_id=name,
        aliases=[name],
        producer="coder",
        generation_mode="agent",
    )


# --- causal effect scale + CI ----------------------------------------------


def test_effect_scale_info_recognises_canonical_full_names():
    for ratio in (
        "odds_ratio",
        "or",
        "hazard_ratio",
        "hr",
        "risk_ratio",
        "rr",
        "relative_risk",
        "rate_ratio",
    ):
        is_ratio, _ = _effect_scale_info(ratio)
        assert is_ratio is True, ratio
    for diff in ("risk_difference", "rd", "ate", "mean_difference", "effect", "coef"):
        is_ratio, _ = _effect_scale_info(diff)
        assert is_ratio is False, diff


def test_load_effect_reads_odds_ratio_scale_and_ci(tmp_path: Path):
    evidence = EvidenceStore(tmp_path)
    # mirrors deterministic_causal.py causal_effect.csv
    _register(
        evidence,
        tmp_path,
        "causal_effect",
        pd.DataFrame(
            [
                {
                    "contrast_id": "c1",
                    "point_estimate": 3.04,
                    "ci_low": 2.87,
                    "ci_high": 3.21,
                    "scale": "odds_ratio",
                }
            ]
        ),
    )
    out = _load_effect(evidence, tmp_path)
    assert out is not None
    _rec, est, lo, hi, scale = out
    assert round(est, 2) == 3.04
    # the CI must NOT collapse to the point estimate
    assert lo != est and hi != est
    assert round(lo, 2) == 2.87 and round(hi, 2) == 3.21
    assert scale == "odds_ratio"
    is_ratio, label = _effect_scale_info(scale)
    assert is_ratio is True and label == "OR"


# --- survival cox-table selection + CI --------------------------------------


def test_cox_load_skips_metadata_summary_and_reads_ci(tmp_path: Path):
    evidence = EvidenceStore(tmp_path)
    # metadata-only table registered FIRST (mirrors cox_summary.csv)
    _register(
        evidence,
        tmp_path,
        "cox_summary",
        pd.DataFrame(
            [
                {
                    "model_name": "primary",
                    "estimator": "cox",
                    "n": 1000,
                    "events": 200,
                    "converged": True,
                    "primary_term": "vent_24h_any",
                }
            ]
        ),
    )
    # the real HR table (mirrors cox_model.csv)
    _register(
        evidence,
        tmp_path,
        "cox_model",
        pd.DataFrame(
            {
                "term": ["vent_24h_any", "age"],
                "coef": [0.60, 0.02],
                "hazard_ratio": [1.83, 1.02],
                "ci_low": [1.74, 1.02],
                "ci_high": [1.91, 1.02],
            }
        ),
    )
    rec, frame = load_table(
        evidence, tmp_path, _COX_TABLE_NAMES, require_columns=_COX_REQUIRE
    )
    assert rec is not None
    # must have skipped the metadata table for the HR-bearing one
    assert "cox_model" in rec.relative_path
    parsed = _parse_cox(frame)
    assert parsed is not None and len(parsed) >= 1
    # CI must be populated, not NaN
    assert parsed["lower"].notna().any() and parsed["upper"].notna().any()
    primary = parsed[parsed["label"] == "vent_24h_any"].iloc[0]
    assert round(float(primary["hr"]), 2) == 1.83
    assert round(float(primary["lower"]), 2) == 1.74


def test_cox_load_returns_nothing_when_only_metadata_exists(tmp_path: Path):
    # If ONLY the metadata table exists, require_columns yields no table (fail
    # closed) rather than feeding _parse_cox a table with no HR.
    evidence = EvidenceStore(tmp_path)
    _register(
        evidence,
        tmp_path,
        "cox_summary",
        pd.DataFrame(
            [{"model_name": "primary", "estimator": "cox", "n": 10, "events": 3}]
        ),
    )
    rec, frame = load_table(
        evidence, tmp_path, _COX_TABLE_NAMES, require_columns=_COX_REQUIRE
    )
    assert rec is None and frame is None
