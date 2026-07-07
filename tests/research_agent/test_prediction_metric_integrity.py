"""Prediction-figure metric integrity (false-pass audit, 2026-07-07).

Three false passes this locks down:
* a Hosmer-Lemeshow expected/observed COUNT table must not be rendered as the
  calibration hero (it would sit off the [0,1] axes as a meaningless curve);
* a long (metric,value,split) performance table must report the HELD-OUT AUROC,
  not last-write-wins (which could ship an optimistic training AUROC);
* panel C's split filter must select ROC-AUC, not PR-AUC / average precision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.figures.prediction import (
    _is_roc_auc_key,
    _load_calibration,
    _out_of_unit_range,
    _performance_metrics,
)


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


# --- calibration range guard ------------------------------------------------


def test_out_of_unit_range():
    assert _out_of_unit_range(pd.Series([12.0, 15.0, 20.0])) is True
    assert _out_of_unit_range(pd.Series([0.1, 0.5, 0.9])) is False
    assert _out_of_unit_range(pd.Series([0.0, 1.0])) is False


def test_calibration_rejects_count_table(tmp_path: Path):
    evidence = EvidenceStore(tmp_path)
    _register(
        evidence,
        tmp_path,
        "calibration_table",
        pd.DataFrame(
            {
                "decile": [1, 2, 3],
                "expected": [12.3, 15.0, 20.0],
                "observed": [10, 18, 22],
                "n": [100, 100, 100],
            }
        ),
    )
    rec, pred, obs = _load_calibration(evidence, tmp_path)
    assert rec is None and pred is None and obs is None


def test_calibration_accepts_risk_table(tmp_path: Path):
    evidence = EvidenceStore(tmp_path)
    _register(
        evidence,
        tmp_path,
        "calibration_table",
        pd.DataFrame(
            {"mean_predicted": [0.1, 0.4, 0.8], "observed": [0.12, 0.38, 0.79]}
        ),
    )
    rec, pred, obs = _load_calibration(evidence, tmp_path)
    assert rec is not None
    assert pred is not None and obs is not None
    assert float(pred.max()) <= 1.0


# --- split-aware performance metrics ---------------------------------------


def test_performance_prefers_heldout_over_train():
    # train row LAST would win under last-write-wins; must report the test AUROC
    frame = pd.DataFrame(
        {
            "metric": ["auroc", "auroc"],
            "value": [0.71, 0.85],
            "split": ["test", "train"],
        }
    )
    metrics = _performance_metrics(frame)
    assert metrics["auroc"] == 0.71


def test_performance_prefers_heldout_regardless_of_order():
    frame = pd.DataFrame(
        {
            "metric": ["auroc", "auroc"],
            "value": [0.85, 0.71],
            "split": ["train", "validation"],
        }
    )
    metrics = _performance_metrics(frame)
    assert metrics["auroc"] == 0.71


def test_performance_no_split_column_is_last_write():
    frame = pd.DataFrame({"metric": ["auroc", "brier"], "value": [0.78, 0.12]})
    metrics = _performance_metrics(frame)
    assert metrics["auroc"] == 0.78 and metrics["brier"] == 0.12


# --- ROC-AUC vs PR-AUC ------------------------------------------------------


def test_is_roc_auc_key():
    for k in ("auroc", "test_auroc", "roc_auc", "auc_roc", "auc"):
        assert _is_roc_auc_key(k) is True, k
    for k in ("pr_auc", "auc_pr", "auprc", "average_precision", "precision_recall_auc"):
        assert _is_roc_auc_key(k) is False, k
