"""Prediction-model publication figure.

Panels (satisfying the ``prediction`` figure strategy, hero role
``calibration``):

* A -- calibration curve, predicted vs observed risk (hero ``calibration``);
* B -- ROC curve with AUROC annotation (role ``model_performance``);
* C -- discrimination-by-split / metric summary (role ``validation``).

Source evidence: ``calibration_curve`` (predicted/observed bins) and
``roc_curve`` (fpr/tpr) point tables plus the ``model_performance`` metric
table. Returns ``None`` when the curve tables are absent, so the skill's
existing prediction-figure promotion path still handles coder-drawn figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..evidence import EvidenceStore
from ..schema import AnalysisPlan, EvidenceRecord, ResearchContext
from .base import RenderedFigure, load_table, numeric_series, resolve_column

_CALIBRATION_NAMES = [
    "calibration_curve",
    "calibration_bins",
    "calibration_table",
    "reliability_curve",
    "calibration",
]
_ROC_NAMES = [
    "roc_curve",
    "roc_points",
    "roc_table",
    "roc",
]
_PERFORMANCE_NAMES = [
    "model_performance",
    "performance_summary",
    "discrimination_summary",
    "metrics",
]


_SPLIT_NAMES = [
    "split",
    "dataset",
    "set",
    "partition",
    "fold",
    "subset",
    "data_split",
]
_HELDOUT_TOKENS = (
    "test",
    "held",
    "holdout",
    "validation",
    "valid",
    "external",
    "oos",
    "out_of_sample",
)
_TRAIN_TOKENS = ("train", "training", "fit", "in_sample", "apparent", "development")


def _is_roc_auc_key(key: str) -> bool:
    """True for a ROC-AUC metric key, False for PR-AUC / average-precision.

    Panel C hard-labels its y-axis 'AUROC', so a substring ``"auc" in k`` would
    pull a precision-recall AUC (``pr_auc`` / ``auc_pr`` / ``auprc``) into the
    ROC panel and misrepresent it as discrimination.
    """
    k = str(key).strip().lower()
    if any(
        t in k
        for t in ("pr_auc", "auc_pr", "auprc", "average_precision", "precision_recall")
    ):
        return False
    return "auroc" in k or "roc_auc" in k or "auc_roc" in k or k == "auc"


def _out_of_unit_range(series) -> bool:
    """True when a series is not risks/fractions in [0,1] (a count table)."""
    vals = series.dropna()
    if vals.empty:
        return True
    return float(vals.max()) > 1.5 or float(vals.min()) < -0.01


def _load_calibration(evidence: EvidenceStore, run_dir: Path):
    record, frame = load_table(
        evidence,
        run_dir,
        _CALIBRATION_NAMES,
        require_columns=[
            ["predicted", "pred", "mean_predicted", "bin_mid", "expected"],
            ["observed", "obs", "actual", "event_rate", "fraction_positive"],
        ],
    )
    if frame is None:
        return None, None, None
    pred_col = resolve_column(
        frame, ["predicted", "pred", "mean_predicted", "bin_mid", "expected"]
    )
    obs_col = resolve_column(
        frame, ["observed", "obs", "actual", "event_rate", "fraction_positive"]
    )
    pred_s = numeric_series(frame, pred_col)
    obs_s = numeric_series(frame, obs_col)
    # A calibration curve is predicted-vs-observed RISK, both in [0,1]. A
    # Hosmer-Lemeshow expected/observed COUNT table (values >> 1, matched via the
    # "expected"/"observed" candidates) would render off-axis on the [0,1]
    # calibration panel yet pass the role gate as a meaningless hero. Reject it
    # (fail closed -> no prediction figure) rather than ship a wrong curve.
    if _out_of_unit_range(pred_s) or _out_of_unit_range(obs_s):
        return None, None, None
    return record, pred_s, obs_s


def _load_roc(evidence: EvidenceStore, run_dir: Path):
    record, frame = load_table(
        evidence,
        run_dir,
        _ROC_NAMES,
        require_columns=[
            ["fpr", "false_positive", "1-specificity", "specificity"],
            ["tpr", "true_positive", "sensitivity", "recall"],
        ],
    )
    if frame is None:
        return None, None, None
    fpr_col = resolve_column(
        frame, ["fpr", "false_positive_rate", "false_positive", "1-specificity"]
    )
    if fpr_col is None:
        spec_col = resolve_column(frame, ["specificity"])
        fpr = 1.0 - numeric_series(frame, spec_col) if spec_col else None
    else:
        fpr = numeric_series(frame, fpr_col)
    tpr_col = resolve_column(
        frame, ["tpr", "true_positive_rate", "true_positive", "sensitivity", "recall"]
    )
    tpr = numeric_series(frame, tpr_col) if tpr_col else None
    if fpr is None or tpr is None:
        return None, None, None
    return record, fpr, tpr


def _performance_metrics(frame: Optional[pd.DataFrame]) -> Dict[str, float]:
    if frame is None or frame.empty:
        return {}
    metric_col = resolve_column(frame, ["metric", "name", "statistic"])
    value_col = resolve_column(frame, ["value", "estimate", "score"])
    out: Dict[str, float] = {}
    if metric_col and value_col:
        # When a split/dataset column exists, prefer the held-out (test /
        # validation / external) row for each metric. Keying purely on the metric
        # name is last-write-wins, so a training-split AUROC row could overwrite
        # the held-out one and ship an optimistic training number labeled as
        # held-out discrimination.
        split_col = resolve_column(frame, _SPLIT_NAMES)
        chosen: Dict[str, tuple] = {}  # metric -> (priority, value)
        for _, row in frame.iterrows():
            try:
                metric = str(row[metric_col]).strip().lower()
                val = float(row[value_col])
            except (TypeError, ValueError):
                continue
            if split_col is not None:
                split = str(row[split_col]).strip().lower()
                if any(t in split for t in _HELDOUT_TOKENS):
                    prio = 2
                elif any(t in split for t in _TRAIN_TOKENS):
                    prio = 0
                else:
                    prio = 1
            else:
                prio = 1
            prev = chosen.get(metric)
            if prev is None or prio >= prev[0]:
                chosen[metric] = (prio, val)
        return {m: v for m, (_p, v) in chosen.items()}
    # Wide form: one row, columns are metric names.
    for col in frame.columns:
        try:
            out[str(col).strip().lower()] = float(frame.iloc[0][col])
        except (TypeError, ValueError):
            continue
    return out


def render_prediction_figure(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Optional[RenderedFigure]:
    cal_record, pred, obs = _load_calibration(evidence, run_dir)
    roc_record, fpr, tpr = _load_roc(evidence, run_dir)
    # Calibration is the prediction hero; without it the figure cannot be
    # article-grade for this family. ROC provides the discrimination panel.
    if pred is None or fpr is None:
        return None

    perf_record, perf_frame = load_table(evidence, run_dir, _PERFORMANCE_NAMES)
    metrics = _performance_metrics(perf_frame)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ..publication_figures import add_panel_label, apply_publication_style

    palette = apply_publication_style()
    blue = palette.get("blue", "#0F4D92")
    neutral = palette.get("neutral", "#8F8F8F")

    fig = plt.figure(figsize=(183 / 25.4, 74 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.9],
        left=0.075,
        right=0.98,
        top=0.9,
        bottom=0.19,
        wspace=0.42,
    )
    ax_cal = fig.add_subplot(grid[0, 0])
    ax_roc = fig.add_subplot(grid[0, 1])
    ax_val = fig.add_subplot(grid[0, 2])

    # A -- calibration (hero)
    ax_cal.plot([0, 1], [0, 1], color=neutral, linestyle="--", linewidth=0.8)
    order = np.argsort(pred.to_numpy())
    ax_cal.plot(
        pred.to_numpy()[order],
        obs.to_numpy()[order],
        marker="o",
        markersize=3.4,
        color=blue,
        linewidth=1.2,
    )
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("Predicted risk")
    ax_cal.set_ylabel("Observed risk")
    ax_cal.set_title("Calibration", loc="left", pad=4)
    if "brier" in metrics or "brier_score" in metrics:
        brier = metrics.get("brier", metrics.get("brier_score"))
        ax_cal.text(
            0.04,
            0.92,
            f"Brier {brier:.3f}",
            fontsize=6.4,
            color=palette.get("baseline", "#272727"),
        )
    add_panel_label(ax_cal, "A", x=-0.2)

    # B -- ROC
    fpr_v = fpr.to_numpy()
    tpr_v = tpr.to_numpy()
    ro = np.argsort(fpr_v)
    ax_roc.plot([0, 1], [0, 1], color=neutral, linestyle="--", linewidth=0.8)
    ax_roc.plot(fpr_v[ro], tpr_v[ro], color=blue, linewidth=1.3)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("1 - specificity")
    ax_roc.set_ylabel("Sensitivity")
    ax_roc.set_title("Discrimination (ROC)", loc="left", pad=4)
    auroc = metrics.get("auroc", metrics.get("auc", metrics.get("roc_auc")))
    if auroc is not None:
        ax_roc.text(
            0.4,
            0.08,
            f"AUROC {auroc:.3f}",
            fontsize=6.6,
            color=palette.get("baseline", "#272727"),
        )
    add_panel_label(ax_roc, "B", x=-0.2)

    # C -- validation / metric summary
    labels: List[str] = []
    values: List[float] = []
    split_metrics = {k: v for k, v in metrics.items() if _is_roc_auc_key(k)}
    if len(split_metrics) >= 2:
        for k, v in split_metrics.items():
            labels.append(
                k.replace("_auroc", "").replace("auroc", "").strip("_ ") or "AUROC"
            )
            values.append(v)
        ax_val.set_ylabel("AUROC")
    else:
        for key, disp in (
            ("auroc", "AUROC"),
            ("auc", "AUROC"),
            ("brier", "Brier"),
            ("brier_score", "Brier"),
            ("baseline_prevalence", "Prevalence"),
        ):
            if key in metrics and disp not in labels:
                labels.append(disp)
                values.append(metrics[key])
        ax_val.set_ylabel("Value")
    if not labels:
        labels, values = ["AUROC"], [float(auroc) if auroc is not None else 0.0]
    ax_val.bar(range(len(labels)), values, color=blue, width=0.6)
    ax_val.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=6.0)
    ax_val.set_ylim(0, max(1.0, max(values) * 1.15))
    ax_val.set_title("Held-out performance", loc="left", pad=4)
    add_panel_label(ax_val, "C", x=-0.24)

    outcome = context.target_outcome or "the outcome"
    core_claim = (
        f"The {outcome} prediction model is shown with calibration, ROC "
        "discrimination, and held-out performance rendered from registered "
        "model-evaluation evidence."
    )
    source_ids: List[str] = []
    for rec in (cal_record, roc_record, perf_record):
        if rec is not None:
            source_ids.append(rec.evidence_id)

    def _ids(rec: Optional[EvidenceRecord]) -> List[str]:
        return [rec.evidence_id] if rec is not None else []

    panels = [
        {
            "panel_id": "A",
            "title": "Calibration",
            "role": "calibration",
            "chart_type": "calibration_curve",
            "claim": "Predicted versus observed risk shows whether the model's probabilities are usable, not only discriminative.",
            "evidence_ids": _ids(cal_record),
            "review_risk": "A high AUROC with poor calibration can still mislead clinical thresholds.",
        },
        {
            "panel_id": "B",
            "title": "Discrimination (ROC)",
            "role": "model_performance",
            "chart_type": "roc_curve",
            "claim": "The ROC curve summarises discrimination on the held-out split with its AUROC.",
            "evidence_ids": _ids(roc_record),
            "review_risk": "AUROC is imbalance-insensitive; read it with calibration and precision-recall.",
        },
        {
            "panel_id": "C",
            "title": "Held-out performance",
            "role": "validation",
            "chart_type": "validation_panel",
            "claim": "Discrimination and error metrics are reported on the patient-level held-out split.",
            "evidence_ids": _ids(perf_record) or _ids(roc_record),
            "review_risk": "Development-split metrics are optimistic; only held-out/external numbers validate the model.",
        },
    ]

    source_frames: Dict[str, pd.DataFrame] = {
        "calibration_curve": pd.DataFrame({"predicted": pred, "observed": obs}),
        "roc_curve": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
    }

    return RenderedFigure(
        fig=fig,
        figure_id="easyicu_prediction_publication_figure",
        core_claim=core_claim,
        generation_mode="prediction_publication_figure",
        panels=panels,
        source_evidence_ids=source_ids,
        source_frames=source_frames,
    )
