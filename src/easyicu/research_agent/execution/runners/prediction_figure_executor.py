"""Source-bound four-panel renderer for the static prediction adapter."""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .figure_input_capability import TypedInputCapability
from .prediction_model_executor import (
    PREDICTION_CALIBRATION_PRODUCT,
    PREDICTION_INTERNAL_VALIDATION_PRODUCT,
    PREDICTION_PERFORMANCE_PRODUCT,
    PREDICTION_SCORES_PRODUCT,
)
from .typed_input_binding import BoundTypedInput, load_typed_input, sha256_file

PREDICTION_COMPOSITE_FIGURE_INPUTS = (
    PREDICTION_SCORES_PRODUCT,
    PREDICTION_PERFORMANCE_PRODUCT,
    PREDICTION_INTERNAL_VALIDATION_PRODUCT,
    PREDICTION_CALIBRATION_PRODUCT,
)
PREDICTION_FIGURE_ANALYSIS_KIND = "static_prediction_composite_figure"

_CAPABILITY = TypedInputCapability(required=frozenset(PREDICTION_COMPOSITE_FIGURE_INPUTS))
_REQUIRED_COLUMNS = {
    PREDICTION_SCORES_PRODUCT: frozenset(
        {"unit_id", "subject_id", "split", "outcome", "probability"}
    ),
    PREDICTION_PERFORMANCE_PRODUCT: frozenset(
        {
            "authority_scope",
            "development_n",
            "validation_n",
            "patient_overlap_n",
            "auroc",
            "average_precision",
            "brier_score",
        }
    ),
    PREDICTION_INTERNAL_VALIDATION_PRODUCT: frozenset(
        {
            "evaluation_n",
            "event_n",
            "evaluation_subject_n",
            "repeated_subject_n",
            "auroc",
            "brier_score",
            "patient_overlap_n",
        }
    ),
    PREDICTION_CALIBRATION_PRODUCT: frozenset(
        {
            "row_role",
            "n",
            "mean_predicted_probability",
            "observed_event_rate",
            "brier_score",
            "calibration_status",
            "calibration_intercept",
            "calibration_slope",
        }
    ),
}


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if kind != "figure" or not separator or not re.fullmatch(
        r"[a-z][a-z0-9_]{0,127}", product
    ):
        return None
    return product


def _binding_has_columns(binding: Any, key: str) -> bool:
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    columns = contract.get("columns") if isinstance(contract, Mapping) else None
    return bool(
        isinstance(columns, list)
        and all(isinstance(column, str) for column in columns)
        and _REQUIRED_COLUMNS[key] <= set(columns)
    )


def prediction_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    products = [_figure_product(value) for value in step.expected_outputs]
    return bool(
        step.planned_analysis_role == "auxiliary"
        and str(step.method or "").strip().casefold().split(" with ", 1)[0]
        == "visualization"
        and len(step.inputs) == len(PREDICTION_COMPOSITE_FIGURE_INPUTS)
        and set(step.inputs) == set(PREDICTION_COMPOSITE_FIGURE_INPUTS)
        and _CAPABILITY.admits_step(step)
        and len(products) == 1
        and products[0] is not None
        and isinstance(resolved_bindings, Mapping)
        and set(resolved_bindings) == set(PREDICTION_COMPOSITE_FIGURE_INPUTS)
        and all(
            _binding_has_columns(resolved_bindings.get(key), key)
            for key in PREDICTION_COMPOSITE_FIGURE_INPUTS
        )
    )


def prediction_figure_executor_code(step: AnalysisStep) -> str:
    product = _figure_product(step.expected_outputs[0]) if step.expected_outputs else None
    if product is None:
        raise ValueError("prediction figure has no safe figure product")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.prediction_figure_executor import (
            run_prediction_figure,
        )

        run_prediction_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
        )
        """
    ).strip()


def _load_inputs(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> dict[str, BoundTypedInput]:
    return {
        key: load_typed_input(
            input_key=key,
            run_dir=run_dir,
            resolved_inputs=resolved_inputs,
            step_id=step_id,
            expected_declared_kind="table",
            expected_evidence_kind="table",
            require_consumption_contract=True,
            minimum_row_count=1,
        )
        for key in PREDICTION_COMPOSITE_FIGURE_INPUTS
    }


def _finite(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise RuntimeError(f"prediction figure source {column!r} is not finite")
    return values.astype(float)


def run_prediction_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> dict[str, Any]:
    """Render four exact tables without fitting or selecting a model."""

    if _figure_product(f"figure:{figure_product}") is None:
        raise ValueError("unsafe prediction figure product")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bound = _load_inputs(
        run_dir=Path(run_dir), resolved_inputs=resolved_inputs, step_id=step_id
    )
    for key, item in bound.items():
        missing = _REQUIRED_COLUMNS[key] - set(item.frame.columns)
        if missing:
            raise RuntimeError(f"{key} is missing columns {sorted(missing)!r}")
    scores = bound[PREDICTION_SCORES_PRODUCT].frame.copy()
    validation = scores.loc[scores["split"].astype(str).eq("validation")].copy()
    outcomes = pd.to_numeric(validation["outcome"], errors="coerce")
    probabilities = _finite(validation, "probability")
    if outcomes.isna().any() or not outcomes.isin((0, 1)).all() or outcomes.nunique() != 2:
        raise RuntimeError("prediction figure requires binary validation outcomes")
    performance = bound[PREDICTION_PERFORMANCE_PRODUCT].frame.copy()
    internal = bound[PREDICTION_INTERNAL_VALIDATION_PRODUCT].frame.copy()
    calibration = bound[PREDICTION_CALIBRATION_PRODUCT].frame.copy()
    if len(performance) != 1 or len(internal) != 1:
        raise RuntimeError("prediction performance and validation tables require one row")
    if int(performance.iloc[0]["patient_overlap_n"]) != 0 or int(
        internal.iloc[0]["patient_overlap_n"]
    ) != 0:
        raise RuntimeError("prediction source reports patient split leakage")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = []
    for key, item in bound.items():
        filename = f"{key.partition(':')[2]}_source_data.csv"
        item.frame.to_csv(out_dir / filename, index=False)
        source_files.append(filename)

    palette = apply_publication_style(font_size=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)

    fpr, tpr, _ = roc_curve(outcomes.to_numpy(dtype=int), probabilities.to_numpy())
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=palette["blue"], linewidth=1.6)
    ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=0.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate")
    ax.set_title(f"Discrimination (AUROC {float(performance.iloc[0]['auroc']):.3f})", loc="left", pad=12)
    add_panel_label(ax, "A", x=-0.12, y=1.04)

    precision, recall, _ = precision_recall_curve(
        outcomes.to_numpy(dtype=int), probabilities.to_numpy()
    )
    ax = axes[0, 1]
    ax.plot(recall, precision, color=palette["orange"], linewidth=1.6)
    ax.axhline(float(outcomes.mean()), linestyle="--", color="#777777", linewidth=0.8)
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title(
        f"Precision-recall (AP {float(performance.iloc[0]['average_precision']):.3f})",
        loc="left",
        pad=12,
    )
    add_panel_label(ax, "B", x=-0.12, y=1.04)

    bins = calibration.loc[calibration["row_role"].astype(str).eq("calibration_bin")]
    if bins.empty:
        raise RuntimeError("calibration assessment has no calibration bins")
    predicted = _finite(bins, "mean_predicted_probability")
    observed = _finite(bins, "observed_event_rate")
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=0.8)
    ax.plot(predicted, observed, "o-", color=palette["blue"], linewidth=1.3)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted risk", ylabel="Observed risk")
    ax.set_title("Calibration in validation data", loc="left", pad=12)
    add_panel_label(ax, "C", x=-0.12, y=1.04)

    metric_names = ("AUROC", "Average precision", "1 − Brier")
    metric_values = (
        float(performance.iloc[0]["auroc"]),
        float(performance.iloc[0]["average_precision"]),
        1.0 - float(performance.iloc[0]["brier_score"]),
    )
    if not np.isfinite(metric_values).all():
        raise RuntimeError("prediction performance metrics are not finite")
    ax = axes[1, 1]
    positions = np.arange(len(metric_names))
    ax.barh(positions, metric_values, color=(palette["blue"], palette["orange"], palette["blue_soft"]))
    ax.set_yticks(positions, metric_names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Metric value")
    ax.set_title(
        f"Validation n={int(internal.iloc[0]['evaluation_n'])}; overlap=0",
        loc="left",
        pad=12,
    )
    add_panel_label(ax, "D", x=-0.12, y=1.04)

    evidence = {key: item.evidence_id for key, item in bound.items()}
    panel_specs = (
        ("A", "Discrimination", "model_performance", (PREDICTION_SCORES_PRODUCT, PREDICTION_PERFORMANCE_PRODUCT)),
        ("B", "Precision-recall", "model_performance", (PREDICTION_SCORES_PRODUCT, PREDICTION_PERFORMANCE_PRODUCT)),
        ("C", "Calibration", "calibration", (PREDICTION_CALIBRATION_PRODUCT,)),
        ("D", "Validation context", "validation", (PREDICTION_PERFORMANCE_PRODUCT, PREDICTION_INTERNAL_VALIDATION_PRODUCT)),
    )
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The fixed analysis-only model is evaluated on a patient-separated "
            "validation partition with source-bound discrimination and calibration."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=178.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": title,
                "role": role,
                "claim": (
                    "This panel renders only the named validation products; it "
                    "does not establish external validity or clinical benefit."
                ),
                "evidence_ids": [evidence[source] for source in sources],
                "metadata": {
                    "source_products": list(sources),
                    "source_data": [
                        f"{source.partition(':')[2]}_source_data.csv"
                        for source in sources
                    ],
                },
            }
            for panel_id, title, role, sources in panel_specs
        ],
        source_data=source_files,
        statistics_note=(
            "ROC and precision-recall coordinates are deterministic projections "
            "of the sealed validation scores. Calibration bins and split counts "
            "come from their independently registered tables. Results remain "
            "analysis_only and do not demonstrate transportability."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / figure_product,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)
    for item in bound.values():
        if sha256_file(item.path) != item.sha256:
            raise RuntimeError(f"typed figure input changed: {item.input_key}")
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_static_prediction_composite_figure",
        "analysis_family": "prediction",
        "deterministic_standard_analysis": PREDICTION_FIGURE_ANALYSIS_KIND,
        "authority_scope": "analysis_only",
        "paper_authorization_allowed": False,
        "rendering_only": True,
        "source_inputs": list(PREDICTION_COMPOSITE_FIGURE_INPUTS),
        "input_bindings": [
            {
                "input_key": key,
                "evidence_id": item.evidence_id,
                "sha256": item.sha256,
                "loaded": True,
                "row_count": item.row_count,
            }
            for key, item in bound.items()
        ],
        "source_data_files": source_files,
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "PREDICTION_COMPOSITE_FIGURE_INPUTS",
    "PREDICTION_FIGURE_ANALYSIS_KIND",
    "prediction_figure_executor_code",
    "prediction_figure_executor_owns_step",
    "run_prediction_figure",
]
