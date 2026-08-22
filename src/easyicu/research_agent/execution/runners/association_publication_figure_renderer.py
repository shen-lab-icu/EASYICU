"""Code-backed renderer for the association publication four-table profile."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from .typed_input_binding import BoundTypedInput, sha256_file


def _finite(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column!r} must contain only finite numeric values")
    return values.astype(float)


def _integers(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _finite(frame, column)
    if not np.isclose(values, np.rint(values), rtol=0.0, atol=1e-9).all():
        raise ValueError(f"{column!r} must contain only integer-like values")
    return values.astype("int64")


def _label(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "Not reported"
    return re.sub(r"[_\s]+", " ", str(value).strip()) or "Not reported"


def _source_copy(bound: BoundTypedInput, out_dir: Path) -> str:
    source = bound.frame.copy()
    source.insert(0, "source_row_index", source.index.astype(int))
    source.insert(1, "source_table", bound.path.name)
    name = f"{bound.product}_source_data.csv"
    source.to_csv(out_dir / name, index=False)
    return name


def _validate_interval_table(
    frame: pd.DataFrame,
    *,
    estimate_column: str,
    require_fitted: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    result[estimate_column] = _finite(result, estimate_column)
    result["ci_low"] = _finite(result, "ci_low")
    result["ci_high"] = _finite(result, "ci_high")
    if (result["ci_low"] > result[estimate_column]).any() or (
        result[estimate_column] > result["ci_high"]
    ).any():
        raise ValueError("confidence intervals must contain their point estimates")
    if require_fitted and not result["fit_status"].astype(str).eq("fitted").all():
        raise ValueError("adjusted association rows must all have fit_status='fitted'")
    return result


def _forest(
    ax: Any,
    frame: pd.DataFrame,
    *,
    estimate_column: str,
    label_column: str,
    title: str,
    color: str,
) -> None:
    positions = np.arange(len(frame))
    estimates = frame[estimate_column].to_numpy(dtype=float)
    errors = np.vstack(
        [
            estimates - frame["ci_low"].to_numpy(dtype=float),
            frame["ci_high"].to_numpy(dtype=float) - estimates,
        ]
    )
    ax.errorbar(estimates, positions, xerr=errors, fmt="o", color=color, capsize=2.5)
    ax.set_yticks(
        positions, [_label(value) for value in frame[label_column]], fontsize=5.8
    )
    ax.invert_yaxis()
    scales = {str(value).strip().lower() for value in frame["effect_scale"]}
    if scales and scales <= {
        "or",
        "odds_ratio",
        "hazard_ratio",
        "risk_ratio",
        "hr",
        "rr",
    }:
        if (frame[[estimate_column, "ci_low", "ci_high"]] <= 0).any().any():
            raise ValueError("ratio-scale estimates and intervals must be positive")
        ax.axvline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel(_label(next(iter(scales), "estimate")))
    ax.set_title(title, loc="left", pad=12)


def render_association_publication_figure(
    *,
    bound: Mapping[str, BoundTypedInput],
    out_dir: Path,
    step_id: str,
    figure_product: str,
    input_keys: tuple[str, ...],
    display_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Render all four bound products without fitting or selecting a model."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distribution = bound["table:exposure_outcome_distribution"].frame.copy()
    adjusted = _validate_interval_table(
        bound["table:adjusted_association_estimates"].frame,
        estimate_column="estimate",
        require_fitted=True,
    )
    robustness = _validate_interval_table(
        bound["table:robustness_matrix"].frame,
        estimate_column="point_estimate",
    )
    missingness = bound["table:measurement_missingness"].frame.copy()

    levels = distribution.loc[
        distribution["row_role"].astype(str).eq("exposure_level")
    ].copy()
    if levels.empty:
        raise ValueError("exposure/outcome distribution has no exposure-level rows")
    for column in ("outcome_events", "outcome_denominator"):
        levels[column] = _integers(levels, column)
    levels["outcome_rate_pct"] = _finite(levels, "outcome_rate_pct")
    if (levels["outcome_denominator"] <= 0).any() or (
        levels["outcome_events"] > levels["outcome_denominator"]
    ).any():
        raise ValueError("outcome counts do not nest within positive denominators")
    expected_rates = 100.0 * levels["outcome_events"] / levels["outcome_denominator"]
    if not np.isclose(
        levels["outcome_rate_pct"], expected_rates, rtol=0.0, atol=5e-6
    ).all():
        raise ValueError("outcome percentage does not reconcile to counts")
    has_risk_ci = {"ci_low_pct", "ci_high_pct"} <= set(levels.columns)
    if has_risk_ci:
        levels["ci_low_pct"] = _finite(levels, "ci_low_pct")
        levels["ci_high_pct"] = _finite(levels, "ci_high_pct")
        if (levels["ci_low_pct"] > levels["outcome_rate_pct"]).any() or (
            levels["outcome_rate_pct"] > levels["ci_high_pct"]
        ).any():
            raise ValueError("risk confidence intervals must contain reported rates")

    for column in ("n_total", "missing_n"):
        missingness[column] = _integers(missingness, column)
    missingness["missing_pct"] = _finite(missingness, "missing_pct")
    if (missingness["n_total"] <= 0).any() or (
        missingness["missing_n"] > missingness["n_total"]
    ).any():
        raise ValueError("missingness counts do not nest within positive denominators")
    expected_missing = 100.0 * missingness["missing_n"] / missingness["n_total"]
    if not np.isclose(
        missingness["missing_pct"], expected_missing, rtol=0.0, atol=5e-6
    ).all():
        raise ValueError("missingness percentage does not reconcile to counts")
    if not robustness["converged"].astype(bool).all():
        raise ValueError("robustness matrix contains non-converged rows")

    source_files = [_source_copy(bound[key], out_dir) for key in input_keys]
    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    labels = dict(display_labels or {})
    palette = apply_publication_style(font_size=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(levels))
    values = levels["outcome_rate_pct"].to_numpy(dtype=float)
    yerr = None
    if has_risk_ci:
        yerr = np.vstack(
            [values - levels["ci_low_pct"], levels["ci_high_pct"] - values]
        )
    ax.bar(x, values, color=palette["orange"], yerr=yerr, capsize=2.5)
    exposure_name = str(levels.iloc[0].get("exposure_column") or "exposure")
    level_labels = []
    for value in levels["exposure_level"]:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        raw = (
            str(int(numeric))
            if pd.notna(numeric) and float(numeric).is_integer()
            else str(value)
        )
        level_labels.append(labels.get(f"{exposure_name}={raw}", _label(value)))
    ax.set_xticks(x, level_labels)
    ax.set_ylabel("Observed outcome risk (%)")
    ax.set_title("Absolute risk by exposure level", loc="left", pad=12)
    add_panel_label(ax, "A", x=-0.12, y=1.04)

    _forest(
        axes[0, 1],
        adjusted,
        estimate_column="estimate",
        label_column="model_id",
        title="Primary adjusted association",
        color=palette["blue"],
    )
    add_panel_label(axes[0, 1], "B", x=-0.12, y=1.04)

    robustness_labels = "spec_id" if "spec_id" in robustness.columns else "axis"
    _forest(
        axes[1, 0],
        robustness,
        estimate_column="point_estimate",
        label_column=robustness_labels,
        title="Robustness estimates",
        color=palette["blue_soft"],
    )
    add_panel_label(axes[1, 0], "C", x=-0.12, y=1.04)

    missing_order = missingness.sort_values("missing_pct", ascending=True)
    missing_label = "label" if "label" in missing_order.columns else "variable"
    positions = np.arange(len(missing_order))
    axes[1, 1].barh(positions, missing_order["missing_pct"], color=palette["orange"])
    axes[1, 1].set_yticks(
        positions,
        [_label(value) for value in missing_order[missing_label]],
        fontsize=5.5,
    )
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_xlabel("Missing (%)")
    axes[1, 1].set_title("Measurement missingness", loc="left", pad=12)
    add_panel_label(axes[1, 1], "D", x=-0.12, y=1.04)

    panel_specs = (
        ("A", "Absolute risk by exposure level", "absolute_risk", input_keys[0]),
        ("B", "Primary adjusted association", "primary_estimate", input_keys[1]),
        ("C", "Robustness estimates", "robustness", input_keys[2]),
        ("D", "Measurement missingness", "data_quality", input_keys[3]),
    )
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The bound tables jointly show observed absolute risk, the primary "
            "adjusted association, robustness estimates, and measurement missingness."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=178.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": title,
                "role": role,
                "claim": f"This panel visualizes values from {source} without refitting.",
                "evidence_ids": [evidence[source]],
                "metadata": {
                    "source_products": [source],
                    "source_data": [f"{source.partition(':')[2]}_source_data.csv"],
                },
            }
            for panel_id, title, role, source in panel_specs
        ],
        source_data=source_files,
        statistics_note=(
            "All source rows and original columns are preserved in source-data files. "
            "The renderer performs no model fitting or scientific row selection."
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
            raise ValueError(f"typed input changed while rendering: {item.input_key}")
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_composite_association_figure",
        "analysis_family": "association",
        "deterministic_standard_analysis": "composite_association_figure",
        "rendering_only": True,
        "source_inputs": list(input_keys),
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
        "figure_files": [
            path.name for key, path in outputs.items() if key != "contract"
        ],
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["render_association_publication_figure"]
