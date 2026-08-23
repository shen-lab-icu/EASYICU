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


def _association_finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{column!r} must contain only finite numeric values")
    return values.astype(float)


def _integers(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _association_finite_series(frame, column)
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
    result[estimate_column] = _association_finite_series(result, estimate_column)
    result["ci_low"] = _association_finite_series(result, "ci_low")
    result["ci_high"] = _association_finite_series(result, "ci_high")
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


def _robustness_ranges(ax: Any, frame: pd.DataFrame, *, color: str) -> None:
    """Render reported robustness ranges without inventing point estimates."""

    result = frame.copy()
    for column in ("total_specs", "converged_specs", "non_independent_specs"):
        result[column] = _integers(result, column)
    result["range_low"] = _association_finite_series(result, "range_low")
    result["range_high"] = _association_finite_series(result, "range_high")
    if (result["total_specs"] <= 0).any():
        raise ValueError("robustness summary total_specs must be positive")
    if (
        (result["converged_specs"] < 0).any()
        or (result["converged_specs"] > result["total_specs"]).any()
        or (result["non_independent_specs"] < 0).any()
        or (result["non_independent_specs"] > result["total_specs"]).any()
    ):
        raise ValueError("robustness summary counts do not nest within total_specs")
    if (result["range_low"] > result["range_high"]).any():
        raise ValueError("robustness summary ranges are reversed")

    positions = np.arange(len(result))
    for position, (_, row) in zip(positions, result.iterrows()):
        ax.plot(
            [float(row["range_low"]), float(row["range_high"])],
            [position, position],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
        )
    ax.set_yticks(positions, [_label(value) for value in result["axis"]], fontsize=5.8)
    ax.invert_yaxis()
    if (result[["range_low", "range_high"]] > 0).all().all():
        ax.axvline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Reported estimate range")
    ax.set_title("Robustness ranges", loc="left", pad=12)


def _absolute_risk_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["estimate_type"].astype(str).eq("outcome_risk")].copy()
    if result.empty:
        raise ValueError("absolute-risk context has no outcome_risk rows")
    result["n"] = _integers(result, "n")
    result["event_n"] = _integers(result, "event_n")
    result["estimate"] = _association_finite_series(result, "estimate")
    result["ci_low"] = _association_finite_series(result, "ci_low")
    result["ci_high"] = _association_finite_series(result, "ci_high")
    if (
        (result["n"] <= 0).any()
        or (result["event_n"] < 0).any()
        or (result["event_n"] > result["n"]).any()
    ):
        raise ValueError(
            "absolute-risk counts do not nest within positive denominators"
        )
    expected = result["event_n"].astype(float) / result["n"].astype(float)
    if not np.isclose(result["estimate"], expected, rtol=0.0, atol=5e-7).all():
        raise ValueError("absolute-risk estimates do not reconcile to counts")
    if (result["ci_low"] > result["estimate"]).any() or (
        result["estimate"] > result["ci_high"]
    ).any():
        raise ValueError("absolute-risk intervals must contain their estimates")
    return result


def _measurement_availability(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["measured_one_n"] = _integers(result, "measured_one_n")
    result["eligible_n"] = _integers(result, "eligible_n")
    if (result["eligible_n"] <= 0).any() or (
        result["measured_one_n"] > result["eligible_n"]
    ).any():
        raise ValueError("measurement counts do not nest within eligible denominators")
    result["availability_pct"] = 100.0 * result["measured_one_n"] / result["eligible_n"]
    return result


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

    distribution = (
        bound["table:exposure_outcome_distribution"].frame.copy()
        if "table:exposure_outcome_distribution" in bound
        else None
    )
    absolute_context = (
        _absolute_risk_context(bound["table:absolute_risk_context"].frame)
        if "table:absolute_risk_context" in bound
        else None
    )
    adjusted = _validate_interval_table(
        bound["table:adjusted_association_estimates"].frame,
        estimate_column="estimate",
        require_fitted=True,
    )
    robustness_key = (
        "table:robustness_matrix"
        if "table:robustness_matrix" in bound
        else "table:robustness_summary"
    )
    robustness = bound[robustness_key].frame.copy()
    if robustness_key == "table:robustness_matrix":
        robustness = _validate_interval_table(
            robustness,
            estimate_column="point_estimate",
        )
    missingness = (
        bound["table:measurement_missingness"].frame.copy()
        if "table:measurement_missingness" in bound
        else None
    )
    availability = (
        _measurement_availability(bound["table:measurement_process_audit"].frame)
        if "table:measurement_process_audit" in bound
        and "table:measurement_missingness" not in bound
        else None
    )
    robustness_summary = (
        bound["table:robustness_summary"].frame.copy()
        if robustness_key == "table:robustness_matrix"
        and "table:robustness_summary" in bound
        else None
    )

    if distribution is not None:
        levels = distribution.loc[
            distribution["row_role"].astype(str).eq("exposure_level")
        ].copy()
        if levels.empty:
            raise ValueError("exposure/outcome distribution has no exposure-level rows")
        for column in (
            "n_rows",
            "exposure_denominator",
            "outcome_events",
            "outcome_denominator",
        ):
            levels[column] = _integers(levels, column)
        levels["exposure_pct"] = _association_finite_series(
            levels, "exposure_pct"
        )
        levels["outcome_rate_pct"] = _association_finite_series(
            levels, "outcome_rate_pct"
        )
        if (
            (levels["exposure_denominator"] <= 0).any()
            or (levels["n_rows"] > levels["exposure_denominator"]).any()
            or (levels["outcome_denominator"] <= 0).any()
            or (levels["outcome_events"] > levels["outcome_denominator"]).any()
        ):
            raise ValueError("distribution counts do not nest within positive denominators")
        expected_prevalence = (
            100.0 * levels["n_rows"] / levels["exposure_denominator"]
        )
        expected_rates = (
            100.0 * levels["outcome_events"] / levels["outcome_denominator"]
        )
        if not np.isclose(
            levels["exposure_pct"], expected_prevalence, rtol=0.0, atol=5e-6
        ).all() or not np.isclose(
            levels["outcome_rate_pct"], expected_rates, rtol=0.0, atol=5e-6
        ).all():
            raise ValueError("distribution percentages do not reconcile to counts")
        has_risk_ci = {"ci_low_pct", "ci_high_pct"} <= set(levels.columns)
        if has_risk_ci:
            levels["ci_low_pct"] = _association_finite_series(levels, "ci_low_pct")
            levels["ci_high_pct"] = _association_finite_series(
                levels, "ci_high_pct"
            )
            if (levels["ci_low_pct"] > levels["outcome_rate_pct"]).any() or (
                levels["outcome_rate_pct"] > levels["ci_high_pct"]
            ).any():
                raise ValueError(
                    "risk confidence intervals must contain reported rates"
                )
    else:
        levels = absolute_context
        has_risk_ci = True

    if missingness is not None:
        for column in ("n_total", "missing_n"):
            missingness[column] = _integers(missingness, column)
        missingness["missing_pct"] = _association_finite_series(
            missingness, "missing_pct"
        )
        if (missingness["n_total"] <= 0).any() or (
            missingness["missing_n"] > missingness["n_total"]
        ).any():
            raise ValueError(
                "missingness counts do not nest within positive denominators"
            )
        expected_missing = 100.0 * missingness["missing_n"] / missingness["n_total"]
        if not np.isclose(
            missingness["missing_pct"], expected_missing, rtol=0.0, atol=5e-6
        ).all():
            raise ValueError("missingness percentage does not reconcile to counts")
    if (
        robustness_key == "table:robustness_matrix"
        and not robustness["converged"].astype(bool).all()
    ):
        raise ValueError("robustness matrix contains non-converged rows")

    source_files = [_source_copy(bound[key], out_dir) for key in input_keys]
    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    labels = dict(display_labels or {})
    palette = apply_publication_style(font_size=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(levels))
    if distribution is not None:
        prevalence_values = levels["exposure_pct"].to_numpy(dtype=float)
        risk_values = levels["outcome_rate_pct"].to_numpy(dtype=float)
        risk_yerr = None
        if has_risk_ci:
            risk_yerr = np.vstack(
                [
                    risk_values - levels["ci_low_pct"],
                    levels["ci_high_pct"] - risk_values,
                ]
            )
        prevalence_yerr = None
        if {"exposure_ci_low_pct", "exposure_ci_high_pct"} <= set(levels.columns):
            low = _association_finite_series(levels, "exposure_ci_low_pct")
            high = _association_finite_series(levels, "exposure_ci_high_pct")
            if (low > levels["exposure_pct"]).any() or (
                levels["exposure_pct"] > high
            ).any():
                raise ValueError(
                    "prevalence confidence intervals must contain reported prevalence"
                )
            prevalence_yerr = np.vstack(
                [prevalence_values - low, high - prevalence_values]
            )
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
        absolute_title = "Exposure prevalence and observed outcome risk"
    else:
        values = 100.0 * levels["estimate"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                values - 100.0 * levels["ci_low"].to_numpy(dtype=float),
                100.0 * levels["ci_high"].to_numpy(dtype=float) - values,
            ]
        )
        level_labels = [_label(value) for value in levels["label"]]
        absolute_title = "Absolute risk by source state"
    if distribution is not None:
        width = 0.36
        ax.bar(
            x - width / 2,
            prevalence_values,
            width,
            color=palette["blue_soft"],
            yerr=prevalence_yerr,
            capsize=2.5,
            label="Exposure prevalence",
        )
        ax.bar(
            x + width / 2,
            risk_values,
            width,
            color=palette["orange"],
            yerr=risk_yerr,
            capsize=2.5,
            label="Outcome risk",
        )
        ax.legend(frameon=False, fontsize=5.8)
    else:
        ax.bar(x, values, color=palette["orange"], yerr=yerr, capsize=2.5)
    ax.set_xticks(x, level_labels)
    ax.set_ylabel("Percent" if distribution is not None else "Observed outcome risk (%)")
    ax.set_title(absolute_title, loc="left", pad=12)
    add_panel_label(ax, "A", x=-0.12, y=1.04)

    adjusted_label = (
        "contrast"
        if "contrast" in adjusted.columns
        and adjusted["contrast"].notna().all()
        and adjusted["contrast"].astype(str).str.strip().ne("").all()
        else "model_id"
    )
    _forest(
        axes[0, 1],
        adjusted,
        estimate_column="estimate",
        label_column=adjusted_label,
        title="Primary adjusted association",
        color=palette["blue"],
    )
    add_panel_label(axes[0, 1], "B", x=-0.12, y=1.04)

    if robustness_key == "table:robustness_matrix":
        robustness_labels = "spec_id" if "spec_id" in robustness.columns else "axis"
        _forest(
            axes[1, 0],
            robustness,
            estimate_column="point_estimate",
            label_column=robustness_labels,
            title="Robustness estimates",
            color=palette["blue_soft"],
        )
        robustness_title = "Robustness estimates"
    else:
        _robustness_ranges(axes[1, 0], robustness, color=palette["blue_soft"])
        robustness_title = "Robustness ranges"
    add_panel_label(axes[1, 0], "C", x=-0.12, y=1.04)

    if missingness is not None:
        quality = missingness.sort_values("missing_pct", ascending=True)
        quality_label = "label" if "label" in quality.columns else "variable"
        quality_value = "missing_pct"
        quality_xlabel = "Missing (%)"
        quality_title = "Measurement missingness"
    elif availability is not None:
        quality = availability.sort_values("availability_pct", ascending=True)
        quality_label = "concept" if "concept" in quality.columns else "variable"
        quality_value = "availability_pct"
        quality_xlabel = "Available among eligible (%)"
        quality_title = "Measurement availability"
    elif robustness_summary is not None:
        _robustness_ranges(
            axes[1, 1],
            robustness_summary,
            color=palette["orange"],
        )
        quality_title = "Robustness ranges"
        quality_label = None
        quality_value = None
        quality_xlabel = "Reported estimate range"
    else:  # pragma: no cover - guarded by exact typed profiles
        raise ValueError("association composite has no fourth-panel source")
    if robustness_summary is None:
        positions = np.arange(len(quality))
        axes[1, 1].barh(positions, quality[quality_value], color=palette["orange"])
        axes[1, 1].set_yticks(
            positions,
            [_label(value) for value in quality[quality_label]],
            fontsize=5.5,
        )
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_xlabel(quality_xlabel)
        axes[1, 1].set_title(quality_title, loc="left", pad=12)
    add_panel_label(axes[1, 1], "D", x=-0.12, y=1.04)

    fourth_source = (
        "table:robustness_summary"
        if robustness_summary is not None
        else (
            "table:measurement_missingness"
            if missingness is not None
            else "table:measurement_process_audit"
        )
    )
    fourth_role = "robustness" if robustness_summary is not None else "data_quality"

    panel_specs = (
        (
            "A",
            absolute_title,
            "descriptive_result",
            (
                "table:exposure_outcome_distribution"
                if distribution is not None
                else "table:absolute_risk_context"
            ),
        ),
        (
            "B",
            "Primary adjusted association",
            "primary_estimand",
            "table:adjusted_association_estimates",
        ),
        ("C", robustness_title, "robustness", robustness_key),
        (
            "D",
            quality_title,
            fourth_role,
            fourth_source,
        ),
    )
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The bound tables jointly show observed absolute risk, the primary "
            "adjusted association, and the exact supporting context declared "
            "by the Planner's four-table figure contract."
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
