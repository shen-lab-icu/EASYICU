"""Sensitivity publication rendering from registered parent products."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd

from .prior_output_support import (
    publication_label as _publication_label,
    short_figure_label as _short_figure_label,
)
from .prior_output_contracts import _resolve_upstream_analysis_method
from ..reporting.publication_bundles import (
    _explicit_false_figure_value, _sensitivity_plot_label, _truthy_figure_value,
)


def _render_sensitivity_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
    authorized_repair_id: Optional[str] = None,
) -> Optional[str]:
    """Deterministically rebuild a sensitivity figure from parent outputs."""

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None

    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_sources: List[Tuple[Path, Union[Path, bytes]]] = []
    direct_outputs = steps_dir / parent_step_id / "outputs"
    if parent_step_id and parent_step_id != current_step_id and direct_outputs.exists():
        # A split rendering step must not silently borrow a similarly-shaped
        # table from an unrelated sensitivity step.  Search the direct parent
        # only when it exists.  Older plans whose rendering-step name does not
        # share the exact parent stem retain the conservative fallback below.
        if preverified_parent_artifacts is None:
            direct_candidates = [
                (path, path) for path in sorted(direct_outputs.glob("*.csv"))
            ]
        elif authorized_repair_id == (
            "sensitivity_publication_bundle_from_locked_summary_v1"
        ):
            payload = preverified_parent_artifacts.get("robustness_summary.csv")
            if (
                payload is None
                or "step_summary.json" not in preverified_parent_artifacts
            ):
                return None
            direct_candidates = [(direct_outputs / "robustness_summary.csv", payload)]
        else:
            direct_candidates = [
                (direct_outputs / name, payload)
                for name, payload in sorted(preverified_parent_artifacts.items())
                if Path(name).name == name and Path(name).suffix.lower() == ".csv"
            ]
        declared_names: set[str] = set()
        try:
            summary_payload = (
                preverified_parent_artifacts.get("step_summary.json")
                if preverified_parent_artifacts is not None
                else None
            )
            direct_summary = json.loads(
                summary_payload.decode("utf-8")
                if summary_payload is not None
                else (direct_outputs / "step_summary.json").read_text(encoding="utf-8")
            )
        except Exception:
            direct_summary = {}
        if isinstance(direct_summary, dict):
            for mapping_key in ("output_files", "aliases"):
                mapping = direct_summary.get(mapping_key)
                if isinstance(mapping, dict):
                    declared_items = mapping.items()
                elif isinstance(mapping, list):
                    declared_items = ((str(value), value) for value in mapping)
                else:
                    continue
                for alias, value in declared_items:
                    if not any(
                        token in str(alias).lower()
                        for token in ("robustness", "sensitivity")
                    ):
                        continue
                    if isinstance(value, str) and value.lower().endswith(".csv"):
                        declared_names.add(Path(value).name)
        candidate_sources.extend(
            sorted(
                direct_candidates,
                key=lambda item: (
                    item[0].name not in declared_names,
                    item[0].name,
                ),
            )
        )
    elif preverified_parent_artifacts is None:
        for step_dir in sorted(steps_dir.iterdir()):
            if not step_dir.is_dir() or step_dir.name == current_step_id:
                continue
            if "sensitivity" not in step_dir.name.lower():
                continue
            outputs_dir = step_dir / "outputs"
            if outputs_dir.exists():
                candidate_sources.extend(
                    (path, path) for path in sorted(outputs_dir.glob("*.csv"))
                )

    parent: Optional[tuple[Path, pd.DataFrame]] = None
    for csv_path, source in candidate_sources:
        try:
            # Preserve round-trippable confidence-limit values; the default
            # fast parser can change the final binary digit and create a false
            # source-trace mismatch on an otherwise identical table.
            frame = pd.read_csv(
                io.BytesIO(source) if isinstance(source, bytes) else source,
                float_precision="round_trip",
            )
        except Exception:
            continue
        required = {"spec_id", "effect_scale", "point_estimate", "ci_low", "ci_high"}
        if required <= set(frame.columns):
            parent = (csv_path, frame)
            break
    if parent is None:
        return None

    table_path, frame = parent
    source_step_id = table_path.parents[1].name
    source_data = frame.copy()
    source_data["source_table"] = table_path.name
    source_data["source_step_id"] = source_step_id
    for col in (
        "point_estimate",
        "ci_low",
        "ci_high",
        "modeled_analytic_n",
        "event_n",
        "membership_n",
    ):
        if col in source_data.columns:
            source_data[col] = pd.to_numeric(source_data[col], errors="coerce")
    if "modeled_analytic_n" not in source_data.columns:
        for count_alias in ("analysis_n", "n"):
            if count_alias in source_data.columns:
                source_data["modeled_analytic_n"] = pd.to_numeric(
                    source_data[count_alias], errors="coerce"
                )
                break
    for count_col in ("modeled_analytic_n", "event_n", "membership_n"):
        if count_col not in source_data.columns:
            continue
        numeric = pd.to_numeric(source_data[count_col], errors="coerce")
        finite = numeric.dropna()
        if finite.empty or ((finite % 1) == 0).all():
            source_data[count_col] = numeric.astype("Int64")
    if "display_label" not in source_data.columns:
        source_data["display_label"] = source_data["spec_id"].map(_publication_label)
    if "axis" not in source_data.columns:
        source_data["axis"] = "sensitivity"
    if "converged" not in source_data.columns:
        source_data["converged"] = source_data["point_estimate"].notna()
    source_data["axis_label"] = source_data["axis"].map(_publication_label)
    source_data["plot_label"] = [
        _sensitivity_plot_label(row) for row in source_data.to_dict(orient="records")
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    estimated_mask = (
        source_data[["point_estimate", "ci_low", "ci_high"]].notna().all(axis=1)
    )
    if "converged" in source_data.columns:
        estimated_mask &= source_data["converged"].map(_truthy_figure_value)
    if "reportable" in source_data.columns:
        estimated_mask &= source_data["reportable"].map(_truthy_figure_value)
    if "independent_variant" in source_data.columns:
        independent = source_data["independent_variant"]
        estimated_mask &= ~independent.map(_explicit_false_figure_value)
    plot_df = source_data.loc[estimated_mask].copy()
    if plot_df.empty:
        return None
    ratio_df = plot_df[
        plot_df["effect_scale"].astype(str).str.upper().isin({"OR", "RR", "HR"})
    ].copy()
    additive_scales = {
        "RD",
        "RISK_DIFFERENCE",
        "MEAN_DIFFERENCE",
        "MEDIAN_DIFFERENCE",
        "CONDITIONAL_MEAN_DIFFERENCE",
        "CONDITIONAL_MEDIAN_DIFFERENCE",
    }
    rd_df = plot_df[
        plot_df["effect_scale"].astype(str).str.upper().isin(additive_scales)
    ].copy()
    plotted_indexes = ratio_df.index.union(rd_df.index)
    figure_source_data = source_data.loc[plotted_indexes].copy()
    estimability_source_data = source_data.drop(index=plotted_indexes).copy()
    estimability_source_data = estimability_source_data.drop(
        columns=[
            "modeled_analytic_n",
            "model_contract_n",
            "event_n",
            "model_id",
            "source_model_id",
            "exposure_source",
            "exposure_expression",
            "exposure_role",
            "analysis_role",
            "analysis_set",
            "baseline_missing_policy",
            "fit_status",
            "fit_method",
            "replay_mode",
            "coefficient_source_table",
            "coefficient_term",
            "model_contract_source",
            "source_script_sha256",
        ],
        errors="ignore",
    )
    figure_source_data.to_csv(
        out_dir / "sensitivity_forest_source_data.csv",
        index=False,
    )
    estimability_source_filename: Optional[str] = None
    if not estimability_source_data.empty:
        estimability_source_filename = "sensitivity_estimability_source_data.csv"
        estimability_source_data.to_csv(
            out_dir / estimability_source_filename,
            index=False,
        )
    if not rd_df.empty:
        rd_df["plot_label"] = [
            _sensitivity_plot_label(row) for row in rd_df.to_dict(orient="records")
        ]
    n_df = figure_source_data.copy()
    if "modeled_analytic_n" in n_df.columns:
        n_df["modeled_analytic_n"] = pd.to_numeric(
            n_df["modeled_analytic_n"],
            errors="coerce",
        )
    else:
        n_df["modeled_analytic_n"] = pd.NA
    n_df = n_df[n_df["modeled_analytic_n"].fillna(0).gt(0)].copy()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from easyicu.research_agent.figures.publication import (
        add_panel_label,
        apply_publication_style,
        make_figure_contract,
        save_publication_figure,
    )

    palette = apply_publication_style()
    if ratio_df.empty and rd_df.empty:
        return None
    n_plot = n_df.dropna(subset=["modeled_analytic_n"]).copy()
    max_rows = max(len(ratio_df), len(rd_df), len(n_plot), 1)
    figure_height_mm = float(max(88, min(145, 24 + 15 * max_rows)))
    fig = plt.figure(
        figsize=(183 / 25.4, figure_height_mm / 25.4),
        constrained_layout=False,
    )
    ax_ratio = None
    ax_rd = None
    ax_n = None
    if not ratio_df.empty and not rd_df.empty:
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.42, 0.92],
            height_ratios=[1.0, 0.82],
            left=0.28,
            right=0.98,
            top=0.92,
            bottom=0.17,
            wspace=0.78,
            hspace=0.62,
        )
        ax_ratio = fig.add_subplot(grid[:, 0])
        ax_rd = fig.add_subplot(grid[0, 1])
        if not n_plot.empty:
            ax_n = fig.add_subplot(grid[1, 1])
    else:
        has_denominator = not n_plot.empty
        grid = fig.add_gridspec(
            1,
            2 if has_denominator else 1,
            width_ratios=[1.42, 0.92] if has_denominator else [1.0],
            left=0.25 if has_denominator else 0.30,
            right=0.98,
            top=0.90,
            bottom=0.20,
            wspace=0.82,
        )
        effect_axis = fig.add_subplot(grid[0, 0])
        if not ratio_df.empty:
            ax_ratio = effect_axis
        else:
            ax_rd = effect_axis
        if has_denominator:
            ax_n = fig.add_subplot(grid[0, 1])

    def _plot_interval_panel(
        ax,
        data: pd.DataFrame,
        *,
        title: str,
        xlabel: str,
        null_value: float,
        color: str,
    ) -> None:
        data = data.reset_index(drop=True)
        y = list(range(len(data)))
        center = data["point_estimate"].astype(float).to_numpy()
        lo = data["ci_low"].astype(float).to_numpy()
        hi = data["ci_high"].astype(float).to_numpy()
        labels = [
            _short_figure_label(label)
            for label in data["plot_label"].fillna(data["display_label"]).astype(str)
        ]
        ax.errorbar(
            center,
            y,
            xerr=[
                [max(0.0, c - lower) for c, lower in zip(center, lo)],
                [max(0.0, h - c) for c, h in zip(center, hi)],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markersize=3.9,
        )
        ax.axvline(
            null_value,
            color=palette.get("neutral", "#8F8F8F"),
            linestyle="--",
            linewidth=0.8,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=4)
        ax.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )

    contract_panels: List[Dict[str, Any]] = []
    next_panel_ord = ord("A")

    def _next_panel_id() -> str:
        nonlocal next_panel_ord
        panel_id = chr(next_panel_ord)
        next_panel_ord += 1
        return panel_id

    source_evidence = ["sensitivity_forest_source_data.csv"]
    all_source_evidence = list(source_evidence)
    if estimability_source_filename:
        all_source_evidence.append(estimability_source_filename)
    if ax_ratio is not None:
        ratio_scales = sorted(
            set(ratio_df["effect_scale"].dropna().astype(str).str.upper())
        )
        ratio_xlabel = {
            ("OR",): "Adjusted odds ratio (95% CI)",
            ("RR",): "Adjusted risk ratio (95% CI)",
            ("HR",): "Hazard ratio (95% CI)",
        }.get(tuple(ratio_scales), "Ratio estimate (95% CI)")
        panel_id = _next_panel_id()
        _plot_interval_panel(
            ax_ratio,
            ratio_df,
            title="Ratio-scale sensitivity",
            xlabel=ratio_xlabel,
            null_value=1.0,
            color=palette.get("blue", "#0F4D92"),
        )
        add_panel_label(ax_ratio, panel_id, x=-0.24)
        contract_panels.append(
            {
                "panel_id": panel_id,
                "title": "Ratio-scale sensitivity",
                "role": "robustness",
                "claim": (
                    "Converged, independently estimable ratio-scale sensitivity "
                    "estimates are read from the registered parent table."
                ),
                "evidence_ids": source_evidence,
            }
        )

    if ax_rd is not None:
        additive_values = sorted(
            set(rd_df["effect_scale"].dropna().astype(str).str.upper())
        )
        if set(additive_values) <= {"RD", "RISK_DIFFERENCE"}:
            additive_title = "Risk-difference sensitivity"
            additive_xlabel = "Risk difference (95% CI)"
        elif additive_values and all("MEDIAN" in value for value in additive_values):
            additive_title = "Median-difference sensitivity"
            additive_xlabel = "Adjusted median difference (95% CI)"
        elif additive_values and all("MEAN" in value for value in additive_values):
            additive_title = "Mean-difference sensitivity"
            additive_xlabel = "Adjusted mean difference (95% CI)"
        else:
            additive_title = "Additive-scale sensitivity"
            additive_xlabel = "Adjusted difference (95% CI)"
        panel_id = _next_panel_id()
        _plot_interval_panel(
            ax_rd,
            rd_df,
            title=additive_title,
            xlabel=additive_xlabel,
            null_value=0.0,
            color=palette.get("green", "#008B5E"),
        )
        add_panel_label(ax_rd, panel_id, x=-0.24, y=1.06, fontsize=10.0)
        contract_panels.append(
            {
                "panel_id": panel_id,
                "title": additive_title,
                "role": "robustness",
                "claim": (
                    "Converged, reportable additive-scale sensitivity estimates "
                    "are shown on their declared scale."
                ),
                "evidence_ids": source_evidence,
            }
        )

    non_independent_count = 0
    if "independent_variant" in estimability_source_data.columns:
        non_independent_count = int(
            estimability_source_data["independent_variant"]
            .map(_explicit_false_figure_value)
            .sum()
        )
    if ax_n is not None:
        n_plot = n_plot.reset_index(drop=True)
        y_n = list(range(len(n_plot)))
        colors = [
            (
                palette.get("blue", "#0F4D92")
                if _truthy_figure_value(value)
                else palette.get("neutral_light", "#D8D8D8")
            )
            for value in n_plot["converged"].fillna(False)
        ]
        ax_n.barh(
            y_n,
            n_plot["modeled_analytic_n"].astype(float),
            color=colors,
            height=0.56,
        )
        ax_n.set_yticks(y_n)
        ax_n.set_yticklabels(
            [
                _short_figure_label(label, limit=26)
                for label in n_plot["plot_label"]
                .fillna(n_plot["display_label"])
                .astype(str)
            ]
        )
        ax_n.invert_yaxis()
        if non_independent_count:
            # Reserve an in-axis status row below the final denominator bar;
            # placing the note at a negative axes fraction pushes SVG text
            # outside the canvas even when the raster preview looks acceptable.
            ax_n.set_ylim(len(n_plot) + 0.75, -0.6)
        event_values = (
            pd.to_numeric(n_plot["event_n"], errors="coerce")
            if "event_n" in n_plot.columns
            else pd.Series([pd.NA] * len(n_plot))
        )
        max_n = float(n_plot["modeled_analytic_n"].max())
        if event_values.notna().any():
            for row_index, (analytic_n, event_n) in enumerate(
                zip(n_plot["modeled_analytic_n"], event_values)
            ):
                if pd.isna(event_n):
                    continue
                ax_n.text(
                    float(analytic_n) + max_n * 0.015,
                    row_index,
                    f"{int(event_n):,} events",
                    va="center",
                    fontsize=6.0,
                    color=palette.get("baseline", "#272727"),
                )
            ax_n.set_xlim(0, max_n * 1.29)
            ax_n.set_xlabel("Analytic sample size")
        else:
            ax_n.set_xlabel("Analytic sample size")
        ax_n.set_title("Model denominator audit", loc="left", pad=4)
        ax_n.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
            alpha=0.8,
        )
        if non_independent_count:
            ax_n.text(
                0.0,
                len(n_plot) + 0.25,
                f"Non-independent outcome variants: {non_independent_count}",
                ha="left",
                va="center",
                fontsize=6.0,
                color=palette.get("neutral", "#8F8F8F"),
            )
        panel_id = _next_panel_id()
        add_panel_label(ax_n, panel_id, x=-0.24, y=1.06, fontsize=10.0)
        contract_panels.append(
            {
                "panel_id": panel_id,
                "title": "Model denominator audit",
                "role": "audit",
                "claim": (
                    "Positive analytic sample sizes and available event counts "
                    "are shown for fitted sensitivity models; non-independent "
                    "variants are reported separately rather than encoded as N=0."
                ),
                "evidence_ids": all_source_evidence,
            }
        )

    for panel in contract_panels:
        panel_role = str(panel.get("role") or "")
        if panel_role == "robustness":
            panel["metadata"] = {"planner_product_slots": ["robustness_plot"]}
        elif panel_role == "audit":
            panel["metadata"] = {
                "planner_product_slots": ["robustness_denominator_audit"]
            }

    contract = make_figure_contract(
        figure_id="sensitivity_forest",
        core_claim=(
            "Pre-specified sensitivity estimates are rendered from the "
            "registered sensitivity-comparison table with effect-scale and "
            "denominator context."
        ),
        panels=contract_panels,
        height_mm=figure_height_mm,
        source_data=all_source_evidence,
        statistics_note=(
            "Generated deterministically from the registered parent-step "
            "sensitivity-comparison table after the rendering step lacked a "
            "canonical figure contract."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "sensitivity_forest",
        contract=contract,
        dpi=300,
    )
    plt.close(fig)

    step_summary_path = out_dir / "step_summary.json"
    existing_summary: Dict[str, Any] = {}
    if step_summary_path.exists():
        try:
            loaded = json.loads(step_summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing_summary = loaded
        except Exception:
            existing_summary = {}
    existing_summary.update(
        {
            "step_id": current_step_id,
            "method": "deterministic_sensitivity_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": source_step_id,
            "source_sensitivity_table": str(table_path),
            "source_data_csv": str(out_dir / "sensitivity_forest_source_data.csv"),
            "source_data_files": all_source_evidence,
            "n_rows_plotted": int(len(figure_source_data)),
            "n_denominator_rows": int(len(n_plot)),
            "n_non_independent_variants": non_independent_count,
            "source_model_ids": sorted(
                set(
                    figure_source_data.get("model_id", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                )
            ),
            "effect_scales_plotted": sorted(
                set(figure_source_data["effect_scale"].dropna().astype(str))
            ),
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "sensitivity_forest.png",
            "figure_contract": "sensitivity_forest.figure_contract.json",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if table_path.name == "robustness_summary.csv" and (
        authorized_repair_id == "sensitivity_publication_bundle_from_locked_summary_v1"
        or _resolve_upstream_analysis_method(run_dir, current_step_id)
        == "cohort_definition_sensitivity"
    ):
        return "sensitivity_publication_bundle_from_locked_summary_v1"
    return "sensitivity_publication_bundle_from_parent_outputs_v2"
