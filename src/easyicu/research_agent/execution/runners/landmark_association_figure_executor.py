"""Deterministic four-table display for a landmark spline association."""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...contracts.figure_plan import (
    LANDMARK_ASSOCIATION_COMPOSITE_INPUTS,
    landmark_association_composite_panels,
)
from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .figure_input_capability import TypedInputCapability
from .typed_input_binding import BoundTypedInput, load_typed_input, sha256_file


_REQUIRED_COLUMNS = {
    "curve": frozenset(
        {
            "adjusted_odds_ratio",
            "ci_low",
            "ci_high",
        }
    ),
    "table:absolute_risk_context": frozenset(
        {"label", "estimate_type", "estimate", "ci_low", "ci_high"}
    ),
    "table:robustness_summary": frozenset(
        {"axis", "total_specs", "converged_specs", "range_low", "range_high"}
    ),
    "measurement_process": frozenset(
        {"concept", "n_total", "measured_one_n"}
    ),
}


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != "figure"
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", product)
    ):
        return None
    return product


def _curve_input(inputs: list[str] | tuple[str, ...]) -> str | None:
    matches = [
        value
        for value in inputs
        if value.startswith("table:")
        and value.partition(":")[2].endswith("landmark_rcs_curve")
    ]
    return matches[0] if len(matches) == 1 else None


def _measurement_input(inputs: list[str] | tuple[str, ...]) -> str | None:
    matches = [
        value
        for value in inputs
        if value.startswith("table:")
        and value.partition(":")[2] in {"measurement_process", "measurement_process_audit"}
    ]
    return matches[0] if len(matches) == 1 else None


def _exposure_columns(frame: pd.DataFrame) -> tuple[str, str]:
    pairs = [
        (column.removeprefix("reference_"), column)
        for column in frame.columns
        if column.startswith("reference_")
        and column.removeprefix("reference_") in frame.columns
    ]
    if len(pairs) != 1:
        raise ValueError("curve table requires one exposure/reference column pair")
    return pairs[0]


def landmark_association_figure_input_profile(
    inputs: list[str] | tuple[str, ...],
) -> tuple[str, ...] | None:
    values = tuple(str(value or "").strip() for value in inputs)
    curve = _curve_input(values)
    measurement = _measurement_input(values)
    if (
        curve is None
        or measurement is None
        or len(values) != 4
        or len(values) != len(set(values))
        or not LANDMARK_ASSOCIATION_COMPOSITE_INPUTS <= set(values)
    ):
        return None
    return values


def _binding_has_columns(binding: Any, columns: frozenset[str]) -> bool:
    if not isinstance(binding, Mapping):
        return False
    contract = binding.get("product_contract")
    declared = contract.get("columns") if isinstance(contract, Mapping) else None
    return bool(isinstance(declared, list) and columns <= set(declared))


def landmark_association_figure_executor_owns_step(
    step: AnalysisStep,
    *,
    resolved_bindings: Mapping[str, Any] | None = None,
) -> bool:
    profile = landmark_association_figure_input_profile(tuple(step.inputs))
    product = (
        _figure_product(step.expected_outputs[0])
        if len(step.expected_outputs) == 1
        else None
    )
    if (
        profile is None
        or product is None
        or step.planned_analysis_role != "auxiliary"
        or str(step.method or "").strip().lower().split(" with ", 1)[0]
        != "visualization"
        or not TypedInputCapability(required=frozenset(profile)).admits_step(step)
        or not isinstance(resolved_bindings, Mapping)
        or set(resolved_bindings) != set(profile)
    ):
        return False
    curve = _curve_input(profile)
    measurement = _measurement_input(profile)
    assert curve is not None and measurement is not None
    return all(
        _binding_has_columns(
            resolved_bindings.get(key),
            _REQUIRED_COLUMNS[
                "curve" if key == curve else "measurement_process" if key == measurement else key
            ],
        )
        for key in profile
    )


def landmark_association_figure_executor_code(step: AnalysisStep) -> str:
    product = (
        _figure_product(step.expected_outputs[0])
        if len(step.expected_outputs) == 1
        else None
    )
    profile = landmark_association_figure_input_profile(tuple(step.inputs))
    if product is None or profile is None:
        raise ValueError("landmark association figure contract is incomplete")
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from easyicu.research_agent.execution.runners.landmark_association_figure_executor import run_landmark_association_figure

        run_landmark_association_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
            input_keys={profile!r},
        )
        """
    ).strip()


def _load(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    input_keys: tuple[str, ...],
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
        for key in input_keys
    }


def _require_finite_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column!r} must contain finite numeric values")


def _label(value: Any) -> str:
    return re.sub(r"[_\s]+", " ", str(value or "").strip()) or "Value"


def _measurement_state_label(value: Any) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if token in {"observed", "measured", "source_present", "with_source"}:
        return "Measured"
    if token in {"no_source", "not_measured", "unmeasured", "source_absent"}:
        return "Not measured"
    return _label(value)


def _continuous_exposure_label(column: str) -> str:
    """Derive a reader-facing exposure label and common unit from its field."""

    unit_suffixes = {
        "_mmol_l": "mmol/L",
        "_mg_dl": "mg/dL",
        "_mg_l": "mg/L",
        "_g_dl": "g/dL",
    }
    token = str(column or "").strip().lower()
    for suffix, unit in unit_suffixes.items():
        if token.endswith(suffix):
            name = token[: -len(suffix)]
            return f"{_label(name).title()} ({unit})"
    return _label(column).title()


def run_landmark_association_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    input_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Render four exact source tables without fitting or filtering a model."""

    if _figure_product(f"figure:{figure_product}") is None:
        raise ValueError("unsafe figure product")
    profile = landmark_association_figure_input_profile(input_keys)
    if profile is None:
        raise ValueError("unsupported landmark association figure profile")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bound = _load(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
        input_keys=profile,
    )
    curve_key = _curve_input(profile)
    measurement_key = _measurement_input(profile)
    assert curve_key is not None and measurement_key is not None
    curve = bound[curve_key].frame.copy()
    risk = bound["table:absolute_risk_context"].frame.copy()
    robustness = bound["table:robustness_summary"].frame.copy()
    process = bound[measurement_key].frame.copy()
    for key, frame in (
        (curve_key, curve),
        ("table:absolute_risk_context", risk),
        ("table:robustness_summary", robustness),
        (measurement_key, process),
    ):
        required = _REQUIRED_COLUMNS[
            "curve" if key == curve_key else "measurement_process" if key == measurement_key else key
        ]
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{key} is missing required columns: {sorted(missing)!r}")
    exposure_column, reference_column = _exposure_columns(curve)
    _require_finite_columns(
        curve,
        (exposure_column, reference_column, "adjusted_odds_ratio", "ci_low", "ci_high"),
    )
    shown_risk = risk.loc[
        risk["estimate_type"].astype(str).isin(["outcome_risk", "prevalence"])
    ].copy()
    if shown_risk.empty:
        raise ValueError("absolute-risk context has no displayable estimate rows")
    _require_finite_columns(shown_risk, ("estimate", "ci_low", "ci_high"))
    _require_finite_columns(robustness, ("range_low", "range_high"))
    _require_finite_columns(process, ("n_total", "measured_one_n"))

    source_files: list[str] = []
    for key, item in bound.items():
        name = f"{key.partition(':')[2]}_source_data.csv"
        source = item.frame.copy()
        source.insert(0, "source_row_index", source.index.astype(int))
        source.insert(1, "source_table", item.path.name)
        source.to_csv(out_dir / name, index=False)
        source_files.append(name)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    display_curve = curve.sort_values(exposure_column, kind="stable")
    x = pd.to_numeric(display_curve[exposure_column]).to_numpy(dtype=float)
    y = pd.to_numeric(display_curve["adjusted_odds_ratio"]).to_numpy(dtype=float)
    low = pd.to_numeric(display_curve["ci_low"]).to_numpy(dtype=float)
    high = pd.to_numeric(display_curve["ci_high"]).to_numpy(dtype=float)
    ax.fill_between(x, low, high, color=palette["blue_soft"], alpha=0.45, linewidth=0)
    ax.plot(x, y, color=palette["blue"], linewidth=1.5)
    ax.axhline(1.0, color=palette["neutral"], linestyle="--", linewidth=0.8)
    references = pd.to_numeric(
        display_curve[reference_column], errors="coerce"
    ).dropna()
    exposure_label = _continuous_exposure_label(exposure_column)
    if references.nunique() == 1:
        reference = float(references.iloc[0])
        exposure_label = (
            f"{exposure_label[:-1]}; reference {reference:g})"
            if exposure_label.endswith(")")
            else f"{exposure_label} (reference {reference:g})"
        )
    ax.set_xlabel(exposure_label)
    ax.set_ylabel("Adjusted odds ratio")
    ax.set_title("Landmark association curve", loc="left", pad=12)
    add_panel_label(ax, "A", x=-0.12, y=1.04)

    ax = axes[0, 1]
    group_column = "group_value" if "group_value" in shown_risk.columns else "label"
    display = shown_risk.copy()
    display["group_key"] = display[group_column].astype(str)
    group_keys = list(dict.fromkeys(display["group_key"].tolist()))
    positions = np.arange(len(group_keys), dtype=float)
    height = 0.34
    series_specs = (
        ("prevalence", "Cohort share", palette["blue_soft"], -height / 2),
        ("outcome_risk", "Observed outcome risk", palette["orange"], height / 2),
    )
    for estimate_type, legend_label, color, offset in series_specs:
        subset = display.loc[display["estimate_type"].astype(str).eq(estimate_type)]
        if subset["group_key"].duplicated().any():
            raise ValueError(
                f"absolute-risk context repeats {estimate_type!r} within a group"
            )
        by_group = subset.set_index("group_key")
        values = np.array(
            [
                100.0 * float(by_group.loc[key, "estimate"])
                if key in by_group.index
                else np.nan
                for key in group_keys
            ],
            dtype=float,
        )
        valid = np.isfinite(values)
        if not valid.any():
            continue
        lower = np.array(
            [
                100.0 * float(by_group.loc[key, "ci_low"])
                if key in by_group.index
                else np.nan
                for key in group_keys
            ],
            dtype=float,
        )
        upper = np.array(
            [
                100.0 * float(by_group.loc[key, "ci_high"])
                if key in by_group.index
                else np.nan
                for key in group_keys
            ],
            dtype=float,
        )
        xerr = np.vstack([values[valid] - lower[valid], upper[valid] - values[valid]])
        ax.barh(
            positions[valid] + offset,
            values[valid],
            height=height,
            color=color,
            xerr=xerr,
            capsize=2.2,
            label=legend_label,
        )
    ax.set_yticks(
        positions,
        [_measurement_state_label(value) for value in group_keys],
        fontsize=5.8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Percent")
    ax.set_title("Measurement state and outcome risk", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    add_panel_label(ax, "B", x=-0.12, y=1.04)

    ax = axes[1, 0]
    total_specs = pd.to_numeric(robustness["total_specs"]).to_numpy(dtype=float)
    converged_specs = pd.to_numeric(robustness["converged_specs"]).to_numpy(
        dtype=float
    )
    if (
        (total_specs <= 0).any()
        or (converged_specs < 0).any()
        or (converged_specs > total_specs).any()
    ):
        raise ValueError("robustness specification counts do not nest")
    convergence_pct = 100.0 * converged_specs / total_specs
    positions = np.arange(len(robustness))
    ax.barh(positions, convergence_pct, color=palette["blue_soft"])
    for position, pct, low_value, high_value in zip(
        positions,
        convergence_pct,
        pd.to_numeric(robustness["range_low"]),
        pd.to_numeric(robustness["range_high"]),
        strict=True,
    ):
        ax.text(
            min(float(pct) + 2.0, 98.0),
            position,
            f"range {low_value:.2f}–{high_value:.2f}",
            va="center",
            ha="right" if pct > 80 else "left",
            fontsize=5.7,
        )
    ax.set_yticks(positions, [_label(value) for value in robustness["axis"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Converged specifications (%)")
    ax.set_title("Sensitivity execution audit", loc="left", pad=12)
    ax.text(
        0.0,
        -0.18,
        "Estimate ranges are within-axis; estimands may differ.",
        transform=ax.transAxes,
        fontsize=5.4,
        color=palette["neutral"],
    )
    add_panel_label(ax, "C", x=-0.12, y=1.04)

    ax = axes[1, 1]
    denominator = pd.to_numeric(process["n_total"])
    numerator = pd.to_numeric(process["measured_one_n"])
    if (
        (denominator <= 0).any()
        or (numerator < 0).any()
        or (numerator > denominator).any()
    ):
        raise ValueError("measurement-process counts do not nest")
    pct = 100.0 * numerator / denominator
    positions = np.arange(len(process))
    ax.barh(positions, pct, color=palette["blue_soft"])
    ax.set_yticks(
        positions, [_label(value) for value in process["concept"]], fontsize=5.8
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Measured at least once (%)")
    ax.set_title("Measurement process", loc="left", pad=12)
    add_panel_label(ax, "D", x=-0.12, y=1.04)

    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    panels = landmark_association_composite_panels(profile)
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The four typed source tables jointly describe the landmark association, absolute-risk context, robustness, and measurement process."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=178.0,
        panels=[
            {
                "panel_id": panel.panel_id,
                "title": _label(panel.panel_id),
                "role": panel.article_role,
                "claim": "This panel renders the complete registered source table without model refitting.",
                "evidence_ids": [evidence[source] for source in panel.source_products],
                "metadata": {
                    "chart_type": panel.chart_type,
                    "source_products": list(panel.source_products),
                    "estimate_geometry": (
                        "continuous_fitted_curve_with_95ci"
                        if panel.panel_id == "association_curve"
                        else "direct_table_projection"
                    ),
                    "source_data": [
                        f"{source.partition(':')[2]}_source_data.csv"
                        for source in panel.source_products
                    ],
                },
            }
            for panel in panels
        ],
        source_data=source_files,
        statistics_note="All plotted values are direct projections of registered source rows; no model is fit by the renderer.",
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
        "analysis_family": "descriptive",
        "method": "deterministic_landmark_association_composite_figure",
        "deterministic_standard_analysis": "landmark_association_composite_figure",
        "rendering_only": True,
        "source_inputs": list(profile),
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


__all__ = [
    "landmark_association_figure_executor_code",
    "landmark_association_figure_executor_owns_step",
    "landmark_association_figure_input_profile",
    "run_landmark_association_figure",
]
