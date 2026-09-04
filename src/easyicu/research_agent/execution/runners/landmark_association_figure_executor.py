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
from ...figures.display_labels import display_label
from ...figures.robustness import (
    draw_robustness_coverage,
    prepare_robustness_coverage,
)
from ...icu_rules import classify_variable
from ...schema import AnalysisStep
from .figure_input_capability import TypedInputCapability
from .typed_input_binding import BoundTypedInput, load_typed_input, sha256_file


_REQUIRED_COLUMNS = {
    "curve": frozenset(
        {
            "exposure",
            "adjusted_odds_ratio",
            "ci_low",
            "ci_high",
            "exposure_density_n",
            "exposure_density_fraction",
        }
    ),
    "adjusted_risk_curve": frozenset(
        {
            "exposure",
            "adjusted_absolute_risk",
            "ci_low",
            "ci_high",
            "exposure_density_n",
            "exposure_density_fraction",
        }
    ),
    "table:absolute_risk_context": frozenset(
        {"label", "estimate_type", "estimate", "ci_low", "ci_high"}
    ),
    "table:robustness_summary": frozenset({"axis", "total_specs", "converged_specs"}),
    "table:robustness_matrix": frozenset({"spec_id", "axis", "converged"}),
    "measurement_process": frozenset({"concept", "n_total", "measured_one_n"}),
}

_LEGACY_LANDMARK_ARTICLE_INPUTS = frozenset(
    {
        "table:absolute_risk_context",
        "table:robustness_matrix",
        "table:robustness_summary",
    }
)


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
    reserved = {
        "table:robustness_summary",
    }
    adjusted_risk = _adjusted_risk_input(inputs)
    matches = [
        value
        for value in inputs
        if value.startswith("table:")
        and value not in reserved
        and value != adjusted_risk
        and value.partition(":")[2]
        not in {"measurement_process", "measurement_process_audit"}
    ]
    return matches[0] if len(matches) == 1 else None


def _adjusted_risk_input(inputs: list[str] | tuple[str, ...]) -> str | None:
    accepted_tokens = (
        "adjusted_absolute_risk",
        "standardized_absolute_risk",
        "standardised_absolute_risk",
        "absolute_risk_curve",
    )
    matches = [
        value
        for value in inputs
        if value.startswith("table:")
        and any(token in value.partition(":")[2] for token in accepted_tokens)
    ]
    return matches[0] if len(matches) == 1 else None


def _measurement_input(inputs: list[str] | tuple[str, ...]) -> str | None:
    matches = [
        value
        for value in inputs
        if value.startswith("table:")
        and value.partition(":")[2]
        in {"measurement_process", "measurement_process_audit"}
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
    if (
        len(values) == len(_LEGACY_LANDMARK_ARTICLE_INPUTS)
        and len(values) == len(set(values))
        and set(values) == _LEGACY_LANDMARK_ARTICLE_INPUTS
    ):
        return values
    curve = _curve_input(values)
    adjusted_risk = _adjusted_risk_input(values)
    measurement = _measurement_input(values)
    if (
        curve is None
        or adjusted_risk is None
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
    legacy_profile = set(profile) == _LEGACY_LANDMARK_ARTICLE_INPUTS
    curve = None if legacy_profile else _curve_input(profile)
    adjusted_risk = None if legacy_profile else _adjusted_risk_input(profile)
    measurement = None if legacy_profile else _measurement_input(profile)
    return all(
        _binding_has_columns(
            resolved_bindings.get(key),
            _REQUIRED_COLUMNS[
                "curve"
                if key == curve
                else "adjusted_risk_curve"
                if key == adjusted_risk
                else "measurement_process"
                if key == measurement
                else key
            ],
        )
        for key in profile
    ) and (
        legacy_profile
        or (
            curve is not None
            and adjusted_risk is not None
            and measurement is not None
        )
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
            panel_placements={{{", ".join(f"{panel.panel_id!r}: {panel.placement!r}" for panel in step.figure_panels)}}},
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
    """Derive a reader-facing summary, concept, and unit from its field."""

    unit_suffixes = {
        "_mmol_l": "mmol/L",
        "_mg_dl": "mg/dL",
        "_mg_l": "mg/L",
        "_g_dl": "g/dL",
    }
    token = str(column or "").strip().lower()
    unit: str | None = None
    for suffix, candidate_unit in unit_suffixes.items():
        if token.endswith(suffix):
            token = token[: -len(suffix)]
            unit = candidate_unit
            break
    summary_match = re.search(r"_(max|min|mean|median|first|last|value)$", token)
    summary = summary_match.group(1) if summary_match else None
    if summary_match:
        token = token[: summary_match.start()]
    clinical_names = {
        "lact": "Lactate",
        "bili": "Bilirubin",
    }
    concept = clinical_names.get(token, _label(token).title())
    summary_labels = {
        "max": "Maximum",
        "min": "Minimum",
        "mean": "Mean",
        "median": "Median",
        "first": "First",
        "last": "Last",
        "value": "",
    }
    if summary and summary_labels[summary]:
        concept = f"{summary_labels[summary]} {concept.lower()}"
    if unit is None:
        unit = classify_variable(str(column or ""), "float64").unit
    return f"{concept} ({unit})" if unit else concept


def _configure_ratio_y_axis(ax: Any, *, lows: np.ndarray, highs: np.ndarray) -> None:
    """Use a compact plain-number log axis for ratio-scale estimates."""

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    positive = np.concatenate([lows[lows > 0], highs[highs > 0]])
    if not positive.size:
        raise ValueError("ratio-scale interval requires positive bounds")
    lower_limit = float(positive.min()) / 1.06
    upper_limit = float(positive.max()) * 1.06
    candidate_ticks = (
        0.1,
        0.2,
        0.3,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        20.0,
    )
    visible = [
        tick for tick in candidate_ticks if lower_limit <= tick <= upper_limit
    ]
    if 1.0 not in visible and lower_limit <= 1.0 <= upper_limit:
        visible.append(1.0)
    if len(visible) > 5:
        visible = [visible[0], *visible[1:-1:2], visible[-1]][:5]
        if lower_limit <= 1.0 <= upper_limit and 1.0 not in visible:
            visible = sorted({*visible[:4], 1.0})
    ax.set_yscale("log")
    ax.set_ylim(lower_limit, upper_limit)
    ax.yaxis.set_major_locator(FixedLocator(sorted(set(visible))))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _draw_exposure_distribution(
    ax: Any,
    *,
    x: np.ndarray,
    fractions: np.ndarray,
    color: str,
    exposure_label: str,
) -> None:
    """Draw the exact published grid-bin distribution as a quiet strip."""

    if len(x) != len(fractions) or not len(x):
        raise ValueError("exposure density must align with the curve grid")
    if (fractions < 0).any() or not np.isfinite(fractions).all():
        raise ValueError("exposure density fractions must be finite and non-negative")
    if not np.isclose(float(fractions.sum()), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("displayed exposure density fractions must sum to one")
    ax.fill_between(
        x,
        100.0 * fractions,
        color=color,
        alpha=0.24,
        linewidth=0,
        step="mid",
    )
    ax.plot(x, 100.0 * fractions, color=color, alpha=0.72, linewidth=0.55)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0, max(float((100.0 * fractions).max()) * 1.14, 1.0))
    ax.set_yticks([])
    ax.set_xlabel(exposure_label, labelpad=3)
    ax.text(
        0.0,
        1.02,
        "Exposure distribution",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.8,
        color="#616971",
        clip_on=False,
    )
    for name in ("left", "right", "top"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color("#C9CED3")
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(axis="x", labelsize=5.8, length=2.2, width=0.55)


def _run_legacy_landmark_article_figure(
    *,
    out_dir: Path,
    step_id: str,
    figure_product: str,
    profile: tuple[str, ...],
    bound: Mapping[str, BoundTypedInput],
) -> dict[str, Any]:
    """Render the already-approved three-table article display safely.

    This compatibility path deliberately preserves the reviewed plan and its
    exact input edges.  It projects registered rows only; it never reopens the
    cohort, model, or estimand during crash recovery.
    """

    risk = bound["table:absolute_risk_context"].frame.copy()
    robustness = bound["table:robustness_summary"].frame.copy()
    matrix = bound["table:robustness_matrix"].frame.copy()
    for key, frame in (
        ("table:absolute_risk_context", risk),
        ("table:robustness_summary", robustness),
        ("table:robustness_matrix", matrix),
    ):
        missing = _REQUIRED_COLUMNS[key] - set(frame.columns)
        if missing:
            raise ValueError(f"{key} is missing required columns: {sorted(missing)!r}")

    shown_risk = risk.loc[
        risk["estimate_type"].astype(str).isin(["outcome_risk", "prevalence"])
    ].copy()
    if shown_risk.empty:
        raise ValueError("absolute-risk context has no displayable estimate rows")
    _require_finite_columns(shown_risk, ("estimate", "ci_low", "ci_high"))

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
    fig = plt.figure(figsize=(183 / 25.4, 112 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.72))
    ax_context = fig.add_subplot(grid[0, 0])
    ax_robustness = fig.add_subplot(grid[0, 1])
    ax_matrix = fig.add_subplot(grid[1, :])

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
        valid = np.isfinite(values) & np.isfinite(lower) & np.isfinite(upper)
        if not valid.any():
            continue
        # Numerical rounding can put a bound a few ulps inside its estimate.
        # Error-bar geometry is display-only, so clip that tiny distance to 0.
        xerr = np.vstack(
            [
                np.maximum(values[valid] - lower[valid], 0.0),
                np.maximum(upper[valid] - values[valid], 0.0),
            ]
        )
        ax_context.barh(
            positions[valid] + offset,
            values[valid],
            height=height,
            color=color,
            xerr=xerr,
            capsize=2.2,
            label=legend_label,
        )
    ax_context.set_yticks(
        positions,
        [_measurement_state_label(value) for value in group_keys],
        fontsize=5.8,
    )
    ax_context.invert_yaxis()
    ax_context.set_xlabel("Percent")
    ax_context.set_title("Observed data context", loc="left", pad=7)
    ax_context.legend(frameon=False, fontsize=5.4, loc="lower right")
    add_panel_label(ax_context, "a", x=-0.15, y=1.04, fontsize=8.0)

    robustness_display = draw_robustness_coverage(
        ax_robustness,
        robustness,
        color=palette["blue"],
        label_formatter=lambda value: display_label(value),
    )
    add_panel_label(ax_robustness, "b", x=-0.12, y=1.04, fontsize=8.0)

    matrix_display = matrix.copy()
    matrix_display["converged_bool"] = (
        matrix_display["converged"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    matrix_display = matrix_display.reset_index(drop=True)
    matrix_positions = np.arange(len(matrix_display), dtype=float)
    colors = [
        palette["blue"] if value else palette["neutral"]
        for value in matrix_display["converged_bool"]
    ]
    ax_matrix.barh(matrix_positions, np.ones(len(matrix_display)), color=colors)
    ax_matrix.set_yticks(
        matrix_positions,
        [display_label(value) for value in matrix_display["spec_id"]],
        fontsize=5.6,
    )
    ax_matrix.set_xlim(0, 1)
    ax_matrix.set_xticks([])
    ax_matrix.invert_yaxis()
    ax_matrix.set_title("Prespecified checks completed", loc="left", pad=7)
    add_panel_label(ax_matrix, "c", x=-0.07, y=1.04, fontsize=8.0)

    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "Registered source tables show observed risk context and completion of the prespecified robustness checks."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=112.0,
        panels=[
            {
                "panel_id": "absolute_risk_context",
                "title": "Observed data context",
                "role": "descriptive_result",
                "claim": "Direct projection of registered prevalence and observed outcome-risk rows.",
                "evidence_ids": [evidence["table:absolute_risk_context"]],
                "metadata": {
                    "chart_type": "dot_interval_absolute_risk",
                    "source_products": ["table:absolute_risk_context"],
                    "estimate_geometry": "direct_table_projection",
                    "source_data": ["absolute_risk_context_source_data.csv"],
                },
            },
            {
                "panel_id": "robustness_summary",
                "title": "Robustness coverage",
                "role": "robustness",
                "claim": "Registered, converged, and independent specification counts; heterogeneous effects are not pooled.",
                "evidence_ids": [evidence["table:robustness_summary"]],
                "metadata": {
                    "chart_type": robustness_display["chart_type"],
                    "source_products": ["table:robustness_summary"],
                    "source_data": ["robustness_summary_source_data.csv"],
                    **robustness_display,
                },
            },
            {
                "panel_id": "robustness_matrix",
                "title": "Prespecified checks completed",
                "role": "robustness",
                "claim": "Each registered specification is shown once by execution status only.",
                "evidence_ids": [evidence["table:robustness_matrix"]],
                "metadata": {
                    "chart_type": "status_matrix",
                    "source_products": ["table:robustness_matrix"],
                    "source_data": ["robustness_matrix_source_data.csv"],
                },
            },
        ],
        source_data=source_files,
        statistics_note=(
            "All plotted values are direct projections of registered rows; no model is fit by the renderer. "
            "Robustness estimates with different scientific meanings are not compared on one effect axis."
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
        "analysis_family": "descriptive",
        "method": "deterministic_legacy_landmark_article_figure",
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
        "supplementary_panel_ids": [],
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


def run_landmark_association_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
    input_keys: tuple[str, ...],
    panel_placements: Mapping[str, str] | None = None,
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
    if set(profile) == _LEGACY_LANDMARK_ARTICLE_INPUTS:
        return _run_legacy_landmark_article_figure(
            out_dir=out_dir,
            step_id=step_id,
            figure_product=figure_product,
            profile=profile,
            bound=bound,
        )
    curve_key = _curve_input(profile)
    adjusted_risk_key = _adjusted_risk_input(profile)
    measurement_key = _measurement_input(profile)
    assert (
        curve_key is not None
        and adjusted_risk_key is not None
        and measurement_key is not None
    )
    curve = bound[curve_key].frame.copy()
    adjusted_risk = bound[adjusted_risk_key].frame.copy()
    robustness = bound["table:robustness_summary"].frame.copy()
    process = bound[measurement_key].frame.copy()
    for key, frame in (
        (curve_key, curve),
        (adjusted_risk_key, adjusted_risk),
        ("table:robustness_summary", robustness),
        (measurement_key, process),
    ):
        required = _REQUIRED_COLUMNS[
            "curve"
            if key == curve_key
            else "adjusted_risk_curve"
            if key == adjusted_risk_key
            else "measurement_process"
            if key == measurement_key
            else key
        ]
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{key} is missing required columns: {sorted(missing)!r}")
    exposure_column, reference_column = _exposure_columns(curve)
    _require_finite_columns(
        curve,
        (
            exposure_column,
            reference_column,
            "adjusted_odds_ratio",
            "ci_low",
            "ci_high",
            "exposure_density_n",
            "exposure_density_fraction",
        ),
    )
    risk_exposure_column, risk_reference_column = _exposure_columns(adjusted_risk)
    _require_finite_columns(
        adjusted_risk,
        (
            risk_exposure_column,
            risk_reference_column,
            "adjusted_absolute_risk",
            "ci_low",
            "ci_high",
            "exposure_density_n",
            "exposure_density_fraction",
        ),
    )
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
    placements = dict(panel_placements or {})
    # The registered article contract keeps audit-only coverage and routine
    # measurement availability outside the primary scientific figure.  Their
    # source tables remain exported and clickable in the run dossier.
    show_process = placements.get("measurement_process", "supplementary") == "main"
    show_robustness = placements.get("robustness_summary", "supplementary") == "main"
    if show_process or show_robustness:
        raise ValueError(
            "landmark audit panels require a supplementary display, not the primary curve figure"
        )
    figure_height_mm = 78.0
    fig = plt.figure(
        figsize=(183 / 25.4, figure_height_mm / 25.4),
    )
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(5.2, 0.72),
        hspace=0.12,
        wspace=0.32,
        left=0.08,
        right=0.985,
        bottom=0.19,
        top=0.90,
    )
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_risk = fig.add_subplot(grid[0, 1])
    ax_curve_density = fig.add_subplot(grid[1, 0], sharex=ax_curve)
    ax_risk_density = fig.add_subplot(grid[1, 1], sharex=ax_risk)

    ax = ax_curve
    display_curve = curve.sort_values(exposure_column, kind="stable")
    x = pd.to_numeric(display_curve[exposure_column]).to_numpy(dtype=float)
    y = pd.to_numeric(display_curve["adjusted_odds_ratio"]).to_numpy(dtype=float)
    low = pd.to_numeric(display_curve["ci_low"]).to_numpy(dtype=float)
    high = pd.to_numeric(display_curve["ci_high"]).to_numpy(dtype=float)
    density = pd.to_numeric(
        display_curve["exposure_density_fraction"]
    ).to_numpy(dtype=float)
    ax.fill_between(
        x,
        low,
        high,
        color=palette["blue_soft"],
        alpha=0.50,
        linewidth=0,
    )
    ax.plot(x, y, color=palette["blue"], linewidth=1.45)
    ax.axhline(1.0, color="#7A8188", linestyle=(0, (3, 3)), linewidth=0.75)
    _configure_ratio_y_axis(ax, lows=low, highs=high)
    references = pd.to_numeric(
        display_curve[reference_column], errors="coerce"
    ).dropna()
    if references.nunique() != 1:
        raise ValueError("association curve requires one reference exposure value")
    reference = float(references.iloc[0])
    source_exposures = {
        str(value or "").strip() for value in display_curve["exposure"]
    }
    if "" in source_exposures or len(source_exposures) != 1:
        raise ValueError("association curve requires one named exposure")
    source_exposure = next(iter(source_exposures))
    risk_source_exposures = {
        str(value or "").strip() for value in adjusted_risk["exposure"]
    }
    if risk_source_exposures != {source_exposure}:
        raise ValueError("adjusted association and absolute-risk exposure names differ")
    exposure_label = _continuous_exposure_label(source_exposure)
    ax.axvline(reference, color="#7A8188", linestyle=(0, (3, 3)), linewidth=0.75)
    ax.text(
        reference,
        0.98,
        f"Reference {reference:g}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=4.8,
        color="#616971",
    )
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylabel("Adjusted odds ratio (95% CI)")
    ax.set_title(
        "Adjusted association",
        loc="left",
        pad=4,
        fontsize=7.0,
        fontweight="semibold",
    )
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="y", color="#E7EAED", linewidth=0.5)
    ax.set_axisbelow(True)
    add_panel_label(ax, "a", x=-0.08, y=1.04, fontsize=7.0)
    _draw_exposure_distribution(
        ax_curve_density,
        x=x,
        fractions=density,
        color=palette["blue"],
        exposure_label=exposure_label,
    )

    display_risk = adjusted_risk.sort_values(
        risk_exposure_column, kind="stable"
    )
    risk_x = pd.to_numeric(display_risk[risk_exposure_column]).to_numpy(dtype=float)
    risk_reference = pd.to_numeric(
        display_risk[risk_reference_column]
    ).to_numpy(dtype=float)
    risk_values = 100.0 * pd.to_numeric(
        display_risk["adjusted_absolute_risk"]
    ).to_numpy(dtype=float)
    risk_low = 100.0 * pd.to_numeric(display_risk["ci_low"]).to_numpy(dtype=float)
    risk_high = 100.0 * pd.to_numeric(display_risk["ci_high"]).to_numpy(dtype=float)
    risk_density = pd.to_numeric(
        display_risk["exposure_density_fraction"]
    ).to_numpy(dtype=float)
    if not np.allclose(risk_x, x, rtol=0.0, atol=1e-10):
        raise ValueError("adjusted association and absolute-risk grids do not align")
    if not np.allclose(risk_reference, reference, rtol=0.0, atol=1e-10):
        raise ValueError("adjusted association and absolute-risk references do not align")
    if not np.allclose(risk_density, density, rtol=0.0, atol=1e-10):
        raise ValueError("adjusted association and absolute-risk densities do not align")
    if (risk_low < 0).any() or (risk_high > 100).any():
        raise ValueError("model-standardised absolute-risk intervals must be in [0, 1]")

    ax = ax_risk
    risk_color = "#2C7F86"
    ax.fill_between(
        risk_x,
        risk_low,
        risk_high,
        color="#D9EFEE",
        alpha=0.72,
        linewidth=0,
    )
    ax.plot(risk_x, risk_values, color=risk_color, linewidth=1.45)
    ax.axvline(reference, color="#7A8188", linestyle=(0, (3, 3)), linewidth=0.75)
    risk_span = max(float(risk_high.max() - risk_low.min()), 0.5)
    ax.set_ylim(
        max(0.0, float(risk_low.min()) - 0.08 * risk_span),
        min(100.0, float(risk_high.max()) + 0.08 * risk_span),
    )
    ax.set_xlim(float(risk_x.min()), float(risk_x.max()))
    ax.set_ylabel("Model-standardised outcome risk (%)")
    ax.set_title(
        "Absolute risk",
        loc="left",
        pad=4,
        fontsize=7.0,
        fontweight="semibold",
    )
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(axis="y", color="#E7EAED", linewidth=0.5)
    ax.set_axisbelow(True)
    add_panel_label(ax, "b", x=-0.08, y=1.04, fontsize=7.0)
    _draw_exposure_distribution(
        ax_risk_density,
        x=risk_x,
        fractions=risk_density,
        color=risk_color,
        exposure_label=exposure_label,
    )

    # Validate the supplementary audit sources even though they do not compete
    # with the primary result for visual salience.
    prepare_robustness_coverage(robustness)
    robustness_display = {
        "chart_type": "sensitivity_coverage_matrix",
        "effect_comparison_authorized": False,
        "reason_code": "ROBUSTNESS_EFFECT_COMPARABILITY_UNRESOLVED",
        "display_authority": "audit_only",
    }
    denominator = pd.to_numeric(process["n_total"])
    numerator = pd.to_numeric(process["measured_one_n"])
    if (
        (denominator <= 0).any()
        or (numerator < 0).any()
        or (numerator > denominator).any()
    ):
        raise ValueError("measurement-process counts do not nest")

    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    panel_templates = landmark_association_composite_panels(profile)
    panels = tuple(
        panel
        for panel in panel_templates
        if placements.get(panel.panel_id, panel.placement) == "main"
    )
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "The aligned main panels show the adjusted ratio-scale association and model-standardised absolute outcome risk with 95% confidence intervals across the prespecified exposure grid. "
            "The source-backed distribution strips show where the complete-case cohort contributes information; audit-only coverage and measurement-process tables remain supplementary."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=figure_height_mm,
        panels=[
            {
                "panel_id": panel.panel_id,
                "title": (
                    "Sensitivity-analysis coverage"
                    if panel.panel_id == "robustness_summary"
                    else _label(panel.panel_id)
                ),
                "role": panel.article_role,
                "claim": (
                    "This audit panel reports registered, converged, and independent specification counts without comparing heterogeneous effects."
                    if panel.panel_id == "robustness_summary"
                    else "This panel renders the complete registered source table without model refitting."
                ),
                "evidence_ids": [evidence[source] for source in panel.source_products],
                "metadata": {
                    "chart_type": (
                        robustness_display["chart_type"]
                        if panel.panel_id == "robustness_summary"
                        else panel.chart_type
                    ),
                    "source_products": list(panel.source_products),
                    "estimate_geometry": (
                        "continuous_fitted_curve_with_95ci"
                        if panel.panel_id
                        in {"association_curve", "absolute_risk_curve"}
                        else "direct_table_projection"
                    ),
                    "source_data": [
                        f"{source.partition(':')[2]}_source_data.csv"
                        for source in panel.source_products
                    ],
                    **(
                        robustness_display
                        if panel.panel_id == "robustness_summary"
                        else {}
                    ),
                },
            }
            for panel in panels
        ],
        source_data=source_files,
        statistics_note=(
            "All plotted values and exposure-grid densities are direct projections of registered source rows; no model is fit and no patient rows are read by the renderer. "
            "The two curves share one exposure grid and reference value. Robustness summaries remain audit-only counts and are not displayed as confidence intervals or comparable effects."
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
        "supplementary_panel_ids": sorted(
            panel.panel_id
            for panel in panel_templates
            if placements.get(panel.panel_id, panel.placement) == "supplementary"
        ),
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
