"""Code-backed renderer for the association publication four-table profile."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...contracts.figure_plan import (
    ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS,
    BALANCE_ASSOCIATION_COMPOSITE_INPUTS,
    COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS,
)
from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...figures.display_labels import display_label, label_lookup, scoped_label_lookup
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


def _measurement_state_label(value: Any) -> str:
    """Reader-facing label for generic measurement-source states."""

    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if token in {"observed", "measured", "source_present", "with_source"}:
        return "Measured"
    if token in {"no_source", "not_measured", "unmeasured", "source_absent"}:
        return "Not measured"
    return _label(value)


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
    label_formatter: Any | None = None,
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
    formatter = label_formatter or (lambda value: display_label(value))
    ax.set_yticks(
        positions, [formatter(value) for value in frame[label_column]], fontsize=6.3
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
    ax.set_title(title, loc="left", pad=7)


def _reader_contrast_labels(
    frame: pd.DataFrame,
    *,
    exposure_name: str | None,
    display_labels: Mapping[str, str],
) -> pd.Series | None:
    """Build reader-facing contrasts from typed levels without changing rows."""

    if not exposure_name or not {"exposure_level", "reference_level"} <= set(
        frame.columns
    ):
        return None
    labels: list[str] = []
    ordinal_scope = re.sub(r"_(max|min|first|last)$", "", exposure_name)
    for row in frame.itertuples(index=False):
        comparison_value = getattr(row, "exposure_level")
        reference_value = getattr(row, "reference_level")
        comparison = scoped_label_lookup(
            exposure_name, comparison_value, display_labels
        )
        reference = scoped_label_lookup(exposure_name, reference_value, display_labels)
        for value, current, side in (
            (comparison_value, comparison, "comparison"),
            (reference_value, reference, "reference"),
        ):
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            level_token = (
                str(int(numeric))
                if pd.notna(numeric) and float(numeric).is_integer()
                else str(value)
            )
            declared = label_lookup(f"{ordinal_scope}_{level_token}", display_labels)
            if declared:
                concise = declared.split(":", 1)[0].strip()
                if side == "comparison":
                    comparison = concise
                else:
                    reference = concise
        scope = display_label(exposure_name, display_labels)
        if comparison is None:
            comparison_token = (
                f"{comparison_value:g}"
                if isinstance(comparison_value, (int, float, np.number))
                else str(comparison_value)
            )
            comparison = f"{scope} {comparison_token}"
        if reference is None:
            reference_token = (
                f"{reference_value:g}"
                if isinstance(reference_value, (int, float, np.number))
                else str(reference_value)
            )
            reference = f"{scope} {reference_token}"
        labels.append(f"{comparison} vs {reference}")
    return pd.Series(labels, index=frame.index, dtype="string")


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


def _measurement_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["n_total"] = _integers(result, "n_total")
    result["missing_n"] = _integers(result, "missing_n")
    result["missing_pct"] = _association_finite_series(result, "missing_pct")
    if (result["n_total"] <= 0).any() or (
        result["missing_n"] > result["n_total"]
    ).any():
        raise ValueError("missingness counts do not nest within positive denominators")
    expected = 100.0 * result["missing_n"] / result["n_total"]
    if not np.isclose(result["missing_pct"], expected, rtol=0.0, atol=5e-6).all():
        raise ValueError("missingness percentage does not reconcile to counts")
    return result


def _component_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["concept"] = result["concept"].astype(str)
    result["exposure_category"] = result["exposure_category"].astype(str)
    result["n_stratum"] = _integers(result, "n_stratum")
    result["measured_n"] = _integers(result, "measured_n")
    result["measured_pct"] = _association_finite_series(result, "measured_pct")
    if (result["n_stratum"] <= 0).any() or (
        result["measured_n"] > result["n_stratum"]
    ).any():
        raise ValueError(
            "component-completeness counts do not nest within positive denominators"
        )
    expected = 100.0 * result["measured_n"] / result["n_stratum"]
    if not np.isclose(result["measured_pct"], expected, rtol=0.0, atol=5e-6).all():
        raise ValueError(
            "component-completeness percentage does not reconcile to counts"
        )
    keys = result[["concept", "exposure_category"]]
    if keys.duplicated().any():
        raise ValueError(
            "component-completeness rows must be unique by concept and exposure category"
        )
    return result


def _draw_missingness(ax: Any, frame: pd.DataFrame, *, color: str) -> None:
    quality = frame.sort_values("missing_pct", ascending=True)
    label_column = "label" if "label" in quality.columns else "variable"
    positions = np.arange(len(quality))
    ax.barh(positions, quality["missing_pct"], color=color)
    ax.set_yticks(
        positions,
        [display_label(value) for value in quality[label_column]],
        fontsize=5.5,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Measurement missingness", loc="left", pad=12)


def _draw_component_completeness(ax: Any, frame: pd.DataFrame) -> None:
    concepts = list(dict.fromkeys(frame["concept"].astype(str)))
    categories = list(dict.fromkeys(frame["exposure_category"].astype(str)))
    matrix = frame.pivot(
        index="concept",
        columns="exposure_category",
        values="measured_pct",
    ).reindex(index=concepts, columns=categories)
    if matrix.isna().any().any():
        raise ValueError(
            "component-completeness grid must contain every declared concept-category cell"
        )
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        vmin=0,
        vmax=100,
        cmap="Blues",
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(categories)),
        [display_label(value) for value in categories],
        rotation=25,
        ha="right",
        fontsize=5.5,
    )
    ax.set_yticks(
        np.arange(len(concepts)),
        [display_label(value) for value in concepts],
        fontsize=5.2,
    )
    for row_index in range(len(concepts)):
        for column_index in range(len(categories)):
            value = float(matrix.iloc[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if value >= 55 else "#202020",
            )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Measured (%)")
    ax.set_title("Component completeness", loc="left", pad=12)


def _render_cohort_balance_association_figure(
    *,
    bound: Mapping[str, BoundTypedInput],
    out_dir: Path,
    step_id: str,
    figure_product: str,
    input_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Render cohort, balance, primary estimate, and robustness without refitting."""

    flow = bound["table:cohort_flow"].frame.copy()
    flow["n_remaining"] = _integers(flow, "n_remaining")
    if (flow["n_remaining"] < 0).any():
        raise ValueError("cohort-flow counts must be non-negative")

    table_one = bound["table:table_one"].frame.copy()
    computed = table_one.loc[
        table_one["standardized_difference_status"].astype(str).eq("computed")
    ].copy()
    computed["absolute_standardized_mean_difference"] = pd.to_numeric(
        computed["absolute_standardized_mean_difference"], errors="coerce"
    )
    computed = computed.loc[
        computed["absolute_standardized_mean_difference"].notna()
    ].copy()
    if computed.empty:
        raise ValueError("Table 1 has no computed standardized differences")
    if (computed["absolute_standardized_mean_difference"] < 0).any() or not np.isfinite(
        computed["absolute_standardized_mean_difference"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Table 1 standardized differences must be finite and non-negative"
        )
    balance = (
        computed.groupby("variable", as_index=False)[
            "absolute_standardized_mean_difference"
        ]
        .max()
        .sort_values("absolute_standardized_mean_difference", ascending=True)
    )

    adjusted = _validate_interval_table(
        bound["table:adjusted_association_estimates"].frame,
        estimate_column="estimate",
        require_fitted=True,
    )
    robustness = _validate_interval_table(
        bound["table:robustness_matrix"].frame,
        estimate_column="point_estimate",
    )
    if not robustness["converged"].astype(bool).all():
        raise ValueError("robustness matrix contains non-converged rows")

    source_files = [_source_copy(bound[key], out_dir) for key in input_keys]
    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    palette = apply_publication_style(font_size=7.0)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(183 / 25.4, 132 / 25.4),
        gridspec_kw={"width_ratios": (1.18, 1.0), "height_ratios": (1.12, 0.88)},
        constrained_layout=True,
    )

    positions = np.arange(len(flow))
    axes[0, 0].barh(positions, flow["n_remaining"], color=palette["blue_soft"])
    flow_labels = []
    for index, row in flow.iterrows():
        label = row.get("concept_id")
        if label is None or (not isinstance(label, str) and pd.isna(label)):
            label = row.get("predicate_kind")
        flow_labels.append(_label(label) if label is not None else f"Step {index + 1}")
    axes[0, 0].set_yticks(positions, flow_labels, fontsize=5.5)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("ICU stays remaining")
    axes[0, 0].set_title("Cohort accounting", loc="left", pad=12)
    add_panel_label(axes[0, 0], "A", x=-0.12, y=1.04)

    positions = np.arange(len(balance))
    axes[0, 1].barh(
        positions,
        balance["absolute_standardized_mean_difference"],
        color=palette["orange"],
    )
    axes[0, 1].set_yticks(
        positions,
        [_label(value) for value in balance["variable"]],
        fontsize=5.5,
    )
    axes[0, 1].axvline(0.1, color="#777777", linewidth=0.8, linestyle="--")
    axes[0, 1].set_xlabel("Absolute standardized difference")
    axes[0, 1].set_title("Baseline balance", loc="left", pad=12)
    add_panel_label(axes[0, 1], "B", x=-0.12, y=1.04)

    adjusted_label = next(
        (
            candidate
            for candidate in ("contrast", "exposure", "model_id")
            if candidate in adjusted.columns
            and adjusted[candidate].notna().all()
            and adjusted[candidate].astype(str).str.strip().ne("").all()
        ),
        "model_id",
    )
    _forest(
        axes[1, 0],
        adjusted,
        estimate_column="estimate",
        label_column=adjusted_label,
        title="Primary adjusted association",
        color=palette["blue"],
    )
    add_panel_label(axes[1, 0], "C", x=-0.12, y=1.04)

    robustness_label = "spec_id" if "spec_id" in robustness.columns else "axis"
    _forest(
        axes[1, 1],
        robustness,
        estimate_column="point_estimate",
        label_column=robustness_label,
        title="Robustness estimates",
        color=palette["blue_soft"],
    )
    add_panel_label(axes[1, 1], "D", x=-0.12, y=1.04)

    panel_rows = (
        ("A", "Cohort accounting", "cohort_accounting", "cohort_flow", input_keys[0]),
        (
            "B",
            "Baseline balance",
            "descriptive_result",
            "standardized_difference",
            input_keys[1],
        ),
        (
            "C",
            "Primary adjusted association",
            "primary_estimand",
            "forest_plot",
            input_keys[2],
        ),
        ("D", "Robustness estimates", "robustness", "forest_plot", input_keys[3]),
    )
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "Cohort accounting and baseline balance contextualize the primary "
            "adjusted association and its prespecified robustness estimates."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=178.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": title,
                "role": role,
                "article_role": role,
                "chart_type": chart_type,
                "claim": f"This panel visualizes values from {source} without refitting.",
                "evidence_ids": [evidence[source]],
                "metadata": {
                    "source_products": [source],
                    "source_data": [f"{source.partition(':')[2]}_source_data.csv"],
                },
            }
            for panel_id, title, role, chart_type, source in panel_rows
        ],
        source_data=source_files,
        statistics_note=(
            "Panel B reports source-table absolute standardized differences; "
            "Panels C and D preserve the reported point estimates and confidence "
            "intervals. The renderer performs no fitting or row selection."
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
        "method": "deterministic_cohort_balance_association_figure",
        "analysis_family": "association",
        "deterministic_standard_analysis": "cohort_balance_association_figure",
        "rendering_only": True,
        "source_inputs": list(input_keys),
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


def _render_balance_association_figure(
    *,
    bound: Mapping[str, BoundTypedInput],
    out_dir: Path,
    step_id: str,
    figure_product: str,
    input_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Render a source-bound balance, estimate, and robustness suite."""

    balance = bound["table:balance_positivity_context"].frame.copy()
    balance = balance.loc[
        balance["standardized_difference_status"].astype(str).eq("computed")
    ].copy()
    balance["absolute_standardized_mean_difference"] = pd.to_numeric(
        balance["absolute_standardized_mean_difference"], errors="coerce"
    )
    balance = balance.loc[
        balance["absolute_standardized_mean_difference"].notna()
    ].copy()
    if balance.empty:
        raise ValueError("balance context has no computed standardized differences")
    if (balance["absolute_standardized_mean_difference"] < 0).any() or not np.isfinite(
        balance["absolute_standardized_mean_difference"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "balance-context standardized differences must be finite and non-negative"
        )
    balance = (
        balance.groupby("variable", as_index=False)[
            "absolute_standardized_mean_difference"
        ]
        .max()
        .sort_values("absolute_standardized_mean_difference", ascending=True)
    )
    adjusted = _validate_interval_table(
        bound["table:adjusted_association_estimates"].frame,
        estimate_column="estimate",
        require_fitted=True,
    )
    robustness = _validate_interval_table(
        bound["table:robustness_matrix"].frame,
        estimate_column="point_estimate",
    )
    if not robustness["converged"].astype(bool).all():
        raise ValueError("robustness matrix contains non-converged rows")
    robustness_summary = bound["table:robustness_summary"].frame.copy()

    source_files = [_source_copy(bound[key], out_dir) for key in input_keys]
    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    palette = apply_publication_style(font_size=7.0)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)

    positions = np.arange(len(balance))
    axes[0, 0].barh(
        positions,
        balance["absolute_standardized_mean_difference"],
        color=palette["orange"],
    )
    axes[0, 0].set_yticks(
        positions,
        [_label(value) for value in balance["variable"]],
        fontsize=5.5,
    )
    axes[0, 0].axvline(0.1, color="#777777", linewidth=0.8, linestyle="--")
    axes[0, 0].set_xlabel("Absolute standardized difference")
    axes[0, 0].set_title("Baseline balance", loc="left", pad=12)
    add_panel_label(axes[0, 0], "A", x=-0.12, y=1.04)

    adjusted_label = next(
        (
            candidate
            for candidate in ("contrast", "exposure", "model_id")
            if candidate in adjusted.columns
            and adjusted[candidate].notna().all()
            and adjusted[candidate].astype(str).str.strip().ne("").all()
        ),
        "model_id",
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

    robustness_label = "spec_id" if "spec_id" in robustness.columns else "axis"
    _forest(
        axes[1, 0],
        robustness,
        estimate_column="point_estimate",
        label_column=robustness_label,
        title="Robustness estimates",
        color=palette["blue_soft"],
    )
    add_panel_label(axes[1, 0], "C", x=-0.12, y=1.04)

    _robustness_ranges(axes[1, 1], robustness_summary, color=palette["orange"])
    add_panel_label(axes[1, 1], "D", x=-0.12, y=1.04)

    panels = (
        (
            "baseline_balance",
            "Baseline balance",
            "descriptive_result",
            "standardized_difference",
            input_keys[0],
        ),
        (
            "primary_adjusted_association",
            "Primary adjusted association",
            "primary_estimand",
            "forest",
            input_keys[1],
        ),
        (
            "robustness_estimates",
            "Robustness estimates",
            "robustness",
            "sensitivity_forest",
            input_keys[2],
        ),
        (
            "robustness_ranges",
            "Robustness ranges",
            "robustness",
            "specification_grid",
            input_keys[3],
        ),
    )
    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "Observed baseline balance contextualizes the primary adjusted "
            "association and its prespecified robustness estimates."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=178.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": title,
                "role": role,
                "article_role": role,
                "chart_type": chart_type,
                "claim": f"This panel visualizes values from {source} without refitting.",
                "evidence_ids": [evidence[source]],
                "metadata": {
                    "source_products": [source],
                    "source_data": [f"{source.partition(':')[2]}_source_data.csv"],
                },
            }
            for panel_id, title, role, chart_type, source in panels
        ],
        source_data=source_files,
        statistics_note=(
            "All source rows and original columns are preserved in source-data "
            "files. The renderer performs no model fitting or row selection."
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
        "method": "deterministic_balance_association_figure",
        "analysis_family": "association",
        "deterministic_standard_analysis": "balance_association_figure",
        "rendering_only": True,
        "source_inputs": list(input_keys),
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


def render_association_publication_figure(
    *,
    bound: Mapping[str, BoundTypedInput],
    out_dir: Path,
    step_id: str,
    figure_product: str,
    input_keys: tuple[str, ...],
    display_labels: Mapping[str, str] | None = None,
    panel_placements: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Render all four bound products without fitting or selecting a model."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if tuple(input_keys) == COHORT_BALANCE_ASSOCIATION_COMPOSITE_INPUTS:
        return _render_cohort_balance_association_figure(
            bound=bound,
            out_dir=out_dir,
            step_id=step_id,
            figure_product=figure_product,
            input_keys=input_keys,
        )
    if tuple(input_keys) == BALANCE_ASSOCIATION_COMPOSITE_INPUTS:
        return _render_balance_association_figure(
            bound=bound,
            out_dir=out_dir,
            step_id=step_id,
            figure_product=figure_product,
            input_keys=input_keys,
        )

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
    robustness_key = next(
        (
            key
            for key in ("table:robustness_matrix", "table:robustness_summary")
            if key in bound
        ),
        None,
    )
    robustness = (
        bound[robustness_key].frame.copy() if robustness_key is not None else None
    )
    if robustness_key == "table:robustness_matrix" and robustness is not None:
        robustness = _validate_interval_table(
            robustness,
            estimate_column="point_estimate",
        )
    missingness_key = next(
        (
            key
            for key in (
                "table:measurement_missingness",
                "table:missingness_measurement_audit",
            )
            if key in bound
        ),
        None,
    )
    missingness = (
        _measurement_missingness(bound[missingness_key].frame)
        if missingness_key is not None
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
    completeness = (
        _component_completeness(
            bound["table:exposure_component_completeness_audit"].frame
        )
        if "table:exposure_component_completeness_audit" in bound
        else None
    )
    scientific_sensitivity_key = next(
        (
            key
            for key in bound
            if key
            not in {
                "table:exposure_outcome_distribution",
                "table:adjusted_association_estimates",
                "table:exposure_component_completeness_audit",
                "table:absolute_risk_context",
                "table:robustness_matrix",
                "table:robustness_summary",
                "table:measurement_missingness",
                "table:missingness_measurement_audit",
                "table:measurement_process_audit",
            }
        ),
        None,
    )
    scientific_sensitivity = None
    if scientific_sensitivity_key is not None:
        scientific_sensitivity = _validate_interval_table(
            bound[scientific_sensitivity_key].frame,
            estimate_column="estimate",
        )
        if not scientific_sensitivity["converged"].astype(bool).all():
            raise ValueError("scientific sensitivity table contains non-converged rows")
        scientific_sensitivity["effect_scale"] = scientific_sensitivity[
            "effect_measure"
        ]

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
        levels["exposure_pct"] = _association_finite_series(levels, "exposure_pct")
        levels["outcome_rate_pct"] = _association_finite_series(
            levels, "outcome_rate_pct"
        )
        if (
            (levels["exposure_denominator"] <= 0).any()
            or (levels["n_rows"] > levels["exposure_denominator"]).any()
            or (levels["outcome_denominator"] <= 0).any()
            or (levels["outcome_events"] > levels["outcome_denominator"]).any()
        ):
            raise ValueError(
                "distribution counts do not nest within positive denominators"
            )
        expected_prevalence = 100.0 * levels["n_rows"] / levels["exposure_denominator"]
        expected_rates = (
            100.0 * levels["outcome_events"] / levels["outcome_denominator"]
        )
        if (
            not np.isclose(
                levels["exposure_pct"], expected_prevalence, rtol=0.0, atol=5e-6
            ).all()
            or not np.isclose(
                levels["outcome_rate_pct"], expected_rates, rtol=0.0, atol=5e-6
            ).all()
        ):
            raise ValueError("distribution percentages do not reconcile to counts")
        has_risk_ci = {"ci_low_pct", "ci_high_pct"} <= set(levels.columns)
        if has_risk_ci:
            levels["ci_low_pct"] = _association_finite_series(levels, "ci_low_pct")
            levels["ci_high_pct"] = _association_finite_series(levels, "ci_high_pct")
            if (levels["ci_low_pct"] > levels["outcome_rate_pct"]).any() or (
                levels["outcome_rate_pct"] > levels["ci_high_pct"]
            ).any():
                raise ValueError(
                    "risk confidence intervals must contain reported rates"
                )
    else:
        levels = absolute_context
        has_risk_ci = True

    if (
        robustness_key == "table:robustness_matrix"
        and robustness is not None
        and not robustness["converged"].astype(bool).all()
    ):
        raise ValueError("robustness matrix contains non-converged rows")

    source_files = [_source_copy(bound[key], out_dir) for key in input_keys]
    evidence = {key: str(item.evidence_id or "") for key, item in bound.items()}
    labels = dict(display_labels or {})
    palette = apply_publication_style(font_size=7.0)
    placements = dict(panel_placements or {})
    show_panel_d = placements.get("D", "main") == "main"
    if show_panel_d:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.28))
        axes = np.empty((2, 2), dtype=object)
        axes[0, 0] = fig.add_subplot(grid[0, 0])
        axes[0, 1] = fig.add_subplot(grid[0, 1])
        axes[1, 0] = fig.add_subplot(grid[1, :])
        axes[1, 1] = None

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
            ordinal_scope = re.sub(r"_(max|min|first|last)$", "", exposure_name)
            declared_level = label_lookup(f"{ordinal_scope}_{raw}", labels)
            level_labels.append(
                labels.get(
                    f"{exposure_name}={raw}",
                    declared_level.split(":", 1)[0]
                    if declared_level
                    else _label(value),
                )
            )
        absolute_title = "Exposure prevalence and observed outcome risk"
    else:
        values = 100.0 * levels["estimate"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                values - 100.0 * levels["ci_low"].to_numpy(dtype=float),
                100.0 * levels["ci_high"].to_numpy(dtype=float) - values,
            ]
        )
        state_column = "group_value" if "group_value" in levels.columns else "label"
        level_labels = [
            _measurement_state_label(value) for value in levels[state_column]
        ]
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
    rotate_levels = len(level_labels) > 3 or max(map(len, level_labels), default=0) > 18
    ax.set_xticks(
        x,
        level_labels,
        rotation=20 if rotate_levels else 0,
        ha="right" if rotate_levels else "center",
        fontsize=6.2 if rotate_levels else None,
    )
    ax.set_ylabel(
        "Percent" if distribution is not None else "Observed outcome risk (%)"
    )
    ax.set_title(absolute_title, loc="left", pad=7)
    add_panel_label(ax, "a", x=-0.12, y=1.04, fontsize=8.0)

    adjusted_label = "model_id"
    for candidate in ("contrast", "exposure", "model_id"):
        if (
            candidate in adjusted.columns
            and adjusted[candidate].notna().all()
            and adjusted[candidate].astype(str).str.strip().ne("").all()
        ):
            adjusted_label = candidate
            break
    exposure_name = None
    if distribution is not None and "exposure_column" in levels.columns:
        exposure_name = str(levels.iloc[0]["exposure_column"])
    reader_contrasts = _reader_contrast_labels(
        adjusted, exposure_name=exposure_name, display_labels=labels
    )
    if reader_contrasts is not None:
        adjusted = adjusted.copy()
        adjusted["_reader_contrast"] = reader_contrasts
        adjusted_label = "_reader_contrast"
    _forest(
        axes[0, 1],
        adjusted,
        estimate_column="estimate",
        label_column=adjusted_label,
        title="Primary adjusted association",
        color=palette["blue"],
        label_formatter=(lambda value: str(value))
        if adjusted_label == "_reader_contrast"
        else None,
    )
    add_panel_label(axes[0, 1], "b", x=-0.12, y=1.04, fontsize=8.0)

    if scientific_sensitivity is not None and scientific_sensitivity_key is not None:
        _forest(
            axes[1, 0],
            scientific_sensitivity,
            estimate_column="estimate",
            label_column="analysis_id",
            title="Scientific sensitivity analyses",
            color=palette["blue_soft"],
        )
        panel_c = (
            "Scientific sensitivity analyses",
            "robustness",
            scientific_sensitivity_key,
        )
    elif robustness_key == "table:robustness_matrix" and robustness is not None:
        robustness_labels = "spec_id" if "spec_id" in robustness.columns else "axis"
        _forest(
            axes[1, 0],
            robustness,
            estimate_column="point_estimate",
            label_column=robustness_labels,
            title="Robustness estimates",
            color=palette["blue_soft"],
        )
        panel_c = (
            "Robustness estimates",
            "robustness",
            "table:robustness_matrix",
        )
    elif robustness_key == "table:robustness_summary" and robustness is not None:
        _robustness_ranges(axes[1, 0], robustness, color=palette["blue_soft"])
        panel_c = (
            "Robustness ranges",
            "robustness",
            "table:robustness_summary",
        )
    elif missingness is not None and missingness_key is not None:
        _draw_missingness(axes[1, 0], missingness, color=palette["blue_soft"])
        panel_c = ("Measurement missingness", "data_quality", missingness_key)
    else:  # pragma: no cover - guarded by exact typed profiles
        raise ValueError("association composite has no third-panel source")
    add_panel_label(axes[1, 0], "c", x=-0.12, y=1.04, fontsize=8.0)

    if not show_panel_d:
        panel_d = None
    elif completeness is not None:
        _draw_component_completeness(axes[1, 1], completeness)
        panel_d = (
            "Component completeness",
            "data_quality",
            "table:exposure_component_completeness_audit",
        )
    elif missingness is not None and missingness_key is not None:
        _draw_missingness(axes[1, 1], missingness, color=palette["orange"])
        panel_d = ("Measurement missingness", "data_quality", missingness_key)
    elif availability is not None:
        quality = availability.sort_values("availability_pct", ascending=True)
        label_column = "concept" if "concept" in quality.columns else "variable"
        positions = np.arange(len(quality))
        axes[1, 1].barh(positions, quality["availability_pct"], color=palette["orange"])
        axes[1, 1].set_yticks(
            positions,
            [display_label(value) for value in quality[label_column]],
            fontsize=5.5,
        )
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_xlabel("Available among eligible (%)")
        axes[1, 1].set_title("Measurement availability", loc="left", pad=7)
        panel_d = (
            "Measurement availability",
            "data_quality",
            "table:measurement_process_audit",
        )
    elif robustness_summary is not None:
        _robustness_ranges(
            axes[1, 1],
            robustness_summary,
            color=palette["orange"],
        )
        panel_d = ("Robustness ranges", "robustness", "table:robustness_summary")
    else:  # pragma: no cover - guarded by exact typed profiles
        raise ValueError("association composite has no fourth-panel source")
    if show_panel_d:
        add_panel_label(axes[1, 1], "d", x=-0.12, y=1.04, fontsize=8.0)

    if scientific_sensitivity is not None:
        panel_specs = (
            (
                "absolute_risk_context",
                absolute_title,
                "descriptive_result",
                "grouped_absolute_risk",
                "table:exposure_outcome_distribution",
            ),
            (
                "primary_adjusted_association",
                "Primary adjusted association",
                "primary_estimand",
                "forest_plot",
                "table:adjusted_association_estimates",
            ),
            (
                "scientific_sensitivity",
                "Scientific sensitivity analyses",
                "robustness",
                "sensitivity_forest_plot",
                scientific_sensitivity_key,
            ),
            (
                "component_completeness",
                "Component completeness",
                "data_quality",
                "availability_heatmap",
                "table:exposure_component_completeness_audit",
            ),
        )
        if not show_panel_d:
            panel_specs = panel_specs[:3]
    else:
        panel_specs = (
            (
                "A",
                absolute_title,
                "descriptive_result",
                (
                    "event_rate_panel"
                    if distribution is not None
                    else "dot_interval_absolute_risk"
                ),
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
                "forest",
                "table:adjusted_association_estimates",
            ),
            (
                "C",
                panel_c[0],
                panel_c[1],
                "sensitivity_forest" if panel_c[1] == "robustness" else "bar",
                panel_c[2],
            ),
        )
        if show_panel_d and panel_d is not None:
            panel_specs = (
                *panel_specs,
                (
                    "D",
                    panel_d[0],
                    panel_d[1],
                    (
                        "specification_grid"
                        if panel_d[1] == "robustness"
                        else "availability_panel"
                    ),
                    panel_d[2],
                ),
            )
        if tuple(input_keys) == ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS:
            panel_specs = tuple(
                (panel_id, *spec[1:])
                for panel_id, spec in zip(
                    (
                        "absolute_risk_context",
                        "primary_adjusted_association",
                        "robustness_estimates",
                        "robustness_ranges",
                    ),
                    panel_specs,
                )
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
        height_mm=132.0,
        panels=[
            {
                "panel_id": panel_id,
                "title": title,
                "role": role,
                "article_role": role,
                "chart_type": chart_type,
                "claim": f"This panel visualizes values from {source} without refitting.",
                "evidence_ids": [evidence[source]],
                "metadata": {
                    "source_products": [source],
                    "source_data": [f"{source.partition(':')[2]}_source_data.csv"],
                },
            }
            for panel_id, title, role, chart_type, source in panel_specs
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
        "supplementary_panel_ids": sorted(
            panel_id
            for panel_id, placement in placements.items()
            if placement == "supplementary"
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


__all__ = ["render_association_publication_figure"]
