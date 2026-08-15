"""Deterministic missingness and measurement publication renderer.

This module owns the complete prior-output rescue: source-table selection,
count-derived availability semantics, source-data projection, plotting, figure
contract, and step summary.  It never chooses a new scientific variable or
reads outside the authorized parent-step boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .prior_output_support import (
    figure_parent_candidate_step_dirs,
    publication_label,
    short_figure_label,
)


def render_missingness_publication_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
) -> Optional[str]:
    """Deterministically rebuild a missingness/measurement audit figure."""

    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None
    parent_step_id = current_step_id.removesuffix("_figure")
    candidate_paths: List[Path] = []
    candidate_step_dirs, direct_parent_only = figure_parent_candidate_step_dirs(
        steps_dir=steps_dir, current_step_id=current_step_id
    )
    for step_dir in candidate_step_dirs:
        text = step_dir.name.lower()
        if not direct_parent_only and not any(
            token in text for token in ("missing", "measurement", "quality")
        ):
            continue
        outputs_dir = step_dir / "outputs"
        if outputs_dir.exists():
            candidate_paths.extend(sorted(outputs_dir.glob("*.csv")))

    def _first_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name in frame.columns:
                return name
        return None

    parent: Optional[tuple[Path, pd.DataFrame]] = None
    parent_score = -1
    for csv_path in candidate_paths:
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        has_label = _first_col(
            frame,
            ("variable", "exposure_or_variable", "concept", "label", "value_col"),
        )
        has_total = _first_col(
            frame,
            ("total_n", "n_total", "denominator", "denominator_n", "n"),
        )
        has_missing = _first_col(
            frame,
            ("missing_n", "n_missing", "value_missing_n", "raw_missing_n"),
        )
        has_unavailable = _first_col(
            frame,
            ("analysis_unavailable_n", "unavailable_n", "invalid_or_missing_n"),
        )
        has_measured = _first_col(
            frame, ("measured_n", "measured_one_n", "n_nonmissing")
        )
        has_pct = _first_col(
            frame,
            (
                "missing_pct",
                "value_missing_pct",
                "raw_missing_pct",
                "analysis_unavailable_pct",
                "measured_pct",
                "measured_one_pct",
                "percentage",
            ),
        )
        if not (
            has_label
            and (
                has_total
                and (has_missing or has_unavailable or has_measured)
                or has_pct
            )
        ):
            continue
        name = csv_path.name.lower()
        score = 0
        if "missingness" in name:
            score += 100
        if "measurement" in name:
            score += 100
        if has_missing:
            score += 20
        if has_unavailable:
            score += 20
        if "metric" in frame.columns and any(
            column in frame.columns for column in ("table_section", "section")
        ):
            score += 10
        if score > parent_score:
            parent = (csv_path, frame)
            parent_score = score
    if parent is None:
        return None

    table_path, frame = parent
    label_col = _first_col(
        frame,
        ("variable", "exposure_or_variable", "concept", "value_col", "label"),
    )
    display_col = _first_col(
        frame,
        (
            "display_label",
            "label",
            "concept",
            "variable",
            "exposure_or_variable",
            "value_col",
        ),
    )
    section_col = _first_col(frame, ("table_section", "section"))
    metric_col = _first_col(frame, ("metric",))
    cohort_col = _first_col(frame, ("cohort", "scope"))
    category_col = _first_col(frame, ("category", "status_category"))
    total_col = _first_col(
        frame,
        ("total_n", "n_total", "denominator", "denominator_n", "n"),
    )
    missing_n_col = _first_col(
        frame,
        ("missing_n", "n_missing", "value_missing_n", "raw_missing_n"),
    )
    unavailable_n_col = _first_col(
        frame,
        ("analysis_unavailable_n", "unavailable_n", "invalid_or_missing_n"),
    )
    measured_n_col = _first_col(frame, ("measured_n", "measured_one_n", "n_nonmissing"))
    missing_pct_col = _first_col(
        frame,
        ("missing_pct", "value_missing_pct", "raw_missing_pct"),
    )
    unavailable_pct_col = _first_col(
        frame,
        ("analysis_unavailable_pct", "unavailable_pct", "invalid_or_missing_pct"),
    )
    measured_pct_col = _first_col(frame, ("measured_pct", "measured_one_pct"))
    if label_col is None:
        return None

    rich_process_table = section_col is not None and metric_col is not None
    source_all = frame.copy()
    if rich_process_table:
        source_all["source_row_index"] = range(len(source_all))
    source = source_all.copy()
    source_row_filter = "all_compatible_rows"
    if rich_process_table:
        section = source[section_col].astype(str).str.lower()
        metric = source[metric_col].astype(str).str.lower()
        raw_rows = section.eq("column_missingness") & metric.eq("raw_missing")
        unavailable_rows = section.eq("column_missingness") & metric.str.contains(
            "analysis_unavailable",
            regex=False,
        )
        if raw_rows.any():
            source = source.loc[raw_rows].copy()
            source_row_filter = "column_missingness:raw_missing"
        elif unavailable_rows.any():
            source = source.loc[unavailable_rows].copy()
            source_row_filter = "column_missingness:analysis_unavailable"
        if missing_n_col is None and "n" in source.columns:
            missing_n_col = "n"
        if missing_pct_col is None and "percentage" in source.columns:
            missing_pct_col = "percentage"
    total = (
        pd.to_numeric(source[total_col], errors="coerce")
        if total_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    missing_n = (
        pd.to_numeric(source[missing_n_col], errors="coerce")
        if missing_n_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    measured_n = (
        pd.to_numeric(source[measured_n_col], errors="coerce")
        if measured_n_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    unavailable_n = (
        pd.to_numeric(source[unavailable_n_col], errors="coerce")
        if unavailable_n_col is not None
        else pd.Series(pd.NA, index=source.index, dtype="Float64")
    )
    if measured_n_col is None and total_col is not None:
        if unavailable_n_col is not None:
            measured_n = total - unavailable_n
        elif missing_n_col is not None:
            measured_n = total - missing_n
    present_but_measured_zero_col = _first_col(
        source,
        ("value_present_but_measured_zero_n", "value_present_but_n_zero_n"),
    )
    if (
        present_but_measured_zero_col is not None
        and total_col is not None
        and missing_n_col is not None
    ):
        present_but_unflagged = pd.to_numeric(
            source[present_but_measured_zero_col],
            errors="coerce",
        ).fillna(0)
        use_value_availability = present_but_unflagged > 0
        measured_n = measured_n.mask(use_value_availability, total - missing_n)
    missing_pct = (
        100.0 * missing_n / total
        if total_col is not None and missing_n_col is not None
        else (
            pd.to_numeric(source[missing_pct_col], errors="coerce")
            if missing_pct_col is not None
            else pd.Series(pd.NA, index=source.index, dtype="Float64")
        )
    )
    unavailable_pct = (
        100.0 * unavailable_n / total
        if total_col is not None and unavailable_n_col is not None
        else (
            pd.to_numeric(source[unavailable_pct_col], errors="coerce")
            if unavailable_pct_col is not None
            else pd.Series(pd.NA, index=source.index, dtype="Float64")
        )
    )
    measured_pct = (
        100.0 * measured_n / total
        if total_col is not None and measured_n.notna().any()
        else (
            pd.to_numeric(source[measured_pct_col], errors="coerce")
            if measured_pct_col is not None
            else 100.0 - missing_pct
        )
    )
    labels = source[label_col].astype(str)
    display_labels = (
        source[display_col].astype(str)
        if display_col is not None
        else labels.map(publication_label)
    )
    indicator_semantics_col = _first_col(source, ("indicator_semantics",))
    event_status_mask = pd.Series(False, index=source.index)
    if indicator_semantics_col is not None:
        event_status_mask = (
            source[indicator_semantics_col].astype(str).eq("binary_event_presence")
        )
        display_labels = display_labels.mask(
            event_status_mask,
            display_labels.astype(str) + " — analytic event status",
        )
    label_output_col = "variable_name" if rich_process_table else "variable"
    source_data_payload: Dict[str, Any] = {
        label_output_col: labels,
        "display_label": display_labels,
        "missing_pct": missing_pct.astype(float),
        "missing_n": missing_n.astype(float),
        "n_nonmissing": measured_n.astype(float),
        "total_n": total.astype(float),
        "measured_pct": measured_pct.astype(float),
        "measured_n": measured_n.astype(float),
        "source_table": table_path.name,
        "source_transform": "missingness_measurement_summary_v1",
        "source_row_filter": source_row_filter,
    }
    if rich_process_table:
        source_data_payload["source_row_index"] = source["source_row_index"].astype(int)
    else:
        if "concept" in source.columns:
            source_data_payload["concept"] = source["concept"].astype(str)
        else:
            source_data_payload["concept"] = labels
        if "label" in source.columns:
            source_data_payload["label"] = source["label"].astype(str)
    if missing_n_col is not None:
        source_data_payload["value_missing_n"] = pd.to_numeric(
            source[missing_n_col],
            errors="coerce",
        )
        if rich_process_table:
            source_data_payload[missing_n_col] = pd.to_numeric(
                source[missing_n_col],
                errors="coerce",
            )
    if missing_pct_col is not None:
        source_data_payload["value_missing_pct"] = pd.to_numeric(
            source[missing_pct_col],
            errors="coerce",
        )
        if rich_process_table:
            source_data_payload[missing_pct_col] = pd.to_numeric(
                source[missing_pct_col],
                errors="coerce",
            )
    if unavailable_n_col is not None:
        source_data_payload["analysis_unavailable_n"] = unavailable_n
    if unavailable_pct_col is not None or unavailable_n_col is not None:
        source_data_payload["analysis_unavailable_pct"] = unavailable_pct
    if total_col is not None:
        source_data_payload["n_total"] = pd.to_numeric(
            source[total_col], errors="coerce"
        )
        if rich_process_table:
            source_data_payload[total_col] = pd.to_numeric(
                source[total_col],
                errors="coerce",
            )
    if cohort_col is not None:
        cohort_output_col = "cohort_name" if rich_process_table else "cohort"
        source_data_payload[cohort_output_col] = source[cohort_col].astype(str)
    if "measured_one_n" in source.columns:
        source_data_payload["measured_one_n"] = pd.to_numeric(
            source["measured_one_n"],
            errors="coerce",
        )
    if "measured_one_pct" in source.columns:
        source_data_payload["measured_one_pct"] = pd.to_numeric(
            source["measured_one_pct"],
            errors="coerce",
        )
    if indicator_semantics_col is not None:
        source_data_payload["indicator_semantics"] = source[
            indicator_semantics_col
        ].astype(str)
    if "raw_indicator_one_n" in source.columns:
        source_data_payload["raw_indicator_one_n"] = pd.to_numeric(
            source["raw_indicator_one_n"],
            errors="coerce",
        )
    if "event_count_column" in source.columns:
        source_data_payload["event_count_column"] = (
            source["event_count_column"].fillna("").astype(str)
        )
    source_data = pd.DataFrame(source_data_payload).dropna(
        subset=["missing_pct", "measured_pct"],
        how="all",
    )
    if source_data.empty:
        return None
    if rich_process_table:
        variable_order = (
            source_data.groupby(label_output_col, sort=False)["missing_pct"]
            .max()
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        source_data = source_data[
            source_data[label_output_col].isin(variable_order)
        ].copy()
        source_data[label_output_col] = pd.Categorical(
            source_data[label_output_col],
            categories=list(variable_order),
            ordered=True,
        )
        source_data = source_data.sort_values(
            [label_output_col, "cohort_name"],
        )
        source_data[label_output_col] = source_data[label_output_col].astype(str)
    else:
        source_data = source_data.sort_values("missing_pct", ascending=False).head(12)

    has_event_status_rows = bool(
        "indicator_semantics" in source_data.columns
        and source_data["indicator_semantics"]
        .astype(str)
        .eq("binary_event_presence")
        .any()
    )
    availability_title = (
        "Analytic availability" if has_event_status_rows else "Measurement availability"
    )
    availability_axis_label = (
        "Value / event status available (%)"
        if has_event_status_rows
        else "Available observations (%)"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    availability_source_path = out_dir / "missingness_measurement_panel_source_data.csv"
    source_data.to_csv(availability_source_path, index=False)

    status_source_data = pd.DataFrame()
    status_source_path = out_dir / "missingness_status_matrix_source_data.csv"
    if (
        rich_process_table
        and label_col is not None
        and category_col is not None
        and cohort_col is not None
        and total_col is not None
        and {"n", "percentage"}.issubset(source_all.columns)
    ):
        status_mask = source_all[section_col].astype(str).str.lower().eq(
            "source_status"
        ) & source_all[metric_col].astype(str).str.lower().isin(
            ("mutually_exclusive_source_status", "source_status")
        )
        status_rows = source_all.loc[status_mask].copy()
        if not status_rows.empty:
            status_source_data = pd.DataFrame(
                {
                    "variable_name": status_rows[label_col].astype(str),
                    "display_label": status_rows[label_col]
                    .astype(str)
                    .map(publication_label),
                    "cohort_name": status_rows[cohort_col].astype(str),
                    "status_category": status_rows[category_col].astype(str),
                    "n": pd.to_numeric(status_rows["n"], errors="coerce"),
                    "denominator": pd.to_numeric(
                        status_rows[total_col], errors="coerce"
                    ),
                    "percentage": pd.to_numeric(
                        status_rows["percentage"], errors="coerce"
                    ),
                    "source_row_index": status_rows["source_row_index"].astype(int),
                    "source_table": table_path.name,
                    "source_transform": "source_status_matrix_v1",
                }
            ).dropna(subset=["percentage"])
            if not status_source_data.empty:
                status_source_data.to_csv(status_source_path, index=False)

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
    rich_matrix_rendered = not status_source_data.empty
    if rich_matrix_rendered:
        fig = plt.figure(
            figsize=(183 / 25.4, 126 / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.18, 0.82],
            left=0.16,
            right=0.98,
            top=0.88,
            bottom=0.25,
            wspace=0.62,
        )
        ax_missing = fig.add_subplot(grid[0, 0])
        ax_measured = fig.add_subplot(grid[0, 1])

        status_plot = status_source_data.copy()
        multi_cohort = status_plot["cohort_name"].nunique() > 1
        status_plot["row_label"] = status_plot["display_label"].astype(str)
        if multi_cohort:
            status_plot["row_label"] = (
                status_plot["row_label"]
                + "\n"
                + status_plot["cohort_name"].map(publication_label)
            )
        status_rows = list(dict.fromkeys(status_plot["row_label"].tolist()))
        status_columns = list(
            dict.fromkeys(status_plot["status_category"].astype(str).tolist())
        )
        status_matrix = status_plot.pivot_table(
            index="row_label",
            columns="status_category",
            values="percentage",
            aggfunc="first",
        ).reindex(index=status_rows, columns=status_columns)

        def _status_display(value: str) -> str:
            text = value.lower().replace("_", " ")
            if "valid observed" in text:
                return "Observed"
            if "no recorded source" in text:
                return "No source"
            if "summary missing" in text:
                return "Source present;\nsummary missing"
            if "contradictory" in text or "invalid" in text:
                return "Contradictory /\ninvalid"
            return short_figure_label(value, limit=24)

        status_values = status_matrix.to_numpy(dtype=float)
        ax_missing.imshow(
            status_values,
            aspect="auto",
            vmin=0,
            vmax=100,
            cmap="Blues",
        )
        ax_missing.set_xticks(range(len(status_columns)))
        ax_missing.set_xticklabels(
            [_status_display(value) for value in status_columns],
            rotation=28,
            ha="right",
        )
        ax_missing.set_yticks(range(len(status_rows)))
        ax_missing.set_yticklabels(status_rows)
        for row_idx in range(len(status_rows)):
            for col_idx in range(len(status_columns)):
                value = status_values[row_idx, col_idx]
                if pd.isna(value):
                    continue
                ax_missing.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value >= 55 else "black",
                )
        ax_missing.set_xlabel("Source-status share of cohort (%)")
        ax_missing.set_title("Measurement-source status", loc="left", pad=4)
        add_panel_label(ax_missing, "A", x=0.0, y=1.08)

        availability_plot = source_data.copy()
        variable_rows = list(
            dict.fromkeys(availability_plot["variable_name"].astype(str).tolist())
        )
        cohort_columns = list(
            dict.fromkeys(availability_plot["cohort_name"].astype(str).tolist())
        )
        availability_matrix = availability_plot.pivot_table(
            index="variable_name",
            columns="cohort_name",
            values="measured_pct",
            aggfunc="first",
        ).reindex(index=variable_rows, columns=cohort_columns)
        availability_values = availability_matrix.to_numpy(dtype=float)

        def _measurement_display_map(values: Sequence[str]) -> Dict[str, str]:
            raw_values = list(dict.fromkeys(str(value) for value in values))
            display = {value: publication_label(value) for value in raw_values}
            counts: Dict[str, int] = {}
            for label in display.values():
                counts[label] = counts.get(label, 0) + 1
            suffix_labels = {
                "_first": "First value",
                "_max": "Maximum",
                "_min": "Minimum",
                "_mean": "Mean",
                "_n": "Observation count",
                "_measured": "Measured flag",
            }
            always_expand = ("_first", "_n", "_measured")
            for value in raw_values:
                lower_value = value.lower()
                if counts.get(display[value], 0) <= 1 and not lower_value.endswith(
                    always_expand
                ):
                    continue
                for suffix, suffix_label in suffix_labels.items():
                    if lower_value.endswith(suffix):
                        base = value[: -len(suffix)]
                        display[value] = f"{publication_label(base)} — {suffix_label}"
                        break
                else:
                    display[value] = value.replace("_", " ").title()
            return display

        variable_display = _measurement_display_map(variable_rows)
        ax_measured.imshow(
            availability_values,
            aspect="auto",
            vmin=0,
            vmax=100,
            cmap="Blues",
        )
        ax_measured.set_xticks(range(len(cohort_columns)))
        ax_measured.set_xticklabels(
            [publication_label(value) for value in cohort_columns],
            rotation=28,
            ha="right",
        )
        ax_measured.set_yticks(range(len(variable_rows)))
        ax_measured.set_yticklabels(
            [
                short_figure_label(variable_display[value], limit=32).replace(
                    " — ", "\n"
                )
                for value in variable_rows
            ]
        )
        for row_idx in range(len(variable_rows)):
            for col_idx in range(len(cohort_columns)):
                value = availability_values[row_idx, col_idx]
                if pd.isna(value):
                    continue
                ax_measured.text(
                    col_idx,
                    row_idx,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value >= 55 else "black",
                )
        ax_measured.set_xlabel(availability_axis_label)
        ax_measured.set_title(availability_title, loc="left", pad=4)
        add_panel_label(ax_measured, "B", x=0.0, y=1.08)
    else:
        plot_df = source_data.reset_index(drop=True)
        y = list(range(len(plot_df)))
        labels = [
            short_figure_label(label, limit=30)
            for label in plot_df["display_label"].astype(str)
        ]
        fig = plt.figure(
            figsize=(183 / 25.4, 104 / 25.4),
            constrained_layout=False,
        )
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=[1.05, 0.95],
            left=0.28,
            right=0.98,
            top=0.88,
            bottom=0.16,
            wspace=0.50,
        )
        ax_missing = fig.add_subplot(grid[0, 0])
        ax_measured = fig.add_subplot(grid[0, 1], sharey=ax_missing)

        missing = pd.to_numeric(plot_df["missing_pct"], errors="coerce").fillna(0)
        measured = pd.to_numeric(plot_df["measured_pct"], errors="coerce").fillna(0)
        ax_missing.barh(
            y,
            missing.clip(0, 100),
            color=palette.get("red", "#B2182B"),
            height=0.56,
        )
        ax_missing.axvline(
            20,
            color=palette.get("neutral", "#8F8F8F"),
            linestyle="--",
            linewidth=0.8,
        )
        ax_missing.set_yticks(y)
        ax_missing.set_yticklabels(labels)
        ax_missing.invert_yaxis()
        ax_missing.set_xlabel("Missing values (%)")
        ax_missing.set_title("Value missingness", loc="left", pad=4)
        ax_missing.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
        )
        add_panel_label(ax_missing, "A", x=0.0, y=1.08)

        ax_measured.barh(
            y,
            measured.clip(0, 100),
            color=palette.get("blue", "#0F4D92"),
            height=0.56,
        )
        ax_measured.set_xlim(0, 100)
        ax_measured.set_xlabel(availability_axis_label)
        ax_measured.set_title(availability_title, loc="left", pad=4)
        ax_measured.tick_params(axis="y", labelleft=False)
        ax_measured.grid(
            axis="x",
            color=palette.get("neutral_light", "#D8D8D8"),
            linewidth=0.55,
        )
        add_panel_label(ax_measured, "B", x=0.0, y=1.08)

    parent_evidence_id = table_path.stem
    availability_source_id = availability_source_path.stem
    status_source_id = status_source_path.stem
    panel_a_title = (
        "Measurement-source status" if rich_matrix_rendered else "Value missingness"
    )
    panel_a_claim = (
        "Mutually exclusive source-status percentages are shown for each audited "
        "measurement summary and cohort."
        if rich_matrix_rendered
        else "Missing percentages are recomputed from missing counts and denominators "
        "in the parent audit table."
    )
    panel_a_evidence = [
        parent_evidence_id,
        status_source_id if rich_matrix_rendered else availability_source_id,
    ]
    # Figure contracts name concrete local CSV files; upstream evidence ids stay
    # in the panel bindings above. A stem-only list leaves the strict source-data
    # validator with no verifiable file and must not pass.
    contract_source_data = [availability_source_path.name]
    if rich_matrix_rendered:
        contract_source_data.append(status_source_path.name)

    contract = make_figure_contract(
        figure_id="missingness_measurement_panel",
        core_claim=(
            "First-24h variable availability is shown directly from the "
            "registered missingness and measurement audit table."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": panel_a_title,
                "role": "data_quality",
                "chart_type": (
                    "missingness_matrix" if rich_matrix_rendered else "missingness_bar"
                ),
                "claim": panel_a_claim,
                "evidence_ids": panel_a_evidence,
            },
            {
                "panel_id": "B",
                "title": availability_title,
                "role": "data_quality",
                "chart_type": "availability_panel",
                "claim": (
                    "Value availability percentages are recomputed from counts and "
                    "denominators in the parent audit table. Registered binary-event "
                    "rows report analytic status availability under the locked "
                    "absence-as-negative convention, not literal measurement capture."
                    if has_event_status_rows
                    else "Measured or available percentages are recomputed from "
                    "measurement counts and denominators in the parent audit table."
                ),
                "evidence_ids": [parent_evidence_id, availability_source_id],
            },
        ],
        source_data=contract_source_data,
        statistics_note=(
            "Generated deterministically from the registered parent-step "
            "missingness/measurement audit; percentages are count-derived."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / "missingness_measurement_panel",
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
            "method": "deterministic_missingness_publication_figure_repair",
            "rendering_only": True,
            "source_step_id": parent_step_id,
            "source_missingness_table": str(table_path),
            "source_row_filter": source_row_filter,
            "source_data_csv": str(availability_source_path),
            "source_data_files": [
                availability_source_path.name,
                *([status_source_path.name] if rich_matrix_rendered else []),
            ],
            "n_variables_plotted": int(source_data[label_output_col].nunique()),
            "n_availability_rows": int(len(source_data)),
            "n_source_status_rows": int(len(status_source_data)),
            "n_binary_event_status_rows": int(
                source_data.get("indicator_semantics", pd.Series(dtype=str))
                .astype(str)
                .eq("binary_event_presence")
                .sum()
            ),
            "rich_missingness_matrix_rendered": rich_matrix_rendered,
            "figure_files": [
                path.name for key, path in outputs.items() if key != "contract"
            ],
            "figure_path": "missingness_measurement_panel.png",
        }
    )
    step_summary_path.write_text(
        json.dumps(existing_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return "missingness_publication_bundle_from_parent_outputs_v1"


__all__ = ["render_missingness_publication_bundle_from_prior_outputs"]
