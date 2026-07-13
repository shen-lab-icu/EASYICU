"""Deterministic publication figure for ordered-category QC distributions.

The renderer is deliberately case-neutral.  It activates from a controlled
parent method and a source-table schema (ordered level + count), never from a
clinical variable name.  Panel A shows the distribution conditional on a valid
observed category; panel B accounts for valid-observed versus unavailable rows
against the locked cohort.  This separation prevents a locked-cohort percentage
from being paired with the valid-observed denominator.
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ..publication_figures import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)


_LEVEL_COLUMNS = ("stage", "level", "ordered_level", "exposure_level", "category")
_COUNT_COLUMNS = ("n", "count", "frequency")
_CONDITIONAL_PERCENT_COLUMNS = (
    "percentage_of_valid_observed_stage",
    "percentage_of_valid_observed",
    "percentage_within_observed",
    "percentage_within_available",
)
_STATUS_COLUMNS = ("source_status", "source_state", "availability_status")
_LOCKED_PERCENT_COLUMNS = (
    "percentage_of_locked_cohort",
    "percentage_of_analysis_cohort",
)
_CONDITIONAL_FRACTION_COLUMNS = (
    "fraction_of_valid_observed_stage",
    "fraction_of_valid_observed",
    "fraction_within_observed",
    "fraction_within_available",
)
_LOCKED_FRACTION_COLUMNS = (
    "fraction_of_locked_cohort",
    "fraction_of_analysis_cohort",
)
_VALID_OBSERVED_ROLES = {"valid_observed", "observed", "valid", "available"}


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _resolve_column(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    by_name = {_normalise(column): str(column) for column in frame.columns}
    for name in names:
        match = by_name.get(_normalise(name))
        if match is not None:
            return match
    return None


def _nonnegative_integer(series: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & numeric.ge(0) & numeric.mod(1).eq(0)
    if not bool(valid.all()):
        return None
    return numeric.astype(int)


def _candidate_table(
    parent_out: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
) -> Optional[Tuple[Path, pd.DataFrame, str, str]]:
    candidates: list[Tuple[int, Path, pd.DataFrame, str, str]] = []
    if preverified_parent_artifacts is None:
        sources = [(path, path) for path in sorted(parent_out.glob("*.csv"))]
    else:
        sources = [
            (parent_out / name, payload)
            for name, payload in sorted(preverified_parent_artifacts.items())
            if Path(name).name == name and Path(name).suffix.lower() == ".csv"
        ]
    for path, source in sources:
        if "source_data" in path.name.lower() or path.name == "cohort_flow.csv":
            continue
        try:
            frame = pd.read_csv(
                io.BytesIO(source) if isinstance(source, bytes) else source
            )
        except Exception:
            continue
        level_col = _resolve_column(frame, _LEVEL_COLUMNS)
        count_col = _resolve_column(frame, _COUNT_COLUMNS)
        if level_col is None or count_col is None:
            continue
        levels = pd.to_numeric(frame[level_col], errors="coerce")
        level_rows = levels.notna()
        if int(level_rows.sum()) < 2:
            continue
        unique_levels = levels[level_rows].drop_duplicates()
        if len(unique_levels) != int(level_rows.sum()):
            # A source table with repeated levels needs another grouping key;
            # choosing one row per level here would silently collapse strata.
            continue
        score = 0
        stem = _normalise(path.stem)
        if "distribution" in stem:
            score += 100
        if _resolve_column(frame, _CONDITIONAL_PERCENT_COLUMNS):
            score += 20
        if _resolve_column(frame, _STATUS_COLUMNS):
            score += 10
        candidates.append((score, path, frame, level_col, count_col))
    # Fail closed when more than one sibling table has the required shape.
    # Ranking multiple plausible tables would turn a schema contract back into
    # a heuristic search over filenames.
    if len(candidates) != 1:
        return None
    _, path, frame, level_col, count_col = candidates[0]
    return path, frame, level_col, count_col


def _optional_values_match(
    *,
    frame: pd.DataFrame,
    column_names: Sequence[str],
    expected: pd.Series,
    tolerance: float,
) -> bool:
    column = _resolve_column(frame, column_names)
    if column is None:
        return True
    observed = pd.to_numeric(frame[column], errors="coerce").reset_index(drop=True)
    expected = expected.astype(float).reset_index(drop=True)
    return bool(
        len(observed) == len(expected)
        and observed.notna().all()
        and (observed - expected).abs().le(tolerance).all()
    )


def _declared_count(summary: Dict[str, Any], *names: str) -> Optional[int]:
    for name in names:
        value = pd.to_numeric(pd.Series([summary.get(name)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value).is_integer() and float(value) >= 0:
            return int(value)
    return None


def _availability_distribution(
    *,
    frame: pd.DataFrame,
    parent_out: Path,
    parent_summary: Dict[str, Any],
    level_col: str,
    count_col: str,
    observed_n: int,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
) -> Optional[Tuple[pd.DataFrame, int]]:
    status_col = _resolve_column(frame, _STATUS_COLUMNS)
    if status_col is None:
        return None
    levels = pd.to_numeric(frame[level_col], errors="coerce")
    status_rows = frame.loc[
        levels.isna() & frame[status_col].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if status_rows.empty:
        return None
    status_rows = status_rows.reset_index().rename(
        columns={"index": "__source_row_index"}
    )
    counts = _nonnegative_integer(status_rows[count_col])
    if counts is None:
        return None
    status_rows["__count"] = counts.to_numpy()
    status_rows["__status_role"] = status_rows[status_col].map(_normalise)
    if (
        status_rows["__status_role"].eq("").any()
        or status_rows["__status_role"].duplicated().any()
    ):
        return None
    valid_mask = status_rows["__status_role"].isin(_VALID_OBSERVED_ROLES)
    if int(valid_mask.sum()) != 1:
        return None
    if int(status_rows.loc[valid_mask, "__count"].sum()) != observed_n:
        return None
    locked_n = int(status_rows["__count"].sum())
    if locked_n < observed_n:
        return None

    expected_pct = 100.0 * status_rows["__count"].astype(float) / float(locked_n)
    expected_fraction = status_rows["__count"].astype(float) / float(locked_n)
    if not _optional_values_match(
        frame=status_rows,
        column_names=_LOCKED_PERCENT_COLUMNS,
        expected=expected_pct,
        tolerance=0.05,
    ):
        return None
    if not _optional_values_match(
        frame=status_rows,
        column_names=_LOCKED_FRACTION_COLUMNS,
        expected=expected_fraction,
        tolerance=0.0005,
    ):
        return None

    declared_locked_n = _declared_count(
        parent_summary,
        "locked_analysis_cohort_n",
        "n_analysis_cohort",
        "analysis_cohort_n",
    )
    if declared_locked_n is not None and declared_locked_n != locked_n:
        return None
    declared_observed_n = _declared_count(
        parent_summary,
        "valid_observed_n",
        "n_valid_observed",
    )
    if declared_observed_n is not None and declared_observed_n != observed_n:
        return None

    flow_path = parent_out / "cohort_flow.csv"
    flow_payload = (
        preverified_parent_artifacts.get("cohort_flow.csv")
        if preverified_parent_artifacts is not None
        else None
    )
    # Once the host supplies a digest-bound snapshot, that mapping is the
    # complete input authority.  Do not let an optional file that appears on
    # disk after sealing influence the rendered result.
    flow_source = (
        io.BytesIO(flow_payload)
        if flow_payload is not None
        else flow_path
        if preverified_parent_artifacts is None and flow_path.exists()
        else None
    )
    if flow_source is not None:
        try:
            flow = pd.read_csv(flow_source)
        except Exception:
            flow = pd.DataFrame()
        flow_n_col = _resolve_column(flow, ("n", "count", "retained_n"))
        flow_step_col = _resolve_column(flow, ("step", "row_type", "label"))
        if flow_n_col is not None and not flow.empty:
            rows = flow
            if flow_step_col is not None:
                preferred = (
                    flow[flow_step_col]
                    .map(_normalise)
                    .isin({"locked_analysis_cohort", "analysis_cohort", "locked_input"})
                )
                if bool(preferred.any()):
                    rows = flow.loc[preferred]
            values = _nonnegative_integer(rows[flow_n_col])
            if values is None or not len(values) or not bool(values.eq(locked_n).all()):
                return None
    return status_rows, locked_n


def _display_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Ordered exposure"
    return text.replace("_", " ").strip().title()


def _status_display_label(role: str, raw_value: Any) -> str:
    labels = {
        "valid_observed": "Valid observed",
        "observed": "Valid observed",
        "valid": "Valid observed",
        "available": "Valid observed",
        "no_source": "No source",
        "measured_source_present_but_summary_missing": (
            "Source present, summary missing"
        ),
        "contradictory_invalid": "Contradictory / invalid",
    }
    return labels.get(role, _display_label(raw_value))


def _declared_figure_data_families(summary: Dict[str, Any]) -> set[str]:
    families = {
        _normalise(summary.get("figure_data_family")),
    }
    contracts = summary.get("figure_data_contracts")
    if isinstance(contracts, list):
        families.update(
            _normalise(item.get("family"))
            for item in contracts
            if isinstance(item, dict)
        )
    families.discard("")
    return families


def _exposure_display_label(raw_value: Any, category_labels: Sequence[str]) -> str:
    prefixes: list[str] = []
    for label in category_labels:
        match = re.match(r"^(.+?)\s+(?:stage|level|category)\b", label, re.I)
        if match is None:
            prefixes = []
            break
        prefixes.append(match.group(1).strip())
    if prefixes and len({_normalise(value) for value in prefixes}) == 1:
        return prefixes[0]
    return _display_label(raw_value)


def render_ordered_distribution_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
) -> Optional[str]:
    """Render a two-panel ordered distribution from the exact parent outputs."""

    parent_step_id = str(current_step_id).removesuffix("_figure")
    if not parent_step_id or parent_step_id == str(current_step_id):
        return None
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    parent_summary_path = parent_out / "step_summary.json"
    try:
        summary_payload = (
            preverified_parent_artifacts.get("step_summary.json")
            if preverified_parent_artifacts is not None
            else None
        )
        parent_summary = json.loads(
            summary_payload.decode("utf-8")
            if summary_payload is not None
            else parent_summary_path.read_text(encoding="utf-8")
        )
    except Exception:
        return None
    if not isinstance(parent_summary, dict):
        return None
    method_is_legacy_adapter = _normalise(parent_summary.get("method")) == (
        "ordinal_exposure_derivation_and_quality_control"
    )
    declared_families = _declared_figure_data_families(parent_summary)
    if declared_families:
        if declared_families != {"ordered_category_distribution"}:
            return None
    elif not method_is_legacy_adapter:
        return None

    candidate = _candidate_table(parent_out, preverified_parent_artifacts)
    if candidate is None:
        return None
    source_path, frame, level_col, count_col = candidate

    levels = pd.to_numeric(frame[level_col], errors="coerce")
    level_rows = levels.notna()
    plot = frame.loc[level_rows].copy()
    plot["__level"] = levels[level_rows].astype(int).to_numpy()
    plot = (
        plot.sort_values("__level")
        .reset_index()
        .rename(columns={"index": "__source_row_index"})
    )
    counts = _nonnegative_integer(plot[count_col])
    if counts is None or plot["__level"].duplicated().any():
        return None
    observed_n = int(counts.sum())
    if observed_n <= 0:
        return None

    # Rebuild the level-row mask after sorting so source percentage rows align.
    conditional_col = _resolve_column(frame, _CONDITIONAL_PERCENT_COLUMNS)
    expected_pct = 100.0 * counts.astype(float) / float(observed_n)
    if conditional_col is not None:
        source_pct = pd.to_numeric(plot[conditional_col], errors="coerce")
        if source_pct.isna().any() or bool(
            (source_pct - expected_pct).abs().gt(0.05).any()
        ):
            return None
        # Plot and export the exact count/denominator calculation.  The source
        # percentage is a cross-check, not a value to copy: rounded source
        # percentages must never violate the source-data identity
        # ``percentage == 100 * count / denominator``.
        percentages = expected_pct
        percentage_source = conditional_col
        percentage_derived = True
    else:
        percentages = expected_pct
        percentage_source = "derived_from_count_over_valid_observed"
        percentage_derived = True

    if not _optional_values_match(
        frame=plot,
        column_names=_CONDITIONAL_FRACTION_COLUMNS,
        expected=counts.astype(float) / float(observed_n),
        tolerance=0.0005,
    ):
        return None

    availability = _availability_distribution(
        frame=frame,
        parent_out=parent_out,
        parent_summary=parent_summary,
        level_col=level_col,
        count_col=count_col,
        observed_n=observed_n,
        preverified_parent_artifacts=preverified_parent_artifacts,
    )
    if availability is None:
        return None
    status_rows, locked_n = availability
    if not _optional_values_match(
        frame=plot,
        column_names=_LOCKED_PERCENT_COLUMNS,
        expected=100.0 * counts.astype(float) / float(locked_n),
        tolerance=0.05,
    ):
        return None
    if not _optional_values_match(
        frame=plot,
        column_names=_LOCKED_FRACTION_COLUMNS,
        expected=counts.astype(float) / float(locked_n),
        tolerance=0.0005,
    ):
        return None
    unavailable_n = int(locked_n - observed_n)
    availability_counts = status_rows["__count"].astype(int).tolist()
    availability_pct = [100.0 * value / locked_n for value in availability_counts]
    availability_roles = status_rows["__status_role"].tolist()
    status_col = _resolve_column(frame, _STATUS_COLUMNS)
    if status_col is None:
        return None
    availability_labels = [
        _status_display_label(role, value)
        for role, value in zip(availability_roles, status_rows[status_col].tolist())
    ]

    label_col = _resolve_column(
        plot, ("stage_label", "level_label", "label", "category_label")
    )
    if label_col is not None:
        labels = [str(value) for value in plot[label_col].tolist()]
    else:
        labels = [f"Level {int(value)}" for value in plot["__level"].tolist()]
    exposure_value = (
        parent_summary.get("primary_exposure")
        or parent_summary.get("exposure")
        or parent_summary.get("primary_exposure_source")
        or "ordered exposure"
    )
    exposure_label = _exposure_display_label(exposure_value, labels)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    source_copy = out_dir / f"{stem}_source_data.csv"
    source_rows: list[Dict[str, Any]] = []
    for index, row in plot.iterrows():
        source_rows.append(
            {
                "panel_id": "A",
                "panel_role": "distribution",
                "category": labels[index],
                "ordered_level": int(row["__level"]),
                "source_status_role": "valid_observed",
                "n": int(counts.iloc[index]),
                "count": int(counts.iloc[index]),
                "percentage": float(percentages.iloc[index]),
                "denominator": observed_n,
                "denominator_definition": "valid_observed",
                "source_table": source_path.name,
                "source_row_index": int(row["__source_row_index"]),
                "source_percentage_column": percentage_source,
                "source_transform": "count_over_valid_observed_denominator",
            }
        )
    for row_index, category, role, count, percentage in zip(
        status_rows["__source_row_index"].astype(int).tolist(),
        availability_labels,
        availability_roles,
        availability_counts,
        availability_pct,
    ):
        source_rows.append(
            {
                "panel_id": "B",
                "panel_role": "data_quality",
                "category": category,
                "ordered_level": None,
                "source_status_role": role,
                "n": int(count),
                "count": int(count),
                "percentage": float(percentage),
                "denominator": int(locked_n),
                "denominator_definition": "locked_analysis_cohort",
                "source_table": source_path.name,
                "source_row_index": int(row_index),
                "source_percentage_column": "derived_from_count_over_locked_cohort",
                "source_transform": "count_over_locked_analysis_cohort",
            }
        )
    pd.DataFrame(source_rows).to_csv(source_copy, index=False)

    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 90 / 25.4),
        gridspec_kw={"width_ratios": [1.35, 0.85]},
    )
    category_colors = [
        palette["blue_soft"],
        "#7FA6C9",
        palette["teal"],
        palette["blue"],
    ]
    colors = [
        category_colors[index % len(category_colors)] for index in range(len(plot))
    ]
    bars = ax_a.bar(range(len(plot)), percentages, color=colors, width=0.68)
    ax_a.set_xticks(range(len(plot)))
    ax_a.set_xticklabels(labels, rotation=0)
    ax_a.set_ylabel("Valid-observed records (%)")
    ax_a.set_xlabel(f"{exposure_label} (ordered category)")
    ax_a.set_title("Ordered category distribution", loc="left", pad=4)
    ax_a.set_ylim(0, min(100.0, max(20.0, float(percentages.max()) * 1.28)))
    ax_a.grid(axis="y", color=palette["neutral_light"], linewidth=0.55, zorder=0)
    for bar, count, percentage in zip(bars, counts, percentages):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.3,
            f"{float(percentage):.1f}%\n(n={int(count):,})",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    add_panel_label(ax_a, "A", x=-0.12, y=1.03)

    availability_colors = [
        palette["blue"]
        if role in _VALID_OBSERVED_ROLES
        else palette["orange"]
        if count > 0
        else palette["neutral_light"]
        for role, count in zip(availability_roles, availability_counts)
    ]
    bars_b = ax_b.barh(
        range(len(availability_labels)),
        availability_pct,
        color=availability_colors,
        height=0.52,
    )
    ax_b.set_yticks(range(len(availability_labels)))
    ax_b.set_yticklabels(availability_labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Locked analysis cohort (%)")
    ax_b.set_title("Source availability", loc="left", pad=4)
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55, zorder=0)
    for bar, count, percentage in zip(bars_b, availability_counts, availability_pct):
        text_x = min(max(float(percentage) + 1.0, 3.0), 94.0)
        ax_b.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{float(percentage):.2f}% (n={int(count):,})",
            va="center",
            ha="left" if percentage < 92 else "right",
            color=palette["baseline"] if percentage < 92 else "white",
            fontsize=6.5,
        )
    add_panel_label(ax_b, "B", x=-0.18, y=1.03)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.89, wspace=0.48)

    contract = make_figure_contract(
        figure_id=f"figure:{stem}",
        core_claim=(
            "The authoritative ordered exposure distribution is shown among "
            "valid-observed records, with source availability accounted against "
            "the locked analysis cohort."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=90.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Ordered category distribution",
                "role": "distribution",
                "claim": (
                    "Ordered-category counts and percentages use the same "
                    "valid-observed denominator."
                ),
                "evidence_ids": [source_copy.name],
                "metadata": {"planner_product_slots": ["distribution"]},
            },
            {
                "panel_id": "B",
                "title": "Source availability",
                "role": "data_quality",
                "claim": (
                    "Valid-observed and unavailable records reconcile to the "
                    "locked analysis cohort."
                ),
                "evidence_ids": [source_copy.name],
                "metadata": {"planner_product_slots": ["availability"]},
            },
        ],
        source_data=[source_copy.name],
        statistics_note=(
            "Percentages are deterministic count/denominator calculations. "
            "Panel A is conditional on valid-observed records; panel B uses the "
            "locked analysis cohort."
        ),
    )
    outputs = save_publication_figure(
        fig,
        out_dir / stem,
        contract=contract,
        formats=("png", "svg", "pdf", "tiff"),
        dpi=300,
    )
    plt.close(fig)

    summary = {
        "step_id": current_step_id,
        "method": "deterministic_ordered_category_distribution_figure",
        "analysis_family": "descriptive",
        "rendering_only": True,
        "status": "completed",
        "source_step_id": parent_step_id,
        "source_table": str(source_path),
        "source_data_csv": str(source_copy),
        "figure_files": [
            path.name for key, path in outputs.items() if key != "contract"
        ],
        "source_data_files": [source_copy.name],
        "figure_path": f"{stem}.png",
        "figure_contract": f"{stem}.figure_contract.json",
        "ordered_levels": [int(value) for value in plot["__level"].tolist()],
        "valid_observed_n": observed_n,
        "unavailable_n": unavailable_n,
        "availability_statuses": availability_roles,
        "locked_analysis_cohort_n": locked_n,
        "availability_reconciles_to_locked_cohort": bool(
            observed_n + unavailable_n == locked_n
        ),
        "conditional_percentage_sum": float(percentages.sum()),
        "conditional_percentages_sum_to_100": bool(
            math.isclose(float(percentages.sum()), 100.0, abs_tol=0.05)
        ),
        "percentage_source": percentage_source,
        "percentage_derived": percentage_derived,
        "denominator_contract": {
            "panel_a": "valid_observed",
            "panel_b": "locked_analysis_cohort",
        },
        "warnings": [],
        "skipped": [],
        "errors": [],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return "ordered_category_distribution_publication_bundle_v1"


__all__ = ["render_ordered_distribution_bundle_from_prior_outputs"]
