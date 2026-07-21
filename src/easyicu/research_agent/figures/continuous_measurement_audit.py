"""Render a closed continuous-distribution and measurement-availability audit.

The adapter consumes three exact Planner products from the direct parent:
distribution, missingness, and measurement-process tables.  It validates their
shared exposure, denominator, counts, percentages, and ordered row identities
before rendering.  It never selects a variable, cohort, time window, or model.
"""

from __future__ import annotations

import io
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from ..authority.parent_artifact import (
    _resolve_upstream_manifest_step,
    _verified_direct_parent_artifact_digests,
)
from ..contracts.declared_product import (
    read_digest_bound_artifact_snapshot,
    typed_product,
)
from .publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)

REPAIR_ID = "continuous_measurement_audit_publication_bundle_v1"
CONTROLLED_METHOD = "right_skewed_distribution_and_measurement_availability_audit"

_ROLE_SUFFIXES = {
    "distribution": ("distribution",),
    "missingness": ("missingness",),
    "measurement_process": ("measurement", "process"),
}
_DISTRIBUTION_COLUMNS = {
    "variable",
    "metric",
    "unit",
    "n",
    "denominator",
    "percentage",
    "median",
    "q25",
    "q75",
    "min",
    "max",
}
_MISSINGNESS_COLUMNS = {
    "variable",
    "status",
    "count",
    "denominator",
    "percentage",
    "missing_n",
    "missing_pct",
}
_MEASUREMENT_COLUMNS = {
    "variable",
    "status",
    "count",
    "denominator",
    "percentage",
    "unit",
    "measured_column",
    "count_column",
    "summary_column",
}
_MISSINGNESS_STATUSES = (
    "authoritative_value_observed",
    "authoritative_value_missing",
)
_MEASUREMENT_STATUSES = (
    "valid_observed",
    "no_source",
    "measured_source_present_but_summary_missing",
    "contradictory_invalid",
)


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _safe_csv_name(value: Any) -> Optional[str]:
    name = str(value or "").strip()
    path = Path(name)
    if not name or path.name != name or path.suffix.lower() != ".csv":
        return None
    return name


def _finite(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> Optional[int]:
    parsed = _finite(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _same(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=tolerance)


def _read_csv(payload: bytes, required: set[str]) -> Optional[pd.DataFrame]:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception:
        return None
    return frame if required <= set(frame.columns) else None


def _planner_table_roles(expected_outputs: Sequence[Any]) -> Optional[dict[str, str]]:
    products = [
        parsed[1]
        for raw in expected_outputs
        if (parsed := typed_product(raw)) is not None and parsed[0] == "table"
    ]
    selected: dict[str, str] = {}
    for role, suffix in _ROLE_SUFFIXES.items():
        matches = [
            product
            for product in products
            if tuple(product.split("_")[-len(suffix) :]) == suffix
        ]
        if len(matches) != 1:
            return None
        selected[role] = matches[0]
    return selected if len(set(selected.values())) == 3 else None


@dataclass(frozen=True)
class ContinuousMeasurementAuditInputs:
    distribution_path: Path
    missingness_path: Path
    measurement_path: Path
    exposure_column: str
    exposure_label: str
    unit: str
    metric_values: Mapping[str, float]
    distribution_row: pd.DataFrame
    missingness_rows: pd.DataFrame
    measurement_rows: pd.DataFrame
    observed_n: int
    denominator_n: int
    status_schema: tuple[str, ...]


def prepare_continuous_measurement_audit_inputs(
    *,
    parent_out: Path,
    parent_summary: Mapping[str, Any],
    planner_roles: Mapping[str, str],
    preverified_table_bytes: Mapping[str, bytes],
) -> Optional[ContinuousMeasurementAuditInputs]:
    """Validate one exact three-table parent snapshot."""

    if _normalise(parent_summary.get("method")) != CONTROLLED_METHOD:
        return None
    exposure = str(parent_summary.get("primary_exposure") or "").strip()
    unit = str(parent_summary.get("unit") or "").strip()
    output_files = parent_summary.get("output_files")
    cohort_policy = parent_summary.get("cohort_policy")
    if (
        not exposure
        or not isinstance(output_files, Mapping)
        or not isinstance(cohort_policy, Mapping)
    ):
        return None
    denominator_n = _integer(cohort_policy.get("final_cohort_n"))
    if denominator_n is None or denominator_n <= 0:
        return None

    names: dict[str, str] = {}
    for role, product in planner_roles.items():
        name = _safe_csv_name(output_files.get(f"table:{product}"))
        if name is None:
            return None
        names[role] = name
    if len(set(names.values())) != 3 or set(names.values()) != set(
        preverified_table_bytes
    ):
        return None

    distribution = _read_csv(
        preverified_table_bytes[names["distribution"]], _DISTRIBUTION_COLUMNS
    )
    missingness = _read_csv(
        preverified_table_bytes[names["missingness"]], _MISSINGNESS_COLUMNS
    )
    measurement = _read_csv(
        preverified_table_bytes[names["measurement_process"]], _MEASUREMENT_COLUMNS
    )
    if distribution is None or missingness is None or measurement is None:
        return None

    distribution_rows = distribution.loc[
        distribution["variable"].astype(str).eq(exposure)
        & distribution["metric"].map(_normalise).eq("distribution_summary")
    ].copy()
    if len(distribution_rows) != 1:
        return None
    distribution_row = distribution_rows.iloc[0]
    observed_n = _integer(distribution_row["n"])
    distribution_denominator = _integer(distribution_row["denominator"])
    distribution_percentage = _finite(distribution_row["percentage"])
    if (
        observed_n is None
        or observed_n <= 0
        or distribution_denominator != denominator_n
        or distribution_percentage is None
        or not _same(
            distribution_percentage,
            100.0 * observed_n / denominator_n,
        )
        or (unit and _normalise(distribution_row["unit"]) != _normalise(unit))
    ):
        return None
    metric_values = {
        key: _finite(distribution_row[key])
        for key in ("median", "q25", "q75", "min", "max")
    }
    if any(value is None for value in metric_values.values()):
        return None
    finite_metrics = {key: float(value) for key, value in metric_values.items()}
    if not (
        finite_metrics["min"]
        <= finite_metrics["q25"]
        <= finite_metrics["median"]
        <= finite_metrics["q75"]
        <= finite_metrics["max"]
    ):
        return None

    missingness_rows = missingness.loc[
        missingness["variable"].astype(str).eq(exposure)
    ].copy()
    missingness_rows["__status"] = missingness_rows["status"].map(_normalise)
    if missingness_rows["__status"].duplicated().any() or set(
        missingness_rows["__status"]
    ) != set(_MISSINGNESS_STATUSES):
        return None
    missingness_rows.insert(0, "source_row_index", missingness_rows.index.astype(int))
    missingness_rows = (
        missingness_rows.set_index("__status")
        .loc[list(_MISSINGNESS_STATUSES)]
        .reset_index(drop=True)
    )
    missing_counts = [_integer(value) for value in missingness_rows["count"]]
    missing_denominators = {
        _integer(value) for value in missingness_rows["denominator"]
    }
    if (
        any(value is None for value in missing_counts)
        or missing_denominators != {denominator_n}
        or [int(value) for value in missing_counts if value is not None][0]
        != observed_n
        or sum(int(value) for value in missing_counts if value is not None)
        != denominator_n
    ):
        return None
    missing_n = denominator_n - observed_n
    for row, count in zip(missingness_rows.to_dict("records"), missing_counts):
        assert count is not None
        missing_percentage = _finite(row["missing_pct"])
        row_percentage = _finite(row["percentage"])
        if (
            _integer(row["missing_n"]) != missing_n
            or missing_percentage is None
            or row_percentage is None
            or not _same(missing_percentage, 100.0 * missing_n / denominator_n)
            or not _same(row_percentage, 100.0 * count / denominator_n)
        ):
            return None

    measurement_rows = measurement.loc[
        measurement["summary_column"].astype(str).eq(exposure)
    ].copy()
    measurement_rows["__status"] = measurement_rows["status"].map(_normalise)
    measurement_rows = measurement_rows.loc[
        measurement_rows["__status"].isin(_MEASUREMENT_STATUSES)
    ].copy()
    if (
        measurement_rows["__status"].duplicated().any()
        or set(measurement_rows["__status"]) != set(_MEASUREMENT_STATUSES)
        or measurement_rows["measured_column"].astype(str).nunique() != 1
        or measurement_rows["count_column"].astype(str).nunique() != 1
    ):
        return None
    measurement_rows.insert(0, "source_row_index", measurement_rows.index.astype(int))
    measurement_rows = (
        measurement_rows.set_index("__status")
        .loc[list(_MEASUREMENT_STATUSES)]
        .reset_index(drop=True)
    )
    measurement_counts = [_integer(value) for value in measurement_rows["count"]]
    measurement_denominators = {
        _integer(value) for value in measurement_rows["denominator"]
    }
    if (
        any(value is None for value in measurement_counts)
        or measurement_denominators != {denominator_n}
        or sum(int(value) for value in measurement_counts if value is not None)
        != denominator_n
        or [int(value) for value in measurement_counts if value is not None][:2]
        != [observed_n, missing_n]
        or (unit and measurement_rows["unit"].map(_normalise).nunique() != 1)
        or (unit and _normalise(measurement_rows["unit"].iloc[0]) != _normalise(unit))
    ):
        return None
    for percentage, count in zip(measurement_rows["percentage"], measurement_counts):
        assert count is not None
        parsed_percentage = _finite(percentage)
        if parsed_percentage is None or not _same(
            parsed_percentage, 100.0 * count / denominator_n
        ):
            return None

    distribution_rows.insert(0, "source_row_index", distribution_rows.index.astype(int))
    return ContinuousMeasurementAuditInputs(
        distribution_path=parent_out / names["distribution"],
        missingness_path=parent_out / names["missingness"],
        measurement_path=parent_out / names["measurement_process"],
        exposure_column=exposure,
        exposure_label=exposure.replace("_", " ").strip(),
        unit=unit,
        metric_values=finite_metrics,
        distribution_row=distribution_rows,
        missingness_rows=missingness_rows,
        measurement_rows=measurement_rows,
        observed_n=observed_n,
        denominator_n=denominator_n,
        status_schema=_MEASUREMENT_STATUSES,
    )


def _continuous_measurement_audit_parent_digest_seal(
    run_dir: Path,
    figure_step_id: str,
) -> Optional[dict[str, str]]:
    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if (
        not isinstance(request_step, Mapping)
        or _normalise(request_step.get("method")) != CONTROLLED_METHOD
    ):
        return None
    planner_roles = _planner_table_roles(request_step.get("expected_outputs") or [])
    if planner_roles is None:
        return None
    digests = _verified_direct_parent_artifact_digests(run_dir, figure_step_id)
    if not digests or "step_summary.json" not in digests:
        return None
    parent_step_id = str(figure_step_id or "").removesuffix("_figure")
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    try:
        full_snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests=digests,
        )
        summary = json.loads(full_snapshot["step_summary.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(summary, Mapping):
        return None
    output_files = summary.get("output_files")
    if not isinstance(output_files, Mapping):
        return None
    table_names = {
        _safe_csv_name(output_files.get(f"table:{product}"))
        for product in planner_roles.values()
    }
    if None in table_names or len(table_names) != 3:
        return None
    names = {str(name) for name in table_names}
    if not names <= set(full_snapshot):
        return None
    selected_bytes = {name: full_snapshot[name] for name in names}
    prepared = prepare_continuous_measurement_audit_inputs(
        parent_out=parent_out,
        parent_summary=summary,
        planner_roles=planner_roles,
        preverified_table_bytes=selected_bytes,
    )
    if prepared is None:
        return None
    required_names = {"step_summary.json", *names}
    return {name: digests[name] for name in sorted(required_names)}


def _wrapped(value: str, width: int = 23) -> str:
    lines = textwrap.wrap(
        value.replace("_", " "),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else value


def render_continuous_measurement_audit_bundle(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_artifacts: Mapping[str, bytes],
) -> Optional[str]:
    """Render one already sealed three-table audit snapshot."""

    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    if not parent_step_id or parent_step_id == str(current_step_id or ""):
        return None
    try:
        summary = json.loads(preverified_parent_artifacts["step_summary.json"].decode())
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    request_step = _resolve_upstream_manifest_step(Path(run_dir), current_step_id)
    if not isinstance(summary, Mapping) or not isinstance(request_step, Mapping):
        return None
    planner_roles = _planner_table_roles(request_step.get("expected_outputs") or [])
    if planner_roles is None:
        return None
    table_bytes = {
        name: payload
        for name, payload in preverified_parent_artifacts.items()
        if name != "step_summary.json"
    }
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    prepared = prepare_continuous_measurement_audit_inputs(
        parent_out=parent_out,
        parent_summary=summary,
        planner_roles=planner_roles,
        preverified_table_bytes=table_bytes,
    )
    if prepared is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_source = out_dir / "distribution_panel_source_data.csv"
    missingness_source = out_dir / "missingness_audit_source_data.csv"
    availability_source = out_dir / "measurement_process_source_data.csv"
    prepared.distribution_row.to_csv(distribution_source, index=False)
    prepared.missingness_rows.to_csv(missingness_source, index=False)
    prepared.measurement_rows.to_csv(availability_source, index=False)

    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 88 / 25.4),
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )
    values = prepared.metric_values
    ax_a.hlines(
        0, values["min"], values["max"], color=palette["neutral"], linewidth=1.2
    )
    ax_a.hlines(0, values["q25"], values["q75"], color=palette["blue"], linewidth=7.0)
    ax_a.plot(values["median"], 0, "o", color=palette["baseline"], markersize=4.5)
    ax_a.set_yticks([])
    axis_label = prepared.exposure_label
    if prepared.unit:
        axis_label = f"{axis_label} ({prepared.unit})"
    ax_a.set_xlabel(axis_label)
    ax_a.set_title("Observed distribution", loc="left", pad=4)
    ax_a.text(
        0.5,
        0.72,
        f"median {values['median']:g}  (IQR {values['q25']:g}–{values['q75']:g})\n"
        f"n={prepared.observed_n:,}",
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
    )
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    add_panel_label(ax_a, "A", x=-0.08, y=1.03)

    counts = pd.to_numeric(prepared.measurement_rows["count"]).astype(int)
    percentages = counts.astype(float) * 100.0 / prepared.denominator_n
    positions = range(len(prepared.status_schema))
    bars = ax_b.barh(positions, percentages, color=palette["blue_soft"], height=0.58)
    ax_b.set_yticks(list(positions))
    ax_b.set_yticklabels([_wrapped(status) for status in prepared.status_schema])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Analysis cohort (%)")
    ax_b.set_title("Measurement availability", loc="left", pad=4)
    ax_b.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for bar, percentage, count in zip(bars, percentages, counts):
        ax_b.text(
            min(float(percentage) + 1.0, 96.0),
            bar.get_y() + bar.get_height() / 2,
            f"{float(percentage):.1f}% (n={int(count):,})",
            va="center",
            ha="left" if percentage < 94 else "right",
            fontsize=6.3,
        )
    add_panel_label(ax_b, "B", x=-0.12, y=1.03)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.86, wspace=0.46)

    stem = "continuous_distribution_measurement_availability"
    source_files = [
        distribution_source.name,
        missingness_source.name,
        availability_source.name,
    ]
    contract = make_figure_contract(
        figure_id=f"figure:{stem}",
        core_claim=(
            "The Planner-selected continuous distribution and measurement "
            "availability are rendered from the verified direct-parent tables."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=88.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Observed distribution",
                "role": "descriptive_result",
                "claim": "Median, interquartile range, and range among observed records.",
                "evidence_ids": [distribution_source.name],
            },
            {
                "panel_id": "B",
                "title": "Measurement availability",
                "role": "data_quality",
                "claim": "Predeclared measurement-status counts partition the cohort.",
                "evidence_ids": [missingness_source.name, availability_source.name],
            },
        ],
        source_data=source_files,
        statistics_note=(
            "No bins, cohort filters, exposure definitions, outcomes, or models "
            "are selected by this rendering adapter."
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
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    rendered_summary = {
        "step_id": current_step_id,
        "method": "deterministic_continuous_measurement_audit_figure",
        "analysis_family": "descriptive",
        "rendering_only": True,
        "status": "completed",
        "source_step_id": parent_step_id,
        "source_tables": [
            prepared.distribution_path.name,
            prepared.missingness_path.name,
            prepared.measurement_path.name,
        ],
        "source_data_files": source_files,
        "figure_files": figure_files,
        "figure_path": f"{stem}.png",
        "figure_contract": f"{stem}.figure_contract.json",
        "output_files": {"figure:publication_figure": f"{stem}.png"},
        "warnings": [],
        "skipped": [],
        "errors": [],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(rendered_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return REPAIR_ID


__all__ = [
    "CONTROLLED_METHOD",
    "REPAIR_ID",
    "_continuous_measurement_audit_parent_digest_seal",
    "prepare_continuous_measurement_audit_inputs",
    "render_continuous_measurement_audit_bundle",
]
