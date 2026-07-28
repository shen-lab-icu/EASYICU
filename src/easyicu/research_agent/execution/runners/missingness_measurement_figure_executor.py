"""Deterministic renderer for a closed missingness / measurement-process pair.

The executor consumes exactly the two digest-bound audit tables declared by one
direct parent under ``all_rows`` contracts and renders the availability and
measurement-process context the parent already measured.  It does not read the
cohort, choose variables, redefine missingness, or fit a model.

The two tables carry **two structurally different row kinds**, and conflating
them is what this executor exists to get right.  Counting rows (source
missingness, valid-observed totals, categorical level distributions) reconcile
``count`` against ``denominator``.  Distribution-summary rows (median, IQR)
legitimately carry an empty ``count`` because a median is not a tally; they
reconcile through ``summary_value``/``q1``/``q3`` instead.  A validator that
demands a count from every row rejects the summary rows as malformed
accounting when they are exactly what the parent's schema prescribes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping

import pandas as pd

from ...figures.publication import (
    add_panel_label,
    apply_publication_style,
    make_figure_contract,
    save_publication_figure,
)
from ...schema import AnalysisStep
from .figure_input_capability import TypedInputCapability

__all__ = [
    "MEASUREMENT_PROCESS_AUDIT_INPUT",
    "MISSINGNESS_MEASUREMENT_AUDIT_INPUT",
    "MISSINGNESS_MEASUREMENT_FIGURE_INPUTS",
    "missingness_measurement_figure_executor_code",
    "missingness_measurement_figure_executor_owns_step",
    "run_missingness_measurement_figure",
]


MISSINGNESS_MEASUREMENT_AUDIT_INPUT = "table:missingness_measurement_audit"
MEASUREMENT_PROCESS_AUDIT_INPUT = "table:measurement_process_audit"
MISSINGNESS_MEASUREMENT_FIGURE_INPUTS = (
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT,
    MEASUREMENT_PROCESS_AUDIT_INPUT,
)
_SUPPORTED_FIGURE_PRODUCTS = frozenset(
    {
        "data_quality",
        "missingness_measurement_audit",
        "missingness_measurement",
    }
)
_AUDIT_COLUMNS = (
    "variable",
    "metric",
    "level",
    "count",
    "percentage",
    "denominator",
    "valid_observed_denominator",
    "raw_nonfinite_n",
    "plausibility_flag_n",
    "summary_value",
    "q1",
    "q3",
    "notes",
)
_PROCESS_COLUMNS = (
    "variable",
    "process_measure",
    "level",
    "count",
    "percentage",
    "denominator",
    "valid_observed_denominator",
    "median",
    "q1",
    "q3",
    "nonfinite_n",
    "notes",
)
# Row kinds in the missingness audit.  ``_COUNT_METRICS`` tally stays against a
# denominator; ``_SUMMARY_METRICS`` describe the distribution of the observed
# values and carry no count by construction.
_COUNT_METRICS = frozenset({"missing", "valid_observed", "level_distribution"})
_SUMMARY_METRICS = frozenset({"median", "iqr"})
_PRODUCT_BY_INPUT = {
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT: "missingness_measurement_audit",
    MEASUREMENT_PROCESS_AUDIT_INPUT: "measurement_process_audit",
}
_COLUMNS_BY_INPUT = {
    MISSINGNESS_MEASUREMENT_AUDIT_INPUT: _AUDIT_COLUMNS,
    MEASUREMENT_PROCESS_AUDIT_INPUT: _PROCESS_COLUMNS,
}


def _method_head(value: Any) -> str:
    return str(value or "").strip().lower().split(" with ", 1)[0]


def _figure_product(value: Any) -> str | None:
    kind, separator, product = str(value or "").strip().partition(":")
    if (
        kind != "figure"
        or not separator
        or not re.fullmatch(r"[a-z][a-z0-9_]*", product)
    ):
        return None
    return product


#: ``run_missingness_measurement_figure`` indexes both bindings and builds a
#: panel from each, so neither may be declared optional.  A plan that names
#: only the missingness audit is refused here and stays refused: the E1
#: protocol layer is what requires the process audit be produced, and this
#: renderer is not the place to paper over a plan that did not promise it.
MISSINGNESS_MEASUREMENT_FIGURE_CAPABILITY = TypedInputCapability(
    required=frozenset(MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
)


def missingness_measurement_figure_executor_owns_step(step: AnalysisStep) -> bool:
    """Return whether every scientific choice is fixed by the typed contract."""

    products = [_figure_product(value) for value in step.expected_outputs]
    if not MISSINGNESS_MEASUREMENT_FIGURE_CAPABILITY.admits_step(step):
        return False
    return bool(
        step.planned_analysis_role == "auxiliary"
        and _method_head(step.method) == "visualization"
        and len(products) == 1
        and products[0] in _SUPPORTED_FIGURE_PRODUCTS
        and not step.model_requirements
        and step.table_one_spec is None
        and step.trajectory_stability_spec is None
    )


def missingness_measurement_figure_executor_code(step: AnalysisStep) -> str:
    """Return the small sandbox entrypoint for the exact declared figure."""

    if not missingness_measurement_figure_executor_owns_step(step):
        raise ValueError(
            "The step is not owned by the missingness/measurement renderer"
        )
    product = _figure_product(step.expected_outputs[0])
    assert product is not None
    return textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        from easyicu.research_agent.execution.runners.missingness_measurement_figure_executor import (
            run_missingness_measurement_figure,
        )

        run_missingness_measurement_figure(
            out_dir=Path(os.environ["STEP_OUT_DIR"]),
            run_dir=Path(os.environ["EASYICU_RUN_DIR"]),
            resolved_inputs=Path(os.environ["EASYICU_RESOLVED_INPUTS_JSON"]),
            step_id={step.step_id!r},
            figure_product={product!r},
        )
        """
    ).strip()


def _canonical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_one_binding(
    *,
    run_dir: Path,
    inputs: Mapping[str, Any],
    input_key: str,
) -> tuple[pd.DataFrame, Mapping[str, Any], str]:
    binding = inputs.get(input_key)
    if not isinstance(binding, dict):
        raise ValueError(f"{input_key} binding is absent")
    product = _PRODUCT_BY_INPUT[input_key]
    expected_columns = list(_COLUMNS_BY_INPUT[input_key])
    expected_sha256 = str(binding.get("sha256") or "")
    relative_path = binding.get("relative_path")
    product_contract = binding.get("product_contract")
    consumption = binding.get("consumption_contract")
    identity = binding.get("identity_row")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(product_contract, dict)
        or not isinstance(consumption, dict)
        or not isinstance(identity, dict)
        or binding.get("declared_kind") != "table"
        or binding.get("evidence_kind") != "table"
        or binding.get("product") != product
        or identity.get("input_key") != input_key
        or identity.get("product") != product
        or identity.get("sha256") != expected_sha256
        or consumption.get("input_key") != input_key
        or consumption.get("mode") != "all_rows"
        or consumption.get("artifact_sha256") != expected_sha256
    ):
        raise ValueError(f"{input_key} authority binding is incomplete")

    path = (Path(run_dir).resolve() / relative_path).resolve()
    try:
        path.relative_to(Path(run_dir).resolve())
    except ValueError as exc:
        raise ValueError(f"{input_key} binding escapes the run directory") from exc
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".csv":
        raise ValueError(f"{input_key} must be a regular bound CSV")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError(f"{input_key} digest verification failed")

    row_count = product_contract.get("row_count")
    if (
        product_contract.get("columns") != expected_columns
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 1
        or consumption.get("verified_row_count") != row_count
    ):
        raise ValueError(f"{input_key} product contract is unsupported")
    frame = pd.read_csv(path)
    if list(frame.columns) != expected_columns or len(frame) != row_count:
        raise ValueError(f"{input_key} bytes disagree with its product contract")
    if _canonical_sha256(path) != expected_sha256:
        raise ValueError(f"{input_key} changed while it was being read")
    return frame, binding, f"{product}.csv"


def _load_bindings(
    *,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
) -> dict[str, tuple[pd.DataFrame, Mapping[str, Any], str]]:
    if isinstance(resolved_inputs, Mapping):
        payload = dict(resolved_inputs)
    else:
        payload = json.loads(Path(resolved_inputs).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("step_id") != step_id:
        raise ValueError("resolved-input manifest does not belong to this step")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(
        MISSINGNESS_MEASUREMENT_FIGURE_INPUTS
    ):
        raise ValueError("exact audit-table bindings are absent or widened")
    return {
        input_key: _load_one_binding(
            run_dir=run_dir,
            inputs=inputs,
            input_key=input_key,
        )
        for input_key in MISSINGNESS_MEASUREMENT_FIGURE_INPUTS
    }


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    parsed = _finite(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def _percentage_matches(value: Any, count: int, denominator: int) -> bool:
    parsed = _finite(value)
    if parsed is None:
        return False
    return math.isclose(
        parsed,
        100.0 * count / denominator,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def _summary_reconciles(
    metric: str, summary_value: float, q1: float, q3: float
) -> bool:
    """Check a distribution summary against the quartiles reported with it."""

    if metric == "median":
        return q1 <= summary_value <= q3
    if metric == "iqr":
        return math.isclose(summary_value, q3 - q1, rel_tol=1e-6, abs_tol=1e-6)
    return True


def _validate_audit_rows(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Validate every audit row against the kind its ``metric`` declares."""

    per_variable: dict[str, dict[str, Any]] = {}
    for index, row in frame.iterrows():
        variable = _text(row["variable"])
        metric = _text(row["metric"])
        if not variable:
            raise ValueError(f"missingness audit row {index} names no variable")
        if metric not in _COUNT_METRICS and metric not in _SUMMARY_METRICS:
            raise ValueError(
                f"missingness audit row {index} declares unsupported metric {metric!r}"
            )
        denominator = _integer(row["denominator"])
        if denominator is None or denominator <= 0:
            raise ValueError(
                f"missingness audit row {index} has no positive denominator"
            )
        entry = per_variable.setdefault(
            variable,
            {
                "denominator": denominator,
                "missing": None,
                "valid_observed": None,
                "level_total": 0,
                "level_rows": 0,
                "summary_metrics": set(),
            },
        )
        if entry["denominator"] != denominator:
            raise ValueError(
                f"variable {variable!r} mixes two denominators in the missingness audit"
            )

        if metric in _SUMMARY_METRICS:
            # A median or IQR is a distribution summary, not a tally: an empty
            # ``count``/``percentage`` here is the schema, not a defect.
            if not _blank(row["count"]) or not _blank(row["percentage"]):
                raise ValueError(
                    f"missingness audit row {index} reports {metric!r} as a tally"
                )
            summary_value = _finite(row["summary_value"])
            q1 = _finite(row["q1"])
            q3 = _finite(row["q3"])
            if summary_value is None or q1 is None or q3 is None or q1 > q3:
                raise ValueError(
                    f"missingness audit row {index} has an incomplete {metric!r} summary"
                )
            # ``q1 <= q3`` alone accepts a summary_value belonging to another
            # variable. Each summary must reconcile with the quartiles printed
            # beside it: a median falls inside its own range, an IQR is that
            # range's width.
            if not _summary_reconciles(metric, summary_value, q1, q3):
                raise ValueError(
                    f"missingness audit row {index} reports a {metric!r} that "
                    "does not reconcile with its own q1-q3 range"
                )
            entry["summary_metrics"].add(metric)
            continue

        if (
            not _blank(row["summary_value"])
            or not _blank(row["q1"])
            or not _blank(row["q3"])
        ):
            raise ValueError(
                f"missingness audit row {index} reports {metric!r} as a distribution"
            )
        count = _integer(row["count"])
        if count is None or count > denominator:
            raise ValueError(f"missingness audit row {index} has an invalid count")
        if not _percentage_matches(row["percentage"], count, denominator):
            raise ValueError(
                f"missingness audit row {index} percentage does not reconcile"
            )
        if metric == "level_distribution":
            if not _text(row["level"]):
                raise ValueError(
                    f"missingness audit row {index} is a level row with no level"
                )
            entry["level_total"] += count
            entry["level_rows"] += 1
        else:
            if entry[metric] is not None:
                raise ValueError(f"variable {variable!r} repeats its {metric!r} row")
            entry[metric] = count

    for variable, entry in per_variable.items():
        denominator = entry["denominator"]
        if entry["missing"] is None:
            raise ValueError(f"variable {variable!r} has no source-missingness row")
        observed = entry["valid_observed"]
        if observed is None and entry["level_rows"] == 0:
            raise ValueError(
                f"variable {variable!r} reports neither a valid-observed total nor levels"
            )
        if entry["summary_metrics"] and observed is None:
            raise ValueError(
                f"variable {variable!r} summarises a distribution it never counted"
            )
        if (
            observed is not None
            and entry["level_rows"]
            and entry["level_total"] != observed
        ):
            # A variable that reports both is reporting the same observed rows
            # twice; if the two disagree the figure would draw one of them and
            # silently contradict the other.
            raise ValueError(
                f"variable {variable!r} level counts sum to {entry['level_total']} "
                f"but its valid-observed total is {observed}"
            )
        closure = observed if observed is not None else entry["level_total"]
        if entry["missing"] + closure != denominator:
            raise ValueError(
                f"variable {variable!r} missing and observed counts do not "
                "partition its denominator"
            )
        entry["available"] = denominator - entry["missing"]
        entry["available_pct"] = 100.0 * entry["available"] / denominator
    return per_variable


def _validate_process_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate every measurement-process row and key it by measure and level."""

    cells: list[dict[str, Any]] = []
    partitions: dict[tuple[str, str], dict[str, int]] = {}
    seen_cells: set[tuple[str, str, str]] = set()
    for index, row in frame.iterrows():
        variable = _text(row["variable"])
        measure = _text(row["process_measure"])
        if not variable or not measure:
            raise ValueError(
                f"measurement-process row {index} names no variable or measure"
            )
        denominator = _integer(row["denominator"])
        count = _integer(row["count"])
        if denominator is None or denominator <= 0:
            raise ValueError(
                f"measurement-process row {index} has no positive denominator"
            )
        if count is None or count > denominator:
            raise ValueError(f"measurement-process row {index} has an invalid count")
        if not _percentage_matches(row["percentage"], count, denominator):
            raise ValueError(
                f"measurement-process row {index} percentage does not reconcile"
            )
        summary = [row["median"], row["q1"], row["q3"]]
        if any(not _blank(value) for value in summary):
            median, q1, q3 = (_finite(value) for value in summary)
            if median is None or q1 is None or q3 is None or q1 > q3:
                raise ValueError(
                    f"measurement-process row {index} has an incomplete summary"
                )
            if not _summary_reconciles("median", median, q1, q3):
                raise ValueError(
                    f"measurement-process row {index} reports a median outside "
                    "its own q1-q3 range"
                )
        level = _text(row["level"])
        # Every cell is one point on the panel, so the same coordinate may not
        # be stated twice. Levelled rows are additionally a partition, checked
        # below; an unlevelled measure is a single cell and has no such cover.
        coordinate = (variable, measure, level)
        if coordinate in seen_cells:
            raise ValueError(
                f"measurement-process row {index} repeats the cell "
                f"{measure!r} for {variable!r}"
            )
        seen_cells.add(coordinate)
        if level:
            # Levels of one measure are a partition of that measure's
            # denominator; this is structural and never keyed on the measure's
            # name.
            bucket = partitions.setdefault((variable, measure), {})
            if level in bucket:
                raise ValueError(
                    f"measurement-process row {index} repeats level {level!r}"
                )
            bucket[level] = count
        cells.append(
            {
                "variable": variable,
                "process_measure": measure,
                "level": level,
                "column": f"{measure}={level}" if level else measure,
                "count": count,
                "denominator": denominator,
                "percentage": 100.0 * count / denominator,
            }
        )

    for (variable, measure), bucket in partitions.items():
        denominators = {
            cell["denominator"]
            for cell in cells
            if cell["variable"] == variable and cell["process_measure"] == measure
        }
        if len(denominators) != 1 or sum(bucket.values()) != next(iter(denominators)):
            raise ValueError(
                f"levels of {measure!r} for {variable!r} do not partition its denominator"
            )
    return cells


def _reader_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", str(value or "")).strip()
    return cleaned if cleaned else "Variable"


def _level_text(value: str) -> str:
    """Render a level for a reader without inventing or rounding a category."""

    text = str(value or "").strip()
    parsed = _finite(text)
    if parsed is not None and parsed.is_integer():
        return str(int(parsed))
    return text


def _column_label(column: str) -> str:
    """Label one coverage column, keeping its level distinct from its measure."""

    measure, separator, level = str(column or "").partition("=")
    label = _reader_label(measure)
    if not separator:
        return label
    return f"{label}\n({_level_text(level)})"


def _write_source_projection(
    rows: pd.DataFrame,
    *,
    path: Path,
    source_table: str,
) -> None:
    export = rows.copy()
    export.insert(0, "source_row_index", export.index.astype(int))
    export.insert(1, "source_table", source_table)
    export.to_csv(path, index=False)


def run_missingness_measurement_figure(
    *,
    out_dir: Path,
    run_dir: Path,
    resolved_inputs: Path | Mapping[str, Any],
    step_id: str,
    figure_product: str,
) -> Mapping[str, Any]:
    """Render the verified missingness/measurement pair and write its contract."""

    if figure_product not in _SUPPORTED_FIGURE_PRODUCTS:
        raise ValueError("unsupported missingness/measurement figure product")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bindings = _load_bindings(
        run_dir=Path(run_dir),
        resolved_inputs=resolved_inputs,
        step_id=step_id,
    )
    audit_frame, audit_binding, audit_table = bindings[
        MISSINGNESS_MEASUREMENT_AUDIT_INPUT
    ]
    process_frame, process_binding, process_table = bindings[
        MEASUREMENT_PROCESS_AUDIT_INPUT
    ]
    per_variable = _validate_audit_rows(audit_frame)
    process_cells = _validate_process_rows(process_frame)

    audit_source = out_dir / f"{figure_product}_missingness_source_data.csv"
    process_source = out_dir / f"{figure_product}_measurement_process_source_data.csv"
    missingness_panel_source = (
        out_dir / f"{figure_product}_source_missingness_panel_source_data.csv"
    )
    _write_source_projection(audit_frame, path=audit_source, source_table=audit_table)
    _write_source_projection(
        process_frame,
        path=process_source,
        source_table=process_table,
    )
    # Every panel projection is a verbatim row subset of its parent, keeping the
    # parent's own values and row positions.  A derived column (an availability
    # complement, a reshaped coverage grid) cannot be traced back to any single
    # upstream row, so it is computed for validation and reported in the step
    # summary rather than published as if it were source data.
    missing_rows = audit_frame.loc[
        audit_frame["metric"].map(_text).eq("missing")
    ].sort_values("percentage", ascending=False)
    _write_source_projection(
        missing_rows,
        path=missingness_panel_source,
        source_table=audit_table,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    variables = [_text(value) for value in missing_rows["variable"]]
    columns = sorted({str(cell["column"]) for cell in process_cells})
    grid_variables = sorted({str(cell["variable"]) for cell in process_cells})
    height_mm = max(86.0, 26.0 + 5.4 * max(len(variables), len(grid_variables)))
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, height_mm / 25.4),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )

    positions = list(range(len(variables)))
    missing_pct = pd.to_numeric(missing_rows["percentage"]).to_numpy()
    missing_counts = pd.to_numeric(missing_rows["count"]).astype(int).to_numpy()
    bars = ax_a.barh(
        positions,
        missing_pct,
        color=palette["blue_soft"],
        height=0.62,
    )
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels([_reader_label(name) for name in variables])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Stays with no source value (%)")
    ax_a.set_title("Source missingness", loc="left", pad=4)
    ax_a.grid(axis="x", color=palette["neutral_light"], linewidth=0.55)
    for bar, percentage, missing_n in zip(bars, missing_pct, missing_counts):
        ax_a.text(
            min(float(percentage) + 1.0, 97.0),
            bar.get_y() + bar.get_height() / 2,
            f"{float(percentage):.1f}%  n={int(missing_n):,}",
            va="center",
            ha="left" if percentage < 88 else "right",
            fontsize=6.1,
        )
    add_panel_label(ax_a, "A", x=-0.30, y=1.02)

    matrix = [
        [
            next(
                (
                    float(cell["percentage"])
                    for cell in process_cells
                    if cell["variable"] == variable and cell["column"] == column
                ),
                float("nan"),
            )
            for column in columns
        ]
        for variable in grid_variables
    ]
    frame = pd.DataFrame(matrix, index=grid_variables, columns=columns)
    image = ax_b.imshow(
        frame.to_numpy(dtype=float),
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        aspect="auto",
    )
    ax_b.set_xticks(range(len(columns)))
    ax_b.set_xticklabels(
        [_column_label(column) for column in columns],
        rotation=35,
        ha="right",
    )
    ax_b.set_yticks(range(len(grid_variables)))
    ax_b.set_yticklabels([_reader_label(name) for name in grid_variables])
    ax_b.set_title("Measurement-process coverage", loc="left", pad=4)
    for row_index, variable in enumerate(grid_variables):
        for column_index, column in enumerate(columns):
            value = frame.iat[row_index, column_index]
            if math.isnan(float(value)):
                ax_b.text(
                    column_index,
                    row_index,
                    "–",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color=palette["neutral"],
                )
                continue
            ax_b.text(
                column_index,
                row_index,
                f"{float(value):.1f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if float(value) >= 55.0 else palette["blue"],
            )
    colorbar = fig.colorbar(image, ax=ax_b, fraction=0.032, pad=0.02)
    colorbar.set_label("Share of the measure's denominator (%)", fontsize=6.2)
    colorbar.ax.tick_params(labelsize=5.8)
    add_panel_label(ax_b, "B", x=-0.30, y=1.02)
    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.22, top=0.88, wspace=0.72)

    contract = make_figure_contract(
        figure_id=f"figure:{figure_product}",
        core_claim=(
            "Source availability and measurement-process coverage are rendered "
            "from two digest-verified parent audit tables."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=height_mm,
        panels=[
            {
                "panel_id": "A",
                "title": "Source missingness",
                "role": "data_quality",
                "claim": (
                    "Each audited variable's source-missingness share is the "
                    "parent table's own value; missing and observed counts "
                    "partition the parent-locked denominator."
                ),
                "evidence_ids": [missingness_panel_source.name],
                "metadata": {
                    "chart_type": "availability_panel",
                    "source_data": [missingness_panel_source.name],
                },
            },
            {
                "panel_id": "B",
                "title": "Measurement-process coverage",
                "role": "data_quality",
                "claim": (
                    "Each measurement-process measure is shown as a share of "
                    "its own denominator; blank cells are measures the parent "
                    "did not report for that variable."
                ),
                "evidence_ids": [process_source.name],
                "metadata": {
                    "chart_type": "coverage_heatmap",
                    "source_data": [process_source.name],
                },
            },
        ],
        source_data=[
            audit_source.name,
            process_source.name,
            missingness_panel_source.name,
        ],
        statistics_note=(
            "Percentages are recomputed from the sealed integer counts. "
            "Distribution-summary rows (median, IQR) are validated on "
            "summary_value/q1/q3 and are not counted as tallies. The executor "
            "validates all source rows and introduces no cohort, variable, "
            "missing-data, or modeling decision."
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
    figure_files = [path.name for key, path in outputs.items() if key != "contract"]
    source_files = [
        audit_source.name,
        process_source.name,
        missingness_panel_source.name,
    ]
    summary = {
        "step_id": step_id,
        "status": "ok",
        "analysis_status": "ok",
        "method": "deterministic_missingness_measurement_figure",
        "analysis_family": "data_quality",
        "deterministic_standard_analysis": "missingness_measurement_figure",
        "rendering_only": True,
        "source_inputs": list(MISSINGNESS_MEASUREMENT_FIGURE_INPUTS),
        "source_evidence_ids": {
            MISSINGNESS_MEASUREMENT_AUDIT_INPUT: audit_binding.get("evidence_id"),
            MEASUREMENT_PROCESS_AUDIT_INPUT: process_binding.get("evidence_id"),
        },
        "source_sha256": {
            MISSINGNESS_MEASUREMENT_AUDIT_INPUT: audit_binding.get("sha256"),
            MEASUREMENT_PROCESS_AUDIT_INPUT: process_binding.get("sha256"),
        },
        "source_rows_consumed": {
            MISSINGNESS_MEASUREMENT_AUDIT_INPUT: int(len(audit_frame)),
            MEASUREMENT_PROCESS_AUDIT_INPUT: int(len(process_frame)),
        },
        "audited_variable_count": int(len(per_variable)),
        "measurement_process_cell_count": int(len(process_cells)),
        "source_data_files": source_files,
        "figure_files": figure_files,
        "figure_path": f"{figure_product}.png",
        "figure_contract": f"{figure_product}.figure_contract.json",
        "contract_files": [f"{figure_product}.figure_contract.json"],
        "output_files": {f"figure:{figure_product}": f"{figure_product}.png"},
        "input_bindings": [
            {
                "input_key": input_key,
                "evidence_id": binding.get("evidence_id"),
                "sha256": binding.get("sha256"),
                "loaded": True,
                "row_count": int(len(loaded)),
            }
            for input_key, (loaded, binding, _) in bindings.items()
        ],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
