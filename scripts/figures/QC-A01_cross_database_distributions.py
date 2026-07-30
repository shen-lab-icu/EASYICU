#!/usr/bin/env python3
"""QC-A01: Publication QC distributions for EasyICU module exports.

This is a record-level extraction-QC figure set, except that binary concepts are
summarised as the percentage of cohort stays with at least one positive record.
Continuous densities use every finite value inside a shared display range; exact
tail counts remain in the source-data and audit exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d


def _prefer_checkout_src(script_path: Path | None = None) -> Path | None:
    """Put this checkout's ``src`` first when the script runs from a clone.

    A server may also have an older editable EasyICU installation.  Direct
    execution from ``scripts/figures`` must use the matching source tree on
    Windows, macOS and Linux without relying on a shell-specific ``PYTHONPATH``.
    Installed/copied scripts simply leave ``sys.path`` unchanged.
    """

    script = (script_path or Path(__file__)).resolve()
    try:
        checkout_root = script.parents[2]
    except IndexError:
        return None
    checkout_src = checkout_root / "src"
    if not (checkout_src / "easyicu" / "__init__.py").is_file():
        return None

    normalized_src = os.path.normcase(os.path.realpath(os.fspath(checkout_src)))
    retained: list[str] = []
    for entry in sys.path:
        try:
            normalized_entry = os.path.normcase(
                os.path.realpath(os.fspath(entry or Path.cwd()))
            )
        except TypeError:
            retained.append(entry)
            continue
        if normalized_entry != normalized_src:
            retained.append(entry)
    sys.path[:] = [os.fspath(checkout_src), *retained]
    return checkout_src


CHECKOUT_SRC = _prefer_checkout_src()

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DATABASES = ("aumc", "eicu", "hirid", "mimic", "miiv", "sic")
DATABASE_LABELS = {
    "aumc": "AUMCdb",
    "eicu": "eICU-CRD",
    "hirid": "HiRID",
    "mimic": "MIMIC-III",
    "miiv": "MIMIC-IV",
    "sic": "SICdb",
}
DATABASE_COLORS = {
    "aumc": "#0F4D92",
    "eicu": "#D55E00",
    "hirid": "#009E73",
    "mimic": "#8E5EA2",
    "miiv": "#E6A400",
    "sic": "#4E88C7",
}
DATABASE_MARKERS = {
    "aumc": "o",
    "eicu": "s",
    "hirid": "^",
    "mimic": "D",
    "miiv": "P",
    "sic": "X",
}
ID_COLUMNS = {
    "admissionid",
    "patientunitstayid",
    "patientid",
    "icustay_id",
    "stay_id",
    "CaseID",
}
INDEX_COLUMNS = ID_COLUMNS | {"charttime"}
MAX_DISCRETE_LEVELS = 41
MAX_TRACKED_LEVELS = 65
MAX_CATEGORIES_SHOWN = 12
SAMPLE_LIMIT_PER_DATABASE = 50_000
BATCH_SIZE = 500_000
DISPLAY_Q_LOW = 0.005
DISPLAY_Q_HIGH = 0.995
width_mm = 183
DOUBLE_COLUMN_WIDTH_IN = width_mm / 25.4
PANELS_PER_PAGE = 12
NON_DISPLAY_UNITS = frozenset({"boolean", "category", "datetime"})


@dataclass
class ColumnStats:
    database: str
    row_count: int = 0
    non_null: int = 0
    null_count: int = 0
    non_finite: int = 0
    minimum: float | None = None
    maximum: float | None = None
    sample: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    value_counts: Counter[Any] = field(default_factory=Counter)
    level_tracking_complete: bool = True
    integer_like: bool = True
    arrow_type: str = ""


@dataclass
class PlotPayload:
    module: str
    variable: str
    description: str
    unit: str | None
    kind: str
    data: pd.DataFrame
    subtitle: str
    footnote: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        help=(
            "Concept catalog JSON. Required for a full scan; when supplied with "
            "--render-only, refreshes audit metadata without rescanning Parquet."
        ),
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        help=(
            "Source run_metadata.json. Defaults to the sibling of --input-root "
            "(for example, RUN/exports -> RUN/run_metadata.json)."
        ),
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        help="Optional module subset. Default: every module found in all six exports.",
    )
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--panels-per-page",
        type=int,
        default=PANELS_PER_PAGE,
        help="Maximum panels per physical page (default: 12).",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild module figures from existing source_data and audit CSVs.",
    )
    return parser.parse_args()


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Concept catalog must be a JSON object keyed by concept name")
    catalog = {str(k): v for k, v in data.items() if isinstance(v, dict)}

    # The JSON catalog is the source of truth for directly extracted concepts.
    # Derived concepts (for example, rolling urine-output rates) are registered
    # in the runtime catalog, so use it only to fill metadata absent from JSON.
    from easyicu.concept.catalog import CONCEPT_DICTIONARY

    for concept, metadata in CONCEPT_DICTIONARY.items():
        description, _, unit = metadata
        item = catalog.setdefault(concept, {})
        if item.get("description") in (None, ""):
            item["description"] = description
        if item.get("unit") in (None, "", []):
            item["unit"] = unit or None
    return catalog


def concept_metadata(
    catalog: dict[str, dict[str, Any]], variable: str
) -> tuple[str, str | None, float | None, float | None]:
    item = catalog.get(variable, {})
    description = str(item.get("description") or variable.replace("_", " "))
    unit_value = item.get("unit")
    if isinstance(unit_value, list):
        unit = str(unit_value[0]) if unit_value else None
    elif unit_value in (None, ""):
        unit = None
    else:
        unit = str(unit_value)
    lower = numeric_or_none(item.get("min"))
    upper = numeric_or_none(item.get("max"))
    return description, unit, lower, upper


def numeric_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def display_unit(unit: str | None) -> str | None:
    """Return a reader-facing unit while retaining canonical audit metadata."""

    if unit is None:
        return None
    value = str(unit).strip()
    if not value or value.casefold() in NON_DISPLAY_UNITS:
        return None
    return value


def module_names(root: Path) -> list[str]:
    module_sets = [
        {path.stem for path in (root / database).glob("*.parquet")}
        for database in DATABASES
    ]
    if not all(module_sets):
        missing = [db for db, modules in zip(DATABASES, module_sets) if not modules]
        raise FileNotFoundError(f"No Parquet modules found for: {', '.join(missing)}")
    union = set.union(*module_sets)
    return sorted(union)


def module_variables(root: Path, module: str) -> list[str]:
    variables: set[str] = set()
    for database in DATABASES:
        path = root / database / f"{module}.parquet"
        if not path.exists():
            continue
        variables.update(pq.ParquetFile(path).schema_arrow.names)
    return sorted(variables - INDEX_COLUMNS)


def field_type(path: Path, column: str) -> pa.DataType | None:
    schema = pq.ParquetFile(path).schema_arrow
    index = schema.get_field_index(column)
    return schema.field(index).type if index >= 0 else None


def _finite_numeric_values(array: pa.Array) -> tuple[np.ndarray, int]:
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=float)
    finite = np.isfinite(values)
    return values[finite], int((~finite & ~np.isnan(values)).sum())


def _bounded_systematic_sample(
    current: np.ndarray, values: np.ndarray, limit: int
) -> np.ndarray:
    if values.size == 0:
        return current
    take = min(values.size, max(512, limit // 10))
    if take < values.size:
        indices = np.linspace(0, values.size - 1, take, dtype=np.int64)
        incoming = values[indices]
    else:
        incoming = values
    merged = np.concatenate((current, incoming))
    if merged.size <= limit:
        return merged
    indices = np.linspace(0, merged.size - 1, limit, dtype=np.int64)
    return merged[indices]


def analyse_column(
    path: Path, database: str, column: str, batch_size: int
) -> ColumnStats:
    parquet = pq.ParquetFile(path)
    dtype = field_type(path, column)
    stats = ColumnStats(database=database, row_count=parquet.metadata.num_rows)
    if dtype is None:
        stats.null_count = stats.row_count
        return stats
    stats.arrow_type = str(dtype)
    is_text = (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_dictionary(dtype)
    )
    is_boolean = pa.types.is_boolean(dtype)
    for batch in parquet.iter_batches(columns=[column], batch_size=batch_size):
        array = batch.column(0)
        stats.null_count += array.null_count
        if is_text:
            values = [str(value) for value in array.to_pylist() if value is not None]
            stats.non_null += len(values)
            stats.value_counts.update(values)
            if len(stats.value_counts) > MAX_TRACKED_LEVELS:
                stats.level_tracking_complete = False
            continue
        if is_boolean:
            values = np.asarray(
                array.fill_null(False).to_numpy(zero_copy_only=False), dtype=bool
            )
            valid = np.asarray(array.is_valid().to_numpy(zero_copy_only=False))
            numeric = values[valid].astype(float)
            stats.non_null += int(numeric.size)
            stats.value_counts.update(numeric.astype(int).tolist())
            stats.sample = _bounded_systematic_sample(
                stats.sample, numeric, SAMPLE_LIMIT_PER_DATABASE
            )
            continue
        numeric, non_finite = _finite_numeric_values(array)
        stats.non_null += int(numeric.size)
        stats.non_finite += non_finite
        if numeric.size == 0:
            continue
        batch_min = float(numeric.min())
        batch_max = float(numeric.max())
        stats.minimum = (
            batch_min if stats.minimum is None else min(stats.minimum, batch_min)
        )
        stats.maximum = (
            batch_max if stats.maximum is None else max(stats.maximum, batch_max)
        )
        stats.sample = _bounded_systematic_sample(
            stats.sample, numeric, SAMPLE_LIMIT_PER_DATABASE
        )
        if stats.level_tracking_complete and stats.integer_like:
            rounded = np.rint(numeric)
            if not np.all(np.abs(numeric - rounded) <= 1e-7):
                stats.integer_like = False
                stats.level_tracking_complete = False
                stats.value_counts.clear()
            else:
                unique, counts = np.unique(rounded.astype(np.int64), return_counts=True)
                stats.value_counts.update(
                    {int(value): int(count) for value, count in zip(unique, counts)}
                )
                if len(stats.value_counts) > MAX_TRACKED_LEVELS:
                    stats.level_tracking_complete = False
                    stats.value_counts.clear()
    # Arrow nulls do not include NaN; make row accounting explicit.
    accounted = stats.non_null + stats.null_count + stats.non_finite
    if accounted < stats.row_count:
        stats.null_count += stats.row_count - accounted
    return stats


def classify_variable(stats_by_db: dict[str, ColumnStats]) -> str:
    nonempty = [stats for stats in stats_by_db.values() if stats.non_null > 0]
    if not nonempty:
        return "unavailable"
    text_types = {
        "string",
        "large_string",
        "dictionary",
    }
    if any(any(token in stats.arrow_type for token in text_types) for stats in nonempty):
        return "categorical"
    all_complete = all(stats.level_tracking_complete for stats in nonempty)
    all_integer = all(stats.integer_like for stats in nonempty)
    levels: set[Any] = set()
    if all_complete:
        for stats in nonempty:
            levels.update(stats.value_counts)
    if all_complete and all_integer and levels and levels.issubset({0, 1}):
        return "binary"
    if (
        all_complete
        and all_integer
        and 1 <= len(levels) <= MAX_DISCRETE_LEVELS
    ):
        return "ordinal"
    return "continuous"


def cohort_denominators(root: Path, batch_size: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for database in DATABASES:
        path = root / database / "demographics.parquet"
        parquet = pq.ParquetFile(path)
        id_column = next(
            (name for name in parquet.schema_arrow.names if name in ID_COLUMNS), None
        )
        if id_column is None:
            raise ValueError(f"No stay identifier in demographics for {database}")
        identifiers: set[Any] = set()
        for batch in parquet.iter_batches(columns=[id_column], batch_size=batch_size):
            identifiers.update(value for value in batch.column(0).to_pylist() if value is not None)
        result[database] = len(identifiers)
    return result


def positive_stay_counts(
    path: Path, column: str, batch_size: int
) -> tuple[int, int]:
    parquet = pq.ParquetFile(path)
    id_column = next(
        (name for name in parquet.schema_arrow.names if name in ID_COLUMNS), None
    )
    if id_column is None or column not in parquet.schema_arrow.names:
        return 0, 0
    positive: set[Any] = set()
    observed: set[Any] = set()
    dtype = field_type(path, column)
    for batch in parquet.iter_batches(
        columns=[id_column, column], batch_size=batch_size
    ):
        ids = batch.column(0).to_pylist()
        values = batch.column(1)
        if pa.types.is_boolean(dtype):
            raw = values.to_pylist()
            for identifier, value in zip(ids, raw):
                if identifier is None or value is None:
                    continue
                observed.add(identifier)
                if value:
                    positive.add(identifier)
            continue
        numeric = np.asarray(values.to_numpy(zero_copy_only=False), dtype=float)
        finite = np.isfinite(numeric)
        for identifier, value, valid in zip(ids, numeric, finite):
            if identifier is None or not valid:
                continue
            observed.add(identifier)
            if value > 0:
                positive.add(identifier)
    return len(positive), len(observed)


def _catalog_clamped_range(
    samples: list[np.ndarray],
    catalog_min: float | None,
    catalog_max: float | None,
) -> tuple[float, float]:
    pooled_parts = []
    for sample in samples:
        if sample.size > SAMPLE_LIMIT_PER_DATABASE:
            indices = np.linspace(
                0, sample.size - 1, SAMPLE_LIMIT_PER_DATABASE, dtype=np.int64
            )
            pooled_parts.append(sample[indices])
        elif sample.size:
            pooled_parts.append(sample)
    if not pooled_parts:
        return 0.0, 1.0
    pooled = np.concatenate(pooled_parts)
    if pooled.size < 100:
        lower, upper = float(pooled.min()), float(pooled.max())
    else:
        lower, upper = np.quantile(pooled, [DISPLAY_Q_LOW, DISPLAY_Q_HIGH]).tolist()
    if catalog_min is not None:
        lower = max(lower, catalog_min)
    if catalog_max is not None:
        upper = min(upper, catalog_max)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        lower, upper = float(pooled.min()), float(pooled.max())
    if upper <= lower:
        pad = max(abs(lower) * 0.05, 0.5)
        lower, upper = lower - pad, upper + pad
    return lower, upper


def continuous_payload(
    root: Path,
    module: str,
    variable: str,
    stats_by_db: dict[str, ColumnStats],
    description: str,
    unit: str | None,
    catalog_min: float | None,
    catalog_max: float | None,
    bins: int,
    batch_size: int,
) -> tuple[PlotPayload, dict[str, dict[str, int | float]]]:
    lower, upper = _catalog_clamped_range(
        [stats.sample for stats in stats_by_db.values()],
        catalog_min,
        catalog_max,
    )
    edges = np.linspace(lower, upper, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    frames = []
    tails: dict[str, dict[str, int | float]] = {}
    for database in DATABASES:
        path = root / database / f"{module}.parquet"
        stats = stats_by_db[database]
        counts = np.zeros(bins, dtype=np.int64)
        below = 0
        above = 0
        if stats.non_null > 0 and variable in pq.ParquetFile(path).schema_arrow.names:
            for batch in pq.ParquetFile(path).iter_batches(
                columns=[variable], batch_size=batch_size
            ):
                values, _ = _finite_numeric_values(batch.column(0))
                if values.size == 0:
                    continue
                below += int((values < lower).sum())
                above += int((values > upper).sum())
                counts += np.histogram(values, bins=edges)[0]
        denominator = max(stats.non_null, 1)
        density = counts.astype(float) / (denominator * width)
        smoothed = gaussian_filter1d(density, sigma=1.25, mode="nearest")
        displayed = int(counts.sum())
        coverage = displayed / denominator if stats.non_null else 0.0
        tails[database] = {
            "display_lower": lower,
            "display_upper": upper,
            "n_below": below,
            "n_above": above,
            "n_displayed": displayed,
            "display_coverage": coverage,
        }
        frames.append(
            pd.DataFrame(
                {
                    "database": database,
                    "database_label": DATABASE_LABELS[database],
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "bin_center": centres,
                    "count": counts,
                    "density": density,
                    "density_smoothed": smoothed,
                    "total_finite": stats.non_null,
                    "n_below": below,
                    "n_above": above,
                    "display_coverage": coverage,
                }
            )
        )
    payload = PlotPayload(
        module=module,
        variable=variable,
        description=description,
        unit=unit,
        kind="continuous",
        data=pd.concat(frames, ignore_index=True),
        subtitle="Record-level smoothed density across harmonized databases",
        footnote="Shared display range uses per-database deterministic q0.5–q99.5 samples; exact tails are retained in source data.",
    )
    return payload, tails


def binary_payload(
    root: Path,
    module: str,
    variable: str,
    description: str,
    unit: str | None,
    denominators: dict[str, int],
    batch_size: int,
) -> PlotPayload:
    rows = []
    for database in DATABASES:
        path = root / database / f"{module}.parquet"
        positive, observed = positive_stay_counts(path, variable, batch_size)
        denominator = denominators[database]
        rows.append(
            {
                "database": database,
                "database_label": DATABASE_LABELS[database],
                "positive_stays": positive,
                "observed_stays": observed,
                "cohort_stays": denominator,
                "prevalence": (
                    positive / denominator
                    if denominator and observed > 0
                    else np.nan
                ),
            }
        )
    return PlotPayload(
        module=module,
        variable=variable,
        description=description,
        unit=unit,
        kind="binary",
        data=pd.DataFrame(rows),
        subtitle="Cohort stays with at least one positive record",
        footnote="Denominator is the database-level EasyICU cohort from demographics; missing is not treated as negative at record level.",
    )


def ordinal_payload(
    module: str,
    variable: str,
    stats_by_db: dict[str, ColumnStats],
    description: str,
    unit: str | None,
) -> PlotPayload:
    rows = []
    for database, stats in stats_by_db.items():
        denominator = max(sum(stats.value_counts.values()), 1)
        for value, count in sorted(stats.value_counts.items(), key=lambda item: item[0]):
            rows.append(
                {
                    "database": database,
                    "database_label": DATABASE_LABELS[database],
                    "value": value,
                    "count": count,
                    "total_non_null": stats.non_null,
                    "proportion": count / denominator,
                }
            )
    return PlotPayload(
        module=module,
        variable=variable,
        description=description,
        unit=unit,
        kind="ordinal",
        data=pd.DataFrame(rows),
        subtitle="Record-level probability mass",
        footnote="Repeated within-stay measurements contribute repeated records; curves describe extracted records, not independent patients.",
    )


def _clean_category(value: Any, variable: str) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if variable == "sex":
        normalized = text.casefold()
        if normalized in {"m", "male", "man"}:
            return "Male"
        if normalized in {"f", "female", "woman"}:
            return "Female"
        if normalized in {"u", "unk", "unknown", "other"}:
            return "Unknown / other"
    return text if text else "(blank)"


def categorical_payload(
    module: str,
    variable: str,
    stats_by_db: dict[str, ColumnStats],
    description: str,
    unit: str | None,
) -> PlotPayload:
    global_counts: Counter[str] = Counter()
    per_database: dict[str, Counter[str]] = {}
    for database, stats in stats_by_db.items():
        cleaned: Counter[str] = Counter()
        for value, count in stats.value_counts.items():
            cleaned[_clean_category(value, variable)] += count
        per_database[database] = cleaned
        global_counts.update(cleaned)
    categories = [
        category
        for category, _ in global_counts.most_common(MAX_CATEGORIES_SHOWN)
    ]
    has_other = len(global_counts) > len(categories)
    rows = []
    for database in DATABASES:
        counts = per_database[database]
        denominator = max(sum(counts.values()), 1)
        for category in categories:
            count = counts.get(category, 0)
            rows.append(
                {
                    "database": database,
                    "database_label": DATABASE_LABELS[database],
                    "category": category,
                    "count": count,
                    "total_non_null": sum(counts.values()),
                    "proportion": count / denominator,
                }
            )
        if has_other:
            other = sum(
                count for category, count in counts.items() if category not in categories
            )
            rows.append(
                {
                    "database": database,
                    "database_label": DATABASE_LABELS[database],
                    "category": "Other",
                    "count": other,
                    "total_non_null": sum(counts.values()),
                    "proportion": other / denominator,
                }
            )
    return PlotPayload(
        module=module,
        variable=variable,
        description=description,
        unit=unit,
        kind="categorical",
        data=pd.DataFrame(rows),
        subtitle="Record-level category composition",
        footnote=(
            "Top categories are shown by pooled record count; remaining levels are grouped as Other."
            if has_other
            else "All observed categories are shown."
        ),
    )


def unavailable_payload(
    module: str, variable: str, description: str, unit: str | None
) -> PlotPayload:
    return PlotPayload(
        module=module,
        variable=variable,
        description=description,
        unit=unit,
        kind="unavailable",
        data=pd.DataFrame(columns=["database", "status"]),
        subtitle="No non-missing values in the six audited exports",
        footnote="The Parquet schema is retained even when a concept is structurally unavailable.",
    )


def wrap_identifier(value: str, width: int = 20) -> str:
    parts = value.split("_")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if current and len(candidate) > width:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def axis_label(payload: PlotPayload, *, compact: bool) -> str:
    label = wrap_identifier(payload.variable, 18) if compact else payload.variable
    unit = display_unit(payload.unit)
    if unit:
        label += f"\n({unit})" if compact else f" ({unit})"
    return label


def value_axis_label(payload: PlotPayload) -> str:
    """Short value-axis label; the concept identifier remains in the title."""
    unit = display_unit(payload.unit)
    return f"Value ({unit})" if unit else "Value"


def panel_label(index: int) -> str:
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("a") + remainder) + label
        if value == 0:
            return label
        value -= 1


def draw_payload(
    ax: mpl.axes.Axes,
    payload: PlotPayload,
    *,
    compact: bool,
    show_legend: bool,
) -> None:
    data = payload.data
    if payload.kind == "continuous":
        for database in DATABASES:
            subset = data[data["database"] == database]
            if subset.empty or subset["total_finite"].iloc[0] == 0:
                continue
            x = subset["bin_center"].to_numpy()
            y = subset["density_smoothed"].to_numpy()
            label = f"{DATABASE_LABELS[database]} (n={int(subset['total_finite'].iloc[0]):,})"
            ax.plot(x, y, color=DATABASE_COLORS[database], lw=1.35, label=label)
        # Dense multi-panel pages have limited horizontal room.  Cap the
        # number of major ticks so decimal labels (for example 0.00--1.25)
        # cannot collide in narrow panels.
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, min_n_ticks=3))
        ax.set_xlabel(value_axis_label(payload))
        ax.set_ylabel("Density")
    elif payload.kind == "binary":
        positions = np.arange(len(DATABASES))[::-1]
        values = []
        for database in DATABASES:
            subset = data[data["database"] == database]
            values.append(float(subset["prevalence"].iloc[0]) * 100 if not subset.empty else np.nan)
        finite_values = [value for value in values if math.isfinite(value)]
        upper = max(finite_values, default=0.0)
        label_pad = max(upper * 0.035, 0.8)
        for position, database, value in zip(positions, DATABASES, values):
            if math.isfinite(value):
                ax.hlines(
                    position,
                    0,
                    value,
                    color=DATABASE_COLORS[database],
                    linewidth=1.1,
                    alpha=0.65,
                )
                ax.scatter(
                    value,
                    position,
                    s=22,
                    marker=DATABASE_MARKERS[database],
                    color=DATABASE_COLORS[database],
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
                ax.text(
                    value + label_pad,
                    position,
                    f"{value:.1f}",
                    ha="left",
                    va="center",
                    fontsize=6.0 if compact else 6.5,
                )
            else:
                ax.text(
                    label_pad * 0.45,
                    position,
                    "N/A",
                    ha="left",
                    va="center",
                    fontsize=6.0 if compact else 6.5,
                    color="#767676",
                )
        ax.set_yticks(
            positions,
            ["AUMC", "eICU", "HiRID", "M-III", "M-IV", "SIC"],
        )
        ax.set_xlabel("Cohort stays positive (%)")
        ax.set_xlim(0, max(upper + 4 * label_pad, 1.0))
    elif payload.kind == "ordinal":
        for database in DATABASES:
            subset = data[data["database"] == database].sort_values("value")
            if subset.empty:
                continue
            ax.plot(
                subset["value"],
                subset["proportion"] * 100,
                color=DATABASE_COLORS[database],
                marker=DATABASE_MARKERS[database],
                ms=2.5 if compact else 3.5,
                lw=1.2,
                label=DATABASE_LABELS[database],
            )
        ax.set_xlabel(value_axis_label(payload))
        ax.set_ylabel("Record proportion (%)")
        ax.set_ylim(bottom=0)
    elif payload.kind == "categorical":
        categories = list(dict.fromkeys(data["category"].tolist()))
        matrix = np.zeros((len(categories), len(DATABASES)), dtype=float)
        for index, database in enumerate(DATABASES):
            subset = data[data["database"] == database].set_index("category")
            if subset.empty or float(subset["total_non_null"].max()) <= 0:
                matrix[:, index] = np.nan
                continue
            matrix[:, index] = [
                float(subset.loc[category, "proportion"]) * 100
                if category in subset.index
                else 0.0
                for category in categories
            ]
        color_map = mpl.colormaps["Blues"].copy()
        color_map.set_bad("#E6E6E6")
        image = ax.imshow(
            matrix,
            cmap=color_map,
            vmin=0,
            vmax=100,
            aspect="auto",
            interpolation="nearest",
        )
        image.set_rasterized(True)
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        colorbar.set_label("Record (%)", fontsize=5.5)
        colorbar.ax.tick_params(labelsize=5.5, length=2)
        for database_index in range(len(DATABASES)):
            if np.isnan(matrix[:, database_index]).all():
                ax.text(
                    database_index,
                    (len(categories) - 1) / 2,
                    "N/A",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=5.5,
                    color="#666666",
                )
        ax.set_xticks(
            np.arange(len(DATABASES)),
            ["AUMC", "eICU", "HiRID", "M-III", "M-IV", "SIC"],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(
            np.arange(len(categories)),
            [
                textwrap.fill(str(category), width=18, break_long_words=False)
                for category in categories
            ],
        )
        ax.set_xlabel("Database")
        ax.set_ylabel("Category")
        ax.grid(False)
    else:
        ax.text(
            0.5,
            0.55,
            "No observed values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#767676",
        )
        ax.text(
            0.5,
            0.43,
            "Schema retained / structurally unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="#767676",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.55)
    title = wrap_identifier(payload.variable, 24)
    unit = display_unit(payload.unit)
    if unit:
        title = f"{title} [{unit}]"
    ax.set_title(
        title,
        loc="left",
        fontsize=7.5 if compact else 9,
        fontweight="semibold",
    )
    if show_legend and payload.kind in {"continuous", "ordinal", "categorical"}:
        ax.legend(loc="best", ncol=2, fontsize=6.5)


def save_module_atlas(
    module: str,
    payloads: list[PlotPayload],
    output_base: Path,
    dpi: int,
    panels_per_page: int = PANELS_PER_PAGE,
) -> int:
    """Save a fixed-width, paginated module atlas and return page count."""
    panel_count = len(payloads)
    if panels_per_page < 1 or panels_per_page > 12:
        raise ValueError("--panels-per-page must be between 1 and 12")
    page_count = max(1, math.ceil(panel_count / panels_per_page))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DATABASE_COLORS[database],
            marker=DATABASE_MARKERS[database],
            lw=1.5,
            ms=4,
            label=DATABASE_LABELS[database],
        )
        for database in DATABASES
    ]
    pdf_path = output_base.with_suffix(".pdf")
    with PdfPages(pdf_path) as pdf:
        for page_index in range(page_count):
            page_payloads = payloads[
                page_index * panels_per_page : (page_index + 1) * panels_per_page
            ]
            page_panel_count = len(page_payloads)
            columns = 1 if page_panel_count == 1 else (2 if page_panel_count <= 4 else 3)
            rows = math.ceil(page_panel_count / columns)
            figure_width = DOUBLE_COLUMN_WIDTH_IN
            figure_height = min(9.55, max(4.2, 0.82 + rows * 2.15))
            fig, axes = plt.subplots(
                rows,
                columns,
                figsize=(figure_width, figure_height),
                squeeze=False,
            )
            first_panel = page_index * panels_per_page
            for local_index, (ax, payload) in enumerate(
                zip(axes.flat, page_payloads)
            ):
                draw_payload(ax, payload, compact=True, show_legend=False)
                ax.text(
                    -0.14,
                    1.02,
                    panel_label(first_panel + local_index),
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )
            for ax in axes.flat[page_panel_count:]:
                ax.axis("off")
            page_suffix = (
                f" — page {page_index + 1}/{page_count}"
                if page_count > 1
                else ""
            )
            fig.suptitle(
                f"EasyICU — {module} distributions{page_suffix}",
                x=0.018,
                y=0.992,
                ha="left",
                fontsize=10,
                fontweight="bold",
            )
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=6,
                bbox_to_anchor=(0.5, 0.012),
                fontsize=6.5,
            )
            fig.text(
                0.015,
                0.068,
                (
                    "Density: finite records within shared q0.5–q99.5 display range; "
                    "binary: stays with ≥1 positive record; categorical: record %."
                ),
                ha="left",
                va="bottom",
                fontsize=5.5,
                color="#606060",
            )
            fig.tight_layout(
                rect=(0.012, 0.13, 0.995, 0.94),
                h_pad=1.15,
                w_pad=0.9,
            )
            pdf.savefig(fig)
            page_base = (
                output_base
                if page_count == 1
                else output_base.with_name(
                    f"{output_base.name}_p{page_index + 1:02d}"
                )
            )
            fig.savefig(page_base.with_suffix(".svg"))
            fig.savefig(page_base.with_suffix(".png"), dpi=dpi)
            fig.savefig(
                page_base.with_suffix(".tiff"),
                dpi=600,
                pil_kwargs={"compression": "tiff_lzw"},
            )
            plt.close(fig)
    return page_count


def write_source_data(payload: PlotPayload, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload.data.to_csv(output_path, index=False)


def payload_notes(kind: str) -> tuple[str, str]:
    if kind == "continuous":
        return (
            "Record-level smoothed density across harmonized databases",
            "Shared display range uses per-database deterministic q0.5–q99.5 samples; exact tails are retained in source data.",
        )
    if kind == "binary":
        return (
            "Cohort stays with at least one positive record",
            "Denominator is the database-level EasyICU cohort from demographics; missing is not treated as negative at record level.",
        )
    if kind == "ordinal":
        return (
            "Record-level probability mass",
            "Repeated within-stay measurements contribute repeated records; curves describe extracted records, not independent patients.",
        )
    if kind == "categorical":
        return (
            "Record-level category composition",
            "Category counts are exact among non-missing extracted values.",
        )
    return (
        "No non-missing values in the six audited exports",
        "The Parquet schema is retained even when a concept is structurally unavailable.",
    )


def source_run_lineage(run_metadata_path: Path) -> dict[str, str]:
    """Bind a QC artifact to the exact source run metadata bytes."""

    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"Missing source run metadata: {run_metadata_path}")
    raw_metadata = run_metadata_path.read_bytes()
    metadata = json.loads(raw_metadata)
    run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(
            f"Source run metadata has no non-empty run_id: {run_metadata_path}"
        )
    return {
        "source_run_id": run_id.strip(),
        "source_run_metadata_sha256": hashlib.sha256(raw_metadata).hexdigest(),
    }


def refresh_audit_catalog_metadata(
    audit: pd.DataFrame,
    catalog: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Refresh catalog-only columns without touching record-level audit values."""

    required = {
        "variable",
        "description",
        "unit",
        "catalog_min",
        "catalog_max",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise ValueError(
            "Render audit is missing catalog metadata columns: "
            + ", ".join(missing)
        )
    refreshed = audit.copy()
    for column in ("description", "unit"):
        refreshed[column] = refreshed[column].astype("object")
    for column in ("catalog_min", "catalog_max"):
        refreshed[column] = pd.to_numeric(refreshed[column], errors="coerce")
    for variable in refreshed["variable"].dropna().astype(str).unique():
        description, unit, lower, upper = concept_metadata(catalog, variable)
        selector = refreshed["variable"].astype(str) == variable
        refreshed.loc[selector, "description"] = description
        refreshed.loc[selector, "unit"] = unit
        refreshed.loc[selector, "catalog_min"] = lower
        refreshed.loc[selector, "catalog_max"] = upper
    return refreshed


def render_from_source(
    output_root: Path,
    modules: list[str] | None,
    dpi: int,
    panels_per_page: int,
    *,
    catalog: dict[str, dict[str, Any]] | None,
    catalog_sha256: str | None,
    lineage: dict[str, str],
) -> int:
    audit_path = output_root / "audit" / "variable_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing render audit: {audit_path}")
    audit = pd.read_csv(audit_path)
    if catalog is not None:
        audit = refresh_audit_catalog_metadata(audit, catalog)
        audit.to_csv(audit_path, index=False)
    available_modules = list(dict.fromkeys(audit["module"].astype(str)))
    selected_modules = modules or available_modules
    unknown = sorted(set(selected_modules) - set(available_modules))
    if unknown:
        raise ValueError(f"Unknown render-only modules: {', '.join(unknown)}")
    figures_root = output_root / "figures"
    source_root = output_root / "source_data"
    for module_index, module in enumerate(selected_modules, start=1):
        module_audit = audit[audit["module"] == module]
        variables = list(dict.fromkeys(module_audit["variable"].astype(str)))
        payloads: list[PlotPayload] = []
        for variable in variables:
            metadata = module_audit[module_audit["variable"] == variable].iloc[0]
            source_path = source_root / module / f"{variable}.csv"
            data = pd.read_csv(source_path)
            kind = str(metadata["plot_kind"])
            subtitle, footnote = payload_notes(kind)
            unit_value = metadata.get("unit")
            unit = None if pd.isna(unit_value) else str(unit_value)
            payloads.append(
                PlotPayload(
                    module=module,
                    variable=variable,
                    description=str(metadata["description"]),
                    unit=unit,
                    kind=kind,
                    data=data,
                    subtitle=subtitle,
                    footnote=footnote,
                )
            )
        save_module_atlas(
            module,
            payloads,
            figures_root / module,
            dpi,
            panels_per_page,
        )
        print(
            f"[{module_index}/{len(selected_modules)}] rendered {module}: "
            f"{len(payloads)} panels",
            flush=True,
        )
    manifest_path = output_root / "audit" / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest.update(lineage)
    manifest["last_rendered_at_utc"] = datetime.now(UTC).isoformat()
    manifest["render_layout"] = "adaptive_module_atlas"
    rendered_modules = set(manifest.get("modules") or [])
    rendered_modules.update(manifest.get("rendered_modules") or [])
    rendered_modules.update(selected_modules)
    manifest["rendered_modules"] = sorted(rendered_modules)
    manifest["render_dpi"] = dpi
    manifest["panels_per_page"] = panels_per_page
    if catalog_sha256 is not None:
        manifest["catalog_sha256"] = catalog_sha256
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> int:
    args = parse_args()
    apply_publication_style()
    started = time.perf_counter()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    run_metadata_path = (
        args.run_metadata.resolve()
        if args.run_metadata is not None
        else input_root.parent / "run_metadata.json"
    )
    lineage = source_run_lineage(run_metadata_path)
    catalog_path = args.catalog.resolve() if args.catalog is not None else None
    catalog = load_catalog(catalog_path) if catalog_path is not None else None
    if args.render_only:
        return render_from_source(
            output_root,
            args.modules,
            args.dpi,
            args.panels_per_page,
            catalog=catalog,
            catalog_sha256=(
                file_sha256(catalog_path) if catalog_path is not None else None
            ),
            lineage=lineage,
        )
    if catalog_path is None or catalog is None:
        raise ValueError("--catalog is required unless --render-only is used")
    available_modules = module_names(input_root)
    modules = args.modules or available_modules
    unknown = sorted(set(modules) - set(available_modules))
    if unknown:
        raise ValueError(f"Unknown modules: {', '.join(unknown)}")

    figures_root = output_root / "figures"
    source_root = output_root / "source_data"
    audit_root = output_root / "audit"
    figures_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    print("Computing six-database cohort denominators...", flush=True)
    denominators = cohort_denominators(input_root, args.batch_size)
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    variable_panel_count = 0
    module_figure_count = 0
    kind_counts: Counter[str] = Counter()

    for module_index, module in enumerate(modules, start=1):
        variables = module_variables(input_root, module)
        print(
            f"[{module_index}/{len(modules)}] {module}: {len(variables)} variables",
            flush=True,
        )
        payloads: list[PlotPayload] = []
        for variable_index, variable in enumerate(variables, start=1):
            print(
                f"  [{variable_index}/{len(variables)}] {variable}",
                flush=True,
            )
            description, unit, catalog_min, catalog_max = concept_metadata(
                catalog, variable
            )
            stats_by_db: dict[str, ColumnStats] = {}
            try:
                for database in DATABASES:
                    path = input_root / database / f"{module}.parquet"
                    stats_by_db[database] = analyse_column(
                        path, database, variable, args.batch_size
                    )
                kind = classify_variable(stats_by_db)
                tail_stats: dict[str, dict[str, int | float]] = {}
                if kind == "continuous":
                    payload, tail_stats = continuous_payload(
                        input_root,
                        module,
                        variable,
                        stats_by_db,
                        description,
                        unit,
                        catalog_min,
                        catalog_max,
                        args.bins,
                        args.batch_size,
                    )
                elif kind == "binary":
                    payload = binary_payload(
                        input_root,
                        module,
                        variable,
                        description,
                        unit,
                        denominators,
                        args.batch_size,
                    )
                elif kind == "ordinal":
                    payload = ordinal_payload(
                        module, variable, stats_by_db, description, unit
                    )
                elif kind == "categorical":
                    payload = categorical_payload(
                        module, variable, stats_by_db, description, unit
                    )
                else:
                    payload = unavailable_payload(
                        module, variable, description, unit
                    )

                write_source_data(
                    payload, source_root / module / f"{variable}.csv"
                )
                payloads.append(payload)
                variable_panel_count += 1
                kind_counts[kind] += 1

                for database, stats in stats_by_db.items():
                    sample_q = (
                        np.quantile(stats.sample, [DISPLAY_Q_LOW, 0.5, DISPLAY_Q_HIGH])
                        if stats.sample.size
                        else [np.nan, np.nan, np.nan]
                    )
                    tails = tail_stats.get(database, {})
                    audit_rows.append(
                        {
                            "module": module,
                            "variable": variable,
                            "description": description,
                            "unit": unit,
                            "catalog_min": catalog_min,
                            "catalog_max": catalog_max,
                            "plot_kind": kind,
                            "database": database,
                            "parquet": f"exports/{database}/{module}.parquet",
                            "arrow_type": stats.arrow_type,
                            "row_count": stats.row_count,
                            "non_null_or_finite": stats.non_null,
                            "null_count": stats.null_count,
                            "non_finite_count": stats.non_finite,
                            "minimum": stats.minimum,
                            "q0_5_sample": sample_q[0],
                            "median_sample": sample_q[1],
                            "q99_5_sample": sample_q[2],
                            "maximum": stats.maximum,
                            "display_lower": tails.get("display_lower"),
                            "display_upper": tails.get("display_upper"),
                            "n_below_display": tails.get("n_below"),
                            "n_above_display": tails.get("n_above"),
                            "n_displayed": tails.get("n_displayed"),
                            "display_coverage": tails.get("display_coverage"),
                            "cohort_stays": denominators[database],
                        }
                    )
            except Exception as exc:
                failures.append(
                    {
                        "module": module,
                        "variable": variable,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"    ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if payloads:
            save_module_atlas(
                module,
                payloads,
                figures_root / module,
                args.dpi,
                args.panels_per_page,
            )
            module_figure_count += 1

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_root / "variable_audit.csv", index=False)
    pd.DataFrame(failures).to_csv(audit_root / "failures.csv", index=False)
    denominator_frame = pd.DataFrame(
        [
            {
                "database": database,
                "database_label": DATABASE_LABELS[database],
                "cohort_stays": denominators[database],
            }
            for database in DATABASES
        ]
    )
    denominator_frame.to_csv(audit_root / "cohort_denominators.csv", index=False)

    elapsed = time.perf_counter() - started
    manifest = {
        "status": "passed" if not failures else "partial",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": "python",
        "databases": list(DATABASES),
        "modules": list(modules),
        "module_count": len(modules),
        "module_figure_count": module_figure_count,
        "rendered_modules": list(modules),
        "variable_panel_count": variable_panel_count,
        "plot_kind_counts": dict(sorted(kind_counts.items())),
        "failure_count": len(failures),
        "panels_per_page": args.panels_per_page,
        "figure_width_mm": 183,
        "elapsed_seconds": round(elapsed, 3),
        "display_quantiles": [DISPLAY_Q_LOW, DISPLAY_Q_HIGH],
        "continuous_density_rule": (
            "All finite values inside the shared display range contribute to "
            "histogram density; Gaussian sigma=1.25 bins is applied only to "
            "the rendered density line. Tail counts remain in source data."
        ),
        "binary_rule": (
            "Unique cohort stays with at least one positive record divided by "
            "unique stays in demographics."
        ),
        "ordinal_and_categorical_rule": (
            "Exact record counts among non-missing extracted values."
        ),
        "excluded_columns": sorted(INDEX_COLUMNS),
        "catalog_sha256": file_sha256(catalog_path),
        **lineage,
    }
    (audit_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run())
