"""Render a planned continuous-distribution and source-availability audit.

This adapter is intentionally rendering-only.  It accepts one exact controlled
parent method and the two CSV files named by that parent's structured summary.
It never scans sibling files, reads an outcome column, chooses an exposure,
defines bins, or changes the cohort.
"""

from __future__ import annotations

import io
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Optional

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

REPAIR_ID = "distribution_availability_publication_bundle_from_parent_outputs_v1"
CONTROLLED_METHOD = "exposure_distribution_and_missingness_audit"
_DISTRIBUTION_COLUMNS = {
    "row_type",
    "variable",
    "n",
    "denominator_n",
    "percentage",
    "fraction",
}
_DISTRIBUTION_OPTIONAL_COLUMNS = {
    "category",
    "analysis_set",
    "time_window",
    "unit",
}
_MEASUREMENT_COLUMNS = {
    "row_type",
    "variable",
    "category",
    "source_status",
    "n",
    "denominator_n",
    "percentage",
    "fraction",
}
_METRICS = ("median", "q25", "q75", "min", "max")


def _distribution_availability_parent_digest_seal(
    run_dir: Path,
    figure_step_id: str,
) -> Optional[dict[str, str]]:
    """Seal only the three parent files selected by the closed renderer contract.

    A parent may legitimately publish other tables or parquet products.  They
    are neither renderer inputs nor CSVs, so including them in the child seal
    would make an unrelated artifact capable of disabling this renderer.
    """

    request_step = _resolve_upstream_manifest_step(run_dir, figure_step_id)
    if not isinstance(request_step, Mapping):
        return None
    if str(request_step.get("method") or "").strip().lower() != CONTROLLED_METHOD:
        return None

    digests = _verified_direct_parent_artifact_digests(run_dir, figure_step_id)
    if not digests or "step_summary.json" not in digests:
        return None
    parent_step_id = str(figure_step_id or "").removesuffix("_figure")
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    try:
        snapshot = read_digest_bound_artifact_snapshot(
            parent_out=parent_out,
            artifact_digests=digests,
        )
        parent_summary = json.loads(snapshot["step_summary.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parent_summary, Mapping):
        return None
    distribution_contract = parent_summary.get("distribution")
    measurement_contract = parent_summary.get("measurement_audit")
    if not isinstance(distribution_contract, Mapping) or not isinstance(
        measurement_contract, Mapping
    ):
        return None
    selected_names = {
        str(distribution_contract.get("table") or "").strip(),
        str(measurement_contract.get("table") or "").strip(),
    }
    if (
        len(selected_names) != 2
        or any(
            Path(name).name != name or not name.endswith(".csv")
            for name in selected_names
        )
        or not selected_names <= set(snapshot)
    ):
        return None
    selected_table_bytes = {name: snapshot[name] for name in selected_names}
    prepared = prepare_distribution_availability_inputs(
        parent_out=parent_out,
        parent_summary=parent_summary,
        verified_table_names=set(selected_table_bytes),
        preverified_table_bytes=selected_table_bytes,
    )
    if prepared is None:
        return None
    declared_tables, distribution_roles, availability_roles = (
        distribution_availability_planner_table_roles(
            request_step.get("expected_outputs") or []
        )
    )
    if len(distribution_roles) != 1 or len(availability_roles) != 1:
        return None
    required_table_products = {
        prepared.distribution_path.stem,
        prepared.measurement_path.stem,
    }
    if (
        required_table_products != distribution_roles | availability_roles
        or not required_table_products <= declared_tables
    ):
        return None
    required_names = {
        "step_summary.json",
        prepared.distribution_path.name,
        prepared.measurement_path.name,
    }
    if not required_names <= set(digests):
        return None
    return {name: digests[name] for name in sorted(required_names)}


def _normalise(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _safe_csv_name(value: Any) -> Optional[str]:
    name = str(value or "").strip()
    path = Path(name)
    if not name or path.name != name or path.suffix.lower() != ".csv":
        return None
    return name


def _unique_metric_name(
    names: Collection[Any], base: str, *, unit: str
) -> Optional[str]:
    """Return one metric whose name is authorized by the exposure unit.

    A free suffix is not a unit contract: for example, ``median_response``
    must not be accepted merely because it begins with ``median_``.  The only
    supported spellings are the bare metric or the metric followed by the
    normalized authoritative exposure unit.
    """

    allowed = {base}
    normalized_unit = _normalise(unit)
    if normalized_unit:
        allowed.add(f"{base}_{normalized_unit}")
    matches = [str(name) for name in names if _normalise(name) in allowed]
    if len({_normalise(name) for name in matches}) != len(matches):
        return None
    if normalized_unit:
        qualified = [
            name for name in matches if _normalise(name) == f"{base}_{normalized_unit}"
        ]
        if len(qualified) == 1:
            return qualified[0]
    bare = [name for name in matches if _normalise(name) == base]
    return bare[0] if len(bare) == 1 else None


def _metric_alias_names(
    names: Collection[Any], base: str, *, unit: str
) -> tuple[str, ...]:
    """Return every closed-schema alias for one metric."""

    allowed = {base}
    normalized_unit = _normalise(unit)
    if normalized_unit:
        allowed.add(f"{base}_{normalized_unit}")
    aliases = tuple(str(name) for name in names if _normalise(name) in allowed)
    if len({_normalise(name) for name in aliases}) != len(aliases):
        return ()
    return aliases


def _finite_number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_integer(value: Any) -> Optional[int]:
    parsed = _finite_number(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _same(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=tolerance)


def _same_alias(left: float, right: float) -> bool:
    """Compare duplicate representations without scale-relative slack."""

    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _normalised_optional(value: Any) -> tuple[bool, str]:
    """Distinguish an absent cell from a present but invalid token."""

    if value is None or pd.isna(value) or not str(value).strip():
        return False, ""
    return True, _normalise(value)


def _wrapped_axis_label(value: Any, *, width: int = 22) -> str:
    """Wrap display-only category text without changing the source contract."""

    label = str(value).strip()
    lines = textwrap.wrap(
        label,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else label


def _read_selected_columns(
    source: Path | bytes,
    allowed: set[str],
    *,
    metrics: bool,
    unit: str = "",
    optional: Collection[str] = (),
) -> Optional[pd.DataFrame]:
    def _reader() -> Path | io.BytesIO:
        return io.BytesIO(source) if isinstance(source, bytes) else source

    try:
        header = pd.read_csv(_reader(), nrows=0)
    except Exception:
        return None
    selected = [
        str(column)
        for column in header.columns
        if str(column) in allowed or str(column) in optional
    ]
    if metrics:
        for base in _METRICS:
            aliases = _metric_alias_names(header.columns, base, unit=unit)
            if not aliases:
                return None
            selected.extend(aliases)
    if not allowed.issubset(set(selected)):
        return None
    try:
        return pd.read_csv(_reader(), usecols=list(dict.fromkeys(selected)))
    except Exception:
        return None


@dataclass(frozen=True)
class DistributionAvailabilityInputs:
    distribution_path: Path
    measurement_path: Path
    exposure_column: str
    exposure_label: str
    unit: str
    metric_columns: Mapping[str, str]
    metric_values: Mapping[str, float]
    distribution_row: pd.DataFrame
    status_rows: pd.DataFrame
    observed_n: int
    denominator_n: int
    status_schema: tuple[str, ...]


def distribution_availability_planner_table_roles(
    expected_outputs: Collection[Any],
) -> tuple[set[str], set[str], set[str]]:
    """Return declared tables and the exact renderer-role candidates."""

    declared_tables = {
        parsed[1]
        for raw in expected_outputs
        if (parsed := typed_product(raw)) is not None and parsed[0] == "table"
    }
    declared_tokens = {
        product: tuple(token for token in product.split("_") if token)
        for product in declared_tables
    }

    def _matches(*suffixes: tuple[str, ...]) -> set[str]:
        return {
            product
            for product, tokens in declared_tokens.items()
            if any(
                len(tokens) >= len(suffix) and tokens[-len(suffix) :] == suffix
                for suffix in suffixes
            )
        }

    return (
        declared_tables,
        _matches(("distribution",)),
        _matches(
            ("measurement", "audit"),
            ("availability",),
            ("measurement", "coverage"),
            ("source", "coverage"),
            ("missingness",),
        ),
    )


def distribution_availability_parent_contract_issue(
    *,
    planned_method: Any,
    parent_out: Optional[Path],
    parent_summary: Mapping[str, Any],
    expected_outputs: Collection[Any],
    planned_inputs: Collection[Any] = (),
    host_context: Any = None,
) -> Optional[dict[str, Any]]:
    """Return one host-owned issue for an unrenderable controlled parent.

    This is the pre-registration counterpart of the digest seal below.  It
    validates only the two table roles already selected by the Planner and the
    agent-authored summary; it never scans sibling products or chooses a
    variable, cohort, denominator, category schema, or statistic.
    """

    if str(planned_method or "").strip().lower() != CONTROLLED_METHOD:
        return None
    # This is an output-family contract, not a free method-name hook.  Only a
    # Planner step with both renderer roles enters it. Auxiliary tables remain
    # outside the renderer permission set. ``missingness`` is the generic
    # legacy terminal for a source-availability table.
    declared_tables, distribution_roles, availability_roles = (
        distribution_availability_planner_table_roles(expected_outputs)
    )
    if not distribution_roles or not availability_roles:
        return None
    # Some compatibility/unit-test callers do not have execution outputs.  The
    # real early and final gates always pass an output directory; do not turn a
    # missing optional test argument into a new global failure mode.
    if parent_out is None:
        return None

    distribution = parent_summary.get("distribution")
    measurement = parent_summary.get("measurement_audit")
    exposure = parent_summary.get("primary_exposure")
    selected_names: set[str] = set()
    contract_issue: Optional[str] = None
    expected_exposure: Optional[str] = None
    expected_unit: Optional[str] = None
    reported_exposure = (
        str(exposure.get("column") or "").strip()
        if isinstance(exposure, Mapping)
        else ""
    )
    reported_unit = (
        str(exposure.get("unit") or "").strip() if isinstance(exposure, Mapping) else ""
    )
    if len(distribution_roles) != 1 or len(availability_roles) != 1:
        contract_issue = "planner_table_roles_ambiguous"
    elif host_context is not None:
        expected_exposure = str(
            getattr(host_context, "primary_exposure", None) or ""
        ).strip()
        if not expected_exposure:
            contract_issue = "host_primary_exposure_unavailable"
        elif expected_exposure not in {str(value) for value in planned_inputs}:
            contract_issue = "host_primary_exposure_not_planner_input"
        elif reported_exposure != expected_exposure:
            contract_issue = "summary_primary_exposure_mismatch"
        else:
            variable_getter = getattr(host_context, "variable", None)
            descriptor = (
                variable_getter(expected_exposure)
                if callable(variable_getter)
                else None
            )
            expected_unit = str(getattr(descriptor, "unit", None) or "").strip()
            if expected_unit and _normalise(reported_unit) != _normalise(expected_unit):
                contract_issue = "summary_primary_exposure_unit_mismatch"
    if contract_issue is None and (
        not isinstance(distribution, Mapping) or not isinstance(measurement, Mapping)
    ):
        contract_issue = "summary_table_roles_missing"
    elif contract_issue is None:
        distribution_name = _safe_csv_name(distribution.get("table"))
        measurement_name = _safe_csv_name(measurement.get("table"))
        if (
            distribution_name is None
            or measurement_name is None
            or distribution_name == measurement_name
        ):
            contract_issue = "summary_table_roles_invalid"
        else:
            selected_names = {distribution_name, measurement_name}
            required_products = {Path(name).stem for name in selected_names}
            role_products = distribution_roles | availability_roles
            if required_products != role_products:
                contract_issue = "summary_tables_not_exact_planner_roles"
            else:
                prepared = prepare_distribution_availability_inputs(
                    parent_out=Path(parent_out),
                    parent_summary=parent_summary,
                    verified_table_names=selected_names,
                )
                if prepared is not None:
                    return None
                contract_issue = "closed_schema_rejected"
    if contract_issue is None:
        raise AssertionError("invalid parent contract did not record a reason")
    return {
        "kind": "controlled_renderer_parent_contract_invalid",
        "reason": "distribution_availability_parent_contract_invalid",
        "contract_issue": contract_issue,
        "controlled_method": CONTROLLED_METHOD,
        "declared_table_products": sorted(declared_tables),
        "distribution_role_products": sorted(distribution_roles),
        "availability_role_products": sorted(availability_roles),
        "selected_table_files": sorted(selected_names),
        "expected_primary_exposure": expected_exposure,
        "reported_primary_exposure": reported_exposure or None,
        "planner_inputs": [str(value) for value in planned_inputs],
        "expected_primary_exposure_unit": expected_unit,
        "reported_primary_exposure_unit": reported_unit or None,
        "required_summary_slots": [
            "primary_exposure",
            "distribution",
            "measurement_audit",
        ],
        "required_table_roles": [
            "distribution.table",
            "measurement_audit.table",
        ],
        "required_schema": {
            "primary_exposure": [
                "column",
                "authoritative=true",
                "role=authoritative_primary_exposure",
            ],
            "distribution_summary": [
                "table",
                "observed_n",
                *list(_METRICS),
            ],
            "distribution_table_columns": sorted(_DISTRIBUTION_COLUMNS),
            "measurement_summary": [
                "table",
                "source_status_schema",
                "source_status_counts",
                "status_assignment_n",
            ],
            "measurement_table_columns": sorted(_MEASUREMENT_COLUMNS),
            "distribution_invariants": [
                "exactly one selected row for primary_exposure.column",
                "row_type identifies the exposure distribution",
                "category or analysis_set selects valid_observed",
                "n <= denominator_n",
                "percentage == 100 * n / denominator_n",
                "fraction == n / denominator_n",
                "min <= q25 <= median <= q75 <= max",
                "five-number values exactly match the distribution summary",
            ],
            "measurement_invariants": [
                "source_status rows exactly match source_status_schema",
                "category == source_status for every status row",
                "source_status_counts exactly match table counts",
                "all status rows share the distribution denominator_n",
                "sum(source_status_counts) == denominator_n",
                "status_assignment_n == denominator_n",
                "valid_observed count == distribution.observed_n",
            ],
        },
    }


def prepare_distribution_availability_inputs(
    *,
    parent_out: Path,
    parent_summary: Mapping[str, Any],
    verified_table_names: Collection[str],
    preverified_table_bytes: Optional[Mapping[str, bytes]] = None,
) -> Optional[DistributionAvailabilityInputs]:
    """Validate the closed parent contract and return only plottable fields."""

    if _normalise(parent_summary.get("method")) != CONTROLLED_METHOD:
        return None
    exposure = parent_summary.get("primary_exposure")
    distribution = parent_summary.get("distribution")
    measurement = parent_summary.get("measurement_audit")
    if not all(
        isinstance(item, Mapping) for item in (exposure, distribution, measurement)
    ):
        return None
    exposure = dict(exposure)
    distribution = dict(distribution)
    measurement = dict(measurement)
    exposure_column = str(exposure.get("column") or "").strip()
    unit = str(exposure.get("unit") or "").strip()
    if (
        not exposure_column
        or exposure.get("authoritative") is not True
        or _normalise(exposure.get("role")) != "authoritative_primary_exposure"
    ):
        return None
    distribution_name = _safe_csv_name(distribution.get("table"))
    measurement_name = _safe_csv_name(measurement.get("table"))
    if (
        distribution_name is None
        or measurement_name is None
        or distribution_name == measurement_name
        or not {distribution_name, measurement_name} <= set(verified_table_names)
    ):
        return None
    distribution_path = parent_out / distribution_name
    measurement_path = parent_out / measurement_name
    if preverified_table_bytes is None:
        if any(
            path.is_symlink() or not path.is_file()
            for path in (distribution_path, measurement_path)
        ):
            return None
        distribution_source: Path | bytes = distribution_path
        measurement_source: Path | bytes = measurement_path
    else:
        if set(preverified_table_bytes) != {distribution_name, measurement_name}:
            return None
        distribution_source = preverified_table_bytes[distribution_name]
        measurement_source = preverified_table_bytes[measurement_name]

    distribution_frame = _read_selected_columns(
        distribution_source,
        _DISTRIBUTION_COLUMNS,
        metrics=True,
        unit=unit,
        optional=_DISTRIBUTION_OPTIONAL_COLUMNS,
    )
    measurement_frame = _read_selected_columns(
        measurement_source, _MEASUREMENT_COLUMNS, metrics=False
    )
    if distribution_frame is None or measurement_frame is None:
        return None

    metric_columns: dict[str, str] = {}
    metric_alias_columns: dict[str, tuple[str, ...]] = {}
    summary_metric_aliases: dict[str, tuple[str, ...]] = {}
    summary_metric_names: dict[str, str] = {}
    for base in _METRICS:
        summary_name = _unique_metric_name(distribution.keys(), base, unit=unit)
        table_aliases = _metric_alias_names(distribution_frame.columns, base, unit=unit)
        declared_aliases = _metric_alias_names(distribution.keys(), base, unit=unit)
        if summary_name is None or not table_aliases or not declared_aliases:
            return None
        exact = [
            name
            for name in table_aliases
            if _normalise(name) == _normalise(summary_name)
        ]
        table_name = exact[0] if len(exact) == 1 else None
        if table_name is None and len(table_aliases) == 1:
            table_name = table_aliases[0]
        if table_name is None:
            return None
        metric_columns[base] = table_name
        metric_alias_columns[base] = table_aliases
        summary_metric_aliases[base] = declared_aliases
        summary_metric_names[base] = summary_name

    row_type = distribution_frame["row_type"].fillna("").map(_normalise)
    distribution_candidates = distribution_frame.loc[
        distribution_frame["variable"].fillna("").astype(str).eq(exposure_column)
        & row_type.str.endswith("_distribution")
    ].copy()
    if distribution_candidates.empty:
        return None
    normalized_exposure = _normalise(exposure_column)
    if not normalized_exposure:
        return None
    allowed_categories = {
        "valid_observed",
        f"valid_observed_{normalized_exposure}",
    }
    selectors: list[str] = []
    category_presence: list[bool] = []
    for _, row in distribution_candidates.iterrows():
        category_present, normalized_category = _normalised_optional(
            row.get("category")
        )
        analysis_set_present, normalized_analysis_set = _normalised_optional(
            row.get("analysis_set")
        )
        if (category_present and not normalized_category) or (
            analysis_set_present and not normalized_analysis_set
        ):
            return None
        if category_present:
            category_is_authoritative = normalized_category in allowed_categories
            analysis_set_is_authoritative = (
                normalized_analysis_set in allowed_categories
            )
            if analysis_set_present and (
                category_is_authoritative != analysis_set_is_authoritative
            ):
                return None
            selectors.append(normalized_category)
        else:
            selectors.append(normalized_analysis_set)
        category_presence.append(category_present)
    if not selectors or selectors[0] not in allowed_categories:
        return None
    selector_series = pd.Series(selectors, index=distribution_candidates.index)
    candidates = distribution_candidates.loc[
        selector_series.isin(allowed_categories)
    ].copy()
    if len(candidates) != 1:
        return None
    candidate = candidates.iloc[0]
    candidate_position = distribution_candidates.index.get_loc(candidate.name)
    category_present = category_presence[int(candidate_position)]
    if (
        not category_present
        and row_type.loc[candidate.name] != f"{normalized_exposure}_distribution"
    ):
        return None

    row_window_present, row_window = _normalised_optional(candidate.get("time_window"))
    row_unit_present, row_unit = _normalised_optional(candidate.get("unit"))
    declared_window_present, declared_window = _normalised_optional(
        exposure.get("time_window")
    )
    declared_unit_present, declared_unit = _normalised_optional(unit)
    if (
        (row_window_present and not row_window)
        or (row_unit_present and not row_unit)
        or (
            row_window_present
            and (not declared_window_present or row_window != declared_window)
        )
        or (
            row_unit_present
            and (not declared_unit_present or row_unit != declared_unit)
        )
    ):
        return None
    observed_n = _nonnegative_integer(distribution.get("observed_n"))
    if observed_n is None or observed_n <= 0:
        return None
    candidate_n = pd.to_numeric(candidates["n"], errors="coerce")
    candidates = candidates.loc[candidate_n.eq(observed_n)].copy()
    if candidates.empty:
        return None

    numeric_columns = [
        "n",
        "denominator_n",
        "percentage",
        "fraction",
        *(alias for aliases in metric_alias_columns.values() for alias in aliases),
    ]
    numeric = candidates[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or len(numeric) != 1:
        return None
    reference = numeric.iloc[0]
    source_index = int(candidates.index[0])
    selected_distribution = candidates.iloc[[0]].copy()
    selected_distribution.insert(0, "source_row_index", source_index)

    denominator_n = _nonnegative_integer(reference["denominator_n"])
    if denominator_n is None or denominator_n <= 0 or observed_n > denominator_n:
        return None
    expected_percentage = 100.0 * observed_n / denominator_n
    expected_fraction = observed_n / denominator_n
    if not _same(float(reference["percentage"]), expected_percentage, tolerance=1e-6):
        return None
    if not _same(float(reference["fraction"]), expected_fraction):
        return None

    metric_values: dict[str, float] = {}
    for base, column in metric_columns.items():
        value = _finite_number(reference[column])
        expected = _finite_number(distribution.get(summary_metric_names[base]))
        if value is None or expected is None or not _same_alias(value, expected):
            return None
        if any(
            (alias_value := _finite_number(reference[alias])) is None
            or not _same_alias(alias_value, value)
            for alias in metric_alias_columns[base]
        ):
            return None
        if any(
            (alias_value := _finite_number(distribution.get(alias))) is None
            or not _same_alias(alias_value, expected)
            for alias in summary_metric_aliases[base]
        ):
            return None
        if _normalise(column) != _normalise(summary_metric_names[base]):
            if (
                not row_unit_present
                or not declared_unit_present
                or row_unit != declared_unit
            ):
                return None
        metric_values[base] = value
    if not (
        metric_values["min"]
        <= metric_values["q25"]
        <= metric_values["median"]
        <= metric_values["q75"]
        <= metric_values["max"]
    ):
        return None

    schema_raw = measurement.get("source_status_schema")
    counts_raw = measurement.get("source_status_counts")
    if not isinstance(schema_raw, list) or not isinstance(counts_raw, Mapping):
        return None
    status_schema = tuple(str(value).strip() for value in schema_raw)
    if (
        not status_schema
        or any(not value for value in status_schema)
        or len(set(status_schema)) != len(status_schema)
    ):
        return None
    status_rows = measurement_frame.loc[
        measurement_frame["row_type"].fillna("").map(_normalise).eq("source_status")
    ].copy()
    if status_rows.empty or status_rows["source_status"].isna().any():
        return None
    status_variables = status_rows["variable"].fillna("").astype(str).str.strip()
    if status_variables.eq("").any() or status_variables.nunique() != 1:
        return None
    statuses = status_rows["source_status"].astype(str).str.strip()
    if statuses.duplicated().any() or set(statuses) != set(status_schema):
        return None
    categories = status_rows["category"].fillna("").astype(str).str.strip()
    if not categories.equals(statuses):
        return None
    declared_status_variable = str(
        measurement.get("source_status_variable") or ""
    ).strip()
    if (
        declared_status_variable
        and status_variables.iloc[0] != declared_status_variable
    ):
        return None
    status_rows = (
        status_rows.assign(
            __status_order=statuses.map(
                {value: index for index, value in enumerate(status_schema)}
            )
        )
        .sort_values("__status_order")
        .drop(columns="__status_order")
    )
    status_rows.insert(0, "source_row_index", status_rows.index.astype(int))
    status_n = pd.to_numeric(status_rows["n"], errors="coerce")
    status_denominator = pd.to_numeric(status_rows["denominator_n"], errors="coerce")
    if status_n.isna().any() or status_denominator.isna().any():
        return None
    parsed_counts = [_nonnegative_integer(value) for value in status_n]
    parsed_denominators = [_nonnegative_integer(value) for value in status_denominator]
    if any(value is None for value in (*parsed_counts, *parsed_denominators)):
        return None
    counts = [int(value) for value in parsed_counts if value is not None]
    denominators = [int(value) for value in parsed_denominators if value is not None]
    if set(denominators) != {denominator_n} or sum(counts) != denominator_n:
        return None
    declared_assignment_n = _nonnegative_integer(measurement.get("status_assignment_n"))
    if declared_assignment_n is None or declared_assignment_n != denominator_n:
        return None
    for status, count, percentage_raw, fraction_raw in zip(
        status_schema,
        counts,
        status_rows["percentage"],
        status_rows["fraction"],
    ):
        declared = _nonnegative_integer(counts_raw.get(status))
        if declared != count:
            return None
        percentage_missing = pd.isna(percentage_raw)
        fraction_missing = pd.isna(fraction_raw)
        if percentage_missing or fraction_missing:
            if count != 0 or not (percentage_missing and fraction_missing):
                return None
            continue
        percentage = _finite_number(percentage_raw)
        fraction = _finite_number(fraction_raw)
        if percentage is None or fraction is None:
            return None
        if not _same(float(percentage), 100.0 * count / denominator_n, tolerance=1e-6):
            return None
        if not _same(float(fraction), count / denominator_n):
            return None
    valid_observed_positions = [
        index
        for index, status in enumerate(status_schema)
        if _normalise(status) == "valid_observed"
    ]
    # The legacy summary does not declare the status-variable column name.
    # Bind the two tables through exact planned table names, the closed status
    # schema/count map, one common denominator, and this observed-count
    # identity; never infer a semantic link from variable-name substrings.
    if (
        len(valid_observed_positions) != 1
        or counts[valid_observed_positions[0]] != observed_n
    ):
        return None

    label = str(
        exposure.get("display_label") or exposure.get("label") or exposure_column
    )
    return DistributionAvailabilityInputs(
        distribution_path=distribution_path,
        measurement_path=measurement_path,
        exposure_column=exposure_column,
        exposure_label=label.replace("_", " ").strip(),
        unit=unit,
        metric_columns=metric_columns,
        metric_values=metric_values,
        distribution_row=selected_distribution,
        status_rows=status_rows,
        observed_n=observed_n,
        denominator_n=denominator_n,
        status_schema=status_schema,
    )


def render_distribution_availability_bundle_from_prior_outputs(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    preverified_parent_digests: Optional[Mapping[str, str]] = None,
    preverified_parent_artifacts: Optional[Mapping[str, bytes]] = None,
) -> Optional[str]:
    """Render the exact direct parent's verified descriptive audit product."""

    parent_step_id = str(current_step_id).removesuffix("_figure")
    if not parent_step_id or parent_step_id == str(current_step_id):
        return None
    parent_out = Path(run_dir) / "steps" / parent_step_id / "outputs"
    preverified_table_bytes: Optional[dict[str, bytes]] = None
    if preverified_parent_artifacts is not None:
        names = {str(name) for name in preverified_parent_artifacts}
        csv_names = {name for name in names if name != "step_summary.json"}
        if (
            "step_summary.json" not in names
            or len(names) != 3
            or len(csv_names) != 2
            or any(_safe_csv_name(name) is None for name in csv_names)
        ):
            return None
        snapshot = dict(preverified_parent_artifacts)
        try:
            summary = json.loads(snapshot["step_summary.json"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        preverified_table_bytes = {
            name: payload
            for name, payload in snapshot.items()
            if name != "step_summary.json"
        }
        verified = set(preverified_table_bytes)
    elif preverified_parent_digests is None:
        host_seal = _distribution_availability_parent_digest_seal(
            Path(run_dir), current_step_id
        )
        if host_seal is None:
            return None
        verified = {name for name in host_seal if name != "step_summary.json"}
        try:
            summary = json.loads((parent_out / "step_summary.json").read_text("utf-8"))
        except Exception:
            return None
    else:
        names = {str(name) for name in preverified_parent_digests}
        csv_names = {name for name in names if name != "step_summary.json"}
        if (
            "step_summary.json" not in names
            or len(names) != 3
            or len(csv_names) != 2
            or any(_safe_csv_name(name) is None for name in csv_names)
        ):
            return None
        try:
            snapshot = read_digest_bound_artifact_snapshot(
                parent_out=parent_out,
                artifact_digests=preverified_parent_digests,
            )
        except ValueError:
            return None
        try:
            summary = json.loads(snapshot["step_summary.json"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        preverified_table_bytes = {
            name: payload
            for name, payload in snapshot.items()
            if name != "step_summary.json"
        }
        verified = set(preverified_table_bytes)
    if not verified:
        return None
    if not isinstance(summary, Mapping):
        return None
    prepared = prepare_distribution_availability_inputs(
        parent_out=parent_out,
        parent_summary=summary,
        verified_table_names=verified,
        preverified_table_bytes=preverified_table_bytes,
    )
    if prepared is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_source = out_dir / "distribution_panel_source_data.csv"
    availability_source = out_dir / "availability_panel_source_data.csv"
    distribution_export = prepared.distribution_row.copy()
    distribution_export.insert(0, "panel_id", "A")
    distribution_export.insert(1, "source_table", prepared.distribution_path.name)
    availability_export = prepared.status_rows.copy()
    # The status-table ``variable`` is constant across all rows and therefore
    # cannot be a row key.  Keep the unique predeclared category plus the exact
    # source position so trace validation cannot create a many-to-many join.
    availability_export = availability_export.drop(columns="variable")
    availability_export.insert(0, "panel_id", "B")
    availability_export.insert(1, "source_table", prepared.measurement_path.name)
    distribution_export.to_csv(distribution_source, index=False)
    availability_export.to_csv(availability_source, index=False)

    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(183 / 25.4, 88 / 25.4), gridspec_kw={"width_ratios": [1.0, 1.35]}
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
    ax_a.set_title("Observed exposure distribution", loc="left", pad=4)
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

    counts = pd.to_numeric(prepared.status_rows["n"]).astype(int)
    percentages = counts.astype(float) * 100.0 / prepared.denominator_n
    positions = range(len(prepared.status_schema))
    bars = ax_b.barh(positions, percentages, color=palette["blue_soft"], height=0.58)
    ax_b.set_yticks(list(positions))
    ax_b.set_yticklabels(
        [_wrapped_axis_label(status) for status in prepared.status_schema]
    )
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Analysis cohort (%)")
    ax_b.set_title("Source availability", loc="left", pad=4)
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
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.86, wspace=0.42)

    stem = "distribution_availability"
    contract = make_figure_contract(
        figure_id=f"figure:{stem}",
        core_claim=(
            "The agent-planned exposure distribution and source availability "
            "are rendered from the verified direct-parent audit tables."
        ),
        archetype="quantitative_grid",
        width_mm=183.0,
        height_mm=88.0,
        panels=[
            {
                "panel_id": "A",
                "title": "Observed exposure distribution",
                "role": "descriptive_result",
                "claim": "Median, interquartile range, and range among observed records.",
                "evidence_ids": [distribution_source.name],
                "metadata": {"planner_product_slots": ["distribution"]},
            },
            {
                "panel_id": "B",
                "title": "Source availability",
                "role": "data_quality",
                "claim": "Predeclared source-status counts reconcile to the analysis cohort.",
                "evidence_ids": [availability_source.name],
                "metadata": {"planner_product_slots": ["availability"]},
            },
        ],
        source_data=[distribution_source.name, availability_source.name],
        statistics_note=(
            "No bins, cohort filters, exposure definitions, or inferential estimates "
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
        "method": "deterministic_distribution_availability_figure",
        "analysis_family": "descriptive",
        "rendering_only": True,
        "status": "completed",
        "source_step_id": parent_step_id,
        "source_tables": [
            prepared.distribution_path.name,
            prepared.measurement_path.name,
        ],
        "source_data_files": [distribution_source.name, availability_source.name],
        "figure_files": figure_files,
        "figure_path": f"{stem}.png",
        "figure_contract": f"{stem}.figure_contract.json",
        "output_files": {
            "figure:publication_figure": f"{stem}.png",
        },
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
    "distribution_availability_parent_contract_issue",
    "distribution_availability_planner_table_roles",
    "prepare_distribution_availability_inputs",
    "render_distribution_availability_bundle_from_prior_outputs",
]
