"""Deterministic publication figure for ordered-category QC distributions.

The renderer is deliberately case-neutral. Legacy v1 activates from one
controlled parent method; additive v2 activates from Planner-declared product
roles plus a digest-bound ordinal/count/availability schema. Neither path uses
a clinical variable name. Panel A shows the distribution conditional on a
valid observed category; panel B accounts for availability against the locked
cohort. This separation prevents a locked-cohort percentage from being paired
with the valid-observed denominator.
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .publication import (
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
ORDERED_DISTRIBUTION_REPAIR_V1 = "ordered_category_distribution_publication_bundle_v1"
ORDERED_DISTRIBUTION_AVAILABILITY_REPAIR_V2 = (
    "ordered_category_distribution_availability_publication_bundle_v2"
)


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
        else (
            flow_path
            if preverified_parent_artifacts is None and flow_path.exists()
            else None
        )
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


def _primary_ordinal_contract(
    parent_summary: Mapping[str, Any],
) -> Optional[Tuple[str, Tuple[int, ...], int]]:
    """Return the exact ordinal exposure contract recorded by the parent.

    This is a compatibility proof for sealed, already-audited parent outputs.
    It never infers a clinical variable, level set, or denominator from a name.
    """

    exposure = parent_summary.get("primary_exposure")
    locked = parent_summary.get("locked_cohort")
    if not isinstance(exposure, Mapping) or not isinstance(locked, Mapping):
        return None
    if _normalise(exposure.get("scale")) != "ordinal":
        return None
    variable = str(exposure.get("variable") or "").strip()
    raw_levels = exposure.get("declared_levels")
    if not variable or not isinstance(raw_levels, list) or len(raw_levels) < 2:
        return None
    numeric_levels = pd.to_numeric(pd.Series(raw_levels), errors="coerce")
    if (
        numeric_levels.isna().any()
        or not bool(numeric_levels.mod(1).eq(0).all())
        or numeric_levels.duplicated().any()
    ):
        return None
    locked_n = _declared_count(dict(locked), "n_rows")
    if locked_n is None or locked_n <= 0:
        return None
    return variable, tuple(int(value) for value in numeric_levels.tolist()), locked_n


def _candidate_separate_availability_table(
    *,
    parent_out: Path,
    exposure_variable: str,
    preverified_parent_artifacts: Mapping[str, bytes],
) -> Optional[Tuple[Path, pd.DataFrame, str, str]]:
    """Select one schema-bound availability table for the exact exposure.

    Selection uses a column that explicitly binds rows to the authoritative
    exposure variable.  Filenames, row labels, and clinical vocabulary never
    participate.  Multiple matching tables or groups fail closed.
    """

    candidates: list[Tuple[Path, pd.DataFrame, str, str]] = []
    for name, payload in sorted(preverified_parent_artifacts.items()):
        if Path(name).name != name or Path(name).suffix.lower() != ".csv":
            continue
        try:
            frame = pd.read_csv(io.BytesIO(payload))
        except Exception:
            continue
        status_col = _resolve_column(
            frame, ("status", "source_status", "source_state", "availability_status")
        )
        count_col = _resolve_column(frame, _COUNT_COLUMNS)
        binding_col = _resolve_column(
            frame, ("value_column", "source_variable", "exposure_source")
        )
        if status_col is None or count_col is None or binding_col is None:
            continue
        exact = (
            frame[binding_col].fillna("").astype(str).str.strip().eq(exposure_variable)
        )
        rows = frame.loc[exact].copy()
        if rows.empty:
            continue
        rows = rows.reset_index().rename(columns={"index": "__source_row_index"})
        candidates.append((parent_out / name, rows, status_col, count_col))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _validate_separate_availability_rows(
    *,
    rows: pd.DataFrame,
    status_col: str,
    count_col: str,
    observed_n: int,
    locked_n: int,
) -> Optional[pd.DataFrame]:
    counts = _nonnegative_integer(rows[count_col])
    if counts is None:
        return None
    rows = rows.copy()
    rows["__count"] = counts.to_numpy()
    rows["__status_role"] = rows[status_col].map(_normalise)
    if (
        rows["__status_role"].eq("").any()
        or rows["__status_role"].duplicated().any()
        or int(rows["__count"].sum()) != locked_n
    ):
        return None
    valid_mask = rows["__status_role"].isin(_VALID_OBSERVED_ROLES)
    if int(valid_mask.sum()) != 1:
        return None
    if int(rows.loc[valid_mask, "__count"].sum()) != observed_n:
        return None
    denominator_col = _resolve_column(rows, ("denominator",))
    if denominator_col is None:
        return None
    denominators = _nonnegative_integer(rows[denominator_col])
    if denominators is None or not bool(denominators.eq(locked_n).all()):
        return None
    expected_pct = 100.0 * rows["__count"].astype(float) / float(locked_n)
    if _resolve_column(rows, _LOCKED_PERCENT_COLUMNS) is None:
        return None
    if not _optional_values_match(
        frame=rows,
        column_names=_LOCKED_PERCENT_COLUMNS,
        expected=expected_pct,
        tolerance=0.05,
    ):
        return None
    if not _optional_values_match(
        frame=rows,
        column_names=_LOCKED_FRACTION_COLUMNS,
        expected=rows["__count"].astype(float) / float(locked_n),
        tolerance=0.0005,
    ):
        return None
    return rows


def _declared_optional_missingness_audits(
    *,
    parent_out: Path,
    parent_summary: Mapping[str, Any],
    preverified_parent_artifacts: Mapping[str, bytes],
    locked_n: int,
) -> Optional[
    Tuple[
        Tuple[Path, pd.DataFrame, bytes],
        Tuple[Path, pd.DataFrame, bytes],
    ]
]:
    """Return a schema-verified optional missingness/structural-audit pair.

    The Planner-owned product roles select the tables.  Both roles must be
    present together; filenames and clinical vocabulary never authorize a
    table.  ``None`` therefore means either "no optional pair declared" or an
    invalid/incomplete pair, and the caller distinguishes those states from
    the declared role set.
    """

    output_files = parent_summary.get("output_files")
    if not isinstance(output_files, Mapping):
        return None
    role_names: Dict[str, str] = {}
    for raw_product, raw_name in output_files.items():
        product = str(raw_product or "").strip()
        if ":" not in product:
            continue
        kind, role = product.split(":", 1)
        role = _normalise(role)
        if _normalise(kind) not in {"artifact", "dataset", "table"}:
            continue
        if role in {"missingness_audit", "structural_missingness_audit"}:
            name = str(raw_name or "").strip()
            if (
                not name
                or Path(name).name != name
                or Path(name).suffix.lower() != ".csv"
                or role in role_names
            ):
                return None
            role_names[role] = name
    if not role_names:
        return None
    if set(role_names) != {"missingness_audit", "structural_missingness_audit"}:
        return None

    loaded: Dict[str, Tuple[Path, pd.DataFrame, bytes]] = {}
    for role, name in role_names.items():
        payload = preverified_parent_artifacts.get(name)
        if not isinstance(payload, bytes):
            return None
        try:
            frame = pd.read_csv(io.BytesIO(payload))
        except Exception:
            return None
        if frame.empty:
            return None
        loaded[role] = (parent_out / name, frame, payload)

    missingness_path, missingness, missingness_payload = loaded["missingness_audit"]
    structural_path, structural, structural_payload = loaded[
        "structural_missingness_audit"
    ]
    missing_variable = _resolve_column(missingness, ("variable",))
    missing_total = _resolve_column(missingness, ("n_total",))
    missing_nonmissing = _resolve_column(missingness, ("n_nonmissing",))
    missing_count = _resolve_column(missingness, ("missing_n",))
    missing_pct = _resolve_column(missingness, ("missing_pct",))
    if None in {
        missing_variable,
        missing_total,
        missing_nonmissing,
        missing_count,
        missing_pct,
    }:
        return None
    missing_variables = missingness[missing_variable].fillna("").astype(str).str.strip()
    totals = _nonnegative_integer(missingness[missing_total])
    nonmissing = _nonnegative_integer(missingness[missing_nonmissing])
    missing = _nonnegative_integer(missingness[missing_count])
    percentages = pd.to_numeric(missingness[missing_pct], errors="coerce")
    if (
        missing_variables.eq("").any()
        or missing_variables.duplicated().any()
        or totals is None
        or nonmissing is None
        or missing is None
        or not bool(totals.eq(locked_n).all())
        or not bool(nonmissing.add(missing).eq(totals).all())
        or percentages.isna().any()
        or not bool(
            percentages.sub(100.0 * missing.astype(float) / totals.astype(float))
            .abs()
            .le(0.05)
            .all()
        )
    ):
        return None

    structural_variable = _resolve_column(structural, ("variable",))
    structural_total = _resolve_column(structural, ("n_total",))
    structural_missing = _resolve_column(structural, ("missing_n",))
    structural_nonmissing = _resolve_column(structural, ("nonmissing_n",))
    structural_unique = _resolve_column(structural, ("nonmissing_unique_n",))
    structural_pct = _resolve_column(structural, ("missing_pct",))
    structural_status = _resolve_column(structural, ("structural_status",))
    if None in {
        structural_variable,
        structural_total,
        structural_missing,
        structural_nonmissing,
        structural_unique,
        structural_pct,
        structural_status,
    }:
        return None
    structural_variables = (
        structural[structural_variable].fillna("").astype(str).str.strip()
    )
    structural_totals = _nonnegative_integer(structural[structural_total])
    structural_missing_n = _nonnegative_integer(structural[structural_missing])
    structural_nonmissing_n = _nonnegative_integer(structural[structural_nonmissing])
    structural_unique_n = _nonnegative_integer(structural[structural_unique])
    structural_percentages = pd.to_numeric(structural[structural_pct], errors="coerce")
    structural_statuses = (
        structural[structural_status].fillna("").astype(str).str.strip()
    )
    if (
        structural_variables.eq("").any()
        or structural_variables.duplicated().any()
        or not set(structural_variables).issubset(set(missing_variables))
        or structural_statuses.eq("").any()
        or structural_totals is None
        or structural_missing_n is None
        or structural_nonmissing_n is None
        or structural_unique_n is None
        or not bool(structural_totals.eq(locked_n).all())
        or not bool(
            structural_missing_n.add(structural_nonmissing_n)
            .eq(structural_totals)
            .all()
        )
        or not bool(structural_unique_n.le(structural_nonmissing_n).all())
        or structural_percentages.isna().any()
        or not bool(
            structural_percentages.sub(
                100.0
                * structural_missing_n.astype(float)
                / structural_totals.astype(float)
            )
            .abs()
            .le(0.05)
            .all()
        )
    ):
        return None
    return (
        (missingness_path, missingness, missingness_payload),
        (structural_path, structural, structural_payload),
    )


def _declares_optional_missingness_audits(summary: Mapping[str, Any]) -> bool:
    output_files = summary.get("output_files")
    if not isinstance(output_files, Mapping):
        return False
    roles = {
        _normalise(str(product).split(":", 1)[1])
        for product in output_files
        if ":" in str(product)
    }
    return bool(roles & {"missingness_audit", "structural_missingness_audit"})


def ordered_distribution_availability_snapshot_is_valid(
    preverified_parent_artifacts: Mapping[str, bytes],
) -> bool:
    """Validate the additive v2 parent contract without producing outputs."""

    try:
        summary = json.loads(
            preverified_parent_artifacts["step_summary.json"].decode("utf-8")
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(summary, Mapping):
        return False
    contract = _primary_ordinal_contract(summary)
    if contract is None:
        return False
    exposure_variable, declared_levels, locked_n = contract
    parent_out = Path(".")
    candidate = _candidate_table(parent_out, preverified_parent_artifacts)
    if candidate is None:
        return False
    _source_path, frame, level_col, count_col = candidate
    levels = pd.to_numeric(frame[level_col], errors="coerce")
    plot = frame.loc[levels.notna()].copy()
    plot["__level"] = levels[levels.notna()].astype(int).to_numpy()
    plot = plot.sort_values("__level").reset_index(drop=True)
    counts = _nonnegative_integer(plot[count_col])
    if counts is None or tuple(plot["__level"].tolist()) != declared_levels:
        return False
    observed_n = int(counts.sum())
    if observed_n <= 0 or observed_n > locked_n:
        return False
    variable_col = _resolve_column(plot, ("variable", "source_variable"))
    if variable_col is not None and not bool(
        plot[variable_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(exposure_variable)
        .all()
    ):
        return False
    denominator_col = _resolve_column(plot, ("denominator",))
    if denominator_col is None:
        return False
    denominators = _nonnegative_integer(plot[denominator_col])
    if denominators is None or not bool(denominators.eq(observed_n).all()):
        return False
    percentage_col = _resolve_column(
        plot, (*_CONDITIONAL_PERCENT_COLUMNS, "percentage")
    )
    if percentage_col is None:
        return False
    expected_pct = 100.0 * counts.astype(float) / float(observed_n)
    observed_pct = pd.to_numeric(plot[percentage_col], errors="coerce")
    if observed_pct.isna().any() or bool(
        (observed_pct.reset_index(drop=True) - expected_pct.reset_index(drop=True))
        .abs()
        .gt(0.05)
        .any()
    ):
        return False
    availability = _candidate_separate_availability_table(
        parent_out=parent_out,
        exposure_variable=exposure_variable,
        preverified_parent_artifacts=preverified_parent_artifacts,
    )
    if availability is None:
        return False
    _path, rows, status_col, availability_count_col = availability
    availability_rows = _validate_separate_availability_rows(
        rows=rows,
        status_col=status_col,
        count_col=availability_count_col,
        observed_n=observed_n,
        locked_n=locked_n,
    )
    if availability_rows is None:
        return False
    optional_audits = _declares_optional_missingness_audits(summary)
    verified_audits = _declared_optional_missingness_audits(
        parent_out=parent_out,
        parent_summary=summary,
        preverified_parent_artifacts=preverified_parent_artifacts,
        locked_n=locked_n,
    )
    return verified_audits is not None if optional_audits else True


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
    authorized_repair_id: str = ORDERED_DISTRIBUTION_REPAIR_V1,
) -> Optional[str]:
    """Render a digest-bound ordered-distribution and data-quality bundle."""

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
    v2_separate_availability = (
        authorized_repair_id == ORDERED_DISTRIBUTION_AVAILABILITY_REPAIR_V2
    )
    if v2_separate_availability:
        if preverified_parent_artifacts is None or not (
            ordered_distribution_availability_snapshot_is_valid(
                preverified_parent_artifacts
            )
        ):
            return None
    elif authorized_repair_id != ORDERED_DISTRIBUTION_REPAIR_V1:
        return None
    else:
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
    conditional_col = _resolve_column(
        frame,
        (
            *_CONDITIONAL_PERCENT_COLUMNS,
            *(("percentage",) if v2_separate_availability else ()),
        ),
    )
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

    availability_source_path = source_path
    if v2_separate_availability:
        contract = _primary_ordinal_contract(parent_summary)
        if contract is None or preverified_parent_artifacts is None:
            return None
        exposure_variable, declared_levels, locked_n = contract
        if tuple(plot["__level"].tolist()) != declared_levels:
            return None
        separate = _candidate_separate_availability_table(
            parent_out=parent_out,
            exposure_variable=exposure_variable,
            preverified_parent_artifacts=preverified_parent_artifacts,
        )
        if separate is None:
            return None
        (
            availability_source_path,
            raw_status_rows,
            status_col,
            availability_count_col,
        ) = separate
        status_rows = _validate_separate_availability_rows(
            rows=raw_status_rows,
            status_col=status_col,
            count_col=availability_count_col,
            observed_n=observed_n,
            locked_n=locked_n,
        )
        if status_rows is None:
            return None
    else:
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
        status_col = _resolve_column(frame, _STATUS_COLUMNS)
        if status_col is None:
            return None
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
    availability_labels = [
        _status_display_label(role, value)
        for role, value in zip(availability_roles, status_rows[status_col].tolist())
    ]
    optional_audits = None
    if v2_separate_availability and preverified_parent_artifacts is not None:
        optional_audits = _declared_optional_missingness_audits(
            parent_out=parent_out,
            parent_summary=parent_summary,
            preverified_parent_artifacts=preverified_parent_artifacts,
            locked_n=locked_n,
        )
        if (
            _declares_optional_missingness_audits(parent_summary)
            and optional_audits is None
        ):
            return None

    label_col = _resolve_column(
        plot, ("stage_label", "level_label", "label", "category_label")
    )
    if label_col is not None:
        labels = [str(value) for value in plot[label_col].tolist()]
    else:
        labels = [f"Level {int(value)}" for value in plot["__level"].tolist()]
    primary_exposure = parent_summary.get("primary_exposure")
    exposure_value = (
        (
            primary_exposure.get("variable")
            if isinstance(primary_exposure, Mapping)
            else primary_exposure
        )
        or parent_summary.get("exposure")
        or parent_summary.get("primary_exposure_source")
        or "ordered exposure"
    )
    exposure_label = _exposure_display_label(exposure_value, labels)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    source_copy = out_dir / f"{stem}_source_data.csv"
    source_data_paths = [source_copy]
    source_table_names = [source_path.name, availability_source_path.name]
    missingness_plot: Optional[pd.DataFrame] = None
    structural_variables: set[str] = set()
    if optional_audits is not None:
        (missingness_path, missingness, _missingness_payload), (
            structural_path,
            structural,
            _structural_payload,
        ) = optional_audits
        missingness_copy = out_dir / f"{missingness_path.stem}_source_data.csv"
        structural_copy = out_dir / f"{structural_path.stem}_source_data.csv"
        source_data_paths.extend((missingness_copy, structural_copy))
        source_table_names.extend((missingness_path.name, structural_path.name))
        variable_col = _resolve_column(missingness, ("variable",))
        missing_pct_col = _resolve_column(missingness, ("missing_pct",))
        structural_variable_col = _resolve_column(structural, ("variable",))
        if (
            variable_col is None
            or missing_pct_col is None
            or structural_variable_col is None
        ):
            return None
        missingness_source = missingness[
            [
                variable_col,
                _resolve_column(missingness, ("n_total",)),
                _resolve_column(missingness, ("n_nonmissing",)),
                _resolve_column(missingness, ("missing_n",)),
                missing_pct_col,
            ]
        ].copy()
        structural_source = structural[
            [
                structural_variable_col,
                _resolve_column(structural, ("n_total",)),
                _resolve_column(structural, ("missing_n",)),
                _resolve_column(structural, ("missing_pct",)),
                _resolve_column(structural, ("nonmissing_n",)),
                _resolve_column(structural, ("nonmissing_unique_n",)),
                _resolve_column(structural, ("structural_status",)),
            ]
        ].copy()
        for source_frame, source_name in (
            (missingness_source, missingness_path.name),
            (structural_source, structural_path.name),
        ):
            source_frame["source_table"] = source_name
            source_frame["source_step_id"] = parent_step_id
        missingness_source.to_csv(missingness_copy, index=False)
        structural_source.to_csv(structural_copy, index=False)
        missingness_plot = missingness[[variable_col, missing_pct_col]].copy()
        missingness_plot.columns = ["variable", "missing_pct"]
        missingness_plot["variable"] = (
            missingness_plot["variable"].fillna("").astype(str).str.strip()
        )
        missingness_plot["missing_pct"] = pd.to_numeric(
            missingness_plot["missing_pct"], errors="coerce"
        )
        missingness_plot = (
            missingness_plot.sort_values(
                ["missing_pct", "variable"], ascending=[False, True]
            )
            .head(12)
            .reset_index(drop=True)
        )
        structural_variables = set(
            structural[structural_variable_col].fillna("").astype(str).str.strip()
        )
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
                "source_table": availability_source_path.name,
                "source_row_index": int(row_index),
                "source_percentage_column": "derived_from_count_over_locked_cohort",
                "source_transform": "count_over_locked_analysis_cohort",
            }
        )
    pd.DataFrame(source_rows).to_csv(source_copy, index=False)

    import matplotlib.pyplot as plt

    palette = apply_publication_style(font_size=7.0)
    if missingness_plot is None:
        figure_height_mm = 90.0
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(183 / 25.4, figure_height_mm / 25.4),
            gridspec_kw={"width_ratios": [1.35, 0.85]},
        )
        ax_a, ax_b = axes
        ax_c = None
    else:
        figure_height_mm = 100.0
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(183 / 25.4, figure_height_mm / 25.4),
            gridspec_kw={"width_ratios": [1.1, 0.8, 1.25]},
        )
        ax_a, ax_b, ax_c = axes
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
        (
            palette["blue"]
            if role in _VALID_OBSERVED_ROLES
            else palette["orange"] if count > 0 else palette["neutral_light"]
        )
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
    panels = [
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
    ]
    if ax_c is not None and missingness_plot is not None:
        panel_c_colors = [
            (
                palette["teal"]
                if variable in structural_variables
                else palette["neutral_light"]
            )
            for variable in missingness_plot["variable"]
        ]
        bars_c = ax_c.barh(
            range(len(missingness_plot)),
            missingness_plot["missing_pct"],
            color=panel_c_colors,
            height=0.58,
        )
        ax_c.set_yticks(range(len(missingness_plot)))
        ax_c.set_yticklabels(
            [_display_label(value) for value in missingness_plot["variable"]],
            fontsize=5.8,
        )
        ax_c.invert_yaxis()
        ax_c.set_xlim(0, 100)
        ax_c.set_xlabel("Locked analysis cohort missing (%)")
        ax_c.set_title("Variable missingness", loc="left", pad=4)
        ax_c.grid(axis="x", color=palette["neutral_light"], linewidth=0.55, zorder=0)
        for bar, percentage in zip(bars_c, missingness_plot["missing_pct"]):
            ax_c.text(
                min(float(percentage) + 1.0, 97.0),
                bar.get_y() + bar.get_height() / 2,
                f"{float(percentage):.1f}%",
                va="center",
                ha="right" if percentage > 92 else "left",
                fontsize=5.8,
                color=palette["baseline"],
            )
        add_panel_label(ax_c, "C", x=-0.18, y=1.03)
        panels.append(
            {
                "panel_id": "C",
                "title": "Variable missingness",
                "role": "missingness_audit",
                "claim": (
                    "The highest missingness percentages are reproduced from "
                    "the declared audit, with structural-audit coverage shown "
                    "only where that companion table contains the variable."
                ),
                "evidence_ids": [
                    missingness_copy.name,
                    structural_copy.name,
                ],
                "metadata": {"planner_product_slots": ["availability"]},
            }
        )
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.90, wspace=0.58)
    else:
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
        height_mm=figure_height_mm,
        panels=panels,
        source_data=[path.name for path in source_data_paths],
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
        "source_tables": source_table_names,
        "source_data_csv": str(source_copy),
        "figure_files": [
            path.name for key, path in outputs.items() if key != "contract"
        ],
        "source_data_files": [path.name for path in source_data_paths],
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
            **(
                {"panel_c": "locked_analysis_cohort"}
                if missingness_plot is not None
                else {}
            ),
        },
        "warnings": [],
        "skipped": [],
        "errors": [],
    }
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return authorized_repair_id


__all__ = [
    "ORDERED_DISTRIBUTION_AVAILABILITY_REPAIR_V2",
    "ORDERED_DISTRIBUTION_REPAIR_V1",
    "ordered_distribution_availability_snapshot_is_valid",
    "render_ordered_distribution_bundle_from_prior_outputs",
]
