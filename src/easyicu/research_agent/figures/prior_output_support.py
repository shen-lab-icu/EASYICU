"""Shared support for deterministic figures rebuilt from prior step outputs.

The parent-selection rule is an evidence boundary: a split figure may read only
its direct parent, while a legacy terminal overview may inspect prior steps.
Label helpers preserve the historical renderer vocabulary without putting this
policy back into the pipeline entry module. Context-table readers and
sensitivity labels also live here so renderers do not import host publication
and readiness workflows through the reporting compatibility facade.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def figure_parent_candidate_step_dirs(
    *,
    steps_dir: Path,
    current_step_id: str,
) -> tuple[list[Path], bool]:
    """Return direct parent only for split figures, else legacy prior steps."""

    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    direct_parent = Path(steps_dir) / parent_step_id
    is_split = parent_step_id != str(current_step_id or "")
    if is_split and direct_parent.is_dir():
        return [direct_parent], True
    return (
        [
            step_dir
            for step_dir in sorted(Path(steps_dir).iterdir())
            if step_dir.is_dir() and step_dir.name != current_step_id
        ],
        False,
    )


def publication_label(value: Any) -> str:
    """Return the stable reader-facing fallback used by legacy renderers."""

    token = str(value or "").strip()
    mapping = {
        "sepsis3": "Sepsis-3",
        "sep3_sofa2_max": "Experimental SOFA-2 Sepsis-3 phenotype",
        "age": "Age",
        "age_filled": "Age",
        "age_per_10y": "Age, per 10 years",
        "sex_m": "Male sex",
        "sex_male": "Male sex",
        "male": "Male sex",
        "hr_first": "Heart rate",
        "hr_first_filled": "Heart rate",
        "hr_max_per_10bpm": "Maximum heart rate, per 10 bpm",
        "map_first": "Mean arterial pressure",
        "map_first_filled": "Mean arterial pressure",
        "map_min": "Minimum mean arterial pressure",
        "resp_max_per_5": "Maximum respiratory rate, per 5/min",
        "temp_max_c": "Maximum temperature, per 1 deg C",
        "lactate": "Lactate",
        "lact": "Lactate",
        "lact_max_mmol_l": "Maximum lactate, per 1 mmol/L",
        "lact_measured": "Lactate measured",
        "bun_max_per_10": "Maximum BUN, per 10 units",
        "wbc_max_per_10": "Maximum WBC, per 10 units",
        "sofa2": "SOFA-2",
        "death": "In-hospital mortality",
        "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
        "alt_adult_los2_all_vitals_sepsis3_derivable": "ICU LOS >=2 d",
    }
    lower = token.lower()
    if lower in mapping:
        return mapping[lower]
    cleaned = lower
    for suffix in ("_filled", "_first", "_measured"):
        cleaned = cleaned.removesuffix(suffix)
    return cleaned.replace("_", " ").strip().title() or token


def short_figure_label(value: Any, *, limit: int = 38) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "..."


def iter_prior_output_tables(
    *,
    run_dir: Path,
    current_step_id: str,
) -> Sequence[Tuple[Path, pd.DataFrame]]:
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return []
    tables: List[Tuple[Path, pd.DataFrame]] = []
    for step_dir in sorted(steps_dir.iterdir()):
        if not step_dir.is_dir() or step_dir.name == current_step_id:
            continue
        outputs_dir = step_dir / "outputs"
        if not outputs_dir.exists():
            continue
        for csv_path in sorted(outputs_dir.glob("*.csv")):
            try:
                tables.append((csv_path, pd.read_csv(csv_path)))
            except Exception:
                continue
    return tables


def find_column(
    frame: pd.DataFrame,
    *,
    exact: Sequence[str] = (),
    suffixes: Sequence[str] = (),
    contains: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Optional[str]:
    excluded = {item.lower() for item in exclude}
    lower_to_orig = {str(c).lower(): c for c in frame.columns}
    for candidate in exact:
        key = candidate.lower()
        if key in lower_to_orig and key not in excluded:
            return str(lower_to_orig[key])
    for column in frame.columns:
        key = str(column).lower()
        if key in excluded:
            continue
        if suffixes and any(key.endswith(suffix.lower()) for suffix in suffixes):
            return str(column)
        if contains and any(token.lower() in key for token in contains):
            return str(column)
    return None


def as_percent(row: pd.Series, column: Optional[str]) -> Optional[float]:
    if not column:
        return None
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def event_count_column(
    frame: pd.DataFrame, denominator_col: Optional[str]
) -> Optional[str]:
    excluded = {
        "n",
        "n_total",
        "total_n",
        "denominator",
        "n_denominator",
        str(denominator_col or "").lower(),
    }
    column = find_column(
        frame,
        exact=("event_n", "events", "outcome_n", "n_positive"),
        suffixes=("_event_n", "_events", "_n"),
        exclude=tuple(excluded),
    )
    return column


def label_column(frame: pd.DataFrame) -> Optional[str]:
    return find_column(
        frame,
        exact=(
            "label",
            "group_label",
            "exposure_label",
            "stratum_label",
            "category_label",
        ),
        suffixes=("_label",),
    )


BINARY_GROUP_EXCLUDED_TOKENS = (
    "n",
    "count",
    "event",
    "events",
    "death",
    "mort",
    "risk",
    "rate",
    "prevalence",
    "incidence",
    "ci",
    "lower",
    "upper",
    "pct",
    "percent",
    "source",
    "row",
)


def binary_group_column(frame: pd.DataFrame) -> Optional[str]:
    binary_tokens = {"0", "1", "0.0", "1.0", "false", "true", "no", "yes"}
    for column in frame.columns:
        key = str(column).lower()
        if any(token in key for token in BINARY_GROUP_EXCLUDED_TOKENS):
            continue
        values = [
            str(value).strip().lower()
            for value in frame[column].dropna().tolist()
            if str(value).strip()
        ]
        if not values:
            continue
        binary_values = [value for value in values if value in binary_tokens]
        allowed_extra_values = [
            value
            for value in values
            if value not in binary_tokens and "risk_difference" not in value
        ]
        if len(set(binary_values)) >= 2 and not allowed_extra_values:
            return str(column)
    return None


def binary_group_label(column: str, value: Any) -> str:
    normalized = str(value).strip().lower()
    base = publication_label(column)
    if normalized in {"1", "1.0", "true", "yes"}:
        return f"{base} positive"
    if normalized in {"0", "0.0", "false", "no"}:
        return f"{base} negative"
    return publication_label(value)


def is_risk_difference_row(row: pd.Series, *values: Any) -> bool:
    haystack = " ".join([str(value or "") for value in values])
    haystack = (
        f"{haystack} {' '.join(str(value or '') for value in row.to_dict().values())}"
    )
    return (
        "risk_difference" in haystack.lower() or "risk difference" in haystack.lower()
    )


def context_axis_label(metric: Any, group: Any) -> str:
    metric_text = str(metric or "").strip()
    group_text = str(group or "").strip()
    if metric_text.lower() == "exposure prevalence":
        suffix = " prevalence"
        if group_text.lower().endswith(suffix) and len(group_text) > len(suffix):
            return (
                f"{short_figure_label(group_text[: -len(suffix)].strip(), limit=24)}\n"
                "prevalence"
            )
        return short_figure_label(group_text or metric_text, limit=28)
    if metric_text and group_text and metric_text.lower() not in group_text.lower():
        return (
            f"{short_figure_label(group_text, limit=24)}\n"
            f"{short_figure_label(metric_text, limit=24)}"
        )
    return short_figure_label(group_text or metric_text or "Context", limit=28)


def association_descriptive_context(
    *,
    run_dir: Path,
    current_step_id: str,
    out_dir: Path,
    primary_exposure: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect source-backed prevalence or absolute-risk rows for association figures.

    The helper is deliberately keyed to generic column semantics
    (prevalence/risk/event-rate), not to a benchmark variable name.
    """

    plot_rows: List[Dict[str, Any]] = []
    source_files: List[str] = []
    has_prevalence = False
    has_outcome_risk = False

    def _canonical_exposure_token(value: Any) -> str:
        token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
        suffixes = (
            "_log1p_active",
            "_log1p",
            "_active",
            "_maximum",
            "_minimum",
            "_median",
            "_mean",
            "_first",
            "_last",
            "_value",
            "_max",
            "_min",
        )
        changed = True
        while changed and token:
            changed = False
            for suffix in suffixes:
                if token.endswith(suffix) and len(token) > len(suffix):
                    token = token[: -len(suffix)]
                    changed = True
                    break
        return token

    primary_token = _canonical_exposure_token(primary_exposure)

    for table_path, frame in iter_prior_output_tables(
        run_dir=run_dir,
        current_step_id=current_step_id,
    ):
        if frame.empty:
            continue
        frame = frame.copy()
        if primary_token:
            exposure_col = find_column(
                frame,
                exact=(
                    "exposure",
                    "exposure_source",
                    "source_variable",
                    "concept",
                    "variable",
                ),
            )
            if exposure_col:
                matches = (
                    frame[exposure_col].map(_canonical_exposure_token).eq(primary_token)
                )
                if matches.any():
                    frame = frame.loc[matches].copy()
                elif frame[exposure_col].nunique(dropna=True) > 1:
                    # This is explicitly a multi-exposure context table but it
                    # contains no row for the locked primary exposure.
                    continue
        prevalence_pct_col = find_column(
            frame,
            exact=("prevalence_pct", "incidence_pct"),
        )
        prevalence_prop_col = find_column(frame, exact=("prevalence", "incidence"))
        if not has_prevalence and (prevalence_pct_col or prevalence_prop_col):
            denominator_col = find_column(
                frame,
                exact=("n_denominator", "denominator", "n_total", "total_n", "n"),
            )
            event_col = find_column(frame, exact=("n_positive", "event_n", "events"))
            label_col = find_column(
                frame,
                exact=("label", "exposure", "variable", "concept"),
            )
            source_rows: List[Dict[str, Any]] = []
            for idx, row in frame.iterrows():
                estimate = as_percent(row, prevalence_pct_col or prevalence_prop_col)
                if estimate is None:
                    continue
                base_label = row.get(label_col) if label_col else "Exposure"
                display_label = f"{publication_label(base_label)} prevalence"
                record = row.to_dict()
                record.update(
                    {
                        "plot_metric": "Exposure prevalence",
                        "plot_group_label": display_label,
                        "plot_estimate_pct": estimate,
                        "plot_ci_low_pct": as_percent(
                            row,
                            find_column(frame, exact=("ci_low_pct", "lower_pct"))
                            or find_column(frame, exact=("ci_low", "lower")),
                        ),
                        "plot_ci_high_pct": as_percent(
                            row,
                            find_column(frame, exact=("ci_high_pct", "upper_pct"))
                            or find_column(frame, exact=("ci_high", "upper")),
                        ),
                        "plot_denominator": (
                            row.get(denominator_col) if denominator_col else None
                        ),
                        "plot_event_n": row.get(event_col) if event_col else None,
                        "source_table": table_path.name,
                        "source_row_index": int(idx),
                    }
                )
                source_rows.append(record)
                plot_rows.append(record)
            if source_rows:
                source_path = out_dir / "publication_figure_prevalence_source_data.csv"
                pd.DataFrame(source_rows).to_csv(source_path, index=False)
                source_files.append(source_path.name)
                has_prevalence = True

        risk_pct_col = find_column(
            frame,
            exact=("outcome_risk_pct", "risk_pct", "event_rate_pct"),
            suffixes=("_risk_pct", "_rate_pct"),
            exclude=("prevalence_pct", "incidence_pct"),
        )
        risk_prop_col = find_column(
            frame,
            exact=("outcome_risk", "risk", "event_rate"),
            suffixes=("_risk", "_rate"),
            exclude=("prevalence", "incidence"),
        )
        if not has_outcome_risk and (risk_pct_col or risk_prop_col):
            denominator_col = find_column(
                frame,
                exact=("n", "n_total", "total_n", "denominator", "n_denominator"),
            )
            event_col = event_count_column(frame, denominator_col)
            label_col = label_column(frame) or find_column(
                frame,
                exact=("group", "category", "stratum", "exposure"),
            )
            binary_group_col = None if label_col else binary_group_column(frame)
            metric_source = str(risk_pct_col or risk_prop_col or "outcome risk")
            metric_label = publication_label(
                metric_source.replace("_pct", "").replace("_risk", " risk")
            )
            source_rows = []
            for idx, row in frame.iterrows():
                estimate = as_percent(row, risk_pct_col or risk_prop_col)
                if estimate is None:
                    continue
                if is_risk_difference_row(
                    row,
                    row.get(label_col) if label_col else None,
                    row.get(binary_group_col) if binary_group_col else None,
                ):
                    continue
                if label_col:
                    group_label = row.get(label_col)
                elif binary_group_col:
                    group_label = binary_group_label(
                        binary_group_col,
                        row.get(binary_group_col),
                    )
                else:
                    group_label = f"Group {idx + 1}"
                record = row.to_dict()
                record.update(
                    {
                        "plot_metric": metric_label,
                        "plot_group_label": publication_label(group_label),
                        "plot_estimate_pct": estimate,
                        "plot_ci_low_pct": as_percent(
                            row,
                            find_column(frame, exact=("ci_low_pct", "lower_pct"))
                            or find_column(
                                frame,
                                exact=("ci_low", "lower"),
                                suffixes=(
                                    "_ci_low_pct",
                                    "_ci_low",
                                    "_lower_pct",
                                    "_lower",
                                ),
                            ),
                        ),
                        "plot_ci_high_pct": as_percent(
                            row,
                            find_column(frame, exact=("ci_high_pct", "upper_pct"))
                            or find_column(
                                frame,
                                exact=("ci_high", "upper"),
                                suffixes=(
                                    "_ci_high_pct",
                                    "_ci_high",
                                    "_upper_pct",
                                    "_upper",
                                ),
                            ),
                        ),
                        "plot_denominator": (
                            row.get(denominator_col) if denominator_col else None
                        ),
                        "plot_event_n": row.get(event_col) if event_col else None,
                        "source_table": table_path.name,
                        "source_row_index": int(idx),
                    }
                )
                source_rows.append(record)
                plot_rows.append(record)
            if source_rows:
                source_path = (
                    out_dir / "publication_figure_absolute_risk_source_data.csv"
                )
                pd.DataFrame(source_rows).to_csv(source_path, index=False)
                source_files.append(source_path.name)
                has_outcome_risk = True

        if has_prevalence and has_outcome_risk:
            break

    if has_prevalence and has_outcome_risk:
        title = "Prevalence and absolute outcome risk"
        claim = "Exposure prevalence and absolute outcome risk are shown before adjusted relative estimates."
    elif has_prevalence:
        title = "Exposure prevalence"
        claim = "Exposure prevalence is shown before adjusted relative estimates."
    elif has_outcome_risk:
        title = "Absolute outcome risk"
        claim = "Absolute outcome risk is shown before adjusted relative estimates."
    else:
        title = ""
        claim = ""
    return {
        "plot_rows": plot_rows,
        "source_files": source_files,
        "has_prevalence": has_prevalence,
        "has_outcome_risk": has_outcome_risk,
        "title": title,
        "claim": claim,
    }


def truthy_figure_value(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}


def explicit_false_figure_value(value: Any) -> bool:
    if value is False:
        return True
    if value is True or value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"false", "0", "no"}


def sensitivity_plot_label(row: Mapping[str, Any]) -> str:
    spec_id = str(row.get("spec_id") or "").strip().lower()
    if spec_id.endswith("_crude_rd"):
        spec_id = spec_id.removesuffix("_crude_rd")
    mapping = {
        "full_export_step03_scope": "Full export",
        "primary_los_ge_1d": "Primary cohort",
        "primary": "Primary cohort",
        "cohort_no_los_restriction": "No ICU LOS restriction",
        "cohort_los_ge_2d": "ICU LOS >=2 d",
        "cohort_core_physiology_present": "Core physiology present",
        "primary_adult_los1_all_vitals_sepsis3_derivable": "Primary cohort",
        "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
        "alt_adult_los2_all_vitals_sepsis3_derivable": "ICU LOS >=2 d",
        "primary_lactate_complete_case": "Lactate obs.",
        "primary_without_lactate_adjustment": "No lactate adj.",
        "missing_raw_complete_case": "Complete-case",
        "missing_drop_lactate": "Drop lactate",
        "effect_robust_poisson_rr": "Risk ratio",
        "effect_marginal_standardized_rd": "Risk difference",
    }
    if spec_id in mapping:
        return mapping[spec_id]
    los_match = re.fullmatch(r"alt_cohort_los_ge_(\d+(?:p\d+)?)([hd])", spec_id)
    if los_match:
        value = los_match.group(1).replace("p", ".")
        unit = "h" if los_match.group(2) == "h" else "d"
        return f"ICU LOS >= {value} {unit}"
    if "complete_case" in spec_id:
        return "Complete-case"
    if "source_aware" in spec_id:
        return "Source-aware"
    label = str(row.get("display_label") or row.get("label") or spec_id).strip()
    return short_figure_label(label.replace("LOS ≥", "LOS >="), limit=30)


__all__ = [
    "iter_prior_output_tables",
    "find_column",
    "as_percent",
    "event_count_column",
    "label_column",
    "binary_group_column",
    "binary_group_label",
    "is_risk_difference_row",
    "context_axis_label",
    "association_descriptive_context",
    "truthy_figure_value",
    "explicit_false_figure_value",
    "sensitivity_plot_label",
    "BINARY_GROUP_EXCLUDED_TOKENS",
    "figure_parent_candidate_step_dirs",
    "publication_label",
    "short_figure_label",
]
