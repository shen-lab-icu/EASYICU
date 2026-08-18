"""Figure validators — source-data and contract-quality owners.

FigureSourceDataValidator and FigureContractQualityValidator reference each
other, so they share one owner module rather than an import cycle.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from ..canonical_json import sha256_file as _sha256_file
from ..planning.analysis_method_suite import figure_product_source_obligations
from ..contracts.declared_product import (
    effect_adjustment_family,
    effect_bearing_product,
    effect_estimand_tier,
    effect_measure_family,
    effect_role_family,
    typed_product,
)
from ..schema import (
    AnalysisStep,
    ValidationFinding,
)
from ..authority.runtime_artifacts import (
    current_run_evidence_records,
    current_successful_step_records,
    verified_run_evidence_path,
)

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and


class FigureSourceDataValidator:
    """Verify figure source-data tables are traceable to upstream step tables."""

    name = "figure_source_data"
    _SOURCE_DATA_GLOB = "*source_data*.csv"
    _KEY_COLUMNS = (
        "definition_id",
        "comparison_definition",
        "spec_id",
        "row_id",
        "concept",
        "label",
        # Model-level result tables key rows by the fitted-model label
        # (e.g. adjusted_association.csv from an association model step);
        # the deterministic figure renderer preserves that column verbatim
        # in publication_figure_source_data.csv, so it is a valid trace key.
        "model_label",
        "variable",
        "term",
        "exposure",
        "contrast",
        # Causal effect-estimation steps key each estimated contrast by
        # ``contrast_id`` (e.g. causal_effect.csv); the deterministic forest
        # renderer preserves it verbatim in publication_figure_source_data.csv,
        # so it is a valid per-row trace key. Without it a faithfully-derived
        # causal figure was rejected as "no shared key" (H2 fix3).
        "contrast_id",
        # Ordinal dose-response steps key each graded-exposure level by
        # ``stage`` (e.g. dose_response.csv rows stage=0..K); the figure renderer
        # carries it verbatim into publication_figure_source_data.csv, so it is a
        # valid per-row trace key. Without it a faithfully-derived ordinal forest
        # (odds_ratio per stage identical to the upstream table) was rejected as
        # "no shared key" (E3). The subset + numeric-equality checks below still
        # run, so this only lets a genuinely-traceable figure be verified.
        "stage",
        # A graded-categorical association forest keys each row by the ordinal
        # ``level`` / ``band`` / ``category`` of a single exposure (the exposure
        # NAME is constant across rows; the level is what varies). The association
        # bundle renderer now labels/keys rows by the varying column and carries
        # it into publication_figure_source_data.csv (M1: odds_ratio per
        # sofa2_liver_cat level). Same subset + numeric-equality guards apply.
        "level",
        "band",
        "category",
    )
    _COMPOSITE_KEY_COLUMNS = (
        ("spec_id", "model_id", "term"),
        ("spec_id", "model_id"),
        # Coefficient tables repeat ordinary terms (age/sex/etc.) across
        # multiple models.  ``term`` alone therefore creates a many-to-many
        # join and can falsely compare a primary-model estimate with its
        # complete-case or secondary-model counterpart.
        ("model_id", "term"),
        ("definition_a", "definition_b"),
        ("primary_definition", "comparison_definition"),
    )
    _NUMERIC_COLUMNS = (
        "missing_pct",
        "missing_n",
        "value_missing_pct",
        "value_missing_n",
        "measured_pct",
        "measured_n",
        "measured_one_pct",
        "measured_one_n",
        "n_nonmissing",
        "total_n",
        "n_total",
        "n_included",
        "n_excluded",
        "included_pct_of_rows",
        "overlap_with_primary_n",
        "overlap_with_primary_pct_of_primary",
        "overlap_with_primary_pct_of_definition",
        "moved_in_vs_primary_n",
        "moved_out_vs_primary_n",
        "n_a",
        "n_b",
        "intersection_n",
        "union_n",
        "jaccard",
        "a_in_b_pct",
        "b_in_a_pct",
        "point_estimate",
        "modeled_analytic_n",
        "model_contract_n",
        "event_n",
        "membership_n",
        "estimate",
        "ci_low",
        "ci_high",
        "se",
        "odds_ratio",
        "risk_ratio",
        "risk_difference",
        "p_value",
    )
    _TEXT_COLUMNS = (
        "row_type",
        "group_type",
        "estimate_type",
        "estimate_unit",
        "effect_scale",
        "model_id",
        "source_model_id",
        "source_step_id",
        "exposure_source",
        "exposure_expression",
        "exposure_role",
        "analysis_role",
        "analysis_set",
        "baseline_missing_policy",
        "fit_status",
        "fit_method",
        "value_type",
        "replay_mode",
        "coefficient_source_table",
        "coefficient_term",
        "model_contract_source",
        "source_script_sha256",
        "estimability_status",
    )
    _PCT_COUNT_RULES = (
        ("missing_pct", "missing_n", "total_n"),
        ("measured_pct", "measured_n", "total_n"),
        ("measured_pct", "n_nonmissing", "total_n"),
        # Generic long-form figure source-data contract. This catches a common
        # denominator drift where a renderer copies a percent computed against
        # the locked cohort but pairs it with the valid-observed count sum.
        ("percentage", "count", "denominator"),
    )
    _DEFAULT_NUMERIC_ABS_TOL = 1e-9
    # Deterministic summary tables commonly serialize percentages to six
    # decimal places, while figure renderers recompute the same percentages
    # from integer counts at full precision.  Treat only that serialization
    # difference as equivalent; counts, effects, intervals, and p-values keep
    # the stricter default tolerance below.
    _PERCENTAGE_ABS_TOL = 1e-6
    _POSITIONAL_ROW_INDEX_COLUMNS = (
        "source_row_index",
        "_source_row_index",
    )
    _TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".parquet", ".feather"})
    _PURE_RENDER_METHODS = frozenset(
        {
            "chart_generation",
            "figure",
            "figure_generation",
            "plotting",
            "publication_figure",
            "publication_figure_generation",
            "render_figure",
            "visualisation",
            "visualization",
        }
    )
    _PREDICTION_METHODS = frozenset(
        {
            "classification_model",
            "model_validation",
            "prediction",
            "prediction_model",
            "risk_prediction",
        }
    )
    _PREDICTION_SOURCE_ROLES = frozenset(
        {
            "auc",
            "auroc",
            "brier",
            "c_statistic",
            "calibration",
            "calibration_curve",
            "calibration_intercept",
            "calibration_slope",
            "decision_curve",
            "discrimination",
            "false_positive_rate",
            "fpr",
            "horizon_performance",
            "model_performance",
            "observed_risk",
            "predicted_probability",
            "predicted_risk",
            "prediction",
            "prediction_performance",
            "predictions",
            "risk_prediction",
            "risk_predictions",
            "risk_score",
            "roc",
            "roc_curve",
            "true_positive_rate",
            "tpr",
            "validation_performance",
        }
    )
    _PREDICTED_VALUE_ROLES = frozenset(
        {
            "predicted_probability",
            "predicted_risk",
            "prediction",
            "risk_prediction",
            "risk_score",
        }
    )
    _PREDICTED_PROBABILITY_ROLES = frozenset(
        {
            "predicted_probability",
            "predicted_risk",
            "prediction",
            "risk_prediction",
        }
    )
    _PREDICTED_SCORE_ROLES = frozenset({"risk_score"})
    _OBSERVED_OUTCOME_ROLES = frozenset(
        {
            "event",
            "label",
            "observed_outcome",
            "outcome",
            "target",
            "y_true",
        }
    )
    _OBSERVED_CALIBRATION_ROLES = frozenset(
        {
            "observed_probability",
            "observed_rate",
            "observed_risk",
        }
    )
    _PREDICTION_PERFORMANCE_METRICS = frozenset(
        {
            "auc",
            "auroc",
            "brier",
            "brier_score",
            "c_statistic",
            "calibration_intercept",
            "calibration_slope",
            "discrimination",
            "roc_auc",
        }
    )
    _PREDICTION_TIME_ROLES = frozenset(
        {
            "horizon",
            "landmark",
            "prediction_horizon",
            "prediction_time",
            "time_horizon",
        }
    )
    _FALSE_POSITIVE_RATE_ROLES = frozenset({"false_positive_rate", "fpr"})
    _TRUE_POSITIVE_RATE_ROLES = frozenset({"true_positive_rate", "tpr"})
    _UNIT_INTERVAL_PREDICTION_METRICS = frozenset(
        {
            "auc",
            "auroc",
            "brier",
            "brier_score",
            "c_statistic",
            "discrimination",
            "roc_auc",
        }
    )
    _DISCRIMINATION_PREDICTION_METRICS = frozenset(
        {
            "auc",
            "auroc",
            "c_statistic",
            "discrimination",
            "roc_auc",
        }
    )

    @staticmethod
    def _normalise(value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    @classmethod
    def _read_tabular(cls, path: Path) -> pd.DataFrame:
        """Read every tabular format accepted by the typed-evidence registry."""

        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix == ".feather":
            return pd.read_feather(path)
        raise ValueError(f"unsupported tabular suffix: {suffix or '<none>'}")

    @classmethod
    def _normalised_method_head(cls, method: Any) -> str:
        normalised = cls._normalise(method)
        return normalised.split("_with_", 1)[0]

    @classmethod
    def _figure_result_family(
        cls,
        *,
        step: AnalysisStep,
        figure_product: str,
    ) -> Optional[str]:
        parsed = typed_product(figure_product)
        if parsed is None or parsed[0] != "figure":
            return None
        obligations = set(figure_product_source_obligations(figure_product))
        if effect_bearing_product(figure_product) or any(
            item.startswith("effect:") for item in obligations
        ):
            return "effect"
        if any(item.startswith("prediction:") for item in obligations):
            return "prediction"
        if cls._normalised_method_head(step.method) in cls._PREDICTION_METHODS:
            return "prediction"
        return None

    @classmethod
    def _figure_source_obligations(
        cls,
        *,
        step: AnalysisStep,
        figure_product: str,
    ) -> Set[str]:
        obligations = set(figure_product_source_obligations(figure_product))
        if obligations:
            return obligations
        family = cls._figure_result_family(
            step=step,
            figure_product=figure_product,
        )
        if family == "effect":
            return {"effect"}
        if family == "prediction":
            return {"prediction:performance"}
        return set()

    @classmethod
    def _planned_result_families(cls, step: AnalysisStep) -> Set[str]:
        families: Set[str] = set()
        for raw in step.expected_outputs or []:
            family = cls._figure_result_family(step=step, figure_product=str(raw))
            if family is not None:
                families.add(family)
        return families

    @staticmethod
    def _role_present(value: Any, role: str) -> bool:
        normalised = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip(
            "_"
        )
        return normalised == role or f"_{role}_" in f"_{normalised}_"

    @classmethod
    def _column_role_present(cls, column: Any, role: str) -> bool:
        """Match a declared semantic column role without substring capture.

        Product identifiers and long-form metric labels may carry namespace
        riders, but tabular columns are the actual replay schema.  Treating a
        token anywhere in a column name as its value role makes metadata such
        as ``auroc_ci_method`` or ``prediction_horizon_hours`` masquerade as a
        numeric value column.
        """

        return cls._normalise(column) == cls._normalise(role)

    @classmethod
    def _time_column_role_present(cls, column: Any, role: str) -> bool:
        normalised = cls._normalise(column)
        expected = cls._normalise(role)
        if normalised == expected:
            return True
        unit_suffixes = {
            "day",
            "days",
            "hour",
            "hours",
            "minute",
            "minutes",
            "month",
            "months",
            "week",
            "weeks",
            "year",
            "years",
        }
        if not normalised.startswith(f"{expected}_"):
            return False
        return normalised.removeprefix(f"{expected}_") in unit_suffixes

    @classmethod
    def _prediction_metric_column_roles(
        cls,
        column: Any,
    ) -> List[Tuple[str, str, str]]:
        """Return structured metric roles as ``(role, group, value_kind)``.

        A closed set of value/bound suffixes keeps numeric metric payloads
        auditable while excluding prose metadata such as ``*_ci_method``.
        Interval bounds are validated but never establish performance without
        a point estimate.
        """

        normalised = cls._normalise(column)
        contexts = {
            "development",
            "external",
            "internal",
            "test",
            "train",
            "validation",
        }
        point_suffixes = {"estimate", "point_estimate", "value"}
        lower_suffixes = {
            "ci_low",
            "ci_lower",
            "confidence_interval_low",
            "confidence_interval_lower",
            "lcl",
            "lower",
        }
        upper_suffixes = {
            "ci_high",
            "ci_upper",
            "confidence_interval_high",
            "confidence_interval_upper",
            "ucl",
            "upper",
        }
        matches: List[Tuple[str, str, str]] = []
        for role in sorted(cls._PREDICTION_PERFORMANCE_METRICS, key=len, reverse=True):
            candidates = [("", normalised)]
            prefix, separator, remainder = normalised.partition("_")
            if separator and prefix in contexts:
                candidates.append((prefix, remainder))
            for context, candidate in candidates:
                group = f"{context}:{role}" if context else role
                if candidate == role:
                    matches.append((role, group, "point"))
                    break
                role_prefix = f"{role}_"
                if not candidate.startswith(role_prefix):
                    continue
                suffix = candidate.removeprefix(role_prefix)
                if suffix in point_suffixes:
                    matches.append((role, group, "point"))
                    break
                if suffix in lower_suffixes:
                    matches.append((role, group, "lower"))
                    break
                if suffix in upper_suffixes:
                    matches.append((role, group, "upper"))
                    break
        return matches

    @classmethod
    def _has_row_paired_prediction_outcome(
        cls,
        frame: pd.DataFrame,
        predictor_columns: Sequence[str],
        outcome_columns: Sequence[str],
        *,
        require_both_classes: bool,
    ) -> bool:
        for predictor in predictor_columns:
            for outcome in outcome_columns:
                paired = frame[[predictor, outcome]].dropna()
                if paired.empty:
                    continue
                if not cls._finite_numeric_values(paired[predictor]):
                    continue
                if not cls._finite_numeric_values(paired[outcome]):
                    continue
                if require_both_classes and not cls._series_is_binary_outcome(
                    paired[outcome]
                ):
                    continue
                return True
        return False

    @classmethod
    def _has_complete_numeric_rows(
        cls,
        frame: pd.DataFrame,
        column_groups: Sequence[Sequence[str]],
        *,
        minimum_rows: int = 1,
        require_distinct_first: bool = False,
    ) -> bool:
        if not column_groups or any(not group for group in column_groups):
            return False
        for columns in itertools.product(*column_groups):
            paired = frame[list(columns)].dropna()
            if len(paired) < minimum_rows:
                continue
            if not all(
                cls._finite_numeric_values(paired[column]) for column in columns
            ):
                continue
            if require_distinct_first:
                first_values = cls._finite_numeric_values(paired[columns[0]])
                if len(set(first_values)) < 2:
                    continue
            return True
        return False

    @staticmethod
    def _series_has_finite_numeric(series: pd.Series) -> bool:
        numeric = pd.to_numeric(series, errors="coerce")
        return any(math.isfinite(float(value)) for value in numeric.dropna().tolist())

    @staticmethod
    def _finite_numeric_values(series: pd.Series) -> List[float]:
        raw = series.dropna()
        if raw.empty:
            return []
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.isna().any():
            return []
        values = [float(value) for value in numeric.tolist()]
        if not values or not all(math.isfinite(value) for value in values):
            return []
        return values

    @classmethod
    def _series_in_unit_interval(cls, series: pd.Series) -> bool:
        values = cls._finite_numeric_values(series)
        return bool(values) and all(0.0 <= value <= 1.0 for value in values)

    @classmethod
    def _series_is_binary_outcome(cls, series: pd.Series) -> bool:
        values = cls._finite_numeric_values(series)
        if not values:
            return False
        has_zero = any(math.isclose(value, 0.0, abs_tol=1e-12) for value in values)
        has_one = any(math.isclose(value, 1.0, abs_tol=1e-12) for value in values)
        return (
            has_zero
            and has_one
            and all(
                math.isclose(value, 0.0, abs_tol=1e-12)
                or math.isclose(value, 1.0, abs_tol=1e-12)
                for value in values
            )
        )

    @classmethod
    def _matching_domain_columns(
        cls,
        frame: pd.DataFrame,
        roles: Set[str] | frozenset[str],
        predicate: Callable[[pd.Series], bool],
    ) -> List[str]:
        matching = [
            str(column)
            for column in frame.columns
            if any(cls._column_role_present(column, role) for role in roles)
        ]
        if not matching or not all(predicate(frame[column]) for column in matching):
            return []
        return matching

    @classmethod
    def _prediction_metric_values_valid(
        cls,
        metric: Any,
        series: pd.Series,
    ) -> bool:
        metric_name = cls._normalise(metric)
        values = cls._finite_numeric_values(series)
        if not values:
            return False
        if any(
            cls._role_present(metric_name, role)
            for role in cls._UNIT_INTERVAL_PREDICTION_METRICS
        ):
            return all(0.0 <= value <= 1.0 for value in values)
        return any(
            cls._role_present(metric_name, role)
            for role in cls._PREDICTION_PERFORMANCE_METRICS
        )

    @classmethod
    def _prediction_metric_interval_valid(
        cls,
        *,
        metric: str,
        point: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
    ) -> bool:
        lower_present = lower.notna()
        upper_present = upper.notna()
        if not lower_present.equals(upper_present) or not bool(lower_present.any()):
            return False
        if not bool(point[lower_present].notna().all()):
            return False
        point_slice = point[lower_present]
        lower_slice = lower[lower_present]
        upper_slice = upper[upper_present]
        if not all(
            cls._prediction_metric_values_valid(metric, values)
            for values in (point_slice, lower_slice, upper_slice)
        ):
            return False
        point_values = cls._finite_numeric_values(point_slice)
        lower_values = cls._finite_numeric_values(lower_slice)
        upper_values = cls._finite_numeric_values(upper_slice)
        return bool(point_values) and all(
            low <= estimate <= high
            for estimate, low, high in zip(
                point_values,
                lower_values,
                upper_values,
            )
        )

    @classmethod
    def _matching_finite_columns(
        cls,
        frame: pd.DataFrame,
        roles: Set[str] | frozenset[str],
    ) -> List[str]:
        matching = [
            str(column)
            for column in frame.columns
            if any(cls._column_role_present(column, role) for role in roles)
        ]
        if not matching or not all(
            bool(cls._finite_numeric_values(frame[column])) for column in matching
        ):
            return []
        return matching

    @classmethod
    def _prediction_source_obligations(
        cls,
        *,
        product: str,
        frame: Optional[pd.DataFrame],
        statistic_value: Optional[float] = None,
    ) -> Set[str]:
        """Return replayable prediction display obligations for one source."""

        parsed_product = typed_product(product)
        product_supported = any(
            cls._role_present(product, role) for role in cls._PREDICTION_SOURCE_ROLES
        )
        if not product_supported:
            return set()
        if parsed_product is not None and parsed_product[0] == "statistic":
            metric_role = next(
                (
                    role
                    for role in cls._PREDICTION_PERFORMANCE_METRICS
                    if cls._role_present(product, role)
                ),
                None,
            )
            if metric_role is None:
                return set()
            if statistic_value is not None and not (
                cls._prediction_metric_values_valid(
                    metric_role,
                    pd.Series([statistic_value]),
                )
            ):
                return set()
            return {"prediction:performance"}
        if frame is None:
            return set()

        obligations: Set[str] = set()
        probability_columns = cls._matching_domain_columns(
            frame,
            cls._PREDICTED_PROBABILITY_ROLES,
            cls._series_in_unit_interval,
        )
        score_columns = cls._matching_finite_columns(
            frame,
            cls._PREDICTED_SCORE_ROLES,
        )
        observed_outcome_columns = cls._matching_domain_columns(
            frame,
            cls._OBSERVED_OUTCOME_ROLES,
            cls._series_is_binary_outcome,
        )
        probability_outcome_paired = cls._has_row_paired_prediction_outcome(
            frame,
            probability_columns,
            observed_outcome_columns,
            require_both_classes=True,
        )
        score_outcome_paired = cls._has_row_paired_prediction_outcome(
            frame,
            score_columns,
            observed_outcome_columns,
            require_both_classes=True,
        )
        if probability_outcome_paired:
            # Patient-level predictions plus observed outcomes are sufficient to
            # replay discrimination, calibration, aggregate performance, and DCA.
            obligations.update(
                {
                    "prediction:calibration",
                    "prediction:decision",
                    "prediction:performance",
                    "prediction:roc",
                }
            )
        elif score_outcome_paired:
            # An arbitrary continuous score can replay rank discrimination, but
            # it is not a calibrated probability and cannot authorize calibration,
            # Brier, or decision-curve displays.
            obligations.update({"prediction:performance", "prediction:roc"})

        observed_calibration_columns = cls._matching_domain_columns(
            frame,
            cls._OBSERVED_CALIBRATION_ROLES,
            cls._series_in_unit_interval,
        )
        if cls._has_complete_numeric_rows(
            frame,
            (probability_columns, observed_calibration_columns),
            minimum_rows=2,
            require_distinct_first=True,
        ):
            obligations.add("prediction:calibration")

        false_positive_rate_columns = cls._matching_domain_columns(
            frame,
            cls._FALSE_POSITIVE_RATE_ROLES,
            cls._series_in_unit_interval,
        )
        true_positive_rate_columns = cls._matching_domain_columns(
            frame,
            cls._TRUE_POSITIVE_RATE_ROLES,
            cls._series_in_unit_interval,
        )
        threshold_columns = cls._matching_finite_columns(
            frame,
            frozenset({"threshold"}),
        )
        if cls._has_complete_numeric_rows(
            frame,
            (
                threshold_columns,
                false_positive_rate_columns,
                true_positive_rate_columns,
            ),
            minimum_rows=2,
            require_distinct_first=True,
        ):
            obligations.add("prediction:roc")
        net_benefit_columns = cls._matching_finite_columns(
            frame,
            frozenset({"net_benefit"}),
        )
        probability_threshold_columns = cls._matching_domain_columns(
            frame,
            frozenset({"threshold"}),
            cls._series_in_unit_interval,
        )
        if cls._has_complete_numeric_rows(
            frame,
            (probability_threshold_columns, net_benefit_columns),
            minimum_rows=2,
            require_distinct_first=True,
        ):
            obligations.add("prediction:decision")

        performance_rows: Set[Any] = set()
        discrimination_rows: Set[Any] = set()
        performance_payload_valid = True
        performance_payload_has_valid_value = False
        metric_payloads: Dict[str, Dict[str, Any]] = {}
        generic_metric_intervals: Dict[str, List[str]] = {
            "lower": [],
            "upper": [],
        }
        for column in frame.columns:
            parsed_interval = cls._confidence_interval_bound(column)
            if parsed_interval is not None and not parsed_interval[0]:
                generic_metric_intervals[parsed_interval[1]].append(str(column))
        for column in frame.columns:
            matching_metrics = cls._prediction_metric_column_roles(column)
            if not matching_metrics:
                continue
            column_valid = all(
                cls._prediction_metric_values_valid(metric, frame[column])
                for metric, _group, _kind in matching_metrics
            )
            if not column_valid:
                performance_payload_valid = False
                continue
            for metric, group, kind in matching_metrics:
                payload = metric_payloads.setdefault(
                    group,
                    {"role": metric, "point": [], "lower": [], "upper": []},
                )
                payload[kind].append(str(column))
                if kind != "point":
                    continue
                performance_payload_has_valid_value = True
                point_rows = set(frame[column].dropna().index.tolist())
                performance_rows.update(point_rows)
                if metric in cls._DISCRIMINATION_PREDICTION_METRICS:
                    discrimination_rows.update(point_rows)
        if generic_metric_intervals["lower"] or generic_metric_intervals["upper"]:
            point_payloads = [
                payload for payload in metric_payloads.values() if payload["point"]
            ]
            if len(point_payloads) == 1 and not (
                point_payloads[0]["lower"] or point_payloads[0]["upper"]
            ):
                point_payloads[0]["lower"].extend(generic_metric_intervals["lower"])
                point_payloads[0]["upper"].extend(generic_metric_intervals["upper"])
            elif point_payloads:
                performance_payload_valid = False
        for payload in metric_payloads.values():
            has_interval = bool(payload["lower"] or payload["upper"])
            if not has_interval:
                continue
            if not (
                len(payload["point"]) == 1
                and len(payload["lower"]) == 1
                and len(payload["upper"]) == 1
            ):
                performance_payload_valid = False
                continue
            if not cls._prediction_metric_interval_valid(
                metric=str(payload["role"]),
                point=frame[payload["point"][0]],
                lower=frame[payload["lower"][0]],
                upper=frame[payload["upper"][0]],
            ):
                performance_payload_valid = False
        label_columns = [
            column
            for column in frame.columns
            if cls._normalise(column) in {"metric", "name", "statistic"}
        ]
        value_columns = [
            column
            for column in frame.columns
            if cls._normalise(column) in {"estimate", "result", "value"}
        ]
        long_interval_columns = generic_metric_intervals
        for label_column in label_columns:
            for row_index, metric_label in frame[label_column].items():
                metric_role = next(
                    (
                        role
                        for role in cls._PREDICTION_PERFORMANCE_METRICS
                        if cls._normalise(metric_label) == role
                    ),
                    None,
                )
                if metric_role is None:
                    continue
                present_values = [
                    value_column
                    for value_column in value_columns
                    if pd.notna(frame.at[row_index, value_column])
                ]
                row_valid = bool(present_values) and all(
                    cls._prediction_metric_values_valid(
                        metric_role,
                        frame.loc[[row_index], value_column],
                    )
                    for value_column in present_values
                )
                present_lower = [
                    column
                    for column in long_interval_columns["lower"]
                    if pd.notna(frame.at[row_index, column])
                ]
                present_upper = [
                    column
                    for column in long_interval_columns["upper"]
                    if pd.notna(frame.at[row_index, column])
                ]
                if present_lower or present_upper:
                    row_valid = row_valid and (
                        len(present_values) == 1
                        and len(present_lower) == 1
                        and len(present_upper) == 1
                        and cls._prediction_metric_interval_valid(
                            metric=metric_role,
                            point=frame.loc[[row_index], present_values[0]],
                            lower=frame.loc[[row_index], present_lower[0]],
                            upper=frame.loc[[row_index], present_upper[0]],
                        )
                    )
                if row_valid:
                    performance_payload_has_valid_value = True
                    performance_rows.add(row_index)
                    if metric_role in cls._DISCRIMINATION_PREDICTION_METRICS:
                        discrimination_rows.add(row_index)
                else:
                    performance_payload_valid = False

        if performance_payload_has_valid_value and performance_payload_valid:
            obligations.add("prediction:performance")
        elif not performance_payload_valid:
            # A valid sibling must not launder an out-of-domain value carrying
            # the same semantic metric role in the same source-data payload.
            obligations.discard("prediction:performance")

        candidate_time_columns = [
            column
            for column in frame.columns
            if any(
                cls._time_column_role_present(column, role)
                for role in cls._PREDICTION_TIME_ROLES
            )
        ]
        valid_time_varying_discrimination = False
        for time_column in candidate_time_columns:
            time_values = cls._finite_numeric_values(frame[time_column])
            if len(set(time_values)) < 2:
                continue

            paired_metric_values = cls._finite_numeric_values(
                frame.loc[list(discrimination_rows), time_column]
                if discrimination_rows
                else pd.Series(dtype=float)
            )
            if len(set(paired_metric_values)) >= 2:
                valid_time_varying_discrimination = True
                break

            replayable_raw_horizons: Set[float] = set()
            for horizon_value, group in frame.dropna(subset=[time_column]).groupby(
                time_column
            ):
                probability_replay = cls._has_row_paired_prediction_outcome(
                    group,
                    probability_columns,
                    observed_outcome_columns,
                    require_both_classes=True,
                )
                score_replay = cls._has_row_paired_prediction_outcome(
                    group,
                    score_columns,
                    observed_outcome_columns,
                    require_both_classes=True,
                )
                if probability_replay or score_replay:
                    numeric_horizon = cls._as_float(horizon_value)
                    if numeric_horizon is not None:
                        replayable_raw_horizons.add(numeric_horizon)
            if len(replayable_raw_horizons) >= 2:
                valid_time_varying_discrimination = True
                break

        if (
            "prediction:performance" in obligations
            and valid_time_varying_discrimination
        ):
            obligations.add("prediction:time_varying_discrimination")
        return obligations

    @staticmethod
    def _effect_semantics_support_figure(
        *,
        semantic_signals: Sequence[str],
        figure_product: str,
    ) -> bool:
        """Require one source to preserve the figure's scientific semantics."""

        output_measure = effect_measure_family(figure_product)
        output_role = effect_role_family(figure_product)
        registered_roles = {
            obligation.split(":", 1)[1]
            for obligation in figure_product_source_obligations(figure_product)
            if obligation.startswith("effect:")
        }
        output_tier = effect_estimand_tier(figure_product)
        output_adjustment = effect_adjustment_family(figure_product)
        input_measures = {
            family
            for signal in semantic_signals
            if (family := effect_measure_family(signal)) is not None
        }
        input_roles = {
            family
            for signal in semantic_signals
            if (family := effect_role_family(signal)) is not None
        }
        input_tiers = {
            family
            for signal in semantic_signals
            if (family := effect_estimand_tier(signal)) is not None
        }
        input_adjustments = {
            family
            for signal in semantic_signals
            if (family := effect_adjustment_family(signal)) is not None
        }
        if output_measure is not None and output_measure not in input_measures:
            return False
        required_roles = ({output_role} if output_role is not None else set()) | (
            registered_roles
        )
        if registered_roles and not input_measures:
            # A specialised effect display (for example subgroup or interaction)
            # must preserve both its role and an explicit effect scale. A generic
            # ``estimate`` column is not enough to establish forest-plot meaning.
            return False
        if required_roles:
            if not required_roles.issubset(input_roles):
                return False
        elif input_roles:
            return False
        if output_tier is not None:
            if output_tier not in input_tiers:
                return False
        elif input_tiers & {"secondary", "sensitivity", "corroborative"}:
            return False
        if output_adjustment is not None and output_adjustment not in input_adjustments:
            return False
        return True

    @classmethod
    def _contract_scoped_effect_product(
        cls,
        *,
        product: str,
        source_frame: pd.DataFrame,
        upstream_frame: Optional[pd.DataFrame] = None,
        upstream_step_id: str,
        completed_step_records: Optional[Sequence[Dict[str, Any]]],
    ) -> str:
        """Add an estimand tier only when rows match validated model contracts.

        A generic coefficient table name cannot by itself prove that selected
        rows are primary, secondary, or sensitivity estimates.  Once the
        figure source has value-matched that table, its exact ``model_id`` and
        exposure rows may inherit the tier from the successful parent step's
        machine-readable model contracts.  Free text and variable names are
        never routing authority.
        """

        parsed = typed_product(product)
        contract_frame = source_frame
        if (
            "model_id" not in contract_frame.columns
            and upstream_frame is not None
            and "source_row_index" in source_frame.columns
        ):
            positions = pd.to_numeric(source_frame["source_row_index"], errors="coerce")
            if (
                positions.notna().all()
                and positions.mod(1).eq(0).all()
                and positions.between(0, len(upstream_frame) - 1).all()
            ):
                contract_frame = upstream_frame.iloc[
                    positions.astype(int).tolist()
                ].reset_index(drop=True)
        if (
            parsed is None
            or not effect_bearing_product(product)
            or "model_id" not in contract_frame.columns
            or not completed_step_records
        ):
            return product
        model_ids = {
            str(value).strip()
            for value in contract_frame["model_id"].dropna().tolist()
            if str(value).strip()
        }
        if not model_ids:
            return product
        parent_records = [
            record
            for record in current_successful_step_records(completed_step_records)
            if str(record.get("step_id") or "").strip() == upstream_step_id
        ]
        if len(parent_records) != 1:
            return product
        summary = parent_records[0].get("step_summary")
        contracts = (
            summary.get("model_contracts") if isinstance(summary, Mapping) else None
        )
        contract_by_model: Dict[str, Mapping[str, Any]] = {}
        for contract in contracts or []:
            if not isinstance(contract, Mapping):
                continue
            model_id = str(contract.get("model_id") or "").strip()
            if not model_id or model_id in contract_by_model:
                return product
            contract_by_model[model_id] = contract
        if not model_ids <= set(contract_by_model):
            return product
        selected_contracts = [contract_by_model[model_id] for model_id in model_ids]
        tiers = {
            cls._normalise(contract.get("analysis_role"))
            for contract in selected_contracts
        }
        allowed_tiers = {"primary", "secondary", "sensitivity", "corroborative"}
        if (
            len(tiers) != 1
            or not tiers <= allowed_tiers
            or any(
                cls._normalise(contract.get("fit_status")) != "fitted"
                for contract in selected_contracts
            )
        ):
            return product
        if {"term_role", "source_variable"} <= set(contract_frame.columns):
            exposure_rows = contract_frame.loc[
                contract_frame["term_role"].map(cls._normalise).eq("exposure")
            ]
            if exposure_rows.empty:
                return product
            for _, row in exposure_rows.iterrows():
                contract = contract_by_model.get(str(row.get("model_id") or "").strip())
                if (
                    contract is None
                    or str(row.get("source_variable") or "").strip()
                    != str(contract.get("exposure_source") or "").strip()
                ):
                    return product
        elif "exposure" in contract_frame.columns:
            # One-row-per-model display tables may use the exact exposure
            # source as their row identity instead of re-exporting the richer
            # term-level ``term_role``/``source_variable`` columns.  Accept
            # that compact shape only when every row maps exactly to a fitted
            # host model contract and preserves its locked exposure source.
            for _, row in contract_frame.iterrows():
                contract = contract_by_model.get(str(row.get("model_id") or "").strip())
                if (
                    contract is None
                    or str(row.get("exposure") or "").strip()
                    != str(contract.get("exposure_source") or "").strip()
                ):
                    return product
        else:
            return product
        tier = next(iter(tiers))
        kind, name = parsed
        if effect_estimand_tier(product) is not None:
            return product
        return f"{kind}:{tier}_{name}"

    @classmethod
    def _confidence_interval_bound(
        cls,
        column: Any,
    ) -> Optional[Tuple[str, str]]:
        normalised = cls._normalise(column)
        patterns = (
            r"^(?P<prefix>.*?)(?:_)?(?:ci|confidence_interval)_"
            r"(?P<bound>low|lower|lcl|high|upper|ucl)$",
            r"^(?P<prefix>.*?)(?:_)?(?P<bound>low|lower|lcl|high|upper|ucl)_"
            r"(?:ci|confidence_interval)$",
            r"^(?P<prefix>.*?)(?:_)?(?P<bound>lcl|ucl)$",
            r"^(?P<prefix>.*?)(?:_)?(?P<bound>lower|upper)$",
        )
        for pattern in patterns:
            matched = re.fullmatch(pattern, normalised)
            if matched is None:
                continue
            bound = matched.group("bound")
            side = "lower" if bound in {"low", "lower", "lcl"} else "upper"
            return matched.group("prefix").strip("_"), side
        return None

    @classmethod
    def _ratio_intervals_valid(
        cls,
        frame: pd.DataFrame,
        ratio_point_columns: Sequence[str],
    ) -> bool:
        interval_columns: Dict[str, Dict[str, List[str]]] = {}
        for column in frame.columns:
            parsed = cls._confidence_interval_bound(column)
            if parsed is None:
                continue
            prefix, side = parsed
            interval_columns.setdefault(
                prefix,
                {"lower": [], "upper": []},
            )[side].append(str(column))
        normalised_points = {
            str(column): cls._normalise(column) for column in ratio_point_columns
        }

        def matched_ratio_points(prefix: str) -> List[str]:
            prefix_family = effect_measure_family(f"table:{prefix}")
            return [
                column
                for column, normalised in normalised_points.items()
                if normalised == prefix
                or (
                    prefix_family is not None
                    and effect_measure_family(f"table:{normalised}") == prefix_family
                )
            ]

        explicitly_covered_points = {
            column
            for prefix in interval_columns
            if prefix
            for column in matched_ratio_points(prefix)
        }
        for prefix, sides in interval_columns.items():
            if prefix:
                matched_points = matched_ratio_points(prefix)
            else:
                matched_points = [
                    column
                    for column in normalised_points
                    if column not in explicitly_covered_points
                ]
            if not matched_points:
                # A signed interval for another estimand in the same table is
                # not a ratio-scale interval and must not poison the ratio.
                continue
            if len(matched_points) != 1:
                return False
            if len(sides["lower"]) != 1 or len(sides["upper"]) != 1:
                return False
            point_column = matched_points[0]
            lower_column = sides["lower"][0]
            upper_column = sides["upper"][0]
            point_raw = frame[point_column]
            lower_raw = frame[lower_column]
            upper_raw = frame[upper_column]
            lower_present = lower_raw.notna()
            upper_present = upper_raw.notna()
            if not lower_present.equals(upper_present) or not bool(lower_present.any()):
                return False
            if not bool(point_raw[lower_present].notna().all()):
                return False
            point = pd.to_numeric(point_raw[lower_present], errors="coerce")
            lower = pd.to_numeric(lower_raw[lower_present], errors="coerce")
            upper = pd.to_numeric(upper_raw[upper_present], errors="coerce")
            if point.isna().any() or lower.isna().any() or upper.isna().any():
                return False
            point_values = [float(value) for value in point.tolist()]
            lower_values = [float(value) for value in lower.tolist()]
            upper_values = [float(value) for value in upper.tolist()]
            if not all(
                math.isfinite(estimate)
                and math.isfinite(low)
                and math.isfinite(high)
                and 0.0 < low <= estimate <= high
                for estimate, low, high in zip(
                    point_values,
                    lower_values,
                    upper_values,
                )
            ):
                return False
        return True

    @classmethod
    def _source_supports_result_family(
        cls,
        *,
        product: str,
        frame: Optional[pd.DataFrame] = None,
        family: Optional[str],
        figure_products: Sequence[str] = (),
    ) -> bool:
        """Return whether a typed value source can authenticate the figure family.

        The source product and its immutable table schema are host-owned.  Figure
        contract prose and panel roles are intentionally not consulted.
        """

        if family is None:
            return True
        parsed_product = typed_product(product)
        columns = list(frame.columns) if frame is not None else []
        if family == "prediction":
            source_obligations = cls._prediction_source_obligations(
                product=product,
                frame=frame,
            )
            if not source_obligations:
                return False
            if not figure_products:
                return True
            return all(
                {
                    obligation
                    for obligation in (
                        figure_product_source_obligations(figure)
                        or ("prediction:performance",)
                    )
                    if obligation.startswith("prediction:")
                }.issubset(source_obligations)
                for figure in figure_products
            )
        if family != "effect":
            return True

        if not effect_bearing_product(product):
            return False
        if parsed_product is not None and parsed_product[0] == "statistic":
            semantic_signals = [product]
            return all(
                cls._effect_semantics_support_figure(
                    semantic_signals=semantic_signals,
                    figure_product=figure,
                )
                for figure in figure_products
                if effect_bearing_product(figure)
                or any(
                    obligation.startswith("effect:")
                    for obligation in figure_product_source_obligations(figure)
                )
            )
        if frame is None:
            return False
        typed_columns = [f"table:{column}" for column in columns]
        generic_value_columns = {
            "coef",
            "coefficient",
            "effect",
            "effect_estimate",
            "estimate",
            "point_estimate",
            "value",
        }
        finite_effect_columns = [
            signal
            for signal, column in zip(typed_columns, columns)
            if (
                effect_bearing_product(signal)
                or effect_measure_family(signal) is not None
                or cls._normalise(column) in generic_value_columns
            )
            and cls._series_has_finite_numeric(frame[column])
        ]
        if not finite_effect_columns:
            return False
        source_measure = effect_measure_family(product)
        ratio_families = {"hazard_ratio", "odds_ratio", "risk_ratio"}
        ratio_point_columns: List[str] = []
        for signal, column in zip(typed_columns, columns):
            column_measure = effect_measure_family(signal)
            if column_measure not in ratio_families and not (
                source_measure in ratio_families
                and cls._normalise(column) in generic_value_columns
            ):
                continue
            ratio_point_columns.append(str(column))
            values = cls._finite_numeric_values(frame[column])
            if not values or any(value <= 0.0 for value in values):
                return False
        if ratio_point_columns and not cls._ratio_intervals_valid(
            frame,
            ratio_point_columns,
        ):
            return False
        semantic_signals = [
            product,
            *(
                signal
                for signal in finite_effect_columns
                if effect_bearing_product(signal)
                or effect_measure_family(signal) is not None
            ),
        ]
        effect_figures = [
            figure
            for figure in figure_products
            if effect_bearing_product(figure)
            or any(
                obligation.startswith("effect:")
                for obligation in figure_product_source_obligations(figure)
            )
        ]
        if not effect_figures:
            return True
        return all(
            cls._effect_semantics_support_figure(
                semantic_signals=semantic_signals,
                figure_product=figure,
            )
            for figure in effect_figures
        )

    @classmethod
    def _source_supports_figures(
        cls,
        *,
        step: AnalysisStep,
        product: str,
        frame: Optional[pd.DataFrame],
        figure_products: Sequence[str],
        require_all: bool,
    ) -> bool:
        checks = [
            cls._source_supports_result_family(
                product=product,
                frame=frame,
                family=cls._figure_result_family(
                    step=step,
                    figure_product=figure,
                ),
                figure_products=[figure],
            )
            for figure in figure_products
        ]
        if not checks:
            return True
        return all(checks) if require_all else any(checks)

    @classmethod
    def _extract_statistic_value(
        cls,
        step_summary: Any,
        product_name: str,
    ) -> Optional[float]:
        """Extract one unambiguous finite scalar for an exact statistic product."""

        target = cls._normalise(product_name)
        candidates: List[float] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                declared_name = value.get("name") or value.get("statistic")
                if (
                    declared_name is not None
                    and cls._normalise(declared_name) == target
                ):
                    for field in ("value", "estimate", "result"):
                        numeric = cls._as_float(value.get(field))
                        if numeric is not None:
                            candidates.append(numeric)
                for key, child in value.items():
                    if cls._normalise(key) == target:
                        numeric = cls._as_float(child)
                        if numeric is not None:
                            candidates.append(numeric)
                    if isinstance(child, (Mapping, list, tuple)):
                        visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(step_summary)
        if not candidates:
            return None
        first = candidates[0]
        if any(
            not math.isclose(item, first, rel_tol=1e-9, abs_tol=1e-9)
            for item in candidates[1:]
        ):
            return None
        return first

    @classmethod
    def _source_contains_statistic(
        cls,
        source_df: pd.DataFrame,
        *,
        product_name: str,
        expected: float,
    ) -> bool:
        target = cls._normalise(product_name)

        def values_match(series: pd.Series) -> bool:
            raw = series.dropna()
            if raw.empty:
                return False
            values = pd.to_numeric(raw, errors="coerce")
            if values.isna().any():
                return False
            return all(
                math.isfinite(float(value))
                and math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-9)
                for value in values
            )

        target_family = cls._statistic_family(target)
        for column in source_df.columns:
            column_name = cls._normalise(column)
            if column_name != target and (
                target_family is None
                or cls._statistic_family(column_name) != target_family
            ):
                continue
            if values_match(source_df[column]):
                return True

        label_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"metric", "name", "product", "statistic"}
        ]
        value_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"estimate", "result", "value"}
        ]
        for label_column in label_columns:
            normalised_labels = source_df[label_column].map(cls._normalise)
            matching_rows = normalised_labels.eq(target)
            if target_family is not None:
                matching_rows |= normalised_labels.map(cls._statistic_family).eq(
                    target_family
                )
            if not matching_rows.any():
                continue
            for value_column in value_columns:
                if values_match(source_df.loc[matching_rows, value_column]):
                    return True
        return False

    @classmethod
    def _statistic_family(cls, value: Any) -> Optional[str]:
        normalised = cls._normalise(value)
        effect_family = effect_measure_family(f"statistic:{normalised}")
        if effect_family is not None:
            return f"effect:{effect_family}"
        for family, aliases in {
            "auroc": {"auc", "auroc", "c_statistic", "roc_auc"},
            "brier": {"brier", "brier_score"},
            "calibration_intercept": {"calibration_intercept"},
            "calibration_slope": {"calibration_slope"},
        }.items():
            if normalised in aliases:
                return f"prediction:{family}"
        return None

    @classmethod
    def _statistic_payload_issue(
        cls,
        source_df: pd.DataFrame,
        *,
        required_statistics: Mapping[str, tuple[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """Return the first unbound numeric cell in a statistic-only source.

        Finding one truthful scalar must not authenticate unrelated plotted
        numbers in the same source-data file.  Table-backed sources are checked
        by the table comparator instead; this helper governs the scalar-only
        fallback.
        """

        required = [
            (
                cls._normalise(product_name),
                cls._statistic_family(product_name),
                expected,
            )
            for product_name, expected in required_statistics.values()
        ]

        def matching_expected(label: Any) -> List[float]:
            normalised = cls._normalise(label)
            family = cls._statistic_family(normalised)
            return [
                expected
                for target, target_family, expected in required
                if normalised == target
                or (
                    family is not None
                    and target_family is not None
                    and family == target_family
                )
            ]

        def agrees(value: Any, expected_values: Sequence[float]) -> bool:
            numeric = cls._as_float(value)
            return numeric is not None and any(
                math.isclose(numeric, expected, rel_tol=1e-9, abs_tol=1e-9)
                for expected in expected_values
            )

        label_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"metric", "name", "product", "statistic"}
        ]
        value_columns = [
            column
            for column in source_df.columns
            if cls._normalise(column) in {"estimate", "result", "value"}
        ]
        verified_cells: Set[tuple[Any, Any]] = set()
        if label_columns and value_columns:
            for row_index, row in source_df.iterrows():
                expected_values = [
                    expected
                    for label_column in label_columns
                    for expected in matching_expected(row[label_column])
                ]
                for value_column in value_columns:
                    if pd.isna(row[value_column]):
                        continue
                    if not agrees(row[value_column], expected_values):
                        return {
                            "reason": "unbound_statistic_value",
                            "column": str(value_column),
                            "row": str(row_index),
                            "value": row[value_column],
                        }
                    verified_cells.add((row_index, value_column))

        exempt_columns = {
            *cls._KEY_COLUMNS,
            *cls._POSITIONAL_ROW_INDEX_COLUMNS,
            "source_step_id",
            "source_table",
        }
        for column in source_df.columns:
            normalised_column = cls._normalise(column)
            if column in label_columns or normalised_column in exempt_columns:
                continue
            for row_index, value in source_df[column].items():
                if (row_index, column) in verified_cells or pd.isna(value):
                    continue
                numeric = cls._as_float(value)
                if numeric is None:
                    continue
                expected_values = matching_expected(column)
                if not agrees(numeric, expected_values):
                    return {
                        "reason": "unbound_statistic_value",
                        "column": str(column),
                        "row": str(row_index),
                        "value": numeric,
                    }
        return None

    @staticmethod
    def _iter_string_values(value: Any) -> List[str]:
        values: List[str] = []
        if isinstance(value, str):
            if value.strip():
                values.append(value.strip())
        elif isinstance(value, Mapping):
            for child in value.values():
                values.extend(FigureSourceDataValidator._iter_string_values(child))
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                values.extend(FigureSourceDataValidator._iter_string_values(child))
        return values

    @classmethod
    def _registered_figure_paths(
        cls,
        *,
        step: AnalysisStep,
        step_summary: Mapping[str, Any],
        out_dir: Path,
    ) -> Dict[tuple[str, str], List[Path]]:
        """Resolve exact planned figure roles to their registered files.

        A directory-wide contract/source-data scan is insufficient: an honest
        decoy bundle must never authenticate a different file registered under
        the Planner's figure role.  Exact typed registry keys are authoritative;
        a same-name file fallback is retained for legacy summaries.
        """

        declared = {
            parsed
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        }
        resolved: Dict[tuple[str, str], List[Path]] = {
            product: [] for product in declared
        }

        def candidate_paths(value: Any) -> List[Path]:
            paths: List[Path] = []
            for raw_path in cls._iter_string_values(value):
                suffix = Path(raw_path).suffix.lower()
                if suffix not in {".png", ".svg", ".pdf", ".tif", ".tiff"}:
                    continue
                relative = Path(raw_path)
                candidate = relative if relative.is_absolute() else out_dir / relative
                paths.append(candidate)
            return paths

        for container_key in ("output_files", "outputs"):
            container = step_summary.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for raw_role, value in container.items():
                role = typed_product(raw_role)
                if role in declared:
                    resolved[role].extend(candidate_paths(value))

        legacy_paths: List[Path] = []
        for key in ("figure_files", "figure_file", "figure_path"):
            legacy_paths.extend(candidate_paths(step_summary.get(key)))
        for product in declared:
            if resolved[product]:
                continue
            resolved[product].extend(
                path for path in legacy_paths if path.stem == product[1]
            )

        return {
            product: list(dict.fromkeys(paths)) for product, paths in resolved.items()
        }

    @classmethod
    def _registered_same_step_tables(
        cls,
        *,
        step: AnalysisStep,
        step_summary: Mapping[str, Any],
        out_dir: Path,
        run_dir: Path,
        excluded_paths: Sequence[Path] = (),
    ) -> Dict[Path, str]:
        """Return distinct planned tabular outputs available to a mixed step.

        A figure's own contract-declared source CSV is never eligible as the
        upstream value source.  Otherwise the writable output could register the
        same file as both ``table:*`` and ``*source_data.csv`` and authenticate
        arbitrary values by comparing the file with itself.
        """

        declared = {
            parsed: f"{parsed[0]}:{parsed[1]}"
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None
            and parsed[0] in {"artifact", "dataset", "table"}
        }
        excluded = {path.resolve() for path in excluded_paths if path.exists()}
        result_families = cls._planned_result_families(step)
        tables: Dict[Path, str] = {}
        for container_key in ("output_files", "outputs"):
            container = step_summary.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for raw_role, value in container.items():
                role = typed_product(raw_role)
                if role not in declared:
                    continue
                for raw_path in cls._iter_string_values(value):
                    if Path(raw_path).suffix.lower() not in cls._TABULAR_SUFFIXES:
                        continue
                    relative = Path(raw_path)
                    candidate = (
                        relative if relative.is_absolute() else out_dir / relative
                    )
                    if (
                        cls._safe_regular_run_file(candidate, run_dir=run_dir)
                        and candidate.parent.resolve() == out_dir.resolve()
                        and candidate.resolve() not in excluded
                    ):
                        try:
                            frame = cls._read_tabular(candidate)
                        except Exception:
                            continue
                        product = declared[role]
                        if (
                            not any(
                                cls._source_supports_result_family(
                                    product=product,
                                    frame=frame,
                                    family=family,
                                )
                                for family in result_families
                            )
                            and result_families
                        ):
                            continue
                        tables[candidate.resolve()] = product
        return tables

    @classmethod
    def _declared_bundle_source_tables(
        cls,
        *,
        step: AnalysisStep,
        step_summary: Mapping[str, Any],
        out_dir: Path,
        run_dir: Path,
        resolved_input_bindings: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> tuple[Optional[Dict[Path, Set[str]]], List[ValidationFinding]]:
        """Bind each planned numeric figure to its exact local source bundle.

        ``None`` means the step has no typed planned figure and the legacy
        source-data scan may be used.  A returned mapping is authoritative: it
        binds each local source table to the exact planned figure product(s)
        whose contract declared it, so one honest family cannot launder another.
        """

        planned = {
            parsed: str(raw)
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        }
        if not planned:
            return None, []

        declared_input_kinds = {
            parsed[0]
            for raw in (step.inputs or [])
            if (parsed := typed_product(raw)) is not None
        }
        declared_input_kinds.update(
            str(binding.get("declared_kind") or "").strip().lower()
            for binding in (resolved_input_bindings or {}).values()
            if isinstance(binding, Mapping)
        )
        declared_result_kinds = {
            parsed[0]
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None and parsed[0] != "figure"
        }
        has_data_input = bool(
            (declared_input_kinds | declared_result_kinds)
            & {"artifact", "dataset", "model", "statistic", "table"}
        )
        has_untyped_input = any(
            typed_product(raw) is None for raw in (step.inputs or [])
        )
        planned_result_families = cls._planned_result_families(step)
        method_head = cls._normalised_method_head(step.method)
        compute_and_render = (
            bool(planned) and method_head not in cls._PURE_RENDER_METHODS
        )
        host_requires_source = bool(
            has_data_input
            or has_untyped_input
            or planned_result_families
            or compute_and_render
        )
        registered = cls._registered_figure_paths(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
        )
        source_tables: Dict[Path, Set[str]] = {}
        findings: List[ValidationFinding] = []

        for product, raw_product in planned.items():
            paths = registered.get(product, [])
            if not paths:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' did not bind planned "
                            f"figure {raw_product!r} to an exact output file."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "reason": "missing_declared_figure_registration",
                        },
                    )
                )
                continue

            stems = {path.stem for path in paths}
            if len(stems) != 1:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Planned figure {raw_product!r} is registered to "
                            "multiple unrelated file stems; one figure bundle "
                            "must share a single stem across export formats."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stems": sorted(stems),
                            "reason": "ambiguous_declared_figure_bundle",
                        },
                    )
                )
                continue
            stem = next(iter(stems))
            contract_path = out_dir / f"{stem}.figure_contract.json"
            contract: Any = None
            contract_is_safe = (
                cls._safe_regular_run_file(contract_path, run_dir=run_dir)
                and contract_path.parent.resolve() == out_dir.resolve()
            )
            if contract_is_safe:
                try:
                    contract = json.loads(contract_path.read_text(encoding="utf-8"))
                except Exception:
                    contract = None
            panels = contract.get("panels") if isinstance(contract, Mapping) else []
            result_like = bool(
                isinstance(contract, dict)
                and FigureContractQualityValidator._is_result_like_contract(
                    contract,
                    panels if isinstance(panels, list) else [],
                )
            )
            unsafe_exports = [
                path.name
                for path in paths
                if not cls._safe_regular_run_file(path, run_dir=run_dir)
                or path.parent.resolve() != out_dir.resolve()
            ]
            if unsafe_exports:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Planned figure bundle '{stem}' contains an unsafe "
                            "or missing registered export."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "unsafe_figure_exports": sorted(unsafe_exports),
                            "reason": "unsafe_declared_figure_path",
                        },
                    )
                )
                continue
            if not isinstance(contract, Mapping) or not contract_is_safe:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Planned figure bundle '{stem}' has no readable, "
                            "same-stem .figure_contract.json file."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stem": stem,
                            "reason": "missing_figure_contract",
                        },
                    )
                )
                continue
            raw_figure_id = str(contract.get("figure_id") or "").strip()
            safe_figure_id = re.fullmatch(
                r"(?:figure:)?([A-Za-z0-9][A-Za-z0-9_.-]*)",
                raw_figure_id,
                flags=re.IGNORECASE,
            )
            figure_id = (
                cls._normalise(safe_figure_id.group(1))
                if safe_figure_id is not None
                else ""
            )
            if not figure_id or figure_id != cls._normalise(stem):
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure contract '{contract_path.name}' identifies "
                            "a different figure than its registered export."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stem": stem,
                            "contract_figure_id": contract.get("figure_id"),
                            "reason": "figure_contract_export_mismatch",
                        },
                    )
                )
                continue

            requires_source = host_requires_source or result_like
            if not requires_source:
                continue

            declared_sources = contract.get("source_data")
            raw_source_names = (
                [declared_sources]
                if isinstance(declared_sources, str)
                else (
                    list(declared_sources)
                    if isinstance(declared_sources, (list, tuple, set))
                    else ([] if declared_sources is None else [declared_sources])
                )
            )
            invalid_source_descriptors = [
                {
                    "index": index,
                    "value_type": type(value).__name__,
                }
                for index, value in enumerate(raw_source_names)
                if not isinstance(value, str)
            ]
            if invalid_source_descriptors:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure contract '{contract_path.name}' must declare "
                            "source_data as local CSV basename strings."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "invalid_source_data_descriptors": (
                                invalid_source_descriptors
                            ),
                            "reason": "invalid_contract_source_data",
                        },
                    )
                )
                continue
            source_names = [str(value) for value in raw_source_names]
            local_sources: List[Path] = []
            unsafe_sources: List[str] = []
            for value in source_names:
                name = str(value or "").strip()
                if not name or Path(name).suffix.lower() != ".csv":
                    continue
                if Path(name).name != name or "/" in name or "\\" in name:
                    unsafe_sources.append(name)
                    continue
                source_path = out_dir / name
                if (
                    not cls._safe_regular_run_file(source_path, run_dir=run_dir)
                    or source_path.parent.resolve() != out_dir.resolve()
                ):
                    unsafe_sources.append(name)
                    continue
                local_sources.append(source_path)
            if unsafe_sources:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Figure contract '{contract_path.name}' declares "
                            "unsafe or missing local source-data files."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "unsafe_source_data": sorted(set(unsafe_sources)),
                            "reason": "invalid_contract_source_data",
                        },
                    )
                )
                continue
            if not local_sources:
                findings.append(
                    ValidationFinding(
                        validator=cls.name,
                        severity="error",
                        message=(
                            f"Result figure bundle '{stem}' has no local CSV "
                            "declared in contract.source_data."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "figure_product": raw_product,
                            "figure_stem": stem,
                            "reason": "missing_source_data",
                        },
                    )
                )
                continue
            canonical_figure = f"{product[0]}:{product[1]}"
            for source_path in local_sources:
                source_tables.setdefault(source_path.resolve(), set()).add(
                    canonical_figure
                )

        return source_tables, findings

    def audit(
        self,
        *,
        step: AnalysisStep,
        out_dir: Path,
        run_dir: Path,
        step_summary: Dict[str, Any],
        completed_step_records: Optional[Sequence[Dict[str, Any]]] = None,
        resolved_input_bindings: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> List[ValidationFinding]:
        if not self._is_rendering_step(step=step, step_summary=step_summary):
            return []
        figure_products = [
            f"{parsed[0]}:{parsed[1]}"
            for raw in (step.expected_outputs or [])
            if (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
        ]
        declared_sources, bundle_findings = self._declared_bundle_source_tables(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
            run_dir=run_dir,
            resolved_input_bindings=resolved_input_bindings,
        )
        if bundle_findings:
            return bundle_findings
        if declared_sources is None:
            source_tables = sorted(out_dir.glob(self._SOURCE_DATA_GLOB))
            source_figure_products = {
                path.resolve(): set(figure_products) for path in source_tables
            }
        else:
            source_tables = sorted(declared_sources)
            source_figure_products = {
                path.resolve(): set(products)
                for path, products in declared_sources.items()
            }
        if not source_tables:
            return []

        result_families = self._planned_result_families(step)
        same_step_tables = self._registered_same_step_tables(
            step=step,
            step_summary=step_summary,
            out_dir=out_dir,
            run_dir=run_dir,
            excluded_paths=source_tables,
        )
        same_step_statistics: Dict[str, tuple[str, float]] = {}
        for raw_output in step.expected_outputs or []:
            product = typed_product(raw_output)
            if product is None or product[0] != "statistic":
                continue
            canonical = f"{product[0]}:{product[1]}"
            if result_families and not any(
                self._source_supports_result_family(
                    product=canonical,
                    family=family,
                )
                for family in result_families
            ):
                continue
            value = self._extract_statistic_value(step_summary, product[1])
            if value is not None:
                same_step_statistics[f"same_step:{canonical}"] = (
                    product[1],
                    value,
                )

        bound_input_bindings: Dict[str, Mapping[str, Any]] = {}
        if resolved_input_bindings is None:
            upstream_step_ids = self._upstream_step_ids(
                step=step,
                step_summary=step_summary,
            )
            if same_step_tables or same_step_statistics:
                upstream_step_ids.add(str(step.step_id))
        else:
            upstream_step_ids: Set[str] = set()
            invalid_bindings: List[str] = []
            for raw_input, binding in resolved_input_bindings.items():
                if not isinstance(binding, Mapping):
                    invalid_bindings.append(str(raw_input))
                    continue
                declared_kind = str(binding.get("declared_kind") or "").strip().lower()
                producer_id = str(binding.get("produced_by_step") or "").strip()
                evidence_id = str(binding.get("evidence_id") or "").strip()
                digest = str(binding.get("sha256") or "").strip()
                product = str(binding.get("product") or "").strip()
                parsed_input = typed_product(raw_input)
                if (
                    declared_kind
                    not in {"artifact", "dataset", "model", "statistic", "table"}
                    or parsed_input != (declared_kind, self._normalise(product))
                    or not self._safe_step_id(producer_id)
                    or not evidence_id
                    or not product
                    or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
                ):
                    invalid_bindings.append(str(raw_input))
                    continue
                bound_input_bindings[str(raw_input)] = binding
                upstream_step_ids.add(producer_id)
            if same_step_tables or same_step_statistics:
                upstream_step_ids.add(str(step.step_id))
            if invalid_bindings:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' has invalid "
                            "host-resolved typed input bindings; source-data "
                            "provenance cannot be authenticated."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "invalid_resolved_inputs": sorted(invalid_bindings),
                            "reason": "invalid_resolved_input_binding",
                        },
                    )
                ]
            declared_upstream_ids = self._explicit_upstream_step_ids(step_summary)
            contradictory_ids = declared_upstream_ids - upstream_step_ids
            if contradictory_ids:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' reports upstream "
                            "steps that disagree with its host-resolved typed "
                            "bindings."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "declared_upstream_step_ids": sorted(declared_upstream_ids),
                            "resolved_upstream_step_ids": sorted(upstream_step_ids),
                            "reason": "resolved_upstream_binding_mismatch",
                        },
                    )
                ]
        unsafe_step_ids = sorted(
            step_id for step_id in upstream_step_ids if not self._safe_step_id(step_id)
        )
        if unsafe_step_ids:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' declared unsafe upstream "
                        "step identifiers. Upstream lineage must use plain "
                        "run-local step ids, never paths."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "unsafe_upstream_step_ids": unsafe_step_ids,
                    },
                )
            ]
        if not upstream_step_ids:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' wrote source data without "
                        "declaring any upstream step. Source-data provenance "
                        "cannot be verified without an exact run-local parent."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "source_tables": [path.name for path in source_tables],
                        "reason": "missing_upstream_step_binding",
                    },
                )
            ]
        upstream_step_ids = {
            step_id for step_id in upstream_step_ids if self._safe_step_id(step_id)
        }

        authoritative_evidence = self._authoritative_table_evidence(
            run_dir=run_dir,
            completed_step_records=completed_step_records,
        )
        authoritative_tables = (
            None
            if authoritative_evidence is None
            else {item["path"] for item in authoritative_evidence.values()}
        )
        if same_step_tables and authoritative_tables is not None:
            authoritative_tables.update(same_step_tables)

        required_table_paths: Set[Path] = set()
        required_statistics: Dict[str, tuple[str, float]] = dict(same_step_statistics)
        table_products: Dict[Path, str] = dict(same_step_tables)
        table_frames: Dict[Path, pd.DataFrame] = {}
        declared_table_aliases: Dict[str, Set[Path]] = {}
        declared_statistic_artifacts: Dict[str, Set[str]] = {}
        bound_tabular_paths: Set[Path] = set()
        unsupported_value_inputs: List[str] = []
        if resolved_input_bindings is not None:
            invalid_bound_evidence: List[str] = []
            current_records = {
                str(record.get("step_id") or "").strip(): record
                for record in current_successful_step_records(
                    completed_step_records or []
                )
            }
            for raw_input, binding in bound_input_bindings.items():
                declared_kind = str(binding.get("declared_kind") or "").strip().lower()
                evidence_id = str(binding.get("evidence_id") or "").strip()
                producer_id = str(binding.get("produced_by_step") or "").strip()
                product_name = self._normalise(binding.get("product"))
                canonical_product = f"{declared_kind}:{product_name}"
                bound_path = Path(str(binding.get("absolute_path") or ""))
                expected_sha = str(binding.get("sha256") or "").strip().lower()
                if declared_kind == "table":
                    item = (
                        authoritative_evidence.get(evidence_id)
                        if authoritative_evidence is not None
                        else None
                    )
                    if (
                        item is None
                        or item["sha256"] != expected_sha
                        or item["produced_by_step"] != producer_id
                    ):
                        invalid_bound_evidence.append(raw_input)
                        continue
                    bound_path = item["path"]
                    for alias in self._bound_artifact_aliases(
                        input_key=str(raw_input),
                        product=product_name,
                        evidence_path=item.get("evidence_path"),
                    ):
                        declared_table_aliases.setdefault(alias, set()).add(
                            bound_path.resolve()
                        )
                elif (
                    not self._safe_regular_run_file(bound_path, run_dir=run_dir)
                    or _sha256_file(bound_path) != expected_sha
                ):
                    invalid_bound_evidence.append(raw_input)
                    continue
                if declared_kind == "statistic":
                    record = current_records.get(producer_id)
                    if record is None or evidence_id not in {
                        str(item) for item in (record.get("evidence_ids") or [])
                    }:
                        invalid_bound_evidence.append(raw_input)
                        continue
                    try:
                        statistic_payload = json.loads(
                            bound_path.read_text(encoding="utf-8")
                        )
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        invalid_bound_evidence.append(raw_input)
                        continue
                    value = self._extract_statistic_value(
                        statistic_payload,
                        product_name,
                    )
                    if value is None:
                        unsupported_value_inputs.append(raw_input)
                        continue
                    if not result_families or any(
                        self._source_supports_result_family(
                            product=canonical_product,
                            family=family,
                        )
                        for family in result_families
                    ):
                        required_statistics[raw_input] = (product_name, value)
                        declared_statistic_artifacts.setdefault(
                            bound_path.name,
                            set(),
                        ).add(raw_input)
                    else:
                        unsupported_value_inputs.append(raw_input)
                    continue
                if declared_kind == "model":
                    unsupported_value_inputs.append(raw_input)
                    continue

                if bound_path.suffix.lower() not in self._TABULAR_SUFFIXES:
                    unsupported_value_inputs.append(raw_input)
                    continue
                try:
                    frame = self._read_tabular(bound_path)
                except Exception:
                    invalid_bound_evidence.append(raw_input)
                    continue
                if result_families and not any(
                    self._source_supports_result_family(
                        product=canonical_product,
                        frame=frame,
                        family=family,
                    )
                    for family in result_families
                ):
                    unsupported_value_inputs.append(raw_input)
                    continue
                resolved_path = bound_path.resolve()
                bound_tabular_paths.add(resolved_path)
                required_table_paths.add(resolved_path)
                table_products[resolved_path] = canonical_product
                table_frames[resolved_path] = frame
            if invalid_bound_evidence:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' has typed bindings "
                            "that do not resolve to current hash-verified evidence."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "invalid_resolved_inputs": sorted(invalid_bound_evidence),
                            "reason": "resolved_input_evidence_mismatch",
                        },
                    )
                ]
            bound_tabular_paths.update(same_step_tables)
            required_table_paths.update(same_step_tables)
            authoritative_tables = set(bound_tabular_paths)
        if completed_step_records is not None:
            current_parent_ids = {
                str(record.get("step_id") or "").strip()
                for record in current_successful_step_records(completed_step_records)
            }
            same_step_ids = (
                {str(step.step_id)}
                if same_step_tables or same_step_statistics
                else set()
            )
            stale_parent_ids = sorted(
                upstream_step_ids - current_parent_ids - same_step_ids
            )
            if stale_parent_ids:
                return [
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure step '{step.step_id}' cites upstream step(s) "
                            "whose latest checkpoint is not successful. Historical "
                            "outputs cannot authenticate a current figure."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "noncurrent_upstream_step_ids": stale_parent_ids,
                        },
                    )
                ]
        upstream_tables = self._upstream_tables(
            run_dir=run_dir,
            current_out_dir=out_dir,
            upstream_step_ids=upstream_step_ids,
            authoritative_tables=authoritative_tables,
        )
        for bound_path in bound_tabular_paths:
            if bound_path not in upstream_tables:
                upstream_tables.append(bound_path)
        for same_step_table in same_step_tables:
            if same_step_table not in upstream_tables:
                upstream_tables.append(same_step_table)
        if not upstream_tables and not required_statistics:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity=(
                        "error" if authoritative_tables is not None else "warning"
                    ),
                    message=(
                        f"Figure step '{step.step_id}' has no replayable, "
                        "hash-verified upstream table or statistic source for its result "
                        "figure. Model files and non-tabular artifacts cannot "
                        "authenticate plotted values by themselves."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "upstream_step_ids": sorted(upstream_step_ids),
                        "source_tables": [p.name for p in source_tables],
                        "unsupported_value_inputs": sorted(
                            set(unsupported_value_inputs)
                        ),
                        "reason": "non_replayable_figure_input",
                    },
                )
            ]

        findings: List[ValidationFinding] = []
        matched_table_paths: Set[Path] = set()
        matched_statistics: Set[str] = set()
        required_figure_obligations: Dict[str, Set[str]] = {
            figure: self._figure_source_obligations(
                step=step,
                figure_product=figure,
            )
            for figure in figure_products
            if self._figure_result_family(
                step=step,
                figure_product=figure,
            )
            is not None
        }
        matched_figure_obligations: Dict[str, Set[str]] = {
            figure: set() for figure in required_figure_obligations
        }

        def credit_table_source(source_path: Path, source_frame: pd.DataFrame, table_paths: Set[Path]):
            return _figure_audit__credit_table_source(source_path, source_frame, table_paths, completed_step_records=completed_step_records, matched_figure_obligations=matched_figure_obligations, required_figure_obligations=required_figure_obligations, run_dir=run_dir, self=self, source_figure_products=source_figure_products, step=step, table_frames=table_frames, table_products=table_products)

        def credit_statistic_source(source_path: Path, statistic_ids: Set[str]):
            return _figure_audit__credit_statistic_source(source_path, statistic_ids, matched_figure_obligations=matched_figure_obligations, required_figure_obligations=required_figure_obligations, required_statistics=required_statistics, self=self, source_figure_products=source_figure_products, step=step)

        for source_path in source_tables:
            if not self._safe_regular_run_file(source_path, run_dir=run_dir):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table {source_path.name} is not "
                            "a regular, non-symlink file contained by this run."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "reason": "unsafe_source_data_path",
                        },
                    )
                )
                continue
            try:
                source_df = pd.read_csv(source_path)
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=f"Could not read figure source-data table {source_path.name}: {exc}",
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "reason": "source_data_read_failed",
                        },
                    )
                )
                continue
            if source_df.empty:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table {source_path.name} is "
                            "empty and cannot authenticate a rendered result."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "reason": "source_data_empty",
                        },
                    )
                )
                continue
            unsafe_declared_tables = sorted(
                {
                    str(item).strip()
                    for item in source_df.get(
                        "source_table", pd.Series(dtype=object)
                    ).dropna()
                    if str(item).strip()
                    and not self._safe_declared_table_name(str(item).strip())
                }
            )
            if unsafe_declared_tables:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table '{source_path.name}' "
                            "declares an unsafe source_table path. The claim must "
                            "be one plain upstream filename."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "unsafe_declared_source_tables": unsafe_declared_tables,
                        },
                    )
                )
                continue
            unsafe_declared_steps = sorted(
                {
                    str(item).strip()
                    for item in source_df.get(
                        "source_step_id", pd.Series(dtype=object)
                    ).dropna()
                    if str(item).strip() and not self._safe_step_id(item)
                }
            )
            if unsafe_declared_steps:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table '{source_path.name}' "
                            "declares an unsafe source_step_id."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "unsafe_declared_source_step_ids": unsafe_declared_steps,
                        },
                    )
                )
                continue
            findings.extend(
                self._percentage_count_consistency_findings(
                    source_df=source_df,
                    source_path=source_path,
                    step_id=step.step_id,
                )
            )
            findings.extend(
                self._structured_sensitivity_trace_findings(
                    source_df=source_df,
                    source_path=source_path,
                    step_id=step.step_id,
                    run_dir=run_dir,
                    upstream_step_ids=upstream_step_ids,
                )
            )
            source_statistic_matches: Set[str] = set()
            for statistic_id, (
                product_name,
                expected_value,
            ) in required_statistics.items():
                if self._source_contains_statistic(
                    source_df,
                    product_name=product_name,
                    expected=expected_value,
                ):
                    source_statistic_matches.add(statistic_id)
            source_matched_table_paths: Set[Path] = set()

            def finalize_source_match() -> bool:
                if source_matched_table_paths:
                    matched_table_paths.update(source_matched_table_paths)
                    credit_table_source(
                        source_path,
                        source_df,
                        source_matched_table_paths,
                    )
                    if source_statistic_matches:
                        matched_statistics.update(source_statistic_matches)
                        credit_statistic_source(source_path, source_statistic_matches)
                    return True
                if not source_statistic_matches:
                    return False
                statistic_issue = self._statistic_payload_issue(
                    source_df,
                    required_statistics=required_statistics,
                )
                if statistic_issue is not None:
                    findings.append(
                        ValidationFinding(
                            validator=self.name,
                            severity="error",
                            message=(
                                f"Figure source-data table '{source_path.name}' "
                                "contains numeric result payload that is not "
                                "bound to a verified statistic."
                            ),
                            detail={
                                "step_id": step.step_id,
                                "source_table": source_path.name,
                                **statistic_issue,
                            },
                        )
                    )
                    return False
                matched_statistics.update(source_statistic_matches)
                credit_statistic_source(source_path, source_statistic_matches)
                return True

            # A faithful figure may bind a table produced by any exact typed
            # parent, not merely a step-id naming convention.  ``source_table``
            # remains a binding claim: when present, only that basename (and,
            # when supplied, that exact source_step_id) may authenticate rows.
            declared_tables = self._resolve_declared_source_tables_across_run(
                run_dir=run_dir,
                source_df=source_df,
                current_out_dir=out_dir,
                authoritative_tables=authoritative_tables,
                allowed_step_ids=(
                    upstream_step_ids if resolved_input_bindings is not None else None
                ),
            )
            candidate_tables = list(upstream_tables)
            for path in declared_tables:
                if path not in candidate_tables:
                    candidate_tables.append(path)
            declared_row_names = pd.Series("", index=source_df.index, dtype=str)
            if "source_table" in source_df.columns:
                declared_row_names = source_df["source_table"].map(
                    lambda item: (
                        Path(str(item)).name
                        if pd.notna(item) and str(item).strip()
                        else ""
                    )
                )
            declared_names = {
                name for name in declared_row_names.astype(str) if name.strip()
            }
            comparisons: List[Dict[str, Any]] = []
            ordered_upstream_tables: List[Path] = []
            if declared_names:
                # ``source_table`` is a binding provenance claim, not merely a
                # routing hint.  Validate each row group only against tables with
                # the declared basename; an unrelated sibling table must never
                # launder a forged declaration by happening to share keys/values.
                blank_rows = declared_row_names.eq("")
                if blank_rows.any():
                    comparisons.append(
                        {
                            "ok": False,
                            "reason": "missing_declared_source_table",
                            "n_rows": int(blank_rows.sum()),
                            "message": (
                                "source_table is declared for this figure, but "
                                f"{int(blank_rows.sum())} source-data row(s) do "
                                "not name their upstream table"
                            ),
                        }
                    )
                for declared_name in sorted(declared_names):
                    group_df = source_df.loc[
                        declared_row_names.eq(declared_name)
                    ].copy()
                    declared_statistic_ids = declared_statistic_artifacts.get(
                        declared_name,
                        set(),
                    )
                    if declared_statistic_ids and all(
                        statistic_id in source_statistic_matches
                        for statistic_id in declared_statistic_ids
                    ):
                        # A source row may truthfully name the exact JSON file
                        # backing a digest-verified typed statistic.  It is not
                        # an upstream *table*, so do not force that provenance
                        # claim through the tabular basename resolver.  The
                        # statistic payload/value checks below still have to
                        # pass before the source receives credit.
                        continue
                    group_tables = sorted(
                        {
                            path
                            for path in candidate_tables
                            if path.name == declared_name
                            or path.resolve()
                            in declared_table_aliases.get(declared_name, set())
                        },
                        key=str,
                    )
                    declared_parent_step: Optional[str] = None
                    if "source_step_id" in group_df.columns:
                        declared_step_values = group_df["source_step_id"].map(
                            lambda item: str(item).strip() if pd.notna(item) else ""
                        )
                        declared_parent_steps = {
                            item for item in declared_step_values if item
                        }
                        if (
                            len(declared_parent_steps) != 1
                            or declared_step_values.eq("").any()
                        ):
                            comparisons.append(
                                {
                                    "ok": False,
                                    "reason": "ambiguous_declared_source_step",
                                    "declared_source_table": declared_name,
                                    "declared_source_step_ids": sorted(
                                        declared_parent_steps
                                    ),
                                    "message": (
                                        f"declared source table {declared_name} "
                                        "must identify exactly one source_step_id"
                                    ),
                                }
                            )
                            continue
                        declared_parent_step = next(iter(declared_parent_steps))
                        group_tables = [
                            path
                            for path in group_tables
                            if self._table_step_id(path, run_dir=run_dir)
                            == declared_parent_step
                        ]
                    elif len(group_tables) > 1:
                        comparisons.append(
                            {
                                "ok": False,
                                "reason": "ambiguous_declared_source_table_lineage",
                                "declared_source_table": declared_name,
                                "candidate_source_steps": sorted(
                                    {
                                        self._table_step_id(path, run_dir=run_dir)
                                        for path in group_tables
                                    }
                                ),
                                "message": (
                                    f"declared source table {declared_name} exists "
                                    "in multiple upstream steps; source_step_id is "
                                    "required to bind exact lineage"
                                ),
                            }
                        )
                        continue
                    ordered_upstream_tables.extend(group_tables)
                    if not group_tables:
                        comparisons.append(
                            {
                                "ok": False,
                                "reason": "declared_source_table_not_found",
                                "declared_source_table": declared_name,
                                "message": (
                                    f"declared source table {declared_name} was "
                                    "not found in an upstream step"
                                ),
                            }
                        )
                        continue
                    group_comparison_pairs = [
                        (
                            upstream_path,
                            self._compare_source_to_upstream(
                                source_df=group_df,
                                source_path=source_path,
                                upstream_path=upstream_path,
                            ),
                        )
                        for upstream_path in group_tables
                    ]
                    group_comparisons = [item for _, item in group_comparison_pairs]
                    comparisons.extend(group_comparisons)
                    if any(item.get("ok") for item in group_comparisons):
                        source_matched_table_paths.update(
                            path.resolve()
                            for path, item in group_comparison_pairs
                            if item.get("ok")
                        )
                        # Keep only failures from groups that have no matching
                        # declared parent.  A duplicate basename in another step
                        # may legitimately be the referenced parent.
                        comparisons = [
                            item
                            for item in comparisons
                            if item not in group_comparisons or item.get("ok")
                        ]
                failed_comparisons = [
                    item for item in comparisons if not item.get("ok")
                ]
                if not failed_comparisons:
                    if finalize_source_match():
                        continue
                comparisons = failed_comparisons
            else:
                ordered_upstream_tables = self._prioritize_declared_source_tables(
                    source_df=source_df,
                    upstream_tables=candidate_tables,
                )
                comparison_pairs = [
                    (
                        upstream_path,
                        self._compare_source_to_upstream(
                            source_df=source_df,
                            source_path=source_path,
                            upstream_path=upstream_path,
                        ),
                    )
                    for upstream_path in ordered_upstream_tables
                ]
                comparisons = [item for _, item in comparison_pairs]
                successful_paths = {
                    path.resolve() for path, item in comparison_pairs if item.get("ok")
                }
                if successful_paths:
                    source_matched_table_paths.update(successful_paths)
                if finalize_source_match():
                    continue
            if not comparisons:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Figure source-data table '{source_path.name}' does "
                            "not reproduce any bound table or statistic value."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "source_table": source_path.name,
                            "required_statistics": sorted(required_statistics),
                            "reason": "no_verifiable_figure_values",
                        },
                    )
                )
                continue
            actionable = [
                item for item in comparisons if item.get("reason") != "no_shared_key"
            ]
            best = actionable[0] if actionable else comparisons[0]
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure source-data table '{source_path.name}' is not a "
                        "traceable subset of the declared upstream table(s); "
                        f"{best.get('message', 'no matching upstream rows found')}."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "source_table": source_path.name,
                        "upstream_step_ids": sorted(upstream_step_ids),
                        "candidate_upstream_tables": [
                            (
                                str(p.relative_to(run_dir))
                                if p.is_relative_to(run_dir)
                                else str(p)
                            )
                            for p in ordered_upstream_tables
                        ],
                        "best_mismatch": best,
                    },
                )
            )
        missing_table_paths = required_table_paths - matched_table_paths
        missing_statistics = set(required_statistics) - matched_statistics
        if missing_table_paths or missing_statistics:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' source-data bundle does "
                        "not cover every bound result source. Each typed parent "
                        "must be independently value-verified."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "missing_bound_tables": sorted(
                            path.name for path in missing_table_paths
                        ),
                        "missing_bound_statistics": sorted(missing_statistics),
                        "reason": "incomplete_source_lineage_coverage",
                    },
                )
            )
        missing_figure_sources = {
            figure: {
                "declared_sources": sorted(
                    path.name
                    for path, products in source_figure_products.items()
                    if figure in products
                ),
                "missing_obligations": sorted(
                    required_obligations - matched_figure_obligations.get(figure, set())
                ),
            }
            for figure, required_obligations in required_figure_obligations.items()
            if not required_obligations.issubset(
                matched_figure_obligations.get(figure, set())
            )
        }
        if missing_figure_sources:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' has a planned result "
                        "figure whose own source bundle is not backed by a "
                        "semantically compatible, value-verified product."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "missing_figure_sources": missing_figure_sources,
                        "reason": "missing_figure_family_source",
                    },
                )
            )
        return findings

    @classmethod
    def _percentage_count_consistency_findings(
        cls,
        *,
        source_df: pd.DataFrame,
        source_path: Path,
        step_id: str,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        for pct_col, count_col, total_col in cls._PCT_COUNT_RULES:
            if not {pct_col, count_col, total_col} <= set(source_df.columns):
                continue
            pct = pd.to_numeric(source_df[pct_col], errors="coerce")
            count = pd.to_numeric(source_df[count_col], errors="coerce")
            total = pd.to_numeric(source_df[total_col], errors="coerce")
            valid = total > 0
            if not valid.any():
                continue
            expected = 100.0 * count[valid] / total[valid]
            observed = pct[valid]
            diff = (observed - expected).abs()
            bad = diff[(diff > 0.05) & ~(observed.isna() & expected.isna())]
            if bad.empty:
                continue
            idx = int(bad.index[0])
            findings.append(
                ValidationFinding(
                    validator=cls.name,
                    severity="error",
                    message=(
                        f"Figure source-data table '{source_path.name}' has "
                        f"inconsistent percentage/count columns: {pct_col} "
                        f"does not match 100*{count_col}/{total_col}."
                    ),
                    detail={
                        "step_id": step_id,
                        "source_table": source_path.name,
                        "pct_column": pct_col,
                        "count_column": count_col,
                        "total_column": total_col,
                        "row_index": idx,
                        "observed_pct": (
                            None if pd.isna(pct.loc[idx]) else float(pct.loc[idx])
                        ),
                        "expected_pct": (
                            None
                            if pd.isna(expected.loc[idx])
                            else float(expected.loc[idx])
                        ),
                        "abs_diff": float(bad.loc[idx]),
                    },
                )
            )
        return findings

    @classmethod
    def _structured_sensitivity_trace_findings(
        cls,
        *,
        source_df: pd.DataFrame,
        source_path: Path,
        step_id: str,
        run_dir: Path,
        upstream_step_ids: Set[str],
    ) -> List[ValidationFinding]:
        """Require fitted sensitivity rows to identify their exact model.

        Simple legacy sensitivity tables remain valid.  The stronger contract
        activates only when the parent step declares a full
        ``robustness_model_contracts`` grid; in that case a scalar plot row must
        say which ``spec_id x model_id`` contract and coefficient term supplied
        the estimate.
        """

        required_shape = {
            "spec_id",
            "effect_scale",
            "point_estimate",
            "ci_low",
            "ci_high",
        }
        if not required_shape <= set(source_df.columns):
            return []

        parent_payloads: List[tuple[str, Path, Dict[str, Any]]] = []
        for parent_step_id in sorted(upstream_step_ids):
            outputs_dir = run_dir / "steps" / parent_step_id / "outputs"
            summary_path = outputs_dir / "step_summary.json"
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            contracts = payload.get("robustness_model_contracts")
            if isinstance(contracts, list) and contracts:
                parent_payloads.append((parent_step_id, outputs_dir, payload))
        if not parent_payloads:
            return []

        point = pd.to_numeric(source_df["point_estimate"], errors="coerce")
        low = pd.to_numeric(source_df["ci_low"], errors="coerce")
        high = pd.to_numeric(source_df["ci_high"], errors="coerce")
        estimated = point.notna() & low.notna() & high.notna()
        if "converged" in source_df.columns:
            estimated &= source_df["converged"].map(
                lambda value: cls._normalise(value) in {"true", "1", "yes"}
            )
        if "independent_variant" in source_df.columns:
            estimated &= ~source_df["independent_variant"].map(
                lambda value: cls._normalise(value) in {"false", "0", "no"}
            )
        rows = source_df.loc[estimated].copy()
        if rows.empty:
            return []

        required_trace = {
            "model_id",
            "event_n",
            "exposure_expression",
            "analysis_set",
            "fit_method",
            "coefficient_source_table",
            "coefficient_term",
            "model_contract_source",
        }
        missing_columns = sorted(required_trace - set(rows.columns))
        issues: List[Dict[str, Any]] = []
        if missing_columns:
            issues.append(
                {
                    "issue": "missing_structured_sensitivity_trace_columns",
                    "columns": missing_columns,
                }
            )
        else:
            parent_step_id, outputs_dir, parent = parent_payloads[0]
            primary_model_id = str(parent.get("primary_model_id") or "")
            all_contracts: List[Dict[str, Any]] = []
            for item in parent.get("model_contracts") or []:
                if not isinstance(item, dict):
                    continue
                contract = dict(item)
                if str(contract.get("model_id") or "") == primary_model_id:
                    contract["spec_id"] = "primary"
                    all_contracts.append(contract)
            all_contracts.extend(
                dict(item)
                for item in parent.get("robustness_model_contracts") or []
                if isinstance(item, dict)
            )
            coefficient_cache: Dict[str, Optional[pd.DataFrame]] = {}
            for row_index, row in rows.iterrows():
                spec_id = str(row.get("spec_id") or "")
                model_id = str(row.get("model_id") or "")
                label = f"{spec_id}:{model_id or '<missing>'}"
                blank_fields = [
                    field
                    for field in required_trace
                    if pd.isna(row.get(field)) or not str(row.get(field)).strip()
                ]
                if blank_fields:
                    issues.append(
                        {
                            "row": label,
                            "row_index": int(row_index),
                            "issue": "blank_structured_sensitivity_trace",
                            "fields": sorted(blank_fields),
                        }
                    )
                    continue
                matched = [
                    item
                    for item in all_contracts
                    if str(item.get("spec_id") or "") == spec_id
                    and str(item.get("model_id") or "") == model_id
                ]
                if len(matched) != 1:
                    issues.append(
                        {
                            "row": label,
                            "issue": "ambiguous_model_contract_trace",
                            "matches": len(matched),
                        }
                    )
                    continue
                contract = matched[0]
                for source_field, contract_field in (
                    ("modeled_analytic_n", "n"),
                    ("event_n", "event_n"),
                ):
                    expected = cls._as_float(contract.get(contract_field))
                    reported = cls._as_float(row.get(source_field))
                    if expected is not None and reported != expected:
                        issues.append(
                            {
                                "row": label,
                                "issue": f"{source_field}_contract_mismatch",
                                "expected": expected,
                                "reported": reported,
                            }
                        )
                for field in (
                    "exposure_expression",
                    "analysis_set",
                    "fit_method",
                ):
                    if cls._normalise(row.get(field)) != cls._normalise(
                        contract.get(field)
                    ):
                        issues.append(
                            {
                                "row": label,
                                "issue": f"{field}_contract_mismatch",
                                "expected": contract.get(field),
                                "reported": row.get(field),
                            }
                        )

                coefficient_name = Path(
                    str(row.get("coefficient_source_table") or "")
                ).name
                if coefficient_name not in coefficient_cache:
                    coefficient_path = outputs_dir / coefficient_name
                    try:
                        coefficient_cache[coefficient_name] = pd.read_csv(
                            coefficient_path, float_precision="round_trip"
                        )
                    except Exception:
                        coefficient_cache[coefficient_name] = None
                coefficients = coefficient_cache[coefficient_name]
                if coefficients is None:
                    issues.append(
                        {
                            "row": label,
                            "issue": "coefficient_source_unreadable",
                            "source": coefficient_name,
                        }
                    )
                    continue
                coefficient_match = coefficients[
                    coefficients.get("model_id", pd.Series(dtype=str))
                    .astype(str)
                    .eq(model_id)
                ]
                if "spec_id" in coefficients.columns:
                    coefficient_match = coefficient_match[
                        coefficient_match["spec_id"].astype(str).eq(spec_id)
                    ]
                if "term" in coefficients.columns:
                    coefficient_match = coefficient_match[
                        coefficient_match["term"]
                        .astype(str)
                        .eq(str(row.get("coefficient_term") or ""))
                    ]
                if len(coefficient_match) != 1:
                    issues.append(
                        {
                            "row": label,
                            "issue": "ambiguous_coefficient_trace",
                            "matches": int(len(coefficient_match)),
                            "source": coefficient_name,
                        }
                    )

            if len(parent_payloads) > 1:
                issues.append(
                    {
                        "issue": "multiple_structured_sensitivity_parents",
                        "parents": [item[0] for item in parent_payloads],
                    }
                )

        if not issues:
            return []
        return [
            ValidationFinding(
                validator=cls.name,
                severity="error",
                message=(
                    f"Figure source-data table '{source_path.name}' does not "
                    "preserve the parent step's structured sensitivity-model trace."
                ),
                detail={
                    "step_id": step_id,
                    "source_table": source_path.name,
                    "issues": issues[:50],
                },
            )
        ]

    @classmethod
    def _is_rendering_step(
        cls, *, step: AnalysisStep, step_summary: Dict[str, Any]
    ) -> bool:
        if bool(
            (step_summary or {}).get("rendering_only")
            or (step_summary or {}).get("render_only")
        ):
            return True
        if any(
            (parsed := typed_product(raw)) is not None and parsed[0] == "figure"
            for raw in (step.expected_outputs or [])
        ):
            return True
        method = cls._normalise(step.method)
        if method in {
            "chart_generation",
            "figure",
            "figure_generation",
            "plotting",
            "publication_figure",
            "publication_figure_generation",
            "render_figure",
            "visualisation",
            "visualization",
        }:
            return True
        return any(
            Path(value).suffix.lower() in {".png", ".svg", ".pdf", ".tif", ".tiff"}
            for value in cls._iter_string_values(step_summary or {})
        )

    @classmethod
    def _upstream_step_ids(
        cls, *, step: AnalysisStep, step_summary: Dict[str, Any]
    ) -> Set[str]:
        found = cls._explicit_upstream_step_ids(step_summary)

        text = f"{step.intent}\n{step.method}\n{json.dumps(step_summary or {}, default=str)}"
        for match in re.finditer(r"\bstep\s*['\"]([A-Za-z0-9_.:-]+)['\"]", text):
            candidate = match.group(1).strip()
            if candidate and candidate != step.step_id:
                found.add(candidate)

        step_id = str(step.step_id)
        for suffix in (
            "_figure",
            "_publication_figure",
            "_figure_generation",
            "_render_figure",
        ):
            if step_id.endswith(suffix) and len(step_id) > len(suffix):
                found.add(step_id[: -len(suffix)])
        return found

    @classmethod
    def _explicit_upstream_step_ids(cls, step_summary: Mapping[str, Any]) -> Set[str]:
        """Return structured producer claims without prose/name inference."""

        found: Set[str] = set()
        for key in (
            "upstream_step_id",
            "source_step_id",
            "producer_step_id",
        ):
            value = (step_summary or {}).get(key)
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
        for key in (
            "upstream_step_ids",
            "source_step_ids",
            "producer_step_ids",
        ):
            value = (step_summary or {}).get(key)
            if isinstance(value, (list, tuple, set)):
                found.update(str(item).strip() for item in value if str(item).strip())
        return found

    @staticmethod
    def _safe_step_id(step_id: Any) -> bool:
        return bool(
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*",
                str(step_id or "").strip(),
            )
        )

    @staticmethod
    def _safe_declared_table_name(value: Any) -> bool:
        text = str(value or "").strip()
        return bool(
            text
            and text not in {".", ".."}
            and Path(text).name == text
            and "/" not in text
            and "\\" not in text
        )

    @staticmethod
    def _bound_artifact_aliases(
        *,
        input_key: str,
        product: str,
        evidence_path: Any,
    ) -> Set[str]:
        """The names one bound table answers to besides its own filename.

        The host publishes the same artifact under four names: the typed input
        key the step declared it as (``table:cohort_flow``), the typed product
        id inside that key (``cohort_flow``), the file the producing step wrote
        (``cohort_flow.csv``) and the evidence copy taken of it.  A figure that
        names its parent with any of them is naming the same digest-verified
        bytes, so all four must resolve; resolving only some makes the verdict
        depend on which spelling the producer happened to pick, which is not a
        property of the figure.  Measured on 2026-07-29: a source-data bundle
        that reconciled row for row against both of its parents was refused as
        ``declared source table event_timing_audit was not found in an upstream
        step`` -- the product id was the one name that did not resolve.

        The written filename is deliberately *not* returned here.  The caller
        already matches a candidate whose ``name`` equals the declared name --
        it has to, because a legacy run has no bindings and no aliases at all --
        so repeating it would be a line that can never decide anything.

        This is vocabulary, not authority.  Aliases *filter* tables the step
        already bound, so none can reach a table the step did not bind; a name
        resolving to more than one artifact is still refused as ambiguous
        lineage; and the rows behind a resolved name are still compared value by
        value against the parent's bytes.  Derived spellings stay out for the
        same reason: the file stem would resolve the measured case (product id
        and stem are the same word there) while also admitting names the host
        never writes, since the audit runner maps product ``measurement_audit``
        onto ``missingness_measurement_audit.csv``.
        """

        names = {
            str(input_key or "").strip(),
            str(product or "").strip(),
        }
        if isinstance(evidence_path, Path):
            names.add(evidence_path.name)
        return {name for name in names if name}

    @classmethod
    def _table_step_id(cls, path: Path, *, run_dir: Path) -> str:
        try:
            relative = Path(path).resolve().relative_to(Path(run_dir).resolve())
        except ValueError:
            return ""
        parts = relative.parts
        if len(parts) < 4 or parts[0] != "steps" or parts[2] != "outputs":
            return ""
        return parts[1] if cls._safe_step_id(parts[1]) else ""

    @staticmethod
    def _safe_regular_run_file(path: Path, *, run_dir: Path) -> bool:
        """Require a regular, non-symlink file contained by the run root."""

        root = Path(run_dir).resolve()
        candidate = Path(path)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            lexical_relative = candidate.absolute().relative_to(root)
        except (OSError, ValueError):
            return False
        if not candidate.is_file() or candidate.is_symlink():
            return False
        cursor = root
        for part in lexical_relative.parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                return False
        return True

    @classmethod
    def _authoritative_table_evidence(
        cls,
        *,
        run_dir: Path,
        completed_step_records: Optional[Sequence[Dict[str, Any]]],
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Resolve active table evidence ids back to immutable step outputs.

        ``None`` is the explicit legacy signal: no modern per-step authority is
        available, so old run fixtures may use the contained filesystem scan.
        A modern run returns a mapping (possibly empty); only current
        successful, hash-matching table artifacts are eligible as parents.
        """

        evidence_records = current_run_evidence_records(
            run_dir,
            per_step_records=completed_step_records,
        )
        if evidence_records is None:
            return None
        current_ids = (
            {
                str(record.get("step_id") or "").strip()
                for record in current_successful_step_records(completed_step_records)
            }
            if completed_step_records is not None
            else None
        )
        root = Path(run_dir).resolve()
        authorised: Dict[str, Dict[str, Any]] = {}
        for record in evidence_records:
            if str(record.get("kind") or "").strip().lower() != "table":
                continue
            step_id = str(record.get("produced_by_step") or "").strip()
            if not cls._safe_step_id(step_id) or (
                current_ids is not None and step_id not in current_ids
            ):
                continue
            expected_sha = str(record.get("sha256") or "").strip().lower()
            evidence_id = str(record.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            evidence_path = verified_run_evidence_path(root, record)
            if evidence_path is None:
                continue
            evidence_name = evidence_path.name
            logical_name = (
                evidence_name.split("__", 1)[1]
                if "__" in evidence_name
                else evidence_name
            )
            if not cls._safe_declared_table_name(logical_name):
                continue
            output_path = root / "steps" / step_id / "outputs" / logical_name
            if (
                cls._safe_regular_run_file(output_path, run_dir=root)
                and _sha256_file(output_path) == expected_sha
            ):
                authorised[evidence_id] = {
                    "path": output_path.resolve(),
                    "evidence_path": evidence_path.resolve(),
                    "sha256": expected_sha,
                    "produced_by_step": step_id,
                }
        return authorised

    @classmethod
    def _upstream_tables(
        cls,
        *,
        run_dir: Path,
        current_out_dir: Path,
        upstream_step_ids: Set[str],
        authoritative_tables: Optional[Set[Path]] = None,
    ) -> List[Path]:
        tables: List[Path] = []
        root = Path(run_dir).resolve()
        for step_id in sorted(upstream_step_ids):
            if not cls._safe_step_id(step_id):
                continue
            outputs = run_dir / "steps" / step_id / "outputs"
            if not outputs.exists() or outputs.is_symlink():
                continue
            for path in sorted(outputs.iterdir()):
                if (
                    path.suffix.lower() not in cls._TABULAR_SUFFIXES
                    or not cls._safe_regular_run_file(path, run_dir=root)
                ):
                    continue
                if path.parent.resolve() == current_out_dir.resolve():
                    continue
                if (
                    authoritative_tables is not None
                    and path.resolve() not in authoritative_tables
                ):
                    continue
                tables.append(path)
        return tables

    @classmethod
    def _prioritize_declared_source_tables(
        cls,
        *,
        source_df: pd.DataFrame,
        upstream_tables: Sequence[Path],
    ) -> List[Path]:
        """Put explicitly declared parent tables first.

        Figure source-data tables are often clean, manuscript-facing
        summaries derived from a registered audit table rather than byte-for-
        byte row subsets. A ``source_table`` column is the deterministic
        breadcrumb that says which parent table should be used for provenance
        checks. Keep all tables as fallbacks, but score the declared parent
        first so a coincidental key in an unrelated audit table does not drive
        the mismatch explanation.
        """

        if "source_table" not in source_df.columns:
            return list(upstream_tables)
        declared = {
            Path(str(item)).name
            for item in source_df["source_table"].dropna().astype(str)
            if str(item).strip()
        }
        if not declared:
            return list(upstream_tables)
        return sorted(
            upstream_tables,
            key=lambda path: (path.name not in declared, str(path)),
        )

    @classmethod
    def _resolve_declared_source_tables_across_run(
        cls,
        *,
        run_dir: Path,
        source_df: pd.DataFrame,
        current_out_dir: Path,
        authoritative_tables: Optional[Set[Path]] = None,
        allowed_step_ids: Optional[Set[str]] = None,
    ) -> List[Path]:
        """Locate the figure's self-declared ``source_table`` parents anywhere.

        The ``source_table`` column names the upstream table each figure row was
        derived from. That table may live in ANY prior step's ``outputs/`` (a
        probe/audit table, not just the ``_figure``-suffix sibling), so resolve
        the declared filenames across ``run_dir/steps/*/outputs`` rather than
        only the steps ``_upstream_step_ids`` found. The figure's own output dir
        is excluded so a figure can never be declared traceable to itself.

        Returns the matched parent paths (first-seen order); ``[]`` when the
        column is absent or nothing matches. This only ADDS comparison
        candidates; the caller still runs the subset + value-equality checks, so
        a figure whose values do not match the table it names still fails.
        """

        if "source_table" not in source_df.columns:
            return []
        declared_names = {
            str(item).strip()
            for item in source_df["source_table"].dropna().astype(str)
            if cls._safe_declared_table_name(item)
        }
        if not declared_names:
            return []
        steps_dir = Path(run_dir) / "steps"
        if not steps_dir.exists():
            return []
        current_resolved = current_out_dir.resolve()
        resolved: List[Path] = []
        seen: Set[Path] = set()
        for path in sorted(steps_dir.glob("*/outputs/*")):
            if (
                path.suffix.lower() not in cls._TABULAR_SUFFIXES
                or path.name not in declared_names
                or not cls._safe_regular_run_file(path, run_dir=run_dir)
            ):
                continue
            if (
                allowed_step_ids is not None
                and cls._table_step_id(path, run_dir=run_dir) not in allowed_step_ids
            ):
                continue
            if path.parent.resolve() == current_resolved:
                continue
            rp = path.resolve()
            if authoritative_tables is not None and rp not in authoritative_tables:
                continue
            if rp not in seen:
                seen.add(rp)
                resolved.append(path)
        return resolved

    @classmethod
    def _identify_rows_uniquely(
        cls,
        *,
        source: pd.DataFrame,
        upstream: pd.DataFrame,
        key_cols: tuple,
    ) -> tuple:
        """Widen ``key_cols`` until it identifies one row *upstream*.

        Returns ``(key_cols, [])`` once every source row can match at most one
        parent row, or ``(None, duplicate_examples)`` when no available
        combination gets there.  Refusing is the point: a value comparison
        across a many-to-many join reports differences produced by the join
        itself, which reads as a fabricated figure and is impossible to act on.

        Uniqueness is required on the upstream side only.  The upstream table
        is the authority each source row is checked against, so one duplicated
        parent key is what makes "which row authenticates this one?"
        unanswerable.  A duplicated *source* key is not that: several source
        rows citing the same parent row is a many-to-one join, and every one of
        them is still compared against that parent's values.  It is also a
        shape the projection format supports on purpose -- two panels drawing
        the same parent row each trace it -- so demanding uniqueness on both
        sides rejects a truthful long-form projection for a hazard it does not
        have.  (It did: requiring it broke exactly those cases.)
        """

        def _duplicates(frame: pd.DataFrame, cols: tuple) -> list:
            present = [col for col in cols if col in frame.columns]
            if not present or frame.empty:
                return []
            # Built row by row rather than with astype(str).agg: a row-wise
            # agg re-infers dtypes and hands back the original floats.
            keys = pd.Series(
                [
                    "|".join(str(value) for value in row)
                    for row in frame[present].itertuples(index=False, name=None)
                ],
                index=frame.index,
                dtype=object,
            )
            duplicated = keys[keys.duplicated(keep=False)]
            return sorted(set(duplicated.tolist()))[:10]

        def _identifies_one_parent(cols: tuple) -> bool:
            return not _duplicates(upstream, cols)

        if _identifies_one_parent(key_cols):
            return key_cols, []

        # Any column shared by both frames may help identify the row. Value
        # columns are not excluded here the way they are when *choosing* a
        # key: a level indicator such as ``exposure_level`` is numeric and is
        # exactly what separates the duplicated rows. Correctness comes from
        # the resulting key being unique, not from the columns' names.
        shared = [
            col
            for col in source.columns
            if col in upstream.columns and col not in key_cols
        ]

        def _is_measure(col: str) -> bool:
            """A column that is fully numeric on both sides reads as a value."""

            left = pd.to_numeric(source[col], errors="coerce")
            right = pd.to_numeric(upstream[col], errors="coerce")
            return bool(left.notna().all() and right.notna().all())

        measures = {col: _is_measure(col) for col in shared}
        remaining = list(shared)
        widened = tuple(key_cols)
        while remaining:
            # Greedy, preferring a real identifier over a measurement: a level
            # indicator separates the rows and reads as a key, whereas a
            # confidence bound happens to be distinct and does not. Either
            # yields a correct join -- a figure that altered the column would
            # fail to join at all rather than pass -- but the reported key is
            # also the explanation a reader gets, so it should name what
            # actually distinguishes the rows.
            best_col = min(
                remaining,
                key=lambda col: (
                    measures[col],
                    len(_duplicates(upstream, (*widened, col))),
                    col,
                ),
            )
            candidate = (*widened, best_col)

            def _ambiguity(cols: tuple) -> int:
                return len(_duplicates(upstream, cols))

            if _ambiguity(candidate) >= _ambiguity(widened):
                # The best remaining column separates nothing, so no other
                # will either -- greedy already picked the most separating.
                break
            widened = candidate
            remaining.remove(best_col)
            if _identifies_one_parent(widened):
                return widened, []

        return None, {
            "source": _duplicates(source, key_cols),
            "upstream": _duplicates(upstream, key_cols),
            "attempted_key": list(widened),
        }

    @classmethod
    def _compare_source_to_upstream(
        cls,
        *,
        source_df: pd.DataFrame,
        source_path: Path,
        upstream_path: Path,
    ) -> Dict[str, Any]:
        try:
            upstream_df = cls._read_tabular(upstream_path)
        except Exception as exc:
            return {
                "ok": False,
                "reason": "upstream_read_failed",
                "upstream_table": upstream_path.name,
                "message": f"could not read upstream table {upstream_path.name}: {exc}",
            }
        if upstream_df.empty:
            return {
                "ok": False,
                "reason": "upstream_empty",
                "upstream_table": upstream_path.name,
                "message": f"upstream table {upstream_path.name} is empty",
            }

        used_structural_fallback = False
        key_cols = next(
            (
                tuple(cols)
                for cols in cls._COMPOSITE_KEY_COLUMNS
                if all(
                    col in source_df.columns and col in upstream_df.columns
                    for col in cols
                )
            ),
            None,
        )
        if key_cols is None:
            key = next(
                (
                    col
                    for col in cls._KEY_COLUMNS
                    if col in source_df.columns and col in upstream_df.columns
                ),
                None,
            )
            key_cols = (key,) if key is not None else None
        source = source_df.copy()
        upstream = upstream_df.copy()
        positional_key_label: Optional[str] = None
        selected_position_col: Optional[str] = None
        positional_columns = [
            col for col in cls._POSITIONAL_ROW_INDEX_COLUMNS if col in source.columns
        ]
        parsed_positions: Dict[str, pd.Series] = {}
        for position_col in positional_columns:
            row_index = pd.to_numeric(source[position_col], errors="coerce")
            invalid = (
                row_index.isna()
                | (row_index < 0)
                | (row_index >= len(upstream))
                | (row_index % 1 != 0)
            )
            if invalid.any():
                first_bad = int(invalid[invalid].index[0])
                return {
                    "ok": False,
                    "reason": "source_row_index_out_of_bounds",
                    "key_column": position_col,
                    "upstream_table": upstream_path.name,
                    "message": (
                        f"{position_col} values must be unique integer row "
                        f"positions within {upstream_path.name}; first invalid "
                        f"source-data row is {first_bad}"
                    ),
                }
            parsed_positions[position_col] = row_index.astype(int)

        if len(positional_columns) == 2:
            canonical = parsed_positions["source_row_index"]
            legacy = parsed_positions["_source_row_index"]
            conflict = canonical.ne(legacy)
            if conflict.any():
                first_bad = int(conflict[conflict].index[0])
                return {
                    "ok": False,
                    "reason": "conflicting_source_row_index_aliases",
                    "upstream_table": upstream_path.name,
                    "message": (
                        "source_row_index and _source_row_index must identify "
                        "the same upstream row; first conflict is at source-data "
                        f"row {first_bad}"
                    ),
                }

        if positional_columns:
            selected_position_col = (
                "source_row_index"
                if "source_row_index" in parsed_positions
                else "_source_row_index"
            )

        # A single figure source CSV may use long form for multiple panels: the
        # same parent row then appears once per panel, while a generic column
        # such as ``estimate`` maps to a different upstream measure in each
        # panel.  Validate those panels separately only when every non-empty
        # panel covers the exact same unique parent-position set.  This keeps
        # the grouping structural (not a free-text scientific guess) and still
        # requires every value column in every panel to match its parent rows.
        if selected_position_col is not None and "panel_id" in source.columns:
            panel_ids = source["panel_id"].fillna("").astype(str).str.strip()
            unique_panels = [value for value in panel_ids.unique() if value]
            if len(unique_panels) > 1 and panel_ids.ne("").all():
                panel_position_sets: List[Set[int]] = []
                panel_groups: List[tuple[str, pd.DataFrame]] = []
                panels_are_complete = True
                for panel_id in unique_panels:
                    panel_mask = panel_ids.eq(panel_id)
                    panel_positions = parsed_positions[selected_position_col].loc[
                        panel_mask
                    ]
                    if panel_positions.duplicated().any():
                        panels_are_complete = False
                        break
                    panel_position_sets.append(set(panel_positions.astype(int)))
                    panel_groups.append((panel_id, source.loc[panel_mask].copy()))
                if (
                    panels_are_complete
                    and panel_position_sets
                    and all(
                        positions == panel_position_sets[0]
                        for positions in panel_position_sets[1:]
                    )
                ):
                    panel_results = {
                        panel_id: cls._compare_source_to_upstream(
                            source_df=panel_df,
                            source_path=source_path,
                            upstream_path=upstream_path,
                        )
                        for panel_id, panel_df in panel_groups
                    }
                    failed_panel = next(
                        (
                            (panel_id, result)
                            for panel_id, result in panel_results.items()
                            if not result.get("ok")
                        ),
                        None,
                    )
                    if failed_panel is not None:
                        panel_id, result = failed_panel
                        return {
                            **result,
                            "panel_id": panel_id,
                            "message": (
                                f"panel {panel_id} failed source verification: "
                                f"{result.get('message', 'unknown mismatch')}"
                            ),
                        }
                    return {
                        "ok": True,
                        "reason": "source_subset_matches",
                        "source_table": source_path.name,
                        "upstream_table": upstream_path.name,
                        "key_column": selected_position_col,
                        "n_source_rows": int(len(source_df)),
                        "join_mode": "panel_stratified_positional",
                        "verified_panels": panel_results,
                    }

        if selected_position_col is not None:
            positional_key_label = selected_position_col
            join_col = "__easyicu_parent_row_position"
            while join_col in source.columns or join_col in upstream.columns:
                join_col = f"_{join_col}"
            source[join_col] = parsed_positions[selected_position_col].astype(str)
            upstream[join_col] = pd.Series(
                range(len(upstream)), index=upstream.index, dtype=int
            ).astype(str)
            key_cols = (join_col, *(key_cols or ()))
        if key_cols is None:
            # Structural fallback: no composite / named / positional key matched,
            # but a faithfully-derived figure often preserves the parent's OWN key
            # column under a name not in _KEY_COLUMNS (e.g. category_code,
            # lactate_group, group). Accept ANY column present in BOTH frames that
            # is (a) not a numeric value/measure and (b) identifier-like in the
            # source (mostly-distinct), choosing the one whose source values best
            # join into the upstream. The value-equality checks below still run on
            # every shared numeric column, so this only enables the JOIN and never
            # masks a fabricated value. This moves traceability OFF the hard-coded
            # key-name allowlist that needed a new entry per case
            # (contrast_id/stage/level/... -> group/category_code) onto structural
            # evidence. Because structurally selected identifiers have no
            # semantic contract, the join is allowed to PASS only when every
            # numeric source-data value column has a same-name upstream value
            # column and is actually checked below. A truthful count must never
            # launder an unrelated renamed/forged estimate. Only reached when the
            # existing resolution already returned no_shared_key, so it cannot
            # change any currently-passing figure's key.
            n_src = max(len(source), 1)
            best: Optional[tuple[tuple[float, float], str]] = None
            for col in source.columns:
                if col not in upstream.columns:
                    continue
                if (
                    col in {*cls._POSITIONAL_ROW_INDEX_COLUMNS, "source_table"}
                    or col in cls._NUMERIC_COLUMNS
                ):
                    continue
                left_num = pd.to_numeric(source[col], errors="coerce")
                right_num = pd.to_numeric(upstream[col], errors="coerce")
                # a column fully numeric in both frames is a value/measure, not a key
                if left_num.notna().all() and right_num.notna().all():
                    continue
                s_vals = source[col].dropna().astype(str)
                if s_vals.empty:
                    continue
                distinct_ratio = s_vals.nunique() / n_src
                if distinct_ratio < 0.5:  # a real per-row key is mostly-distinct
                    continue
                u_vals = set(upstream[col].dropna().astype(str))
                overlap = float(s_vals.isin(u_vals).mean())  # joinable fraction
                score = (overlap, distinct_ratio)
                if overlap > 0 and (best is None or score > best[0]):
                    best = (score, col)
            if best is not None:
                key_cols = (best[1],)
                used_structural_fallback = True
        if key_cols is None:
            return {
                "ok": False,
                "reason": "no_shared_key",
                "upstream_table": upstream_path.name,
                "message": f"no shared key column with {upstream_path.name}",
            }
        for key in key_cols:
            source[key] = source[key].astype(str)
            upstream[key] = upstream[key].astype(str)

        # A join key must IDENTIFY a row, not merely vary across rows. Both
        # selectors above could hand back a non-unique key: the named
        # allowlist does it whenever a table repeats a term across models
        # (already noted on _COMPOSITE_KEY_COLUMNS), and the structural
        # selector accepts any shared column that is "mostly distinct"
        # (>= 50% unique), which is not the same as unique.
        #
        # Measured consequence: a three-row distribution table keyed on
        # ``row_role`` -- two ``exposure_level`` rows plus one ``overall`` --
        # scored 2/3 and was accepted. pandas then joined the duplicates
        # many-to-many, so exposure level 0 in the figure's source data was
        # compared against level 1 upstream, and EVERY numeric column was
        # reported as disagreeing between two byte-identical files. The
        # figure was rejected for fabricating values it had copied exactly.
        #
        # Extend the key with further shared columns until it identifies a
        # row. If it cannot be made unique, say so precisely rather than
        # compare cross-matched rows and report the differences as evidence
        # of a forged figure.
        key_cols, duplicate_examples = cls._identify_rows_uniquely(
            source=source,
            upstream=upstream,
            key_cols=key_cols,
        )
        if key_cols is None:
            return {
                "ok": False,
                "reason": "ambiguous_join_key",
                "upstream_table": upstream_path.name,
                "duplicate_key_values": duplicate_examples,
                "message": (
                    f"no key identifies a single row of {upstream_path.name}, "
                    "so no source row can be traced to one parent row; "
                    "comparing values across a many-to-many join would report "
                    "differences that are an artefact of the join, not of the "
                    "figure"
                ),
            }

        def _key_set(frame: pd.DataFrame):
            return _figure_source_compare___key_set(frame, key_cols=key_cols)

        upstream_keys = _key_set(upstream)
        missing_keys = sorted(_key_set(source) - upstream_keys)
        key_label = positional_key_label or "+".join(key_cols)

        def _format_key(row: pd.Series):
            return _figure_source_compare___format_key(row, key_cols=key_cols)

        if missing_keys:
            return {
                "ok": False,
                "reason": "source_rows_not_in_upstream",
                "key_column": key_label,
                "upstream_table": upstream_path.name,
                "missing_keys": ["|".join(item) for item in missing_keys[:20]],
                "n_missing_keys": len(missing_keys),
                "message": (
                    f"{len(missing_keys)} {key_label} value(s) are absent from "
                    f"{upstream_path.name}"
                ),
            }

        merged = source.merge(
            upstream,
            on=list(key_cols),
            how="left",
            suffixes=("_source", "_upstream"),
        )
        mismatches: List[Dict[str, Any]] = []
        ignored_for_dynamic_numeric = {
            *key_cols,
            *cls._POSITIONAL_ROW_INDEX_COLUMNS,
            "source_table",
        }
        text_name = re.compile(
            r"(?:^|_)(?:label|name|category|group|stratum|term|id|level|stage|band|role|status|column|"
            r"method|table|source|description|note)(?:_|$)"
        )
        value_name = re.compile(
            r"(?:^|_)(?:estimate|effect|rate|risk|odds|hazard|ratio|percent|pct|"
            r"count|ci|lower|upper|mean|median|se|p|statistic|value|n)(?:_|$)"
        )

        def _clean_numeric(raw: pd.Series):
            return _figure_source_compare___clean_numeric(raw)

        def _is_value_column(frame: pd.DataFrame, col: str):
            return _figure_source_compare___is_value_column(frame, col, _clean_numeric=_clean_numeric, cls=cls, ignored_for_dynamic_numeric=ignored_for_dynamic_numeric, text_name=text_name, value_name=value_name)

        source_value_columns = {
            col for col in source.columns if _is_value_column(source, col)
        }
        upstream_value_columns = {
            col for col in upstream.columns if _is_value_column(upstream, col)
        }

        def _merged_source(col: str):
            return _figure_source_compare___merged_source(col, merged=merged, upstream=upstream)

        def _merged_upstream(col: str):
            return _figure_source_compare___merged_upstream(col, merged=merged, source=source)

        def _numeric_comparison(source_name: str, upstream_name: str):
            return _figure_source_compare___numeric_comparison(source_name, upstream_name, _clean_numeric=_clean_numeric, _merged_source=_merged_source, _merged_upstream=_merged_upstream, cls=cls)

        def _value_family(col: str):
            return _figure_source_compare___value_family(col)

        def _structured_source_family(source_name: str):
            return _figure_source_compare___structured_source_family(source_name, _value_family=_value_family, cls=cls, source=source)

        def _cross_name_families_compatible(source_name: str, upstream_name: str):
            return _figure_source_compare___cross_name_families_compatible(source_name, upstream_name, _structured_source_family=_structured_source_family, _value_family=_value_family)

        def _explicit_semantic_target_columns(source_name: str):
            return _figure_source_compare___explicit_semantic_target_columns(source_name, _cross_name_families_compatible=_cross_name_families_compatible, cls=cls, source=source, upstream_value_columns=upstream_value_columns)

        verified_value_mappings: Dict[str, str] = {}
        used_upstream_value_columns: Set[str] = set()
        ambiguous_value_mappings: Dict[str, List[str]] = {}
        for source_col in sorted(source_value_columns):
            # A same-name value is authoritative: if it disagrees, never search
            # another column for a coincidental numeric match that could launder
            # the mismatch.
            if source_col in upstream_value_columns:
                verified, disagrees, bad, left, right, diff = _numeric_comparison(
                    source_col, source_col
                )
                if verified:
                    verified_value_mappings[source_col] = source_col
                    used_upstream_value_columns.add(source_col)
                elif disagrees:
                    idx = int(bad[bad].index[0])
                    abs_tolerance = (
                        cls._PERCENTAGE_ABS_TOL
                        if any(
                            token in source_col.lower() for token in ("_pct", "percent")
                        )
                        else cls._DEFAULT_NUMERIC_ABS_TOL
                    )
                    mismatches.append(
                        {
                            "column": source_col,
                            "upstream_column": source_col,
                            "key": _format_key(merged.loc[idx]),
                            "source": (
                                None if pd.isna(left.loc[idx]) else float(left.loc[idx])
                            ),
                            "upstream": (
                                None
                                if pd.isna(right.loc[idx])
                                else float(right.loc[idx])
                            ),
                            "abs_diff": (
                                None if pd.isna(diff.loc[idx]) else float(diff.loc[idx])
                            ),
                            "abs_tolerance": abs_tolerance,
                        }
                    )
                continue

            explicit_targets = _explicit_semantic_target_columns(source_col)
            if explicit_targets:
                if len(explicit_targets) > 1:
                    ambiguous_value_mappings[source_col] = explicit_targets
                    continue
                target = explicit_targets[0]
                verified, disagrees, bad, left, right, diff = _numeric_comparison(
                    source_col, target
                )
                if verified:
                    verified_value_mappings[source_col] = target
                    used_upstream_value_columns.add(target)
                elif disagrees:
                    idx = int(bad[bad].index[0])
                    mismatches.append(
                        {
                            "column": source_col,
                            "upstream_column": target,
                            "semantic_binding": True,
                            "key": _format_key(merged.loc[idx]),
                            "source": (
                                None if pd.isna(left.loc[idx]) else float(left.loc[idx])
                            ),
                            "upstream": (
                                None
                                if pd.isna(right.loc[idx])
                                else float(right.loc[idx])
                            ),
                            "abs_diff": (
                                None if pd.isna(diff.loc[idx]) else float(diff.loc[idx])
                            ),
                        }
                    )
                # A concrete declaration is binding: never search a sibling
                # same-family column after its named target disagrees.
                continue

            # Renderers may use a presentation-neutral alias (for example
            # ``ci_low`` for upstream ``or_ci_low``).  Verify renamed values by
            # their complete row-aligned numeric vector and record the mapping;
            # zero comparisons or a partial/mixed match never count as proof.
            if source_col.startswith("plot_"):
                continue
            matching_upstream_columns: List[str] = []
            for upstream_col in sorted(upstream_value_columns):
                if upstream_col in used_upstream_value_columns:
                    continue
                if not _cross_name_families_compatible(source_col, upstream_col):
                    continue
                verified, _disagrees, _bad, _left, _right, _diff = _numeric_comparison(
                    source_col, upstream_col
                )
                if verified:
                    matching_upstream_columns.append(upstream_col)
            if len(matching_upstream_columns) == 1:
                matched = matching_upstream_columns[0]
                verified_value_mappings[source_col] = matched
                used_upstream_value_columns.add(matched)
            elif len(matching_upstream_columns) > 1:
                ambiguous_value_mappings[source_col] = matching_upstream_columns

        def _derived_matches(source_col: str, expected_vectors: Sequence[pd.Series], *, tolerance: Optional[float]=None):
            return _figure_source_compare___derived_matches(source_col, expected_vectors, tolerance=tolerance, _clean_numeric=_clean_numeric, _merged_source=_merged_source, cls=cls)

        # Derived display columns remain fail-closed, but can be authenticated
        # from already verified source values.  This preserves honest renderer
        # aliases without allowing an unrelated truthful count to launder a
        # forged estimate.
        for source_col in sorted(source_value_columns - set(verified_value_mappings)):
            source_family = _structured_source_family(source_col)
            for verified_source_col in sorted(verified_value_mappings):
                verified_family = _structured_source_family(verified_source_col)
                compatible_alias = source_family == verified_family
                if source_family == "generic_estimate":
                    compatible_alias = verified_family in {"rate", "ratio"}
                if not compatible_alias or source_family in {
                    "generic_value",
                    "other_numeric",
                    "ordering",
                }:
                    continue
                if _derived_matches(
                    source_col,
                    [_clean_numeric(_merged_source(verified_source_col))],
                ):
                    verified_value_mappings[source_col] = (
                        f"derived:alias({verified_source_col})"
                    )
                    break

        for width_col in ("ci_width", "errorbar_width"):
            if (
                width_col in source_value_columns
                and "ci_low" in verified_value_mappings
                and "ci_high" in verified_value_mappings
                and _derived_matches(
                    width_col,
                    [
                        _clean_numeric(_merged_source("ci_high"))
                        - _clean_numeric(_merged_source("ci_low"))
                    ],
                )
            ):
                verified_value_mappings[width_col] = "derived:ci_high-ci_low"

        # A renderer may make an upstream long table presentation-ready by
        # adding a denominator and percentage. Authenticate that denominator
        # only when it equals the complete upstream count total within an
        # explicit structural stratum (row/group/estimate type), then derive
        # the percentage from two already-authenticated values. This cannot
        # bless an arbitrary display column or a subset-dependent denominator.
        if (
            "denominator" in source_value_columns
            and "denominator" not in verified_value_mappings
        ):
            grouped_total_candidates: List[tuple[str, pd.Series]] = []
            seen_grouped_totals: Set[tuple[str, str]] = set()
            for source_count_col in ("count", "n", "membership_n", "n_included"):
                upstream_count_col = verified_value_mappings.get(source_count_col)
                if upstream_count_col not in upstream.columns:
                    continue
                upstream_counts = _clean_numeric(upstream[upstream_count_col])
                grouped_total_candidates.append(
                    (
                        f"derived:sum({upstream_count_col})",
                        pd.Series(
                            upstream_counts.sum(min_count=1),
                            index=merged.index,
                            dtype=float,
                        ),
                    )
                )
                for group_col in ("row_type", "group_type", "estimate_type"):
                    pair = (str(upstream_count_col), group_col)
                    if pair in seen_grouped_totals or group_col not in upstream.columns:
                        continue
                    seen_grouped_totals.add(pair)
                    group_key = upstream[group_col].fillna("<missing>").astype(str)
                    totals = upstream_counts.groupby(group_key).sum(min_count=1)
                    merged_group_key = (
                        _merged_upstream(group_col).fillna("<missing>").astype(str)
                    )
                    grouped_total_candidates.append(
                        (
                            f"derived:sum({upstream_count_col})_by_{group_col}",
                            merged_group_key.map(totals),
                        )
                    )
            for derivation, expected in grouped_total_candidates:
                if _derived_matches("denominator", [expected]):
                    verified_value_mappings["denominator"] = derivation
                    break

        if (
            "percentage" in source_value_columns
            and "percentage" not in verified_value_mappings
            and "denominator" in verified_value_mappings
        ):
            denominator = _clean_numeric(_merged_source("denominator")).replace(
                0.0, float("nan")
            )
            percentage_vectors = [
                100.0 * _clean_numeric(_merged_source(count_col)) / denominator
                for count_col in ("count", "n", "membership_n", "n_included")
                if count_col in verified_value_mappings and count_col in source.columns
            ]
            if percentage_vectors and _derived_matches(
                "percentage",
                percentage_vectors,
                tolerance=cls._PERCENTAGE_ABS_TOL,
            ):
                verified_value_mappings["percentage"] = (
                    "derived:100*verified_count/verified_denominator"
                )

        total_candidates = [
            col
            for col in ("total_n", "n_total", "denominator", "denominator_n")
            if col in verified_value_mappings and col in source.columns
        ]
        missing_candidates = [
            col
            for col in (
                "missing_n",
                "value_missing_n",
                "raw_missing_n",
                "analysis_unavailable_n",
            )
            if col in verified_value_mappings and col in source.columns
        ]
        complement_vectors = [
            _clean_numeric(_merged_source(total_col))
            - _clean_numeric(_merged_source(missing_col))
            for total_col in total_candidates
            for missing_col in missing_candidates
        ]
        for measured_col in ("measured_n", "n_nonmissing"):
            if (
                measured_col in source_value_columns
                and complement_vectors
                and _derived_matches(measured_col, complement_vectors)
            ):
                verified_value_mappings[measured_col] = (
                    "derived:denominator-minus-unavailable"
                )
        if (
            "measured_pct" in source_value_columns
            and "measured_n" in verified_value_mappings
            and total_candidates
        ):
            measured = _clean_numeric(_merged_source("measured_n"))
            pct_vectors = [
                100.0
                * measured
                / _clean_numeric(_merged_source(total_col)).replace(0.0, float("nan"))
                for total_col in total_candidates
            ]
            if _derived_matches(
                "measured_pct",
                pct_vectors,
                tolerance=cls._PERCENTAGE_ABS_TOL,
            ):
                verified_value_mappings["measured_pct"] = (
                    "derived:100*measured_n/denominator"
                )

        for plot_col in sorted(
            col
            for col in source_value_columns
            if col.startswith("plot_") and col not in verified_value_mappings
        ):
            if _value_family(plot_col) == "ordering":
                matching_order_columns = []
                for upstream_col in sorted(upstream.columns):
                    if _value_family(upstream_col) != "ordering":
                        continue
                    verified, _disagrees, _bad, _left, _right, _diff = (
                        _numeric_comparison(plot_col, upstream_col)
                    )
                    if verified:
                        matching_order_columns.append(upstream_col)
                if len(matching_order_columns) == 1:
                    verified_value_mappings[plot_col] = (
                        f"derived:ordering({matching_order_columns[0]})"
                    )
                elif len(matching_order_columns) > 1:
                    ambiguous_value_mappings[plot_col] = matching_order_columns
                continue
            target = plot_col.removeprefix("plot_").removesuffix("_pct")
            if "ci_low" in target:
                source_candidates = [
                    col for col in verified_value_mappings if "ci_low" in col
                ]
            elif "ci_high" in target:
                source_candidates = [
                    col for col in verified_value_mappings if "ci_high" in col
                ]
            elif "estimate" in target:
                source_candidates = [
                    col
                    for col in verified_value_mappings
                    if any(token in col for token in ("estimate", "risk", "rate"))
                    and "ci_" not in col
                ]
            else:
                source_candidates = []
            plot_vectors: List[pd.Series] = []
            for candidate in source_candidates:
                base = _clean_numeric(_merged_source(candidate))
                plot_vectors.extend([base, 100.0 * base])
            if plot_vectors and _derived_matches(
                plot_col,
                plot_vectors,
                tolerance=cls._PERCENTAGE_ABS_TOL,
            ):
                verified_value_mappings[plot_col] = (
                    f"derived:display-scale({','.join(source_candidates)})"
                )
        for col in cls._TEXT_COLUMNS:
            source_col = f"{col}_source"
            upstream_col = f"{col}_upstream"
            if source_col not in merged.columns or upstream_col not in merged.columns:
                continue
            left = merged[source_col].fillna("").astype(str).str.strip().str.lower()
            right = merged[upstream_col].fillna("").astype(str).str.strip().str.lower()
            bad = left != right
            if bad.any():
                idx = int(bad[bad].index[0])
                mismatches.append(
                    {
                        "column": col,
                        "key": _format_key(merged.loc[idx]),
                        "source": merged.loc[idx, source_col],
                        "upstream": merged.loc[idx, upstream_col],
                    }
                )
        if mismatches:
            return {
                "ok": False,
                "reason": "source_values_disagree",
                "key_column": key_label,
                "upstream_table": upstream_path.name,
                "mismatches": mismatches[:20],
                "n_mismatches": len(mismatches),
                "message": f"source-data values disagree with {upstream_path.name}",
            }
        unverified_source_columns = sorted(
            source_value_columns - set(verified_value_mappings)
        )
        if not source_value_columns and not upstream_value_columns:
            # A DESIGN TABLE HAS NOTHING TO VERIFY, AND DEMANDING IT VERIFIES
            # NOTHING.
            #
            # The rule above exists so a figure cannot claim a number no bound
            # parent supports. A parent carrying no value column supports no
            # number, so "verify its values" is a demand nothing can meet: an
            # exact row-for-row copy of it fails here with
            # ``no_verifiable_values``, which is what happened to the robustness
            # renderer's specification-grid companion on e2 (2026-08-03) -- the
            # grid is the plan's own description of what each specification
            # CHANGES (``spec_id``/``axis``/``description`` plus override
            # columns that are empty), and none of that is a result.
            #
            # The filter that already exists for this -- ``result_families``
            # deciding a bound table is not a value source -- is skipped
            # whenever the step declares no result family, and a rendering-only
            # figure step never declares one: measured over every recorded plan,
            # 912 of 1052 visualization steps (87%). So the guard that would
            # have excused this case is disabled in exactly the case it is for.
            #
            # BOTH SIDES MUST BE VALUE-LESS. If the upstream carries values and
            # the source does not, the source dropped them and must still fail
            # -- which is why this is not keyed on the source alone.
            #
            # AND THE COPY MUST BE EXACT ON EVERY SHARED COLUMN, not only on
            # ``_TEXT_COLUMNS``. That list is a fixed 22 names and the grid's
            # own ``description`` -- the sentence the plan registered for a
            # specification -- is not among them, so the text check above would
            # have let an altered description through. With no value to verify,
            # faithful reproduction is the only verification left, so it is the
            # one this branch performs.
            infidelities: List[Dict[str, Any]] = []
            for column in sorted(set(source.columns) & set(upstream.columns)):
                if column in cls._TEXT_COLUMNS:
                    continue  # already compared, and already reported above
                left = _merged_source(column).fillna("").astype(str).str.strip()
                right = _merged_upstream(column).fillna("").astype(str).str.strip()
                disagreeing = left != right
                if disagreeing.any():
                    index = int(disagreeing[disagreeing].index[0])
                    infidelities.append(
                        {
                            "column": column,
                            "key": _format_key(merged.loc[index]),
                            "source": str(left.loc[index]),
                            "upstream": str(right.loc[index]),
                        }
                    )
            if infidelities:
                return {
                    "ok": False,
                    "reason": "source_values_disagree",
                    "key_column": key_label,
                    "upstream_table": upstream_path.name,
                    "mismatches": infidelities[:20],
                    "n_mismatches": len(infidelities),
                    "message": (
                        f"{upstream_path.name} carries no value to verify, so the "
                        "source data must reproduce it exactly; it does not"
                    ),
                }
            return {
                "ok": True,
                "reason": "valueless_parent_reproduced",
                "source_table": source_path.name,
                "upstream_table": upstream_path.name,
                "key_column": key_label,
                "n_source_rows": int(len(source_df)),
                "verified_value_mappings": {},
                "join_mode": (
                    "structural_fallback"
                    if used_structural_fallback
                    else "declared_key"
                ),
            }
        if not verified_value_mappings or unverified_source_columns:
            # WHY no column matched, when the answer is mechanical.
            #
            # A source table that carries SEVERAL rows per upstream row -- one
            # per panel, per statistic, per level -- holds values from several
            # upstream columns in one source column, so by construction no
            # single upstream vector matches it and every column arrives here
            # unverified. The reader is then told which columns failed but not
            # the one fact that explains all of them, and the repair that
            # follows tends to rename columns, which the Coder prompt already
            # says "is not a repair".
            #
            # MEASURED over the recorded corpus: 12 of 361 source-data tables
            # carry duplicate keys, and 6 of the 8 whose step status is known
            # failed. It is not fatal on its own -- 2 passed -- so this reports
            # the shape and does not judge it.
            #
            # m1's 09_missingness_audit_figure, 2026-08-04: 6 rows over 3
            # upstream rows, one per panel, so ``count`` alternated between the
            # upstream's ``missing_n`` and ``measured_n``.
            rows_per_upstream_row: dict[str, object] = {}
            try:
                distinct_keys = int(source[list(key_cols)].drop_duplicates().shape[0])
                if distinct_keys and len(source) > distinct_keys:
                    rows_per_upstream_row = {
                        "source_rows_per_upstream_row": round(
                            len(source) / distinct_keys, 3
                        ),
                        "n_source_rows": int(len(source)),
                        "n_distinct_source_keys": distinct_keys,
                    }
            except Exception:  # noqa: BLE001 - a diagnostic must never fail the audit
                rows_per_upstream_row = {}
            if unverified_source_columns:
                verification_detail = (
                    "these source-data value columns were not verified against "
                    "any row-aligned upstream value vector: "
                    f"{unverified_source_columns}; one verified column cannot "
                    "authenticate another renamed, formatted, or transformed value"
                )
                if rows_per_upstream_row:
                    verification_detail += (
                        f"; the source carries {rows_per_upstream_row['n_source_rows']}"
                        f" rows over {rows_per_upstream_row['n_distinct_source_keys']}"
                        " upstream rows, so one source column holds values from"
                        " several upstream columns and no single upstream vector"
                        " can match it"
                    )
            else:
                verification_detail = (
                    "no source-data value column was available for a real "
                    "row-aligned comparison"
                )
            return {
                "ok": False,
                "reason": "no_verifiable_values",
                "key_column": key_label,
                "upstream_table": upstream_path.name,
                "unverified_source_value_columns": unverified_source_columns,
                "verified_source_value_columns": sorted(verified_value_mappings),
                "verified_value_mappings": verified_value_mappings,
                "ambiguous_value_mappings": ambiguous_value_mappings,
                **rows_per_upstream_row,
                "message": (
                    f"source rows joined to {upstream_path.name} on {key_label}, "
                    f"but {verification_detail}"
                ),
            }
        return {
            "ok": True,
            "reason": "source_subset_matches",
            "source_table": source_path.name,
            "upstream_table": upstream_path.name,
            "key_column": key_label,
            "n_source_rows": int(len(source_df)),
            "verified_value_mappings": verified_value_mappings,
            "join_mode": (
                "structural_fallback" if used_structural_fallback else "declared_key"
            ),
        }


class FigureContractQualityValidator:
    """Audit manuscript-facing figure contracts beyond file/source existence."""

    name = "figure_contract_quality"
    _CONTRACT_GLOB = "*.figure_contract.json"
    _FALLBACK_TERMS = (
        "rescue",
        "fallback",
        "placeholder",
        "did not emit exports",
        "no generated figure",
    )
    _RESULT_ROLES = {
        "relationship",
        "robustness",
        "forest_odds_ratio",
        "forest_risk_difference",
        "forest_risk_ratio",
        "association",
        "effect",
        "descriptive_result",
        "primary_estimand",
        "model_performance",
        "calibration",
        "temporal_absolute_risk",
        "survival_effect",
        "phenotype_structure",
        "phenotype_profile",
        "stability",
        "causal_contrast",
        "distribution",
    }
    # Supporting/context panel roles. A figure whose EVERY panel carries one of
    # these roles is an audit/diagnostic/overview figure — legitimately allowed to
    # be single-panel — and must NOT be gated by the manuscript-facing result-figure
    # ">= 2 panels" rule. Decided on the structured panel ``role`` rather than free
    # text, because a supporting figure's id or core_claim can contain a result-role
    # word (e.g. "distribution", "effect") without being a primary result figure.
    _SUPPORTING_ROLES = {
        "audit",
        "diagnostic",
        "qa",
        "qa_only",
        "exploratory",
        "overview",
        "context",
        "data_quality",
        "missingness",
    }
    _RAW_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+){1,}\b")

    def audit(
        self,
        *,
        step: AnalysisStep,
        out_dir: Path,
        run_dir: Path,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        if not FigureSourceDataValidator._is_rendering_step(
            step=step,
            step_summary=step_summary,
        ):
            return []
        findings: List[ValidationFinding] = []
        contract_paths = sorted(out_dir.glob(self._CONTRACT_GLOB))
        if not contract_paths and self._has_figure_exports(out_dir):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Figure step '{step.step_id}' wrote figure exports "
                        "without a .figure_contract.json file; manuscript-facing "
                        "figures must declare panel claims and source evidence."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "out_dir": str(out_dir),
                    },
                )
            )
            return findings
        for contract_path in contract_paths:
            findings.extend(
                self.audit_contract_file(
                    contract_path,
                    step=step,
                    step_summary=step_summary,
                    manuscript_facing=True,
                )
            )
        return findings

    @staticmethod
    def _has_figure_exports(out_dir: Path) -> bool:
        figure_suffixes = {".svg", ".pdf", ".png", ".tiff", ".tif", ".pptx"}
        return any(
            path.is_file() and path.suffix.lower() in figure_suffixes
            for path in out_dir.iterdir()
        )

    def audit_contract_file(
        self,
        contract_path: Path,
        *,
        step: Optional[AnalysisStep] = None,
        step_summary: Optional[Dict[str, Any]] = None,
        manuscript_facing: Optional[bool] = None,
    ) -> List[ValidationFinding]:
        try:
            raw = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=f"Could not read figure contract {contract_path.name}: {exc}",
                    detail={"path": str(contract_path)},
                )
            ]
        if not isinstance(raw, dict):
            return []

        is_manuscript = (
            bool(manuscript_facing)
            if manuscript_facing is not None
            else self._looks_manuscript_facing(raw, contract_path, step, step_summary)
        )
        if not is_manuscript:
            return []

        figure_id = str(raw.get("figure_id") or contract_path.stem)
        panels = raw.get("panels")
        panels_list = panels if isinstance(panels, list) else []
        text_blob = self._contract_text(raw)
        findings: List[ValidationFinding] = []

        fallback_terms = [
            term for term in self._FALLBACK_TERMS if term in text_blob.lower()
        ]
        if fallback_terms:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"{figure_id} is marked as a fallback/rescue figure; "
                        "manuscript-facing figures must be regenerated from "
                        "registered source data instead of accepted as rescue output."
                    ),
                    detail={
                        "path": str(contract_path),
                        "terms": sorted(set(fallback_terms)),
                        "step_id": getattr(step, "step_id", None),
                    },
                )
            )

        result_like = self._is_result_like_contract(raw, panels_list)
        if (
            result_like
            and len(panels_list) < 2
            and not self._is_supporting_figure_step(step)
            and not self._contract_looks_supporting_figure(raw, figure_id)
        ):
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"{figure_id} has only {len(panels_list)} panel(s); "
                        "manuscript-facing result figures need at least two "
                        "data-backed panels so the primary estimate, robustness, "
                        "and audit context are not collapsed into one forest plot."
                    ),
                    detail={
                        "path": str(contract_path),
                        "panel_count": len(panels_list),
                        "step_id": getattr(step, "step_id", None),
                    },
                )
            )

        blank_titles = [
            str(panel.get("panel_id") or idx + 1)
            for idx, panel in enumerate(panels_list)
            if isinstance(panel, dict) and not str(panel.get("title") or "").strip()
        ]
        if blank_titles:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"{figure_id} has panel(s) without titles: "
                        + ", ".join(blank_titles)
                    ),
                    detail={"path": str(contract_path), "panel_ids": blank_titles},
                )
            )

        weak_claims = [
            str(panel.get("panel_id") or idx + 1)
            for idx, panel in enumerate(panels_list)
            if isinstance(panel, dict)
            and len(str(panel.get("claim") or "").strip()) < 24
        ]
        if weak_claims:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"{figure_id} has panel(s) with weak or missing claims: "
                        + ", ".join(weak_claims)
                    ),
                    detail={"path": str(contract_path), "panel_ids": weak_claims},
                )
            )

        machine_labels = sorted(
            {
                token
                for token in self._RAW_IDENTIFIER_RE.findall(
                    self._reader_facing_text(raw)
                )
                if token not in {"figure_id", "source_data", "evidence_ids"}
            }
        )
        if machine_labels:
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=(
                        f"{figure_id} includes machine-style labels in the "
                        "figure contract; manuscript figures should expose "
                        "reader-facing labels."
                    ),
                    detail={
                        "path": str(contract_path),
                        "examples": machine_labels[:10],
                    },
                )
            )
        return findings

    @classmethod
    def _looks_manuscript_facing(
        cls,
        raw: Dict[str, Any],
        contract_path: Path,
        step: Optional[AnalysisStep],
        step_summary: Optional[Dict[str, Any]],
    ) -> bool:
        haystack = cls._contract_text(raw) + f"\n{contract_path.name}"
        if step is not None:
            haystack += f"\n{step.step_id}\n{step.intent}\n{step.method}"
            haystack += "\n" + json.dumps(
                getattr(step, "expected_outputs", []) or [],
                default=str,
            )
        if step_summary:
            haystack += "\n" + json.dumps(step_summary, default=str)
        lowered = haystack.lower()
        if any(token in lowered for token in ("exploratory", "diagnostic", "qa only")):
            return False
        return any(
            token in lowered
            for token in ("figure", "publication", "manuscript", "render")
        )

    @classmethod
    def _is_result_like_contract(
        cls,
        raw: Dict[str, Any],
        panels: Sequence[Any],
    ) -> bool:
        panel_roles = [
            cls._normalise_supporting_identifier(panel.get("role"))
            for panel in panels
            if isinstance(panel, dict)
        ]
        # An all-supporting-role figure (every panel is audit/diagnostic/overview/…)
        # is not a manuscript-facing PRIMARY result figure. Exclude it here so the
        # ">= 2 panels" rule does not fire on a legitimately single-panel audit or
        # overview figure (e.g. probe_overview, reporting_followup_distribution) whose
        # id/core_claim happens to contain a result-role substring.
        labelled_roles = [role for role in panel_roles if role]
        if (
            panels
            and len(panel_roles) == len(panels)
            and all(role in cls._SUPPORTING_ROLES for role in panel_roles)
        ):
            return False
        if any(role in cls._RESULT_ROLES for role in labelled_roles):
            return True
        text_blob = cls._contract_text(raw).lower()
        return any(role in text_blob for role in cls._RESULT_ROLES)

    # Exact artifact roles retained for legacy contracts whose panel was
    # mistakenly labelled with a result role (for example, role="robustness"
    # on the separate audit_panel figure). Free-text substrings are deliberately
    # excluded: ``audited_primary_effect`` is a primary result, not an audit
    # artifact merely because its identifier contains "audit".
    _SUPPORTING_ARTIFACT_IDS = {
        "audit",
        "audit_panel",
        "data_completeness_panel",
        "data_quality",
        "data_quality_panel",
        "diagnostic",
        "diagnostic_panel",
        "measurement_process_audit",
        "missingness",
        "missingness_measurement_panel",
        "overview",
        "probe_overview",
        "qa",
        "qa_panel",
        "quality_control",
        "quality_control_panel",
    }

    @staticmethod
    def _normalise_supporting_identifier(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"^figure\s*:\s*", "", text)
        return re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    @classmethod
    def _is_supporting_figure_step(cls, step: Optional[AnalysisStep]) -> bool:
        """True when the step is a SUPPORTING audit/QC figure, not the primary
        result figure.

        Such a supplementary figure must not be held to the primary-result
        ">= 2 data-backed panels" rule: its very existence as a SEPARATE figure
        means the audit context is not collapsed into the primary result figure
        (which the rule exists to prevent). Without this, an LLM coder that tags
        a lone audit panel with a result role ("robustness"/"stability") makes a
        supplementary figure hard-fail the whole run — the M3 subphenotype block.
        The deterministic audit renderer additionally emits >= 2 supporting-role
        panels, so this is the belt to that renderer's suspenders.
        """
        if step is None:
            return False
        if str(getattr(step, "planned_analysis_role", "") or "").strip().lower() == (
            "auxiliary"
        ):
            return True
        step_id = cls._normalise_supporting_identifier(getattr(step, "step_id", ""))
        step_id = re.sub(r"^\d+_", "", step_id)
        if step_id.endswith("_figure"):
            step_id = step_id[: -len("_figure")]
        if step_id in cls._SUPPORTING_ARTIFACT_IDS:
            return True
        expected_outputs = getattr(step, "expected_outputs", None) or []
        return any(
            cls._normalise_supporting_identifier(output) in cls._SUPPORTING_ARTIFACT_IDS
            for output in expected_outputs
            if str(output or "").strip().lower().startswith("figure:")
        )

    @classmethod
    def _contract_looks_supporting_figure(
        cls, raw: Dict[str, Any], figure_id: str
    ) -> bool:
        """True when the CONTRACT itself identifies a supporting audit/QC figure.

        The real figures.skill call sites do not thread the step, so an exact
        normalized figure_id remains as a compatibility signal for separately
        registered supporting artifacts. Panel roles are handled structurally
        by :meth:`_is_result_like_contract`; titles, claims, and identifier
        substrings never grant this exemption.
        """
        fid = cls._normalise_supporting_identifier(figure_id)
        return fid in cls._SUPPORTING_ARTIFACT_IDS

    @staticmethod
    def _contract_text(raw: Dict[str, Any]) -> str:
        parts: List[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(raw)
        return "\n".join(parts)

    @staticmethod
    def _reader_facing_text(raw: Dict[str, Any]) -> str:
        parts = [
            str(raw.get("title") or ""),
            str(raw.get("core_claim") or ""),
            str(raw.get("statistics_note") or ""),
        ]
        panels = raw.get("panels")
        if isinstance(panels, list):
            for panel in panels:
                if not isinstance(panel, dict):
                    continue
                parts.extend(
                    [
                        str(panel.get("title") or ""),
                        str(panel.get("claim") or ""),
                        str(panel.get("review_risk") or ""),
                    ]
                )
        return "\n".join(part for part in parts if part)

def _figure_source_compare___key_set(frame: pd.DataFrame, *, key_cols: Any) -> Set[tuple[str, ...]]:
    return set(
        frame[list(key_cols)]
        .dropna()
        .astype(str)
        .itertuples(index=False, name=None)
    )


def _figure_source_compare___format_key(row: pd.Series, *, key_cols: Any) -> str:
    return "|".join(str(row[col]) for col in key_cols)


def _figure_source_compare___clean_numeric(raw: pd.Series) -> pd.Series:
    text = raw.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace("−", "-", regex=False)
    text = text.str.replace(
        r"^\(([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\)$",
        r"-\1",
        regex=True,
    )
    return pd.to_numeric(text, errors="coerce").astype(float)


def _figure_source_compare___is_value_column(frame: pd.DataFrame, col: str, *, _clean_numeric: Any, cls: Any, ignored_for_dynamic_numeric: Any, text_name: Any, value_name: Any) -> bool:
    if col in ignored_for_dynamic_numeric or col in cls._TEXT_COLUMNS:
        return False
    raw = frame[col]
    if pd.api.types.is_bool_dtype(raw) or str(col).lower() in {
        "is_continuous",
        "treated",
    }:
        return False
    present = raw.notna() & raw.astype(str).str.strip().ne("")
    if not present.any():
        return False
    # CSV round-trips commonly represent nullable boolean metadata as
    # object dtype (for example ``[False, NaN]``). A name such as
    # ``estimate_identical_to_primary`` contains the token
    # ``estimate`` but remains a flag, not a numeric result. Treating
    # it as numeric turns two identical ``False`` values into matching
    # parse failures and falsely rejects an exact parent projection.
    if pd.api.types.infer_dtype(raw[present], skipna=True) == "boolean":
        return False
    parsed = _clean_numeric(raw[present])
    numeric_evidence = bool(
        pd.api.types.is_numeric_dtype(raw) or parsed.notna().all()
    )
    # Text-like suffixes normally identify labels/roles rather than
    # values.  A name that also declares a value role (for example a
    # numeric ``estimate_label``) must not escape verification merely
    # because it contains ``label``.
    if text_name.search(str(col).lower()) and not (
        value_name.search(str(col).lower()) and numeric_evidence
    ):
        return False
    return bool(
        col in cls._NUMERIC_COLUMNS
        or numeric_evidence
        or value_name.search(str(col).lower())
    )


def _figure_source_compare___merged_source(col: str, *, merged: Any, upstream: Any) -> pd.Series:
    suffixed = f"{col}_source"
    return (
        merged[suffixed]
        if col in upstream.columns and suffixed in merged.columns
        else merged[col]
    )


def _figure_source_compare___merged_upstream(col: str, *, merged: Any, source: Any) -> pd.Series:
    suffixed = f"{col}_upstream"
    return (
        merged[suffixed]
        if col in source.columns and suffixed in merged.columns
        else merged[col]
    )


def _figure_source_compare___numeric_comparison(source_name: str, upstream_name: str, *, _clean_numeric: Any, _merged_source: Any, _merged_upstream: Any, cls: Any) -> tuple[bool, bool, pd.Series, pd.Series, pd.Series, pd.Series]:
    left_raw = _merged_source(source_name)
    right_raw = _merged_upstream(upstream_name)
    left_present = left_raw.notna() & left_raw.astype(str).str.strip().ne("")
    right_present = right_raw.notna() & right_raw.astype(str).str.strip().ne("")
    left = _clean_numeric(left_raw)
    right = _clean_numeric(right_raw)
    left_finite = left.notna() & left.map(math.isfinite)
    right_finite = right.notna() & right.map(math.isfinite)
    comparable = left_present & right_present & left_finite & right_finite
    abs_tolerance = (
        cls._PERCENTAGE_ABS_TOL
        if any(
            token in name.lower()
            for name in (source_name, upstream_name)
            for token in ("_pct", "percent")
        )
        else cls._DEFAULT_NUMERIC_ABS_TOL
    )
    diff = (left - right).abs()
    same_nonfinite = (
        left_present & right_present & left.eq(right) & ~left_finite & ~right_finite
    )
    # Equal nonnumeric receipt text is faithful, not a parse failure.
    same_semantic_text = (
        left_present & right_present & left.isna() & right.isna()
        & left_raw.astype(str).str.strip().eq(right_raw.astype(str).str.strip())
    )
    parse_failure = ((left_present & left.isna()) | (right_present & right.isna())) & ~same_semantic_text
    bad = (
        (left_present ^ right_present)
        | parse_failure
        | (comparable & (diff > abs_tolerance))
        | (
            left_present
            & right_present
            & ~comparable
            & ~same_semantic_text
            & ~same_nonfinite
            & ~parse_failure
        )
    )
    return (
        bool((comparable | same_semantic_text).any() and not bad.any()),
        bool(bad.any()),
        bad,
        left,
        right,
        diff,
    )


def _figure_source_compare___value_family(col: str) -> str:
    """Classify value-column names for safe cross-name matching.

    Row-aligned equality proves that a numeric vector came from the
    parent table, but it does not by itself prove semantic identity. A
    count vector must not authenticate an ``estimate`` merely because
    the numbers happen to coincide.  Keep the families deliberately
    small and case-neutral; same-name comparisons remain authoritative.
    """

    name = re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
    tokens = set(name.split("_")) if name else set()
    if tokens & {"percent", "percentage", "pct"}:
        return "percent"
    if (
        name in {"n", "count", "denominator", "sample_size"}
        or tokens & {"count", "events", "deaths"}
        or "denominator" in tokens
        or "sample" in tokens
        and "size" in tokens
        or name.startswith("n_")
        or name.endswith("_n")
    ):
        return "count"
    if tokens & {
        "risk",
        "rate",
        "proportion",
        "prevalence",
        "incidence",
        "probability",
    }:
        return "rate"
    if ("ci" in tokens and tokens & {"low", "lower", "lcl"}) or tokens & {
        "lcl"
    }:
        return "ci_low"
    if ("ci" in tokens and tokens & {"high", "upper", "ucl"}) or tokens & {
        "ucl"
    }:
        return "ci_high"
    if name in {
        "se",
        "stderr",
        "std_err",
        "std_error",
        "standard_err",
        "standard_error",
    } or (
        bool(tokens & {"std", "standard"}) and bool(tokens & {"err", "error"})
    ):
        return "standard_error"
    if name in {"p", "pval", "p_val", "pvalue", "p_value"} or (
        "p" in tokens and bool(tokens & {"val", "value"})
    ):
        return "p_value"
    if tokens & {"mean", "median", "quantile"}:
        return "location_summary"
    if tokens & {"order", "position", "rank"}:
        return "ordering"
    if tokens & {"ratio", "odds", "hazard"} or name in {
        "or",
        "hr",
        "rr",
    }:
        return "ratio"
    if tokens & {"estimate", "effect", "statistic"}:
        return "generic_estimate"
    if "value" in tokens:
        return "generic_value"
    return "other_numeric"


def _figure_source_compare___structured_source_family(source_name: str, *, _value_family: Any, cls: Any, source: Any) -> str:
    family = _value_family(source_name)
    if family != "generic_estimate":
        return family
    semantic_values: Set[str] = set()
    for semantic_col in ("value_type", "estimate_type", "effect_scale"):
        if semantic_col not in source.columns:
            continue
        semantic_values.update(
            cls._normalise(item)
            for item in source[semantic_col].dropna().astype(str)
            if str(item).strip()
        )
    semantic_families: Set[str] = set()
    if semantic_values & {
        "distribution",
        "continuous_distribution",
        "distribution_mean",
        "distribution_median",
        "location_summary",
        "mean",
        "median",
        "quantile",
    }:
        semantic_families.add("location_summary")
    if semantic_values & {
        "risk",
        "rate",
        "probability",
        "prevalence",
        "incidence",
        "absolute_risk",
        "event_rate",
        "mortality_rate",
    }:
        semantic_families.add("rate")
    if semantic_values & {
        "odds",
        "hazard",
        "ratio",
        "association",
        "effect",
        "odds_ratio",
        "hazard_ratio",
        "risk_ratio",
        "association_estimate",
        "effect_estimate",
        "or",
        "hr",
        "rr",
    }:
        semantic_families.add("ratio")
    if len(semantic_families) == 1:
        return next(iter(semantic_families))
    return family


def _figure_source_compare___cross_name_families_compatible(source_name: str, upstream_name: str, *, _structured_source_family: Any, _value_family: Any) -> bool:
    source_family = _structured_source_family(source_name)
    upstream_family = _value_family(upstream_name)
    # Unknown/generic numeric names have no semantic contract.  Exact
    # same-name columns were already handled above; across names, an
    # equal vector such as ``display_metric`` == ``age`` must not be
    # treated as proof that the displayed quantity came from the
    # claimed upstream measure.
    if source_family in {
        "generic_value",
        "other_numeric",
    } or upstream_family in {"generic_value", "other_numeric"}:
        return False
    # Ordering is presentation metadata, not a scientific value
    # family. Only the explicit ``plot_*`` derivation below may bind
    # it to a complete row-aligned upstream ordering vector.
    if "ordering" in {source_family, upstream_family}:
        return False
    if "count" in {source_family, upstream_family}:
        return source_family == upstream_family
    # Percent-labelled columns require either a same-family raw vector
    # or the explicit derived percentage logic below; they cannot
    # silently inherit the scale of a 0-1 risk/rate column.
    if "percent" in {source_family, upstream_family}:
        return source_family == upstream_family
    inferential_specific = {
        "ci_low",
        "ci_high",
        "standard_error",
        "p_value",
    }
    if (
        source_family in inferential_specific
        or upstream_family in inferential_specific
    ):
        return source_family == upstream_family
    # A presentation-neutral estimate may project a rate/risk or ratio
    # when its complete vector matches.  Location summaries require a
    # structured source semantic (value_type/estimate_type/effect_scale)
    # so an unrelated mean-age vector cannot authenticate an outcome
    # estimate merely because the numbers happen to coincide.
    # e.g. a renderer's ``estimate`` may faithfully project an
    # upstream ``mortality_rate``.
    if source_family == "generic_estimate":
        return upstream_family in {"rate", "ratio"}
    return source_family == upstream_family


def _figure_source_compare___explicit_semantic_target_columns(source_name: str, *, _cross_name_families_compatible: Any, cls: Any, source: Any, upstream_value_columns: Any) -> List[str]:
    """Resolve a concrete source declaration to its named parent value.

    A declaration such as ``value_type=mortality_rate`` is stronger
    than the broad ``rate`` family inferred from it. When that exact
    normalised value column exists upstream, bind to it so a sibling
    rate/effect column with coincident values cannot authenticate the
    claim. Generic declarations that do not name an upstream column
    retain the family-level compatibility path below.
    """

    declared = {
        cls._normalise(item)
        for semantic_col in ("value_type", "estimate_type", "effect_scale")
        if semantic_col in source.columns
        for item in source[semantic_col].dropna().astype(str)
        if str(item).strip()
    }
    if not declared:
        return []
    return sorted(
        upstream_col
        for upstream_col in upstream_value_columns
        if cls._normalise(upstream_col) in declared
        and _cross_name_families_compatible(source_name, upstream_col)
    )


def _figure_source_compare___derived_matches(source_col: str, expected_vectors: Sequence[pd.Series], tolerance: Optional[float], _clean_numeric: Any, _merged_source: Any, cls: Any) -> bool:
    tolerance = cls._DEFAULT_NUMERIC_ABS_TOL if tolerance is None else tolerance
    left_raw = _merged_source(source_col)
    left_present = left_raw.notna() & left_raw.astype(str).str.strip().ne("")
    left = _clean_numeric(left_raw)
    if not left_present.any() or (left_present & left.isna()).any():
        return False
    for expected in expected_vectors:
        expected = pd.to_numeric(expected, errors="coerce").astype(float)
        comparable = (
            left_present
            & left.notna()
            & expected.notna()
            & left.map(math.isfinite)
            & expected.map(math.isfinite)
        )
        matched = comparable & ((left - expected).abs() <= tolerance)
        if comparable.any() and (matched | ~left_present).all():
            return True
    return False

def _figure_audit__credit_table_source(source_path: Path, source_frame: pd.DataFrame, table_paths: Set[Path], *, completed_step_records: Any, matched_figure_obligations: Any, required_figure_obligations: Any, run_dir: Any, self: Any, source_figure_products: Any, step: Any, table_frames: Any, table_products: Any) -> None:
    for table_path in table_paths:
        resolved = table_path.resolve()
        product = table_products.get(resolved, f"table:{table_path.stem}")
        frame = table_frames.get(resolved)
        if frame is None:
            try:
                frame = self._read_tabular(resolved)
            except Exception:
                continue
            table_frames[resolved] = frame
        semantic_product = self._contract_scoped_effect_product(
            product=product,
            source_frame=source_frame,
            upstream_frame=frame,
            upstream_step_id=self._table_step_id(table_path, run_dir=run_dir),
            completed_step_records=completed_step_records,
        )
        for figure in source_figure_products.get(source_path.resolve(), set()):
            if figure not in required_figure_obligations:
                continue
            family = self._figure_result_family(
                step=step,
                figure_product=figure,
            )
            if family == "prediction":
                matched_figure_obligations[figure].update(
                    required_figure_obligations[figure]
                    & self._prediction_source_obligations(
                        product=product,
                        frame=frame,
                    )
                )
            elif self._source_supports_figures(
                step=step,
                product=semantic_product,
                frame=frame,
                figure_products=[figure],
                require_all=True,
            ):
                matched_figure_obligations[figure].update(
                    required_figure_obligations[figure]
                )


def _figure_audit__credit_statistic_source(source_path: Path, statistic_ids: Set[str], *, matched_figure_obligations: Any, required_figure_obligations: Any, required_statistics: Any, self: Any, source_figure_products: Any, step: Any) -> None:
    for statistic_id in statistic_ids:
        product_name, expected = required_statistics[statistic_id]
        product = f"statistic:{product_name}"
        for figure in source_figure_products.get(source_path.resolve(), set()):
            if figure not in required_figure_obligations:
                continue
            family = self._figure_result_family(
                step=step,
                figure_product=figure,
            )
            if family == "prediction":
                matched_figure_obligations[figure].update(
                    required_figure_obligations[figure]
                    & self._prediction_source_obligations(
                        product=product,
                        frame=None,
                        statistic_value=expected,
                    )
                )
            elif self._source_supports_figures(
                step=step,
                product=product,
                frame=None,
                figure_products=[figure],
                require_all=True,
            ):
                matched_figure_obligations[figure].update(
                    required_figure_obligations[figure]
                )


