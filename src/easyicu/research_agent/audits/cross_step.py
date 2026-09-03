"""Cross-step lock/registered-output/fraction/contract/source-status validators."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import pandas as pd

from ..contracts.fraction_scale import (
    is_scale_descriptor_field,
    normalize_metric_key,
)
from ..contracts.model_tokens import canonical_association_method
from ..contracts.model_contract_match import reported_model_requirement_fields
from ..schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT,
    PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    AnalysisStep,
    ResearchContext,
    ValidationFinding,
)
from ..authority.runtime_artifacts import (
    current_successful_step_records,
)

# ---------------------------------------------------------------------------
# CohortAuditor
# ---------------------------------------------------------------------------


# Patient-level identifier column names. Their presence means a cohort can
# be reasoned about at the patient level (within-patient non-independence,
# first-stay selection). Stay-level keys (stay_id, icustay_id) and

# ---------------------------------------------------------------------------
# Cross-step validators
# ---------------------------------------------------------------------------


class CrossStepCohortLockValidator:
    """Prevent a fixed-cohort step from silently re-filtering prior rows.

    Some reconciliation steps explicitly promise to keep the completed
    analytic cohort fixed.  A generated script must then operate on the
    already-materialised cohort rather than add a new eligibility rule.  This
    gate activates only for that explicit intent and compares a
    machine-readable current cohort count with the most recent successful
    analysis-step count.  ``n_universe`` is intentionally not accepted as the
    current count because it can remain unchanged while ``n_final_cohort`` is
    silently reduced.
    """

    name = "cross_step_cohort_lock"

    _SUCCESSFUL_STATUSES = {
        "ok",
        "complete",
        "completed",
        "repaired",
        "runner_repaired",
    }
    _COUNT_PATHS: tuple[tuple[str, ...], ...] = (
        # First, because it is the only entry the HOST writes. Everything below
        # is a spelling this validator hopes a producer happened to choose, and
        # a hoped-for name can mean something else in another producer's
        # vocabulary. ``n_total`` is exactly that: 8 recorded robustness
        # summaries use it for the analysis cohort, and one uses it for the
        # number of variants compared -- so the cohort-lock guard read an
        # analysis cohort of 2 against a locked 1,000, with the correct value
        # sitting in the same file under this key.
        #
        # Measured over 819 recorded summaries: 53 carry ``analysis_cohort_n``.
        # Of those the search below lands on ``n_total`` 35 times, ``cohort_n``
        # 3 times, and on NOTHING 15 times -- so in 15 the guard was blind while
        # the value it needed was present. Putting this first corrects the one
        # disagreement, gives the 15 a reading, and leaves the 37 that already
        # agree untouched.
        ("analysis_cohort_n",),
        ("n_final_cohort",),
        ("final_analytic_cohort_n",),
        ("final_cohort_n",),
        ("cohort_count_final",),
        ("cohort", "n_final_rows"),
        ("cohort", "final_analytic_cohort_n"),
        ("locked_cohort", "n_output"),
        ("locked_cohort", "n_final_rows"),
        ("analytic_cohort_n",),
        ("adult_analytic_cohort_n",),
        ("adult_cohort_n",),
        ("cohort_definition", "adult_analytic_cohort_n"),
        ("cohort_definition", "analytic_cohort_n"),
        ("cohort_definition", "adult_cohort_n"),
        ("cohort_counts", "n_adult_analysis_cohort_rows"),
        ("cohort_counts", "n_analytic_cohort_rows"),
        ("cohort_n",),
        ("n_cohort",),
        ("n_total",),
        # The deterministic probe uses ``n_rows`` and is a final fallback when
        # no later analysis summary exposes a cohort count.
        ("n_rows",),
    )

    @staticmethod
    def _as_count(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @classmethod
    def _extract_count(cls, summary: Dict[str, Any]) -> Optional[tuple[int, str]]:
        for path in cls._COUNT_PATHS:
            value: Any = summary
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    break
                value = value[key]
            else:
                count = cls._as_count(value)
                if count is not None:
                    return count, ".".join(path)
        return None

    @staticmethod
    def _requires_fixed_cohort(step: AnalysisStep) -> bool:
        text = re.sub(r"\s+", " ", str(step.intent or "").strip().lower())
        if not text:
            return False

        # A true alternative-cohort/sensitivity step is allowed to change N.
        # Do not infer this from a generated summary's analysis_family: the M1
        # failure that motivated the gate mislabeled a fixed reconciliation as
        # cohort-definition sensitivity.
        varying_cohort = any(
            re.search(pattern, text)
            for pattern in (
                r"\balternative cohort(?: definition)?\b",
                r"\bvary(?:ing)? (?:the )?cohort(?: definition)?\b",
                r"\bcohort(?: definition)? sensitivity\b",
                r"\bcompare (?:an |the )?alternative eligibility\b",
            )
        )
        if varying_cohort:
            return False

        return any(
            re.search(pattern, text)
            for pattern in (
                r"\bkeep(?:ing)?\b.{0,160}\bcohort\b.{0,160}\b(?:fixed|unchanged|constant)\b",
                r"\b(?:preserve|preserving|hold|holding)\b.{0,120}\bcohort\b.{0,120}\b(?:fixed|unchanged|constant)\b",
                r"\b(?:fixed|locked|unchanged)\b.{0,80}\b(?:completed|current|existing|analytic|analysis)?\s*cohort\b",
                r"\b(?:completed|current|existing|analytic|analysis)?\s*cohort\b.{0,80}\b(?:fixed|locked|unchanged)\b",
                r"\b(?:do not|don't|must not|without)\b.{0,100}\b(?:redefine|change|refilter|restrict)\w*\b.{0,100}\b(?:the )?cohort\b",
            )
        )

    @classmethod
    def _latest_prior_lock(
        cls, completed_step_records: Sequence[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for record in reversed(completed_step_records):
            step_id = str(record.get("step_id") or "prior_step")
            normalised_step_id = re.sub(
                r"[^a-z0-9]+", "_", step_id.strip().lower()
            ).strip("_")
            if normalised_step_id.endswith("_figure"):
                continue

            record_status = str(record.get("status") or "").strip().lower()
            if record_status and record_status not in cls._SUCCESSFUL_STATUSES:
                continue
            summary = record.get("step_summary")
            if not isinstance(summary, dict) or summary.get("rendering_only") is True:
                continue
            summary_status = str(summary.get("status") or "").strip().lower()
            if summary_status and summary_status not in cls._SUCCESSFUL_STATUSES:
                continue
            extracted = cls._extract_count(summary)
            if extracted is None:
                continue
            count, path = extracted
            return {"cohort_n": count, "summary_path": path, "step_id": step_id}
        return None

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[ValidationFinding]:
        normalised_step_id = re.sub(
            r"[^a-z0-9]+", "_", str(step.step_id or "").strip().lower()
        ).strip("_")
        if step_summary.get("rendering_only") is True or normalised_step_id.endswith(
            "_figure"
        ):
            # A split figure step reads registered parent outputs and cannot
            # redefine eligibility. Requiring it to restate the analytic N
            # turns the stock "do not redefine the cohort" rendering prompt
            # into a false cohort-drift error and sends valid figures through
            # an irrelevant model-code repair loop.
            return []
        if not self._requires_fixed_cohort(step):
            return []
        prior = self._latest_prior_lock(completed_step_records)
        if prior is None:
            return []

        current = self._extract_count(step_summary)
        if current is None:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Fixed-cohort step {step.step_id} does not report a "
                        "machine-readable final analytic cohort count. Report "
                        f"the unchanged cohort N locked by completed step "
                        f"{prior['step_id']} ({prior['cohort_n']}) and do not "
                        "re-derive eligibility inside this step."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "expected_cohort_n": prior["cohort_n"],
                        "expected_from_step": prior["step_id"],
                        "expected_summary_path": prior["summary_path"],
                        "reported_summary_path": None,
                    },
                )
            ]

        reported_n, reported_path = current
        if reported_n == prior["cohort_n"]:
            return []
        return [
            ValidationFinding(
                validator=self.name,
                severity="error",
                message=(
                    f"Fixed-cohort drift for step {step.step_id}: "
                    f"{reported_path} reports {reported_n}, but completed step "
                    f"{prior['step_id']} locked {prior['cohort_n']}. Treat the "
                    "input cohort as already eligible; remove any new age, "
                    "length-of-stay, identifier, outcome-availability, or other "
                    "row filter and recompute this step on the locked cohort."
                ),
                detail={
                    "step_id": step.step_id,
                    "reported_cohort_n": reported_n,
                    "reported_summary_path": reported_path,
                    "expected_cohort_n": prior["cohort_n"],
                    "expected_from_step": prior["step_id"],
                    "expected_summary_path": prior["summary_path"],
                },
            )
        ]


class CrossStepRegisteredOutputValidator:
    """Reject a false "upstream table unavailable" reconciliation gap.

    Generated reconciliation code sometimes finds the correct upstream step
    but over-filters its evidence records by a guessed semantic filename and
    therefore declares an existing registered table unavailable.  This gate
    compares that explicit availability claim with the upstream step record's
    table evidence and machine-readable output-file declarations.  Genuine
    gaps remain allowed when the completed upstream step registered no table.
    """

    name = "cross_step_registered_output"
    _SUCCESSFUL_STATUSES = CrossStepCohortLockValidator._SUCCESSFUL_STATUSES
    _TABLE_SUFFIXES = (".csv", ".parquet", ".tsv", ".feather", ".xlsx")

    @classmethod
    def _availability_blocks(cls, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                upstream_step = value.get("upstream_step")
                availability_key = next(
                    (
                        key
                        for key in (
                            "source_table_available",
                            "registered_output_readable",
                            "available",
                        )
                        if isinstance(value.get(key), bool)
                    ),
                    None,
                )
                if isinstance(upstream_step, str) and availability_key is not None:
                    blocks.append(
                        {
                            "upstream_step": upstream_step,
                            "available": value[availability_key],
                            "availability_key": availability_key,
                            "path": ".".join(path) or "step_summary",
                            "reported_path": value.get("source_table_path")
                            or value.get("registered_output_path")
                            or value.get("path"),
                        }
                    )
                for key, child in value.items():
                    visit(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return blocks

    @classmethod
    def _table_artifacts(cls, record: Dict[str, Any]) -> List[str]:
        artifacts: List[str] = []
        for evidence_id in record.get("evidence_ids") or []:
            if isinstance(evidence_id, str) and evidence_id.startswith("table_"):
                artifacts.append(evidence_id)

        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            return sorted(set(artifacts))
        output_files = summary.get("output_files")
        if isinstance(output_files, Mapping):
            for raw_path in output_files.values():
                if isinstance(raw_path, str) and raw_path.strip().lower().endswith(
                    cls._TABLE_SUFFIXES
                ):
                    artifacts.append(raw_path.strip())
        return sorted(set(artifacts))

    @classmethod
    def _upstream_table_lock(
        cls,
        upstream_step: str,
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for record in reversed(completed_step_records):
            if str(record.get("step_id") or "") != upstream_step:
                continue
            record_status = str(record.get("status") or "").strip().lower()
            if record_status and record_status not in cls._SUCCESSFUL_STATUSES:
                continue
            summary = record.get("step_summary")
            if isinstance(summary, dict):
                summary_status = str(summary.get("status") or "").strip().lower()
                if summary_status and summary_status not in cls._SUCCESSFUL_STATUSES:
                    continue
            artifacts = cls._table_artifacts(record)
            if artifacts:
                return {"step_id": upstream_step, "table_artifacts": artifacts}
            return None
        return None

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        for block in self._availability_blocks(step_summary):
            if block["available"]:
                continue
            prior = self._upstream_table_lock(
                block["upstream_step"], completed_step_records
            )
            if prior is None:
                continue
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Registered upstream table was falsely reported "
                        f"unavailable in step {step.step_id}: completed step "
                        f"{prior['step_id']} registered table evidence "
                        f"{prior['table_artifacts']}. Filter manifest records "
                        "by the exact produced_by_step and table kind, resolve "
                        "relative_path from the run directory, and use the sole "
                        "compatible table even when its filename does not repeat "
                        "the current step's semantic label."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "summary_path": block["path"],
                        "availability_key": block["availability_key"],
                        "reported_path": block["reported_path"],
                        "upstream_step": prior["step_id"],
                        "registered_table_artifacts": prior["table_artifacts"],
                    },
                )
            )
        return findings


class StepSummaryFractionValidator:
    """Enforce [0, 1] for probability-like machine-summary fields."""

    name = "step_summary_fraction_scale"

    @classmethod
    def _invalid_fraction_values(
        cls, summary: Dict[str, Any]
    ) -> List[tuple[str, float, str]]:
        invalid: List[tuple[str, float, str]] = []

        effect_scale_names = {
            "hr",
            "or",
            "rd",
            "rr",
            "risk_ratio",
            "relative_risk",
            "odds_ratio",
            "hazard_ratio",
            "risk_difference",
        }

        def is_effect_scale_field(key: Any) -> bool:
            name = normalize_metric_key(key)
            ci_base = re.sub(r"_(?:ci_)?(?:low|high|lower|upper)$", "", name)
            return ci_base in effect_scale_names or any(
                ci_base.endswith(f"_{effect_scale}")
                for effect_scale in effect_scale_names
            )

        def bounded_field_kind(key: Any) -> Optional[str]:
            """Identify fields whose *value* is contractually in [0, 1].

            Do not propagate merely because a structural or methodological key
            contains the substring ``fraction``.  Names such as
            ``fractional_polynomial_power`` and
            ``sampling_fraction_denominator`` are not values on a [0, 1]
            scale.  A mapping directly owned by a true ``*_fraction`` field is
            still allowed to encode category -> fraction values.
            """

            name = normalize_metric_key(key)
            if not name or any(
                token in name for token in ("pct", "percent", "percentage")
            ):
                return None
            if name.startswith("fractional_") or name == "fractional":
                return None
            if name.endswith(("_numerator", "_denominator")):
                return None
            ci_base = re.sub(r"_(?:ci_)?(?:low|high|lower|upper)$", "", name)
            if is_effect_scale_field(key):
                return None
            if ci_base == "at_risk" or ci_base.endswith("_at_risk"):
                # Survival risk-set counts/statuses are not probabilities.
                return None
            if ci_base in {
                "attributable_fraction",
                "population_attributable_fraction",
            }:
                # These effect measures can legitimately be negative.
                return None
            if ci_base == "fraction" or ci_base.endswith("_fraction"):
                return "fraction"
            if ci_base == "probability" or ci_base.endswith("_probability"):
                return "probability"
            if ci_base == "prevalence" or ci_base.endswith("_prevalence"):
                return "prevalence"
            if ci_base.startswith("prevalence_ci"):
                return "prevalence"
            if ci_base == "risk" or ci_base.endswith("_risk"):
                if ci_base in {"excess_risk", "attributable_risk"} or ci_base.endswith(
                    ("_excess_risk", "_attributable_risk")
                ):
                    return None
                return "risk"
            return None

        structural_children = {
            "count",
            "cases",
            "deaths",
            "denominator",
            "event_n",
            "events",
            "n",
            "nobs",
            "non_events",
            "numerator",
            "observations",
            "patients",
            "sample_size",
            "stays",
            "subjects",
            "survivors",
            "total",
            "total_n",
        }
        structural_suffixes = (
            "_count",
            "_denominator",
            "_draws",
            "_folds",
            "_iterations",
            "_n",
            "_numerator",
            "_replicates",
            "_rows",
            "_sample_size",
        )
        structural_prefixes = ("n_", "num_", "number_")
        coordinate_children = {
            "category",
            "category_code",
            "code",
            "cutpoint",
            "decimal_places",
            "df",
            "digits",
            "group",
            "group_id",
            "id",
            "index",
            "label",
            "level",
            "level_id",
            "name",
            "order",
            "precision",
            "rank",
            "random_seed",
            "seed",
            "stratum",
            "stratum_id",
            "threshold",
            "timepoint",
            "timepoint_index",
            "version",
        }
        coordinate_suffixes = (
            "_category",
            "_code",
            "_cutpoint",
            "_days",
            "_places",
            "_group",
            "_hours",
            "_id",
            "_index",
            "_label",
            "_level",
            "_minutes",
            "_months",
            "_name",
            "_order",
            "_precision",
            "_rank",
            "_seconds",
            "_seed",
            "_stratum",
            "_threshold",
            "_timepoint",
            "_version",
            "_years",
        )
        scalar_value_children = {
            "estimate",
            "fraction",
            "point_estimate",
            "result",
            "value",
        }
        generic_ci_children = {"ci_low", "ci_high", "ci_lower", "ci_upper"}
        bounded_scale_descriptors = {
            "0_1",
            "dimensionless",
            "proportion",
            "unit_interval",
            "unitless",
            "zero_to_one",
        }
        non_bounded_scale_descriptors = {
            "aic",
            "attributable_fraction",
            "attributable_risk",
            "auc",
            "beta",
            "bic",
            "c_statistic",
            "coefficient",
            "count",
            "counts",
            "deviance",
            "excess_risk",
            "frequency",
            "hazard_ratio",
            "hr",
            "iqr",
            "log_likelihood",
            "log_odds",
            "logit",
            "mae",
            "mean",
            "median",
            "mse",
            "n",
            "odds_ratio",
            "or",
            "pct",
            "percent",
            "percentage",
            "population_attributable_fraction",
            "rd",
            "relative_risk",
            "risk_difference",
            "risk_ratio",
            "rmse",
            "rr",
            "sample_size",
            "sd",
            "se",
            "standard_deviation",
            "standard_error",
            "variance",
        }
        domain_changing_tokens = {
            "audit",
            "audits",
            "bootstrap",
            "bootstraps",
            "coefficient",
            "coefficients",
            "count",
            "counts",
            "diagnostic",
            "diagnostics",
            "distribution",
            "distributions",
            "draw",
            "draws",
            "effect",
            "effects",
            "fit",
            "fits",
            "format",
            "formats",
            "formatting",
            "fold",
            "folds",
            "iteration",
            "iterations",
            "metadata",
            "option",
            "options",
            "parameter",
            "parameters",
            "percentile",
            "percentiles",
            "quantile",
            "quantiles",
            "replicate",
            "replicates",
            "rounding",
            "runtime",
            "sample",
            "samples",
            "size",
            "sizes",
            "setting",
            "settings",
            "statistic",
            "statistics",
            "timing",
        }

        def is_structural_child(name: str) -> bool:
            return (
                name in structural_children
                or name.startswith(structural_prefixes)
                or name.endswith(structural_suffixes)
            )

        def blocks_inherited_context(key: Any, name: str) -> bool:
            ci_base = re.sub(r"_(?:ci_)?(?:low|high|lower|upper)$", "", name)
            name_tokens = set(name.split("_"))
            return (
                is_structural_child(name)
                or name in coordinate_children
                or (name.endswith(coordinate_suffixes) and not name.startswith("by_"))
                or name in non_bounded_scale_descriptors
                or bool(name_tokens & domain_changing_tokens)
                or any(token in name for token in ("pct", "percent", "percentage"))
                or name.startswith("fractional_")
                or is_effect_scale_field(key)
                or ci_base == "at_risk"
                or ci_base.endswith("_at_risk")
                or ci_base
                in {
                    "attributable_fraction",
                    "attributable_risk",
                    "excess_risk",
                    "population_attributable_fraction",
                }
            )

        def mapping_declares_non_bounded_scale(value: Any) -> bool:
            if not isinstance(value, dict):
                return False
            for key, descriptor in value.items():
                if not is_scale_descriptor_field(key) or not isinstance(
                    descriptor, str
                ):
                    continue
                if descriptor.strip() == "%":
                    return True
                descriptor_name = normalize_metric_key(descriptor)
                if not descriptor_name:
                    continue
                if (
                    bounded_field_kind(descriptor_name) is not None
                    or descriptor_name in bounded_scale_descriptors
                ):
                    continue
                if (
                    descriptor_name in non_bounded_scale_descriptors
                    or is_effect_scale_field(descriptor_name)
                ):
                    return True
            return False

        def mapping_has_generic_metric_payload(value: Any) -> bool:
            return isinstance(value, dict) and any(
                normalize_metric_key(key) in scalar_value_children | generic_ci_children
                for key in value
            )

        def visit(
            value: Any,
            path: tuple[str, ...] = (),
            bounded_context: Optional[str] = None,
        ) -> None:
            if isinstance(value, dict):
                local_non_bounded_scale = bool(
                    bounded_context
                    and mapping_has_generic_metric_payload(value)
                    and mapping_declares_non_bounded_scale(value)
                )
                sibling_kinds = {
                    kind
                    for key in value
                    if (kind := bounded_field_kind(key)) is not None
                }
                sibling_context = (
                    next(iter(sibling_kinds)) if len(sibling_kinds) == 1 else None
                )
                has_effect_scale_sibling = any(
                    is_effect_scale_field(key) for key in value
                )
                for key, child in value.items():
                    normalised = normalize_metric_key(key)
                    key_context = bounded_field_kind(key)
                    inherited_context = bounded_context
                    if inherited_context:
                        if blocks_inherited_context(key, normalised):
                            inherited_context = None
                        elif local_non_bounded_scale and normalised in (
                            scalar_value_children | generic_ci_children
                        ):
                            inherited_context = None
                    if (
                        normalised in generic_ci_children
                        and (sibling_context or bounded_context)
                        and not has_effect_scale_sibling
                        and not local_non_bounded_scale
                    ):
                        key_context = sibling_context or bounded_context
                    visit(
                        child,
                        (*path, str(key)),
                        inherited_context or key_context,
                    )
                return
            if isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)), bounded_context)
                return
            if not bounded_context or isinstance(value, bool) or value is None:
                return
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                invalid.append((".".join(path), number, bounded_context))

        visit(summary)
        return invalid

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        for path, value, metric_kind in self._invalid_fraction_values(step_summary):
            roundoff_sized_overflow = bool(
                math.isfinite(value)
                and (value > 1.0 or value < 0.0)
                and min(abs(value), abs(value - 1.0)) <= 1e-12
            )
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Bounded {metric_kind} mismatch in step {step.step_id}: "
                        f"{path}={value} is outside [0, 1]. Do not retain even "
                        "roundoff-sized overflow in a registered summary; "
                        "normalize deterministically before writing the output."
                    ),
                    detail={
                        "issue": "bounded_metric_out_of_range",
                        "step_id": step.step_id,
                        "summary_path": path,
                        "metric_kind": metric_kind,
                        "reported_value": value,
                        "expected_min": 0.0,
                        "expected_max": 1.0,
                        "roundoff_sized_overflow": roundoff_sized_overflow,
                    },
                )
            )
        findings.extend(self._ambiguous_percent_pair_findings(step, step_summary))
        return findings

    @classmethod
    def _ambiguous_percent_pair_findings(
        cls, step: AnalysisStep, summary: Dict[str, Any]
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []

        def numeric_mapping(value: Any) -> Optional[Dict[str, float]]:
            if not isinstance(value, dict) or not value:
                return None
            parsed: Dict[str, float] = {}
            for key, raw in value.items():
                if isinstance(raw, bool):
                    return None
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    return None
                if not pd.notna(number):
                    return None
                parsed[str(key)] = number
            return parsed

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    key_text = str(key)
                    if key_text.endswith("_percent"):
                        pct_key = f"{key_text}_pct"
                        left = numeric_mapping(child)
                        right = numeric_mapping(value.get(pct_key))
                        if (
                            left
                            and right
                            and left.keys() == right.keys()
                            and all(
                                abs(right[item] - 100.0 * left[item]) <= 1e-8
                                for item in left
                            )
                        ):
                            summary_path = ".".join((*path, key_text))
                            findings.append(
                                ValidationFinding(
                                    validator=cls.name,
                                    severity="error",
                                    message=(
                                        f"Ambiguous percent/fraction schema in step "
                                        f"{step.step_id}: {summary_path} contains "
                                        "proportions while its sibling *_pct contains "
                                        "the same values multiplied by 100. Rename the "
                                        "first field to *_fraction and keep the second "
                                        "as *_pct so machine consumers cannot interpret "
                                        "a fraction as a percent."
                                    ),
                                    detail={
                                        "step_id": step.step_id,
                                        "summary_path": summary_path,
                                        "pct_summary_path": ".".join((*path, pct_key)),
                                    },
                                )
                            )
                    visit(child, (*path, key_text))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return findings


class PrimaryModelContractValidator:
    """Fail closed on complex multi-model primary-association contracts.

    This validator deliberately covers binary-logistic and continuous
    linear/quantile adjusted-association steps, not EasyICU's survival,
    prediction, mixed-effects, or clustering families. Supported complex steps
    must expose one fixed ``model_contracts`` record per attempted model and a
    term-level coefficient table for fitted models so primary/secondary roles,
    denominators, adjustment sets, and fit diagnostics are machine-verifiable.
    """

    name = "primary_model_contract"
    _REQUIRED_FIELDS = (
        "model_id",
        "exposure_source",
        "exposure_expression",
        "exposure_role",
        "analysis_role",
        "analysis_set",
        "baseline_missing_policy",
        "n",
        "event_n",
        "fit_status",
        "converged",
        "separation_detected",
        "penalized",
        "fit_method",
    )
    _TERM_ROLES = {"intercept", "exposure", "availability", "adjustment"}
    _EXPOSURE_ROLES = {"primary", "secondary"}
    _ANALYSIS_ROLES = {"primary", "secondary", "sensitivity"}
    _ANALYSIS_SETS = {"source_aware", "complete_case"}
    _BASELINE_MISSING_POLICIES = {
        "drop_missing_baseline",
        "explicit_missing_category",
    }
    _FIT_STATUSES = {"fitted", "not_fitted", "separation_no_estimate"}
    _NONFITTED_RESULT_FIELDS = (
        "estimate",
        "odds_ratio",
        "or",
        "ci_low",
        "ci_high",
        "standard_error",
        "p_value",
    )
    _CLOSED_EFFECT_METHODS = {PLANNED_MODEL_REQUIREMENTS_STEP_METHOD}
    _CLOSED_EFFECT_PRODUCTS = {
        (
            PLANNED_MODEL_REQUIREMENTS_OUTPUT_KIND,
            PLANNED_MODEL_REQUIREMENTS_OUTPUT,
        )
    }
    _OUTCOME_TYPES = {"binary", "continuous"}
    _BINARY_MODEL_FAMILIES = ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES
    _CONTINUOUS_MODEL_FAMILIES = ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES
    _CONTINUOUS_INCOMPATIBLE_SCALES = {
        "log_odds",
        "odds_ratio",
        "or",
        "log_hazard",
        "hazard_ratio",
        "hr",
    }
    _BINARY_INCOMPATIBLE_SCALES = {
        "conditional_quantile_difference",
        "median_difference",
        "median_difference_days",
        "mean_difference",
        "outcome_unit_difference",
    }
    _CONTROLLED_PENALIZED_INTERVAL_METHODS = {
        "bootstrap",
        "firth_profile",
        "profile_likelihood",
        "easyicu_penalized_hessian_v1",
    }
    _CONTROLLED_CONVERGENCE_METHODS = {
        "optimizer_success",
        "kkt_residual",
        "firth_optimizer",
        "bootstrap_refit",
    }
    _OPERATIONAL_NAME_SUFFIXES = {
        "any",
        "count",
        "ever",
        "first",
        "flag",
        "indicator",
        "last",
        "max",
        "mean",
        "measured",
        "median",
        "min",
        "n",
        "observed",
        "raw",
        "sum",
        "value",
    }
    _FIGURE_ONLY_METHODS = {
        "figure_generation",
        "plot_generation",
        "publication_figure_generation",
        "visualization",
    }
    _SOURCE_VARIABLE_SEMANTICS = (
        "term may be an encoded or transformed design column; source_variable "
        "must name the unique original authoritative cohort column"
    )

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

    @classmethod
    def _authoritative_completed_records(
        cls,
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Use current checkpoints when a status-bearing ledger is present.

        Status-less records predate the append-only execution ledger and are
        retained only as a legacy compatibility path.  In a modern ledger, a
        later failed checkpoint must revoke an earlier successful summary.
        """

        records = [
            record
            for record in (completed_step_records or [])
            if isinstance(record, Mapping)
        ]
        if not any("status" in record for record in records):
            return records
        return list(current_successful_step_records(records))

    @classmethod
    def _is_closed_planner_owned_step(cls, step: AnalysisStep) -> bool:
        method = cls._normalise(str(step.method or "").split(" with ", 1)[0])
        products = set()
        for output in step.expected_outputs or []:
            kind, separator, name = str(output or "").partition(":")
            if separator:
                products.add((cls._normalise(kind), cls._normalise(name)))
        return method in cls._CLOSED_EFFECT_METHODS and bool(
            products & cls._CLOSED_EFFECT_PRODUCTS
        )

    @classmethod
    def _method_declares_penalty(
        cls,
        contract: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        values = [
            contract.get("fit_method"),
            contract.get("model_family"),
            contract.get("estimator"),
        ]
        if metadata:
            values.extend(
                (
                    metadata.get("fit_method"),
                    metadata.get("model_family"),
                    metadata.get("estimator"),
                )
            )
        blob = "_".join(cls._normalise(value) for value in values if value)
        return any(
            re.search(pattern, blob)
            for pattern in (
                r"(?:^|_)firth(?:_|$)",
                r"(?:^|_)ridge(?:_|$)",
                r"(?:^|_)lasso(?:_|$)",
                r"(?:^|_)elastic_?net(?:_|$)",
                r"(?:^|_)regulari[sz]ed(?:_|$)",
                r"(?:^|_)penali[sz]ed(?:_|$)",
            )
        )

    @classmethod
    def _planned_model_requirement_issues(
        cls,
        *,
        step: AnalysisStep,
        contracts: Sequence[Mapping[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        requirements = {
            item.requirement_id: item.model_dump(mode="python")
            for item in (getattr(step, "model_requirements", []) or [])
        }
        if not requirements:
            return [], {}

        issues: List[Dict[str, Any]] = []
        contracts_by_requirement: Dict[str, List[Mapping[str, Any]]] = {}
        for contract in contracts:
            requirement_id = str(contract.get("requirement_id") or "").strip()
            model_id = str(contract.get("model_id") or "")
            if not requirement_id:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "model_requirement_id_required",
                    }
                )
                continue
            if requirement_id not in requirements:
                issues.append(
                    {
                        "model_id": model_id,
                        "requirement_id": requirement_id,
                        "issue": "unplanned_model_requirement_id",
                    }
                )
                continue
            contracts_by_requirement.setdefault(requirement_id, []).append(contract)

        for requirement_id, requirement in requirements.items():
            matched = contracts_by_requirement.get(requirement_id, [])
            if not matched:
                if cls._planned_requirement_is_required(requirement):
                    issues.append(
                        {
                            "requirement_id": requirement_id,
                            "issue": "required_model_missing",
                            "expected": requirement,
                        }
                    )
                continue
            if len(matched) != 1:
                issues.append(
                    {
                        "requirement_id": requirement_id,
                        "issue": "duplicate_model_requirement_contract",
                        "reported": len(matched),
                    }
                )
                continue

            contract = matched[0]
            reported = reported_model_requirement_fields(contract)
            mismatches: Dict[str, Dict[str, Any]] = {}
            for field in (
                "outcome",
                "outcome_type",
                "method_family",
                "analysis_role",
                "analysis_set",
            ):
                reported_value = reported[field]
                required_value = requirement[field]
                if field == "method_family":
                    reported_value = canonical_association_method(reported_value)
                    required_value = canonical_association_method(required_value)
                if cls._normalise(reported_value) != cls._normalise(required_value):
                    mismatches[field] = {
                        "expected": requirement[field],
                        "reported": reported[field],
                    }
            if not cls._names_match(
                requirement["exposure_source"], reported["exposure_source"]
            ):
                mismatches["exposure_source"] = {
                    "expected": requirement["exposure_source"],
                    "reported": reported["exposure_source"],
                }
            field = "dependence"
            if requirement.get(field) != reported.get(field):
                mismatches[field] = {
                    "expected": requirement.get(field),
                    "reported": reported.get(field),
                }
            if mismatches:
                issues.append(
                    {
                        "model_id": contract.get("model_id"),
                        "requirement_id": requirement_id,
                        "issue": "model_requirement_field_mismatch",
                        "mismatches": mismatches,
                    }
                )
        return issues, requirements

    @classmethod
    def _planned_requirement_is_required(
        cls,
        requirement: Mapping[str, Any],
    ) -> bool:
        return bool(requirement.get("required_for_step_success")) or cls._normalise(
            requirement.get("analysis_role")
        ) in {"primary", "secondary"}

    @staticmethod
    def _fit_failure_reason(metadata: Mapping[str, Any]) -> str:
        return str(metadata.get("fit_failure_reason") or "").strip()

    @classmethod
    def _finite_nonfitted_result_fields(cls, rows: pd.DataFrame) -> List[str]:
        fields: List[str] = []
        for column in cls._NONFITTED_RESULT_FIELDS:
            if column not in rows.columns:
                continue
            if any(cls._finite_number(value) is not None for value in rows[column]):
                fields.append(column)
        return fields

    @classmethod
    def _finite_nonfitted_summary_result_fields(
        cls,
        step_summary: Mapping[str, Any],
        *,
        model_id: str,
    ) -> List[str]:
        """Find finite inferential results attached to one non-fitted model."""

        fields: Set[str] = set()

        def visit(value: Any, inherited_model_id: str = "") -> None:
            if isinstance(value, Mapping):
                active_model_id = str(
                    value.get("model_id") or inherited_model_id
                ).strip()
                if active_model_id == model_id:
                    for field in cls._NONFITTED_RESULT_FIELDS:
                        if (
                            field in value
                            and cls._finite_number(value.get(field)) is not None
                        ):
                            fields.add(field)
                for child in value.values():
                    visit(child, active_model_id)
            elif isinstance(value, list):
                for child in value:
                    visit(child, inherited_model_id)

        visit(step_summary)
        return sorted(fields)

    @classmethod
    def _names_match(cls, left: Any, right: Any) -> bool:
        left_text = str(left or "").strip().lower()
        right_text = str(right or "").strip().lower()
        a = re.sub(r"[^a-z0-9]", "", left_text)
        b = re.sub(r"[^a-z0-9]", "", right_text)
        if not a or not b:
            return False
        if a == b:
            return True
        left_tokens = [token for token in re.split(r"[^a-z0-9]+", left_text) if token]
        right_tokens = [token for token in re.split(r"[^a-z0-9]+", right_text) if token]

        def is_operational_alias(base: List[str], candidate: List[str]) -> bool:
            return bool(
                base
                and len(candidate) > len(base)
                and candidate[: len(base)] == base
                and all(
                    token in cls._OPERATIONAL_NAME_SUFFIXES
                    for token in candidate[len(base) :]
                )
            )

        return is_operational_alias(left_tokens, right_tokens) or is_operational_alias(
            right_tokens, left_tokens
        )

    @classmethod
    def _activates(
        cls,
        step: AnalysisStep,
        context: ResearchContext,
        step_summary: Mapping[str, Any],
    ) -> bool:
        has_planned_requirements = bool(getattr(step, "model_requirements", []) or [])
        if (
            not (context.primary_exposure or "").strip()
            and not has_planned_requirements
        ):
            return False
        method = cls._normalise(str(step.method or "").lower().split(" with ", 1)[0])
        raw_outputs = [
            str(output or "").strip().lower()
            for output in (step.expected_outputs or [])
        ]
        outputs = set()
        for output in raw_outputs:
            output_kind, separator, output_name = output.partition(":")
            if not separator:
                continue
            outputs.add((cls._normalise(output_kind), cls._normalise(output_name)))
        figure_only_outputs = bool(raw_outputs) and all(
            output.startswith("figure:") for output in raw_outputs
        )
        if method in cls._FIGURE_ONLY_METHODS or figure_only_outputs:
            return False
        canonical_method = canonical_association_method(method)
        supported_direct_method = canonical_method in (
            cls._BINARY_MODEL_FAMILIES | cls._CONTINUOUS_MODEL_FAMILIES
        )
        if has_planned_requirements:
            # AnalysisStep validation normally guarantees this scope. Keep the
            # runtime predicate defensive because model_copy(update=...) can
            # construct an unvalidated object in internal/test code.
            return method in cls._CLOSED_EFFECT_METHODS and bool(
                outputs & cls._CLOSED_EFFECT_PRODUCTS
            )
        # Once a step emits the machine contract key, even an empty or malformed
        # value must be audited rather than escaping through a prose router, but
        # only for the adjusted-association families this validator implements.
        # Survival, prediction, mixed-effects, and clustering outputs belong to
        # their own family-specific validators.
        if "model_contracts" in step_summary:
            return method in cls._CLOSED_EFFECT_METHODS or supported_direct_method
        return method in cls._CLOSED_EFFECT_METHODS and bool(
            outputs & cls._CLOSED_EFFECT_PRODUCTS
        )

    @staticmethod
    def _as_nonnegative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @staticmethod
    def _as_bool(value: Any) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    @classmethod
    def _latest_planned_adjustment(
        cls, completed_step_records: Sequence[Dict[str, Any]]
    ) -> tuple[List[str], List[str]]:
        for record in reversed(
            cls._authoritative_completed_records(completed_step_records)
        ):
            summary = record.get("step_summary")
            if not isinstance(summary, Mapping):
                continue
            planned = summary.get("planned_adjustment_context")
            if not isinstance(planned, Mapping):
                continue
            candidates = planned.get("candidate_covariates")
            excluded = planned.get("not_adjusted_for")
            if isinstance(candidates, list):
                return (
                    [str(value) for value in candidates if str(value).strip()],
                    [
                        str(value)
                        for value in (excluded if isinstance(excluded, list) else [])
                        if str(value).strip()
                    ],
                )
        return [], []

    @classmethod
    def _locked_primary_expression(
        cls,
        *,
        primary_exposure: str,
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> Optional[str]:
        locks: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    if cls._normalise(key) == "representation_locked" and isinstance(
                        child, str
                    ):
                        locks.append(child)
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for record in cls._authoritative_completed_records(completed_step_records):
            summary = record.get("step_summary")
            if isinstance(summary, Mapping):
                visit(summary)
        for lock in reversed(locks):
            match = re.search(
                r"(?:np\.)?log1p\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
                lock,
                flags=re.IGNORECASE,
            )
            if match and cls._names_match(primary_exposure, match.group(1)):
                return f"log1p({match.group(1)})"
        return None

    @classmethod
    def _operational_primary_sources(
        cls,
        *,
        declared_primary: str,
        completed_step_records: Sequence[Dict[str, Any]],
        step_summary: Mapping[str, Any],
    ) -> List[str]:
        """Resolve structured context-exposure -> operational-column aliases."""

        sources: List[str] = []

        def primary_matches(value: Any) -> bool:
            if isinstance(value, Mapping):
                value = next(
                    (
                        value.get(key)
                        for key in (
                            "authoritative_context_exposure",
                            "context_exposure",
                            "name",
                        )
                        if value.get(key) is not None
                    ),
                    None,
                )
            return cls._names_match(declared_primary, value)

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                primary = value.get("primary_exposure")
                if primary_matches(primary):
                    for key in (
                        "primary_exposure_source",
                        "operational_column",
                        "exposure_source",
                    ):
                        candidate = value.get(key)
                        if candidate is not None and str(candidate).strip():
                            sources.append(str(candidate).strip())
                    if isinstance(primary, Mapping):
                        candidate = primary.get("operational_column")
                        if candidate is not None and str(candidate).strip():
                            sources.append(str(candidate).strip())
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for record in cls._authoritative_completed_records(completed_step_records):
            summary = record.get("step_summary")
            if isinstance(summary, Mapping):
                visit(summary)
        return list(dict.fromkeys(sources))

    @classmethod
    def _expression_key(cls, value: Any) -> str:
        text = str(value or "").lower().replace("np.", "")
        return re.sub(r"\s+", "", text)

    @classmethod
    def _coefficient_rows(cls, out_dir: Path) -> Optional[pd.DataFrame]:
        frames: List[tuple[Path, pd.DataFrame]] = []
        required = {"model_id", "term", "term_role", "source_variable"}
        for path in sorted(Path(out_dir).glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if not required.issubset(frame.columns):
                continue
            if not {"ci_low", "ci_high"}.issubset(frame.columns):
                continue
            if not {"estimate", "odds_ratio", "or"}.intersection(frame.columns):
                continue
            # Figure/source-data bundles can be wide unions containing these
            # columns for only a subset of rows.  Ignore their non-model rows
            # instead of converting missing term roles into the literal role
            # ``nan`` and falsely rejecting an otherwise valid coefficient
            # table.
            frame = frame.loc[
                frame["model_id"].notna()
                & frame["term"].notna()
                & frame["term_role"].notna()
            ].copy()
            if frame.empty:
                continue
            frames.append((path, frame))
        if not frames:
            return None
        # Result tables can share model_id/term/effect columns while carrying
        # marginal risks or contrasts rather than fitted coefficients. Prefer
        # the term-level coefficient schema so missing standard errors created
        # only by a heterogeneous concat do not become false fit failures.
        coefficient_frames = [
            frame
            for path, frame in frames
            if "estimate_type" not in frame.columns
            or "coefficient" in cls._normalise(path.stem)
        ]
        selected = coefficient_frames or [frame for _, frame in frames]
        return pd.concat(selected, ignore_index=True)

    @classmethod
    def _current_adjustment_context(
        cls, step_summary: Mapping[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Read current-step adjustment declarations without case vocabulary."""

        candidates: List[str] = []
        excluded: List[str] = []

        def collect(raw: Any, target: List[str]) -> None:
            if not isinstance(raw, list):
                return
            for item in raw:
                value: Any = item
                if isinstance(item, Mapping):
                    value = next(
                        (
                            item.get(key)
                            for key in ("variable", "name", "source_variable")
                            if item.get(key) is not None
                        ),
                        None,
                    )
                text = str(value or "").strip()
                if text and text.lower() not in {"none", "nan", "null"}:
                    target.append(text)

        collect(step_summary.get("adjustment_covariates"), candidates)
        collect(step_summary.get("excluded_covariates"), excluded)
        planned = step_summary.get("planned_adjustment_context")
        if isinstance(planned, Mapping):
            collect(planned.get("candidate_covariates"), candidates)
            collect(planned.get("not_adjusted_for"), excluded)
        return list(dict.fromkeys(candidates)), list(dict.fromkeys(excluded))

    @classmethod
    def _actual_adjustment_sources_by_model(
        cls, coefficient_rows: pd.DataFrame
    ) -> Dict[str, List[str]]:
        sources: Dict[str, List[str]] = {}
        adjustment_rows = coefficient_rows[
            coefficient_rows["_term_role"].eq("adjustment")
        ]
        for model_id, rows in adjustment_rows.groupby("_model_id", sort=False):
            values = [
                str(value).strip()
                for value in rows["source_variable"]
                if pd.notna(value)
                and str(value).strip()
                and str(value).strip().lower() not in {"none", "nan", "null"}
            ]
            sources[str(model_id)] = list(dict.fromkeys(values))
        return sources

    @classmethod
    def _model_metadata_by_id(
        cls, step_summary: Mapping[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect structured model metadata repeated across summary sections."""

        metadata: Dict[str, Dict[str, Any]] = {}
        fields = {
            "outcome",
            "outcome_type",
            "family",
            "model_family",
            "fit_method",
            "interval_method",
            "intervals_approximate",
            "convergence_method",
            "optimizer_success",
            "max_abs_kkt",
            "convergence_tolerance",
            "fit_failure_reason",
            "categorical_covariates",
            "categorical_predictors",
            "categorical_sources",
            "categorical_variables",
        }

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                model_id = str(value.get("model_id") or "").strip()
                if model_id:
                    bucket = metadata.setdefault(model_id, {})
                    for field in fields:
                        if field in value and value.get(field) is not None:
                            bucket[field] = value.get(field)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(step_summary)
        return metadata

    @classmethod
    def _model_metadata(
        cls,
        contract: Mapping[str, Any],
        metadata_by_id: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        model_id = str(contract.get("model_id") or "")
        combined = dict(metadata_by_id.get(model_id, {}))
        combined.update(contract)
        if "model_family" not in combined and combined.get("family") is not None:
            combined["model_family"] = combined.get("family")
        cls._apply_nested_ridge_convergence_alias(
            contract=contract,
            metadata=combined,
        )
        return combined

    @classmethod
    def _apply_nested_ridge_convergence_alias(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """Map model-bound sklearn ridge diagnostics to the controlled fields."""

        if (
            "convergence_method" in metadata
            or "optimizer_success" in metadata
            or cls._as_bool(metadata.get("penalized")) is not True
        ):
            return
        fit_method = cls._normalise(metadata.get("fit_method"))
        penalty = cls._normalise(metadata.get("penalty"))
        if not (re.search(r"(?:^|_)ridge(?:_|$)", fit_method) or penalty == "ridge"):
            return
        diagnostics = contract.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            return
        model_id = str(contract.get("model_id") or "").strip()
        diagnostics_model_id = str(diagnostics.get("model_id") or "").strip()
        if diagnostics_model_id and diagnostics_model_id != model_id:
            return
        iterations = cls._as_nonnegative_int(diagnostics.get("ridge_iterations"))
        if (
            cls._as_bool(diagnostics.get("ridge_converged")) is not True
            or iterations is None
            or iterations < 1
        ):
            return
        metadata["convergence_method"] = "optimizer_success"
        metadata["optimizer_success"] = True

    @classmethod
    def _declared_outcome_type(
        cls,
        metadata: Mapping[str, Any],
        *,
        frame: Optional[pd.DataFrame] = None,
        outcome: str = "",
    ) -> str:
        explicit = cls._normalise(metadata.get("outcome_type"))
        if explicit in cls._OUTCOME_TYPES:
            return explicit
        family = cls._normalise(
            metadata.get("model_family") or metadata.get("fit_method")
        )
        if family in cls._BINARY_MODEL_FAMILIES:
            return "binary"
        if family in cls._CONTINUOUS_MODEL_FAMILIES:
            return "continuous"
        if frame is not None and outcome in frame.columns:
            values = pd.to_numeric(frame[outcome], errors="coerce").dropna()
            if not values.empty and set(values.unique()).issubset({0, 1}):
                return "binary"
            if not values.empty:
                return "continuous"
        # Backward compatibility for older single-binary-outcome contracts.
        return "binary"

    @staticmethod
    def _finite_number(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _fitted_term_interval_issues(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Mapping[str, Any],
        rows: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        if cls._normalise(contract.get("fit_status")) != "fitted" or rows.empty:
            return []
        model_id = str(contract.get("model_id") or "")
        penalized = cls._as_bool(contract.get("penalized")) is True
        interval_method = cls._normalise(metadata.get("interval_method"))
        point_only = penalized and interval_method == "unavailable"
        issues: List[Dict[str, Any]] = []
        row_interval_methods = {
            normalised
            for raw in rows.get(
                "interval_method",
                pd.Series(index=rows.index, dtype=object),
            )
            if pd.notna(raw)
            and (normalised := cls._normalise(raw))
            not in {"nan", "none", "not_applicable_reference", "null"}
        }
        if (
            interval_method
            and interval_method not in {"nan", "none", "null"}
            and row_interval_methods
            and row_interval_methods != {interval_method}
        ):
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "interval_method_contract_mismatch",
                    "contract_interval_method": metadata.get("interval_method"),
                    "term_interval_methods": sorted(row_interval_methods),
                }
            )
        effect_columns = [
            column
            for column in ("estimate", "odds_ratio", "or")
            if column in rows.columns
        ]
        has_standard_error = "standard_error" in rows.columns
        for _, row in rows.iterrows():
            term = str(row.get("term") or "")
            row_interval_method = cls._normalise(row.get("interval_method"))
            if (
                "reference" in cls._normalise(term)
                or row_interval_method == "not_applicable_reference"
            ):
                continue
            estimate = next(
                (
                    number
                    for column in effect_columns
                    if (number := cls._finite_number(row.get(column))) is not None
                ),
                None,
            )
            low = cls._finite_number(row.get("ci_low"))
            high = cls._finite_number(row.get("ci_high"))
            standard_error = (
                cls._finite_number(row.get("standard_error"))
                if has_standard_error
                else 0.0
            )
            has_any_interval = (
                low is not None
                or high is not None
                or (has_standard_error and standard_error is not None)
            )
            if point_only:
                reasons: List[str] = []
                if estimate is None:
                    reasons.append("nonfinite_estimate")
                if has_any_interval:
                    reasons.append("point_only_contract_contains_interval")
                if reasons:
                    issues.append(
                        {
                            "model_id": model_id,
                            "term": term,
                            "term_role": row.get("term_role"),
                            "issue": "fitted_term_missing_or_invalid_interval",
                            "reasons": reasons,
                        }
                    )
                continue
            reasons: List[str] = []
            if estimate is None:
                reasons.append("nonfinite_estimate")
            if low is None or high is None:
                reasons.append("missing_or_nonfinite_ci")
            elif low > high:
                reasons.append("reversed_ci")
            if has_standard_error and (standard_error is None or standard_error < 0):
                reasons.append("missing_nonfinite_or_negative_standard_error")
            if reasons:
                issues.append(
                    {
                        "model_id": model_id,
                        "term": term,
                        "term_role": row.get("term_role"),
                        "issue": "fitted_term_missing_or_invalid_interval",
                        "reasons": reasons,
                    }
                )
        return issues

    @classmethod
    def _effect_scale_issues(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Mapping[str, Any],
        rows: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        if rows.empty:
            return []
        outcome_type = cls._declared_outcome_type(metadata)
        issues: List[Dict[str, Any]] = []
        if (
            outcome_type == "continuous"
            and cls._normalise(contract.get("fit_status")) == "fitted"
        ):
            if "effect_scale" not in rows.columns:
                return [
                    {
                        "model_id": contract.get("model_id"),
                        "issue": "continuous_fitted_term_requires_effect_scale",
                        "terms": [str(value) for value in rows["term"].tolist()],
                    }
                ]
            missing_scale_terms = [
                str(row.get("term") or "")
                for _, row in rows.iterrows()
                if not str(row.get("effect_scale") or "").strip()
                or str(row.get("effect_scale") or "").strip().lower()
                in {"nan", "none", "null"}
            ]
            if missing_scale_terms:
                issues.append(
                    {
                        "model_id": contract.get("model_id"),
                        "issue": "continuous_fitted_term_requires_effect_scale",
                        "terms": missing_scale_terms,
                    }
                )
        if "effect_scale" not in rows.columns:
            return issues
        scales = {
            cls._normalise(value)
            for value in rows["effect_scale"]
            if pd.notna(value) and str(value).strip()
        }
        incompatible = (
            scales & cls._CONTINUOUS_INCOMPATIBLE_SCALES
            if outcome_type == "continuous"
            else scales & cls._BINARY_INCOMPATIBLE_SCALES
        )
        if incompatible:
            issues.append(
                {
                    "model_id": contract.get("model_id"),
                    "issue": "effect_scale_model_family_mismatch",
                    "outcome_type": outcome_type,
                    "model_family": metadata.get("model_family")
                    or metadata.get("fit_method"),
                    "reported_effect_scales": sorted(incompatible),
                }
            )
        return issues

    @classmethod
    def _penalized_provenance_issues(
        cls,
        *,
        contract: Mapping[str, Any],
        metadata: Mapping[str, Any],
        rows: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        if cls._as_bool(
            contract.get("penalized")
        ) is not True and not cls._method_declares_penalty(contract, metadata):
            return []
        model_id = str(contract.get("model_id") or "")
        interval_method = cls._normalise(metadata.get("interval_method"))
        finite_intervals = False
        if not rows.empty:
            finite_intervals = any(
                cls._finite_number(row.get("ci_low")) is not None
                and cls._finite_number(row.get("ci_high")) is not None
                for _, row in rows.iterrows()
            )
        issues: List[Dict[str, Any]] = []
        if (
            finite_intervals
            and interval_method not in cls._CONTROLLED_PENALIZED_INTERVAL_METHODS
        ):
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "penalized_intervals_require_controlled_provenance",
                    "reported_interval_method": metadata.get("interval_method"),
                    "allowed": sorted(cls._CONTROLLED_PENALIZED_INTERVAL_METHODS),
                }
            )
        if interval_method == "unavailable" and finite_intervals:
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "point_only_contract_contains_interval",
                }
            )
        if cls._as_bool(contract.get("converged")) is True:
            convergence_method = cls._normalise(metadata.get("convergence_method"))
            optimizer_success = cls._as_bool(metadata.get("optimizer_success"))
            verified = (
                convergence_method in cls._CONTROLLED_CONVERGENCE_METHODS
                and optimizer_success is True
            )
            if verified and convergence_method == "kkt_residual":
                residual = cls._finite_number(metadata.get("max_abs_kkt"))
                tolerance = cls._finite_number(metadata.get("convergence_tolerance"))
                if tolerance is None:
                    tolerance = 1e-6
                verified = residual is not None and residual <= tolerance
            if not verified:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "penalized_convergence_not_verified",
                        "reported_convergence_method": metadata.get(
                            "convergence_method"
                        ),
                        "optimizer_success": metadata.get("optimizer_success"),
                    }
                )
        if (
            interval_method == "easyicu_penalized_hessian_v1"
            and cls._as_bool(metadata.get("intervals_approximate")) is not True
        ):
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "penalized_hessian_interval_must_be_approximate",
                }
            )
        return issues

    @classmethod
    def _expected_denominator(
        cls,
        *,
        frame: pd.DataFrame,
        outcome: str,
        outcome_type: str,
        covariates: Sequence[str],
        contract: Mapping[str, Any],
        raw_exposure_source: Optional[str] = None,
    ) -> Optional[tuple[int, Optional[int]]]:
        if outcome not in frame.columns:
            return None
        outcome_values = pd.to_numeric(frame[outcome], errors="coerce")
        if outcome_type == "binary":
            mask = outcome_values.isin([0, 1])
        elif outcome_type == "continuous":
            mask = outcome_values.notna() & outcome_values.map(math.isfinite)
        else:
            return None
        policy = cls._normalise(contract.get("baseline_missing_policy"))
        if policy in {"drop_missing", "drop_missing_baseline", "complete_case"}:
            for covariate in covariates:
                if covariate not in frame.columns:
                    return None
                mask &= frame[covariate].notna()
        elif policy not in {"explicit_missing_category", "missing_category"}:
            return None

        analysis_set = cls._normalise(contract.get("analysis_set"))
        exposure_requires_observation = analysis_set == "complete_case" or (
            analysis_set == "source_aware"
            and policy in {"drop_missing", "drop_missing_baseline", "complete_case"}
        )
        if exposure_requires_observation:
            exposure = raw_exposure_source or str(contract.get("exposure_source") or "")
            if exposure not in frame.columns:
                return None
            values = frame[exposure]
            mask &= values.notna()
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                mask &= numeric.map(
                    lambda value: pd.notna(value) and abs(value) != float("inf")
                )
        elif analysis_set != "source_aware":
            return None
        event_n = (
            int(outcome_values.loc[mask].sum()) if outcome_type == "binary" else None
        )
        return int(mask.sum()), event_n

    @classmethod
    def _raw_exposure_source(
        cls,
        *,
        frame: Optional[pd.DataFrame],
        contract: Mapping[str, Any],
        exposure_rows: pd.DataFrame,
    ) -> Optional[str]:
        """Resolve a derived exposure to one host-verifiable physical column."""

        declared = str(contract.get("exposure_source") or "").strip()
        if frame is not None and declared in frame.columns:
            return declared
        if exposure_rows.empty:
            return None
        sources = {
            str(value).strip()
            for value in exposure_rows["_source"].tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        }
        if len(sources) != 1:
            return None
        source = next(iter(sources))
        if frame is not None and source not in frame.columns:
            return None
        expression = str(contract.get("exposure_expression") or "")
        if not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])",
            expression,
        ):
            return None
        return source

    @classmethod
    def _coefficient_source_authority_issues(
        cls,
        *,
        frame: pd.DataFrame,
        coefficient_rows: pd.DataFrame,
        contracts: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Require coefficient lineage to name a physical cohort column."""

        fitted_model_ids = {
            str(contract.get("model_id") or "")
            for contract in contracts
            if cls._normalise(contract.get("fit_status")) == "fitted"
        }
        authoritative_columns = [str(column) for column in frame.columns]
        issues: List[Dict[str, Any]] = []
        for _, row in coefficient_rows.iterrows():
            model_id = str(row.get("model_id") or "")
            term_role = cls._normalise(row.get("term_role"))
            if model_id not in fitted_model_ids or term_role == "intercept":
                continue
            raw_source = row.get("source_variable")
            source = "" if pd.isna(raw_source) else str(raw_source).strip()
            match_count = authoritative_columns.count(source) if source else 0
            if match_count == 1:
                continue
            reason = (
                "source_variable_missing_from_authoritative_cohort"
                if match_count == 0
                else "source_variable_not_unique_in_authoritative_cohort"
            )
            issues.append(
                {
                    "model_id": model_id,
                    "term": str(row.get("term") or ""),
                    "term_role": term_role,
                    "issue": "coefficient_source_variable_unresolvable",
                    "reason": reason,
                    "reported_source_variable": source or None,
                    "missing_raw_source_variables": (
                        [source or "<blank>"] if match_count == 0 else []
                    ),
                    "authoritative_match_count": match_count,
                    "required_semantics": cls._SOURCE_VARIABLE_SEMANTICS,
                }
            )
        return issues

    @classmethod
    def _denominator_resolution_detail(
        cls,
        *,
        frame: pd.DataFrame,
        outcome: str,
        outcome_type: str,
        covariates: Sequence[str],
        contract: Mapping[str, Any],
        raw_exposure_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        policy = cls._normalise(contract.get("baseline_missing_policy"))
        analysis_set = cls._normalise(contract.get("analysis_set"))
        required_sources = [outcome] if outcome else []
        if policy in {"drop_missing", "drop_missing_baseline", "complete_case"}:
            required_sources.extend(str(value) for value in covariates)
        if analysis_set == "complete_case":
            required_sources.append(
                raw_exposure_source or str(contract.get("exposure_source") or "")
            )
        required_sources = [value for value in dict.fromkeys(required_sources) if value]
        authoritative_columns = [str(column) for column in frame.columns]
        missing = [
            source for source in required_sources if source not in authoritative_columns
        ]
        ambiguous = [
            source
            for source in required_sources
            if authoritative_columns.count(source) > 1
        ]
        if missing:
            reason = "required_raw_source_missing_from_authoritative_cohort"
        elif ambiguous:
            reason = "required_raw_source_not_unique_in_authoritative_cohort"
        elif outcome_type not in cls._OUTCOME_TYPES:
            reason = "unsupported_outcome_type"
        elif policy not in cls._BASELINE_MISSING_POLICIES:
            reason = "unsupported_baseline_missing_policy"
        elif analysis_set not in cls._ANALYSIS_SETS:
            reason = "unsupported_analysis_set"
        else:
            reason = "denominator_inputs_not_machine_resolvable"
        return {
            "reason": reason,
            "missing_raw_source_variables": missing,
            "ambiguous_raw_source_variables": ambiguous,
            "required_semantics": cls._SOURCE_VARIABLE_SEMANTICS,
        }

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        context: ResearchContext,
        completed_step_records: Sequence[Dict[str, Any]],
        out_dir: Path,
        cohort_path: Path,
    ) -> List[ValidationFinding]:
        if not self._activates(step, context, step_summary):
            return []
        issues: List[Dict[str, Any]] = []
        if self._is_closed_planner_owned_step(step) and not (
            getattr(step, "model_requirements", []) or []
        ):
            issues.append(
                {
                    "issue": "planned_model_requirements_required",
                    "method": step.method,
                    "expected_outputs": list(step.expected_outputs or []),
                }
            )
        raw_contracts = step_summary.get("model_contracts")
        if not isinstance(raw_contracts, list) or not raw_contracts:
            issues.append(
                {
                    "issue": "missing_model_contracts",
                    "required_fields": list(self._REQUIRED_FIELDS),
                }
            )
            contracts: List[Mapping[str, Any]] = []
        else:
            contracts = [item for item in raw_contracts if isinstance(item, Mapping)]
            if len(contracts) != len(raw_contracts):
                issues.append({"issue": "model_contract_must_be_object"})

        model_ids: Set[str] = set()
        for index, contract in enumerate(contracts):
            missing = [
                field for field in self._REQUIRED_FIELDS if field not in contract
            ]
            if missing:
                issues.append(
                    {
                        "model_index": index,
                        "issue": "missing_model_contract_fields",
                        "fields": missing,
                    }
                )
                continue
            model_id = str(contract.get("model_id") or "").strip()
            if not model_id or model_id in model_ids:
                issues.append(
                    {
                        "model_index": index,
                        "model_id": model_id,
                        "issue": "blank_or_duplicate_model_id",
                    }
                )
            model_ids.add(model_id)
            controlled_fields = (
                ("exposure_role", self._EXPOSURE_ROLES),
                ("analysis_role", self._ANALYSIS_ROLES),
                ("analysis_set", self._ANALYSIS_SETS),
                (
                    "baseline_missing_policy",
                    self._BASELINE_MISSING_POLICIES,
                ),
                ("fit_status", self._FIT_STATUSES),
            )
            for field, allowed in controlled_fields:
                reported = self._normalise(contract.get(field))
                if reported not in allowed:
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": f"noncanonical_{field}",
                            "reported": contract.get(field),
                            "allowed": sorted(allowed),
                        }
                    )

        requirement_issues, requirements_by_id = self._planned_model_requirement_issues(
            step=step,
            contracts=contracts,
        )
        issues.extend(requirement_issues)

        primary_models = [
            contract
            for contract in contracts
            if self._normalise(contract.get("analysis_role")) == "primary"
        ]
        planner_primary_requirements = [
            requirement
            for requirement in (getattr(step, "model_requirements", []) or [])
            if self._normalise(requirement.analysis_role) == "primary"
        ]
        planner_authorized_secondary_only = (
            bool(getattr(step, "model_requirements", []) or [])
            and not planner_primary_requirements
        )
        if planner_authorized_secondary_only and primary_models:
            issues.append(
                {
                    "issue": "unplanned_primary_model",
                    "reported": len(primary_models),
                    "planner_model_roles": sorted(
                        {
                            self._normalise(requirement.analysis_role)
                            for requirement in step.model_requirements
                        }
                    ),
                }
            )
        elif not planner_authorized_secondary_only and len(primary_models) != 1:
            issues.append(
                {
                    "issue": "exactly_one_primary_model_required",
                    "reported": len(primary_models),
                }
            )
        declared_primary = str(context.primary_exposure or "")
        declared_primary_sources = list(
            step.primary_exposure_authority_sources(
                declared_primary,
                (
                    self._operational_primary_sources(
                        declared_primary=declared_primary,
                        completed_step_records=completed_step_records,
                        step_summary=step_summary,
                    )
                    if declared_primary.strip()
                    else ()
                ),
            )
        )
        if len(primary_models) == 1 and declared_primary_sources:
            primary = primary_models[0]
            if not any(
                self._names_match(source, primary.get("exposure_source"))
                for source in declared_primary_sources
            ):
                issues.append(
                    {
                        "model_id": primary.get("model_id"),
                        "issue": "primary_exposure_mismatch",
                        "expected": declared_primary_sources,
                        "reported": primary.get("exposure_source"),
                    }
                )
            if self._normalise(primary.get("exposure_role")) != "primary":
                issues.append(
                    {
                        "model_id": primary.get("model_id"),
                        "issue": "primary_model_exposure_role_must_be_primary",
                    }
                )
            locked_expression = self._locked_primary_expression(
                primary_exposure=declared_primary,
                completed_step_records=completed_step_records,
            )
            if locked_expression and self._expression_key(
                primary.get("exposure_expression")
            ) != self._expression_key(locked_expression):
                issues.append(
                    {
                        "model_id": primary.get("model_id"),
                        "issue": "locked_primary_expression_mismatch",
                        "expected": locked_expression,
                        "reported": primary.get("exposure_expression"),
                    }
                )

        if declared_primary_sources:
            for contract in contracts:
                is_declared_exposure = any(
                    self._names_match(source, contract.get("exposure_source"))
                    for source in declared_primary_sources
                )
                exposure_role = self._normalise(contract.get("exposure_role"))
                if is_declared_exposure and exposure_role != "primary":
                    issues.append(
                        {
                            "model_id": contract.get("model_id"),
                            "issue": "declared_primary_exposure_role_mismatch",
                            "reported": contract.get("exposure_role"),
                        }
                    )
                if not is_declared_exposure and exposure_role == "primary":
                    issues.append(
                        {
                            "model_id": contract.get("model_id"),
                            "issue": "alternate_exposure_cannot_be_primary",
                            "reported_source": contract.get("exposure_source"),
                        }
                    )

        candidate_covariates, not_adjusted = self._latest_planned_adjustment(
            completed_step_records
        )
        current_candidates, current_excluded = self._current_adjustment_context(
            step_summary
        )
        if not candidate_covariates:
            candidate_covariates = current_candidates
        if not not_adjusted:
            not_adjusted = current_excluded
        metadata_by_id = self._model_metadata_by_id(step_summary)
        nonfitted_result_fields: Dict[str, Set[str]] = {}
        for contract in contracts:
            model_id = str(contract.get("model_id") or "")
            fit_status = self._normalise(contract.get("fit_status"))
            if fit_status == "fitted":
                continue
            analysis_role = self._normalise(contract.get("analysis_role"))
            requirement_id = str(contract.get("requirement_id") or "").strip()
            requirement = requirements_by_id.get(requirement_id)
            required_for_success = (
                self._planned_requirement_is_required(requirement)
                if requirement is not None
                else analysis_role in {"primary", "secondary"}
            )
            if required_for_success:
                issues.append(
                    {
                        "model_id": model_id,
                        "requirement_id": requirement_id or None,
                        "analysis_role": analysis_role,
                        "fit_status": fit_status,
                        "issue": "required_model_not_fitted",
                    }
                )
            metadata = self._model_metadata(contract, metadata_by_id)
            finite_summary_fields = self._finite_nonfitted_summary_result_fields(
                step_summary,
                model_id=model_id,
            )
            if finite_summary_fields:
                nonfitted_result_fields.setdefault(model_id, set()).update(
                    finite_summary_fields
                )
            if not self._fit_failure_reason(metadata):
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "fit_failure_reason_required",
                        "fit_status": fit_status,
                    }
                )
        actual_covariates_by_model: Dict[str, List[str]] = {}
        coefficient_rows = self._coefficient_rows(Path(out_dir))
        if coefficient_rows is None:
            issues.append(
                {
                    "issue": "missing_term_level_coefficient_table",
                    # Named from what ``_coefficient_rows`` actually accepts.
                    # It used to say ``estimate_or_odds_ratio``, which is a
                    # description of the value and not a column this reader
                    # takes: a table written to that spelling was skipped, and
                    # the step was told again that its table was missing. A
                    # fail-closed message whose only implied fix does not
                    # satisfy the check is a trap, and the repair loop reading
                    # it has no other source of truth.
                    "required_columns": [
                        "model_id",
                        "term",
                        "term_role",
                        "source_variable",
                        "ci_low",
                        "ci_high",
                    ],
                    "required_effect_column_one_of": ["estimate", "odds_ratio", "or"],
                }
            )
        else:
            coefficient_rows = coefficient_rows.copy()
            coefficient_rows["_model_id"] = coefficient_rows["model_id"].astype(str)
            coefficient_rows["_term_role"] = coefficient_rows["term_role"].map(
                self._normalise
            )
            coefficient_rows["_source"] = coefficient_rows["source_variable"].astype(
                str
            )
            actual_covariates_by_model = self._actual_adjustment_sources_by_model(
                coefficient_rows
            )
            invalid_roles = sorted(
                set(coefficient_rows["_term_role"]) - self._TERM_ROLES
            )
            if invalid_roles:
                issues.append(
                    {"issue": "invalid_term_roles", "reported": invalid_roles}
                )
            exposure_sources = [
                str(contract.get("exposure_source") or "") for contract in contracts
            ]
            allowed_norm = {self._normalise(value) for value in candidate_covariates}
            excluded_norm = {self._normalise(value) for value in not_adjusted}
            for contract in contracts:
                model_id = str(contract.get("model_id") or "")
                source = str(contract.get("exposure_source") or "")
                metadata = self._model_metadata(contract, metadata_by_id)
                fit_status = self._normalise(contract.get("fit_status"))
                rows = coefficient_rows[coefficient_rows["_model_id"].eq(model_id)]
                if fit_status != "fitted":
                    finite_fields = self._finite_nonfitted_result_fields(rows)
                    if finite_fields:
                        nonfitted_result_fields.setdefault(model_id, set()).update(
                            finite_fields
                        )
                    continue
                if rows.empty:
                    issues.append(
                        {"model_id": model_id, "issue": "missing_coefficient_rows"}
                    )
                    continue
                exposure_rows = rows[rows["_term_role"].eq("exposure")]
                raw_exposure_source = self._raw_exposure_source(
                    frame=None,
                    contract=contract,
                    exposure_rows=exposure_rows,
                )
                exposure_source_matches = not exposure_rows.empty and all(
                    self._names_match(source, value)
                    or (
                        raw_exposure_source is not None
                        and self._names_match(raw_exposure_source, value)
                    )
                    for value in exposure_rows["_source"]
                )
                if not exposure_source_matches:
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": "exposure_terms_do_not_match_model_source",
                        }
                    )
                for _, row in rows[rows["_term_role"].eq("adjustment")].iterrows():
                    adjustment = str(row["_source"])
                    adjustment_norm = self._normalise(adjustment)
                    other_exposure = next(
                        (
                            other
                            for other in exposure_sources
                            if other != source and self._names_match(other, adjustment)
                        ),
                        None,
                    )
                    if other_exposure is not None:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "mutual_exposure_adjustment",
                                "offending_source": adjustment,
                            }
                        )
                    if excluded_norm and adjustment_norm in excluded_norm:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "forbidden_adjustment_source",
                                "offending_source": adjustment,
                            }
                        )
                    if allowed_norm and adjustment_norm not in allowed_norm:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "adjustment_outside_planned_allowlist",
                                "offending_source": adjustment,
                                "allowed": candidate_covariates,
                            }
                        )
                issues.extend(
                    self._fitted_term_interval_issues(
                        contract=contract,
                        metadata=metadata,
                        rows=rows,
                    )
                )
                issues.extend(
                    self._effect_scale_issues(
                        contract=contract,
                        metadata=metadata,
                        rows=rows,
                    )
                )
                issues.extend(
                    self._penalized_provenance_issues(
                        contract=contract,
                        metadata=metadata,
                        rows=rows,
                    )
                )

            # When the script also exposes model-level results, verify that its
            # term table carries the same exposure estimates and intervals.
            # This catches silent coefficient-index shifts (for example,
            # assigning beta[0], the intercept, to the first predictor) even
            # when all required columns and roles are present.
            model_results = step_summary.get("models")
            if isinstance(model_results, list):
                for model_result in model_results:
                    if not isinstance(model_result, Mapping):
                        continue
                    model_id = str(model_result.get("model_id") or "")
                    exposure_terms = model_result.get("exposure_terms")
                    if not model_id or not isinstance(exposure_terms, list):
                        continue
                    model_rows = coefficient_rows[
                        coefficient_rows["_model_id"].eq(model_id)
                        & coefficient_rows["_term_role"].eq("exposure")
                    ]
                    for expected_term in exposure_terms:
                        if not isinstance(expected_term, Mapping):
                            continue
                        term = str(expected_term.get("term") or "")
                        if not term:
                            continue
                        rows = model_rows[model_rows["term"].astype(str).eq(term)]
                        if rows.empty:
                            issues.append(
                                {
                                    "model_id": model_id,
                                    "term": term,
                                    "issue": "model_result_term_missing_from_coefficients",
                                }
                            )
                            continue
                        comparisons = (
                            ("estimate", "estimate"),
                            ("odds_ratio", "odds_ratio"),
                            ("ci_low", "ci_low"),
                            ("ci_high", "ci_high"),
                        )
                        for summary_field, table_field in comparisons:
                            expected_value = expected_term.get(summary_field)
                            if (
                                expected_value is None
                                or table_field not in rows.columns
                            ):
                                continue
                            expected_number = pd.to_numeric(
                                pd.Series([expected_value]), errors="coerce"
                            ).iloc[0]
                            reported_numbers = pd.to_numeric(
                                rows[table_field], errors="coerce"
                            ).dropna()
                            if pd.isna(expected_number) or reported_numbers.empty:
                                continue
                            if not all(
                                abs(float(value) - float(expected_number))
                                <= 1e-9 * max(1.0, abs(float(expected_number)))
                                for value in reported_numbers
                            ):
                                issues.append(
                                    {
                                        "model_id": model_id,
                                        "term": term,
                                        "issue": "coefficient_model_result_mismatch",
                                        "field": summary_field,
                                        "expected": float(expected_number),
                                        "reported": [
                                            float(value)
                                            for value in reported_numbers.unique()[:5]
                                        ],
                                    }
                                )

        for model_id, finite_fields in nonfitted_result_fields.items():
            contract = next(
                (
                    item
                    for item in contracts
                    if str(item.get("model_id") or "") == model_id
                ),
                {},
            )
            issues.append(
                {
                    "model_id": model_id,
                    "issue": "inconsistent_not_fitted_estimate",
                    "fit_status": self._normalise(contract.get("fit_status")),
                    "finite_fields": sorted(finite_fields),
                }
            )

        try:
            cohort = pd.read_parquet(cohort_path)
        except Exception:
            cohort = None
            issues.append({"issue": "cohort_unreadable_for_denominator_audit"})
        if cohort is not None and coefficient_rows is not None:
            issues.extend(
                self._coefficient_source_authority_issues(
                    frame=cohort,
                    coefficient_rows=coefficient_rows,
                    contracts=contracts,
                )
            )
        for contract in contracts:
            model_id = str(contract.get("model_id") or "")
            metadata = self._model_metadata(contract, metadata_by_id)
            outcome = str(metadata.get("outcome") or context.target_outcome or "")
            outcome_type = self._declared_outcome_type(
                metadata,
                frame=cohort,
                outcome=outcome,
            )
            model_covariates = (
                actual_covariates_by_model.get(model_id, candidate_covariates)
                if coefficient_rows is not None
                else candidate_covariates
            )
            fit_status = self._normalise(contract.get("fit_status"))
            converged = self._as_bool(contract.get("converged"))
            separation = self._as_bool(contract.get("separation_detected"))
            penalized = self._as_bool(contract.get("penalized"))
            method_declares_penalty = self._method_declares_penalty(contract, metadata)
            effective_penalized = penalized is True or method_declares_penalty
            reported_n = self._as_nonnegative_int(contract.get("n"))
            reported_events = (
                self._as_nonnegative_int(contract.get("event_n"))
                if outcome_type == "binary"
                else None
            )
            if converged is None or separation is None or penalized is None:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "fit_diagnostics_must_be_boolean",
                    }
                )
            if not str(contract.get("fit_method") or "").strip():
                issues.append({"model_id": model_id, "issue": "fit_method_required"})
            fit_method_text = str(contract.get("fit_method") or "").lower()
            if method_declares_penalty and penalized is False:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "penalized_method_must_report_penalized_true",
                        "fit_method": contract.get("fit_method"),
                    }
                )
            if effective_penalized and reported_n and "firth" not in fit_method_text:
                if "statsmodels" in fit_method_text and any(
                    token in fit_method_text
                    for token in ("regularized", "ridge", "elastic_net")
                ):
                    alpha_match = re.search(
                        r"alpha\s*=\s*([0-9.eE+-]+)", fit_method_text
                    )
                    max_alpha = 1.0 / float(reported_n)
                    if alpha_match is None:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "statsmodels_penalty_strength_not_reported",
                                "required_format": "fit_method includes alpha=<value>",
                                "maximum_weak_ridge_alpha": max_alpha,
                            }
                        )
                    else:
                        try:
                            alpha = float(alpha_match.group(1))
                        except ValueError:
                            alpha = float("nan")
                        if not pd.notna(alpha) or alpha <= 0 or alpha > max_alpha:
                            issues.append(
                                {
                                    "model_id": model_id,
                                    "issue": "statsmodels_penalty_too_strong_for_separation_fallback",
                                    "reported_alpha": alpha,
                                    "maximum_weak_ridge_alpha": max_alpha,
                                    "rationale": (
                                        "The per-observation statsmodels penalty "
                                        "must not dominate an inferential target "
                                        "effect when used only to stabilize separation."
                                    ),
                                }
                            )
                elif "sklearn" in fit_method_text:
                    c_match = re.search(
                        r"(?:^|[^a-z])c\s*=\s*([0-9.eE+-]+)",
                        fit_method_text,
                    )
                    if c_match is None:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "sklearn_penalty_strength_not_reported",
                                "required_format": "fit_method includes C=<value>",
                                "minimum_weak_ridge_c": 1.0,
                            }
                        )
                    else:
                        try:
                            c_value = float(c_match.group(1))
                        except ValueError:
                            c_value = float("nan")
                        if not pd.notna(c_value) or c_value < 1.0:
                            issues.append(
                                {
                                    "model_id": model_id,
                                    "issue": "sklearn_penalty_too_strong_for_separation_fallback",
                                    "reported_c": c_value,
                                    "minimum_weak_ridge_c": 1.0,
                                }
                            )
            if fit_status == "fitted" and converged is not True:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "fitted_model_must_converge",
                    }
                )
            if (
                separation is True
                and penalized is not True
                and fit_status
                not in {
                    "separation_no_estimate",
                    "not_fitted",
                }
            ):
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "separation_requires_penalized_fit_or_no_estimate",
                    }
                )
            if reported_n is None or (
                outcome_type == "binary" and reported_events is None
            ):
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "model_n_and_event_n_must_be_counts",
                    }
                )
            if outcome_type == "continuous" and contract.get("event_n") is not None:
                issues.append(
                    {
                        "model_id": model_id,
                        "issue": "continuous_outcome_event_n_must_be_null",
                        "reported_event_n": contract.get("event_n"),
                    }
                )
            if cohort is not None:
                model_rows = (
                    coefficient_rows[
                        coefficient_rows["_model_id"].eq(model_id)
                        & coefficient_rows["_term_role"].eq("exposure")
                    ]
                    if coefficient_rows is not None
                    else pd.DataFrame()
                )
                raw_exposure_source = self._raw_exposure_source(
                    frame=cohort,
                    contract=contract,
                    exposure_rows=model_rows,
                )
                expected = self._expected_denominator(
                    frame=cohort,
                    outcome=outcome,
                    outcome_type=outcome_type,
                    covariates=model_covariates,
                    contract=contract,
                    raw_exposure_source=raw_exposure_source,
                )
                if expected is None:
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": "denominator_contract_unresolvable",
                            **self._denominator_resolution_detail(
                                frame=cohort,
                                outcome=outcome,
                                outcome_type=outcome_type,
                                covariates=model_covariates,
                                contract=contract,
                                raw_exposure_source=raw_exposure_source,
                            ),
                        }
                    )
                elif (reported_n, reported_events) != expected:
                    issues.append(
                        {
                            "model_id": model_id,
                            "issue": "model_denominator_or_event_mismatch",
                            "expected_n": expected[0],
                            "expected_event_n": expected[1],
                            "reported_n": reported_n,
                            "reported_event_n": reported_events,
                        }
                    )
                if expected is not None and outcome_type == "binary":
                    zero_cells = self._categorical_zero_event_cells(
                        frame=cohort,
                        outcome=outcome,
                        covariates=model_covariates,
                        contract=metadata,
                        coefficient_rows=(
                            coefficient_rows[coefficient_rows["_model_id"].eq(model_id)]
                            if coefficient_rows is not None
                            else None
                        ),
                    )
                    if zero_cells and separation is not True:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "zero_cell_separation_not_reported",
                                "cells": zero_cells[:10],
                            }
                        )
                    if zero_cells and fit_status == "fitted" and penalized is not True:
                        issues.append(
                            {
                                "model_id": model_id,
                                "issue": "zero_cell_separation_requires_penalized_fit",
                                "cells": zero_cells[:10],
                            }
                        )

        if not issues:
            return []
        return [
            ValidationFinding(
                validator=self.name,
                severity="error",
                message=(
                    f"Complex primary-association step {step.step_id} violates "
                    f"the machine-verifiable multi-model contract ({len(issues)} "
                    "issue(s)). Emit step_summary.model_contracts with the fixed "
                    "fields, "
                    + (
                        "preserve the Planner-authorized secondary-only roster "
                        "without inventing a primary exposure, "
                        if planner_authorized_secondary_only
                        else "keep exactly one context-declared primary exposure, "
                    )
                    + "label alternate representations secondary/sensitivity, fit "
                    "separate models without mutual adjustment, use only the "
                    "planned baseline covariates, satisfy every planner-owned "
                    "required model requirement, report honest non-fit reasons, "
                    "report exact n/event_n, and "
                    "write a term-level coefficient table with model_id, term, "
                    "term_role, source_variable, effect and CI columns plus "
                    "convergence/separation/penalization diagnostics."
                ),
                detail={
                    "step_id": step.step_id,
                    "issues": issues[:50],
                    "required_model_contract_fields": list(self._REQUIRED_FIELDS),
                },
            )
        ]

    @classmethod
    def _categorical_zero_event_cells(
        cls,
        *,
        frame: pd.DataFrame,
        outcome: str,
        covariates: Sequence[str],
        contract: Mapping[str, Any],
        coefficient_rows: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """Return categorical baseline cells with zero events or zero survivors."""

        if outcome not in frame.columns:
            return []
        outcome_values = pd.to_numeric(frame[outcome], errors="coerce")
        mask = outcome_values.isin([0, 1])
        policy = cls._normalise(contract.get("baseline_missing_policy"))
        if policy == "drop_missing_baseline":
            for covariate in covariates:
                if covariate not in frame.columns:
                    return []
                mask &= frame[covariate].notna()
        elif policy != "explicit_missing_category":
            return []
        if cls._normalise(contract.get("analysis_set")) == "complete_case":
            exposure = str(contract.get("exposure_source") or "")
            if exposure not in frame.columns:
                return []
            values = frame[exposure]
            mask &= values.notna()
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                mask &= numeric.map(
                    lambda value: pd.notna(value) and abs(value) != float("inf")
                )
        declared_categorical: Set[str] = set()
        for key in (
            "categorical_covariates",
            "categorical_predictors",
            "categorical_sources",
            "categorical_variables",
        ):
            raw = contract.get(key)
            if isinstance(raw, list):
                declared_categorical.update(
                    cls._normalise(value) for value in raw if str(value or "").strip()
                )
        cells: List[Dict[str, Any]] = []
        for covariate in covariates:
            if covariate not in frame.columns:
                continue
            values = frame.loc[mask, covariate]
            modeled_as_categorical = False
            if coefficient_rows is not None and not coefficient_rows.empty:
                source_rows = coefficient_rows[
                    coefficient_rows["source_variable"]
                    .map(cls._normalise)
                    .eq(cls._normalise(covariate))
                ]
                source_rows = source_rows[~source_rows["_term_role"].eq("availability")]
                terms = {str(value) for value in source_rows.get("term", [])}
                modeled_as_categorical = len(terms) > 1 or any(
                    re.search(r"(?:\bC\s*\(|\[T\.|one[_ -]?hot|dummy)", term, re.I)
                    for term in terms
                )
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            low_cardinality_integer = False
            if len(numeric) >= 20:
                unique_n = int(numeric.nunique(dropna=True))
                low_cardinality_integer = bool(
                    1 < unique_n <= 12
                    and unique_n / len(numeric) <= 0.2
                    and ((numeric - numeric.round()).abs() <= 1e-9).all()
                )
            is_categorical = (
                isinstance(values.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(values)
                or pd.api.types.is_string_dtype(values)
                or pd.api.types.is_bool_dtype(values)
                or cls._normalise(covariate) in declared_categorical
                or modeled_as_categorical
                or low_cardinality_integer
            )
            if not is_categorical:
                continue
            if policy == "explicit_missing_category":
                values = values.astype("object").where(values.notna(), "<missing>")
            observed_outcome = outcome_values.loc[mask]
            grouped = (
                pd.DataFrame({"level": values.astype(str), "outcome": observed_outcome})
                .groupby("level", dropna=False)["outcome"]
                .agg(["count", "sum"])
            )
            for level, row in grouped.iterrows():
                count = int(row["count"])
                event_n = int(row["sum"])
                if count > 0 and event_n in {0, count}:
                    cells.append(
                        {
                            "variable": covariate,
                            "level": str(level),
                            "n": count,
                            "event_n": event_n,
                        }
                    )
        return cells


class CrossStepReconciliationTraceValidator:
    """Verify that a reconciliation table selects the correct parent rows.

    The supported absolute-risk table schema carries both prevalence and
    outcome-risk rows for each stratum.  Matching only on label can silently
    bind a prevalence row and then report ``n_denominator`` as the stratum N.
    This validator checks the detailed reconciliation CSV against the exact
    registered parent table selected by the step itself.
    """

    name = "cross_step_reconciliation_trace"

    @staticmethod
    def _normalise(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        try:
            number = float(value)
            if pd.notna(number) and number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @classmethod
    def _status_alias(cls, value: Any) -> str:
        normalised = cls._normalise(value)
        aliases = {
            "valid_observed": "observed",
            "observed_valid": "observed",
            "no_source": "no_source",
            "no_recorded_source_or_observation": "no_source",
        }
        return aliases.get(normalised, normalised)

    @classmethod
    def _ordinal_level_alias(cls, value: Any) -> str:
        """Normalise semantic labels such as ``level_0`` to parent value ``0``."""

        normalised = cls._normalise(value)
        match = re.fullmatch(r"(?:.*_)?level_?([0-9]+)", normalised)
        return match.group(1) if match else normalised

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if pd.notna(number) else None

    @classmethod
    def _registered_parent_path(cls, summary: Dict[str, Any]) -> Optional[Path]:
        candidates: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                upstream_step = value.get("upstream_step") or value.get(
                    "requested_step"
                )
                if isinstance(upstream_step, str):
                    for key, path in value.items():
                        key_text = re.sub(
                            r"[^a-z0-9]+", "_", str(key).strip().lower()
                        ).strip("_")
                        if (
                            "path" in key_text
                            and isinstance(path, str)
                            and path.strip()
                            and Path(path).suffix.lower() in {".csv", ".tsv"}
                        ):
                            candidates.append(path)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(summary)
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}:
                return path
        return None

    @classmethod
    def _reconciliation_table_path(
        cls, summary: Dict[str, Any], out_dir: Path
    ) -> Optional[Path]:
        for path in cls._reconciliation_candidate_paths(summary, out_dir):
            try:
                columns = set(pd.read_csv(path, nrows=1).columns)
            except Exception:
                continue
            variable_present = bool(
                columns.intersection({"source_variable", "variable", "exposure"})
            )
            requested_semantics_present = bool(
                columns.intersection(
                    {
                        "requested_role",
                        "requested_estimate_type",
                        "estimate_type",
                        "stratum_type",
                        "row_role",
                        "row_type",
                    }
                )
            )
            support_present = bool(
                columns.intersection(
                    {
                        "registered_output_status",
                        "row_supported",
                        "registered_supported",
                        "registered_row_supported",
                    }
                )
            )
            if (
                variable_present
                and requested_semantics_present
                and support_present
                and "registered_n" in columns
            ):
                return path
        return None

    @classmethod
    def _reconciliation_candidate_paths(
        cls, summary: Dict[str, Any], out_dir: Path
    ) -> List[Path]:
        names: List[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str) and "reconciliation" in value.lower():
                names.append(value)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(summary.get("output_files"))
        collect(summary.get("outputs"))
        candidates = [out_dir / name for name in names]
        candidates.extend(sorted(out_dir.glob("*reconciliation*.csv")))
        resolved: List[Path] = []
        seen: Set[Path] = set()
        for path in candidates:
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            resolved.append(path)
        return resolved

    @classmethod
    def _canonical_current_rows(cls, current: pd.DataFrame) -> pd.DataFrame:
        if {
            "source_variable",
            "requested_stratum",
            "requested_role",
            "registered_output_status",
        }.issubset(current.columns):
            return current
        rows: List[Dict[str, Any]] = []
        for _, source in current.iterrows():

            def first_value(*names: str) -> Any:
                for name in names:
                    if name not in current.columns:
                        continue
                    value = source.get(name)
                    if value is not None and not pd.isna(value):
                        return value
                return None

            variable = first_value("source_variable", "variable", "exposure")
            if variable is None:
                continue
            explicit_role = first_value("requested_role", "row_role", "row_type")
            explicit_role_normalised = cls._normalise(explicit_role)
            stratum_type = cls._normalise(
                first_value("stratum_type", "requested_group_type")
            )
            estimate_type = cls._normalise(
                first_value("requested_estimate_type", "estimate_type")
            )
            requested_level = first_value("requested_level", "level")
            requested_status = first_value("requested_source_status", "source_status")
            requested_stratum_raw = first_value(
                "requested_stratum", "stratum", "requested_group_value"
            )

            if (
                requested_level is not None
                or stratum_type in {"exposure_level", "level"}
                or explicit_role_normalised
                in {
                    "level",
                    "ordinal_level",
                    "required_valid_ordinal_level",
                }
            ):
                role = "required_valid_ordinal_level"
                requested_stratum = (
                    requested_level
                    if requested_level is not None
                    else requested_stratum_raw
                )
            elif (
                stratum_type == "source_status"
                or explicit_role_normalised
                in {"source_status", "required_source_status"}
                or (
                    explicit_role is None
                    and requested_status is not None
                    and estimate_type == "outcome_risk"
                )
            ):
                role = "required_source_status"
                requested_stratum = (
                    requested_status
                    if requested_status is not None
                    else requested_stratum_raw
                )
            elif (
                stratum_type == "distribution"
                or explicit_role_normalised
                in {"distribution", "required_continuous_representation"}
                or "distribution" in estimate_type
            ):
                role = "required_continuous_representation"
                requested_stratum = (
                    requested_status
                    if requested_status is not None
                    else requested_stratum_raw
                )
            elif explicit_role is not None:
                role = str(explicit_role)
                requested_stratum = requested_stratum_raw
            else:
                continue
            supported = first_value(
                "row_supported",
                "registered_supported",
                "registered_row_supported",
            )
            status_text = first_value("registered_output_status")
            if supported is None and status_text is not None:
                supported = cls._normalise(status_text) == "row_supported"
            if isinstance(supported, str):
                supported = supported.strip().lower() in {"true", "1", "yes"}
            selected_fields = str(
                first_value(
                    "registered_selected_fields",
                    "selected_parent_row_fields",
                    "selected_registered_fields",
                    "selected_parent_row_field_names",
                    "selected_parent_field_names",
                )
                or ""
            )
            selected_field_tokens = {
                cls._normalise(token)
                for token in re.split(r"[;,\s]+", selected_fields)
                if token.strip()
            }
            n_field = first_value("registered_n_field")
            if n_field is None and (
                "n" in selected_field_tokens
                or re.search(r"(?:^|[,\s])n=n(?:[,\s]|$)", selected_fields)
            ):
                n_field = "n"
            event_field = first_value("registered_event_n_field")
            if event_field is None and (
                "event_n" in selected_field_tokens
                or "event_n=event_n" in selected_fields
            ):
                event_field = "event_n"
            risk_field = first_value(
                "registered_risk_field", "registered_outcome_risk_field"
            )
            if risk_field is None and (
                "outcome_risk" in selected_field_tokens
                or "outcome_risk=outcome_risk" in selected_fields
            ):
                risk_field = "outcome_risk"
            distribution_fields: Dict[str, Any] = {}
            for statistic in ("median", "q25", "q75"):
                field = first_value(f"registered_{statistic}_field")
                if field is None and statistic in selected_field_tokens:
                    field = statistic
                distribution_fields[f"registered_{statistic}_field"] = field
            rows.append(
                {
                    "source_variable": variable,
                    "requested_stratum": requested_stratum,
                    "requested_role": role,
                    "registered_output_status": (
                        "row_supported" if supported is True else "row_not_supported"
                    ),
                    "registered_n": source.get("registered_n"),
                    "registered_event_n": first_value("registered_event_n"),
                    "registered_risk": first_value(
                        "registered_risk", "registered_outcome_risk"
                    ),
                    "registered_n_field": n_field,
                    "registered_event_n_field": event_field,
                    "registered_risk_field": risk_field,
                    "registered_median": first_value("registered_median"),
                    "registered_q25": first_value("registered_q25"),
                    "registered_q75": first_value("registered_q75"),
                    **distribution_fields,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def _parent_match(cls, parent: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
        required_parent = {
            "exposure",
            "group_type",
            "group_value",
            "estimate_type",
        }
        if not required_parent.issubset(parent.columns):
            return parent.iloc[0:0]
        source = cls._normalise(row.get("source_variable"))
        role = cls._normalise(row.get("requested_role"))
        target = row.get("requested_stratum")
        work = parent[parent["exposure"].map(cls._normalise).eq(source)].copy()

        if "ordinal_level" in role:
            return work[
                work["group_type"].map(cls._normalise).eq("exposure_level")
                & work["group_value"]
                .map(cls._ordinal_level_alias)
                .eq(cls._ordinal_level_alias(target))
                & work["estimate_type"].map(cls._normalise).eq("outcome_risk")
            ]
        if "source_status" in role:
            target_status = cls._status_alias(target)
            return work[
                work["group_type"].map(cls._normalise).eq("source_state")
                & work["group_value"].map(cls._status_alias).eq(target_status)
                & work["estimate_type"].map(cls._normalise).eq("outcome_risk")
            ]
        if "continuous_representation" in role:
            return work[
                work["group_type"].map(cls._normalise).eq("continuous_summary")
                & work["estimate_type"]
                .map(cls._normalise)
                .eq("continuous_distribution")
            ]
        return work.iloc[0:0]

    @classmethod
    def _trace_issues(
        cls, current: pd.DataFrame, parent: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        canonical = cls._canonical_current_rows(current)
        for _, row in canonical.iterrows():
            role = cls._normalise(row.get("requested_role"))
            if not any(
                token in role
                for token in (
                    "ordinal_level",
                    "source_status",
                    "continuous_representation",
                )
            ):
                continue
            matched = cls._parent_match(parent, row)
            label = f"{row.get('source_variable')}:{row.get('requested_stratum')}"
            reported_status = cls._normalise(row.get("registered_output_status"))
            if len(matched) == 0:
                if reported_status == "row_supported":
                    issues.append({"row": label, "issue": "false_parent_support"})
                continue
            if len(matched) != 1:
                issues.append(
                    {"row": label, "issue": f"ambiguous_parent_rows={len(matched)}"}
                )
                continue
            expected = matched.iloc[0]
            if reported_status != "row_supported":
                issues.append(
                    {"row": label, "issue": "supported_parent_row_reported_missing"}
                )
                continue

            expected_n = cls._as_float(expected.get("n"))
            reported_n = cls._as_float(row.get("registered_n"))
            if expected_n is not None and (
                reported_n is None or abs(reported_n - expected_n) > 1e-8
            ):
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_n_mismatch",
                        "expected": expected_n,
                        "reported": reported_n,
                    }
                )
            n_field = cls._normalise(row.get("registered_n_field"))
            if expected_n is not None and n_field != "n":
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_n_field_must_be_n",
                        "reported": n_field,
                    }
                )

            if "continuous_representation" in role:
                reported_risk = cls._as_float(row.get("registered_risk"))
                if reported_risk is not None:
                    issues.append(
                        {
                            "row": label,
                            "issue": "continuous_distribution_has_false_risk",
                            "reported": reported_risk,
                        }
                    )
                for statistic in ("median", "q25", "q75"):
                    expected_value = cls._as_float(expected.get(statistic))
                    if expected_value is None:
                        continue
                    reported_value = cls._as_float(row.get(f"registered_{statistic}"))
                    if (
                        reported_value is None
                        or abs(reported_value - expected_value) > 1e-10
                    ):
                        issues.append(
                            {
                                "row": label,
                                "issue": f"registered_{statistic}_mismatch",
                                "expected": expected_value,
                                "reported": reported_value,
                            }
                        )
                    reported_field = cls._normalise(
                        row.get(f"registered_{statistic}_field")
                    )
                    if reported_field != statistic:
                        issues.append(
                            {
                                "row": label,
                                "issue": (
                                    f"registered_{statistic}_field_must_be_{statistic}"
                                ),
                                "reported": reported_field,
                            }
                        )
                continue

            expected_event_n = cls._as_float(expected.get("event_n"))
            reported_event_n = cls._as_float(row.get("registered_event_n"))
            if expected_event_n is not None and (
                reported_event_n is None
                or abs(reported_event_n - expected_event_n) > 1e-8
            ):
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_event_n_mismatch",
                        "expected": expected_event_n,
                        "reported": reported_event_n,
                    }
                )
            event_n_field = cls._normalise(row.get("registered_event_n_field"))
            if expected_event_n is not None and event_n_field != "event_n":
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_event_n_field_must_be_event_n",
                        "reported": event_n_field,
                    }
                )

            expected_risk = cls._as_float(expected.get("outcome_risk"))
            reported_risk = cls._as_float(row.get("registered_risk"))
            if expected_risk is not None and (
                reported_risk is None or abs(reported_risk - expected_risk) > 1e-10
            ):
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_risk_mismatch",
                        "expected": expected_risk,
                        "reported": reported_risk,
                    }
                )
            risk_field = cls._normalise(row.get("registered_risk_field"))
            if expected_risk is not None and risk_field != "outcome_risk":
                issues.append(
                    {
                        "row": label,
                        "issue": "registered_risk_field_must_be_outcome_risk",
                        "reported": risk_field,
                    }
                )
        return issues

    @classmethod
    def _declared_range_flag_issues(
        cls,
        *,
        summary: Dict[str, Any],
        current: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Require detailed rows for range flags declared in the summary."""

        declared: Set[str] = set()

        def collect_declared(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if cls._normalise(key) == "range_flag_counts" and isinstance(
                        child, dict
                    ):
                        declared.update(cls._normalise(flag) for flag in child)
                    collect_declared(child)
            elif isinstance(value, list):
                for child in value:
                    collect_declared(child)

        collect_declared(summary)
        declared.discard("")
        if not declared:
            return []

        present: Set[str] = set()
        for column in ("local_range_flag", "range_flag", "requested_range_flag"):
            if column in current.columns:
                present.update(
                    cls._normalise(value) for value in current[column].dropna().tolist()
                )
        role_columns = (
            "requested_role",
            "row_role",
            "row_type",
            "stratum_type",
            "requested_group_type",
        )
        value_columns = (
            "requested_stratum",
            "stratum",
            "requested_group_value",
            "requested_row",
        )
        for _, row in current.iterrows():
            roles = {
                cls._normalise(row.get(column))
                for column in role_columns
                if column in current.columns
            }
            if not any("range_flag" in role for role in roles):
                continue
            present.update(
                cls._normalise(row.get(column))
                for column in value_columns
                if column in current.columns and pd.notna(row.get(column))
            )
        present.discard("")

        issues: List[Dict[str, Any]] = []
        for expected in sorted(declared):
            if any(
                expected == actual or expected in actual or actual in expected
                for actual in present
            ):
                continue
            issues.append(
                {
                    "row": expected,
                    "issue": "missing_declared_range_flag_row",
                }
            )
        return issues

    @classmethod
    def _percentage_issues(cls, out_dir: Path) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for path in sorted(out_dir.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if "row_type" not in frame or "percentage_of_valid_observed" not in frame:
                continue
            source_rows = frame[
                frame["row_type"].map(cls._normalise).eq("source_status")
            ]
            for index, row in source_rows.iterrows():
                fraction = cls._as_float(row.get("percentage_of_valid_observed"))
                pct = cls._as_float(row.get("percentage_of_valid_observed_pct"))
                # A source-status row is outside vs inside the observed subset;
                # its denominator is the locked cohort. Even a numerically
                # bounded value is semantically wrong under a column named
                # ``percentage_of_valid_observed``.
                if fraction is not None or pct is not None:
                    issues.append(
                        {
                            "file": path.name,
                            "row": int(index),
                            "status": row.get("status"),
                            "issue": "source_status_percentage_field_not_applicable",
                            "reported_fraction": fraction,
                            "reported_pct": pct,
                        }
                    )
        return issues

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        out_dir: Path,
    ) -> List[ValidationFinding]:
        parent_path = self._registered_parent_path(step_summary)
        current_path = self._reconciliation_table_path(step_summary, Path(out_dir))
        if parent_path is None:
            return []
        if current_path is None:
            candidates = self._reconciliation_candidate_paths(
                step_summary, Path(out_dir)
            )
            if not candidates:
                return []
            candidate_columns: Dict[str, Any] = {}
            for candidate in candidates:
                try:
                    candidate_columns[str(candidate)] = list(
                        pd.read_csv(candidate, nrows=1).columns
                    )
                except Exception as exc:
                    candidate_columns[str(candidate)] = {"read_error": str(exc)[:300]}
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Step {step.step_id} declared a reconciliation CSV, but "
                        "its schema does not expose the variable, requested-row "
                        "semantics, registered support flag, and registered_n "
                        "needed for parent-table trace verification."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "parent_table": str(parent_path),
                        "issue": "reconciliation_schema_unrecognised",
                        "candidate_columns": candidate_columns,
                    },
                )
            ]
        try:
            parent = pd.read_csv(parent_path)
            current = pd.read_csv(current_path)
        except Exception:
            return []
        issues = self._trace_issues(current, parent)
        issues.extend(
            self._declared_range_flag_issues(
                summary=step_summary,
                current=current,
            )
        )
        issues.extend(self._percentage_issues(Path(out_dir)))
        if not issues:
            return []
        return [
            ValidationFinding(
                validator=self.name,
                severity="error",
                message=(
                    f"Registered-output reconciliation in step {step.step_id} "
                    f"does not trace to the selected parent table ({len(issues)} "
                    "issue(s)). Match outcome-risk rows with "
                    "estimate_type=outcome_risk and use n/event_n/outcome_risk; "
                    "match the parent grouping dimension and value (for example "
                    "group_type/group_value), including level_0 versus 0 aliases; "
                    "map semantic request roles to the parent's actual grouping "
                    "labels (for example an ordinal *_level request may map to "
                    "exposure_level); record selected parent field names; preserve "
                    "detailed rows for every range flag declared in the summary; "
                    "match continuous summaries only with continuous_distribution; "
                    "normalise observed and valid-observed source aliases; keep "
                    "prevalence rows separate. Source-status percentages must use "
                    "the locked cohort denominator (or be NA), never the "
                    "valid-observed denominator."
                ),
                detail={
                    "step_id": step.step_id,
                    "parent_table": str(parent_path),
                    "reconciliation_table": str(current_path),
                    "issues": issues[:30],
                },
            )
        ]


class CrossStepSourceStatusValidator:
    """Keep source-status denominators stable across completed run steps.

    A data-quality step may lock the number of source-consistent observed
    values for a measured concept.  Later descriptive/model steps are free to
    transform the value, but they must not silently redefine which rows were
    observed when they report the same concept on the same cohort.

    The gate is deliberately evidence-driven: it only compares explicit
    ``source_summary`` blocks against an earlier machine-readable
    ``missingness.source_status_counts`` block, and only when the category
    totals match.  Missing or ambiguous evidence is therefore skipped rather
    than guessed.
    """

    name = "cross_step_source_status"

    @staticmethod
    def _normalise(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    @classmethod
    def _is_valid_observed_label(cls, value: Any) -> bool:
        tokens = set(cls._normalise(value).split("_"))
        return (
            "invalid" not in tokens
            and "valid" in tokens
            and bool(tokens.intersection({"observed", "measured", "value", "level"}))
        )

    @classmethod
    def _status_role(cls, value: Any) -> Optional[str]:
        tokens = set(cls._normalise(value).split("_"))
        if cls._is_valid_observed_label(value):
            return "valid_observed"
        if "no" in tokens and tokens.intersection(
            {"source", "recorded", "observation"}
        ):
            return "no_source"
        if (
            tokens.intersection({"measured", "observed"})
            and "missing" in tokens
            and tokens.intersection({"summary", "value"})
        ):
            return "measured_summary_missing"
        if tokens.intersection({"contradictory", "inconsistent", "invalid"}):
            return "contradictory_invalid"
        return None

    @staticmethod
    def _as_count(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    @classmethod
    def _flat_status_counts(
        cls, value: Any
    ) -> Optional[tuple[List[tuple[str, int]], Set[str]]]:
        """Parse one explicit four-role status mapping without guessing roles."""

        if not isinstance(value, dict):
            return None
        parsed = [
            (str(category), count)
            for category, raw_count in value.items()
            if (count := cls._as_count(raw_count)) is not None
        ]
        if not parsed or len(parsed) != len(value):
            return None
        present_roles = {
            role
            for category, _ in parsed
            if (role := cls._status_role(category)) is not None
        }
        return parsed, present_roles

    @classmethod
    def _declared_primary_source_summary(cls, summary: Dict[str, Any]) -> Optional[str]:
        """Return an explicitly declared primary summary column, if unique."""

        primary = summary.get("primary_exposure")
        if isinstance(primary, str) and primary.strip():
            return primary.strip()
        if isinstance(primary, dict):
            candidates = [
                str(primary.get(key) or "").strip()
                for key in ("column", "summary_variable", "source_summary")
            ]
            candidates = [value for value in candidates if value]
            if len(set(candidates)) == 1:
                return candidates[0]
        return None

    @classmethod
    def _prior_locks(
        cls, completed_step_records: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        locks: List[Dict[str, Any]] = []
        successful_statuses = {
            "ok",
            "complete",
            "completed",
            "repaired",
            "runner_repaired",
        }
        for record_index, record in enumerate(completed_step_records):
            status = str(record.get("status") or "").strip().lower()
            if status and status not in successful_statuses:
                continue
            summary = record.get("step_summary")
            if not isinstance(summary, dict):
                continue
            # A value-quality step may publish one exact top-level status map
            # bound to its explicit primary exposure.  This is the same closed
            # contract as the older nested ``missingness`` representation.
            source_summary = cls._declared_primary_source_summary(summary)
            flat = cls._flat_status_counts(summary.get("source_status_counts"))
            if source_summary and flat is not None:
                parsed, present_roles = flat
                required_roles = {
                    "valid_observed",
                    "no_source",
                    "measured_summary_missing",
                    "contradictory_invalid",
                }
                valid_counts = [
                    count
                    for category, count in parsed
                    if cls._is_valid_observed_label(category)
                ]
                if len(valid_counts) == 1 and required_roles <= present_roles:
                    role_counts = {
                        role: count
                        for category, count in parsed
                        if (role := cls._status_role(category)) is not None
                    }
                    locks.append(
                        {
                            "concept": cls._normalise(source_summary),
                            "source_summary": source_summary,
                            "scope": "step_summary.source_status_counts",
                            "total_n": sum(count for _, count in parsed),
                            "valid_observed_n": valid_counts[0],
                            "role_counts": role_counts,
                            "step_id": str(record.get("step_id") or "prior_step"),
                            "record_index": record_index,
                        }
                    )
            missingness = summary.get("missingness")
            if not isinstance(missingness, dict):
                continue
            source_counts = missingness.get("source_status_counts")
            if not isinstance(source_counts, dict):
                continue
            for scope, by_concept in source_counts.items():
                if not isinstance(by_concept, dict):
                    continue
                for concept, categories in by_concept.items():
                    if not isinstance(categories, dict):
                        continue
                    parsed = {
                        str(category): count
                        for category, raw_count in categories.items()
                        if (count := cls._as_count(raw_count)) is not None
                    }
                    valid_counts = [
                        count
                        for category, count in parsed.items()
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) != 1 or not parsed:
                        continue
                    locks.append(
                        {
                            "concept": cls._normalise(concept),
                            "source_summary": str(concept),
                            "scope": str(scope),
                            "total_n": sum(parsed.values()),
                            "valid_observed_n": valid_counts[0],
                            "step_id": str(record.get("step_id") or "prior_step"),
                            "record_index": record_index,
                        }
                    )
        return locks

    @classmethod
    def _current_status_blocks(cls, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        # Current descriptive steps may use ``source_status_schema`` for the
        # same four-category count contract.  Bind it only to an explicit
        # primary exposure; a free-standing map is intentionally ignored.
        source_summary = cls._declared_primary_source_summary(summary)
        flat = cls._flat_status_counts(summary.get("source_status_schema"))
        if source_summary and flat is not None:
            parsed, present_roles = flat
            valid_counts = [
                count
                for category, count in parsed
                if cls._is_valid_observed_label(category)
            ]
            required_roles = {
                "valid_observed",
                "no_source",
                "measured_summary_missing",
                "contradictory_invalid",
            }
            if len(valid_counts) == 1:
                role_counts = {
                    role: count
                    for category, count in parsed
                    if (role := cls._status_role(category)) is not None
                }
                blocks.append(
                    {
                        "concept": cls._normalise(source_summary),
                        "source_summary": source_summary,
                        "path": "source_status_schema",
                        "total_n": sum(count for _, count in parsed),
                        "valid_observed_n": valid_counts[0],
                        "role_counts": role_counts,
                        "missing_status_roles": sorted(required_roles - present_roles),
                    }
                )

        declarations: List[Dict[str, str]] = []

        def collect_declarations(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                summary_variable = value.get("summary_variable")
                if isinstance(summary_variable, str) and summary_variable.strip():
                    alias = cls._normalise(path[-1] if path else "")
                    alias = re.sub(r"_definition$", "", alias)
                    declarations.append(
                        {
                            "alias": alias,
                            "source_summary": summary_variable,
                            "base": re.sub(
                                r"_(?:first|max|min|mean|median)$",
                                "",
                                cls._normalise(summary_variable),
                            ),
                        }
                    )
                for key, child in value.items():
                    collect_declarations(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect_declarations(child, (*path, str(index)))

        collect_declarations(summary)

        def declared_source_for(path: tuple[str, ...]) -> Optional[str]:
            hint = cls._normalise(path[-1] if path else "")
            hint = re.sub(r"_(?:measurement_)?status(?:_counts)?$", "", hint)
            exact = [
                declaration
                for declaration in declarations
                if declaration["alias"] == hint
            ]
            if len(exact) == 1:
                return exact[0]["source_summary"]
            semantic = [
                declaration
                for declaration in declarations
                if declaration["base"] == hint
                or declaration["base"].startswith(f"{hint}_")
                or hint.startswith(f"{declaration['base']}_")
            ]
            if len(semantic) == 1:
                return semantic[0]["source_summary"]
            return None

        def visit(value: Any, path: tuple[str, ...] = ()) -> None:
            if isinstance(value, dict):
                direct_counts = value.get("source_status_counts")
                source_columns = value.get("source_columns")
                if isinstance(direct_counts, dict) and isinstance(source_columns, list):
                    source_summary = next(
                        (
                            str(column)
                            for column in source_columns
                            if isinstance(column, str) and column.strip()
                        ),
                        None,
                    )
                    parsed_direct = [
                        (str(category), count)
                        for category, raw_count in direct_counts.items()
                        if (count := cls._as_count(raw_count)) is not None
                    ]
                    valid_counts = [
                        count
                        for category, count in parsed_direct
                        if cls._is_valid_observed_label(category)
                    ]
                    if source_summary and len(valid_counts) == 1 and parsed_direct:
                        present_roles = {
                            role
                            for category, _ in parsed_direct
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join((*path, "source_status_counts")),
                                "total_n": sum(count for _, count in parsed_direct),
                                "valid_observed_n": valid_counts[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                # Some reconciliation summaries store one concept per mapping
                # with ``counts`` and ``valid_observed_n`` rather than an
                # explicit source_columns list.  The concept key is still a
                # machine-readable source summary name, so preserve the same
                # four-category completeness and denominator lock.
                concept_counts = value.get("counts")
                concept_valid = cls._as_count(value.get("valid_observed_n"))
                if (
                    isinstance(concept_counts, dict)
                    and concept_valid is not None
                    and path
                ):
                    parsed_concept = [
                        (str(category), count)
                        for category, raw_count in concept_counts.items()
                        if (count := cls._as_count(raw_count)) is not None
                    ]
                    valid_counts = [
                        count
                        for category, count in parsed_concept
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) == 1 and parsed_concept:
                        present_roles = {
                            role
                            for category, _ in parsed_concept
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        source_summary = str(path[-1])
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join((*path, "counts")),
                                "total_n": sum(count for _, count in parsed_concept),
                                "valid_observed_n": valid_counts[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                if path and any(
                    "source_status_count" in cls._normalise(segment) for segment in path
                ):
                    parsed_nested = [
                        (str(category), count)
                        for category, raw in value.items()
                        if isinstance(raw, dict)
                        and (count := cls._as_count(raw.get("count", raw.get("n"))))
                        is not None
                    ]
                    valid_nested = [
                        count
                        for category, count in parsed_nested
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_nested) == 1 and parsed_nested:
                        present_roles = {
                            role
                            for category, _ in parsed_nested
                            if (role := cls._status_role(category)) is not None
                        }
                        required_roles = {
                            "valid_observed",
                            "no_source",
                            "measured_summary_missing",
                            "contradictory_invalid",
                        }
                        source_summary = str(path[-1])
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join(path),
                                "total_n": sum(count for _, count in parsed_nested),
                                "valid_observed_n": valid_nested[0],
                                "missing_status_roles": sorted(
                                    required_roles - present_roles
                                ),
                            }
                        )
                source_summary = value.get("source_summary")
                rows = value.get("measurement_status_counts")
                if not isinstance(rows, list):
                    rows = value.get("counts")
                if isinstance(source_summary, str) and isinstance(rows, list):
                    parsed: List[tuple[str, int]] = []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        count = cls._as_count(row.get("count", row.get("n")))
                        category = row.get("category", row.get("status"))
                        if count is not None and category is not None:
                            parsed.append((str(category), count))
                    valid_counts = [
                        count
                        for category, count in parsed
                        if cls._is_valid_observed_label(category)
                    ]
                    if len(valid_counts) == 1 and parsed:
                        blocks.append(
                            {
                                "concept": cls._normalise(source_summary),
                                "source_summary": source_summary,
                                "path": ".".join(path) or "step_summary",
                                "total_n": sum(count for _, count in parsed),
                                "valid_observed_n": valid_counts[0],
                            }
                        )
                # Newer descriptive summaries may expose the same contract as
                # scalar counts under ``missingness_and_measurement_status``
                # instead of a list of category rows.  Bind the status block to
                # an explicit nearby ``summary_variable`` declaration; never
                # guess from a human label alone.
                scalar_valid = cls._as_count(value.get("observed_valid_summary_n"))
                scalar_total = cls._as_count(value.get("denominator_n"))
                if scalar_valid is not None and scalar_total is not None:
                    scalar_source = value.get("source_summary") or value.get(
                        "summary_variable"
                    )
                    if not isinstance(scalar_source, str) or not scalar_source.strip():
                        scalar_source = declared_source_for(path)
                    if isinstance(scalar_source, str) and scalar_source.strip():
                        blocks.append(
                            {
                                "concept": cls._normalise(scalar_source),
                                "source_summary": scalar_source,
                                "path": ".".join(path) or "step_summary",
                                "total_n": scalar_total,
                                "valid_observed_n": scalar_valid,
                            }
                        )
                for key, child in value.items():
                    visit(child, (*path, str(key)))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, (*path, str(index)))

        visit(summary)
        return blocks

    def audit(
        self,
        *,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        completed_step_records: Sequence[Dict[str, Any]],
    ) -> List[ValidationFinding]:
        locks = self._prior_locks(completed_step_records)
        if not locks:
            return []

        findings: List[ValidationFinding] = []
        compared: Set[tuple[str, int, int]] = set()
        for current in self._current_status_blocks(step_summary):
            candidates = [
                lock
                for lock in locks
                if lock["concept"] == current["concept"]
                and lock["total_n"] == current["total_n"]
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda lock: (
                    "analytic" not in self._normalise(lock["scope"]),
                    -int(lock["record_index"]),
                )
            )
            expected = candidates[0]
            comparison_key = (
                current["concept"],
                current["total_n"],
                current["valid_observed_n"],
            )
            if comparison_key in compared:
                continue
            compared.add(comparison_key)
            missing_status_roles = current.get("missing_status_roles") or []
            if missing_status_roles:
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Incomplete source-status schema for "
                            f"{current['source_summary']} in step {step.step_id}: "
                            f"missing categories {missing_status_roles}. Report "
                            "all four source-status categories explicitly, using "
                            "zero counts for supported zero-frequency strata "
                            "rather than omitting their machine-summary keys."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "summary_path": current["path"],
                            "source_summary": current["source_summary"],
                            "cohort_n": current["total_n"],
                            "missing_status_roles": missing_status_roles,
                            "expected_from_step": expected["step_id"],
                        },
                    )
                )
            expected_role_counts = expected.get("role_counts")
            current_role_counts = current.get("role_counts")
            if (
                isinstance(expected_role_counts, dict)
                and isinstance(current_role_counts, dict)
                and not missing_status_roles
                and current_role_counts != expected_role_counts
            ):
                findings.append(
                    ValidationFinding(
                        validator=self.name,
                        severity="error",
                        message=(
                            f"Source-status category drift for "
                            f"{current['source_summary']}: step {step.step_id} "
                            "reallocated rows among observed, no-source, "
                            "measured-summary-missing, or contradictory states "
                            f"relative to completed step {expected['step_id']}. "
                            "Preserve the earlier closed source-status mapping."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "summary_path": current["path"],
                            "source_summary": current["source_summary"],
                            "cohort_n": current["total_n"],
                            "reported_status_counts": current_role_counts,
                            "expected_status_counts": expected_role_counts,
                            "expected_from_step": expected["step_id"],
                            "expected_scope": expected["scope"],
                        },
                    )
                )
                continue
            if current["valid_observed_n"] == expected["valid_observed_n"]:
                continue
            findings.append(
                ValidationFinding(
                    validator=self.name,
                    severity="error",
                    message=(
                        f"Source-status denominator drift for "
                        f"{current['source_summary']}: step {step.step_id} reports "
                        f"{current['valid_observed_n']} valid observed rows of "
                        f"{current['total_n']}, but completed step "
                        f"{expected['step_id']} locked "
                        f"{expected['valid_observed_n']} for the same concept and "
                        "cohort. Preserve the earlier source-status, variable-type, "
                        "and retain/flag range semantics instead of redefining "
                        "validity in this step."
                    ),
                    detail={
                        "step_id": step.step_id,
                        "summary_path": current["path"],
                        "source_summary": current["source_summary"],
                        "cohort_n": current["total_n"],
                        "reported_valid_observed_n": current["valid_observed_n"],
                        "expected_valid_observed_n": expected["valid_observed_n"],
                        "expected_from_step": expected["step_id"],
                        "expected_scope": expected["scope"],
                    },
                )
            )
        return findings
