"""Extract typed evidence used by deterministic step-result validation."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from ..scalar_utils import _first_numeric_scalar_with_key_fragment, _first_present_scalar, _flatten_scalar_dict
from ..schema import AnalysisStep, ClusterSelectionManifest, ResearchContext, ValidationFinding

def _problematic_metric_keys(
    payload: Any,
    fragments: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return metric-like keys that were present but null/non-finite."""

    lowered_fragments = tuple(fragment.lower() for fragment in fragments if fragment)
    if not lowered_fragments:
        return []
    problems: List[Dict[str, Any]] = []

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
            return
        lowered_path = path.lower()
        if not any(fragment in lowered_path for fragment in lowered_fragments):
            return
        bad = False
        if value is None:
            bad = True
        elif isinstance(value, bool):
            bad = False
        elif isinstance(value, (int, float)):
            bad = not math.isfinite(float(value))
        elif isinstance(value, str):
            text = value.strip().lower()
            bad = (
                text in {"", "nan", "none", "null", "model not fitted"}
                or "not fitted" in text
            )
        if bad:
            problems.append({"key": path, "value": value})

    walk(payload)
    return problems


_PRIMARY_EFFECT_DIRECT_KEYS = (
    "estimate",
    "statistic:estimate",
    "primary_or",
    "statistic:primary_or",
    "odds_ratio",
    "statistic:odds_ratio",
    "adjusted_or",
    "statistic:adjusted_or",
    "adjusted_odds_ratio",
    "statistic:adjusted_odds_ratio",
    "primary_association_estimate",
    "statistic:primary_association_estimate",
    "association_estimate",
    "statistic:association_estimate",
    "or",
)

_PRIMARY_EFFECT_VALUE_KEYS = (
    "primary_or",
    "odds_ratio",
    "adjusted_odds_ratio",
    "adjusted_or",
    "or",
    "estimate",
    "value",
)

_PRIMARY_EFFECT_CI_LOW_KEYS = (
    "ci_low",
    "ci_lower",
    "lower_ci",
    "ci_lower_95",
    "confidence_interval_low",
)

_PRIMARY_EFFECT_CI_HIGH_KEYS = (
    "ci_high",
    "ci_upper",
    "upper_ci",
    "ci_upper_95",
    "confidence_interval_high",
)


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _first_finite_present_scalar(
    payload: Dict[str, Any], keys: Sequence[str]
) -> Optional[float]:
    value = _first_present_scalar(payload, keys)
    return _finite_float(value)


def _lookup_first_finite(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[float]:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key.lower() not in lowered:
            continue
        numeric = _finite_float(lowered[key.lower()])
        if numeric is not None:
            return numeric
    return None


def _primary_effect_name_matches(source_path: str) -> bool:
    lowered = source_path.lower()
    return bool(
        "primary" in lowered
        or "odds_ratio" in lowered
        or re.search(r"(?:^|[.:\-\[\]])or(?:$|[.:\-\[\]])", lowered)
    )


def _flattened_primary_effect_key_matches(source_path: str) -> bool:
    """True for flattened scalar paths that represent an effect value.

    Generated step summaries sometimes report the primary association as a
    dictionary of per-level effects, for example
    ``primary.adjusted_odds_ratio_sofa.sofa2_5.0``.  The contract only needs
    to know that a finite primary effect was recorded, while avoiding CI
    bounds and p-values from sibling paths such as
    ``primary.adjusted_odds_ratio_sofa_ci95.sofa2_5.0.low``.
    """

    lowered = source_path.lower()
    if any(
        marker in lowered
        for marker in (
            "ci95",
            "_ci",
            ".ci",
            "confidence",
            "p_value",
            "pvalue",
        )
    ):
        return False
    if re.search(
        r"(?:^|[._:\-\[\]])(?:low|high|lower|upper|p|se|stderr)(?:$|[._:\-\[\]])",
        lowered,
    ):
        return False
    return bool(
        "odds_ratio" in lowered
        or "primary_or" in lowered
        or "adjusted_or" in lowered
        or re.search(r"(?:^|[.:\-\[\]])or(?:$|[.:\-\[\]])", lowered)
        or lowered.endswith("_estimate")
        or lowered.endswith(".estimate")
    )


def _primary_effect_from_mapping(
    payload: Mapping[str, Any],
    *,
    require_ci: bool,
) -> Optional[float]:
    effect = _lookup_first_finite(payload, _PRIMARY_EFFECT_VALUE_KEYS)
    if effect is None:
        return None
    if not require_ci:
        return effect
    ci_low = _lookup_first_finite(payload, _PRIMARY_EFFECT_CI_LOW_KEYS)
    ci_high = _lookup_first_finite(payload, _PRIMARY_EFFECT_CI_HIGH_KEYS)
    if ci_low is None or ci_high is None:
        return None
    return effect


def _primary_effect_from_estimates_list(payload: Mapping[str, Any]) -> Optional[float]:
    estimates = payload.get("primary_estimates")
    if not isinstance(estimates, list):
        return None
    for idx, item in enumerate(estimates):
        if not isinstance(item, Mapping):
            continue
        effect = _primary_effect_from_mapping(
            item,
            require_ci=False,
        )
        if effect is not None:
            return effect
    return None


def _primary_effect_from_statistic_dicts(payload: Mapping[str, Any]) -> Optional[float]:
    for key, value in payload.items():
        source_path = str(key)
        if isinstance(value, Mapping):
            if source_path.lower().startswith(
                "statistic:"
            ) and _primary_effect_name_matches(source_path):
                effect = _primary_effect_from_mapping(
                    value,
                    require_ci=True,
                )
                if effect is not None:
                    return effect
            nested = _primary_effect_from_statistic_dicts(value)
            if nested is not None:
                return nested
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if not isinstance(item, Mapping):
                    continue
                nested = _primary_effect_from_statistic_dicts(
                    {f"{source_path}[{idx}]": item}
                )
                if nested is not None:
                    return nested
    return None


def _primary_effect_from_summary(step_summary: Dict[str, Any]) -> Optional[float]:
    effect = _first_finite_present_scalar(step_summary, _PRIMARY_EFFECT_DIRECT_KEYS)
    if effect is not None:
        return effect
    effect = _primary_effect_from_estimates_list(step_summary)
    if effect is not None:
        return effect
    effect = _primary_effect_from_statistic_dicts(step_summary)
    if effect is not None:
        return effect
    for key, value in _flatten_scalar_dict(step_summary).items():
        lowered = key.lower()
        if (
            lowered.endswith("_or")
            or lowered.endswith("_odds_ratio")
            or lowered.endswith("_estimate")
            or _flattened_primary_effect_key_matches(lowered)
        ):
            effect = _finite_float(value)
            if effect is not None:
                return effect
    # Canonical effects must come from structured numeric fields or tables.
    # Free prose is not an evidence contract and can contain ordinary language
    # such as "or 1.5-1.9 times baseline" that resembles an OR label.
    return None


_AUROC_SCALAR_KEYS = (
    "auroc",
    "statistic:auroc",
    "auroc_test",
    "statistic:auroc_test",
    "auc",
    "statistic:auc",
    "held_out_auroc",
    "statistic:held_out_auroc",
    "cv_auroc",
    "statistic:cv_auroc",
    "cv_auroc_mean",
    "statistic:cv_auroc_mean",
    "mean_auroc",
    "auroc_mean",
    "auroc_median",
)


def _prediction_auroc_from_completed_records(
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    *,
    current_step_id: str,
) -> Optional[Tuple[str, float]]:
    """Find an auditable AUROC in a *sibling* completed step's summary.

    This fallback is limited to the prediction requirement: a figure/rendering
    step (e.g. ``*_model_training_figure``)
    often does not re-register the metric under a key its own renderer
    recognises, but the discrimination estimate is genuinely produced and bound
    (``statistic:auroc``) by the upstream training step it renders. When that is
    so, the requirement is satisfied by the sibling step, not missing.
    """
    if not completed_step_records:
        return None
    for record in completed_step_records:
        if not isinstance(record, dict):
            continue
        source_step_id = str(record.get("step_id") or "")
        if not source_step_id or source_step_id == current_step_id:
            continue
        if record.get("status") != "ok":
            continue
        step_summary = record.get("step_summary")
        if not isinstance(step_summary, dict):
            continue
        value = _first_present_scalar(step_summary, _AUROC_SCALAR_KEYS)
        if value is None:
            value = _first_numeric_scalar_with_key_fragment(
                step_summary, ("auroc", "auc")
            )
        if value is not None:
            return source_step_id, value
    return None


_CALIBRATION_SCALAR_KEYS = (
    "brier_score",
    "statistic:brier_score",
    "brier_test",
    "statistic:brier_test",
    "cv_brier_mean",
    "statistic:cv_brier_mean",
    "brier_mean",
    "held_out_brier",
    "statistic:held_out_brier",
    "brier_median",
    "calibration_slope",
    "statistic:calibration_slope",
    "calibration_slope_median",
    "calibration_intercept",
    "statistic:calibration_intercept",
    "calibration_intercept_median",
)


def _prediction_calibration_from_completed_records(
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    *,
    current_step_id: str,
) -> Optional[Tuple[str, float]]:
    """Calibration/Brier analogue of :func:`_prediction_auroc_from_completed_records`."""
    if not completed_step_records:
        return None
    for record in completed_step_records:
        if not isinstance(record, dict):
            continue
        source_step_id = str(record.get("step_id") or "")
        if not source_step_id or source_step_id == current_step_id:
            continue
        if record.get("status") != "ok":
            continue
        step_summary = record.get("step_summary")
        if not isinstance(step_summary, dict):
            continue
        value = _first_present_scalar(step_summary, _CALIBRATION_SCALAR_KEYS)
        if value is None:
            value = _first_numeric_scalar_with_key_fragment(
                step_summary, ("brier", "calibration_slope", "calibration_intercept")
            )
        if value is not None:
            return source_step_id, value
    return None


_CLUSTER_COUNT_SCALAR_KEYS = (
    "n_clusters",
    "statistic:n_clusters",
    "cluster_count",
    "statistic:cluster_count",
)


def _cluster_count_from_summary(payload: Mapping[str, Any]) -> Optional[float]:
    value = _first_present_scalar(dict(payload), _CLUSTER_COUNT_SCALAR_KEYS)
    numeric = _finite_float(value)
    if numeric is None or numeric < 1 or not numeric.is_integer():
        return None
    return numeric


def _cluster_selection_evidence_key(
    payload: Mapping[str, Any],
    *,
    cluster_count: Optional[float] = None,
) -> Tuple[Optional[str], bool]:
    """Return a typed selection manifest or substantive stability mapping.

    Bare strings and paths are declarations, not evidence, and intentionally do
    not satisfy the scientific step contract.  The boolean return value marks
    an explicitly declared but invalid/contradictory selection manifest; callers
    must fail closed instead of laundering it through stability or sibling
    fallback evidence.
    """

    def valid_stability(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        if cluster_count is None:
            return False
        selected_n_clusters = value.get("selected_n_clusters")
        try:
            selected_valid = (
                int(selected_n_clusters) >= 1
                and float(selected_n_clusters).is_integer()
                and int(selected_n_clusters) == int(cluster_count)
            )
        except (TypeError, ValueError):
            selected_valid = False
        if not selected_valid:
            return False
        n_resamples = value.get("n_resamples")
        try:
            n_valid = int(n_resamples) >= 2 and float(n_resamples).is_integer()
        except (TypeError, ValueError):
            n_valid = False
        if not n_valid:
            resamples = value.get("resamples")
            n_valid = isinstance(resamples, list) and len(resamples) >= 2
        metric_keys = {
            "adjusted_rand_index",
            "mean_adjusted_rand_index",
            "stability_score",
            "mean_jaccard",
        }
        has_metric = any(
            str(key).strip().lower().rsplit(".", 1)[-1] in metric_keys
            and _finite_float(child) is not None
            for key, child in _flatten_scalar_dict(value).items()
        )
        return n_valid and has_metric

    def valid_selection(value: Any) -> bool:
        try:
            manifest = ClusterSelectionManifest.model_validate(value)
        except ValidationError:
            return False
        if cluster_count is not None and manifest.selected_n_clusters != int(
            cluster_count
        ):
            return False
        selected_value = next(
            item.criterion_value
            for item in manifest.candidates
            if item.n_clusters == manifest.selected_n_clusters
        )
        candidate_values = [item.criterion_value for item in manifest.candidates]
        if manifest.selection_rule == "minimum":
            return math.isclose(
                selected_value,
                min(candidate_values),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        if manifest.selection_rule == "maximum":
            return math.isclose(
                selected_value,
                max(candidate_values),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        return True

    explicit_manifests: List[Tuple[str, Any]] = []
    stability_alternatives: List[Tuple[str, Any]] = []

    def collect(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).strip().lower()
                child_path = f"{path}.{key_text}" if path else key_text
                if key_text in {"cluster_selection", "cluster_selection_manifest"}:
                    explicit_manifests.append((child_path, child))
                if key_text in {"cluster_stability", "stability_evidence"}:
                    stability_alternatives.append((child_path, child))
                collect(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, f"{path}[{index}]")

    collect(payload)
    if explicit_manifests:
        # An explicit manifest is authoritative.  If any declared copy is
        # malformed or contradicts cluster_count, neither a stability mapping in
        # the same summary nor a completed sibling may rescue it.
        if any(not valid_selection(value) for _, value in explicit_manifests):
            return None, True
        return explicit_manifests[0][0], False
    for path, value in stability_alternatives:
        if valid_stability(value):
            return path, False
    return None, False


def _clustering_evidence_from_completed_records(
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    *,
    current_step_id: str,
) -> Tuple[Optional[Tuple[str, float, str]], bool]:
    """Find count plus native selection evidence in a completed sibling step.

    Clustering analog of :func:`_prediction_auroc_from_completed_records`. A
    feature-freeze / figure / rendering step often does not fit clusters
    itself. The genuine clustering step may satisfy the contract with its
    agent-selected native criterion (for example BIC, ICL, gap statistic,
    silhouette, or resampling stability); no one metric family is privileged.
    """
    if not completed_step_records:
        return None, False
    for record in completed_step_records:
        if not isinstance(record, dict):
            continue
        source_step_id = str(record.get("step_id") or "")
        if not source_step_id or source_step_id == current_step_id:
            continue
        if record.get("status") != "ok":
            continue
        step_summary = record.get("step_summary")
        if not isinstance(step_summary, dict):
            continue
        count = _cluster_count_from_summary(step_summary)
        selection_key, explicit_manifest_invalid = _cluster_selection_evidence_key(
            step_summary,
            cluster_count=count,
        )
        if explicit_manifest_invalid:
            return None, True
        if count is not None and selection_key is not None:
            return (source_step_id, count, selection_key), False
    return None, False


_EXPOSURE_PREDICTOR_KEYS = (
    "primary_predictor",
    "predictor",
    "exposure",
    "primary_association_term",
    "primary_term",
)
_ASSOCIATION_EFFECT_KEYS = (
    "primary_or",
    "odds_ratio",
    "adjusted_or",
    "primary_odds_ratio",
    "primary_odds_ratio_per_point",
    "primary_association_estimate",
    "hazard_ratio",
)


def _summary_primary_predictor(step_summary: Mapping[str, Any]) -> str:
    for key in _EXPOSURE_PREDICTOR_KEYS:
        value = step_summary.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _summary_has_association_effect(step_summary: Mapping[str, Any]) -> bool:
    return any(step_summary.get(key) is not None for key in _ASSOCIATION_EFFECT_KEYS)


def _exposure_names_match(required: str, actual: str) -> bool:
    """Lenient text match that preserves numeric exposure identity."""
    required_numbers = set(re.findall(r"\d+", required.lower()))
    actual_numbers = set(re.findall(r"\d+", actual.lower()))
    if (not required_numbers and actual_numbers) or not required_numbers.issubset(
        actual_numbers
    ):
        return False
    r = re.sub(r"[^a-z]", "", required.lower())
    a = re.sub(r"[^a-z]", "", actual.lower())
    if not r or not a:
        return True
    if r in a or a in r:
        return True
    n = 4
    if len(r) < n or len(a) < n:
        return False
    grams = {r[i : i + n] for i in range(len(r) - n + 1)}
    return any(a[i : i + n] in grams for i in range(len(a) - n + 1))


def _primary_exposure_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: ResearchContext,
) -> List[ValidationFinding]:
    """Flag when the primary association model estimated the wrong exposure.

    When the question names a required primary exposure
    (``context.primary_exposure``) and this step fitted an association model
    whose declared predictor is clearly a different variable, emit an error
    finding. The exposure named in the question is *what the analysis must
    estimate* — modelling a different one answers a different question — so
    this is an objective contract error, not an analytical-preference call
    (it never dictates the model form, covariates, or estimator). Routed
    through the existing contract-repair loop so the agent re-fits in-run
    without restarting the whole pipeline.
    """
    if not isinstance(step_summary, Mapping):
        return []
    required = (getattr(context, "primary_exposure", None) or "").strip()
    if not required:
        return []
    actual = _summary_primary_predictor(step_summary)
    # Only judge the primary model step: it declares a predictor *and* an
    # association-effect estimate. An effect with no declared predictor is the
    # separate "omitted predictor" case handled by the deterministic repairs.
    if not actual or not _summary_has_association_effect(step_summary):
        return []
    if _exposure_names_match(required, actual):
        return []
    return [
        ValidationFinding(
            validator="exposure_contract_auditor",
            severity="error",
            message=(
                f"The question's primary exposure is `{required}`, but this "
                f"primary model estimated `{actual}`. Re-fit the association "
                f"with `{required}` as the primary exposure using the "
                "prespecified representation and measurement semantics. Label "
                "other exposure representations secondary/corroborative and fit "
                "them separately unless the study contract explicitly justifies "
                "including one in the other's adjustment set."
            ),
            detail={
                "kind": "exposure_contract",
                "step_id": step.step_id,
                "required_exposure": required,
                "actual_predictor": actual,
            },
        )
    ]


def _iter_nested_mappings(payload: Any) -> List[Mapping[str, Any]]:
    mappings: List[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        mappings.append(payload)
        for value in payload.values():
            mappings.extend(_iter_nested_mappings(value))
    elif isinstance(payload, list):
        for value in payload:
            mappings.extend(_iter_nested_mappings(value))
    return mappings


def _numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _mapping_number_for_any_key(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[float]:
    lowered_keys = {key.lower() for key in keys}
    for key, value in mapping.items():
        if str(key).lower() not in lowered_keys:
            continue
        number = _numeric_value(value)
        if number is not None:
            return number
    return None


def _summary_has_single_level_exposure(step_summary: Mapping[str, Any]) -> bool:
    text = json.dumps(step_summary, ensure_ascii=False, default=str).lower()
    if any(
        marker in text
        for marker in (
            "no variation",
            "zero variance",
            "single level",
            "single-level",
            "constant exposure",
            "exposure has no variation",
            "singular design",
        )
    ):
        return True
    for mapping in _iter_nested_mappings(step_summary):
        exposed = _mapping_number_for_any_key(
            mapping,
            (
                "exposed_n",
                "exposure_positive_n",
                "positive_n",
                "event_positive_n",
            ),
        )
        unexposed = _mapping_number_for_any_key(
            mapping,
            (
                "unexposed_n",
                "exposure_negative_n",
                "negative_n",
                "event_negative_n",
            ),
        )
        if exposed is None or unexposed is None:
            continue
        total = exposed + unexposed
        if total >= 10 and (
            (exposed == 0 and unexposed > 0) or (unexposed == 0 and exposed > 0)
        ):
            return True
    return False


def _summary_has_measurement_filter_signal(step_summary: Mapping[str, Any]) -> bool:
    for mapping in _iter_nested_mappings(step_summary):
        for key, value in mapping.items():
            lowered = str(key).lower()
            if not any(
                marker in lowered
                for marker in (
                    "unmeasured",
                    "unascertain",
                    "no_source",
                    "no-source",
                    "no_positive_evidence",
                )
            ):
                continue
            number = _numeric_value(value)
            if number is not None and number > 0:
                return True
    return False


def _payload_mentions_required_exposure(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    required: str,
) -> bool:
    actual = _summary_primary_predictor(step_summary)
    if actual and _exposure_names_match(required, actual):
        return True
    blob = " ".join(
        [
            getattr(step, "step_id", None) or "",
            getattr(step, "intent", None) or "",
            json.dumps(step_summary, ensure_ascii=False, default=str),
        ]
    ).lower()
    required_norm = re.sub(r"[^a-z0-9]", "", required.lower())
    blob_norm = re.sub(r"[^a-z0-9]", "", blob)
    return bool(required_norm and required_norm in blob_norm)


def _primary_exposure_measurement_filter_findings(
    *,
    step: AnalysisStep,
    step_summary: Mapping[str, Any],
    context: ResearchContext,
) -> List[ValidationFinding]:
    """Catch sparse-event exposures collapsed by filtering on measurement flags.

    Some generated scripts treat ``<concept>_measured == 0`` or ``<concept>_n == 0``
    as exposure-missing and remove those stays before modelling. For sparse binary
    event indicators, that often drops event-negative/untriggered patients and
    turns the primary exposure into a constant. This is an objective contract
    failure only when the summary both (1) refers to the question's primary
    exposure and (2) shows a single-level exposure plus a positive unmeasured /
    unascertainable exclusion signal.
    """
    if not isinstance(step_summary, Mapping):
        return []
    required = (getattr(context, "primary_exposure", None) or "").strip()
    if not required:
        return []
    if not _payload_mentions_required_exposure(
        step=step, step_summary=step_summary, required=required
    ):
        return []
    if not _summary_has_single_level_exposure(step_summary):
        return []
    if not _summary_has_measurement_filter_signal(step_summary):
        return []
    return [
        ValidationFinding(
            validator="exposure_contract_auditor",
            severity="error",
            message=(
                f"The primary exposure `{required}` collapsed to a single level "
                "after the step filtered records as unmeasured/unascertainable. "
                "Do not exclude event-negative or untriggered rows solely because "
                "`<concept>_measured == 0` or `<concept>_n == 0`. Rebuild the "
                "binary exposure denominator from the source value columns so "
                "event-absent records remain 0/False unless concept metadata "
                "explicitly says the state is unassessed and uninterpretable. "
                "If the exposure is truly single-level after that audit, report "
                "the model as infeasible with source-data evidence."
            ),
            detail={
                "kind": "exposure_measurement_filter",
                "step_id": step.step_id,
                "required_exposure": required,
            },
        )
    ]
