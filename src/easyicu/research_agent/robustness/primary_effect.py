"""Primary-effect candidate selection helpers extracted from pipeline.py.

When the cross-database aggregator needs one canonical primary effect per
database, it reads the current successful step ledger.  Primary ownership is
granted only by the host-persisted ``planned_analysis_role`` bound both to the
outer step record and its immutable ``analysis_request.step`` snapshot.  A
summary (including a model-contract payload) can describe an effect but cannot
declare itself primary.

Moved out on 2026-05-27 as part of the pipeline.py size-reduction effort.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..research_context.typed import parse_research_context
from ..plan_utils import (
    _finite_float,
    _infer_primary_predictor_from_context,
    _predictor_tokens,
    _primary_effect_from_summary,
)
from ..authority.runtime_artifacts import (
    current_successful_step_records,
    load_run_artifact_authority,
)
from ..authority.planned_role import unique_verified_primary_record
from ..scalar_utils import _first_present_scalar
from ..schema import PipelineResult

__all__ = [
    "_extract_primary_effect_row",
    "_extract_primary_effect_payload_from_records",
    "_extract_primary_effect_payload_from_summary",
    "_infer_primary_predictor_from_run_dir",
    "_primary_effect_payload_is_complete",
    "_primary_effect_candidate_score",
]


def _extract_primary_effect_row(
    *, database: str, result: PipelineResult
) -> Dict[str, Any]:
    run_dir = Path(result.workdir)
    preferred_predictor = _infer_primary_predictor_from_run_dir(run_dir)
    payload: Dict[str, Any] = {
        "database": database,
        "run_id": result.run_id,
        "manifest_path": result.manifest_path,
        "predictor": None,
        "primary_or": None,
        "primary_ci_low": None,
        "primary_ci_high": None,
        "effect_measure": None,
        "status": "missing_primary_association",
    }
    authority = load_run_artifact_authority(run_dir)
    if authority is None:
        # Pre-v1 cleanup deliberately removes filesystem discovery.  An orphaned
        # ``step_summary.json`` has no current-attempt or host role authority.
        return payload
    raw_records = authority.get("per_step_records")
    records = raw_records if isinstance(raw_records, list) else []
    active_payload = _extract_primary_effect_payload_from_records(
        records,
        preferred_predictor=preferred_predictor,
    )
    if active_payload is not None:
        payload.update(active_payload)
    return payload


def _extract_primary_effect_payload_from_records(
    per_step_records: Any,
    *,
    preferred_predictor: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the uniquely host-authorised primary-effect payload.

    Every planned current-successful record must carry the same valid role in
    two host-owned locations: the outer record and the frozen
    ``analysis_request.step`` snapshot.  Host-native auxiliary records are the
    only exception and require an explicit ``step_authority_kind``.  Missing,
    invalid, or disagreeing role bindings fail closed.  Exactly one current
    record must be ``primary``; summary fields and model-contract self-claims
    never participate in this selection.
    """

    successful_records = current_successful_step_records(per_step_records or [])
    primary_record = unique_verified_primary_record(successful_records)
    if primary_record is None:
        return None
    summary = primary_record.get("step_summary")
    if not isinstance(summary, dict):
        return None
    candidate_payload = _extract_primary_effect_payload_from_summary(
        summary,
        path=_record_summary_path(primary_record),
        preferred_predictor=preferred_predictor,
    )
    candidate_payload.pop("_score", None)
    candidate_payload["step_id"] = str(primary_record.get("step_id") or "")
    candidate_payload["evidence_id"] = str(
        primary_record.get("step_summary_evidence_id") or ""
    )
    return candidate_payload


def _effect_measure_from_scale(scale: Any) -> Optional[str]:
    """Map a declared effect-scale string to a compact measure label.

    Propensity-weighted / causal steps write a scale-neutral point estimate under
    ``adjusted_effect`` and declare the scale in a SEPARATE field
    (``primary_effect_scale``/``effect_scale``), e.g. ``odds_ratio``. Returns None
    when the scale is absent or unrecognised, so the caller never binds an
    unlabelled estimate as if it were an OR.
    """
    text = str(scale or "").strip().lower()
    if not text:
        return None
    if "odds" in text or text == "or":
        return "OR"
    if "hazard" in text or text == "hr":
        return "HR"
    if (
        "risk_ratio" in text
        or "relative_risk" in text
        or "rate_ratio" in text
        or text in ("rr", "risk ratio")
    ):
        return "RR"
    if "risk_difference" in text or text in ("rd", "risk diff", "risk difference"):
        return "RD"
    if "mean_difference" in text or text in ("md", "mean diff", "mean difference"):
        return "MD"
    return None


def _primary_effect_payload_is_complete(payload: Any) -> bool:
    """Return whether a payload can support a converged primary panel row.

    A point estimate without its declared scale, uncertainty, or analytic
    denominator is not a primary-result contract.  This deliberately mirrors
    the fail-closed robustness preflight instead of allowing a prose-derived or
    partially populated value to become the run headline.
    """

    if not isinstance(payload, dict):
        return False
    point = _finite_float(payload.get("primary_or"))
    ci_low = _finite_float(payload.get("primary_ci_low"))
    ci_high = _finite_float(payload.get("primary_ci_high"))
    sample_size = _finite_float(payload.get("sample_size"))
    effect_measure = str(payload.get("effect_measure") or "").strip()
    return bool(
        point is not None
        and ci_low is not None
        and ci_high is not None
        and ci_low <= ci_high
        and effect_measure
        and sample_size is not None
        and sample_size > 0
    )


def _extract_primary_effect_payload_from_summary(
    summary: Dict[str, Any],
    *,
    path: Optional[Path],
    preferred_predictor: Optional[str],
) -> Dict[str, Any]:
    nested_primary = _primary_result_payload(summary)
    predictor = (
        (nested_primary or {}).get("predictor")
        or summary.get("primary_predictor")
        or summary.get("primary_exposure")
        or summary.get("predictor")
        or summary.get("predictor_variable")
        or summary.get("variable")
    )
    # The canonical payload field is ``primary_or`` for historical reasons, but it
    # carries the primary effect RATIO whatever its measure: an odds ratio for a
    # logistic/association design, a HAZARD ratio for a survival/time-to-event
    # design. The deterministic Cox runner emits ``hazard_ratio`` (+ CIs); the
    # OR-only key lists do not match it, so a survival estimand can be dropped and
    # a downstream logistic refit can take precedence. Recognise both measures so
    # downstream labels the effect correctly instead of silently calling it "OR".
    direct_or = _finite_float(
        _first_direct_scalar(
            summary,
            ("primary_or", "odds_ratio", "estimate", "adjusted_or"),
        )
    )
    direct_hr = _finite_float(
        _first_direct_scalar(
            summary,
            ("hazard_ratio", "primary_hr", "adjusted_hr"),
        )
    )
    # Scale-neutral point estimate: propensity-weighted causal steps write the
    # estimate under ``adjusted_effect`` and declare the scale separately in
    # ``primary_effect_scale``/``effect_scale``. Without this, a valid effect can
    # be dropped while an unrelated probe scalar wins.
    direct_scaled = _finite_float(
        _first_direct_scalar(summary, ("adjusted_effect", "primary_point_estimate"))
    )
    scaled_measure = _effect_measure_from_scale(
        _first_direct_scalar(
            summary,
            # ``adjusted_effect_scale`` can accompany ``adjusted_effect``;
            # without a declared scale the headline cannot be labeled safely.
            ("primary_effect_scale", "effect_scale", "adjusted_effect_scale"),
        )
    )
    if direct_or is not None:
        direct_primary: Optional[float] = direct_or
        effect_measure: Optional[str] = "OR"
    elif direct_hr is not None:
        direct_primary = direct_hr
        effect_measure = "HR"
    elif direct_scaled is not None and scaled_measure is not None:
        direct_primary = direct_scaled
        effect_measure = scaled_measure
    else:
        direct_primary = None
        effect_measure = None
    nested_primary_or = _finite_float((nested_primary or {}).get("primary_or"))
    if nested_primary_or is not None:
        primary_or = nested_primary_or
        effect_measure = (
            str((nested_primary or {}).get("effect_measure") or "").strip()
            or effect_measure
            or "OR"
        )
    elif direct_primary is not None:
        primary_or = direct_primary
    else:
        primary_or = _primary_effect_from_summary(summary)
    if primary_or is not None and effect_measure is None:
        # Legacy flattened-key fallback (``*_or`` / ``*_estimate``) is OR-shaped.
        effect_measure = "OR"
    ci_low = _finite_float(
        _first_direct_scalar(
            summary,
            (
                "primary_ci_low",
                "primary_or_ci_low",
                "primary_association_ci_low",
                "hazard_ratio_ci_low",
                "primary_hr_ci_low",
                "adjusted_effect_ci_low",
            ),
        )
    )
    if nested_primary is not None and nested_primary.get("primary_ci_low") is not None:
        ci_low = _finite_float(nested_primary.get("primary_ci_low"))
    ci_high = _finite_float(
        _first_direct_scalar(
            summary,
            (
                "primary_ci_high",
                "primary_or_ci_high",
                "primary_association_ci_high",
                "hazard_ratio_ci_high",
                "primary_hr_ci_high",
                "adjusted_effect_ci_high",
            ),
        )
    )
    if nested_primary is not None and nested_primary.get("primary_ci_high") is not None:
        ci_high = _finite_float(nested_primary.get("primary_ci_high"))
    ci_pair = _first_direct_sequence_for_key(
        summary,
        ("primary_or_ci", "primary_ci", "primary_association_ci"),
    )
    if ci_pair is not None and len(ci_pair) >= 2:
        ci_low = _finite_float(ci_pair[0])
        ci_high = _finite_float(ci_pair[1])
    if primary_or is not None and (ci_low is None or ci_high is None):
        se = _finite_float(
            _first_direct_scalar(
                summary,
                (
                    "primary_or_se",
                    "primary_se",
                    "se",
                    "hazard_ratio_se",
                    "adjusted_effect_se",
                ),
            )
        )
        if se is not None and primary_or > 0:
            ci_low = math.exp(math.log(primary_or) - 1.96 * se)
            ci_high = math.exp(math.log(primary_or) + 1.96 * se)
    sample_size = _finite_float(
        _first_direct_scalar(
            summary,
            (
                "n",
                "sample_size",
                "n_total",
                "n_total_stays",
                "n_complete",
                "n_complete_case",
                "complete_case_n",
                "complete_case_measured_lactate_n_from_completed_step",
            ),
        )
    )
    if nested_primary is not None and nested_primary.get("sample_size") is not None:
        sample_size = _finite_float(nested_primary.get("sample_size"))
    model_contracts = summary.get("model_contracts") or []
    if isinstance(model_contracts, list):
        primary_model_id = str(summary.get("primary_model_id") or "").strip()
        primary_contract = next(
            (
                contract
                for contract in model_contracts
                if isinstance(contract, dict)
                and primary_model_id
                and str(contract.get("model_id") or "") == primary_model_id
            ),
            None,
        )
        if primary_contract is None:
            primary_contract = next(
                (
                    contract
                    for contract in model_contracts
                    if isinstance(contract, dict)
                    and str(contract.get("analysis_role") or "").lower() == "primary"
                    and str(contract.get("exposure_role") or "primary").lower()
                    == "primary"
                ),
                None,
            )
        if isinstance(primary_contract, dict):
            if not predictor:
                predictor = primary_contract.get("exposure_source")
            if sample_size is None:
                sample_size = _finite_float(primary_contract.get("n"))
    score_path = path or Path("")
    score = _primary_effect_candidate_score(
        score_path,
        summary=summary,
        preferred_predictor=preferred_predictor,
    )
    if primary_or is not None:
        score += 100
    if direct_primary is not None:
        score += 30
    if ci_low is not None and ci_high is not None:
        score += 25
    if sample_size is not None:
        score += 10
    if nested_primary is not None:
        score += _nested_primary_result_bonus(nested_primary, path=score_path)
    return {
        "predictor": predictor,
        "primary_or": primary_or,
        "primary_ci_low": ci_low,
        "primary_ci_high": ci_high,
        "effect_measure": effect_measure,
        "sample_size": int(sample_size) if sample_size is not None else None,
        "status": ("ok" if primary_or is not None else "summary_missing_primary_or"),
        "step_summary_path": str(path) if path is not None else None,
        "_score": score,
    }


def _primary_result_payload(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return an explicit nested primary-result payload when a repair/reconcile
    step has corrected an earlier off-protocol model.

    Several agents emit the canonical post-repair estimate as
    ``step_summary.primary_result`` while retaining the earlier
    ``offprotocol_reestimated_result`` for disclosure. Treating only top-level
    ``primary_or`` values as authoritative lets the stale model win; this helper
    makes the nested primary contract first-class without hard-coding a case.
    """

    raw = summary.get("primary_result")
    if not isinstance(raw, dict):
        return None
    or_value = _finite_float(
        _first_direct_scalar(
            raw,
            ("primary_or", "adjusted_or", "odds_ratio", "point_estimate", "estimate"),
        )
    )
    hr_value = _finite_float(
        _first_direct_scalar(raw, ("hazard_ratio", "primary_hr", "adjusted_hr"))
    )
    if or_value is not None:
        primary_or = or_value
        effect_measure = "OR"
    elif hr_value is not None:
        primary_or = hr_value
        effect_measure = "HR"
    else:
        return None
    return {
        "primary_or": primary_or,
        "effect_measure": effect_measure,
        "primary_ci_low": _finite_float(
            _first_direct_scalar(
                raw,
                (
                    "primary_ci_low",
                    "primary_or_ci_low",
                    "hazard_ratio_ci_low",
                    "ci_low",
                    "ci_lower",
                    "lower",
                ),
            )
        ),
        "primary_ci_high": _finite_float(
            _first_direct_scalar(
                raw,
                (
                    "primary_ci_high",
                    "primary_or_ci_high",
                    "hazard_ratio_ci_high",
                    "ci_high",
                    "ci_upper",
                    "upper",
                ),
            )
        ),
        "sample_size": _finite_float(
            _first_direct_scalar(
                raw,
                ("n_modeled", "analytic_n", "n", "sample_size", "n_complete_case"),
            )
        ),
        "predictor": raw.get("primary_predictor")
        or raw.get("predictor")
        or summary.get("primary_predictor")
        or summary.get("predictor"),
        "spec_id": str(raw.get("spec_id") or ""),
    }


def _nested_primary_result_bonus(payload: Dict[str, Any], *, path: Path) -> int:
    score = 80
    spec_id = str(payload.get("spec_id") or "").lower()
    path_text = str(path).lower()
    if any(token in spec_id for token in ("locked", "frozen", "primary")):
        score += 60
    if "offprotocol" in spec_id or "off_protocol" in spec_id:
        score -= 80
    if any(
        token in path_text
        for token in ("repair", "reconcile", "reconciliation", "contract", "addendum")
    ):
        score += 30
    return score


def _record_summary_path(record: Dict[str, Any]) -> Optional[Path]:
    for key in ("step_summary_path", "summary_path"):
        value = record.get(key)
        if value:
            return Path(str(value))
    step_id = str(record.get("step_id") or "").strip()
    return Path(step_id) if step_id else None


def _first_direct_scalar(
    payload: Dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _first_direct_sequence_for_key(
    payload: Dict[str, Any],
    keys: tuple[str, ...],
) -> Optional[list[Any]]:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if isinstance(value, (list, tuple)):
            return list(value)
    return None


def _infer_primary_predictor_from_run_dir(run_dir: Path) -> Optional[str]:
    try:
        payload = json.loads(
            (run_dir / "research_context.json").read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            return None
        context = parse_research_context(payload)
    except Exception:
        return None
    return _infer_primary_predictor_from_context(context)


def _primary_effect_candidate_score(
    path: Path,
    *,
    summary: Dict[str, Any],
    preferred_predictor: Optional[str],
) -> int:
    path_text = str(path).lower()
    blob = json.dumps(summary, ensure_ascii=False, default=str).lower()
    predictor = str(
        summary.get("primary_predictor")
        or summary.get("predictor")
        or summary.get("predictor_variable")
        or summary.get("variable")
        or ""
    ).lower()
    score = 0
    if (
        _first_present_scalar(
            summary,
            (
                "primary_or",
                "odds_ratio",
                "estimate",
                "adjusted_or",
                "hazard_ratio",
                "primary_hr",
                "adjusted_effect",
                "primary_point_estimate",
            ),
        )
        is not None
    ):
        score += 100
    if "primary_association" in path_text or "association_model" in path_text:
        score += 30
    if "model" in path_text or "regression" in path_text:
        score += 10
    if summary.get("error"):
        score -= 20
    if "bias" in path_text:
        # Bias-audit runs are downstream cleanup, not primary-association
        # candidates; demote them so cross-database run-summary picks the
        # actual primary-association run for the active question.
        score -= 40
    if preferred_predictor:
        preferred_tokens = _predictor_tokens(preferred_predictor)
        predictor_tokens = _predictor_tokens(predictor)
        path_or_blob_tokens = _predictor_tokens(path_text + " " + blob)
        if (
            preferred_predictor.lower() in predictor
            or preferred_predictor.lower() in path_text
        ):
            score += 80
        elif preferred_tokens & predictor_tokens:
            score += 70
        elif preferred_tokens & path_or_blob_tokens:
            score += 40
        # Demote any candidate whose own predictor / path tokens conflict
        # with the user's preferred predictor — generic anti-cross-contamination
        # rule, not specific to vasopressor or any single benchmark case.
        candidate_predictor_tokens = _predictor_tokens(predictor) | _predictor_tokens(
            path_text
        )
        if preferred_tokens and not (preferred_tokens & candidate_predictor_tokens):
            score -= 60
    elif predictor:
        score += 5
    return score
