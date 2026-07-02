"""Primary-effect candidate selection helpers extracted from pipeline.py.

When the cross-database aggregator needs one canonical primary OR per
database, it walks every ``step_summary.json`` under a finished run and
picks the candidate that best matches the original research question.
The scoring + path inference live here so pipeline.py can stay focused
on orchestration.

Moved out on 2026-05-27 as part of the pipeline.py size-reduction effort;
behaviour is unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

from .context import ResearchContext
from .plan_utils import (
    _finite_float,
    _infer_primary_predictor_from_context,
    _predictor_tokens,
    _primary_effect_from_summary,
)
from .scalar_utils import _first_present_scalar
from .schema import PipelineResult


__all__ = [
    "_extract_primary_effect_row",
    "_extract_primary_effect_payload_from_records",
    "_extract_primary_effect_payload_from_summary",
    "_infer_primary_predictor_from_run_dir",
    "_primary_effect_candidate_score",
]


def _extract_primary_effect_row(
    *, database: str, result: PipelineResult
) -> Dict[str, Any]:
    run_dir = Path(result.workdir)
    preferred_predictor = _infer_primary_predictor_from_run_dir(run_dir)
    summary_candidates = sorted(run_dir.rglob("step_summary.json"))
    payload: Dict[str, Any] = {
        "database": database,
        "run_id": result.run_id,
        "manifest_path": result.manifest_path,
        "predictor": None,
        "primary_or": None,
        "primary_ci_low": None,
        "primary_ci_high": None,
        "status": "missing_primary_association",
    }
    best_payload: Optional[Dict[str, Any]] = None
    best_score = -10_000
    for path in summary_candidates:
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(summary, dict):
            continue
        candidate_payload = _extract_primary_effect_payload_from_summary(
            summary,
            path=path,
            preferred_predictor=preferred_predictor,
        )
        score = int(candidate_payload.pop("_score"))
        if score > best_score:
            best_score = score
            best_payload = candidate_payload
    if best_payload is not None:
        payload.update(best_payload)
    return payload


def _extract_primary_effect_payload_from_records(
    per_step_records: Any,
    *,
    preferred_predictor: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Pick the best primary-effect payload from in-memory step records."""

    best_payload: Optional[Dict[str, Any]] = None
    best_score = -10_000
    for record in per_step_records or []:
        if not isinstance(record, dict):
            continue
        summary = record.get("step_summary")
        if not isinstance(summary, dict):
            continue
        path = _record_summary_path(record)
        candidate_payload = _extract_primary_effect_payload_from_summary(
            summary,
            path=path,
            preferred_predictor=preferred_predictor,
        )
        score = int(candidate_payload.pop("_score"))
        if score > best_score:
            best_score = score
            best_payload = candidate_payload
            best_payload["step_id"] = str(record.get("step_id") or "")
            best_payload["evidence_id"] = str(
                record.get("step_summary_evidence_id") or ""
            )
    return best_payload


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
        or summary.get("predictor")
        or summary.get("predictor_variable")
        or summary.get("variable")
    )
    direct_primary_or = _finite_float(
        _first_direct_scalar(
            summary,
            ("primary_or", "odds_ratio", "estimate", "adjusted_or"),
        )
    )
    nested_primary_or = _finite_float((nested_primary or {}).get("primary_or"))
    primary_or = nested_primary_or
    if primary_or is None:
        primary_or = (
            direct_primary_or
            if direct_primary_or is not None
            else _primary_effect_from_summary(summary)
        )
    ci_low = _finite_float(
        _first_direct_scalar(
            summary,
            ("primary_ci_low", "primary_or_ci_low", "primary_association_ci_low"),
        )
    )
    if nested_primary is not None and nested_primary.get("primary_ci_low") is not None:
        ci_low = _finite_float(nested_primary.get("primary_ci_low"))
    ci_high = _finite_float(
        _first_direct_scalar(
            summary,
            ("primary_ci_high", "primary_or_ci_high", "primary_association_ci_high"),
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
            _first_direct_scalar(summary, ("primary_or_se", "primary_se", "se"))
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
    score_path = path or Path("")
    score = _primary_effect_candidate_score(
        score_path,
        summary=summary,
        preferred_predictor=preferred_predictor,
    )
    if primary_or is not None:
        score += 100
    if direct_primary_or is not None:
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
    primary_or = _finite_float(
        _first_direct_scalar(
            raw,
            ("primary_or", "adjusted_or", "odds_ratio", "point_estimate", "estimate"),
        )
    )
    if primary_or is None:
        return None
    return {
        "primary_or": primary_or,
        "primary_ci_low": _finite_float(
            _first_direct_scalar(
                raw,
                ("primary_ci_low", "primary_or_ci_low", "ci_low", "ci_lower", "lower"),
            )
        ),
        "primary_ci_high": _finite_float(
            _first_direct_scalar(
                raw,
                (
                    "primary_ci_high",
                    "primary_or_ci_high",
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
        context = ResearchContext.model_validate(payload)
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
            ("primary_or", "odds_ratio", "estimate", "adjusted_or"),
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
