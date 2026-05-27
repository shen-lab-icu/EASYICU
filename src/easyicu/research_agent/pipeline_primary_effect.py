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
from pathlib import Path
from typing import Any, Dict, Optional

from .context import ResearchContext
from .plan_utils import _infer_primary_predictor_from_context, _predictor_tokens
from .scalar_utils import _first_present_scalar
from .schema import PipelineResult


__all__ = [
    "_extract_primary_effect_row",
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
        predictor = (
            summary.get("primary_predictor")
            or summary.get("predictor")
            or summary.get("predictor_variable")
            or summary.get("variable")
        )
        primary_or = _first_present_scalar(
            summary,
            ("primary_or", "odds_ratio", "estimate", "adjusted_or", "lactate_or"),
        )
        ci_low = _first_present_scalar(
            summary,
            ("primary_ci_low", "primary_or_ci_low", "ci_low", "ci_lower", "lower"),
        )
        ci_high = _first_present_scalar(
            summary,
            ("primary_ci_high", "primary_or_ci_high", "ci_high", "ci_upper", "upper"),
        )
        if (
            primary_or is None
            and "primary_or_ci" in summary
            and isinstance(summary["primary_or_ci"], (list, tuple))
        ):
            vals = list(summary["primary_or_ci"])
            if len(vals) >= 2:
                ci_low, ci_high = vals[0], vals[1]
        score = _primary_effect_candidate_score(
            path,
            summary=summary,
            preferred_predictor=preferred_predictor,
        )
        candidate_payload = {
            "predictor": predictor,
            "primary_or": primary_or,
            "primary_ci_low": ci_low,
            "primary_ci_high": ci_high,
            "status": (
                "ok" if primary_or is not None else "summary_missing_primary_or"
            ),
            "step_summary_path": str(path),
        }
        if score > best_score:
            best_score = score
            best_payload = candidate_payload
    if best_payload is not None:
        payload.update(best_payload)
    return payload


def _infer_primary_predictor_from_run_dir(run_dir: Path) -> Optional[str]:
    try:
        payload = json.loads((run_dir / "research_context.json").read_text(encoding="utf-8"))
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
    if _first_present_scalar(
        summary,
        ("primary_or", "odds_ratio", "estimate", "adjusted_or", "lactate_or"),
    ) is not None:
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
        if preferred_predictor.lower() in predictor or preferred_predictor.lower() in path_text:
            score += 80
        elif preferred_tokens & predictor_tokens:
            score += 70
        elif preferred_tokens & path_or_blob_tokens:
            score += 40
        # Demote any candidate whose own predictor / path tokens conflict
        # with the user's preferred predictor — generic anti-cross-contamination
        # rule, not specific to vasopressor or any single benchmark case.
        candidate_predictor_tokens = (
            _predictor_tokens(predictor) | _predictor_tokens(path_text)
        )
        if preferred_tokens and not (preferred_tokens & candidate_predictor_tokens):
            score -= 60
    elif predictor:
        score += 5
    return score
