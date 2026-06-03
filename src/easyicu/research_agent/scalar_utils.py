"""Small scalar / nested-dict helpers shared between pipeline phases.

These functions are used in two places:

* :mod:`code_repair` — the deterministic step-summary repair logic needs
  to read primary-effect numerics out of arbitrarily nested coder JSON,
  no matter whether the coder placed the value at ``primary_or`` or at
  ``robustness_analysis_manifest.primary_or`` or inside a list of dicts.
* :mod:`pipeline` (still) — the prediction / publication bundle renderer
  pulls the same numerics back out at write time.

They are intentionally tiny and have no pipeline state. Lifting them
into a dedicated module breaks the import cycle that would otherwise
force ``pipeline`` and ``code_repair`` to import each other.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from .schema import AnalysisStep


def _expected_numeric_annotations_for_step(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
) -> Dict[str, float]:
    """Return the small set of numeric values we expect a figure to annotate."""
    if not isinstance(step_summary, dict) or not step_summary:
        return {}
    keys: List[str] = []
    step_id = (step.step_id or "").lower()
    if "primary_association" in step_id:
        keys = ["primary_or", "primary_ci_low", "primary_ci_high"]
    elif "outcome_incidence" in step_id:
        keys = ["outcome_rate"]
    expected: Dict[str, float] = {}
    for key in keys:
        value = step_summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            expected[key] = float(value)
    return expected


def _coerce_scalar(value: Any) -> Optional[Union[int, float, str, bool]]:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    return None


def _flatten_scalar_dict(
    payload: Any,
    *,
    prefix: str = "",
) -> Dict[str, Union[int, float, str, bool]]:
    flat: Dict[str, Union[int, float, str, bool]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_scalar_dict(value, prefix=child_prefix))
        return flat
    if isinstance(payload, list):
        return flat
    scalar = _coerce_scalar(payload)
    if scalar is not None and prefix:
        flat[prefix] = scalar
    return flat


def _first_present_scalar(
    payload: Dict[str, Any], keys: Sequence[str]
) -> Optional[Union[int, float, str, bool]]:
    flat = _flatten_scalar_dict(payload)
    for key in keys:
        if key not in payload:
            for flat_key, flat_value in flat.items():
                if flat_key.endswith(f".{key}"):
                    value = _coerce_scalar(flat_value)
                    if value is not None:
                        return value
            continue
        value = _coerce_scalar(payload.get(key))
        if value is not None:
            return value
    return None


def _first_numeric_scalar_with_key_fragment(
    payload: Dict[str, Any], fragments: Sequence[str]
) -> Optional[float]:
    """Return the first numeric scalar whose flattened key contains a fragment."""

    lowered_fragments = tuple(fragment.lower() for fragment in fragments if fragment)
    if not lowered_fragments:
        return None
    for key, value in _flatten_scalar_dict(payload).items():
        lowered = key.lower()
        if not any(fragment in lowered for fragment in lowered_fragments):
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                continue
    return None


def _first_numeric_effect_from_text(payload: Any) -> Optional[float]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    patterns = (
        r"\b(?:OR|odds\s+ratio)\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:adjusted\s+OR|adjusted\s+odds\s+ratio)\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


__all__ = [
    "_expected_numeric_annotations_for_step",
    "_coerce_scalar",
    "_first_present_scalar",
    "_first_numeric_scalar_with_key_fragment",
    "_flatten_scalar_dict",
    "_first_numeric_effect_from_text",
]
