"""Small scalar / nested-dict helpers shared between pipeline phases.

These functions are used across three responsibility boundaries:

* :mod:`code_repair` — the deterministic step-summary repair logic needs
  to read primary-effect numerics out of arbitrarily nested coder JSON,
  no matter whether the coder placed the value at ``primary_or`` or at
  ``robustness_analysis_manifest.primary_or`` or inside a list of dicts.
* :mod:`pipeline` (still) — the prediction / publication bundle renderer
  pulls the same numerics back out at write time.
* :mod:`audits.manuscript_claims` — manuscript claim verification resolves
  candidate numeric values without importing the pipeline orchestration layer.

They are intentionally tiny and have no pipeline state. Lifting them
into a dedicated module breaks the import cycle that would otherwise
force ``pipeline`` and ``code_repair`` to import each other.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from .schema import AnalysisStep
from .numeric_scalars import coerce_finite_float


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
        for index, value in enumerate(payload):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flat.update(_flatten_scalar_dict(value, prefix=child_prefix))
        return flat
    scalar = _coerce_scalar(payload)
    if scalar is not None and prefix:
        flat[prefix] = scalar
    return flat


def _first_present_scalar(
    payload: Dict[str, Any], keys: Sequence[str]
) -> Optional[Union[int, float, str, bool]]:
    """Return the value the summary declares for the first matching field.

    Two passes, because a value reached through a list index is not the same
    kind of fact as a value the summary states directly.

    1. Declared fields -- the key itself, or a key nested through mappings only.
    2. Values inside a list. A metric recorded as one row of
       ``prediction_robustness_results`` is still that step's metric, so these
       do answer the lookup -- but only when every list occurrence agrees.

    The ordering and the agreement rule both exist for one observed defect:
    once ``_flatten_scalar_dict`` began recursing into lists,
    ``cluster_selection.candidates[0].n_clusters`` -- the FIRST EVALUATED
    candidate in a search grid -- started answering the lookup for
    ``n_clusters`` ahead of the top-level ``cluster_count``. A two-cluster
    solution reported one cluster, contradicted its own selection manifest, and
    the clustering step contract failed closed on a valid result. A candidate
    roster disagrees with itself by construction, so it now yields nothing and
    the declared count is used instead.
    """

    flat = _flatten_scalar_dict(payload)
    declared = {key: value for key, value in flat.items() if "[" not in key}
    enumerated = {key: value for key, value in flat.items() if "[" in key}

    for key in keys:
        if key not in payload:
            for flat_key, flat_value in declared.items():
                if flat_key.endswith(f".{key}"):
                    value = _coerce_scalar(flat_value)
                    if value is not None:
                        return value
            continue
        value = _coerce_scalar(payload.get(key))
        if value is not None:
            return value

    for key in keys:
        if key in payload:
            continue
        occurrences = [
            _coerce_scalar(flat_value)
            for flat_key, flat_value in enumerated.items()
            if flat_key.endswith(f".{key}")
        ]
        occurrences = [value for value in occurrences if value is not None]
        if not occurrences:
            continue
        if len({repr(value) for value in occurrences}) == 1:
            return occurrences[0]
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
    """Extract only explicitly labelled prose effects.

    ``OR`` is an acronym only when it is written in uppercase.  Treating it
    case-insensitively makes the ordinary English conjunction in text such as
    ``"increase >=0.3 mg/dL or 1.5-1.9 times baseline"`` look like an odds
    ratio.  Spelled-out ``odds ratio`` remains case-insensitive.
    """

    text = json.dumps(payload, ensure_ascii=False, default=str)
    patterns = (
        (r"\bOR\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)", 0),
        (
            r"\bodds\s+ratio\b\s*(?:=|:|of)?\s*([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE,
        ),
    )
    for pattern, flags in patterns:
        match = re.search(pattern, text, flags=flags)
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
    "coerce_finite_float",
    "_expected_numeric_annotations_for_step",
    "_coerce_scalar",
    "_first_present_scalar",
    "_first_numeric_scalar_with_key_fragment",
    "_flatten_scalar_dict",
    "_first_numeric_effect_from_text",
]
