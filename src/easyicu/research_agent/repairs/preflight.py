"""Ordered deterministic repairs for mechanical preflight findings."""

from __future__ import annotations

from typing import Sequence

from ..schema import ValidationFinding
from .concept_preflight import patch_concept_preflight_repairs
from .interval_method import patch_statsmodels_interval_method_label


def patch_preflight_repairs(
    code: str,
    *,
    findings: Sequence[ValidationFinding],
) -> tuple[str, list[str]]:
    """Apply independent coordinate-bound preflight repairs in stable order."""

    repaired, names = patch_concept_preflight_repairs(code, findings=findings)
    candidate = patch_statsmodels_interval_method_label(repaired, findings=findings)
    if candidate != repaired:
        repaired = candidate
        names.append("statsmodels_interval_method_label_v1")
    return repaired, names


__all__ = ["patch_preflight_repairs"]
