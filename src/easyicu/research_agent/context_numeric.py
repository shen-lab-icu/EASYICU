"""Curated NumericClaim registration for frozen research context metadata."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable, Optional

from .evidence import EvidenceStore, NumericClaim
from .schema import ConceptDescriptor, ResearchContext


_AGGREGATION_SUFFIX_RE = re.compile(
    r"_(?:first|last|min|max|mean|median|q25|q75|n|count|measured)$",
    re.IGNORECASE,
)


def _finite_number(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _literal_for_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _variable_family(name: str) -> str:
    family = _AGGREGATION_SUFFIX_RE.sub("", str(name or ""))
    return family or str(name or "")


def _context_missingness_claims(
    variables: Iterable[ConceptDescriptor],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for variable in variables:
        missingness = variable.missingness
        if missingness is None:
            continue
        value = _finite_number(missingness.fraction_missing)
        if value is None:
            continue
        grouped[_variable_family(variable.name)].append(value)

    claims: dict[str, float] = {}
    for family, values in grouped.items():
        if not values:
            continue
        unique_values = sorted({round(value, 12) for value in values})
        if len(unique_values) == 1:
            claims[f"variable_groups.{family}.missingness.fraction_missing"] = values[0]
            continue
        claims[f"variable_groups.{family}.missingness.min_fraction_missing"] = min(values)
        claims[f"variable_groups.{family}.missingness.max_fraction_missing"] = max(values)
    return claims


def register_context_numeric_claims(
    evidence: EvidenceStore,
    *,
    context: ResearchContext,
    evidence_id: str = "research_context",
    step_id: str = "research_context",
) -> list[NumericClaim]:
    """Register manuscript-facing context numbers without expanding the whole tree.

    The writer may legitimately cite source-context facts in Methods and
    limitations, such as the source export size or a variable group's baseline
    missingness. Register only that compact surface so provenance exists without
    flooding the NumericClaim registry with every observed-domain min/max and
    repeated per-variable ``n_total`` value.
    """

    payload: dict[str, float] = {
        "cohort.n_variables": float(len(context.variables)),
    }
    n_stays = _finite_number(context.cohort.n_stays)
    n_patients = _finite_number(context.cohort.n_patients)
    if n_stays is not None and n_patients is not None and n_stays == n_patients:
        payload["cohort.n_stays_and_patients"] = n_stays
    else:
        if n_stays is not None:
            payload["cohort.n_stays"] = n_stays
        if n_patients is not None:
            payload["cohort.n_patients"] = n_patients
    payload.update(_context_missingness_claims(context.variables))

    registered: list[NumericClaim] = []
    for source_field, canonical in payload.items():
        registered.append(
            evidence.register_numeric_claim(
                value=_literal_for_number(canonical),
                canonical=canonical,
                evidence_id=evidence_id,
                step_id=step_id,
                source_field=source_field,
            )
        )
    return registered


__all__ = ["register_context_numeric_claims"]
