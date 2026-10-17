"""A column's ICU reading, with the curated concept hints added since the ICU rules were sealed.

``icu_rules`` belongs to the Figure 2 paper scorer tree, which is sealed by
digest: its bytes change only with a reseal the user approves.  A curated
hint added after the seal lives here, and every production reader of a
column's role, kind, unit or aggregation reads the column through
:func:`classify_variable`, so the pipeline has one reading of it.

A hint here names a concept the ICU rules' own table does not, and applies
as theirs would: to the concept itself and to a summary of it (``adv_resp``,
``adv_resp_max``), never to a longer name that merely begins with its
letters.  A column the ICU rules read as audit metadata, an identifier or a
time keeps their reading: an observation count such as ``adv_resp_n`` is
not the support it counts.  The context's implementation identity binds
this file's bytes beside theirs (``implementation_identity``).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Optional, Sequence

from ..icu_rules import ConceptHint, VariableKind
from ..icu_rules import classify_variable as _icu_classify_variable
from ..schema import AggregationRule, VariableRole

__all__ = ["SUPPLEMENTARY_CONCEPT_HINTS", "classify_variable"]

#: Curated hints for concepts the sealed ICU rules do not name.
SUPPLEMENTARY_CONCEPT_HINTS: Mapping[str, ConceptHint] = MappingProxyType(
    {
        "adv_resp": ConceptHint(
            role=VariableRole.INTERVENTION,
            kind=VariableKind.BINARY,
            aggregation_default=AggregationRule.MAX_LAST,
            pitfalls=(
                "Advanced respiratory support (invasive or non-invasive ventilation, high-flow oxygen) observed in the exposure window may be concurrent treatment or a mediator; do not auto-adjust without pre-time-zero authority.",
            ),
        ),
        "fluid_balance_cumulative": ConceptHint(
            role=VariableRole.INTERVENTION,
            kind=VariableKind.CONTINUOUS,
            unit="mL",
            aggregation_default=AggregationRule.MAX_LAST,
            pitfalls=(
                "Cumulative fluid balance accumulates over the stay: a window mean understates the balance reached at the window's end.",
                "Fluid balance reflects fluid given in response to illness; adjust for it only with pre-time-zero authority.",
            ),
        ),
    }
)

#: Roles the ICU rules give a column that is not a clinical value.
_STRUCTURAL_ROLES = frozenset({VariableRole.META, VariableRole.ID, VariableRole.TIME})


def _supplementary_hint(name: str) -> Optional[ConceptHint]:
    key = str(name or "").lower()
    for concept in sorted(SUPPLEMENTARY_CONCEPT_HINTS, key=len, reverse=True):
        if key == concept or key.startswith(f"{concept}_"):
            return SUPPLEMENTARY_CONCEPT_HINTS[concept]
    return None


def classify_variable(
    name: str,
    dtype: str,
    sample_values: Optional[Sequence] = None,
) -> ConceptHint:
    """The column's hint: the curated one here for its concept, else the ICU rules' reading."""

    reading = _icu_classify_variable(name, dtype, sample_values)
    hint = _supplementary_hint(name)
    if hint is None or reading.role in _STRUCTURAL_ROLES:
        return reading
    return hint
