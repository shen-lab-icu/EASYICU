"""ICU-specific reasoning rules used to constrain the analysis agent.

This file is the *medical knowledge* layer. It encodes the kind of
things an ICU researcher knows but a generic LLM does not reliably
infer from a CSV: that SOFA components are ordinal and should not be
averaged, that creatinine has a meaningful physiological range,
that vasopressor exposure is conventionally summarised over a
specific time window, and so on.

Three concrete uses:

1. ``classify_variable`` — given a concept name and dtype,
   return a sensible :class:`~.schema.ConceptDescriptor` skeleton.
2. ``aggregation_rule_for`` — what aggregation operations are valid
   for a given variable role / kind. The :class:`ConceptUsageAuditor`
   enforces this against generated code.
3. ``default_time_windows`` — opinionated default windows for
   common ICU analyses (first 24 h, full ICU stay, first 6 h).

The rule set is intentionally *opinionated and small*. A reviewer
should be able to read ``ICU_RULES`` end to end in a few minutes and
either accept it or override it for a specific study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .schema import AggregationRule, TimeWindow, VariableRole


class VariableKind(str, Enum):
    """Coarse-grained variable kind, used by the LLM prompt and the auditor.

    Distinct from :class:`~.schema.VariableRole`: ``role`` describes
    *what* the column means in an analysis (outcome, demographic,
    intervention), ``kind`` describes *how* the values behave
    statistically (continuous, ordinal, binary, categorical, count,
    timestamp, identifier).
    """

    CONTINUOUS = "continuous"
    ORDINAL = "ordinal"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    COUNT = "count"
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ConceptHint:
    """Hard-coded ICU knowledge about a frequently-used concept."""

    role: VariableRole
    kind: VariableKind
    unit: Optional[str] = None
    valid_range: Optional[Tuple[float, float]] = None
    is_ordinal: bool = False
    ordinal_levels: Optional[Tuple[int, ...]] = None
    aggregation_default: AggregationRule = AggregationRule.ANY
    pitfalls: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Concept hint table
# ---------------------------------------------------------------------------
#
# The keys are matched as case-insensitive *prefixes* against concept
# / column names. So ``sofa_resp`` and ``sofa_resp_24h`` both pick up
# the ``sofa_resp`` hint. We keep the table small and curated rather
# than trying to be exhaustive — concepts not covered fall back to
# ``UNKNOWN`` and the LLM is told to be conservative.

_SOFA_COMP = ConceptHint(
    role=VariableRole.ORDINAL_SCORE, kind=VariableKind.ORDINAL,
    valid_range=(0.0, 4.0),
    is_ordinal=True,
    ordinal_levels=(0, 1, 2, 3, 4),
    aggregation_default=AggregationRule.MAX_LAST,
    pitfalls=(
        "SOFA components are 0–4 ordinal levels; never average. Aggregate by max within window.",
        "A component value of 0 may mean 'no dysfunction' OR 'inputs unavailable'; check missingness of upstream inputs (e.g. PaO2/FiO2 for resp) before interpreting.",
    ),
)


_CONCEPT_HINTS: Dict[str, ConceptHint] = {
    # --- demographics
    "age": ConceptHint(
        role=VariableRole.DEMOGRAPHIC,
        kind=VariableKind.CONTINUOUS,
        unit="years",
        valid_range=(0.0, 120.0),
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "sex": ConceptHint(
        role=VariableRole.DEMOGRAPHIC,
        kind=VariableKind.CATEGORICAL,
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "weight": ConceptHint(
        role=VariableRole.DEMOGRAPHIC,
        kind=VariableKind.CONTINUOUS,
        unit="kg",
        valid_range=(20.0, 350.0),
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "height": ConceptHint(
        role=VariableRole.DEMOGRAPHIC,
        kind=VariableKind.CONTINUOUS,
        unit="cm",
        valid_range=(120.0, 230.0),
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    # --- common vitals
    "hr": ConceptHint(
        role=VariableRole.VITAL, kind=VariableKind.CONTINUOUS,
        unit="bpm", valid_range=(20.0, 250.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "map": ConceptHint(
        role=VariableRole.VITAL, kind=VariableKind.CONTINUOUS,
        unit="mmHg", valid_range=(20.0, 200.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "sbp": ConceptHint(
        role=VariableRole.VITAL, kind=VariableKind.CONTINUOUS,
        unit="mmHg", valid_range=(40.0, 260.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "dbp": ConceptHint(
        role=VariableRole.VITAL, kind=VariableKind.CONTINUOUS,
        unit="mmHg", valid_range=(20.0, 180.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "spo2": ConceptHint(
        role=VariableRole.VITAL, kind=VariableKind.CONTINUOUS,
        unit="%", valid_range=(40.0, 100.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "temp": ConceptHint(
        role=VariableRole.VITAL, kind=VariableKind.CONTINUOUS,
        unit="C", valid_range=(28.0, 43.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    # --- common labs
    "lact": ConceptHint(
        role=VariableRole.LAB, kind=VariableKind.CONTINUOUS,
        unit="mmol/L", valid_range=(0.0, 30.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
        pitfalls=(
            "Lactate is right-skewed and measurement is often clinically triggered; report median/IQR or clinically meaningful bins, and audit unmeasured lactate separately.",
            "Do not interpret lactate missingness as normal lactate without an explicit measurement-status indicator.",
        ),
    ),
    "creat": ConceptHint(
        role=VariableRole.LAB, kind=VariableKind.CONTINUOUS,
        unit="mg/dL", valid_range=(0.1, 15.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
        pitfalls=(
            "Creatinine is right-skewed; report median (IQR), not mean (SD), unless log-transformed.",
        ),
    ),
    "bili": ConceptHint(
        role=VariableRole.LAB, kind=VariableKind.CONTINUOUS,
        unit="mg/dL", valid_range=(0.0, 50.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "plt": ConceptHint(
        role=VariableRole.LAB, kind=VariableKind.CONTINUOUS,
        unit="K/uL", valid_range=(1.0, 1500.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "pafi": ConceptHint(
        role=VariableRole.LAB, kind=VariableKind.CONTINUOUS,
        unit="ratio", valid_range=(20.0, 700.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
        pitfalls=(
            "PaO2/FiO2 ratios are only meaningful when FiO2 is reliably recorded; "
            "missing FiO2 should not be silently imputed to 0.21.",
        ),
    ),
    # --- ordinal scores
    "gcs": ConceptHint(
        role=VariableRole.ORDINAL_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(3.0, 15.0),
        is_ordinal=True,
        ordinal_levels=tuple(range(3, 16)),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "GCS is ordinal; do not take its mean. Report worst (min) or representative (last/first) GCS.",
        ),
    ),
    "kdigo_stage": ConceptHint(
        role=VariableRole.ORDINAL_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(0.0, 3.0),
        is_ordinal=True,
        ordinal_levels=(0, 1, 2, 3),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "KDIGO AKI stage is ordinal; do not average stages. Report peak/worst stage within the prespecified window.",
            "Creatinine-based KDIGO staging is sensitive to baseline creatinine and measurement frequency; audit missingness and baseline assumptions.",
        ),
    ),
    "kdigo": ConceptHint(
        role=VariableRole.ORDINAL_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(0.0, 3.0),
        is_ordinal=True,
        ordinal_levels=(0, 1, 2, 3),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "KDIGO AKI stage is ordinal; do not average stages. Report peak/worst stage within the prespecified window.",
            "Creatinine-based KDIGO staging is sensitive to baseline creatinine and measurement frequency; audit missingness and baseline assumptions.",
        ),
    ),
    # --- SOFA components and totals (both sofa and sofa2 conventions)
    "sofa_resp": _SOFA_COMP,
    "sofa_coag": _SOFA_COMP,
    "sofa_liver": _SOFA_COMP,
    "sofa_cardio": _SOFA_COMP,
    "sofa_cns": _SOFA_COMP,
    "sofa_renal": _SOFA_COMP,
    "sofa2_resp": _SOFA_COMP,
    "sofa2_coag": _SOFA_COMP,
    "sofa2_liver": _SOFA_COMP,
    "sofa2_cardio": _SOFA_COMP,
    "sofa2_cns": _SOFA_COMP,
    "sofa2_renal": _SOFA_COMP,
    "sofa": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(0.0, 24.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Total SOFA is a sum of six 0–4 components; treat as ordinal/integer-count, not continuous.",
            "A SOFA total of 0 across many patients may indicate that one or more component inputs were missing rather than that organ dysfunction was truly absent. Always cross-check component-level missingness before drawing clinical conclusions.",
        ),
    ),
    "sofa2": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(0.0, 24.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "SOFA-2 follows the same 0–4 component structure as SOFA; treat as ordinal.",
            "EasyICU has empirically observed elevated mortality in the sofa2==0 stratum on at least one source; this often reflects component-level missingness, not low illness severity. Verify component availability before reporting.",
        ),
    ),
    "sirs": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(0.0, 4.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "qsofa": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(0.0, 3.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    # --- outcomes
    "death": ConceptHint(
        role=VariableRole.OUTCOME, kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.FIRST_VALUE,
        pitfalls=(
            "ICU mortality vs hospital mortality vs 28-day mortality are NOT interchangeable; record which one is in use.",
        ),
    ),
    "death_icu": ConceptHint(
        role=VariableRole.OUTCOME, kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "death_hosp": ConceptHint(
        role=VariableRole.OUTCOME, kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "los_icu": ConceptHint(
        role=VariableRole.OUTCOME, kind=VariableKind.CONTINUOUS,
        unit="days", valid_range=(0.0, 365.0),
        aggregation_default=AggregationRule.FIRST_VALUE,
        pitfalls=(
            "ICU LoS is right-skewed and competing-risks-affected; report median (IQR), and consider survival framing rather than linear regression.",
        ),
    ),
    "los_hosp": ConceptHint(
        role=VariableRole.OUTCOME, kind=VariableKind.CONTINUOUS,
        unit="days", valid_range=(0.0, 365.0),
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    # --- interventions
    "vaso": ConceptHint(
        role=VariableRole.INTERVENTION, kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Vasopressor exposure is conventionally captured as 'any vasopressor in window' (binary) "
            "or 'cumulative norepinephrine-equivalent' (continuous); pick one and document it.",
            "Vasopressor exposure is confounded by indication; association models should avoid causal treatment-effect language.",
        ),
    ),
    "norepi_equiv": ConceptHint(
        role=VariableRole.INTERVENTION, kind=VariableKind.CONTINUOUS,
        unit="mcg/kg/min",
        valid_range=(0.0, 10.0),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Norepinephrine-equivalent dose is an intervention intensity measure; summarize peak or cumulative exposure within a prespecified window.",
            "Dose comparisons require unit harmonisation across source databases.",
        ),
    ),
    "circ_event": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE, kind=VariableKind.ORDINAL,
        valid_range=(0.0, 3.0),
        is_ordinal=True,
        ordinal_levels=(0, 1, 2, 3),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Circulatory failure event level is ordinal; use maximum/worst event within the window, not mean event level.",
        ),
    ),
    "circ_failure": ConceptHint(
        role=VariableRole.INTERVENTION, kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Circulatory failure is a derived EasyICU concept based on lactate, MAP and vasoactive support; document the underlying rule and time window.",
        ),
    ),
    "vent_ind": ConceptHint(
        role=VariableRole.INTERVENTION, kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "rrt": ConceptHint(
        role=VariableRole.INTERVENTION, kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    # --- ids / time
    "patient_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "icustay_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "hadm_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "stay_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "subject_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
}


def _lookup_hint(name: str) -> Optional[ConceptHint]:
    """Find the longest matching prefix hint for ``name`` (case-insensitive)."""
    key = name.lower()
    # exact match first
    if key in _CONCEPT_HINTS:
        return _CONCEPT_HINTS[key]
    # longest prefix match second
    candidates = sorted(
        (k for k in _CONCEPT_HINTS if _matches_concept_prefix(key, k)),
        key=len,
        reverse=True,
    )
    if candidates:
        return _CONCEPT_HINTS[candidates[0]]
    return None


def _matches_concept_prefix(name: str, prefix: str) -> bool:
    if not name.startswith(prefix):
        return False
    if len(name) == len(prefix):
        return True
    return name[len(prefix)] == "_"


def classify_variable(
    name: str,
    dtype: str,
    sample_values: Optional[Sequence] = None,
) -> ConceptHint:
    """Return a best-effort :class:`ConceptHint` for an unknown column.

    Falls back to type-based classification when no concept hint matches.
    """
    hint = _lookup_hint(name)
    if hint is not None:
        return hint

    dtype_l = (dtype or "").lower()
    if "datetime" in dtype_l or "timestamp" in dtype_l:
        return ConceptHint(role=VariableRole.TIME, kind=VariableKind.TIMESTAMP)
    if "bool" in dtype_l:
        return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.BINARY)
    if "int" in dtype_l:
        if sample_values is not None:
            uniq = {v for v in sample_values if v is not None}
            if uniq <= {0, 1}:
                return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.BINARY)
            if len(uniq) <= 5:
                return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.ORDINAL,
                                   is_ordinal=True)
        return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.COUNT)
    if "float" in dtype_l or "double" in dtype_l:
        return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.CONTINUOUS,
                           aggregation_default=AggregationRule.MEAN_MEDIAN)
    if "object" in dtype_l or "string" in dtype_l or "category" in dtype_l:
        return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.CATEGORICAL)
    return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.UNKNOWN)


def aggregation_rule_for(role: VariableRole, kind: VariableKind) -> List[AggregationRule]:
    """Return the *allowed* aggregation operations for a (role, kind) pair."""
    if kind == VariableKind.IDENTIFIER:
        return [AggregationRule.NONE, AggregationRule.FIRST_VALUE]
    if kind == VariableKind.TIMESTAMP:
        return [AggregationRule.NONE, AggregationRule.FIRST_VALUE]
    if kind == VariableKind.BINARY:
        return [AggregationRule.MAX_LAST, AggregationRule.FIRST_VALUE, AggregationRule.SUM]
    if kind == VariableKind.ORDINAL:
        return [AggregationRule.MAX_LAST, AggregationRule.FIRST_VALUE, AggregationRule.MEDIAN_ONLY]
    if kind == VariableKind.COUNT:
        return [AggregationRule.SUM, AggregationRule.MEAN_MEDIAN, AggregationRule.MAX_LAST]
    if kind == VariableKind.CONTINUOUS:
        if role == VariableRole.LAB:
            # labs are typically right-skewed → discourage mean
            return [AggregationRule.MEDIAN_ONLY, AggregationRule.MAX_LAST, AggregationRule.FIRST_VALUE]
        return [AggregationRule.MEAN_MEDIAN, AggregationRule.MEDIAN_ONLY,
                AggregationRule.MAX_LAST, AggregationRule.FIRST_VALUE]
    if kind == VariableKind.CATEGORICAL:
        return [AggregationRule.FIRST_VALUE, AggregationRule.NONE]
    return [AggregationRule.ANY]


def default_time_windows() -> List[TimeWindow]:
    """Opinionated default windows for ICU analyses.

    We expose three:

    * ``first_24h`` — the canonical "admission illness severity" window;
    * ``first_6h`` — useful for sepsis-bundle and resuscitation studies;
    * ``full_stay`` — the entire ICU stay (anchored at admission, ends
      at discharge — the analysis script must cap end_hours per patient).
    """
    return [
        TimeWindow(name="first_24h", anchor="icu_admission",
                   start_hours=0.0, end_hours=24.0,
                   rationale="Standard window for admission severity scores (SOFA, APACHE)."),
        TimeWindow(name="first_6h", anchor="icu_admission",
                   start_hours=0.0, end_hours=6.0,
                   rationale="Early-resuscitation window for sepsis / shock studies."),
        TimeWindow(name="full_stay", anchor="icu_admission",
                   start_hours=0.0, end_hours=24.0 * 30,
                   rationale="Whole ICU stay; analysis script must cap end_hours per patient."),
    ]


# A frozen dictionary the agents and validators read. Wrapped in a
# tiny class so the prompt has a single token to refer to.
@dataclass(frozen=True)
class _ICURules:
    aggregation_rule_for: Callable = field(default=aggregation_rule_for)
    classify_variable: Callable = field(default=classify_variable)
    default_time_windows: Callable = field(default=default_time_windows)


ICU_RULES = _ICURules()


__all__ = [
    "VariableKind",
    "ConceptHint",
    "classify_variable",
    "aggregation_rule_for",
    "default_time_windows",
    "ICU_RULES",
]
