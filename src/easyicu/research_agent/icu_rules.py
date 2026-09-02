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

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class MethodologicalPrinciple:
    """A cross-cutting, database-agnostic ICU-analysis reasoning principle.

    Unlike :class:`ConceptHint` (per-concept), these encode reasoning
    hazards that apply *across* concepts and analysis phases. They are
    deliberately **case-neutral** — no single benchmark task, variable,
    score, or database is privileged (prompt hygiene). Database-specific
    *variation* is recorded separately in ``cross_db_note`` so the agent
    reasons comprehensively across the supported databases instead of
    hard-coding one database's behaviour. The ``cross_db_note`` content is
    illustrative ("e.g. ..."), not an exhaustive per-database contract.
    """

    id: str
    # Coarse analysis phase the principle guards: one of
    # ``cohort`` / ``features`` / ``label`` / ``modeling`` /
    # ``clustering`` / ``interpretation``.
    phase: str
    # Impartiality contract. ``error`` = an objective methodological mistake
    # that is wrong under *any* study design (e.g. outcome leakage, averaging
    # an ordinal score, train/test split that leaks a patient across folds) —
    # safe to flag firmly and to gate on. ``caution`` = a defensible analytical
    # *choice* (e.g. mean vs median, dedup vs cluster-robust SE, imputation
    # strategy, which metric) — the agent must only prompt the analyst to
    # state/justify the choice, NEVER override it. This split is deliberate:
    # the rule layer must not impose our analytical preferences on the user.
    kind: Literal["error", "caution"]
    principle: str
    rationale: str
    cross_db_note: str = ""


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
    role=VariableRole.ORDINAL_SCORE,
    kind=VariableKind.ORDINAL,
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
    "adm": ConceptHint(
        role=VariableRole.DEMOGRAPHIC,
        kind=VariableKind.CATEGORICAL,
        aggregation_default=AggregationRule.FIRST_VALUE,
        pitfalls=(
            "Admission type is a baseline admission attribute; preserve its categorical levels rather than treating their stored codes as continuous.",
        ),
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
        role=VariableRole.VITAL,
        kind=VariableKind.CONTINUOUS,
        unit="bpm",
        valid_range=(20.0, 250.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "map": ConceptHint(
        role=VariableRole.VITAL,
        kind=VariableKind.CONTINUOUS,
        unit="mmHg",
        valid_range=(20.0, 200.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "sbp": ConceptHint(
        role=VariableRole.VITAL,
        kind=VariableKind.CONTINUOUS,
        unit="mmHg",
        valid_range=(40.0, 260.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "dbp": ConceptHint(
        role=VariableRole.VITAL,
        kind=VariableKind.CONTINUOUS,
        unit="mmHg",
        valid_range=(20.0, 180.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    "spo2": ConceptHint(
        role=VariableRole.VITAL,
        kind=VariableKind.CONTINUOUS,
        unit="%",
        valid_range=(40.0, 100.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "temp": ConceptHint(
        role=VariableRole.VITAL,
        kind=VariableKind.CONTINUOUS,
        unit="C",
        valid_range=(28.0, 43.0),
        aggregation_default=AggregationRule.MEAN_MEDIAN,
    ),
    # --- common labs
    "lact": ConceptHint(
        role=VariableRole.LAB,
        kind=VariableKind.CONTINUOUS,
        unit="mmol/L",
        valid_range=(0.0, 30.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
        pitfalls=(
            "Lactate is right-skewed and measurement is often clinically triggered; report median/IQR or clinically meaningful bins, and audit unmeasured lactate separately.",
            "Do not interpret lactate missingness as normal lactate without an explicit measurement-status indicator.",
        ),
    ),
    "creat": ConceptHint(
        role=VariableRole.LAB,
        kind=VariableKind.CONTINUOUS,
        unit="mg/dL",
        valid_range=(0.1, 15.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
        pitfalls=(
            "Creatinine is right-skewed; report median (IQR), not mean (SD), unless log-transformed.",
        ),
    ),
    "bili": ConceptHint(
        role=VariableRole.LAB,
        kind=VariableKind.CONTINUOUS,
        unit="mg/dL",
        valid_range=(0.0, 50.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "plt": ConceptHint(
        role=VariableRole.LAB,
        kind=VariableKind.CONTINUOUS,
        unit="K/uL",
        valid_range=(1.0, 1500.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "pafi": ConceptHint(
        role=VariableRole.LAB,
        kind=VariableKind.CONTINUOUS,
        unit="ratio",
        valid_range=(20.0, 700.0),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
        pitfalls=(
            "PaO2/FiO2 ratios are only meaningful when FiO2 is reliably recorded; "
            "missing FiO2 should not be silently imputed to 0.21.",
        ),
    ),
    # --- ordinal scores
    "gcs": ConceptHint(
        role=VariableRole.ORDINAL_SCORE,
        kind=VariableKind.ORDINAL,
        valid_range=(3.0, 15.0),
        is_ordinal=True,
        ordinal_levels=tuple(range(3, 16)),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "GCS is ordinal; do not take its mean. Report worst (min) or representative (last/first) GCS.",
        ),
    ),
    "kdigo_stage": ConceptHint(
        role=VariableRole.ORDINAL_SCORE,
        kind=VariableKind.ORDINAL,
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
        role=VariableRole.ORDINAL_SCORE,
        kind=VariableKind.ORDINAL,
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
        role=VariableRole.COMPOSITE_SCORE,
        kind=VariableKind.ORDINAL,
        valid_range=(0.0, 24.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Total SOFA is a sum of six 0–4 components; treat as ordinal/integer-count, not continuous.",
            "A SOFA total of 0 across many patients may indicate that one or more component inputs were missing rather than that organ dysfunction was truly absent. Always cross-check component-level missingness before drawing clinical conclusions.",
        ),
    ),
    "sofa2": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE,
        kind=VariableKind.ORDINAL,
        valid_range=(0.0, 24.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "SOFA-2 follows the same 0–4 component structure as SOFA; treat as ordinal.",
            "SOFA-2 totals may reflect component-level missingness; cross-check component completeness before drawing clinical conclusions.",
        ),
    ),
    "sirs": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE,
        kind=VariableKind.ORDINAL,
        valid_range=(0.0, 4.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "qsofa": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE,
        kind=VariableKind.ORDINAL,
        valid_range=(0.0, 3.0),
        is_ordinal=True,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    # --- outcomes
    "death": ConceptHint(
        role=VariableRole.OUTCOME,
        kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.FIRST_VALUE,
        pitfalls=(
            "ICU mortality vs hospital mortality vs 28-day mortality are NOT interchangeable; record which one is in use.",
        ),
    ),
    "death_icu": ConceptHint(
        role=VariableRole.OUTCOME,
        kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "death_hosp": ConceptHint(
        role=VariableRole.OUTCOME,
        kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "los_icu": ConceptHint(
        role=VariableRole.OUTCOME,
        kind=VariableKind.CONTINUOUS,
        unit="days",
        valid_range=(0.0, 365.0),
        aggregation_default=AggregationRule.FIRST_VALUE,
        pitfalls=(
            "ICU LoS is right-skewed and competing-risks-affected; report median (IQR), and consider survival framing rather than linear regression.",
        ),
    ),
    "los_hosp": ConceptHint(
        role=VariableRole.OUTCOME,
        kind=VariableKind.CONTINUOUS,
        unit="days",
        valid_range=(0.0, 365.0),
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    # --- interventions
    "vaso": ConceptHint(
        role=VariableRole.INTERVENTION,
        kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Vasopressor exposure is conventionally captured as 'any vasopressor in window' (binary) "
            "or 'cumulative norepinephrine-equivalent' (continuous); pick one and document it.",
            "Vasopressor exposure is confounded by indication; association models should avoid causal treatment-effect language.",
        ),
    ),
    "norepi_equiv": ConceptHint(
        role=VariableRole.INTERVENTION,
        kind=VariableKind.CONTINUOUS,
        unit="mcg/kg/min",
        valid_range=(0.0, 10.0),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Norepinephrine-equivalent dose is an intervention intensity measure; summarize peak or cumulative exposure within a prespecified window.",
            "Dose comparisons require unit harmonisation across source databases.",
        ),
    ),
    "circ_event": ConceptHint(
        role=VariableRole.COMPOSITE_SCORE,
        kind=VariableKind.ORDINAL,
        valid_range=(0.0, 3.0),
        is_ordinal=True,
        ordinal_levels=(0, 1, 2, 3),
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Circulatory failure event level is ordinal; use maximum/worst event within the window, not mean event level.",
        ),
    ),
    "circ_failure": ConceptHint(
        role=VariableRole.INTERVENTION,
        kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
        pitfalls=(
            "Circulatory failure is a derived EasyICU concept based on lactate, MAP and vasoactive support; document the underlying rule and time window.",
        ),
    ),
    "vent_ind": ConceptHint(
        role=VariableRole.INTERVENTION,
        kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "rrt": ConceptHint(
        role=VariableRole.INTERVENTION,
        kind=VariableKind.BINARY,
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    # --- ids / time
    "patient_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "icustay_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "hadm_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "stay_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
    "subject_id": ConceptHint(role=VariableRole.ID, kind=VariableKind.IDENTIFIER),
}


_COMPANION_WINDOW_SUFFIX_RE = re.compile(
    r"_(?:first_)?\d+(?:h|d)$",
    re.IGNORECASE,
)


def companion_count_column_for_measured(name: str) -> Optional[str]:
    """Return the structural count companion for a measured-status column.

    The original spelling and any supported trailing time-window suffix are
    preserved.  For example, ``signal_measured_6h`` pairs with
    ``signal_n_6h`` and ``signal_measured_first_24h`` pairs with
    ``signal_n_first_24h``.  Non-status names return ``None``.
    """

    raw_name = str(name or "").strip()
    if not raw_name:
        return None
    window_match = _COMPANION_WINDOW_SUFFIX_RE.search(raw_name)
    window_suffix = window_match.group(0) if window_match is not None else ""
    structural_name = (
        raw_name[: window_match.start()] if window_match is not None else raw_name
    )
    measured_suffix = "_measured"
    if not structural_name.lower().endswith(measured_suffix):
        return None
    stem = structural_name[: -len(measured_suffix)]
    if not stem:
        return None
    return f"{stem}_n{window_suffix}"


def _companion_audit_hint(name: str) -> Optional[ConceptHint]:
    """Classify structural count/status companions before concept prefixes.

    Wide ICU cohorts commonly pair a value column with an observation count or
    measurement-status field.  Those companions describe provenance, not the
    underlying physiological quantity.  Letting ``<concept>_n`` inherit the
    base concept's unit/range, or ``<score>_measured`` inherit ordinal levels,
    gives the planner contradictory structured metadata.

    The rule is deliberately suffix-structural and concept-neutral.  Explicit
    entries in ``_CONCEPT_HINTS`` still win, so a future curated concept can
    override this generic fallback without weakening it.
    """

    key = str(name or "").strip().lower()
    structural_key = _COMPANION_WINDOW_SUFFIX_RE.sub("", key)
    if structural_key.endswith("_n"):
        return ConceptHint(
            role=VariableRole.META,
            kind=VariableKind.COUNT,
            aggregation_default=AggregationRule.SUM,
            pitfalls=(
                "Auxiliary observation-count/provenance field; use it to audit "
                "source availability, not as a physiological value or automatic "
                "adjustment covariate.",
            ),
        )

    structural_tokens = set(structural_key.split("_"))
    measurement_status = structural_key.endswith("_measured") or (
        structural_key.endswith("_flag")
        and any(
            marker in structural_tokens
            for marker in (
                "measured",
                "measurement",
                "source",
                "available",
                "availability",
                "observed",
                "observation",
            )
        )
    )
    if measurement_status:
        return ConceptHint(
            role=VariableRole.META,
            kind=VariableKind.BINARY,
            valid_range=(0.0, 1.0),
            aggregation_default=AggregationRule.MAX_LAST,
            pitfalls=(
                "Auxiliary measurement/source-status flag; use it for data-"
                "availability audits, not as the measured physiological value "
                "or an automatic adjustment covariate.",
            ),
        )
    return None


def _lookup_hint(name: str) -> Optional[ConceptHint]:
    """Find the longest matching prefix hint for ``name`` (case-insensitive)."""
    key = name.lower()
    # exact match first
    if key in _CONCEPT_HINTS:
        return _CONCEPT_HINTS[key]
    # Structural audit companions must not inherit the base concept's clinical
    # unit, physiological range, or ordinal scale through prefix matching.
    companion_hint = _companion_audit_hint(key)
    if companion_hint is not None:
        return companion_hint
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
                return ConceptHint(
                    role=VariableRole.OTHER, kind=VariableKind.ORDINAL, is_ordinal=True
                )
        return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.COUNT)
    if "float" in dtype_l or "double" in dtype_l:
        return ConceptHint(
            role=VariableRole.OTHER,
            kind=VariableKind.CONTINUOUS,
            aggregation_default=AggregationRule.MEAN_MEDIAN,
        )
    if "object" in dtype_l or "string" in dtype_l or "category" in dtype_l:
        return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.CATEGORICAL)
    return ConceptHint(role=VariableRole.OTHER, kind=VariableKind.UNKNOWN)


def aggregation_rule_for(
    role: VariableRole, kind: VariableKind
) -> List[AggregationRule]:
    """Return the *allowed* aggregation operations for a (role, kind) pair."""
    if kind == VariableKind.IDENTIFIER:
        return [AggregationRule.NONE, AggregationRule.FIRST_VALUE]
    if kind == VariableKind.TIMESTAMP:
        return [AggregationRule.NONE, AggregationRule.FIRST_VALUE]
    if kind == VariableKind.BINARY:
        return [
            AggregationRule.MAX_LAST,
            AggregationRule.FIRST_VALUE,
            AggregationRule.SUM,
        ]
    if kind == VariableKind.ORDINAL:
        return [
            AggregationRule.MAX_LAST,
            AggregationRule.FIRST_VALUE,
            AggregationRule.MEDIAN_ONLY,
        ]
    if kind == VariableKind.COUNT:
        return [
            AggregationRule.SUM,
            AggregationRule.MEAN_MEDIAN,
            AggregationRule.MAX_LAST,
        ]
    if kind == VariableKind.CONTINUOUS:
        if role == VariableRole.LAB:
            # labs are typically right-skewed → discourage mean
            return [
                AggregationRule.MEDIAN_ONLY,
                AggregationRule.MAX_LAST,
                AggregationRule.FIRST_VALUE,
            ]
        return [
            AggregationRule.MEAN_MEDIAN,
            AggregationRule.MEDIAN_ONLY,
            AggregationRule.MAX_LAST,
            AggregationRule.FIRST_VALUE,
        ]
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
        TimeWindow(
            name="first_24h",
            anchor="icu_admission",
            start_hours=0.0,
            end_hours=24.0,
            rationale="Standard window for admission severity scores (SOFA, APACHE).",
        ),
        TimeWindow(
            name="first_6h",
            anchor="icu_admission",
            start_hours=0.0,
            end_hours=6.0,
            rationale="Early-resuscitation window for sepsis / shock studies.",
        ),
        TimeWindow(
            name="full_stay",
            anchor="icu_admission",
            start_hours=0.0,
            end_hours=24.0 * 30,
            rationale="Whole ICU stay; analysis script must cap end_hours per patient.",
        ),
    ]


# ---------------------------------------------------------------------------
# Cross-cutting methodological principles (case-neutral, cross-database)
# ---------------------------------------------------------------------------
#
# These complement the per-concept hints above with the reasoning hazards
# that survive any data layer. They are the general half of the "common ICU
# data-analysis pitfalls" taxonomy (see ``docs/icu_pitfall_crosswalk.md``):
# the structural half (table location, derived reuse, id hierarchy, time
# alignment, unit harmonisation) is handled upstream by the EasyICU concept /
# converter / cohort layer, so it is recorded here only where a *residual*
# reasoning step remains. Everything below is deliberately database-agnostic;
# the per-database variation lives in ``cross_db_note`` as illustration.

GENERAL_ICU_ANALYSIS_PRINCIPLES: Tuple[MethodologicalPrinciple, ...] = (
    # --- objective errors (wrong under any design; safe to flag firmly) ------
    MethodologicalPrinciple(
        id="diagnosis_membership_not_timing",
        phase="cohort",
        kind="error",
        principle=(
            "Use diagnosis codes for cohort membership only, not as event "
            "timing; derive 'when' from a timestamped source."
        ),
        rationale=(
            "A billing/coding diagnosis says a condition was present during the "
            "admission, not when it began or whether it was ICU-acquired."
        ),
        cross_db_note=(
            "Timing availability varies — e.g. MIMIC diagnoses_icd has no "
            "timestamp, eICU diagnosis carries a diagnosisoffset, Amsterdam/HiRID "
            "do not use ICD; only trust a code's time if the source provides one."
        ),
    ),
    MethodologicalPrinciple(
        id="dedupe_to_patient_unit",
        phase="cohort",
        kind="error",
        principle=(
            "Do not treat multiple records or stays from the same patient as "
            "independent observations; account for within-patient correlation "
            "by the study's chosen design (one record per patient, OR "
            "cluster-robust / mixed-effects estimation — the choice is yours)."
        ),
        rationale=(
            "The independence assumption is violated by repeated measures; "
            "ignoring it inflates significance and evaluation metrics. The "
            "*remedy* is a legitimate design choice — the error is leaving the "
            "correlation unaccounted for."
        ),
        cross_db_note=(
            "The patient-level id differs: e.g. MIMIC subject_id>hadm_id>stay_id, "
            "eICU patienthealthsystemstayid>patientunitstayid, Amsterdam "
            "patientid>admissionid; resolve the patient level per database."
        ),
    ),
    MethodologicalPrinciple(
        id="align_time_to_t0",
        phase="features",
        kind="error",
        principle=(
            "Align every event time to a single stated per-stay anchor (t0; ICU "
            "admission by default) before cutting feature and outcome windows."
        ),
        rationale=(
            "Cross-patient comparison and windowing require a common relative "
            "time axis; raw timestamps are not comparable across patients."
        ),
        cross_db_note=(
            "Time representation differs — e.g. MIMIC stores per-patient "
            "date-shifted absolute timestamps, while eICU/HiRID/Amsterdam/SICdb "
            "use relative offsets from admission; normalise before windowing."
        ),
    ),
    MethodologicalPrinciple(
        id="no_outcome_window_leakage",
        phase="features",
        kind="error",
        principle=(
            "Features may use only data observable strictly before the "
            "prediction/outcome window; never include outcome-window or future "
            "values (or the outcome itself)."
        ),
        rationale=(
            "Outcome-window or future values make a model 'time-travel' — the "
            "AUC looks excellent and the model is clinically worthless."
        ),
        cross_db_note="Database-agnostic; the leak is temporal, not source-specific.",
    ),
    MethodologicalPrinciple(
        id="window_aggregation_respects_kind",
        phase="features",
        kind="error",
        principle=(
            "Never average an ordinal or bounded-integer clinical score; "
            "aggregate it within the window by a rank-preserving summary "
            "(worst/max, last, or the level distribution). (How to summarise a "
            "continuous trend — mean, last, slope — is an analyst choice.)"
        ),
        rationale=(
            "The mean of an ordinal score is not interpretable as a score; this "
            "is a category error regardless of study design."
        ),
        cross_db_note=(
            "Sampling resolution differs — high-frequency HiRID/Amsterdam vs "
            "sparser MIMIC/eICU — which changes which continuous aggregates are "
            "reliable."
        ),
    ),
    MethodologicalPrinciple(
        id="exposure_window_precedes_outcome_window",
        phase="label",
        kind="error",
        principle=(
            "State the exposure-ascertainment window and the outcome window "
            "separately, and do not let them overlap. When an exposure is "
            "ascertained over a period after t0, outcome events occurring "
            "inside that same period cannot be treated as ordinary follow-up: "
            "either start follow-up at the end of the ascertainment window "
            "(analysing only units still at risk then), model the exposure as "
            "time-varying, or state and justify the overlap explicitly. Any of "
            "those is a legitimate design; leaving the overlap unstated is not."
        ),
        rationale=(
            "A unit whose outcome occurs during ascertainment had less "
            "opportunity to accumulate the exposure-defining evidence, so the "
            "exposure and the outcome are partly measuring the same episode. "
            "This is distinct from feature leakage, which runs the other way, "
            "and it is not fixed by any amount of covariate adjustment. It "
            "typically biases the association away from the null, and the "
            "affected subgroup is exactly the sickest and earliest events. "
            "Restarting follow-up at the window's end removes the overlap but "
            "changes the population being described, so which estimate is "
            "primary is a scientific choice that has to be reported, not a "
            "detail."
        ),
        cross_db_note=(
            "Database-agnostic: the hazard is the relationship between two "
            "declared windows, not the source of either. What differs is how "
            "early events are recorded — some databases carry event times that "
            "precede the stated anchor, so check the sign of the interval "
            "rather than assuming follow-up starts at zero."
        ),
    ),
    MethodologicalPrinciple(
        id="label_built_in_outcome_window",
        phase="label",
        kind="error",
        principle=(
            "Compute the outcome label inside the outcome window (group by the "
            "analysis unit, then apply the threshold/duration/OR logic); the "
            "label is rarely a ready-made column."
        ),
        rationale=(
            "A mis-defined label invalidates every downstream result no matter "
            "how good the model is."
        ),
        cross_db_note=(
            "Outcome source and granularity differ by database; build the label "
            "from the timestamped source each database provides."
        ),
    ),
    MethodologicalPrinciple(
        id="winsorize_harmonise_then_scale",
        phase="features",
        kind="error",
        principle=(
            "Fit any preprocessing that learns from data (scaling, imputation, "
            "encoders) on the training split only, and harmonise units before "
            "combining sources. (Whether and how to truncate physiologic "
            "outliers is an analyst choice to document.)"
        ),
        rationale=(
            "Fitting preprocessing on all data leaks test information; mixing "
            "unharmonised units silently corrupts pooled values. Both are wrong "
            "under any design; outlier handling is a defensible choice."
        ),
        cross_db_note=(
            "The same concept may be recorded in different units across "
            "databases (e.g. creatinine mg/dL vs µmol/L, temperature C vs F); "
            "rely on EasyICU concept normalisation and never compare raw "
            "cross-database values/doses without harmonisation."
        ),
    ),
    MethodologicalPrinciple(
        id="split_by_patient",
        phase="modeling",
        kind="error",
        principle=(
            "Split train/test by the patient-level id, not by row, so one "
            "patient never appears on both sides."
        ),
        rationale=(
            "Row-level splits leak a patient's own correlated records into the "
            "test set and inflate performance — a second kind of leakage."
        ),
        cross_db_note=(
            "Use whichever id is the patient level in that database (see "
            "dedupe_to_patient_unit)."
        ),
    ),
    MethodologicalPrinciple(
        id="association_is_not_causation",
        phase="interpretation",
        kind="error",
        principle=(
            "Do not state an observational association as causal, and do not "
            "read feature importance (e.g. SHAP) as a mechanism; keep causal "
            "language out unless a causal design is actually used."
        ),
        rationale=(
            "Treatment exposures are confounded by indication; 'SHAP shows it' "
            "is not evidence a feature is clinically causal. Overstating this is "
            "an interpretation error, not a stylistic preference."
        ),
        cross_db_note=(
            "Confounding structure differs with each database's case-mix and "
            "practice patterns."
        ),
    ),
    MethodologicalPrinciple(
        id="no_overadjustment_for_exposure_constituents",
        phase="modeling",
        kind="error",
        principle=(
            "Do not adjust for a covariate that is a definitional component or "
            "an upstream derivation input of the exposure itself; adjust only "
            "for confounders that are neither constituents nor downstream "
            "mediators of the exposure."
        ),
        rationale=(
            "Conditioning on a variable that helps define the exposure removes "
            "the very signal under study (overadjustment / adjusting for a "
            "mediator) and can flip or null the estimate — an objective error, "
            "not a modelling preference. The exposure's own derivation closure "
            "(e.g. a composite score or rule that enters the exposure's "
            "definition) lists the variables to keep out of the adjustment set."
        ),
        cross_db_note=(
            "The exposure's derivation closure comes from the concept "
            "dictionary, not from any one database's columns, so it is "
            "database-agnostic."
        ),
    ),
    # --- cautions (defensible choices; prompt to document, never override) ---
    MethodologicalPrinciple(
        id="robust_fit_under_collinearity_or_separation",
        phase="modeling",
        kind="caution",
        principle=(
            "Before fitting a regression, drop constant and perfectly collinear "
            "predictors (a rank-deficient design makes the fit singular), and if "
            "the model is still singular or fails to converge under "
            "(quasi-)separation, use a penalised fit (e.g. Firth or ridge) rather "
            "than reporting a failed or non-converged estimate."
        ),
        rationale=(
            "A constant column (e.g. a missing-indicator that becomes constant "
            "after imputation) or two collinear covariates make the design "
            "matrix non-invertible — the fit dies with 'Singular matrix' and no "
            "effect size is produced. Removing redundant columns and using a "
            "penalised estimator under separation are standard, defensible "
            "remedies; which remedy to use is the analyst's choice, so this is a "
            "caution, not a mandated estimator."
        ),
        cross_db_note=(
            "Rank-deficiency and separation arise from each database's case-mix "
            "and missingness, so the same robustness step applies everywhere."
        ),
    ),
    MethodologicalPrinciple(
        id="describe_cohort_before_modeling",
        phase="cohort",
        kind="caution",
        principle=(
            "Report the cohort description (counts, outcome prevalence, "
            "missingness) alongside any model result so a metric is "
            "interpretable. The analysis order itself is the analyst's choice."
        ),
        rationale=(
            "A discrimination metric is hard to interpret without the "
            "denominator and outcome prevalence; reporting them is good "
            "practice, not a mandated workflow."
        ),
        cross_db_note=(
            "Outcome availability differs by database — e.g. hospital-discharge "
            "mortality in MIMIC/eICU vs limited long-horizon outcomes in HiRID; "
            "report the denominator actually available."
        ),
    ),
    MethodologicalPrinciple(
        id="incident_not_prevalent",
        phase="cohort",
        kind="caution",
        principle=(
            "When the question concerns incident (new-onset) events, exclude "
            "prevalent cases (event already present before follow-up start); if "
            "prevalence itself is the target, say so. State which, and note this "
            "is a cohort-definition exclusion, distinct from leakage."
        ),
        rationale=(
            "Whether to exclude prevalent cases depends on the question; the "
            "hazard is leaving it implicit or conflating it with leakage, not "
            "the exclusion itself."
        ),
        cross_db_note=(
            "Pre-ICU history depth varies — e.g. MIMIC carries prior admissions "
            "while HiRID/Amsterdam are ICU-centric; judge prevalence with the "
            "history each database actually exposes."
        ),
    ),
    MethodologicalPrinciple(
        id="missingness_is_information",
        phase="features",
        kind="caution",
        principle=(
            "Do not assume ICU missingness is at random; consider whether 'not "
            "measured' is itself informative (MNAR) when choosing a missing-data "
            "strategy. The strategy (indicator, imputation method, "
            "complete-case) is the analyst's choice to state and justify."
        ),
        rationale=(
            "'Not measured' often encodes clinical judgement (e.g. lactate not "
            "drawn because perfusion looked fine); the hazard is assuming MCAR "
            "without justification, not any particular handling."
        ),
        cross_db_note=(
            "Missingness mechanisms differ — routine high-frequency capture vs "
            "clinically-triggered labs — so the same column can be MCAR in one "
            "database and MNAR in another."
        ),
    ),
    MethodologicalPrinciple(
        id="measurement_frequency_is_informative",
        phase="features",
        kind="caution",
        principle=(
            "Consider that how often a variable is measured can itself be "
            "informative: in the ICU, sampling frequency correlates with acuity, "
            "so measurement counts or carry-forward values may encode severity "
            "rather than physiology. Decide deliberately how to handle it."
        ),
        rationale=(
            "Informative sampling can bias estimates if treated as ignorable; "
            "how to model it (e.g. include a measurement-count feature, or not) "
            "is a legitimate analyst choice."
        ),
        cross_db_note=(
            "Sampling regimes differ — high-frequency monitored signals vs "
            "intermittently ordered labs — across and within databases."
        ),
    ),
    MethodologicalPrinciple(
        id="metrics_match_task_and_balance",
        phase="interpretation",
        kind="caution",
        principle=(
            "Report evaluation metrics appropriate to the task and class "
            "balance, and more than one number; accuracy alone is misleading "
            "under class imbalance. Which metrics to emphasise is the analyst's "
            "choice."
        ),
        rationale=(
            "ICU outcomes are usually imbalanced, so a single accuracy figure "
            "can hide poor minority-class performance; the choice of which "
            "balanced metrics to show is not ours to dictate."
        ),
        cross_db_note=(
            "Outcome prevalence (hence the relevant operating point) varies by "
            "database case-mix."
        ),
    ),
    MethodologicalPrinciple(
        id="clusters_need_external_validation",
        phase="clustering",
        kind="caution",
        principle=(
            "Unsupervised clusters have no ground truth; report cluster "
            "stability and, where a relevant outcome exists, relate clusters to "
            "it before interpreting them — and do not present clusters as "
            "established clinical entities without such support."
        ),
        rationale=(
            "Unsupervised structure is easy to find and easy to over-read; "
            "external-outcome separation is what makes a subphenotype credible. "
            "The validation method is a choice; over-claiming is the hazard."
        ),
        cross_db_note=(
            "Cross-database replication is the strongest validation; harmonise "
            "the clustering feature set across databases first."
        ),
    ),
    MethodologicalPrinciple(
        id="state_outcome_definition",
        phase="interpretation",
        kind="caution",
        principle=(
            "State exactly which outcome definition is used (ICU vs hospital vs "
            "28-day mortality, etc.); they are not interchangeable."
        ),
        rationale=(
            "Conflating outcome definitions silently changes the estimand and "
            "breaks cross-study comparison; which outcome to study is the "
            "analyst's choice, stating it is the requirement."
        ),
        cross_db_note=(
            "Which outcome is available differs — e.g. hospital-discharge status "
            "in eICU, limited long-horizon mortality in HiRID."
        ),
    ),
    MethodologicalPrinciple(
        id="control_for_multiplicity",
        phase="interpretation",
        kind="caution",
        principle=(
            "When many hypotheses or associations are tested, pre-specify the "
            "primary analysis or control for multiplicity; an unadjusted scan of "
            "many comparisons inflates false-positive findings. The adjustment "
            "method is the analyst's choice."
        ),
        rationale=(
            "Testing many endpoints without a plan turns noise into 'findings'; "
            "the hazard is the unacknowledged multiplicity, not any specific "
            "correction."
        ),
        cross_db_note=(
            "Database-agnostic; applies wherever multiple endpoints or subgroups "
            "are screened."
        ),
    ),
    MethodologicalPrinciple(
        id="consider_competing_risks",
        phase="modeling",
        kind="caution",
        principle=(
            "For time-to-event outcomes, consider competing events (e.g. in-ICU "
            "death competing with ICU discharge); a naive survival or binary "
            "model can mislead when a competing event precludes the event of "
            "interest. Whether/how to model competing risks is the analyst's "
            "choice."
        ),
        rationale=(
            "Censoring a competing event as if it were independent biases "
            "cumulative-incidence estimates; the hazard is ignoring the "
            "competing structure, not the specific estimator."
        ),
        cross_db_note=(
            "Discharge/transfer practices that create competing events differ by "
            "unit and database."
        ),
    ),
    MethodologicalPrinciple(
        id="report_cohort_attrition",
        phase="cohort",
        kind="caution",
        principle=(
            "Report inclusion/exclusion attrition (how many records drop at each "
            "step) so cohort construction is auditable and selection effects are "
            "visible."
        ),
        rationale=(
            "A transparent attrition flow lets a reader judge selection bias; "
            "the specific filters are the study's choice, their visibility is "
            "the requirement."
        ),
        cross_db_note=(
            "Database-agnostic; the starting population and available filters "
            "differ, so report the flow for each database used."
        ),
    ),
    MethodologicalPrinciple(
        id="harmonise_before_pooling",
        phase="cohort",
        kind="caution",
        principle=(
            "When combining databases, map concepts to the same definitions and "
            "units and assess between-database heterogeneity before pooling; do "
            "not naively concatenate raw cross-database values. Whether to pool "
            "or meta-analyse is the analyst's choice."
        ),
        rationale=(
            "Cross-database case-mix and recording differences can dominate a "
            "naive pooled estimate; the hazard is unexamined heterogeneity, not "
            "the decision to combine."
        ),
        cross_db_note=(
            "EasyICU's concept layer provides the shared definitions; still "
            "check heterogeneity (case-mix, era, unit-of-care) across the "
            "specific databases combined."
        ),
    ),
)


def principles_for_phase(phase: str) -> List[MethodologicalPrinciple]:
    """Return the cross-cutting principles guarding a given analysis ``phase``.

    ``phase`` is one of ``cohort`` / ``features`` / ``label`` / ``modeling`` /
    ``clustering`` / ``interpretation``. Unknown phases return an empty list.
    """
    key = (phase or "").strip().lower()
    return [p for p in GENERAL_ICU_ANALYSIS_PRINCIPLES if p.phase == key]


# A frozen dictionary the agents and validators read. Wrapped in a
# tiny class so the prompt has a single token to refer to.
@dataclass(frozen=True)
class _ICURules:
    aggregation_rule_for: Callable = field(default=aggregation_rule_for)
    classify_variable: Callable = field(default=classify_variable)
    default_time_windows: Callable = field(default=default_time_windows)
    principles_for_phase: Callable = field(default=principles_for_phase)
    general_principles: Tuple[MethodologicalPrinciple, ...] = field(
        default=GENERAL_ICU_ANALYSIS_PRINCIPLES
    )


ICU_RULES = _ICURules()


# ---------------------------------------------------------------------------
# Overadjustment detection (deterministic, case-neutral)
# ---------------------------------------------------------------------------

# Constituent resolution is dictionary-first, not table-first. The authoritative
# source of "what defines the exposure" is the EasyICU concept dictionary's
# ``depends_on`` derivation closure (see ``composite_constituents`` below), so
# ANY declared composite/derived concept — sofa, sofa2, the sofa_* sub-scores,
# and concepts added later — is covered with no edit here. The table below is a
# curated *fallback* used in two situations only:
#   1. the concept dictionary cannot be loaded (e.g. a data-isolated sandbox);
#   2. bridging ricu-style abbreviations in the dictionary (``crea``, ``pafi``)
#      to the clinical spellings a fitted model's covariate columns tend to use
#      (``creatinine``), so substring matching is not brittle to either spelling.
# These are standard ICU score compositions — general, illustrative clinical
# knowledge, NOT benchmark-specific entries. Tokens are matched case-insensitively
# as substrings of a variable name.
COMPOSITE_EXPOSURE_CONSTITUENTS: Dict[str, Tuple[str, ...]] = {
    "sofa": (
        "pao2",
        "fio2",
        "spo2",
        "platelet",
        "bilirubin",
        "gcs",
        "map",
        "vasopressor",
        "norepi",
        "dopamine",
        "epinephrine",
        "creatinine",
        "urine",
    ),
    "sepsis3": (
        "sofa",
        "suspected_infection",
        "susp_inf",
        "pao2",
        "platelet",
        "bilirubin",
        "gcs",
        "map",
        "creatinine",
        "urine",
    ),
    "qsofa": ("gcs", "respiratory_rate", "resp_rate", "sbp", "systolic"),
    "kdigo": ("creatinine", "urine_output", "urine"),
}


# Composites whose composition lives in a Python callback rather than the
# dictionary's ``depends_on`` field, so their declared closure is empty. Each
# value names the immediate concept(s) the exposure is *built from*; those seeds
# are then expanded through the dictionary closure like any other concept. This
# is the minimal structural gap-filler — not a per-benchmark table.
_CALLBACK_COMPOSITE_SEED: Dict[str, Tuple[str, ...]] = {
    "sep3": ("sofa", "susp_inf"),  # sepsis-3 = suspected infection + SOFA rise
    "sepsis3": ("sofa", "susp_inf"),
    "qsofa": ("gcs", "resp_rate", "sbp"),  # qSOFA bedside components
    "kdigo": ("crea", "urine"),  # KDIGO AKI = creatinine + urine-output
    "kdigoaki": ("crea", "urine"),
}

# Bridge dictionary abbreviations to clinical spellings a model's covariates use,
# per constituent (so a sub-score exposure does not pull in unrelated names).
_CONCEPT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "crea": ("creatinine",),
    "pafi": ("pao2", "fio2", "pao2fio2", "pfratio"),
    "safi": ("spo2", "fio2", "spo2fio2"),
    "map": ("meanarterialpressure",),
    "bili": ("bilirubin",),
    "plt": ("platelet", "platelets"),
    "norepi60": ("norepinephrine", "norepi", "noradrenaline"),
    "dopa60": ("dopamine",),
    "epi60": ("epinephrine", "adrenaline"),
    "dobu60": ("dobutamine",),
    "vaso_ind": ("vasopressor", "pressor"),
    "vent_ind": ("ventilation", "mechanicalventilation"),
    "urine24": ("urine", "urineoutput"),
    "urine": ("urineoutput",),
    "susp_inf": ("suspectedinfection", "suspinf"),
    "resp_rate": ("respiratoryrate", "resprate"),
    "sbp": ("systolic", "systolicbp"),
    "gcs": ("glasgow",),
}

# Pure statistic / derivation suffixes appended to a concept to make a model
# variable (``sofa_max``, ``sepsis3_corrected``). Used to recognise the exposure
# variable itself — so it is never mistaken for one of its own constituents.
_STAT_SUFFIXES: frozenset[str] = frozenset(
    {
        "max",
        "min",
        "mean",
        "median",
        "avg",
        "last",
        "first",
        "sum",
        "total",
        "score",
        "corrected",
        "adj",
        "adjusted",
        "baseline",
        "init",
        "initial",
        "admission",
        "worst",
        "peak",
        "nadir",
        "delta",
        "change",
        "value",
        "val",
        "cat",
        "bin",
        "group",
        "level",
        "norm",
        "std",
        "zscore",
        "flag",
    }
)

_DICT_CACHE: Dict[str, object] = {}


def _normalise_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Aliases keyed by normalised concept token, so closure tokens (``vasoind``,
# ``resprate``, ``suspinf``, ``urine24``) actually resolve their clinical
# spelling bridges regardless of the dictionary's punctuation.
_CONCEPT_ALIASES_NORM: Dict[str, Tuple[str, ...]] = {
    _normalise_token(k): v for k, v in _CONCEPT_ALIASES.items()
}


def _strip_stat_suffix(exposure: str) -> str:
    """Normalised exposure with trailing pure stat/derivation suffixes removed."""
    parts = [p for p in re.split(r"[^a-z0-9]+", str(exposure).lower()) if p]
    while len(parts) > 1 and parts[-1] in _STAT_SUFFIXES:
        parts = parts[:-1]
    return "".join(parts)


def _concept_dictionary() -> Optional[object]:
    """The EasyICU concept dictionary, loaded once; ``None`` if unavailable.

    A missing dictionary (e.g. a data-isolated sandbox) is not an error — the
    detector degrades to the curated fallback table rather than failing.
    """
    if "d" in _DICT_CACHE:
        return _DICT_CACHE["d"]  # type: ignore[return-value]
    dictionary: Optional[object] = None
    try:  # local import to avoid import-time cost / cycles
        from ..concept.loader import load_dictionary

        dictionary = load_dictionary()
    except Exception:
        dictionary = None
    _DICT_CACHE["d"] = dictionary
    return dictionary


def _depends_on_closure(name: str, dictionary: object, seen: set) -> None:
    if name in seen:
        return
    try:
        definition = dictionary[name]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError):
        return
    seen.add(name)
    for dep in getattr(definition, "depends_on", ()) or ():
        _depends_on_closure(str(dep), dictionary, seen)


def _resolve_concept(base: str, dictionary: object) -> Optional[str]:
    """Match a normalised exposure to a dictionary concept name, or ``None``."""
    try:
        names = list(dictionary.keys())  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return None
    best: Optional[str] = None
    best_len = 0
    for cname in names:
        ctok = _normalise_token(cname)
        if not ctok:
            continue
        # exact, or the concept name is the exposure stripped of a stat suffix
        if ctok == base and len(ctok) > best_len:
            best, best_len = cname, len(ctok)
    return best


def _fallback_constituents(exposure: str) -> Tuple[str, ...]:
    """Curated table match (longest composite name wins), or ``()``."""
    key = _normalise_token(exposure)
    if not key:
        return ()
    best_parts: Tuple[str, ...] = ()
    best_len = -1
    for comp, parts in COMPOSITE_EXPOSURE_CONSTITUENTS.items():
        comp_tok = _normalise_token(comp)
        if comp_tok and comp_tok in key and len(comp_tok) > best_len:
            best_parts, best_len = parts, len(comp_tok)
    return best_parts


def composite_constituents(
    exposure: str, dictionary: Optional[object] = None
) -> Tuple[str, ...]:
    """Constituent concept tokens of a composite/derived exposure, or ``()``.

    Dictionary-first and general: the exposure's ``depends_on`` derivation
    closure from the concept dictionary is the authoritative source, so any
    declared composite is covered without hard-coding. Callback-defined
    composites (sep3 / qsofa / kdigo, whose declared closure is empty) are
    seeded from ``_CALLBACK_COMPOSITE_SEED`` and then expanded the same way.
    The curated ``COMPOSITE_EXPOSURE_CONSTITUENTS`` table is unioned in as a
    fallback and to bridge abbreviation/clinical spellings.

    ``()`` means the exposure is not a recognised composite, so no
    overadjustment rule applies — the check stays silent rather than guessing.
    """
    base = _strip_stat_suffix(exposure)
    full = _normalise_token(exposure)
    if not full:
        return ()

    tokens: set[str] = set()

    # (1) Seeds: the exposure's own concept + any callback-composite inputs.
    seeds: set[str] = set()
    for comp, parts in _CALLBACK_COMPOSITE_SEED.items():
        if comp in base or comp in full:
            seeds.update(parts)

    dic = dictionary if dictionary is not None else _concept_dictionary()
    resolved: Optional[str] = None
    if dic is not None:
        resolved = _resolve_concept(base, dic)
        if resolved:
            seeds.add(resolved)
        # (2) Expand every seed through the dictionary derivation closure.
        closure: set[str] = set()
        for seed in seeds:
            _depends_on_closure(seed, dic, closure)
        for cname in closure | seeds:
            ctok = _normalise_token(cname)
            if not ctok or ctok == base or ctok == full:
                continue  # the exposure itself is not a constituent
            tokens.add(ctok)
            tokens.update(_CONCEPT_ALIASES_NORM.get(ctok, ()))

    # (3) Curated fallback. The dictionary's ``depends_on`` graph stops at each
    # organ sub-score (sofa_liver / sofa_coag carry their leaf lab in a callback
    # ``source``, not ``depends_on``), so a TOP-LEVEL composite's closure under
    # -covers its leaves (bilirubin / platelet / gcs). Union the table in for a
    # top-level composite (the resolved concept is itself a table key, or the
    # dictionary could not resolve the exposure at all). A resolved *sub-score*
    # (sofa_renal) is trusted as-is, so its narrow closure is never broadened by
    # the coarse substring table.
    fallback_keys = {_normalise_token(k) for k in COMPOSITE_EXPOSURE_CONSTITUENTS}
    if resolved is None or _normalise_token(resolved) in fallback_keys:
        for part in _fallback_constituents(exposure):
            ptok = _normalise_token(part)
            if ptok and ptok != base and ptok != full:
                tokens.add(ptok)
                tokens.update(_CONCEPT_ALIASES_NORM.get(ptok, ()))

    return tuple(sorted(tokens))


def _is_exposure_self(cov_tok: str, exp_tok: str) -> bool:
    """True when a covariate token is the exposure variable itself.

    Recognises the exposure spelled with a pure stat/derivation suffix
    (``sofa_max``, ``sepsis3_corrected``) but NOT a distinct sub-score that
    merely shares a prefix (``sofa_renal`` is a constituent, not the exposure).
    """
    if not exp_tok or not cov_tok:
        return False
    if cov_tok == exp_tok:
        return True
    longer, shorter = (
        (cov_tok, exp_tok) if len(cov_tok) >= len(exp_tok) else (exp_tok, cov_tok)
    )
    if longer.startswith(shorter):
        remainder = longer[len(shorter) :]
        return remainder in _STAT_SUFFIXES
    return False


def _covariate_matches_constituent(covariate: str, constituent: str) -> bool:
    """Match a constituent by identity, never by an incidental substring."""

    constituent_token = _normalise_token(constituent)
    if not constituent_token:
        return False
    raw_parts = [
        part for part in re.split(r"[^a-z0-9]+", str(covariate).lower()) if part
    ]
    covariate_token = "".join(raw_parts)
    if constituent_token == covariate_token or constituent_token in raw_parts:
        return True
    if covariate_token.startswith(constituent_token):
        remainder = covariate_token[len(constituent_token) :]
        if remainder in _STAT_SUFFIXES:
            return True
    return False


def detect_overadjustment(
    exposure: str,
    adjustment_covariates: Sequence[str],
    dictionary: Optional[object] = None,
) -> List[str]:
    """Return the adjustment covariates that constitute / derive ``exposure``.

    Conditioning on a constituent of the exposure is overadjustment. Conservative
    by design: fires only when ``exposure`` is a recognised composite/derived
    score (``composite_constituents`` non-empty) and a covariate names one of its
    constituent tokens or an explicit alias. Incidental substrings do not carry
    scientific identity. The exposure variable itself is never flagged. An
    empty list means no overadjustment detected.
    """
    part_toks = [t for t in composite_constituents(exposure, dictionary) if t]
    if not part_toks:
        return []
    exp_tok = _strip_stat_suffix(exposure)
    offenders: List[str] = []
    for cov in adjustment_covariates:
        cov_tok = _normalise_token(cov)
        if not cov_tok or _is_exposure_self(cov_tok, exp_tok):
            continue  # the exposure itself is not an over-adjustment
        if any(_covariate_matches_constituent(str(cov), pt) for pt in part_toks):
            offenders.append(str(cov))
    return offenders


def is_derived_exposure(exposure: str, dictionary: Optional[object] = None) -> bool:
    """True when the exposure resolves to a *computed* concept.

    A computed concept has a callback or a non-empty ``depends_on`` — its value
    is a function of other concepts (a composite/derived score), which makes it
    overadjustment-prone as an exposure. Callback-defined composites whose name
    is not a clean dictionary key (sep3 / sepsis3 / qsofa / kdigo) are recognised
    too. Returns ``False`` for a raw measurement, a demographic, or when the
    dictionary is unavailable (no false positives without evidence).
    """
    base = _strip_stat_suffix(exposure)
    full = _normalise_token(exposure)
    if not full:
        return False
    if any(comp in base or comp in full for comp in _CALLBACK_COMPOSITE_SEED):
        return True
    dic = dictionary if dictionary is not None else _concept_dictionary()
    if dic is None:
        return False
    resolved = _resolve_concept(base, dic)
    if not resolved:
        return False
    try:
        definition = dic[resolved]  # type: ignore[index]
    except (KeyError, TypeError):
        return False
    has_callback = bool(getattr(definition, "callback", None))
    has_depends = bool(getattr(definition, "depends_on", ()) or ())
    return has_callback or has_depends


def overadjustment_caution(
    exposure: str,
    adjustment_covariates: Sequence[str],
    dictionary: Optional[object] = None,
) -> Optional[str]:
    """A *caution* (not an error) when overadjustment cannot be checked.

    Fires when the exposure is structurally a derived/composite concept (so an
    adjustment covariate *could* be one of its inputs) but its constituents
    could NOT be resolved from the dictionary — e.g. a callback-defined score
    (mews / news / sirs / anion_gap / pafi …) whose ``depends_on`` is empty. The
    deterministic ``detect_overadjustment`` check is silent there, so without
    this the risk would pass unflagged. Returns a one-line caution prompting
    manual verification, or ``None``.

    Distinct from an error: it never gates or re-fits, it asks the analyst to
    confirm the adjustment set. Returns ``None`` when the exposure is not
    derived, when its constituents *were* resolvable (``detect_overadjustment``
    already covers it), or when there are no adjustment covariates.
    """
    has_covariates = any(_normalise_token(c) for c in adjustment_covariates)
    if not has_covariates:
        return None
    if not is_derived_exposure(exposure, dictionary):
        return None
    if composite_constituents(exposure, dictionary):
        return None  # constituents resolvable -> the deterministic check applies
    return (
        f"exposure '{exposure}' is a derived/composite concept, but its "
        "constituent inputs could not be resolved from the concept dictionary, "
        "so overadjustment could not be checked automatically; verify the "
        "adjustment set excludes any variable that feeds its computation"
    )


# ---------------------------------------------------------------------------
# Per-concept methodological profile (advisory layer)
# ---------------------------------------------------------------------------
#
# The deterministic auditors above are the *teeth* (they flag/gate objective
# errors after the fact). This is the *advisory* complement: a methodological
# role profile derived from each concept's STRUCTURE — its dictionary category,
# whether it is computed (callback / depends_on), and whether it is ordinal — so
# the agent reasons about a variable's hazards up front when it assigns roles.
# Derived from structure, not a hand-written 198-row table: adding a concept to
# the dictionary gives it a profile for free. Mirrors the impartiality contract:
# the objective-leakage role (outcome) is stated firmly; the defensible-choice
# role (treatment confounder-vs-mediator) is framed as a prompt to decide.

_TREATMENT_CATEGORIES = frozenset({"medications", "ventilator"})
_MEASUREMENT_CATEGORIES = frozenset(
    {
        "chemistry",
        "hematology",
        "vitals",
        "blood gas",
        "respiratory",
        "renal",
        "output",
        "microbiology",
        "cardiovascular",
        "neurology",
        "neurological",
    }
)

# Full hazard sentence per role (for the profile / deeper injection).
_ROLE_HAZARD: Dict[str, str] = {
    "outcome": (
        "outcome concept — using it as a predictor/covariate for this or a "
        "derived outcome is leakage"
    ),
    "treatment": (
        "intervention — on the causal pathway; as a covariate decide whether it "
        "is a confounder or a mediator/collider before adjusting"
    ),
    "derived_composite": (
        "derived/composite score — if the exposure, keep its component inputs "
        "out of the adjustment set (overadjustment); as a covariate watch "
        "collinearity with its inputs"
    ),
    "ordinal_score": (
        "ordinal score — aggregate by worst/max within the window; do not "
        "average or model as continuous without justification"
    ),
    "measurement": (
        "time-varying measurement — fix the window and aggregation; its "
        "missingness may be informative (not missing-at-random)"
    ),
}

# Compact tag per role for an inline catalog/prompt line. Plain measurements and
# demographics get no tag (the default safe case) to keep the catalog readable.
_ROLE_TAG: Dict[str, str] = {
    "outcome": "outcome→predictor=leakage",
    "treatment": "treatment: confounder vs mediator?",
    "derived_composite": "derived: keep inputs out of adjustment if exposure",
    "ordinal_score": "ordinal: aggregate by worst, don't average",
}


@dataclass(frozen=True)
class ConceptMethodologyProfile:
    """Structure-derived methodological roles + hazards for one concept."""

    concept: str
    category: str
    roles: Tuple[str, ...]
    hazards: Tuple[str, ...]

    def tag(self) -> str:
        """Compact inline advisory tag, or ``""`` for a plain-safe concept."""
        parts = [_ROLE_TAG[r] for r in self.roles if r in _ROLE_TAG]
        return "; ".join(parts)


def _concept_category(name: str, dictionary: Optional[object]) -> str:
    if dictionary is None:
        return ""
    candidates = [name, _resolve_concept(_strip_stat_suffix(name), dictionary)]
    for key in candidates:
        if not key:
            continue
        try:
            return str(getattr(dictionary[key], "category", "") or "")  # type: ignore[index]
        except (KeyError, TypeError):
            continue
    return ""


def concept_methodology_profile(
    name: str,
    *,
    category: Optional[str] = None,
    dictionary: Optional[object] = None,
) -> ConceptMethodologyProfile:
    """Methodological role profile for a concept, derived from its structure.

    ``category`` may be passed when the caller already enriched it (the data
    catalog does), avoiding a second dictionary lookup; otherwise it is read
    from the concept dictionary. Roles are non-exclusive (a derived score can
    also be ordinal).
    """
    dic = dictionary if dictionary is not None else _concept_dictionary()
    cat = category if category is not None else _concept_category(name, dic) or ""
    cat = cat.lower().strip()

    found: set = set()
    # A computed concept (callback / depends_on) — overadjustment/collinearity.
    is_derived = is_derived_exposure(name, dic)
    if is_derived:
        found.add("derived_composite")
    # The dictionary's "outcome" category lumps true endpoints (death, LoS) with
    # derived severity scores (sofa / sep3 / news). Only the NON-derived ones are
    # study endpoints whose use as a predictor is leakage; a derived score in the
    # same category is a score, already covered by ``derived_composite``.
    if cat == "outcome" and not is_derived:
        found.add("outcome")
    if cat in _TREATMENT_CATEGORIES:
        found.add("treatment")
    hint = _lookup_hint(name)
    if hint is not None and hint.is_ordinal:
        found.add("ordinal_score")
    # Plain time-varying measurement (only when it carries no sharper hazard).
    if cat in _MEASUREMENT_CATEGORIES and not (
        found & {"derived_composite", "treatment"}
    ):
        found.add("measurement")

    order = (
        "outcome",
        "treatment",
        "derived_composite",
        "ordinal_score",
        "measurement",
    )
    roles = tuple(r for r in order if r in found)
    hazards = tuple(_ROLE_HAZARD[r] for r in roles if r in _ROLE_HAZARD)
    return ConceptMethodologyProfile(
        concept=name, category=cat, roles=roles, hazards=hazards
    )


def concept_methodology_tag(
    name: str,
    *,
    category: Optional[str] = None,
    dictionary: Optional[object] = None,
) -> str:
    """Compact advisory tag for one concept, or ``""`` for a plain-safe one."""
    return concept_methodology_profile(
        name, category=category, dictionary=dictionary
    ).tag()


# ---------------------------------------------------------------------------
# Outcome-leakage and treatment-mediator detectors (teeth)
# ---------------------------------------------------------------------------
#
# These mirror detect_overadjustment / overadjustment_caution: a firm error for
# the objective-leakage case, a caution for the timing/DAG-dependent case the
# detector cannot adjudicate. They keep the impartiality contract — only an
# unambiguous, definitional error gates; a defensible analytical choice is
# surfaced as a caution and never re-fits or blocks. The advisory profile above
# warns when a role is *assigned*; these flag the hazard once a model has been
# fitted.


def _is_true_outcome_concept(name: str, dictionary: Optional[object]) -> bool:
    """True when ``name`` is a study endpoint, not a derived severity score.

    The dictionary's ``outcome`` category lumps genuine endpoints (death_icu,
    los_icu, readmission) with derived scores (sofa / sep3 / news). Only the
    NON-derived ones are endpoints whose use as a predictor risks leakage; a
    derived score is covered by the overadjustment / derived-composite rules.
    Degrades to ``False`` when the dictionary is unavailable (no false positives
    without evidence).
    """
    cat = (_concept_category(name, dictionary) or "").lower().strip()
    if cat != "outcome":
        return False
    return not is_derived_exposure(name, dictionary)


def detect_outcome_as_predictor(
    predictors: Sequence[str],
    *,
    study_outcome: Optional[str] = None,
    dictionary: Optional[object] = None,
) -> List[str]:
    """Return predictors that ARE the study outcome — self-leakage.

    Conditioning a model on its own dependent variable (the declared
    ``study_outcome`` appearing among the right-hand-side predictors, spelled
    with a stat/derivation suffix too) is target leakage by construction — an
    objective error like overadjustment, never an analytical-preference call.
    Conservative: fires only when ``study_outcome`` is declared and a predictor
    token matches it. A *different* endpoint concept among the predictors is a
    timing-dependent hazard handled by ``outcome_leakage_caution`` (a caution),
    not flagged here. ``dictionary`` is accepted for signature symmetry but the
    self-leakage match is purely token-based and does not need it. An empty list
    means none detected.
    """
    out_tok = _strip_stat_suffix(study_outcome or "")
    if not out_tok:
        return []
    offenders: List[str] = []
    for pred in predictors:
        pred_tok = _normalise_token(pred)
        if not pred_tok:
            continue
        if _is_exposure_self(pred_tok, out_tok):
            offenders.append(str(pred))
    return offenders


def outcome_leakage_caution(
    predictors: Sequence[str],
    *,
    study_outcome: Optional[str] = None,
    dictionary: Optional[object] = None,
) -> Optional[str]:
    """A caution when a (different) endpoint concept is used as a predictor.

    A true endpoint concept (death_icu, los_icu, readmission — dictionary
    category ``outcome`` and not a derived score) used as a covariate is a
    leakage hazard *if* it is concurrent with or downstream of the study
    endpoint. Whether it actually leaks depends on timing the detector cannot
    see (a genuinely pre-baseline endpoint can be a legitimate covariate), so
    this is a caution, never a gating error. The declared ``study_outcome``
    itself is excluded (that self-leakage case is the firm
    ``detect_outcome_as_predictor`` error). Returns a one-line caution naming the
    offenders, or ``None``.
    """
    dic = dictionary if dictionary is not None else _concept_dictionary()
    out_tok = _strip_stat_suffix(study_outcome or "")
    flagged: List[str] = []
    for pred in predictors:
        pred_tok = _normalise_token(pred)
        if not pred_tok:
            continue
        if out_tok and _is_exposure_self(pred_tok, out_tok):
            continue  # the study outcome itself -> firm error path
        if _is_true_outcome_concept(pred, dic):
            flagged.append(str(pred))
    if not flagged:
        return None
    joined = ", ".join(f"'{f}'" for f in flagged)
    return (
        f"predictor(s) {joined} are outcome/endpoint concepts; if they are "
        "concurrent with or downstream of the study endpoint, using them as "
        "covariates leaks the outcome — confirm each is measured before the "
        "exposure/baseline and is not a competing or nested endpoint"
    )


def treatment_mediator_caution(
    exposure: str,
    adjustment_covariates: Sequence[str],
    *,
    dictionary: Optional[object] = None,
) -> Optional[str]:
    """A caution when a treatment/intervention concept is in the adjustment set.

    A treatment that lies on the exposure->outcome causal pathway is a mediator;
    adjusting for it biases the total-effect estimate. But adjusting for a
    *pre-exposure* treatment is legitimate confounder control. The detector
    cannot see the causal DAG or treatment timing, so this is a caution that
    prompts the analyst to confirm the role — never an error, and it never
    re-fits or gates. Fires only when an exposure is declared and at least one
    covariate is a treatment concept (and is not the exposure itself). Returns a
    one-line caution naming the treatment covariates, or ``None``.
    """
    exp_tok = _strip_stat_suffix(exposure or "")
    if not exp_tok:
        return None
    dic = dictionary if dictionary is not None else _concept_dictionary()
    flagged: List[str] = []
    for cov in adjustment_covariates:
        cov_tok = _normalise_token(cov)
        if not cov_tok or _is_exposure_self(cov_tok, exp_tok):
            continue
        if "treatment" in concept_methodology_profile(cov, dictionary=dic).roles:
            flagged.append(str(cov))
    if not flagged:
        return None
    joined = ", ".join(f"'{f}'" for f in flagged)
    return (
        f"adjustment covariate(s) {joined} are treatment/intervention concepts; "
        "if a treatment lies on the exposure->outcome pathway it is a mediator "
        "and adjusting for it biases the total effect — confirm each is a "
        "pre-exposure confounder rather than a mediator before adjusting"
    )


__all__ = [
    "VariableKind",
    "ConceptHint",
    "MethodologicalPrinciple",
    "ConceptMethodologyProfile",
    "companion_count_column_for_measured",
    "classify_variable",
    "aggregation_rule_for",
    "default_time_windows",
    "GENERAL_ICU_ANALYSIS_PRINCIPLES",
    "principles_for_phase",
    "ICU_RULES",
    "COMPOSITE_EXPOSURE_CONSTITUENTS",
    "composite_constituents",
    "detect_overadjustment",
    "is_derived_exposure",
    "overadjustment_caution",
    "detect_outcome_as_predictor",
    "outcome_leakage_caution",
    "treatment_mediator_caution",
    "concept_methodology_profile",
    "concept_methodology_tag",
]
