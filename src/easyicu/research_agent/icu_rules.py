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
            "SOFA-2 totals may reflect component-level missingness; cross-check component completeness before drawing clinical conclusions.",
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


__all__ = [
    "VariableKind",
    "ConceptHint",
    "MethodologicalPrinciple",
    "classify_variable",
    "aggregation_rule_for",
    "default_time_windows",
    "GENERAL_ICU_ANALYSIS_PRINCIPLES",
    "principles_for_phase",
    "ICU_RULES",
]
