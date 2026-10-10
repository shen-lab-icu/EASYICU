"""What a prediction's benchmark comparator is, and how its window stands.

Owner
-----
This module owns one reading of a benchmark comparator concept: whether it is
a probability or a score, which outcome concept it predicts, and the window
its value was computed over, and how that window stands to the model's
prediction time.  The prediction template chooses its comparator column with
these facts, and the static prediction owner reads the same facts when it
compares, so neither re-derives them from a column's name or its values.

The concept dictionary states a concept's units and bounds.  The three facts
a comparison reads beyond them -- which way a score points, what a
probability predicts, the window a native score was computed over -- are
reviewed here (:data:`COMPARATOR_READINGS`), not in the dictionary: the
dictionary is sealed with the extraction release its foundation lock names,
and these facts change no extraction.

- A probability is a concept whose only unit is ``fraction`` and whose
  bounds are exactly [0, 1].  A score is a concept whose ``risk_direction``
  is ``higher_is_worse``, so its AUROC reads the way the model's does.  Any
  other concept is not a comparator (``kind`` is ``None``): a score that
  falls as risk rises (GCS) would need reorienting, and one with no stated
  direction cannot be oriented at all.
- ``predicts`` is the outcome concept id the probability predicts.
  Calibration is compared only when it equals the study outcome's concept
  exactly (``calibration_reason``); no synonym or prefix matches.
- The information window is the window the comparator column's value was
  computed over (``comparator_information_window``): the column's own
  window, which the context records for a summary of observations
  (``sofa_max`` over the first 24 h), else the concept's ``analysis_window``
  here (APACHE IVa's first ICU day).  It is read on the ICU-admission axis by
  the materialization owner; a window on another anchor, or none, is
  ``comparator_window_unstated``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Optional

from ..research_context.materialization_window import column_window_from_label

ComparatorKind = Literal["probability", "score"]
#: The stated direction of a score this owner compares.
HIGHER_IS_WORSE = "higher_is_worse"
LOWER_IS_WORSE = "lower_is_worse"
_FIRST_ICU_DAY = "icu_admission[0,24]h"
#: What a comparison reads of each comparator concept beyond the dictionary's
#: units and bounds: ``risk_direction`` (which way a score's value means a
#: sicker patient), ``predicts`` (the outcome concept a probability predicts)
#: and ``analysis_window`` (the window a native stay-level value was computed
#: over).  A concept absent here states none of them.
COMPARATOR_READINGS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "sofa": MappingProxyType({"risk_direction": HIGHER_IS_WORSE}),
        "qsofa": MappingProxyType({"risk_direction": HIGHER_IS_WORSE}),
        "sirs": MappingProxyType({"risk_direction": HIGHER_IS_WORSE}),
        "mews": MappingProxyType({"risk_direction": HIGHER_IS_WORSE}),
        "news": MappingProxyType({"risk_direction": HIGHER_IS_WORSE}),
        "gcs": MappingProxyType({"risk_direction": LOWER_IS_WORSE}),
        "apache_iv": MappingProxyType(
            {"risk_direction": HIGHER_IS_WORSE, "analysis_window": _FIRST_ICU_DAY}
        ),
        "apache_iv_pred_hosp_mort": MappingProxyType(
            {"predicts": "death", "analysis_window": _FIRST_ICU_DAY}
        ),
        # SICdb records SAPS 3 within the hour around ICU admission.
        "saps3": MappingProxyType(
            {"risk_direction": HIGHER_IS_WORSE, "analysis_window": "icu_admission[-1,1]h"}
        ),
    }
)
CalibrationReason = Literal["score_scale", "predicts_another_outcome", "predicts_unstated"]
InformationWindowRelation = Literal[
    "same",
    "comparator_ends_after_prediction_time",
    "comparator_ends_before_prediction_time",
    "comparator_window_unstated",
]
INFORMATION_WINDOW_RELATIONS: tuple[InformationWindowRelation, ...] = (
    "same",
    "comparator_ends_after_prediction_time",
    "comparator_ends_before_prediction_time",
    "comparator_window_unstated",
)
CALIBRATION_REASONS: tuple[CalibrationReason, ...] = (
    "score_scale",
    "predicts_another_outcome",
    "predicts_unstated",
)


@dataclass(frozen=True)
class BenchmarkComparatorFacts:
    concept: str
    kind: Optional[ComparatorKind]
    predicts: Optional[str]
    information_window: Optional[str]


def benchmark_comparator_facts(concept: str) -> Optional[BenchmarkComparatorFacts]:
    """The facts a comparison reads of ``concept``; ``None`` when the dictionary has no entry."""

    from easyicu.resources import load_dictionary

    name = str(concept or "").strip()
    definition = load_dictionary(include_sofa2=True).get(name) if name else None
    if definition is None:
        return None
    units = [str(unit).strip().casefold() for unit in (definition.units or ())]
    probability = (
        units == ["fraction"]
        and definition.minimum == 0.0
        and definition.maximum == 1.0
    )
    reading = COMPARATOR_READINGS.get(name, {})
    rising = reading.get("risk_direction") == HIGHER_IS_WORSE
    predicts = reading.get("predicts")
    window = reading.get("analysis_window")
    return BenchmarkComparatorFacts(
        concept=name,
        kind="probability" if probability else "score" if rising else None,
        predicts=predicts,
        information_window=window,
    )


def calibration_reason(
    facts: BenchmarkComparatorFacts, *, outcome_concept: str
) -> Optional[CalibrationReason]:
    """Why calibration is not compared, or ``None`` when it is."""

    if facts.kind != "probability":
        return "score_scale"
    if facts.predicts is None:
        return "predicts_unstated"
    if facts.predicts != str(outcome_concept or "").strip():
        return "predicts_another_outcome"
    return None


def comparator_information_window(
    column_window: Optional[str], facts: BenchmarkComparatorFacts
) -> Optional[str]:
    """The window a comparator column's value was computed over, or ``None``.

    The column's own window when the context records one, else the window the
    dictionary states for its concept.
    """

    return str(column_window or "").strip() or facts.information_window


def information_window_relation(
    window_label: Optional[str], *, prediction_time_hours: float
) -> InformationWindowRelation:
    """How the comparator's information window ends against the prediction time."""

    window = column_window_from_label(window_label) if window_label else None
    if window is None or window.anchor != "icu_admission":
        return "comparator_window_unstated"
    if window.end_hours > prediction_time_hours:
        return "comparator_ends_after_prediction_time"
    if window.end_hours < prediction_time_hours:
        return "comparator_ends_before_prediction_time"
    return "same"


__all__ = [
    "BenchmarkComparatorFacts",
    "COMPARATOR_READINGS",
    "HIGHER_IS_WORSE",
    "LOWER_IS_WORSE",
    "CALIBRATION_REASONS",
    "INFORMATION_WINDOW_RELATIONS",
    "benchmark_comparator_facts",
    "calibration_reason",
    "comparator_information_window",
    "information_window_relation",
]
