"""Deterministic ICU temporal semantics helpers.

The runtime should not leave phrases such as "first 24h SOFA" or
"worst lactate before vasopressor" as vague prose. This module turns
common ICU timing phrases into structured, replayable constraints.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import pandas as pd

from ..schema import ResearchContext, TimeWindow, TemporalConstraint


_WS = r"(?:\s|_)+"
_PATTERNS = [
    (
        "first_window",
        re.compile(rf"\bfirst{_WS}(?P<hours>\d+(?:\.\d+)?)\s*h(?:ours?)?\b", re.I),
    ),
    (
        "within_after",
        re.compile(
            rf"\b(?P<concept>aki|sofa|sofa-?2|lactate|creatinine|ventilation|vasopressor)?"
            rf".*?\bwithin{_WS}(?P<hours>\d+(?:\.\d+)?)\s*h(?:ours?)?"
            rf"{_WS}after{_WS}(?P<anchor>icu admission|admission|hospital admission)\b",
            re.I,
        ),
    ),
    (
        "worst_before_event",
        re.compile(
            rf"\bworst{_WS}(?P<concept>[a-z0-9_/-]+)\b.*?\bbefore{_WS}(?P<anchor>vasopressor|vasopressors|intubation|rrt|ventilation)\b",
            re.I,
        ),
    ),
    (
        "relative_to_anchor",
        re.compile(
            rf"\b(?:from|anchored{_WS}(?:at|to)|relative{_WS}to){_WS}"
            rf"(?:the{_WS})?(?P<anchor>"
            rf"icu(?:\s|_|-)+admission|hospital(?:\s|_|-)+admission|"
            rf"event(?:\s|_|-)+onset|suspected(?:\s|_|-)+infection"
            rf"(?:\s|_|-)+onset)\b",
            re.I,
        ),
    ),
    (
        "before_event",
        re.compile(
            rf"\bbefore{_WS}(?P<anchor>vasopressor|vasopressors|intubation|rrt|ventilation)\b",
            re.I,
        ),
    ),
]


def window_extends_after_anchor(analysis_window: str) -> bool:
    """Return whether a textual clinical window includes post-anchor time.

    A dash between digits is a range delimiter (``0-24h``), not a unary
    minus. Genuine negative origins such as ``-24 to 0h`` remain negative.
    Plan-time method binding and publication-readiness share this owner so the
    two gates cannot disagree about the same clinical window.
    """

    window = str(analysis_window or "").strip()
    if not window:
        return False
    numeric_window = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", " to ", window)
    values = [
        float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", numeric_window)
    ]
    return bool(values and max(values) > 0)


def normalise_time_anchor(anchor: str) -> str:
    """Return one stable identity for a declared clinical time anchor.

    This is deliberately an identity normaliser, not an inference engine.  It
    may collapse spelling variants such as ``ICU-admission`` and
    ``icu_admission``; it must never decide that ICU admission and suspected-
    infection onset are interchangeable clinical events.
    """

    anchor = anchor.strip().lower().replace("-", " ").replace("_", " ")
    anchor = re.sub(r"\s+", " ", anchor)
    if anchor in {"icu admission", "admission"}:
        return "icu_admission"
    if anchor == "hospital admission":
        return "hospital_admission"
    if anchor in {"suspected infection", "suspected infection onset"}:
        return "suspected_infection_onset"
    return anchor.replace(" ", "_")


def _normalise_anchor(anchor: str) -> str:
    """Backward-compatible private alias for the public owner function."""

    return normalise_time_anchor(anchor)


@dataclass(frozen=True)
class PrimaryExposureTimeAnchorAlignment:
    """Digest-friendly decision about declared versus materialized time zero.

    ``declared_anchor`` comes only from sealed study/question authority.  A
    clinical definition comes only from the descriptor's typed clinical
    contract.  The physical analysis window remains a separate observation
    coordinate.  Missing evidence stays unresolved and is never filled from a
    generic cohort window or a Planner assertion.
    """

    status: Literal[
        "aligned",
        "mismatch",
        "declared_only",
        "materialized_only",
        "unspecified",
    ]
    primary_exposure: Optional[str]
    declared_anchor: Optional[str]
    definition_anchor: Optional[str]
    observation_window_anchor: Optional[str]
    observation_window_role: Optional[
        Literal["exposure_definition", "outer_observation_window"]
    ]
    declared_source: Optional[str]
    definition_source: Optional[str]
    observation_window_source: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "primary_exposure": self.primary_exposure,
            "declared_anchor": self.declared_anchor,
            "definition_anchor": self.definition_anchor,
            "observation_window_anchor": self.observation_window_anchor,
            "observation_window_role": self.observation_window_role,
            "declared_source": self.declared_source,
            "definition_source": self.definition_source,
            "observation_window_source": self.observation_window_source,
        }


def _mapping_from_json_text(value: object) -> Mapping[str, Any]:
    text = str(value or "").strip()
    if not text.startswith("{"):
        return {}
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _declared_primary_anchor(
    context: ResearchContext,
) -> tuple[Optional[str], Optional[str]]:
    preferences = context.user_preferences
    timing = getattr(preferences, "timing_and_design", None)
    timing_payload = _mapping_from_json_text(timing)
    explicit = str(timing_payload.get("anchor") or "").strip()
    if explicit:
        return normalise_time_anchor(explicit), "user_preferences.timing_and_design.anchor"

    relative = {
        normalise_time_anchor(item.anchor_event)
        for item in context.temporal_constraints
        if item.relation == "relative_to_anchor" and str(item.anchor_event).strip()
    }
    if len(relative) == 1:
        return next(iter(relative)), "temporal_constraints.relative_to_anchor"

    # Historical contexts may contain the exact request but predate the typed
    # constraint projection.  Parsing is acceptable here because it recovers
    # only an explicit phrase; it does not invent a clinical anchor.
    parsed = {
        normalise_time_anchor(item.anchor_event)
        for item in TimeWindowSemanticParser().parse(context.research_question)
        if item.relation == "relative_to_anchor" and str(item.anchor_event).strip()
    }
    if len(parsed) == 1:
        return next(iter(parsed)), "research_question.explicit_relative_anchor"
    return None, None


def _materialized_primary_anchor(
    context: ResearchContext,
) -> tuple[Optional[str], Optional[str]]:
    exposure_name = str(context.primary_exposure or "").strip()
    descriptor = context.variable(exposure_name) if exposure_name else None
    window = str(getattr(descriptor, "analysis_window", "") or "").strip()
    if not window:
        return None, None

    prefix = re.match(
        r"^\s*(?P<anchor>[A-Za-z][A-Za-z0-9 _-]{1,80})\s*\[",
        window,
    )
    if prefix:
        return (
            normalise_time_anchor(prefix.group("anchor")),
            f"variables.{exposure_name}.analysis_window",
        )

    explicit = re.search(
        r"\b(?:after|from|anchored\s+(?:at|to)|relative\s+to)\s+(?:the\s+)?"
        r"(?P<anchor>icu[ _-]+admission|hospital[ _-]+admission|"
        r"event[ _-]+onset|suspected[ _-]+infection[ _-]+onset)\b",
        window,
        re.I,
    )
    if explicit:
        return (
            normalise_time_anchor(explicit.group("anchor")),
            f"variables.{exposure_name}.analysis_window",
        )
    return None, None


def primary_exposure_time_anchor_alignment(
    context: ResearchContext,
) -> PrimaryExposureTimeAnchorAlignment:
    """Compare sealed study time zero with an owner-issued concept contract."""

    declared, declared_source = _declared_primary_anchor(context)
    observation, observation_source = _materialized_primary_anchor(context)
    exposure_name = str(context.primary_exposure or "").strip()
    descriptor = context.variable(exposure_name) if exposure_name else None
    definition = getattr(descriptor, "clinical_definition", None)
    definition_anchor = normalise_time_anchor(definition.definition_time_anchor) if (
        definition is not None and definition.definition_time_anchor
    ) else None
    definition_source = (
        f"variables.{exposure_name}.clinical_definition:{definition.contract_id}"
        if definition_anchor is not None
        else None
    )
    observation_role = getattr(descriptor, "analysis_window_role", None)
    # A dictionary may explicitly declare that its analysis window is the
    # clinical definition.  A materialized cohort derivation window never is.
    comparison_anchor = definition_anchor
    comparison_source = definition_source
    if comparison_anchor is None and observation_role == "exposure_definition":
        comparison_anchor = observation
        comparison_source = observation_source

    if declared and comparison_anchor:
        status: Literal[
            "aligned",
            "mismatch",
            "declared_only",
            "materialized_only",
            "unspecified",
        ] = "aligned" if declared == comparison_anchor else "mismatch"
    elif declared:
        status = "declared_only"
    elif comparison_anchor:
        status = "materialized_only"
    else:
        status = "unspecified"
    return PrimaryExposureTimeAnchorAlignment(
        status=status,
        primary_exposure=str(context.primary_exposure or "").strip() or None,
        declared_anchor=declared,
        definition_anchor=comparison_anchor,
        observation_window_anchor=observation,
        observation_window_role=(
            observation_role if observation is not None else None
        ),
        declared_source=declared_source,
        definition_source=comparison_source,
        observation_window_source=observation_source,
    )


class TimeWindowSemanticParser:
    """Parse common ICU timing phrases into structured constraints."""

    def parse(self, text: str) -> List[TemporalConstraint]:
        out: List[TemporalConstraint] = []
        if not text:
            return out
        for relation, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groupdict()
                anchor = _normalise_anchor(groups.get("anchor") or "icu_admission")
                hours = float(groups["hours"]) if groups.get("hours") else None
                concept = groups.get("concept")
                constraint = TemporalConstraint(
                    raw_text=match.group(0),
                    relation=relation,  # type: ignore[arg-type]
                    anchor_event=anchor,
                    target_concept=(concept.lower() if concept else None),
                    start_hours=(
                        0.0 if relation in {"first_window", "within_after"} else None
                    ),
                    end_hours=(
                        hours
                        if relation in {"first_window", "within_after"}
                        else None
                        if relation == "relative_to_anchor"
                        else 0.0
                    ),
                    aggregation_hint=(
                        "worst" if relation == "worst_before_event" else None
                    ),
                    executable_repr=_render_constraint_repr(
                        relation=relation,
                        anchor=anchor,
                        hours=hours,
                        concept=concept.lower() if concept else None,
                    ),
                )
                out.append(constraint)
        return _deduplicate_constraints(out)


def _render_constraint_repr(
    *,
    relation: str,
    anchor: str,
    hours: Optional[float],
    concept: Optional[str],
) -> str:
    parts = [relation, f"anchor={anchor}"]
    if concept:
        parts.append(f"concept={concept}")
    if hours is not None:
        parts.append(f"hours={hours:g}")
    return "|".join(parts)


def _deduplicate_constraints(
    items: Sequence[TemporalConstraint],
) -> List[TemporalConstraint]:
    seen = set()
    out: List[TemporalConstraint] = []
    for item in items:
        if item.executable_repr in seen:
            continue
        seen.add(item.executable_repr)
        out.append(item)
    return out


class TemporalAlignmentEngine:
    """Turn temporal constraints into canonical analysis windows when possible."""

    def infer(
        self,
        *,
        research_question: str,
        timing_and_design: Optional[str] = None,
        explicit_windows: Optional[Sequence[TimeWindow]] = None,
    ) -> tuple[List[TimeWindow], List[TemporalConstraint]]:
        parser = TimeWindowSemanticParser()
        constraints = parser.parse(research_question or "")
        if timing_and_design:
            constraints.extend(parser.parse(timing_and_design))
        constraints = _deduplicate_constraints(constraints)

        windows = list(explicit_windows or [])
        if not windows:
            for constraint in constraints:
                if (
                    constraint.relation == "first_window"
                    and constraint.end_hours is not None
                ):
                    windows.append(
                        TimeWindow(
                            name=f"first_{int(constraint.end_hours)}h",
                            anchor="icu_admission",
                            start_hours=0.0,
                            end_hours=float(constraint.end_hours),
                            rationale=f"Inferred from request phrase: {constraint.raw_text}",
                        )
                    )
                elif (
                    constraint.relation == "within_after"
                    and constraint.end_hours is not None
                    and constraint.anchor_event
                    in {"icu_admission", "hospital_admission"}
                ):
                    windows.append(
                        TimeWindow(
                            name=f"within_{int(constraint.end_hours)}h_after_{constraint.anchor_event}",
                            anchor=constraint.anchor_event,  # type: ignore[arg-type]
                            start_hours=0.0,
                            end_hours=float(constraint.end_hours),
                            rationale=f"Inferred from request phrase: {constraint.raw_text}",
                        )
                    )
        return windows, constraints


@dataclass(frozen=True)
class EpisodeResolution:
    id_columns: List[str]
    time_columns: List[str]
    outcome_columns: List[str]
    provenance: Dict[str, Any]


class ICUEpisodeResolver:
    """Deterministically resolve cohort id/time/outcome columns."""

    def resolve(
        self,
        *,
        df: pd.DataFrame,
        database: str,
        id_columns: Sequence[str],
        time_columns: Sequence[str],
        outcome_columns: Sequence[str],
        target_outcome: Optional[str],
        cohort_path: Optional[str],
    ) -> EpisodeResolution:
        return EpisodeResolution(
            id_columns=list(id_columns),
            time_columns=list(time_columns),
            outcome_columns=list(outcome_columns),
            provenance={
                "database": database,
                "cohort_path": cohort_path,
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "id_columns": list(id_columns),
                "time_columns": list(time_columns),
                "outcome_columns": list(outcome_columns),
                "target_outcome": target_outcome,
                "resolver": self.__class__.__name__,
            },
        )


class ConceptValidationLayer:
    """Best-effort validation over concept descriptors before planning."""

    def validate_descriptor_payload(
        self,
        *,
        source_info: Optional[Dict[str, Any]],
        column_name: str,
    ) -> Dict[str, Any]:
        info = dict(source_info or {})
        return {
            "source_tables": _coerce_str_list(
                info.get("source_tables") or info.get("tables")
            ),
            "item_ids": _coerce_str_list(
                info.get("item_ids") or info.get("itemid") or info.get("itemid_list")
            ),
            "unit_normalization": _coerce_str(
                info.get("unit_normalization") or info.get("unit_harmonization")
            ),
            "temporal_resolution": _coerce_str(
                info.get("temporal_resolution") or info.get("resolution")
            ),
            "clinical_caveats": _coerce_str_list(
                info.get("clinical_caveats") or info.get("pitfalls")
            ),
            "missingness_semantics": _coerce_str(info.get("missingness_semantics")),
            "source_concept": _coerce_str(info.get("name")) or column_name,
        }


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(k) for k in value.keys()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


__all__ = [
    "ConceptValidationLayer",
    "TemporalAlignmentEngine",
    "ICUEpisodeResolver",
    "EpisodeResolution",
    "TimeWindowSemanticParser",
]
