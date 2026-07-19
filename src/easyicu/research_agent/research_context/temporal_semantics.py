"""Deterministic ICU temporal semantics helpers.

The runtime should not leave phrases such as "first 24h SOFA" or
"worst lactate before vasopressor" as vague prose. This module turns
common ICU timing phrases into structured, replayable constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ..schema import TimeWindow, TemporalConstraint


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
            rf"event(?:\s|_|-)+onset)\b",
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


def _normalise_anchor(anchor: str) -> str:
    anchor = anchor.strip().lower().replace("-", " ").replace("_", " ")
    anchor = re.sub(r"\s+", " ", anchor)
    if anchor in {"icu admission", "admission"}:
        return "icu_admission"
    if anchor == "hospital admission":
        return "hospital_admission"
    return anchor.replace(" ", "_")


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
