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
    generic cohort window or a Planner assertion.  A study without a primary
    exposure is ``not_applicable``; :func:`study_time_origin_alignment` owns
    its time zero.
    """

    status: Literal[
        "aligned",
        "mismatch",
        "declared_only",
        "materialized_only",
        "unspecified",
        "not_applicable",
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


def _window_anchor(window: str) -> Optional[str]:
    """The event a materialized analysis window counts its hours from."""

    window = str(window or "").strip()
    if not window:
        return None
    prefix = re.match(
        r"^\s*(?P<anchor>[A-Za-z][A-Za-z0-9 _-]{1,80})\s*\[",
        window,
    )
    if prefix:
        return normalise_time_anchor(prefix.group("anchor"))

    explicit = re.search(
        r"\b(?:after|from|anchored\s+(?:at|to)|relative\s+to)\s+(?:the\s+)?"
        r"(?P<anchor>icu[ _-]+admission|hospital[ _-]+admission|"
        r"event[ _-]+onset|suspected[ _-]+infection[ _-]+onset)\b",
        window,
        re.I,
    )
    if explicit:
        return normalise_time_anchor(explicit.group("anchor"))
    return None


def _materialized_primary_anchor(
    context: ResearchContext,
) -> tuple[Optional[str], Optional[str]]:
    exposure_name = str(context.primary_exposure or "").strip()
    descriptor = context.variable(exposure_name) if exposure_name else None
    anchor = _window_anchor(str(getattr(descriptor, "analysis_window", "") or ""))
    if anchor is None:
        return None, None
    return anchor, f"variables.{exposure_name}.analysis_window"


def primary_exposure_time_anchor_alignment(
    context: ResearchContext,
) -> PrimaryExposureTimeAnchorAlignment:
    """Compare sealed study time zero with an owner-issued concept contract."""

    declared, declared_source = _declared_primary_anchor(context)
    exposure_name = str(context.primary_exposure or "").strip()
    if not exposure_name:
        # No exposure definition exists to carry the declared time zero.
        return PrimaryExposureTimeAnchorAlignment(
            status="not_applicable",
            primary_exposure=None,
            declared_anchor=declared,
            definition_anchor=None,
            observation_window_anchor=None,
            observation_window_role=None,
            declared_source=declared_source,
            definition_source=None,
            observation_window_source=None,
        )
    observation, observation_source = _materialized_primary_anchor(context)
    descriptor = context.variable(exposure_name)
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


@dataclass(frozen=True)
class StudyTimeOriginAlignment:
    """A declared time zero against the windows of a study with no primary exposure.

    Every materialized analysis window names the event its hours count from.
    A trajectory, descriptive or audit study has no exposure definition to
    carry a declared time zero, so the declaration is compared with those
    windows.  A study with a primary exposure is ``not_applicable`` here: its
    time zero belongs to :func:`primary_exposure_time_anchor_alignment`.
    """

    status: Literal[
        "aligned",
        "mismatch",
        "declared_only",
        "unspecified",
        "not_applicable",
    ]
    declared_anchor: Optional[str]
    declared_source: Optional[str]
    window_anchors: tuple[str, ...]
    window_sources: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "declared_anchor": self.declared_anchor,
            "declared_source": self.declared_source,
            "window_anchors": list(self.window_anchors),
            "window_sources": list(self.window_sources),
        }


def study_time_origin_alignment(context: ResearchContext) -> StudyTimeOriginAlignment:
    """Compare the declared time zero of a study without a primary exposure.

    ``aligned`` when every materialized window counts from the declared event,
    ``mismatch`` when one counts from another.  A declaration with no window
    to compare (``declared_only``) asks nothing of the windows.
    """

    declared, declared_source = _declared_primary_anchor(context)
    if str(context.primary_exposure or "").strip():
        return StudyTimeOriginAlignment(
            "not_applicable", declared, declared_source, (), ()
        )
    windows = [
        (variable.name, anchor)
        for variable in context.variables
        if (anchor := _window_anchor(str(variable.analysis_window or ""))) is not None
    ]
    anchors = tuple(sorted({anchor for _, anchor in windows}))
    sources = tuple(f"variables.{name}.analysis_window" for name, _ in windows)
    if declared is None:
        status: Literal[
            "aligned", "mismatch", "declared_only", "unspecified", "not_applicable"
        ] = "unspecified"
    elif not anchors:
        status = "declared_only"
    else:
        status = "aligned" if anchors == (declared,) else "mismatch"
    return StudyTimeOriginAlignment(status, declared, declared_source, anchors, sources)


#: "first-24h", "first 24-hour", "the initial 3 days", "first day".
_FIRST_DURATION = (
    r"\b(?:first|initial)[\s_-]+(?:(?P<number>\d+(?:\.\d+)?)[\s_-]*"
    r"(?P<unit>hours?|hrs?|h|days?|d)|(?P<day>day))(?![a-z0-9])"
)
_TRAJECTORY_NOUN = r"(?:trajector(?:y|ies)|time[\s-]+courses?)"
#: Words a modifier run never crosses: a duration before them belongs to
#: another part of the question ("intubated within the first 24 h do SOFA-2
#: trajectories cluster" bounds the cohort, not the trajectories).
_CLAUSE_WORDS = (
    r"(?:do|does|did|is|are|was|were|be|been|can|could|will|would|should|may|"
    r"which|what|how|whether|who|that|than|and|or|but|if|when|while|"
    r"among|in|within|with|without|during|for|to|by|at|on|of|after|before|"
    r"following|since|from|post|over|across|throughout|"
    r"patients?|people|adults?|children|cohort|cluster\w*|predict\w*|"
    r"associat\w*|differ\w*|emerge\w*|identify|form|define\w*)"
)
_WORD = rf"(?!{_CLAUSE_WORDS}(?![a-z0-9]))[a-z0-9][a-z0-9/'-]*"
_SEPARATOR = r"(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:(?:and|or)\s+)?)"
_DURATION_BEFORE_TRAJECTORY = re.compile(
    rf"{_FIRST_DURATION}(?:[\s-]+{_WORD}){{0,4}}?[\s-]+{_TRAJECTORY_NOUN}(?![a-z])",
    re.I,
)
#: The first of two coordinated windows: "first-24h and first-72h trajectories".
_COORDINATED_DURATION = re.compile(
    rf"{_FIRST_DURATION}\s+(?:and|or|versus|vs\.?)\s+(?:the\s+)?$", re.I
)
_TRAJECTORY_BEFORE_DURATION = re.compile(
    rf"{_TRAJECTORY_NOUN}(?:\s+of(?:{_SEPARATOR}{_WORD}){{1,8}}?)?[\s,]+"
    rf"(?:over|during|in|within|across|throughout|for)\s+(?:the\s+)?{_FIRST_DURATION}",
    re.I,
)
_ANCHOR_AFTER = re.compile(
    rf"\s+(?P<relation>of|after|following|since|from|post)[\s-]+(?:the\s+)?"
    rf"(?P<anchor>{_WORD}(?:[\s-]+{_WORD}){{0,4}})",
    re.I,
)
#: "首24小时生理轨迹", "入ICU后前72小时内的SOFA轨迹", "插管后48小时的轨迹".
#: "前" after an event means "before" it ("入ICU前72小时"), so it reads as
#: "first" only at the start of a phrase or after 的/在/于.
_ZH_TRAJECTORY_WINDOW = re.compile(
    r"(?:(?P<anchor>[一-鿿A-Za-z0-9-]{1,12}?)后的?\s*"
    r"(?:(?:首|前|最初|头)\s*个?\s*)?"
    r"|(?:首|最初|头|(?:(?<![一-鿿A-Za-z0-9])|(?<=[的在于]))前)\s*个?\s*)"
    r"(?P<number>\d+(?:\.\d+)?)\s*个?\s*(?P<unit>小时|h|天|日)(?:以内|内)?"
    r"(?P<gap>[^，。；、,.;:：？?！!()（）\s]{0,12}?)(?:轨迹|动态变化|时间序列)",
    re.I,
)
_ZH_FIRST_DAY_TRAJECTORY = re.compile(
    r"(?:首日|第一天)(?P<gap>[^，。；、,.;:：？?！!()（）\s]{0,12}?)(?:轨迹|动态变化|时间序列)"
)
#: A window before these words qualifies the patients, not the trajectories.
_ZH_POPULATION_WORDS = ("患者", "病人", "人群", "者", "中")
#: Words of an anchor that is ICU admission itself ("of the ICU stay").
_ICU_TIME_ZERO_WORDS = frozenset(
    {"icu", "intensive", "care", "unit", "stay", "course", "admission", "the",
     "their", "first", "index", "initial", "observation", "monitoring"}
)
_ZH_ICU_TIME_ZERO = ("icu", "重症", "监护", "入科")


@dataclass(frozen=True)
class TrajectoryWindowStatement:
    """A window a question states for its trajectories."""

    hours: float
    anchor: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"hours": self.hours, "anchor": self.anchor, "text": self.text}


def _english_anchor(phrase: Optional[str]) -> str:
    if not phrase:
        return "icu_admission"
    words = re.findall(r"[a-z0-9]+", phrase.lower())
    if words and set(words) <= _ICU_TIME_ZERO_WORDS:
        return "icu_admission"
    return normalise_time_anchor(phrase)


def trajectory_window_statements(
    question: str,
) -> tuple[TrajectoryWindowStatement, ...]:
    """The windows a question states for its trajectories, in question order.

    Only a first-N duration attached to the trajectory wording counts
    ("first-24h trajectories", "trajectories over the first 72 hours",
    "首24小时生理轨迹"): a window elsewhere can bound the cohort, an exposure
    or an outcome and never binds the trajectory.  A window with no stated
    anchor counts from ICU admission, as every first-N-hours phrase does here;
    one stated from another event keeps that event.  A duration the reader
    cannot attach is not read, so the design keeps its own default window.
    """

    text = str(question or "")
    found: list[tuple[int, TrajectoryWindowStatement]] = []

    def hours(match: re.Match[str], *, days: bool) -> float:
        if match.groupdict().get("day"):
            return 24.0
        return float(match.group("number")) * (24.0 if days else 1.0)

    for pattern, relations in (
        (_DURATION_BEFORE_TRAJECTORY, {"after", "following", "since", "from", "post"}),
        (_TRAJECTORY_BEFORE_DURATION, {"of", "after", "following", "since", "from", "post"}),
    ):
        for match in pattern.finditer(text):
            anchored = _ANCHOR_AFTER.match(text, match.end())
            if anchored and anchored.group("relation").lower() not in relations:
                anchored = None
            anchor = _english_anchor(anchored.group("anchor") if anchored else None)
            end = anchored.end() if anchored else match.end()
            durations = [match]
            if pattern is _DURATION_BEFORE_TRAJECTORY:
                coordinated = _COORDINATED_DURATION.search(text[: match.start()])
                if coordinated:
                    durations.insert(0, coordinated)
            for duration in durations:
                unit = (duration.group("unit") or "").lower()
                found.append((
                    duration.start(),
                    TrajectoryWindowStatement(
                        hours=hours(duration, days=unit.startswith("d")),
                        anchor=anchor,
                        text=text[duration.start() : end].strip(),
                    ),
                ))
    for pattern in (_ZH_TRAJECTORY_WINDOW, _ZH_FIRST_DAY_TRAJECTORY):
        for match in pattern.finditer(text):
            if any(word in match.group("gap") for word in _ZH_POPULATION_WORDS):
                continue
            anchor_text = (match.groupdict().get("anchor") or "").strip()
            icu = not anchor_text or any(
                word in anchor_text.lower() for word in _ZH_ICU_TIME_ZERO
            )
            unit = match.groupdict().get("unit") or ""
            found.append((
                match.start(),
                TrajectoryWindowStatement(
                    hours=(
                        24.0
                        if pattern is _ZH_FIRST_DAY_TRAJECTORY
                        else hours(match, days=unit in {"天", "日"})
                    ),
                    anchor="icu_admission" if icu else anchor_text,
                    text=match.group(0).strip(),
                ),
            ))
    statements: list[TrajectoryWindowStatement] = []
    for _, statement in sorted(found, key=lambda item: item[0]):
        if all(
            (statement.hours, statement.anchor) != (seen.hours, seen.anchor)
            for seen in statements
        ):
            statements.append(statement)
    return tuple(statements)


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
