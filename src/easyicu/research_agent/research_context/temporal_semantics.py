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


#: The admissions every materialized window and follow-up can count from.
ADMISSION_TIME_ZEROS = frozenset({"icu_admission", "hospital_admission"})


@dataclass(frozen=True)
class _EventTimeZero:
    """One clinical event a question can count its time from.

    ``punctual`` events name their own time ("after intubation"); a condition
    or a therapy names it only with an onset or start word ("after sepsis
    onset", "开始机械通气后") or a duration ("within 6 h of sepsis"), so
    "mortality after sepsis" states no time zero.  ``at_names_time`` is False
    where "at <event>" reads a status ("mortality at discharge").
    """

    identity: str
    english: str
    chinese: str
    punctual: bool = False
    at_names_time: bool = True


#: A closed vocabulary: an event outside it is never read as a time zero.
#: Longer spellings come first, so "septic shock" is not read as "shock".
_EVENT_TIME_ZEROS = (
    _EventTimeZero("suspected_infection_onset", r"suspected[\s_-]+infection", r"疑似感染"),
    _EventTimeZero(
        "septic_shock_onset", r"septic[\s_-]+shock", r"脓毒性休克|感染性休克|脓毒症休克"
    ),
    _EventTimeZero("sepsis_onset", r"sepsis", r"脓毒症"),
    _EventTimeZero("shock_onset", r"(?:circulatory[\s_-]+)?shock", r"休克"),
    _EventTimeZero(
        "intubation", r"(?:endotracheal[\s_-]+)?intubation", r"气管插管|插管", punctual=True
    ),
    _EventTimeZero("extubation", r"extubation", r"拔管", punctual=True),
    _EventTimeZero(
        "mechanical_ventilation_start",
        r"(?:invasive[\s_-]+)?mechanical[\s_-]+ventilation|invasive[\s_-]+ventilation",
        r"有创机械通气|机械通气|有创通气",
    ),
    _EventTimeZero(
        "vasopressor_start",
        r"vasopressors?(?:[\s_-]+(?:therapy|support))?",
        r"血管活性药物?|血管升压药物?|升压药物?",
    ),
    _EventTimeZero(
        "rrt_start",
        r"(?:continuous[\s_-]+)?renal[\s_-]+replacement[\s_-]+therapy|c?rrt|(?:hemo)?dialysis",
        r"连续性肾脏替代治疗|肾脏替代治疗|CRRT|RRT|血液透析|透析",
    ),
    _EventTimeZero("aki_onset", r"acute[\s_-]+kidney[\s_-]+injury|aki", r"急性肾损伤|AKI"),
    _EventTimeZero(
        "ards_onset",
        r"acute[\s_-]+respiratory[\s_-]+distress[\s_-]+syndrome|ards",
        r"急性呼吸窘迫综合征|ARDS",
    ),
    _EventTimeZero(
        "cardiac_arrest", r"(?:in[\s_-]+hospital[\s_-]+)?cardiac[\s_-]+arrest",
        r"心脏骤停|心搏骤停", punctual=True,
    ),
    _EventTimeZero(
        "rosc", r"return[\s_-]+of[\s_-]+spontaneous[\s_-]+circulation|rosc",
        r"自主循环恢复|ROSC", punctual=True,
    ),
    _EventTimeZero(
        "ed_arrival",
        r"(?:emergency[\s_-]+department|ed)[\s_-]+(?:arrival|presentation|triage)",
        r"急诊(?:到达|就诊|分诊)", punctual=True,
    ),
    _EventTimeZero(
        "icu_discharge",
        r"(?:icu|intensive[\s_-]+care(?:[\s_-]+unit)?)[\s_-]+discharge",
        r"转出ICU|出ICU|出重症监护室|出科", punctual=True, at_names_time=False,
    ),
    _EventTimeZero(
        "hospital_discharge", r"(?:hospital[\s_-]+)?discharge", r"出院",
        punctual=True, at_names_time=False,
    ),
)


def _event_group(event: _EventTimeZero) -> str:
    return f"event_{event.identity}"


def _event_alternation(
    events: Sequence[_EventTimeZero], language: Literal["english", "chinese"]
) -> str:
    return "|".join(
        f"(?P<{_event_group(event)}>{getattr(event, language)})" for event in events
    )


_ONSET_WORDS = (
    r"(?:onset|start|initiation|commencement|diagnosis|recognition|development)"
)
_AFTER_WORDS = r"(?:after|following|since|from|post)"
_DURATION = (
    r"(?P<number>\d+(?:\.\d+)?)[\s_-]*(?P<unit>hours?|hrs?|h|days?|d)(?![a-z0-9])"
)
_EN_EVENTS = _event_alternation(_EVENT_TIME_ZEROS, "english")
_EN_PUNCTUAL = _event_alternation([e for e in _EVENT_TIME_ZEROS if e.punctual], "english")
_EN_PUNCTUAL_AT = _event_alternation(
    [e for e in _EVENT_TIME_ZEROS if e.punctual and e.at_names_time], "english"
)
_EN_ONSET_EVENTS = _event_alternation(
    [e for e in _EVENT_TIME_ZEROS if e.at_names_time], "english"
)
#: "within 6 h of intubation", "the first 24 hours of septic shock",
#: "48 h after sepsis onset": a duration counted from the event.  "Of" and
#: "day" need a leading "within"/"first": "6 h of vasopressor therapy" is a
#: duration of the therapy, not a time counted from it.
_EN_DURATION_FROM_EVENT = re.compile(
    rf"\b(?P<lead>(?:within|first|initial)[\s_-]+(?:the[\s_-]+)?(?:first[\s_-]+)?)?"
    rf"(?:{_DURATION}|(?P<day>day))"
    rf"[\s_-]+(?P<relation>of|after|following|from|since|post)"
    rf"[\s_-]+(?:the[\s_-]+)?(?:{_ONSET_WORDS}[\s_-]+of[\s_-]+(?:the[\s_-]+)?)?"
    rf"(?:{_EN_EVENTS})(?:[\s_-]+{_ONSET_WORDS})?(?![a-z0-9])",
    re.I,
)
_EN_EVENT_TIME_ZERO_PATTERNS = (
    _EN_DURATION_FROM_EVENT,
    # "after the onset of sepsis", "since vasopressor initiation".
    re.compile(
        rf"\b(?:{_AFTER_WORDS}|at|upon)[\s_-]+(?:the[\s_-]+)?{_ONSET_WORDS}[\s_-]+of"
        rf"[\s_-]+(?:the[\s_-]+)?(?:{_EN_ONSET_EVENTS})(?![a-z0-9])",
        re.I,
    ),
    re.compile(
        rf"\b(?:{_AFTER_WORDS}|at|upon)[\s_-]+(?:the[\s_-]+)?(?:{_EN_ONSET_EVENTS})"
        rf"[\s_-]+{_ONSET_WORDS}(?![a-z0-9])",
        re.I,
    ),
    # "after intubation", "post-intubation", "at ROSC", "30 days after discharge".
    re.compile(
        rf"\b{_AFTER_WORDS}[\s_-]+(?:the[\s_-]+)?(?:{_EN_PUNCTUAL})(?![a-z0-9])", re.I
    ),
    re.compile(rf"\b(?:at|upon)[\s_-]+(?:the[\s_-]+)?(?:{_EN_PUNCTUAL_AT})(?![a-z0-9])", re.I),
)
_ZH_ONSET_WORDS = r"(?:发生|出现|开始|启动|起始|诊断|确诊|识别)"
_ZH_AFTER = r"(?:以后|之后|后)"
_ZH_DURATION = (
    r"(?P<number>\d+(?:\.\d+)?)\s*个?\s*(?P<unit>小时|h|天|日)"
)
_ZH_EVENTS = _event_alternation(_EVENT_TIME_ZEROS, "chinese")
_ZH_PUNCTUAL = _event_alternation([e for e in _EVENT_TIME_ZEROS if e.punctual], "chinese")
_ZH_EVENT_TIME_ZERO_PATTERNS = (
    # "脓毒症发生后24小时内", "插管后6小时", "机械通气开始后前48小时".
    re.compile(
        rf"(?:{_ZH_EVENTS}){_ZH_ONSET_WORDS}?{_ZH_AFTER}\s*的?\s*"
        rf"(?:(?:首|前|最初|头)\s*个?\s*)?{_ZH_DURATION}",
        re.I,
    ),
    # "插管后第一天", "脓毒症发生后首日": the first day counted from the event.
    re.compile(
        rf"(?:{_ZH_EVENTS}){_ZH_ONSET_WORDS}?{_ZH_AFTER}\s*的?\s*"
        rf"(?P<day>第\s*(?:一|1)\s*[天日]|首\s*日|头\s*一?\s*天)",
        re.I,
    ),
    # "脓毒症发生后", "机械通气开始时", "自插管起".
    re.compile(rf"(?:{_ZH_EVENTS}){_ZH_ONSET_WORDS}(?:{_ZH_AFTER}|起|时)", re.I),
    re.compile(rf"自\s*(?:{_ZH_EVENTS}){_ZH_ONSET_WORDS}?(?:起|开始)", re.I),
    # "开始机械通气后", "启动肾脏替代治疗后".
    re.compile(
        rf"(?:开始|启动|使用|接受|进行)(?:{_ZH_EVENTS})(?:治疗)?{_ZH_AFTER}", re.I
    ),
    # "插管后", "心脏骤停后", "出院后".
    re.compile(rf"(?:{_ZH_PUNCTUAL}){_ZH_AFTER}", re.I),
)


@dataclass(frozen=True)
class EventTimeZeroStatement:
    """A clinical event, other than an admission, a question counts time from."""

    anchor: str
    hours: Optional[float]
    start: int
    end: int
    text: str


def _matched_event(match: re.Match[str]) -> str:
    for event in _EVENT_TIME_ZEROS:
        if match.groupdict().get(_event_group(event)) is not None:
            return event.identity
    raise ValueError("an event time-zero pattern matched no event")  # pragma: no cover


def _event_readings(text: str) -> tuple[EventTimeZeroStatement, ...]:
    """Every phrase that counts time from a clinical event, in question order.

    Overlapping readings keep the earliest, longest one.
    """

    found: list[EventTimeZeroStatement] = []
    for pattern in (*_EN_EVENT_TIME_ZERO_PATTERNS, *_ZH_EVENT_TIME_ZERO_PATTERNS):
        for match in pattern.finditer(text):
            groups = match.groupdict()
            if pattern is _EN_DURATION_FROM_EVENT and not groups.get("lead") and (
                groups.get("day") or str(groups.get("relation")).lower() == "of"
            ):
                continue
            hours: Optional[float] = None
            if groups.get("day"):
                hours = 24.0
            elif groups.get("number"):
                unit = str(groups.get("unit") or "").lower()
                days = unit.startswith("d") or unit in {"天", "日"}
                hours = float(groups["number"]) * (24.0 if days else 1.0)
            found.append(
                EventTimeZeroStatement(
                    anchor=_matched_event(match),
                    hours=hours,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0).strip(),
                )
            )
    statements: list[EventTimeZeroStatement] = []
    for statement in sorted(found, key=lambda item: (item.start, -item.end)):
        if any(seen.start <= statement.start < seen.end for seen in statements):
            continue
        statements.append(statement)
    return tuple(statements)


_CLAUSE_STOP = re.compile(r"[.;:,!?()\[\]（）。；：，！？、\n]")
_EN_POPULATION_NOUNS = (
    r"(?:patients?|subjects?|adults?|children|survivors?|cases?|individuals?|"
    r"people|persons?|stays?|those|populations?|cohorts?)"
)
#: The event qualifies who is studied: "patients admitted to the ICU after
#: cardiac arrest", "in patients after ROSC", "post-cardiac arrest patients".
#: An admission word reaches the event across a short place ("to the ICU")
#: only.  A relative clause about the patients qualifies nothing by itself:
#: "patients who received steroids after septic shock onset" may state the
#: exposure.
_EN_ADMITTED_BEFORE = re.compile(
    r"\b(?:admitted|admissions|transferred|presenting|presented|"
    r"hospitali[sz]ed|resuscitated)"
    r"(?:\s+(?:to|into|in|at)\s+(?:(?:the|an?)\s+)?(?:[\w-]+\s+){0,2}[\w-]+)?\s*$",
    re.I,
)
_EN_POPULATION_ADJACENT_BEFORE = re.compile(rf"\b{_EN_POPULATION_NOUNS}\s*$", re.I)
_EN_POPULATION_AFTER = re.compile(rf"^\s*{_EN_POPULATION_NOUNS}\b", re.I)
#: The time elapsed since the event is a variable, not a window: "adjusting
#: for hours since sepsis onset", "the time from intubation to extubation".
#: "The first hours since sepsis onset" is a window.
_EN_ELAPSED_TIME_BEFORE = re.compile(
    r"(?<!first )(?<!initial )(?<!early )"
    r"\b(?:time|hours?|days?|minutes?|duration|interval|delay)\s*$",
    re.I,
)
_EN_ELAPSED_RELATION = re.compile(r"^(?:since|from)\b", re.I)
#: The event is negated or excluded: "not after intubation", "excluding values
#: measured after intubation".
_EN_NEGATION = re.compile(
    r"\b(?:not|never|without|excluding|exclude[sd]?|exclusion|except|"
    r"other\s+than|rather\s+than|instead\s+of)\b",
    re.I,
)
_ZH_POPULATION_AFTER = re.compile(
    r"^\s*的?\s*(?:入住|入\s*(?:ICU|重症|监护|科)|入组|入院|转入|收入|收治|"
    r"患者|病人|者|人群|病例)",
    re.I,
)
#: "自疑似感染起的时间": the time elapsed since the event; "插管后的时间窗"
#: is a window.
_ZH_ELAPSED_SINCE = re.compile(r"(?:起|开始)$")
_ZH_ELAPSED_AFTER = re.compile(r"^\s*的?\s*(?:时间|时长|间隔)(?!窗|段)")
#: "不同" (different), "不论"/"不管"/"无论" (regardless) and "无关"
#: (unrelated) negate nothing.
_ZH_NEGATION_BEFORE = re.compile(
    r"(?:排除|除外|不包括|不含|不(?![同论管])|未|非|无(?![论关]))\s*\S{0,2}$"
)
_ZH_TEXT = re.compile(r"[\u4e00-\u9fff]")


def _clause_around(text: str, start: int, end: int) -> tuple[str, str]:
    """The text of the reading's clause before and after it."""

    stops_before = [match.end() for match in _CLAUSE_STOP.finditer(text, 0, start)]
    left = text[stops_before[-1] if stops_before else 0 : start]
    stop_after = _CLAUSE_STOP.search(text, end)
    right = text[end : stop_after.start() if stop_after else len(text)]
    return left, right


def _governs_time_zero(text: str, reading: EventTimeZeroStatement) -> bool:
    """Whether a reading states when the study's time counts from.

    A duration counted from the event ("within 6 hours of intubation") is a
    window wherever it stands.  A bare mention of the event counts nothing
    from it when it qualifies the population, names the time elapsed since
    the event, or is negated or excluded.
    """

    if reading.hours is not None:
        return True
    left, right = _clause_around(text, reading.start, reading.end)
    phrase = reading.text
    if _ZH_TEXT.search(phrase):
        return not (
            _ZH_POPULATION_AFTER.search(right)
            or (_ZH_ELAPSED_SINCE.search(phrase) and _ZH_ELAPSED_AFTER.search(right))
            or _ZH_NEGATION_BEFORE.search(left[-6:])
        )
    return not (
        _EN_ADMITTED_BEFORE.search(left)
        or _EN_POPULATION_ADJACENT_BEFORE.search(left)
        or _EN_POPULATION_AFTER.search(right)
        or (
            _EN_ELAPSED_TIME_BEFORE.search(left) and _EN_ELAPSED_RELATION.search(phrase)
        )
        or _EN_NEGATION.search(" ".join(left.split()[-3:]))
    )


def event_anchored_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Where a question counts time from a clinical event, time zero or not.

    Every reading of :func:`stated_event_time_zeros`'s vocabulary is included,
    also a phrase that states no time zero (a population, the time elapsed
    since the event, a negation): an hour count inside one is never a window
    from ICU admission.  Spans are ``[start, end)``, sorted and merged.
    """

    spans: list[tuple[int, int]] = []
    for reading in _event_readings(str(text or "")):
        if spans and reading.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], reading.end))
        else:
            spans.append((reading.start, reading.end))
    return tuple(spans)


def stated_event_time_zeros(text: str) -> tuple[EventTimeZeroStatement, ...]:
    """The clinical events a question counts time from, in question order.

    Only the closed event vocabulary is read, and a condition or therapy only
    with an onset word or a duration, so a population ("patients with septic
    shock") or a status ("mortality at discharge") states no time zero.  A
    bare mention of an event in that vocabulary counts nothing from it either
    when it qualifies the population ("patients admitted after cardiac
    arrest"), names the time elapsed since the event ("adjusting for hours
    since sepsis onset", "the time from intubation to extubation"), or is
    negated or excluded ("excluding values measured after intubation").  A
    duration counted from an event is always read.  The admissions are not
    read here: ``relative_to_anchor`` owns them, and every materialized window
    already counts from one.
    """

    text = str(text or "")
    return tuple(
        reading
        for reading in _event_readings(text)
        if _governs_time_zero(text, reading)
    )


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

    # The typed constraints come first.  Historical contexts may contain the
    # exact request but predate the typed constraint projection, so the
    # question is parsed too: that recovers only an explicit phrase and never
    # invents a clinical anchor.
    question = str(context.research_question or "")
    stated: list[tuple[int, str, str]] = []
    sources = (
        (context.temporal_constraints, "temporal_constraints"),
        (TimeWindowSemanticParser().parse(question), "research_question"),
    )
    for constraints, origin in sources:
        for item in constraints:
            if item.relation not in {"relative_to_anchor", "after_event"}:
                continue
            if not str(item.anchor_event).strip():
                continue
            if origin == "temporal_constraints":
                source = f"temporal_constraints.{item.relation}"
            elif item.relation == "relative_to_anchor":
                source = "research_question.explicit_relative_anchor"
            else:
                source = "research_question.stated_event_time_zero"
            position = question.find(item.raw_text)
            stated.append((
                position if position >= 0 else len(question),
                normalise_time_anchor(item.anchor_event),
                source,
            ))
    anchors: dict[str, tuple[int, str]] = {}
    for position, anchor, source in stated:
        first = anchors.setdefault(anchor, (position, source))
        if position < first[0]:
            anchors[anchor] = (position, first[1])
    if len(anchors) == 1:
        [(anchor, (_, source))] = anchors.items()
        return anchor, source
    # A question that counts from an admission and from another event cannot
    # be honoured by windows that all count from an admission: the earliest
    # stated event is its time zero.  Two admissions alone stay unresolved.
    events = sorted(
        (position, anchor, source)
        for anchor, (position, source) in anchors.items()
        if anchor not in ADMISSION_TIME_ZEROS
    )
    if events:
        _, anchor, source = events[0]
        return anchor, source
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
    if (
        comparison_anchor is None
        and declared in ADMISSION_TIME_ZEROS
        and observation == declared
    ):
        # An admission is no clinical definition: an exposure with none,
        # whose window counts its hours from the admission the study declares,
        # is measured from that time zero.  A disease or event anchor still
        # needs the owner-issued definition above.
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
        events = stated_event_time_zeros(text)
        anchored = event_anchored_spans(text)
        no_time_zero = [item for item in _event_readings(text) if item not in events]
        for relation, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                if relation == "first_window" and any(
                    start <= match.start() < end for start, end in anchored
                ):
                    # "the first 24 h after sepsis onset" counts from that
                    # event, whether or not it is the study's time zero; ICU
                    # admission is not its default here.
                    continue
                if relation == "relative_to_anchor" and any(
                    item.start <= match.start() < item.end for item in no_time_zero
                ):
                    # "the time from suspected infection onset" names the time
                    # elapsed since the event, not the study's time zero.
                    continue
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
        for event in events:
            out.append(
                TemporalConstraint(
                    raw_text=event.text,
                    relation="after_event",
                    anchor_event=event.anchor,
                    start_hours=0.0 if event.hours is not None else None,
                    end_hours=event.hours,
                    executable_repr=_render_constraint_repr(
                        relation="after_event",
                        anchor=event.anchor,
                        hours=event.hours,
                        concept=None,
                    ),
                )
            )
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
