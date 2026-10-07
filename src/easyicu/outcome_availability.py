"""Dependency-neutral authority for database-specific outcome support.

It also owns the closed fixed-horizon mortality vocabulary: each ``mort_<h>d``
event concept is paired with its ``followup_days_<h>d`` event/censoring time,
measured in days from ICU admission and administratively censored at the
horizon.  The outcome materializer and the Web survival projection read the
pairing here so neither can drift from the other.  The horizons a question
states for death or survival are read here too, so a requested horizon is
compared with an endpoint's in one vocabulary.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# A fixed horizon needs post-discharge follow-up of survivors as well as
# dates of death.  MIMIC-IV censors a null date of death one year after the
# last hospital discharge, and SICdb records each case's survival observation
# time.  MIMIC-III and AmsterdamUMCdb record dates of death but no follow-up
# of survivors, so a horizon there would be defined for the dead only.
FOLLOWUP_OUTCOME_DATABASES = frozenset({"miiv", "miiv_demo", "sic", "sic_demo"})
# Reserved compatibility sets. They remain empty until an owner can prove
# complete ICU/ventilation trajectories and endpoint-specific day-28 survival.
MIMIC_READMISSION_DATABASES = frozenset()
ICU_FREE_DAY_DATABASES = frozenset()
EICU_VENTILATOR_DAY_DATABASES = frozenset()

OUTCOME_CONCEPT_SUPPORTED_DATABASES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "mort_28d": FOLLOWUP_OUTCOME_DATABASES,
        "mort_90d": FOLLOWUP_OUTCOME_DATABASES,
        "mort_365d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_28d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_90d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_365d": FOLLOWUP_OUTCOME_DATABASES,
        "icu_free_days_28": ICU_FREE_DAY_DATABASES,
        "icu_readmission": MIMIC_READMISSION_DATABASES,
        "vent_free_days_28": EICU_VENTILATOR_DAY_DATABASES,
    }
)


@dataclass(frozen=True, slots=True)
class FixedHorizonMortalityEndpoint:
    """One ``mort_<h>d`` event concept with its paired follow-up time concept."""

    event_concept: str
    followup_concept: str
    horizon_days: int
    #: Time zero shared by the event and follow-up concepts.
    time_origin: str = "icu_admission"
    #: Physical unit of ``followup_concept``.
    followup_unit: str = "days"
    #: Piecewise-interval cutpoints (days) a time-varying hazard-ratio audit
    #: uses for this horizon; each lies strictly inside the horizon.
    time_varying_cutpoints_days: tuple[int, ...] = ()

    @property
    def censoring_rule(self) -> str:
        return (
            f"Observed death time or administrative censoring at {self.horizon_days} "
            "days; exclude rows without documented horizon support."
        )


FIXED_HORIZON_MORTALITY_ENDPOINTS: Mapping[str, FixedHorizonMortalityEndpoint] = (
    MappingProxyType(
        {
            "mort_28d": FixedHorizonMortalityEndpoint(
                event_concept="mort_28d",
                followup_concept="followup_days_28d",
                horizon_days=28,
                time_varying_cutpoints_days=(7, 14),
            ),
            "mort_90d": FixedHorizonMortalityEndpoint(
                event_concept="mort_90d",
                followup_concept="followup_days_90d",
                horizon_days=90,
                time_varying_cutpoints_days=(7, 14, 28),
            ),
            "mort_365d": FixedHorizonMortalityEndpoint(
                event_concept="mort_365d",
                followup_concept="followup_days_365d",
                horizon_days=365,
                time_varying_cutpoints_days=(7, 14, 28, 90),
            ),
        }
    )
)


def fixed_horizon_mortality_endpoint(
    concept_id: str,
) -> FixedHorizonMortalityEndpoint | None:
    """Return the closed event/follow-up pairing for one ``mort_<h>d`` concept."""

    return FIXED_HORIZON_MORTALITY_ENDPOINTS.get(str(concept_id).strip())


#: The days one unit of a stated horizon spans.  A month is 30 or 31 days:
#: "3-month" (90 to 93 days) admits the 90-day endpoint and "12-month" the
#: 365-day one, while "1-month" admits no closed horizon (28 days is four
#: weeks, not a month).  An hour is exact: a horizon in hours admits only the
#: endpoint of that many hours.
_UNIT_DAYS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {"day": (1, 1), "week": (7, 7), "month": (30, 31), "year": (365, 366)}
)
_HOURS_PER_DAY = 24


@dataclass(frozen=True, slots=True)
class StatedHorizon:
    """A horizon a text states for death or survival, in the unit it was stated."""

    count: int
    unit: str

    @property
    def adjective(self) -> str:
        """``28-day``, ``48-hour``: the horizon as it qualifies an endpoint."""

        return f"{self.count}-{self.unit}"

    @property
    def noun(self) -> str:
        return f"{self.count} {self.unit}" + ("" if self.count == 1 else "s")

    @property
    def semantic_key(self) -> str:
        """``mortality_28d``, ``mortality_48h``; ``mortality_1year`` in other units."""

        suffix = {"day": "d", "hour": "h"}.get(self.unit, self.unit)
        return f"mortality_{self.count}{suffix}"

    def admits(self, days: float) -> bool:
        if self.unit == "hour":
            return self.count == days * _HOURS_PER_DAY
        low, high = _UNIT_DAYS[self.unit]
        return self.count * low <= days <= self.count * high


@dataclass(frozen=True, slots=True)
class StatedHorizonMention:
    """One place a text states a horizon: ``text[start:end]``."""

    horizon: StatedHorizon
    start: int
    end: int


_NUMBER_WORDS: Mapping[str, int] = MappingProxyType(
    {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }
)
_ZH_DIGITS: Mapping[str, int] = MappingProxyType(
    {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
)
_ZH_PLACES: Mapping[str, int] = MappingProxyType({"十": 10, "百": 100})
_UNIT_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour", "h": "hour", "小时": "hour",
        "day": "day", "days": "day", "d": "day", "天": "day", "日": "day",
        "week": "week", "weeks": "week", "wk": "week", "wks": "week", "周": "week", "星期": "week",
        "month": "month", "months": "month", "mo": "month", "mos": "month", "个月": "month",
        "year": "year", "years": "year", "yr": "year", "yrs": "year", "年": "year",
    }
)
_COUNT = r"(?P<count>\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_UNIT = r"(?P<unit>days?|d|weeks?|wks?|months?|mos?|years?|yrs?)"
_ENDPOINT = r"(?:mortality|survival|deaths?|die[sd]?|dying|alive|survived?|follow(?:ed)?[\s-]*up)"
# Words that may stand between a horizon and its endpoint ("28-day all-cause
# mortality", "90 days of follow-up"); any other word means the number
# qualifies something else ("a 7-day course and survival").
_QUALIFIER = r"(?:all[\s-]+cause|in[\s-]+hospital|hospital|icu|overall|crude|cumulative|of)"
# "mortality at 28 days", "death within 48 hours", "mortality in the first
# 48 hours".  "In" relates an endpoint to a horizon only as "in the first".
_RELATION = (
    r"(?:(?:at|by|to|through|within|until|over|for|of)[\s-]+(?:the[\s-]+(?:first[\s-]+)?)?"
    r"|(?:in|during)[\s-]+the[\s-]+first[\s-]+)"
)
_HOUR_RELATION = (
    r"(?:(?:at|by|to|through|within|until|over)[\s-]+(?:the[\s-]+(?:first[\s-]+)?)?"
    r"|(?:in|during)[\s-]+the[\s-]+first[\s-]+)"
)
# A Chinese numeral is read whole ("二十八天" is 28 days, never 8).
_COUNT_ZH = r"(?<![零一二两三四五六七八九十百])(?P<count>\d{1,4}|[一二两三四五六七八九十百]{1,6})"
_UNIT_ZH = r"(?P<unit>天|日|d|周|星期|个月|年)"
_ENDPOINT_ZH = r"(?:死亡|病死|生存|存活)"
# "28 天院内死亡", "48 小时 ICU 死亡": a setting may stand between the two.
_QUALIFIER_ZH = r"(?:全因|院内|住院|ICU)"
# Hours are read only next to a mortality, death or survival noun or a verb
# of dying ("48-hour mortality", "death within 24 hours of ICU admission",
# "died within 48 hours", "48 小时内死亡"): "alive at 24 hours", "survived the
# first 24 hours" and "did not die within 24 hours" name the landmark a study
# starts from.  Deaths an exclusion names are the cohort's (``_excluded``).
_HOUR = r"(?P<unit>hours?|hrs?|h)"
_HOUR_ENDPOINT = r"(?:mortality|deaths?|survival)"
_HOUR_ZH = r"(?P<unit>小时|h)"
# Several horizons may be listed for one endpoint: "48-hour and 28-day
# mortality", "28-, 90- and 180-day mortality", "mortality at 24 and 48
# hours", "survival at 28 days and one year", "28 天和 90 天死亡率".  A member
# without a unit of its own takes the unit of the horizon it is listed with.
_LIST_COUNT = r"(?:\d{1,4}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_LIST_UNIT = r"(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?|h|d)"
_LIST_JOIN = r"(?:\s*,\s*(?:(?:and/or|and|or)\s+)?|\s+(?:and/or|and|or)\s+)"
#: No list names more horizons than this; the bound keeps a long run of
#: numbers linear to read.
_LIST_LIMIT = 8
# Before an endpoint's horizon a member is an adjective ("48-hour") or
# suspended ("28-").  A noun is not listed: in "between lactate within 24
# hours and 28-day mortality" the 24 hours are the exposure's.
_PREFIX_MEMBERS = (
    rf"(?P<members>(?:\b{_LIST_COUNT}(?:-(?:hour|hr|h|day|d|week|wk|month|mo|year|yr)\b|-"
    rf"|\s(?:hour|day|week|month|year)\b){_LIST_JOIN}){{1,{_LIST_LIMIT}}})"
)
# After an endpoint and its relation, members may precede the horizon ("at 24
# and 48 hours") or follow it ("at 28 days and one year").  Either list ends
# with "and" or "or": a comma alone leaves "within 28 days, 18 years or
# older" a horizon of 28 days.
_POSTFIX_MEMBER = rf"{_LIST_COUNT}(?:[\s-]*{_LIST_UNIT})?"
_POSTFIX_MEMBERS = (
    rf"(?P<members>(?:{_POSTFIX_MEMBER}\s*,\s*){{0,{_LIST_LIMIT - 1}}}"
    rf"{_POSTFIX_MEMBER}\s*,?\s+(?:and/or|and|or)\s+)"
)
_POSTFIX_TAIL_MEMBER = rf"(?:{_LIST_COUNT}[\s-]*{_LIST_UNIT}|day[\s-]*\d{{1,4}})\b"
_POSTFIX_TAIL = (
    rf"(?P<tail>(?:\s*,\s*{_POSTFIX_TAIL_MEMBER}){{0,{_LIST_LIMIT - 1}}}"
    rf"\s*,?\s+(?:and/or|and|or)\s+{_POSTFIX_TAIL_MEMBER})"
)
# A Chinese member carries its own unit, or is listed with "、" ("28、90 天").
_ZH_LIST_JOIN = r"\s*(?:以及|和|或|及|、|，|,)\s*"
_ZH_PREFIX_MEMBERS = (
    r"(?P<members>(?:(?<![零一二两三四五六七八九十百第\d])(?:\d{1,4}|[一二两三四五六七八九十百]{1,6})\s*"
    rf"(?:(?:小时|天|日|周|星期|个月|年|h|d){_ZH_LIST_JOIN}|、\s*)){{1,{_LIST_LIMIT}}})"
)
_STATED_HORIZON_PATTERNS = (
    # "28-day mortality", "one-year survival", "90 days of follow-up".
    re.compile(
        rf"{_PREFIX_MEMBERS}?(?P<head>\b{_COUNT}[\s-]*{_UNIT}\b)"
        rf"(?:[\s-]+{_QUALIFIER}){{0,2}}[\s-]+{_ENDPOINT}",
        re.IGNORECASE,
    ),
    # "mortality at 28 days", "survival to day 28", "followed up for one year".
    re.compile(
        rf"\b{_ENDPOINT}[\s-]+(?:rates?[\s-]+)?{_RELATION}{_POSTFIX_MEMBERS}?"
        rf"(?P<head>day[\s-]*(?P<day>\d{{1,4}})\b|{_COUNT}[\s-]*{_UNIT}\b){_POSTFIX_TAIL}?",
        re.IGNORECASE,
    ),
    # "day-28 mortality".
    re.compile(rf"(?P<head>\bday[\s-]*(?P<day>\d{{1,4}}))[\s-]+{_ENDPOINT}", re.IGNORECASE),
    # "28 天死亡", "一年生存率", "三个月内死亡", "28 天院内死亡".
    re.compile(
        rf"{_ZH_PREFIX_MEMBERS}?(?P<head>{_COUNT_ZH}\s*{_UNIT_ZH})"
        rf"\s*(?:以内|内)?\s*的?\s*(?:{_QUALIFIER_ZH}\s*)?{_ENDPOINT_ZH}"
    ),
    # "第 28 天死亡".
    re.compile(rf"(?P<head>第\s*(?P<day>\d{{1,4}})\s*[天日])\s*的?\s*{_ENDPOINT_ZH}"),
    # "随访一年", "随访至第 90 天".
    re.compile(rf"随访\s*(?:至|到|满)?\s*第?\s*(?P<head>{_COUNT_ZH}\s*{_UNIT_ZH})"),
    # "48-hour mortality", "72 h in-hospital mortality", "24-hour survival".
    re.compile(
        rf"{_PREFIX_MEMBERS}?(?P<head>\b{_COUNT}[\s-]*{_HOUR}\b)"
        rf"(?:[\s-]+{_QUALIFIER}){{0,2}}[\s-]+{_HOUR_ENDPOINT}\b",
        re.IGNORECASE,
    ),
    # "mortality within 48 hours", "death within 24 h of ICU admission",
    # "died within 48 hours".
    re.compile(
        rf"(?<!not )(?<!n't )(?<!never )\b(?:{_HOUR_ENDPOINT}|die[sd]?|dying)[\s-]+(?:rates?[\s-]+)?"
        rf"{_HOUR_RELATION}{_POSTFIX_MEMBERS}?(?P<head>{_COUNT}[\s-]*{_HOUR}\b){_POSTFIX_TAIL}?",
        re.IGNORECASE,
    ),
    # "48 小时死亡", "48 小时内死亡", "四十八小时病死率", "48 小时院内死亡".
    re.compile(
        rf"{_ZH_PREFIX_MEMBERS}?(?P<head>{_COUNT_ZH}\s*{_HOUR_ZH})"
        rf"\s*(?:以内|内)?\s*的?\s*(?:{_QUALIFIER_ZH}\s*)?(?:死亡|病死)"
    ),
)
_LIST_MEMBER = re.compile(
    r"\bday[\s-]*(?P<day>\d{1,4})\b"
    r"|(?P<count>\d{1,4}|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"
    r"|[一二两三四五六七八九十百]{1,6})"
    r"(?:[\s-]*(?P<unit>(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?|h|d)(?![a-z])"
    r"|小时|天|日|周|星期|个月|年))?",
    re.IGNORECASE,
)

# A horizon an exclusion names is the cohort's, not the endpoint's: "excluding
# deaths within 24 hours", "deaths within the first 24 hours were excluded",
# "排除 24 小时内死亡的患者".  An exclusion governs the first horizon a few words
# after it in its clause, or the horizon its passive predicate follows.
_CLAUSE_MARKS = ".;:,!?。；：，！？\n"
_EXCLUSION_CUE = re.compile(
    r"\b(?:exclu(?:de|des|ded|ding|sion)|except|omit(?:s|ted|ting)?)\b|排除|除外|剔除|不纳入|不包括",
    re.IGNORECASE,
)
_PASSIVE_BEFORE = re.compile(r"\b(?:were|was|are|is|be|been|being)\s+$", re.IGNORECASE)
_EXCLUSION_GAP_WORDS = 4
_EXCLUSION_GAP_CHARS = 24
#: How far before a horizon its clause and a governing exclusion are sought.
_EXCLUSION_REACH = 64
_EXCLUDED_AFTER = re.compile(
    r"^(?:\s+(?:of|after|from|since)\s+[\w\s-]{0,30}?)?\s+(?:were|was|are|is|be|been|being)\s+"
    r"(?:excluded|omitted|removed|not\s+included|ineligible|not\s+eligible)\b"
    r"|^[^，。；,;]{0,4}?(?:被|予以|均|则)?(?:排除|除外|剔除|不纳入|不予纳入)",
    re.IGNORECASE,
)
# A labelled list of eligibility criteria names the cohort's horizons, not the
# endpoint's: "Exclusion criteria: death within 24 hours of ICU admission.
# Outcome: in-hospital mortality.", "排除标准：入ICU 24小时内死亡。结局：院内死亡。".
# A label with content on its own line governs it up to the end of the
# sentence; a label that ends its line governs the lines below it up to a
# blank line.  Either ends at the next label.
_SECTION_LABEL = re.compile(
    r"(?:^|(?<=[\n.;!?。；！？,，]))[ \t]*(?:[-*•][ \t]*)?"
    r"(?P<label>[A-Za-z][A-Za-z ()/-]{0,40}?|[一-鿿]{1,8})[ \t]*[:：]"
)
_CRITERIA_LABEL = re.compile(
    r"\b(?:in|ex)clusions?\b|\bexcluded\b|\beligibility\b|排除|纳入|入选", re.IGNORECASE
)
_INLINE_SECTION_END = re.compile(r"\n|[.!?](?=\s|$)|[。！？]")
_BLOCK_SECTION_END = re.compile(r"\n[ \t]*\n")


def _zh_count(numeral: str) -> int | None:
    """``二十八`` is 28, ``三百六十五`` 365, ``十二`` 12; None when malformed."""

    total, digit, last_place = 0, None, None
    for char in numeral:
        if char in _ZH_DIGITS:
            if digit is not None:
                return None
            digit = _ZH_DIGITS[char]
        else:
            place = _ZH_PLACES[char]
            if last_place is not None and place >= last_place:
                return None
            total += (1 if digit is None else digit) * place
            digit, last_place = None, place
    return total + (digit or 0)


def _criteria_sections(text: str) -> tuple[tuple[int, int], ...]:
    """Where labelled lists of eligibility criteria stand, as ``[start, end)``."""

    labels = list(_SECTION_LABEL.finditer(text))
    sections = []
    for index, label in enumerate(labels):
        if not _CRITERIA_LABEL.search(label.group("label")):
            continue
        begin = label.end()
        limit = labels[index + 1].start() if index + 1 < len(labels) else len(text)
        line_end = text.find("\n", begin, limit)
        inline = text[begin : limit if line_end == -1 else line_end].strip()
        stop = (_INLINE_SECTION_END if inline else _BLOCK_SECTION_END).search(text, begin, limit)
        sections.append((begin, stop.start() if stop else limit))
    return tuple(sections)


def _excluded(
    text: str,
    start: int,
    end: int,
    floor: int,
    sections: tuple[tuple[int, int], ...] = (),
) -> bool:
    """Whether an exclusion governs the horizon stated at ``text[start:end]``.

    ``floor`` is where the previous horizon ends: an exclusion before it
    governs that horizon, not this one.  ``sections`` are the labelled lists
    of eligibility criteria (``_criteria_sections``): every horizon in one is
    the cohort's.
    """

    if any(begin <= start < stop for begin, stop in sections):
        return True
    left = max(floor, start - _EXCLUSION_REACH)
    left = max([left] + [text.rfind(mark, left, start) + 1 for mark in _CLAUSE_MARKS])
    for cue in _EXCLUSION_CUE.finditer(text, left, start):
        gap = text[cue.end():start]
        # "Deaths within 24 hours were excluded and 90-day mortality ...":
        # a passive exclusion governs what precedes it.
        if _PASSIVE_BEFORE.search(text, left, cue.start()):
            continue
        if len(gap.split()) <= _EXCLUSION_GAP_WORDS and len(gap) <= _EXCLUSION_GAP_CHARS:
            return True
    return _EXCLUDED_AFTER.match(text[end:end + 80]) is not None


def _stated_horizon(match: re.Match[str]) -> StatedHorizon | None:
    groups = match.groupdict()
    if groups.get("day"):
        count, unit = int(groups["day"]), "day"
    else:
        raw = groups["count"].lower()
        count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw) or _zh_count(raw)
        unit = _UNIT_WORDS[groups["unit"].lower()]
    return StatedHorizon(count=count, unit=unit) if count else None


def _member_horizon(member: re.Match[str], unit: str | None) -> StatedHorizon | None:
    """A listed horizon; one without a unit of its own takes ``unit``."""

    groups = member.groupdict()
    if groups.get("day"):
        return StatedHorizon(count=int(groups["day"]), unit="day")
    raw = groups["count"].lower()
    count = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw) or _zh_count(raw)
    stated = groups.get("unit")
    unit = _UNIT_WORDS[stated.lower()] if stated else unit
    return StatedHorizon(count=count, unit=unit) if count and unit else None


def _match_mentions(match: re.Match[str]) -> tuple[StatedHorizonMention, ...]:
    """Each horizon one phrase states, at its own place in the text.

    The first mention reaches back to the phrase's start and the last one on
    to its end, so exactly one of them covers the endpoint word, on whichever
    side it stands ("28- and 90-day mortality", "mortality at 24 and 48
    hours").
    """

    head = _stated_horizon(match)
    if head is None:
        return ()
    groups = match.groupdict()
    items = sorted(
        [
            (head, *match.span("head")),
            *(
                (horizon, match.start(name) + member.start(), match.start(name) + member.end())
                for name, unit in (("members", head.unit), ("tail", None))
                if groups.get(name)
                for member in _LIST_MEMBER.finditer(groups[name])
                if (horizon := _member_horizon(member, unit)) is not None
            ),
        ],
        key=lambda item: item[1],
    )
    items[0] = (items[0][0], match.start(), items[0][2])
    items[-1] = (items[-1][0], items[-1][1], match.end())
    return tuple(
        StatedHorizonMention(horizon=horizon, start=start, end=end)
        for horizon, start, end in items
    )


def stated_mortality_horizon_mentions(text: str) -> tuple[StatedHorizonMention, ...]:
    """Every place ``text`` states a horizon for death or survival, in text order.

    A horizon is read only next to its endpoint word: "28-day mortality",
    "survival to day 90", "death within 28 days", "one-year survival",
    "48-hour mortality", "died within 48 hours", "mortality in the first 48
    hours", "90 天死亡", "随访一年", and each horizon of a list ("48-hour and
    28-day mortality", "mortality at 24 and 48 hours").  Days that qualify
    something else ("a 7-day course") state no horizon, nor do hours that name
    a landmark ("alive at 24 hours"), nor a horizon an exclusion names
    ("deaths within 24 hours were excluded", "Exclusion criteria: death
    within 24 hours").
    """

    source = str(text or "")
    matches = sorted(
        (
            match
            for pattern in _STATED_HORIZON_PATTERNS
            for match in pattern.finditer(source)
            if _stated_horizon(match) is not None
        ),
        key=lambda match: (match.start(), match.end()),
    )
    sections = _criteria_sections(source)
    # A phrase's floor is the end of the last phrase before it.  Two patterns
    # may read one phrase ("mortality at 48 hours and 28 days"): it has one
    # floor, so one exclusion governs both readings.
    floors, ended, floor = [], [], 0
    for match in matches:
        while ended and ended[0] <= match.start():
            floor = max(floor, heapq.heappop(ended))
        floors.append(floor)
        heapq.heappush(ended, match.end())
    mentions = dict.fromkeys(
        mention
        for match, floor in zip(matches, floors)
        if not _excluded(source, match.start(), match.end(), floor, sections)
        for mention in _match_mentions(match)
    )
    return tuple(sorted(mentions, key=lambda mention: (mention.start, mention.end)))


def mortality_horizon_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Where ``text`` names a horizon for death or survival, read or excluded.

    The hours or days of such a phrase time an endpoint or an exclusion
    ("48-hour mortality", "excluding deaths within 24 hours"); they never
    state an exposure or observation window.
    """

    source = str(text or "")
    return tuple(
        sorted(
            (match.start(), match.end())
            for pattern in _STATED_HORIZON_PATTERNS
            for match in pattern.finditer(source)
            if _stated_horizon(match) is not None
        )
    )


def stated_mortality_horizons(text: str) -> tuple[StatedHorizon, ...]:
    """The distinct horizons ``text`` states for death or survival, in text order."""

    return tuple(dict.fromkeys(mention.horizon for mention in stated_mortality_horizon_mentions(text)))


def fixed_horizon_mortality_endpoint_stated_by(
    horizon: StatedHorizon,
) -> FixedHorizonMortalityEndpoint | None:
    """The one closed fixed-horizon endpoint a stated horizon admits, if exactly one."""

    admitted = [
        endpoint
        for endpoint in FIXED_HORIZON_MORTALITY_ENDPOINTS.values()
        if horizon.admits(endpoint.horizon_days)
    ]
    return admitted[0] if len(admitted) == 1 else None


@dataclass(frozen=True, slots=True)
class OutcomeConceptUnavailability:
    """A known structural absence, distinct from missing observed values."""

    concept_id: str
    database: str
    reason_code: str
    supported_databases: tuple[str, ...]


def structural_outcome_unavailability(
    concept_id: str,
    database: str,
) -> OutcomeConceptUnavailability | None:
    """Return a receipt only for a known unsupported concept/database pair."""

    concept = str(concept_id).strip()
    normalized_database = str(database).strip().lower()
    supported = OUTCOME_CONCEPT_SUPPORTED_DATABASES.get(concept)
    if supported is None or normalized_database in supported:
        return None
    return OutcomeConceptUnavailability(
        concept_id=concept,
        database=normalized_database,
        reason_code="outcome_concept_structurally_unavailable",
        supported_databases=tuple(sorted(supported)),
    )


__all__ = [
    "EICU_VENTILATOR_DAY_DATABASES",
    "FIXED_HORIZON_MORTALITY_ENDPOINTS",
    "FOLLOWUP_OUTCOME_DATABASES",
    "FixedHorizonMortalityEndpoint",
    "ICU_FREE_DAY_DATABASES",
    "MIMIC_READMISSION_DATABASES",
    "OUTCOME_CONCEPT_SUPPORTED_DATABASES",
    "OutcomeConceptUnavailability",
    "StatedHorizon",
    "StatedHorizonMention",
    "fixed_horizon_mortality_endpoint",
    "fixed_horizon_mortality_endpoint_stated_by",
    "mortality_horizon_spans",
    "stated_mortality_horizon_mentions",
    "stated_mortality_horizons",
    "structural_outcome_unavailability",
]
