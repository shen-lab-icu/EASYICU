"""Dependency-neutral authority for database-specific outcome support.

It covers every output of the outcomes loader (``scores/outcomes.py``),
including ``icu_stay_is_readmission``, a stay attribute read from the same
ICU stay sequence as the ``icu_readmission`` outcome.

It also owns the closed fixed-horizon mortality vocabulary: each ``mort_<h>d``
event concept is paired with its ``followup_days_<h>d`` event/censoring time,
measured in days from ICU admission and administratively censored at the
horizon.  The outcome materializer and the Web survival projection read the
pairing here so neither can drift from the other.  The horizons a question
states for death or survival are read here too, so a requested horizon is
compared with an endpoint's in one vocabulary, and so is the setting (ICU or
hospital) a text states for death: a question naming ICU mortality asks for the
native ``icu_death`` endpoint, never the dictionary's hospital ``death``.
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
# ICU readmission needs every ICU stay of a hospitalisation, in order, from
# the source's own stay table: MIMIC-IV icustays grouped by hadm_id, eICU unit
# stays grouped by patienthealthsystemstayid.
ICU_READMISSION_DATABASES = frozenset({"miiv", "miiv_demo", "eicu", "eicu_demo"})
# ICU-free days to day 28 need every ICU stay of a patient in that window and
# survival to day 28: MIMIC-IV has both (icustays by subject_id, and the
# one-year death follow-up above).
ICU_FREE_DAY_DATABASES = frozenset({"miiv", "miiv_demo"})
# Reserved: empty until an owner can prove complete ventilation episodes and
# day-28 survival.
EICU_VENTILATOR_DAY_DATABASES = frozenset()

OUTCOME_CONCEPT_SUPPORTED_DATABASES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "icu_death": frozenset({"mimic", "mimic_demo", "miiv", "miiv_demo", "eicu", "eicu_demo", "hirid", "aumc", "sic"}),
        "mort_28d": FOLLOWUP_OUTCOME_DATABASES,
        "mort_90d": FOLLOWUP_OUTCOME_DATABASES,
        "mort_365d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_28d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_90d": FOLLOWUP_OUTCOME_DATABASES,
        "followup_days_365d": FOLLOWUP_OUTCOME_DATABASES,
        "icu_free_days_28": ICU_FREE_DAY_DATABASES,
        "icu_readmission": ICU_READMISSION_DATABASES,
        "icu_stay_is_readmission": ICU_READMISSION_DATABASES,
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
# mortality", "90 days of follow-up", "28-day restricted mean survival time");
# any other word means the number qualifies something else ("a 7-day course
# and survival").
_QUALIFIER = (
    r"(?:all[\s-]+cause|in[\s-]+hospital|hospital|icu|overall|crude|cumulative|of"
    r"|restricted|mean|average)"
)
# Between an endpoint and its relation: "mortality rate at 28 days", "survival
# time within 28 days", "survive on average within 28 days", "restricted mean
# survival time (RMST) at 28 days".
_ENDPOINT_TAIL = (
    r"(?:(?:rates?|times?|on[\s-]+average)[\s-]+)?(?:\(\s*rmst\s*\)[\s-]*)?"
)
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
# "28 天内平均存活多少天", "28 天限制性平均生存时间": the mean survival over a
# horizon states that horizon.
_MEAN_SURVIVAL_ZH = r"(?:限制性?)?平均\s*(?:生存|存活)"
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
        rf"\b{_ENDPOINT}[\s-]+{_ENDPOINT_TAIL}{_RELATION}{_POSTFIX_MEMBERS}?"
        rf"(?P<head>day[\s-]*(?P<day>\d{{1,4}})\b|{_COUNT}[\s-]*{_UNIT}\b){_POSTFIX_TAIL}?",
        re.IGNORECASE,
    ),
    # "day-28 mortality".
    re.compile(rf"(?P<head>\bday[\s-]*(?P<day>\d{{1,4}}))[\s-]+{_ENDPOINT}", re.IGNORECASE),
    # "28 天死亡", "一年生存率", "三个月内死亡", "28 天院内死亡", "28 天内平均存活".
    re.compile(
        rf"{_ZH_PREFIX_MEMBERS}?(?P<head>{_COUNT_ZH}\s*{_UNIT_ZH})"
        rf"\s*(?:以内|内)?\s*的?\s*(?:{_QUALIFIER_ZH}\s*)?(?:{_ENDPOINT_ZH}|{_MEAN_SURVIVAL_ZH})"
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
    "survive on average within 28 days", "28 天内平均存活",
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


#: Words that ask how long patients lived on average: a restricted mean
#: survival time.  「28 天内平均存活多少天」, 「限制性平均生存时间」, 「存活天数的
#: 平均值」, "RMST", "restricted mean survival time", "mean survival time",
#: "how many days do patients survive on average", "mean days alive", "days
#: alive within 28 days".  A mean survival is a time-to-event estimand, never a
#: share of deaths: 「28 天内平均存活」 asks how long patients lived within 28
#: days, not how many died by day 28.  A rate or a probability (「平均存活率」,
#: "average survival probability") is a share; "survivors were on average
#: older" describes survivors, and so does a relative clause ("patients who
#: survived, on average, spent fewer days ventilated"); "days alive outside
#: the hospital" or "days alive and free of ventilation" counts other days.
#: None is one.
MEAN_SURVIVAL_PATTERN = re.compile(
    r"(?:限制性?)?平均\s*(?:生存|存活)(?:时间|天数|期)?(?!\s*率)"
    r"|(?:生存|存活)\s*(?:时间|天数|期)\s*的?\s*(?:均值|平均值|平均数)"
    r"|\brmst\b|\brestricted[\s-]+mean(?:[\s-]+survival(?:[\s-]+times?)?)?\b"
    r"|\b(?:mean|average)[\s-]+(?:survival(?:[\s-]+times?)?|time[\s-]+alive)\b"
    r"(?![\s-]+(?:rates?|proportions?|probabilit(?:y|ies)|chances?|percentages?)\b)"
    # The verb, never "survivors", with no copula between: "survive on
    # average", "lived, on average,", "alive on average".  It asks a mean
    # survival only beside a duration (``_DURATION_ASKED``).
    r"|(?P<on_average>\b(?:surviv(?:e|es|ed|ing)|li(?:ve|ves|ved|ving)|alive)\b"
    r"(?:[\s,]+(?!(?:is|are|was|were|be|been|being)\b)[a-z]+){0,2}?[\s,]+on[\s-]+average\b)"
    r"|\b(?:mean|average)[\s-]+(?:number[\s-]+of[\s-]+)?days?[\s-]+alive\b"
    r"(?![\s-]+(?:and|&|free|out|outside|without|off|at|in)\b)"
    r"|\bdays?[\s-]+alive[\s-]+(?:within|by|through|until|over|during|to)[\s-]+"
    r"(?:the[\s-]+first[\s-]+|day[\s-]+)?\d+(?:[\s-]*days?)?\b",
    re.IGNORECASE,
)
# A duration asked in the sentence: "how long", "how many days", "days", "time".
_DURATION_ASKED = re.compile(
    r"\bhow\s+long\b|\bdays?\b|\btime\b|\bduration\b|多久|多长时间|几天|多少天",
    re.IGNORECASE,
)
# The verb of "on average" in a relative clause describes survivors ("patients
# who survived, on average, spent …", "those alive on average had …"); it asks
# no mean survival, whatever other days the sentence counts.
_SURVIVORS_DESCRIBED_BEFORE = re.compile(
    r"\b(?:who|whom|that|which|those)\s+(?:(?:had|have|has|were|was|are|is|been|still)\s+){0,2}$",
    re.IGNORECASE,
)
# A mean survival the clause declines ("rather than RMST", 「不用 RMST」).
_MEAN_SURVIVAL_DECLINED = re.compile(
    r"(?:\bnot|n't|\bno|\bnever|\bwithout|\brather\s+than|\binstead\s+of"
    r"|不|无需|而不是|而非|并非|代替|取代)"
    r"(?:\s+(?:an?|the|any|use|using|report|reporting|estimat\w*|as|by|of|a))*\s*"
    r"(?:用|以|采用|使用|报告|计算|估计)?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StatedMeanSurvival:
    """Where a text asks a restricted mean survival time, in its words."""

    start: int
    end: int
    evidence: str


def stated_mean_survival_mentions(text: str) -> tuple[StatedMeanSurvival, ...]:
    """Every place ``text`` asks how long patients lived on average, in text order.

    The words of :data:`MEAN_SURVIVAL_PATTERN`, unless the clause declines
    them ("use logistic regression rather than RMST", 「不用 RMST」).  A
    horizon the same words state is read by
    :func:`stated_mortality_horizon_mentions`; this says the estimand.
    """

    source = str(text or "")
    mentions = []
    for match in MEAN_SURVIVAL_PATTERN.finditer(source):
        if match.group("on_average") is not None:
            if _SURVIVORS_DESCRIBED_BEFORE.search(source[max(0, match.start() - 40):match.start()]):
                continue
            sentence = re.split(r"[;。；！？!?\n]", source[:match.start()])[-1] + re.split(
                r"[;。；！？!?\n]", source[match.start():]
            )[0]
            if _DURATION_ASKED.search(sentence) is None:
                continue
        head = source[max(0, match.start() - 24):match.start()]
        head = re.split(r"[,;:，。；：！？!?\n]", head)[-1]
        if _MEAN_SURVIVAL_DECLINED.search(head):
            continue
        mentions.append(StatedMeanSurvival(match.start(), match.end(), match.group(0)))
    return tuple(mentions)


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


# Where death is counted.  ICU mortality (death before ICU discharge, the
# native ``icu_death`` endpoint) and hospital mortality (death before hospital
# discharge, the dictionary's generic ``death``) are different endpoints: a
# question naming one is never answered with the other unless a reviewed plan
# states the substitution.
ICU_DEATH = "icu_death"
HOSPITAL_DEATH = "hospital_death"
# "ICU", a unit's own ICU ("MICU", "SICU", "CVICU", "neuro-ICU"), or the
# spelled-out unit.
_ICU_PLACE = (
    r"(?:(?:m|s|c|cv|ct|n|p|t|neuro[\s-]*)?icu"
    r"|intensive[\s-]+care(?:[\s-]+units?)?|critical[\s-]+care(?:[\s-]+units?)?)"
)
# "the medical ICU", "a surgical intensive care unit".
_ICU_KIND = (
    r"(?:(?:medical|surgical|cardiac|cardiothoracic|cardiovascular|coronary|neuro(?:logical)?|"
    r"trauma|mixed|general|respiratory|paediatric|pediatric|neonatal)[\s-]+)?"
)
_ICU_PLACE_ZH = r"(?:ICU|重症监护(?:病房|室)?|重症病房|监护病房|监护室)"
_SETTING_NOUN = r"(?:mortality|deaths?|case[\s-]+fatality)"
_SETTING_DEATH = r"(?:deaths?|died|dies|die|dying|mortality)"
_SETTING_DEATH_ZH = r"(?:死亡|病死)"
# "Mortality in ICU patients", "in intensive care unit patients", "in
# ICU-admitted patients", "in ICU septic patients" name a population, not where
# death occurred.  Up to two modifiers may stand before the population noun,
# but never a word that starts a new phrase ("death in the ICU among patients").
_POPULATION_MODIFIER = (
    r"(?:(?!(?:among|with|without|of|in|for|and|or|who|that|which|after|before|during|"
    r"within|by|at|on|from|than|versus|vs)\b)[a-z]+[\s-]+)"
)
_NOT_A_POPULATION = (
    r"(?![\s-]*(?:units?[\s-]+)?(?:admitted[\s-]+)?" + _POPULATION_MODIFIER + r"{0,2}"
    r"(?:patients?|populations?|cohorts?|admissions?|survivors?|subjects?|adults?|children|"
    r"stays?|cases?)\b)"
)
# "an ICU mortality score" names an instrument, not where death is counted.
_NOT_AN_INSTRUMENT = (
    r"(?![\s-]+(?:risk[\s-]+|prediction[\s-]+|predictive[\s-]+|probability[\s-]+)?"
    r"(?:scores?|scoring|index|indices|indexes|scales?|calculators?|tools?)\b)"
)
_NOT_AN_INSTRUMENT_ZH = r"(?!\s*(?:率|风险)?\s*(?:预测)?\s*(?:评分|指数|量表|计算器))"
_ICU_SETTING_PATTERNS = (
    # "ICU mortality", "ICU deaths", "intensive care unit mortality", "in-ICU death",
    # "MICU mortality"; never "post-ICU mortality" or "an ICU mortality score".
    re.compile(
        rf"(?<!post-)(?<!post )\b{_ICU_PLACE}[\s-]+{_SETTING_NOUN}\b{_NOT_AN_INSTRUMENT}",
        re.IGNORECASE,
    ),
    # "death in the ICU", "died during the ICU stay", "die in the medical ICU".
    re.compile(
        rf"\b{_SETTING_DEATH}\s+(?:in|during|within|inside)\s+(?:(?:the|an?)\s+)?"
        rf"{_ICU_KIND}{_ICU_PLACE}(?:[\s-]+stay)?\b{_NOT_A_POPULATION}",
        re.IGNORECASE,
    ),
    # "death before ICU discharge", "died before discharge from the ICU".
    re.compile(
        rf"\b{_SETTING_DEATH}\s+(?:before|prior\s+to)\s+(?:the\s+)?"
        rf"(?:{_ICU_PLACE}[\s-]+discharge|discharge\s+from\s+(?:the\s+)?{_ICU_PLACE})\b",
        re.IGNORECASE,
    ),
    # 「ICU 死亡」「ICU 内死亡」「ICU 病死率」「ICU 住院期间死亡」「重症监护病房死亡」;
    # never 「出 ICU 后 48 小时内死亡」, which counts deaths after the ICU.
    re.compile(
        rf"(?<!出)(?<!出 )(?<!离开)(?<!离开 ){_ICU_PLACE_ZH}"
        rf"\s*(?:内|中|里|期间|住院期间|住院|停留期间|入住期间|治疗期间)?\s*的?\s*{_SETTING_DEATH_ZH}"
        rf"{_NOT_AN_INSTRUMENT_ZH}",
        re.IGNORECASE,
    ),
    # 「死于 ICU」「出 ICU 前死亡」.
    re.compile(rf"(?:死于|死在)\s*{_ICU_PLACE_ZH}", re.IGNORECASE),
    re.compile(
        rf"(?:转出|离开|出)\s*{_ICU_PLACE_ZH}\s*(?:之)?前\s*的?\s*{_SETTING_DEATH_ZH}",
        re.IGNORECASE,
    ),
)
_HOSPITAL_SETTING_PATTERNS = (
    # "in-hospital mortality", "hospital death".
    re.compile(rf"\b(?:in[\s-]+)?hospital[\s-]+{_SETTING_NOUN}\b{_NOT_AN_INSTRUMENT}", re.IGNORECASE),
    # "death in hospital", "died during the hospital stay".
    re.compile(
        rf"\b{_SETTING_DEATH}\s+(?:in|during|within)\s+(?:the\s+)?hospital"
        rf"(?:[\s-]+stay)?\b{_NOT_A_POPULATION}",
        re.IGNORECASE,
    ),
    # "death before hospital discharge".
    re.compile(
        rf"\b{_SETTING_DEATH}\s+(?:before|prior\s+to)\s+(?:the\s+)?"
        rf"(?:hospital[\s-]+discharge|discharge\s+from\s+(?:the\s+)?hospital)\b",
        re.IGNORECASE,
    ),
    # 「院内死亡」「住院期间死亡」「出院前死亡」「死于院内」.
    re.compile(
        rf"(?:医?院内|住院\s*(?:期间)?\s*的?|出院\s*(?:之)?前\s*的?)\s*{_SETTING_DEATH_ZH}"
        rf"{_NOT_AN_INSTRUMENT_ZH}"
    ),
    re.compile(r"(?:死于|死在)\s*(?:医院|院内)"),
)
# Two settings sharing one death noun: "ICU and in-hospital mortality",
# "ICU- and hospital mortality", "hospital and ICU mortality", "ICU or hospital
# death", 「ICU 及院内死亡」「院内和 ICU 死亡」.  The elided setting is read where
# it stands; never a place someone is moved to or from ("admitted to the ICU
# and in-hospital mortality"), a population ("ICU patients and in-hospital
# mortality") or an instrument ("an ICU and hospital mortality score").
_SETTING_JOIN = r"\s*(?:and/or|and|or|&|/|vs\.?|versus)\s*"
_SETTING_JOIN_ZH = r"\s*(?:以及|及|和|与|或|、|/)\s*"
_NOT_A_DESTINATION = (
    r"(?<!to )(?<!to the )(?<!to an )(?<!to a )(?<!into )(?<!into the )(?<!into an )"
    r"(?<!into a )(?<!from )(?<!from the )(?<!from an )(?<!from a )"
)
_HOSPITAL_PLACE_ZH = r"(?:医?院内|住院(?:期间)?)"
_ELIDED_ICU_PATTERNS = (
    re.compile(
        rf"{_NOT_A_DESTINATION}\b{_ICU_PLACE}-?(?={_SETTING_JOIN}(?:in[\s-]+)?hospital[\s-]+"
        rf"{_SETTING_NOUN}\b{_NOT_AN_INSTRUMENT})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?<!出)(?<!出 )(?<!离开)(?<!离开 ){_ICU_PLACE_ZH}\s*(?:内|中)?"
        rf"(?={_SETTING_JOIN_ZH}{_HOSPITAL_PLACE_ZH}\s*的?\s*{_SETTING_DEATH_ZH}"
        rf"{_NOT_AN_INSTRUMENT_ZH})",
        re.IGNORECASE,
    ),
)
_ELIDED_HOSPITAL_PATTERNS = (
    re.compile(
        rf"{_NOT_A_DESTINATION}\b(?:in[\s-]+)?hospital-?(?={_SETTING_JOIN}{_ICU_PLACE}[\s-]+"
        rf"{_SETTING_NOUN}\b{_NOT_AN_INSTRUMENT})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:{_HOSPITAL_PLACE_ZH}|医院)(?={_SETTING_JOIN_ZH}{_ICU_PLACE_ZH}\s*(?:内|中)?\s*的?\s*"
        rf"{_SETTING_DEATH_ZH}{_NOT_AN_INSTRUMENT_ZH})",
        re.IGNORECASE,
    ),
)
# The death noun first, then both places: "mortality in the ICU and in
# hospital", "death in hospital or in the ICU".  The second place (group
# ``second``) is read only where its phrase ends, so "mortality in the ICU and
# hospital length of stay" names hospital length of stay, not hospital death.
_PHRASE_END = (
    r"(?=\s*(?:[,.;:?!)]|$|\b(?:and|or|among|in|for|with|within|by|at|after|during|"
    r"between|compared|vs|versus|is|are|was|were|differ\w*)\b))"
)
_SECOND_PLACE_PATTERNS = (
    (
        HOSPITAL_DEATH,
        re.compile(
            rf"\b{_SETTING_DEATH}\s+(?:in|during|within|inside)\s+(?:(?:the|an?)\s+)?"
            rf"{_ICU_KIND}{_ICU_PLACE}(?:[\s-]+stay)?{_SETTING_JOIN}"
            rf"(?P<second>(?:(?:in|during|within)\s+)?(?:the\s+)?hospital(?:[\s-]+stay)?)"
            rf"{_PHRASE_END}",
            re.IGNORECASE,
        ),
    ),
    (
        ICU_DEATH,
        re.compile(
            rf"\b{_SETTING_DEATH}\s+(?:in|during|within)\s+(?:the\s+)?hospital(?:[\s-]+stay)?"
            rf"{_SETTING_JOIN}(?P<second>(?:(?:in|during|within|inside)\s+)?(?:(?:the|an?)\s+)?"
            rf"{_ICU_KIND}{_ICU_PLACE}(?:[\s-]+stay)?){_PHRASE_END}",
            re.IGNORECASE,
        ),
    ),
)
# A setting the text directly declines is not requested: "do not use ICU
# mortality", "rather than hospital mortality", 「不要使用 ICU 死亡」,
# 「而不是院内死亡」.  Only a cue standing right before the mention declines
# it, so "do not exclude patients with ICU mortality" still names it.
_DECLINED_SETTING_BEFORE = re.compile(
    r"(?:\b(?:not|never|cannot|can't|don't)\s+"
    r"(?:(?:use|using|substitute|substituting|report|reporting|measure|measuring)\s+)?"
    r"|\b(?:rather than|instead of)\s+"
    r"|(?:不能|不得|不要|不应|不用|不使用|不采用|不以|不是|而非|而不是|并非)\s*"
    r"(?:(?:使用|采用|报告|分析|计算|用|以)\s*)?)$",
    re.IGNORECASE,
)
_DECLINED_REACH = 48


@dataclass(frozen=True, slots=True)
class StatedMortalitySetting:
    """One place a text states where death is counted: ``text[start:end]``."""

    #: ``icu_death`` or ``hospital_death``.
    endpoint: str
    start: int
    end: int


def stated_mortality_setting_mentions(text: str) -> tuple[StatedMortalitySetting, ...]:
    """Every place ``text`` asks for ICU or hospital mortality, in text order.

    ICU mortality: "ICU mortality", "ICU death", "death in the ICU", "died
    before ICU discharge", 「ICU 死亡率」「ICU 内死亡」「ICU 病死率」「死于 ICU」
    「出 ICU 前死亡」.  Hospital mortality: "in-hospital mortality", "death in
    hospital", 「院内死亡」「住院期间死亡」「出院前死亡」.  Settings sharing one
    noun ("ICU and in-hospital mortality", 「ICU 及院内死亡」) are both read.  A
    hospital phrase inside an ICU one (「ICU 住院期间死亡」) is the ICU's.  A mention the text
    directly declines ("not ICU mortality", 「不要用 ICU 死亡」) is not one, and
    neither is a population ("mortality in ICU patients") or a death after
    the ICU (「出 ICU 后 48 小时内死亡」).
    """

    source = str(text or "")
    second = {
        endpoint: [match.span("second") for match in pattern.finditer(source)]
        for endpoint, pattern in _SECOND_PLACE_PATTERNS
    }
    icu = sorted(
        [
            (match.start(), match.end())
            for pattern in (*_ICU_SETTING_PATTERNS, *_ELIDED_ICU_PATTERNS)
            for match in pattern.finditer(source)
        ]
        + second[ICU_DEATH]
    )
    masked = list(source)
    for start, end in icu:
        masked[start:end] = " " * (end - start)
    hospital = sorted(
        [
            (match.start(), match.end())
            for pattern in _HOSPITAL_SETTING_PATTERNS
            for match in pattern.finditer("".join(masked))
        ]
        + [
            # Read before the ICU mention it shares a noun with is masked.
            (match.start(), match.end())
            for pattern in _ELIDED_HOSPITAL_PATTERNS
            for match in pattern.finditer(source)
        ]
        + second[HOSPITAL_DEATH]
    )
    mentions: list[StatedMortalitySetting] = []
    for endpoint, start, end in sorted(
        [(ICU_DEATH, *span) for span in icu] + [(HOSPITAL_DEATH, *span) for span in hospital],
        key=lambda item: (item[1], -item[2]),
    ):
        if any(item.start < end and start < item.end for item in mentions):
            continue
        if _DECLINED_SETTING_BEFORE.search(source[max(0, start - _DECLINED_REACH):start]):
            continue
        mentions.append(StatedMortalitySetting(endpoint=endpoint, start=start, end=end))
    return tuple(mentions)


def stated_mortality_settings(text: str) -> tuple[str, ...]:
    """The distinct settings ``text`` asks for death in (``icu_death``, ``hospital_death``)."""

    return tuple(dict.fromkeys(item.endpoint for item in stated_mortality_setting_mentions(text)))


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
    "HOSPITAL_DEATH",
    "ICU_DEATH",
    "FOLLOWUP_OUTCOME_DATABASES",
    "MEAN_SURVIVAL_PATTERN",
    "FixedHorizonMortalityEndpoint",
    "ICU_FREE_DAY_DATABASES",
    "ICU_READMISSION_DATABASES",
    "OUTCOME_CONCEPT_SUPPORTED_DATABASES",
    "OutcomeConceptUnavailability",
    "StatedHorizon",
    "StatedMeanSurvival",
    "StatedHorizonMention",
    "StatedMortalitySetting",
    "fixed_horizon_mortality_endpoint",
    "fixed_horizon_mortality_endpoint_stated_by",
    "mortality_horizon_spans",
    "stated_mean_survival_mentions",
    "stated_mortality_horizon_mentions",
    "stated_mortality_horizons",
    "stated_mortality_setting_mentions",
    "stated_mortality_settings",
    "structural_outcome_unavailability",
]
