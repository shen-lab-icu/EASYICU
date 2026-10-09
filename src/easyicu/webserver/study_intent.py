"""Typed study-intent extraction for the Copilot front door.

Owner of ONE responsibility: turning a user's own sentence into a typed,
closed-set ``StudyContract`` proposal, plus an explicit list of the slots it
could **not** read.

Why this module exists
----------------------
The conversational front door used to infer intent with three regexes over the
question text and then fill every remaining slot from module-level defaults
(``exposure='lactate'``, ``outcome='In-hospital mortality'``, population pinned
to ``Sepsis-3``). A question about fluid balance and AKI therefore became a
question about lactate and mortality, and that substituted string was what got
submitted, persisted and bound to evidence.

The contract here is deliberately narrow:

* Slots come from **closed sets** or from the project's own concept catalog.
* A slot that cannot be read from the user's words stays ``None`` and is named
  in ``unread``. Nothing is ever filled from a default. A caller that wants a
  value must ask the user.
* The LLM path is optional and gated. When it is unavailable, refused, or
  returns anything that fails validation, the deterministic reader answers and
  the reason is reported. The extractor never invents to stay useful.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Tuple

from easyicu.ai_optin import AIOptInError
from easyicu.outcome_availability import (
    fixed_horizon_mortality_endpoint_stated_by,
    mortality_horizon_spans,
    stated_mortality_horizon_mentions,
)
from easyicu.research_agent.research_context.temporal_semantics import (
    event_anchored_spans,
)
from easyicu.webserver import provider_adapter
from easyicu.webserver.provider_gate import ProviderGateError, resolve_provider_gate

__all__ = [
    "ANALYSIS_FAMILIES",
    "OUTCOME_TYPES",
    "StudyIntentError",
    "extract_study_intent",
    "deterministic_intent",
    "explicit_outcome_concepts",
    "explicit_exposure_aggregation",
    "explicit_landmark_hours",
    "SLOTS",
]

# --------------------------------------------------------------------------
# Closed sets. A value outside these is rejected, never coerced.
# --------------------------------------------------------------------------
ANALYSIS_FAMILIES: Tuple[str, ...] = (
    "description",
    "association",
    "prediction",
    "survival",
    "causal",
    "trajectory",
    "cross_database",
    "data_quality",
)
OUTCOME_TYPES: Tuple[str, ...] = (
    "binary",
    "continuous",
    "time_to_event",
    "ordinal",
    "count",
)
SLOTS: Tuple[str, ...] = (
    "population",
    "exposure",
    "outcome",
    "outcome_type",
    "time_window_hours",
    "comparator",
    "analysis_family",
)

_MAX_QUESTION_CHARS = 1200
_MAX_SLOT_CHARS = 160


class StudyIntentError(ValueError):
    """Raised when the request itself is unusable (not when a slot is unread)."""

    def __init__(self, detail: Dict[str, Any]) -> None:
        super().__init__(str(detail.get("error") or "study_intent_error"))
        self.detail = detail


# --------------------------------------------------------------------------
# Vocabulary, grounded in the project's own concept catalog.
# --------------------------------------------------------------------------
def _concept_groups() -> Dict[str, List[str]]:
    try:
        from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

        return {str(k): [str(c) for c in v] for k, v in CONCEPT_GROUPS_INTERNAL.items()}
    except Exception:  # pragma: no cover - catalog is optional at import time
        return {}


# The closed fixed-horizon mortality concepts are read by the shared horizon
# reader: "28-day mortality", "survival to day 90", "one-year survival" and
# "随访一年" each name the one closed endpoint their horizon admits.  A horizon
# no closed endpoint admits ("30-day mortality") is left to the generic
# mortality reading.
_FIXED_HORIZON_MORTALITY = "fixed_horizon_mortality"

# Clinical phrasings (EN + ZH) mapped onto catalog concept ids. This is a
# reading aid, not an allowlist of what a study may be about: an unmatched
# phrase yields an unread slot, never a default.
_PHRASE_TO_CONCEPT: Tuple[Tuple[str, str], ...] = (
    # A spelled-out catalog name is read by its own name ("lactate
    # dehydrogenase" is ldh, not lactate): see ``_resolved_reading``.  These are
    # the abbreviations and name variants the catalog does not spell.
    (r"\bldh\b", "ldh"),
    (r"\bhba1c\b|\ba1c\b|glyc(?:ated|osylated)\s+h(?:ae|e)moglobin", "hba1c"),
    (r"\bplr\b", "plr"),
    (r"\bnlr\b", "nlr"),
    (r"lactate|乳酸", "lact"),
    (r"\bsofa-?2\b|sofa2", "sofa2"),
    (r"\bsofa\b", "sofa"),
    (r"\bqsofa\b", "qsofa"),
    (r"\bsaps\s*3\b|saps3", "saps3"),
    (r"\bapache\b", "apache_iv"),
    (r"charlson|查尔森", "charlson"),
    (r"creatinine|肌酐", "crea"),
    (r"\bbun\b|尿素氮", "bun"),
    (r"bilirubin|胆红素", "bili"),
    (r"platelet|血小板", "plt"),
    (r"albumin|白蛋白", "alb"),
    (r"h(ae|e)moglobin|血红蛋白", "hgb"),
    (r"\bwbc\b|white cell|白细胞", "wbc"),
    (r"\bph\b|酸碱", "ph"),
    (r"base excess|碱剩余", "be"),
    (r"heart rate|心率", "hr"),
    (r"\bmap\b|mean arterial|平均动脉压", "map"),
    (r"blood pressure|血压", "sbp"),
    (r"temperature|体温|发热", "temp"),
    (r"spo2|血氧饱和度", "spo2"),
    (r"pa[o/]?2\s*/\s*fio2|p/f ratio|pafi|氧合指数", "pafi"),
    (r"fio2", "fio2"),
    (r"\bpeep\b", "peep"),
    (r"tidal volume|潮气量", "tidal_vol"),
    (r"mechanical ventilation|ventilat|机械通气|插管", "vent_ind"),
    (r"norepinephrine|noradrenaline|去甲肾上腺素", "norepi_rate"),
    (r"vasopressor|升压药|血管活性药", "norepi_equiv"),
    (r"antibiotic|抗生素|抗菌药", "abx"),
    (r"corticosteroid|steroid|激素|糖皮质", "cort"),
    (r"fluid balance|液体平衡|液体正平衡|入出量|液体复苏", "fluid_balance"),
    (r"urine output|尿量", "urine"),
    (r"\brrt\b|renal replacement|dialysis|透析|肾脏替代", "rrt"),
    # Specific before general: "KDIGO AKI stage" is aki_stage, not aki.
    (r"kdigo|aki stage|aki 分期|肾损伤分期", "aki_stage"),
    (r"\baki\b|acute kidney|急性肾损伤|肾损伤", "aki"),
    (r"\bgcs\b|glasgow|昏迷评分", "gcs"),
    (r"\brass\b|镇静评分", "rass"),
    (r"delirium|谵妄", "gcs"),
    (r"sofa-?2.{0,20}(?:sepsis|脓毒症)|(?:sepsis|脓毒症).{0,20}sofa-?2", "sep3_sofa2"),
    (r"sepsis-?3|脓毒症|sepsis", "sep3"),
    (r"suspected infection|疑似感染", "susp_inf"),
    (r"circulatory failure|循环衰竭|休克", "circ_failure"),
    (r"\bbmi\b|体重指数", "bmi"),
    (r"\bage\b|年龄", "age"),
    (r"\bsex\b|gender|性别", "sex"),
    (r"\badmission (?:type|category)\b|\btype of admission\b|入院类型|入院类别|入科类型", "adm"),
    (r"in-?hospital mortality|hospital mortality|院内死亡|住院death|住院死亡", "death"),
    (_FIXED_HORIZON_MORTALITY, _FIXED_HORIZON_MORTALITY),
    (r"icu mortality|icu 死亡", "death"),
    (r"mortality|death|死亡|病死", "death"),
    (r"length of stay|\blos\b|住院时长|住院时间|icu 时长", "los_icu"),
    (r"readmission|再入院", "icu_readmission"),
    (r"ventilator-?free|无呼吸机天数", "vent_free_days_28"),
)

# Outcome candidates in two tiers. Tier 1 is the catalog's own ``outcome``
# group: these are almost never the cohort. Tier 2 are clinical events that are
# just as often the population or the exposure ("in AKI patients", "AKI stage
# vs LoS"), so they only become the outcome when no tier-1 phrase is present.
# Sepsis-3 is deliberately in NEITHER: in this corpus it is the cohort, and
# guessing it as an outcome is exactly the substitution this module exists to
# stop.
_OUTCOME_CONCEPTS_PRIMARY = frozenset(
    {
        "death",
        "mort_28d",
        "mort_90d",
        "mort_365d",
        "los_icu",
        "los_hosp",
        "icu_free_days_28",
        "vent_free_days_28",
        "icu_readmission",
    }
)
_OUTCOME_CONCEPTS_EVENT = frozenset({"aki", "aki_stage", "rrt", "circ_failure", "vent_ind"})
_OUTCOME_CONCEPTS = _OUTCOME_CONCEPTS_PRIMARY | _OUTCOME_CONCEPTS_EVENT
# A population is only read when the sentence actually names a population.
# Without this, "ICU length of stay" would silently become "ICU patients".
# NOTE: plural "stays" only. "length of stay" is an outcome, not a cohort.
_POPULATION_NOUN = re.compile(r"patients?|adults?|\bstays\b|cohort|subjects?|患者|人群|病人", re.IGNORECASE)
_TIME_TO_EVENT_CONCEPTS = frozenset({"los_icu", "los_hosp"})
_ORDINAL_CONCEPTS = frozenset({"aki_stage"})
_COUNT_CONCEPTS = frozenset({"icu_free_days_28", "vent_free_days_28"})

# (pattern, label, concepts this cohort is defined by). The third element lets
# a disease phrase be dropped as the cohort when that same disease is already
# serving as the outcome — "AKI stage vs LoS" has an AKI outcome, not an AKI
# cohort.
_POPULATION_PATTERNS: Tuple[Tuple[str, str, frozenset], ...] = (
    (r"sepsis-?3|脓毒症|septic", "Sepsis-3 patients", frozenset({"sep3", "susp_inf"})),
    (r"\baki\b|acute kidney|急性肾损伤|kdigo", "Patients with AKI", frozenset({"aki", "aki_stage", "rrt"})),
    (r"ventilated|mechanical ventilation|机械通气", "Mechanically ventilated patients", frozenset({"vent_ind", "vent_free_days_28"})),
    (r"\bards\b", "Patients with ARDS", frozenset()),
    (r"cardiac surgery|心脏外科|心脏手术", "Cardiac surgery patients", frozenset()),
    (r"\bcovid", "COVID-19 patients", frozenset()),
    (r"adults?|成年|成人", "Adult ICU patients", frozenset()),
    (r"\bicu\b|重症|监护", "ICU patients", frozenset()),
)

_FAMILY_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"across\s+(?:\w+\s+){0,2}(databases?|cohorts?|centres?|centers?|sites?)|cross-?database|多个数据库|跨库|external validation", "cross_database"),
    (r"missing|coverage|data quality|数据质量|缺失|覆盖率", "data_quality"),
    (r"predict|prognostic|risk score|auroc|discrimination|预测|预后模型", "prediction"),
    (r"causal|confound|treatment effect|因果|混杂|倾向性评分|propensity", "causal"),
    (r"survival|time-?to-?event|hazard|cox|生存|风险比", "survival"),
    (r"trajectory|over time|longitudinal|轨迹|随时间|纵向", "trajectory"),
    (r"associat|correlat|relationship|related to|相关|关联", "association"),
    (r"describe|distribution|prevalence|characteris|描述|分布|患病率", "description"),
)


# "I am NOT studying death, my outcome is AKI" must not read `death`. Without
# this, a user's correction becomes the very thing they corrected.
_NEGATION = re.compile(
    r"(?:\bnot\b|\bno\b|\bnever\b|\bisn't\b|\baren't\b|\bdon't\b|\bdoesn't\b|\brather than\b|\binstead of\b|不是|不要|不想|不(?:研究|分析|考虑|比较)|并非|而非|非|无关|别)"
    r"[\s\S]{0,16}$",
    re.IGNORECASE,
)
_NEGATION_LOOKBACK = 26


def _negated(text: str, start: int) -> bool:
    """True when a negation marker governs the match.

    The window is generous enough for "I am not studying mortality" and
    "我不是要研究死亡", and stops at sentence boundaries so a negation in one
    clause does not suppress a reading in the next.
    """
    window = text[max(0, start - _NEGATION_LOOKBACK) : start]
    # A sentence break ends a negation's scope.
    for sep in (". ", "; ", "。", "；", "?", "？"):
        if sep in window:
            window = window.rsplit(sep, 1)[1]
    # "非 X 患者的死亡" names a negative-exposure population, not a negated
    # outcome. Close that noun phrase's scope while retaining any subsequent
    # explicit negation such as "非 X 患者中，不研究死亡".
    window = re.sub(r"非[^，,。；;?？]{1,20}?(?:患者|人群|病人)", "", window)
    return bool(_NEGATION.search(window))


# A follow-up handling clause says how follow-up ends: "处理死亡、出院与删失",
# "account for death and discharge as competing events", "censored at death".
# The events it lists end follow-up; they are not endpoints the researcher asks
# to analyse, whichever concept they name.  Every check below is a bounded
# search or split, so a long question cannot make the reader backtrack.
_HANDLING_VERB = re.compile(
    r"(?:处理|考虑|handl(?:e|es|ed|ing)|account(?:s|ed|ing)?\s+for|deal(?:s|t|ing)?\s+with)"
    r"[^。；;?？]{0,24}$",
    re.IGNORECASE,
)
_CENSORING_TERM = re.compile(
    r"删失|截尾|竞争(?:风险|事件)|censor\w*|competing[\s-]+(?:risks?|events?)", re.IGNORECASE
)
_SENTENCE_END = re.compile(r"[。；;.?？!！]")
_CONJUNCTION = re.compile(r"\s*(?:、|与|和|及|或|\band\b|\bor\b)\s*", re.IGNORECASE)
_LIST_SEPARATOR = re.compile(
    r"\s*(?:,\s*(?:and|or)\b|、|,|，|与|和|及|或|\band\b|\bor\b)\s*", re.IGNORECASE
)
# "death and discharge as competing events", "death treated as a competing
# risk", "死亡作为竞争事件", "死亡时删失".
_ENDING_MARKER = re.compile(
    r"\s*(?:(?:(?:treated|considered|regarded|handled|counted|modell?ed|analy[sz]ed)\s+)?\bas\b"
    r"|作为|视为|时)\s*(?:an?\s+)?$",
    re.IGNORECASE,
)
# "以死亡为竞争风险": 以 ... 为 takes the events between them as the term.
_TAKEN_AS_OPENER = re.compile(r"以\s*$")
_TAKEN_AS_MARKER = re.compile(r"\s*为\s*$")
# "censored at death", "the competing risk of death".
_CENSORED_AT = re.compile(
    r"(?:censor(?:ed|ing)?\s+(?:at|on|by)|competing[\s-]+(?:risks?|events?)\s+(?:of|from))\s+$",
    re.IGNORECASE,
)


def _listed_events(segment: str, separator: re.Pattern, *, trailing: bool) -> bool:
    """Whether ``segment`` continues a list of short event names.

    It must start at a separator and hold only short items without "的";
    ``trailing`` requires it to end at a separator too.
    """

    pieces = separator.split(segment)
    if pieces[0].strip():
        return False
    items = pieces[1:-1] if trailing else pieces[1:]
    if trailing and len(pieces) > 1 and pieces[-1].strip():
        return False
    return all(0 < len(item.strip()) <= 16 and "的" not in item for item in items)


def _follow_up_handling(text: str, start: int, end: int) -> bool:
    """True when the match names an event that ends follow-up, not an endpoint."""

    before = text[max(0, start - 40) : start]
    if _CENSORED_AT.search(before):
        return True
    after = _SENTENCE_END.split(text[end : end + 96], maxsplit=1)[0]
    term = _CENSORING_TERM.search(after)
    if term is None:
        return False
    head = after[: term.start()]
    marker = _ENDING_MARKER.search(head)
    if marker is None and _TAKEN_AS_OPENER.search(before):
        marker = _TAKEN_AS_MARKER.search(head)
    if marker is not None and _listed_events(
        head[: marker.start()], _CONJUNCTION, trailing=False
    ):
        return True
    if not _HANDLING_VERB.search(before):
        return False
    # "考虑死亡的竞争风险": the term is the event's own.
    if head.rstrip().endswith("的"):
        return _listed_events(head.rstrip()[:-1], _LIST_SEPARATOR, trailing=False)
    # "处理死亡、出院与删失": a handled list that ends in censoring.
    return _listed_events(head, _LIST_SEPARATOR, trailing=True)


# Whom a study includes is not what it studies.  An age or a stay length stated
# with a bound ("年龄 ≥ 18 岁", "ICU 住院时长至少 24 小时", "length of stay > 48
# h") restricts the population, as the population spec's age and stay-length
# kinds do, and so does every concept named inside an inclusion or exclusion
# clause ("纳入机械通气的成人患者", "excluding patients with AKI on
# admission").  Such a mention is never read as the exposure or the outcome:
# the slot is read from another mention or stays unread.  A question that
# compares age groups ("年龄 ≥ 65 岁与 < 65 岁") therefore leaves its exposure
# unread for the plan to propose, the safe side of the same rule.
_BOUNDED_RESTRICTION_CONCEPTS = frozenset({"age", "los_icu", "los_hosp"})
_NUMBER = r"\d+(?:\.\d+)?"
_BOUND_WORD = (
    r"(?:≥|≤|⩾|⩽|>=|<=|=>|=<|>|<|＞|＜|≧|≦"
    r"|\bat\s+(?:least|most)\b|\b(?:no|not)\s+(?:less|more|fewer|older|younger)\s+than\b"
    r"|\b(?:more|less|greater|fewer|longer|shorter|older|younger)\s+than\b"
    r"|\b(?:over|under|above|below|exceeding|between)\b"
    r"|大于或等于|大于等于|小于或等于|小于等于|大于|小于|高于|低于|多于|少于|长于|短于"
    r"|不少于|不低于|不小于|不超过|不足|超过|至少|最少|最多|满|达到|介于)"
)
_BOUND_UNIT = r"(?:周岁|岁|小时|小時|天|日|周|years?(?:\s+old)?|yrs?|hours?|hrs?|h|days?|d|weeks?|wks?)"
_BOUND_TAIL = (
    r"(?:及以上|或以上|以上|及以下|或以下|以下|以内|之间"
    r"|\bor\s+(?:more|older|above|greater|longer|less|younger|fewer|over|under)\b"
    r"|\band\s+(?:over|above|older|under)\b|\+)"
)
# The bound follows the mention ("年龄 ≥ 18 岁", "年龄 18 岁以上", "年龄 18-80
# 岁", "length of stay of at least 24 h") or, less often, precedes it ("至少
# 48 小时的 ICU 住院时长").
_BOUND_AFTER = re.compile(
    rf"\s*(?:(?:of|is|was|were)\s+|为|在|是|[:：])?\s*[(（]?\s*"
    rf"(?:{_BOUND_WORD}\s*{_NUMBER}"
    rf"|{_NUMBER}\s*{_BOUND_UNIT}?\s*(?:[-–—~～至到]|\bto\b)\s*{_NUMBER}"
    rf"|{_NUMBER}\s*{_BOUND_UNIT}?\s*{_BOUND_TAIL})",
    re.IGNORECASE,
)
_BOUND_BEFORE = re.compile(
    rf"{_BOUND_WORD}\s*{_NUMBER}\s*{_BOUND_UNIT}?\s*(?:的|\bof\b)?\s*"
    r"(?:icu|hospital|院内|住院)?\s*$",
    re.IGNORECASE,
)
# An inclusion or exclusion clause runs from its opening word to the next
# clause break.  "包括" and a bare "include" are not openers: "结局包括死亡" and
# "outcomes include mortality" list endpoints, not eligibility.
_ELIGIBILITY_OPENER = re.compile(
    r"纳入|入选|排除|剔除|仅限|限于"
    r"|\b(?:inclusion|exclusion)\s+criteria\b|\beligib(?:le|ility)\b"
    r"|\bexclud(?:e|es|ed|ing)\b|\b(?:restricted|limited)\s+to\b"
    r"|\binclud(?:e|es|ed|ing)\s+(?:only\s+)?(?:all\s+)?(?:adult\s+)?(?:icu\s+)?"
    r"(?:patients|stays|admissions|subjects|adults)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[，,。；;？?！!]")


def _population_restriction(text: str, concept: str, start: int, end: int) -> bool:
    """Whether the mention at ``start:end`` states whom the study includes."""

    if concept in _BOUNDED_RESTRICTION_CONCEPTS and (
        _BOUND_AFTER.match(text, end)
        or _BOUND_BEFORE.search(text[max(0, start - 40) : start])
    ):
        return True
    for opener in _ELIGIBILITY_OPENER.finditer(text, 0, start):
        stop = _CLAUSE_BREAK.search(text, opener.end())
        if stop is None or start < stop.start():
            return True
    return False


# Concepts that name the same clinical thing at different granularity. Used to
# stop one phrase from filling two different slots.
_CONCEPT_FAMILIES: Tuple[frozenset, ...] = (
    frozenset({"aki", "aki_stage", "rrt"}),
    frozenset({"death", "mort_28d", "mort_90d", "mort_365d"}),
    frozenset({"los_icu", "los_hosp"}),
    frozenset({"vent_ind", "vent_free_days_28", "peep", "tidal_vol"}),
    frozenset({"sep3", "sep3_sofa2", "susp_inf"}),
)


def _family_of(concept: Optional[str]) -> Optional[frozenset]:
    if not concept:
        return None
    for family in _CONCEPT_FAMILIES:
        if concept in family:
            return family
    return None


def _entry_readings(pattern: str, concept: str, text: str) -> Iterator[Tuple[str, int, int]]:
    """Each reading of one phrase-table entry in ``text``: concept, start, end."""

    if concept == _FIXED_HORIZON_MORTALITY:
        for mention in stated_mortality_horizon_mentions(text):
            endpoint = fixed_horizon_mortality_endpoint_stated_by(mention.horizon)
            if endpoint is not None:
                yield endpoint.event_concept, mention.start, mention.end
        return
    for match in re.finditer(pattern, text, re.IGNORECASE):
        yield concept, match.start(), match.end()


@lru_cache(maxsize=1)
def _catalog_name_patterns() -> Tuple[Tuple[str, "re.Pattern[str]"], ...]:
    """Every catalog concept's own names, compiled once."""

    try:
        from easyicu.concept.catalog import CONCEPT_DICTIONARY
    except Exception:  # pragma: no cover - catalog is optional at import time
        return ()
    compiled = []
    for concept_id in CONCEPT_DICTIONARY:
        pattern = _concept_name_pattern(concept_id)
        if pattern is not None:
            compiled.append((concept_id, re.compile(pattern, re.IGNORECASE)))
    return tuple(compiled)


@lru_cache(maxsize=64)
def _catalog_name_readings(text: str) -> Tuple[Tuple[str, int, int], ...]:
    """Each catalog concept ``text`` names by its own name: concept, start, end."""

    return tuple(
        (concept_id, match.start(), match.end())
        for concept_id, pattern in _catalog_name_patterns()
        for match in pattern.finditer(text)
    )


#: A reading joined to another analyte in a ratio ("lactate/pyruvate ratio",
#: "ratio of BUN to creatinine", "乳酸与丙酮酸比值"), or followed by a clearance,
#: names that derived measure, not the analyte.
_RATIO_WORD = r"(?:ratio\b|比值|比率|之比)"
_RATIO_JOIN = r"(?:[\s\-‐–]+to[\s\-‐–]+|\s*[/／:]\s*|与|和)"
_RATIO_TERM = r"[^\s,.;:，。；：、]{1,24}"
_DERIVED_AFTER = re.compile(
    rf"(?:[\s\-‐–]*(?:clearance\b|清除率)"
    rf"|{_RATIO_JOIN}{_RATIO_TERM}(?:[\s\-‐–]+{_RATIO_TERM}){{0,2}}?[\s\-‐–]*的?{_RATIO_WORD})",
    re.IGNORECASE,
)
_RATIO_PARTNER_BEFORE = re.compile(rf"{_RATIO_TERM}{_RATIO_JOIN}$", re.IGNORECASE)
_RATIO_AFTER = re.compile(rf"[\s\-‐–]*的?{_RATIO_WORD}", re.IGNORECASE)
_RATIO_OF_BEFORE = re.compile(
    r"\bratio\s+of\s+(?:(?:[^\s,.;:]+\s+){0,3}?|.{1,40}?\bto\s+)$", re.IGNORECASE
)


def _derived_measure_component(text: str, start: int, end: int) -> bool:
    """Whether the reading at ``start:end`` is one term of a ratio or a clearance."""

    before, after = text[:start], text[end:]
    return bool(
        _DERIVED_AFTER.match(after)
        or (_RATIO_PARTNER_BEFORE.search(before) and _RATIO_AFTER.match(after))
        or _RATIO_OF_BEFORE.search(before)
    )


def _resolved_reading(
    text: str, concept: str, start: int, end: int
) -> Optional[Tuple[str, int, int]]:
    """What one phrase-table reading names, or ``None`` when it names nothing read here.

    A reading inside another concept's longer catalog name is that concept:
    "lactate dehydrogenase" is ldh, "直接胆红素" is direct bilirubin, "hospital
    length of stay" is the hospital stay; the longest such name wins.  A term
    of a ratio or a clearance the catalog does not name is not read at all, so
    the slot stays unread rather than naming the analyte it is computed from.
    """

    longest: Optional[Tuple[str, int, int]] = None
    for other, begin, finish in _catalog_name_readings(text):
        if (
            other != concept
            and begin <= start
            and end <= finish
            and finish - begin > end - start
            and (longest is None or finish - begin > longest[2] - longest[1])
        ):
            longest = (other, begin, finish)
    if longest is not None:
        return longest
    if _derived_measure_component(text, start, end):
        return None
    return concept, start, end


def _match_concept(text: str) -> List[Tuple[str, str]]:
    """Return concept/phrase pairs in dictionary-specificity order.

    A phrase the sentence explicitly negates is not a reading — it is skipped,
    which leaves the slot unread rather than wrong.  So is a restriction on
    whom the study includes (``_population_restriction``).  A phrase inside
    another concept's longer name is that concept (``_resolved_reading``).
    """
    found: List[Tuple[str, str]] = []
    seen = set()
    for pattern, entry in _PHRASE_TO_CONCEPT:
        for reading in _entry_readings(pattern, entry, text):
            resolved = _resolved_reading(text, *reading)
            if resolved is None:
                continue
            concept, start, end = resolved
            if (
                concept in seen
                or _negated(text, start)
                or _follow_up_handling(text, start, end)
                or _population_restriction(text, concept, start, end)
            ):
                continue
            seen.add(concept)
            found.append((concept, text[start:end]))
    return found


def explicit_outcome_concepts(question: str) -> tuple[str, ...]:
    """Read all explicit endpoint phrases without changing the primary slot.

    Clinical events can also name a population or exposure, so this roster
    only adds the closed, high-specificity endpoint vocabulary. A configured
    event outcome remains the caller's authority. Specific phrases reserve
    their text span: ``28-day mortality`` must not add generic ``death`` too.
    An event a follow-up handling clause lists ends follow-up, and a stay
    length that bounds the population restricts it; neither is read.
    This is intent, not evidence that the source can supply these endpoints.
    """

    text = str(question or "")
    values: list[str] = []
    covered: list[tuple[int, int]] = []
    for pattern, entry in _PHRASE_TO_CONCEPT:
        if entry not in _OUTCOME_CONCEPTS_PRIMARY and entry != _FIXED_HORIZON_MORTALITY:
            continue
        for reading in _entry_readings(pattern, entry, text):
            resolved = _resolved_reading(text, *reading)
            if resolved is None or resolved[0] not in _OUTCOME_CONCEPTS_PRIMARY:
                continue
            concept, start, end = resolved
            if (
                _negated(text, start)
                or _follow_up_handling(text, start, end)
                or _population_restriction(text, concept, start, end)
                or any(start < stop and begin < end for begin, stop in covered)
            ):
                continue
            covered.append((start, end))
            if concept not in values:
                values.append(concept)
    return tuple(values)


#: Conditions a question studies, or studies a population with.  One that
#: frames where the question is asked ("mortality in sepsis", "among AKI",
#: "sepsis mortality", "脓毒症28天死亡") is the setting, not the exposure, so
#: an exposure the reader cannot name stays unread instead of becoming the
#: setting's condition.
_CONDITION_CONCEPTS = frozenset().union(
    *(family for _pattern, _label, family in _POPULATION_PATTERNS)
) | frozenset({"sep3_sofa2", "circ_failure"})
_SETTING_BEFORE = re.compile(r"\b(?:in|among|amongst)\s+$", re.IGNORECASE)
_SETTING_AFTER = re.compile(
    r"\s*(?:-?related\s+|的\s*)?"
    r"(?:(?:\d{1,3}|[一二三四五六七八九十百]+)\s*(?:-?days?\s+|天|日)\s*)?"
    r"(?:in-?hospital\s+|hospital\s+|icu\s+|院内|住院)?"
    r"(?:mortality|death|survival|死亡|病死|生存|存活)",
    re.IGNORECASE,
)


def _exposure_candidates_in_text_order(
    text: str, candidates: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Read the studied marker before later definitions or method acronyms.

    Dictionary order ranks synonyms, not scientific roles. Explicit population
    phrases are not exposure assignments, nor is a condition that frames the
    question's setting (``_SETTING_BEFORE``, ``_SETTING_AFTER``). Ties retain
    dictionary specificity (for example, SOFA-2 before the overlapping
    original SOFA token).
    """

    positioned = []
    for rank, (concept, phrase) in enumerate(candidates):
        for match in re.finditer(re.escape(phrase), text, re.IGNORECASE):
            if _negated(text, match.start()) or _population_restriction(
                text, concept, match.start(), match.end()
            ):
                continue
            after = text[match.end():]
            before = text[max(0, match.start() - 35):match.start()]
            if re.match(r"\s*(?:patients?\b|cohort\b|患者|人群|病人)", after, re.IGNORECASE):
                continue
            if re.search(r"\bpatients?\s+with\s*$", before, re.IGNORECASE):
                continue
            if concept in _CONDITION_CONCEPTS and (
                _SETTING_BEFORE.search(text[:match.start()]) or _SETTING_AFTER.match(after)
            ):
                continue
            positioned.append((match.start(), rank, concept, phrase))
            break
    return [(concept, phrase) for _, _, concept, phrase in sorted(positioned)]


def _clean_question(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise StudyIntentError({"error": "study_intent_question_required"})
    if len(text) > _MAX_QUESTION_CHARS:
        raise StudyIntentError(
            {
                "error": "study_intent_question_too_long",
                "max_chars": _MAX_QUESTION_CHARS,
            }
        )
    return text


def _slot(value: Any, provenance: str, evidence: Optional[str] = None) -> Dict[str, Any]:
    return {
        "value": value,
        "provenance": provenance,
        "evidence": (str(evidence)[:_MAX_SLOT_CHARS] if evidence else None),
    }


def _empty_slot() -> Dict[str, Any]:
    return {"value": None, "provenance": "unread", "evidence": None}


@dataclass(frozen=True)
class ExplicitExposureAggregation:
    """An operation attached to one named concept in the actual question.

    This is proposal input, not an execution or plan-approval receipt. Table
    summary defaults never supply this coordinate.
    """

    concept_id: str
    aggregation: Literal["max", "min", "mean", "median", "first", "last", "sum"]
    evidence: str


_MEASUREMENT_OPERATIONS = (
    ("max", r"\b(?:maximum|highest|peak)\b|最高(?:值)?|最大(?:值)?|峰值"),
    ("min", r"\b(?:minimum|lowest|nadir)\b|最低(?:值)?|最小(?:值)?"),
    ("mean", r"\b(?:mean|average)\b|平均(?:值)?"),
    ("median", r"\bmedian\b|中位数"),
    ("first", r"\b(?:first|initial)\b|首次|初次"),
    ("last", r"\b(?:last|final)\b|末次|最后一次"),
    ("sum", r"\b(?:cumulative|total|sum)\b|累计|累积|总量"),
)


_LANDMARK_HOURS = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*-?\s*(?:h\b|hrs?\b|hours?\b|小时|小時)\s*(?:的|为|作为)?\s*landmark"
    r"|landmark\s*(?:at|of|=|:|：|为|设在|定在|于)?\s*(?:第\s*)?"
    r"(\d{1,3}(?:\.\d+)?)\s*-?\s*(?:h\b|hrs?\b|hours?\b|小时|小時)",
    re.IGNORECASE,
)


# "不采用 24 小时 landmark" refuses a design rather than naming one; the general
# negation vocabulary is about slots and does not cover design verbs.
_LANDMARK_REFUSAL = re.compile(
    r"(?:不采用|不使用|不用|不做|无需|不需要|避免|\bwithout\b|\bno\b|\bnot\b)[^，,。；;.]{0,12}$",
    re.IGNORECASE,
)


def explicit_landmark_hours(question: str) -> Optional[float]:
    """Read a landmark time the researcher stated, in hours after ICU admission.

    Only an explicit, non-negated statement of one landmark is a reading
    ("24 小时 landmark", "landmark at 24 h").  "不采用 landmark" names no time
    and is never a landmark; two different stated times are left unread for the
    plan to resolve rather than picking one.
    """

    # A reading aid, not an intake gate: an empty or over-long question simply
    # states no landmark here; ``_clean_question`` owns rejecting it.
    text = str(question or "").strip()[:_MAX_QUESTION_CHARS]
    stated = set()
    for match in _LANDMARK_HOURS.finditer(text):
        before = text[max(0, match.start() - _NEGATION_LOOKBACK):match.start()]
        if _negated(text, match.start()) or _LANDMARK_REFUSAL.search(before):
            continue
        hours = float(match.group(1) or match.group(2))
        if 0 < hours <= 720:
            stated.add(hours)
    return next(iter(stated)) if len(stated) == 1 else None


#: How a catalog name's words may be joined in a question: "platelet-to-
#: lymphocyte ratio", "platelet to lymphocyte ratio", "platelet/lymphocyte
#: ratio"; "血小板/淋巴细胞比值", "血小板与淋巴细胞比值".
_NAME_WORD_SPLIT = re.compile(r"[\s\-‐–/／]+")
_NAME_SEPARATOR = r"[\s\-‐–/／]+"
_NAME_TO_JOIN = r"(?:[\s\-‐–]+to[\s\-‐–]+|\s*[/／:]\s*)"
_NAME_CJK_SEPARATOR = r"(?:[\s\-‐–/／]+|与|和)?"
_CJK = re.compile(r"[\u3400-\u9fff]")


def _catalog_name_regex(name: str) -> str:
    """One catalog name as a phrase: its words in order, however they are joined."""

    words = [word for word in _NAME_WORD_SPLIT.split(name.strip()) if word]
    pattern = ""
    index = 0
    while index < len(words):
        word = words[index]
        if not pattern:
            pattern = re.escape(word)
        elif word.lower() == "to" and index + 1 < len(words):
            index += 1
            pattern += _NAME_TO_JOIN + re.escape(words[index])
        elif _CJK.search(words[index - 1][-1:]) or _CJK.search(word[:1]):
            pattern += _NAME_CJK_SEPARATOR + re.escape(word)
        else:
            pattern += _NAME_SEPARATOR + re.escape(word)
        index += 1
    return pattern


@lru_cache(maxsize=1024)
def _concept_name_pattern(concept_id: str) -> Optional[str]:
    """The concept's own catalog names, read as whole concept phrases.

    "Total Bilirubin" and "总胆红素" name one analyte: "total" is part of the
    name there, not a cumulative operation.
    """

    try:
        from easyicu.concept.catalog import CONCEPT_DICTIONARY
    except Exception:  # pragma: no cover - catalog is optional at import time
        return None
    entry = CONCEPT_DICTIONARY.get(concept_id)
    names = [
        str(value).strip()
        for value in (entry[:2] if isinstance(entry, tuple) else ())
        if value is not None and str(value).strip()
    ]
    if not names:
        return None
    return "|".join(
        rf"(?<![a-z0-9]){_catalog_name_regex(name)}(?![a-z0-9])" for name in names
    )


def explicit_exposure_aggregation(
    question: str, *, concept_id: str,
) -> Optional[ExplicitExposureAggregation]:
    """Read only an adjacent, unambiguous measurement operation.

    Match the concept phrase as a whole before looking outside it: the word
    "mean" in "mean arterial pressure" does not request temporal averaging,
    and "total" in "total bilirubin", the analyte's own name, requests no
    sum.  A phrase inside a longer phrase of the same concept is that
    phrase.  A remote table-summary instruction or another variable's
    operation cannot bind this exposure. Negated and conflicting operations
    remain unread for complete-plan resolution, never an internal-field
    questionnaire.
    """

    text = _clean_question(question)
    lowered = text.lower()
    matches: Dict[str, str] = {}
    patterns = [pattern for pattern, concept in _PHRASE_TO_CONCEPT if concept == concept_id]
    name_pattern = _concept_name_pattern(concept_id)
    if name_pattern is not None:
        patterns.append(name_pattern)
    spans = {
        (named.start(), named.end())
        for pattern in patterns
        for named in re.finditer(pattern, text, re.IGNORECASE)
    }
    spans = {
        span
        for span in spans
        if not any(
            other != span and other[0] <= span[0] and span[1] <= other[1] for other in spans
        )
        and _resolved_reading(text, concept_id, *span) == (concept_id, *span)
    }
    for begin, finish in sorted(spans):
        for operation, expression in _MEASUREMENT_OPERATIONS:
            before = re.search(
                rf"(?:{expression})\s*(?:(?:serum|blood|plasma)\s+|血清|血浆)?$",
                text[:begin], re.IGNORECASE,
            )
            after = re.match(
                rf"\s*(?:(?:levels?|values?)\s+|的|值|水平)?(?:{expression})",
                text[finish:], re.IGNORECASE,
            )
            if before is not None:
                start, end = before.start(), finish
            elif after is not None:
                start, end = begin, finish + after.end()
            else:
                continue
            if not _negated(lowered, start):
                matches[operation] = text[start:end]
    if len(matches) != 1:
        return None
    operation, evidence = next(iter(matches.items()))
    return ExplicitExposureAggregation(
        concept_id=concept_id, aggregation=operation, evidence=evidence,
    )


# --------------------------------------------------------------------------
# Deterministic reader (always available, offline, no provider)
# --------------------------------------------------------------------------
def deterministic_intent(question: str) -> Dict[str, Any]:
    """Read what the sentence actually says. Leave the rest unread."""
    text = _clean_question(question)
    lowered = text.lower()
    slots: Dict[str, Dict[str, Any]] = {name: _empty_slot() for name in SLOTS}

    concepts = _match_concept(lowered)
    primary = [(c, p) for c, p in concepts if c in _OUTCOME_CONCEPTS_PRIMARY]
    events = [(c, p) for c, p in concepts if c in _OUTCOME_CONCEPTS_EVENT]

    outcome_concept: Optional[str] = None
    outcome_phrase: Optional[str] = None
    if primary:
        outcome_concept, outcome_phrase = primary[0]
    elif events:
        outcome_concept, outcome_phrase = events[0]

    if outcome_concept:
        slots["outcome"] = _slot(outcome_concept, "user_text", outcome_phrase)
        if outcome_concept in _TIME_TO_EVENT_CONCEPTS:
            kind = "time_to_event"
        elif outcome_concept in _ORDINAL_CONCEPTS:
            kind = "ordinal"
        elif outcome_concept in _COUNT_CONCEPTS:
            kind = "count"
        else:
            kind = "binary"
        slots["outcome_type"] = _slot(kind, "user_text", outcome_phrase)

    # Everything else the sentence names is an exposure candidate — including a
    # tier-2 event concept that did not win the outcome slot. A concept from the
    # SAME clinical family as the outcome is not an exposure though: "my outcome
    # is AKI (KDIGO stage)" names one thing twice, not an exposure and an
    # outcome. Leaving it unread is what makes the card ask.  Nor is any other
    # concept the question names as an endpoint: "compare mortality, ICU length
    # of stay and readmission across the groups" lists three outcomes, so the
    # groups it compares are the exposure, not the second outcome.  When an
    # endpoint is the factor studied ("is ICU length of stay associated with
    # 1-year mortality?"), the slot stays unread for the plan to propose,
    # rather than guessing which is which.
    outcome_family = _family_of(outcome_concept)
    listed_outcomes = set(explicit_outcome_concepts(text))
    exposures = [
        (c, p)
        for c, p in concepts
        if c != outcome_concept
        and c not in listed_outcomes
        and not (outcome_family and _family_of(c) == outcome_family)
    ]
    exposures = _exposure_candidates_in_text_order(text, exposures)
    if exposures:
        concept, phrase = exposures[0]
        slots["exposure"] = _slot(concept, "user_text", phrase)

    if _POPULATION_NOUN.search(text):
        for pattern, label, family in _POPULATION_PATTERNS:
            match = re.search(pattern, lowered, re.IGNORECASE)
            if not match or _negated(lowered, match.start()):
                continue
            # A disease already serving as the outcome is not also the cohort
            # ("...与急性肾损伤的风险相关" is an outcome, not a population).
            if outcome_concept and outcome_concept in family:
                continue
            slots["population"] = _slot(label, "user_text", match.group(0))
            break

    # The hours of "48-hour mortality" or "excluding deaths within 24 hours"
    # time an endpoint or an exclusion, not the window.  Those of "the first 24
    # hours after suspected infection onset" count from that event, not from
    # ICU admission, whether or not the event is the study's time zero, so
    # they are not the study's ICU window either.
    elsewhere = [*mortality_horizon_spans(lowered), *event_anchored_spans(lowered)]
    window = next(
        (
            match
            for match in re.finditer(r"(?:first\s*)?(\d{1,3})\s*(?:h\b|hr|hour|小时)", lowered)
            if not any(start <= match.start(1) < end for start, end in elsewhere)
        ),
        None,
    )
    first_day = next(
        (
            match
            for match in re.finditer(r"首日|第一天|first day", lowered)
            if not any(start <= match.start() < end for start, end in elsewhere)
        ),
        None,
    )
    if window:
        slots["time_window_hours"] = _slot(
            int(window.group(1)), "user_text", window.group(0)
        )
    elif first_day:
        slots["time_window_hours"] = _slot(24, "user_text", "first day")

    for pattern, family in _FAMILY_PATTERNS:
        for match in re.finditer(pattern, lowered, re.IGNORECASE):
            # "Not a prediction study; is PEEP associated with ..." is an
            # association study. The same negation rule applies here.
            if _negated(lowered, match.start()):
                continue
            slots["analysis_family"] = _slot(family, "user_text", match.group(0))
            break
        if slots["analysis_family"]["value"]:
            break

    return _finalize(question=text, slots=slots, source="deterministic", notes=[])


def _finalize(
    *,
    question: str,
    slots: Dict[str, Dict[str, Any]],
    source: str,
    notes: List[str],
) -> Dict[str, Any]:
    unread = [name for name in SLOTS if slots[name]["value"] in (None, "")]
    return {
        "ok": True,
        "question": question,
        "slots": {name: slots[name] for name in SLOTS},
        "unread": unread,
        "read_count": len(SLOTS) - len(unread),
        "slot_count": len(SLOTS),
        "source": source,
        "notes": notes,
        # A contract is only runnable once the user has supplied or confirmed
        # everything. This flag exists so no caller can mistake a partial read
        # for a ready study.
        "complete": not unread,
    }


# --------------------------------------------------------------------------
# LLM reader (optional, gated, validated)
# --------------------------------------------------------------------------
_LLM_SYSTEM = (
    "You extract a structured study contract from one ICU research question. "
    "Return STRICT JSON only, no prose. Every field must be present. Use null "
    "for anything the question does not state — never guess, never substitute a "
    "more common study. Do not add fields."
)


def _llm_user_prompt(question: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"population": string|null, "exposure": string|null, '
        '"outcome": string|null, "outcome_type": one of '
        f"{list(OUTCOME_TYPES)}|null, "
        '"time_window_hours": integer|null, "comparator": string|null, '
        f'"analysis_family": one of {list(ANALYSIS_FAMILIES)}|null'
        "}\n"
        "Rules: population/exposure/outcome are short clinical phrases taken "
        "from the question. If the question names no comparator, return null - "
        "do not invent one. If it does not state a time window, return null."
    )


def _validate_llm_slots(payload: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise StudyIntentError({"error": "study_intent_llm_payload_not_object"})
    unknown = sorted(set(payload) - set(SLOTS))
    if unknown:
        raise StudyIntentError(
            {"error": "study_intent_llm_unknown_fields", "fields": unknown}
        )
    slots: Dict[str, Dict[str, Any]] = {name: _empty_slot() for name in SLOTS}
    for name in SLOTS:
        raw = payload.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        if name == "analysis_family":
            text = str(raw).strip().lower()
            if text not in ANALYSIS_FAMILIES:
                raise StudyIntentError(
                    {"error": "study_intent_llm_bad_family", "value": text}
                )
            slots[name] = _slot(text, "llm")
        elif name == "outcome_type":
            text = str(raw).strip().lower()
            if text not in OUTCOME_TYPES:
                raise StudyIntentError(
                    {"error": "study_intent_llm_bad_outcome_type", "value": text}
                )
            slots[name] = _slot(text, "llm")
        elif name == "time_window_hours":
            try:
                hours = int(raw)
            except (TypeError, ValueError) as exc:
                raise StudyIntentError(
                    {"error": "study_intent_llm_bad_window", "value": str(raw)[:40]}
                ) from exc
            if not 1 <= hours <= 24 * 365:
                raise StudyIntentError(
                    {"error": "study_intent_llm_window_out_of_range", "value": hours}
                )
            slots[name] = _slot(hours, "llm")
        else:
            text = str(raw).strip()[:_MAX_SLOT_CHARS]
            if text:
                slots[name] = _slot(text, "llm")
    return slots


def _llm_intent(
    question: str,
    *,
    provider_meta: Dict[str, Any],
    transport: Optional[Callable[[Dict[str, Any], Dict[str, str]], Dict[str, Any]]],
    environ: Optional[Mapping[str, str]],
) -> Dict[str, Dict[str, Any]]:
    credentials = provider_adapter._load_external_credentials(  # noqa: SLF001
        str(provider_meta.get("provider") or ""), environ=environ
    )
    request = {
        "model": credentials["model"],
        "temperature": 0,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": _llm_user_prompt(question)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {credentials['api_key']}",
        "Content-Type": "application/json",
    }
    if transport is None:
        response = provider_adapter._post_chat_completion(  # noqa: SLF001
            url=credentials["base_url"],
            request=request,
            headers=headers,
            timeout=30,
        )
    else:
        response = transport(request, headers)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise StudyIntentError({"error": "study_intent_llm_response_malformed"}) from exc
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StudyIntentError({"error": "study_intent_llm_not_json"}) from exc
    return _validate_llm_slots(payload)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def extract_study_intent(
    question: Any,
    *,
    llm_provider: str = "offline",
    external_llm_opt_in: bool = False,
    ai_enabled: bool = False,
    language: str = "en",
    transport: Optional[Callable[[Dict[str, Any], Dict[str, str]], Dict[str, Any]]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return a typed study-contract proposal for the user's own question.

    The deterministic reader always runs. An external provider is consulted
    only when the canonical AI opt-in gate allows it, and only its *validated*
    output is used; any refusal or malformed answer falls back to the
    deterministic result with the reason recorded in ``notes``.
    """
    text = _clean_question(question)
    baseline = deterministic_intent(text)

    provider_text = str(llm_provider or "offline").strip().lower() or "offline"
    if provider_text in {"offline", "mock", "none", ""}:
        baseline["notes"].append("llm_not_requested")
        return baseline

    try:
        provider_meta = resolve_provider_gate(
            # Intent extraction really does leave the machine, so it is gated
            # as a full external call rather than as a local preflight.
            run_type="full",
            llm_provider=provider_text,
            external_llm_opt_in=external_llm_opt_in,
            ai_enabled=ai_enabled,
            language=language,
        )
    except (ProviderGateError, AIOptInError) as exc:
        baseline["notes"].append("llm_blocked_by_opt_in_gate")
        detail = getattr(exc, "detail", None)
        baseline["provider_block"] = (
            {k: v for k, v in detail.items() if k != "message"}
            if isinstance(detail, dict)
            else {"error": "external_llm_opt_in_required"}
        )
        return baseline

    try:
        slots = _llm_intent(
            text,
            provider_meta=provider_meta,
            transport=transport,
            environ=environ,
        )
    except (StudyIntentError, provider_adapter.ProviderAdapterError) as exc:
        detail = getattr(exc, "detail", {}) or {}
        baseline["notes"].append(
            f"llm_rejected:{detail.get('error') or 'study_intent_llm_failed'}"
        )
        return baseline

    # A deterministic read is grounded directly in the user's own wording.
    # The optional model may fill only unread slots; it may not reinterpret or
    # overwrite a slot that the deterministic reader has already established.
    for name in SLOTS:
        if baseline["slots"][name]["value"] not in (None, ""):
            slots[name] = baseline["slots"][name]

    result = _finalize(question=text, slots=slots, source="llm", notes=[])
    result["provider"] = {
        "provider": provider_meta.get("provider"),
        "external": provider_meta.get("external"),
        "provider_gate": provider_meta.get("provider_gate"),
    }
    return result
