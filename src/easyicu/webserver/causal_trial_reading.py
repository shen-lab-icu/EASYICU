"""Whether a question asks a target trial's causal question, read from its words.

Owner of one decision for the Copilot front door: does a researcher's question
ask what would happen if a treatment were started within a grace period after
a time zero, rather than not, for a fixed-horizon death -- the shape of a
target trial -- or does it name a causal method outright?  The study setup
records the causal design from this reading (``causal_trial_design``), and the
research launch stops a study without one whose question it reads as causal,
so the workflow and the planner never decide the family each by a rule of
its own.

A reading carries, element by element, the words of the question it rests
on.  Each ``evidence`` is a contiguous substring of the question, never a
paraphrase, so the receipt and the trial card show the researcher why the host
treats the question as a trial.

``question_trial_shape`` needs three elements together:

* a numeric time zero ("以入 ICU 后第 6 小时为时间零点", "time zero at hour 6");
* two start strategies in one comparison: start the treatment within a grace
  period of G hours, against not starting it within those hours, or starting
  it only after them ("24 小时内开始 X，与这 24 小时内不开始", "24 小时后才开始").
  The second is read as the trial's one comparison strategy, defer: not
  started within the grace period, unrestricted after it;
* a fixed-horizon death a closed endpoint admits (``outcome_availability``).

Two bounded start windows ("6 小时内开始" against "6 到 24 小时之间开始"), and an
early-against-late comparison without hours, are not this shape: the v1 trial
emulates start against not-start within one grace period.  A risk difference
or ratio is not required.

``question_causal_method`` reads a question that names a causal method (a
target trial, a causal effect, propensity scores, inverse probability
weighting, ...).  A question that disclaims a causal reading, or asks only for
an association or a description, is read as neither, whatever else it says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Literal, Mapping, Optional, Tuple

from easyicu.outcome_availability import (
    fixed_horizon_mortality_endpoint_stated_by,
    stated_mortality_horizon_mentions,
)

__all__ = [
    "CAUSAL_TRIAL_ELEMENTS",
    "QUESTION_CAUSAL_METHOD",
    "QUESTION_TRIAL_SHAPE",
    "CausalTrialEvidence",
    "CausalTrialReading",
    "causal_trial_reading",
    "normalize_causal_trial_reading",
    "reading_rests_on",
    "time_zero_spans",
]

QUESTION_TRIAL_SHAPE = "question_trial_shape"
QUESTION_CAUSAL_METHOD = "question_causal_method"
CausalTrialSource = Literal["question_trial_shape", "question_causal_method"]

#: The elements a reading may rest on, in the order a reading lists them.
CAUSAL_TRIAL_ELEMENTS: Tuple[str, ...] = (
    "time_zero",
    "initiate_strategy",
    "defer_strategy",
    "fixed_horizon_outcome",
    "causal_method",
)


@dataclass(frozen=True)
class CausalTrialEvidence:
    """The words of the question one element rests on, and where they are."""

    element: str
    start: int
    end: int
    evidence: str


@dataclass(frozen=True)
class CausalTrialReading:
    """Why the host reads a question as a target trial's causal question."""

    source: str
    elements: Tuple[CausalTrialEvidence, ...]
    time_zero_hours: Optional[float] = None
    grace_period_hours: Optional[float] = None
    #: Where the initiate strategy names what it starts.
    treatment: Optional[Tuple[int, int]] = None

    def record(self) -> Dict[str, Any]:
        """The reading as the receipt, the stop and the study record state it."""

        return {
            "source": self.source,
            "elements": [
                {"element": item.element, "evidence": item.evidence}
                for item in self.elements
            ],
        }


_SOURCES = (QUESTION_TRIAL_SHAPE, QUESTION_CAUSAL_METHOD)
_MAX_EVIDENCE_CHARS = 300


def normalize_causal_trial_reading(value: Any) -> Optional[Dict[str, Any]]:
    """The stored shape of a reading (:meth:`CausalTrialReading.record`); ``None`` clears it.

    Raises ``ValueError`` for any other shape: a closed source, each element
    once, each with non-empty evidence.
    """

    if value is None or value == {}:
        return None
    if not isinstance(value, Mapping) or set(value) != {"source", "elements"}:
        raise ValueError("a reading states its source and its elements only")
    if value["source"] not in _SOURCES:
        raise ValueError("a reading's source is a trial shape or a causal method")
    elements = value["elements"]
    if not isinstance(elements, list) or not elements:
        raise ValueError("a reading rests on at least one element")
    stored, seen = [], set()
    for item in elements:
        if not isinstance(item, Mapping) or set(item) != {"element", "evidence"}:
            raise ValueError("an element states its name and its evidence only")
        element, evidence = item["element"], item["evidence"]
        if element not in CAUSAL_TRIAL_ELEMENTS or element in seen:
            raise ValueError("each element is a known element, named once")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > _MAX_EVIDENCE_CHARS:
            raise ValueError("an element's evidence is the question's words")
        seen.add(element)
        stored.append({"element": element, "evidence": evidence})
    return {"source": value["source"], "elements": stored}


def reading_rests_on(record: Mapping[str, Any], question: Any) -> bool:
    """Whether every evidence of a stored reading is words of ``question``."""

    text = str(question or "")
    return all(str(item["evidence"]) in text for item in record["elements"])


_NUMBER = r"\d+(?:\.\d+)?"
_SENTENCE_END = re.compile(r"[。；;？?！!]|\.(?=\s|$)")

# ---------------------------------------------------------------- disclaimers
# A question that disclaims a causal reading, or asks only for an association
# or a description, is not read as causal.  "Not starting" is a strategy, not
# a disclaimer, so a negation followed by a start verb is not one.
_DISCLAIMER = re.compile(
    r"\b(?:do\s+not|don't|not|avoid|without)\b"
    r"(?!\s+(?:start|initiat|begin|receiv|giv|administer))"
    r"[^.;?!。；？！]{0,40}\bcausal(?:ity|ly)?\b"
    r"|\b(?:rather\s+than|instead\s+of|as\s+opposed\s+to)\b[^.;?!。；？！]{0,30}"
    r"\bcausal(?:ity|ly)?\b"
    r"|\bnon[-_\s]?causal\b"
    r"|\b(?:association|descriptive)\s+only\b"
    r"|\bonly\s+(?:an?\s+)?(?:association|descriptive|description)\b"
    r"|\bpurely\s+(?:descriptive|associational)\b"
    r"|(?:不(?:作|做|进行|用于|支持|解释为?)|避免|无意)[^。；？！]{0,24}因果"
    r"|因果[^。；？！]{0,16}(?:不成立|不支持|不解释)"
    r"|(?:而非|不是|并非)[^。；？！]{0,12}因果|非因果"
    r"|(?:不得|禁止|拒绝)[^。；？！]{0,24}因果"
    r"|(?:只|仅|只是|仅仅)(?:做|作|进行|分析|研究|看|估计)?(?:观察性)?(?:关联|相关性|描述)",
    re.IGNORECASE,
)

# ------------------------------------------------------------- causal method
# The methods whose naming asks for a causal estimate.  The planner's own
# keyword routing (``analysis_types.infer_analysis_type``) also reads weaker
# cues ("positivity", "covariate balance", "weighted estimate"); a question
# only those name is not read here, so the host writes no design from them,
# and the launch stops it with the trial entry instead
# (``causal_trial_design``).
_CAUSAL_METHOD = re.compile(
    r"target[\s-]+trial(?:\s+emulation)?|trial\s+emulation"
    r"|\bcausal(?:ly)?(?:\s+(?:effect|effects|inference|difference|estimate))?\b"
    r"|treatment\s+effects?|propensity(?:\s+scores?)?|\biptw?\b"
    r"|inverse\s+probability(?:\s+(?:of\s+treatment\s+)?weight(?:ing|s)?)?"
    r"|g-?formula|g-?computation|instrumental\s+variables?|marginal\s+structural"
    r"|clone[\s-]+censor|doubly\s+robust|\btmle\b"
    r"|目标试验(?:模拟)?|(?:模拟)?试验模拟|因果(?:效应|差异|推断|分析|关系)?"
    r"|治疗效应|处理效应|倾向(?:性)?评分|逆概率(?:加权)?|工具变量|边际结构|双重稳健",
    re.IGNORECASE,
)

# ------------------------------------------------------------------ time zero
_ZH_HOUR = (
    r"(?:入\s*(?:ICU|重症监护(?:病房|室)?|监护室|科)\s*(?:后\s*)?)?第?\s*"
    rf"(?P<hours>{_NUMBER})\s*(?:个\s*)?小时"
)
_EN_AFTER = r"(?:\s+(?:after|from|post)\s+(?:icu\s+)?admission)?"
_EN_ZERO = r"time[\s-]*zero"
_EN_LINK = r"(?:(?:is|was|at|=|of|set\s+(?:at|to)|defined\s+as|placed\s+at)\s+)+"
_TIME_ZERO: Tuple["re.Pattern[str]", ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"(?P<ev>{_ZH_HOUR})\s*(?:作为|为|设为|定为|是)\s*(?:时间)?零点",
        rf"(?:时间)?零点\s*(?:为|是|设为|定为|取|在)\s*(?P<ev>{_ZH_HOUR})",
        rf"{_EN_ZERO}\s+{_EN_LINK}(?P<ev>hour\s+(?P<hours>{_NUMBER}){_EN_AFTER})",
        rf"{_EN_ZERO}\s+{_EN_LINK}(?P<ev>(?P<hours>{_NUMBER})\s*(?:h|hrs?|hours?)\b{_EN_AFTER})",
        rf"(?P<ev>hour\s+(?P<hours>{_NUMBER}){_EN_AFTER})\s+(?:as|is|was)\s+(?:the\s+)?{_EN_ZERO}",
        rf"(?P<ev>(?P<hours>{_NUMBER})\s*(?:h|hrs?|hours?)\s+(?:after|from|post)\s+"
        rf"(?:icu\s+)?admission)\s+(?:as|is|was)\s+(?:the\s+)?{_EN_ZERO}",
    )
)

# ----------------------------------------------------------------- strategies
_ZH_START = r"(?:开始|启动|起始|给予|使用|接受|应用)"
_ZH_STOP = r"，,。；;？?！!、与和及或对"
_INITIATE_ZH = re.compile(
    rf"(?P<ev>(?P<grace>{_NUMBER})\s*(?:个\s*)?小时\s*(?:内|以内|之内)\s*{_ZH_START}\s*"
    rf"(?P<treatment>[^{_ZH_STOP}\s]+(?:\s+[^{_ZH_STOP}\s]+)*?))"
    rf"(?=\s*(?:[{_ZH_STOP}]|vs\b|versus\b|相比|比较|$))",
    re.IGNORECASE,
)
_EN_START = (
    r"(?:start(?:s|ed|ing)?|initiat(?:e|es|ed|ing)|begin(?:s|ning)?|began"
    r"|receiv(?:e|es|ed|ing)|giv(?:e|es|en|ing)|administer(?:s|ed|ing)?)"
)
_EN_WORD = (
    r"(?!(?:compar\w*|whether|vs|versus|and|or|with|of|the|a|an|to|for|who|that"
    r"|in|on|patients?|stays?)\b)[a-z][a-z0-9\-]*"
)
_EN_GRACE = (
    r"(?:within|in\s+the\s+first|during\s+the\s+first|in\s+the\s+next|during\s+the\s+next)"
    rf"\s+(?P<grace>{_NUMBER})\s*(?:h|hrs?|hours?)\b"
)
_INITIATE_EN = (
    re.compile(
        rf"(?P<ev>{_EN_START}\s+(?P<treatment>{_EN_WORD}(?:\s+{_EN_WORD}){{0,3}}?)\s+{_EN_GRACE})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<ev>(?P<treatment>{_EN_WORD}(?:\s+{_EN_WORD}){{0,2}}?)\s+"
        rf"(?:started|initiated|begun|given|administered)\s+{_EN_GRACE})",
        re.IGNORECASE,
    ),
)
# The second strategy: not started within the grace period, or only after it.
_DEFER = (
    re.compile(
        rf"(?P<ev>(?:(?:这|该|此|同样的?)\s*)?(?:(?P<grace>{_NUMBER})\s*(?:个\s*)?小时\s*"
        rf"(?:内|以内|之内)\s*)?(?:不|未|没有|暂不|不予)\s*{_ZH_START})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<ev>(?P<grace>{_NUMBER})\s*(?:个\s*)?小时\s*(?:后|以后|之后)\s*(?:才|再)?\s*{_ZH_START}"
        rf"|推迟(?:到|至)\s*(?P<grace_b>{_NUMBER})\s*(?:个\s*)?小时\s*(?:后|以后|之后)"
        rf"(?:\s*(?:才|再)?\s*{_ZH_START})?"
        rf"|(?:推迟|延迟)\s*{_ZH_START})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<ev>(?:not|never)\s+(?:start(?:ing)?|initiat(?:e|ing)|begin(?:ning)?"
        r"|receiv(?:e|ing)|giv(?:e|ing)|administer(?:ing)?)\b(?:\s+(?:it|them|treatment))?"
        rf"(?:\s+(?:within|during|in)\s+(?:th(?:at|ose|is|e\s+same)\s+)?(?:(?P<grace>{_NUMBER})"
        r"\s*(?:h|hrs?|hours?)|window|period|grace\s+period))?"
        r"|defer(?:ring|red|ral\s+of)?(?:\s+(?:it|them|treatment|initiation))?"
        r"|withhold(?:ing)?(?:\s+(?:it|them|treatment))?"
        r"|delay(?:ing|ed)?\s+(?:it|them|treatment|initiation|the\s+start))",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<ev>(?:start(?:ing|ed)?|initiat(?:e|ing|ed|ion)|begin(?:ning)?)\s+(?:only\s+)?"
        rf"(?:after|beyond)\s+(?P<grace>{_NUMBER})\s*(?:h|hrs?|hours?)\b)",
        re.IGNORECASE,
    ),
)
# A comparison joins the two strategies.
_COMPARISON = re.compile(
    r"与|和|对比|对|相比|或|\bvs\b\.?|\bversus\b|\bcompared\s+(?:with|to)\b|\bagainst\b|\bor\b",
    re.IGNORECASE,
)
# A second strategy that starts within a bounded later window is a different
# comparison: two start windows, which the v1 trial does not emulate.
_BOUNDED_WINDOW = re.compile(
    rf"{_NUMBER}\s*(?:个\s*)?小时?\s*(?:到|至|-|–|—|~)\s*{_NUMBER}\s*(?:个\s*)?小时"
    rf"|between\s+{_NUMBER}\s*(?:h|hrs?|hours?)?\s+and\s+{_NUMBER}\s*(?:h|hrs?|hours?)",
    re.IGNORECASE,
)
_RANGE_BEFORE = re.compile(r"\d\s*(?:个\s*)?(?:小时)?\s*(?:到|至|-|–|—|~)\s*$")


def _sentence(text: str, position: int) -> Tuple[int, int]:
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.end() <= position:
            start = match.end()
        else:
            return start, match.start()
    return start, len(text)


def _evidence(element: str, match: "re.Match[str]", group: str = "ev") -> CausalTrialEvidence:
    start, end = match.span(group)
    return CausalTrialEvidence(element, start, end, match.group(group))


def _same(first: float, second: Optional[str]) -> bool:
    return second is None or float(second) == first


def time_zero_spans(text: str) -> Tuple[Tuple[int, int], ...]:
    """Where ``text`` states a numeric time zero; its hours are no window."""

    return tuple(sorted({m.span() for p in _TIME_ZERO for m in p.finditer(str(text or ""))}))


def _time_zero(text: str) -> Optional[Tuple[CausalTrialEvidence, float]]:
    found = sorted(
        (match for pattern in _TIME_ZERO for match in pattern.finditer(text)),
        key=lambda match: match.start(),
    )
    if not found:
        return None
    match = found[0]
    return _evidence("time_zero", match), float(match.group("hours"))


def _initiations(text: str) -> Iterator["re.Match[str]"]:
    for pattern in (_INITIATE_ZH, *_INITIATE_EN):
        for match in pattern.finditer(text):
            # "6 到 24 小时内开始" bounds the start on both sides: a window, not
            # a grace period from time zero.
            if _RANGE_BEFORE.search(text[: match.start("grace")]):
                continue
            yield match


def _strategies(
    text: str,
) -> Optional[Tuple[CausalTrialEvidence, CausalTrialEvidence, float, Tuple[int, int]]]:
    for initiate in sorted(_initiations(text), key=lambda match: match.start()):
        grace = float(initiate.group("grace"))
        _, sentence_end = _sentence(text, initiate.start())
        rest = text[initiate.end() : sentence_end]
        if _BOUNDED_WINDOW.search(rest):
            return None
        seconds = sorted(
            (
                match
                for pattern in _DEFER
                for match in pattern.finditer(text, initiate.end(), sentence_end)
            ),
            key=lambda match: match.start(),
        )
        for second in seconds:
            if not _COMPARISON.search(text[initiate.end() : second.start()]):
                continue
            hours = second.groupdict().get("grace") or second.groupdict().get("grace_b")
            if not _same(grace, hours):
                return None
            return (
                _evidence("initiate_strategy", initiate),
                _evidence("defer_strategy", second),
                grace,
                initiate.span("treatment"),
            )
    return None


def _fixed_horizon_outcome(text: str) -> Optional[CausalTrialEvidence]:
    for mention in stated_mortality_horizon_mentions(text):
        if fixed_horizon_mortality_endpoint_stated_by(mention.horizon) is not None:
            return CausalTrialEvidence(
                "fixed_horizon_outcome",
                mention.start,
                mention.end,
                text[mention.start : mention.end],
            )
    return None


def _causal_method(text: str) -> Optional[CausalTrialEvidence]:
    match = _CAUSAL_METHOD.search(text)
    if match is None:
        return None
    return CausalTrialEvidence("causal_method", match.start(), match.end(), match.group(0))


def causal_trial_reading(question: Any) -> Optional[CausalTrialReading]:
    """Read ``question`` as a target trial's causal question, or ``None``.

    A trial-shaped question is read as ``question_trial_shape``, with the
    causal method it also names, if any; otherwise a question that names a
    causal method is read as ``question_causal_method``, with whatever trial
    elements it states.  Every evidence is a substring of ``question``.
    """

    text = str(question or "").strip()
    if not text or _DISCLAIMER.search(text):
        return None
    zero = _time_zero(text)
    strategies = _strategies(text)
    outcome = _fixed_horizon_outcome(text)
    method = _causal_method(text)
    if zero is not None and strategies is not None and outcome is not None:
        source = QUESTION_TRIAL_SHAPE
    elif method is not None:
        source = QUESTION_CAUSAL_METHOD
    else:
        return None
    found = [
        *((zero[0],) if zero is not None else ()),
        *(strategies[:2] if strategies is not None else ()),
        *((outcome,) if outcome is not None else ()),
        *((method,) if method is not None else ()),
    ]
    elements = tuple(sorted(found, key=lambda item: CAUSAL_TRIAL_ELEMENTS.index(item.element)))
    if any(text[item.start : item.end] != item.evidence for item in elements):
        return None
    return CausalTrialReading(
        source=source,
        elements=elements,
        time_zero_hours=zero[1] if zero is not None else None,
        grace_period_hours=strategies[2] if strategies is not None else None,
        treatment=strategies[3] if strategies is not None else None,
    )
