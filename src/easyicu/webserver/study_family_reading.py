"""Which analysis family a question asks for, read from its own words.

Owner of one reading for the Copilot front door, beside the target-trial one
(``causal_trial_reading``): does a researcher's question ask whether two
things are related -- an association study -- or how common something is,
without any adjusted, causal or predictive estimate -- descriptive
epidemiology?  The study setup records the design from this reading
(``study_family_design``), so a study whose question states its family is
planned on that family, never on the planner's keyword routing.

A reading carries the words it rests on.  Each ``evidence`` is a contiguous
substring of the question, never a paraphrase, so the receipt and the study
card show the researcher why the host reads the question as it does.

* ``relationship``: the question asks whether two things are related ("is X
  associated with Y", "the relationship between X and Y", "risk factors for Y",
  "X 是否与 Y 相关", "X 与 Y 是否有关", "X 与 Y 的关系").  "与本研究相关的变量"
  qualifies a noun and asks nothing; an association the question declines
  ("不研究…关系", "not … association") is not asked, and one it asks to
  describe, show or compare ("描述…与死亡的关系", "describe the relationship")
  is a descriptive comparison, as the study setup's descriptive default reads
  a prevalence question that also describes a relationship.
* ``proportion``: the question asks how common something is ("what proportion",
  "prevalence of", "how common", "比例是多少", "多大比例", "患病率", "发生率").
  Comparing groups ("differ between", "有何不同") stays descriptive.

A relationship makes the question an association study, whatever proportions
it also asks for (Table 1 reports them).  A proportion alone makes it
descriptive, unless the question asks for an adjusted, causal or predictive
estimate.  A question about the data themselves (missingness, completeness,
data quality) is read as neither, and so is any other: the host then records
nothing and the study states its family in the conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

__all__ = [
    "FAMILY_READING_ELEMENTS",
    "FAMILY_READING_FAMILIES",
    "QUESTION_WORDING",
    "StudyFamilyEvidence",
    "StudyFamilyReading",
    "family_reading_rests_on",
    "normalize_study_family_reading",
    "study_family_reading",
]

#: The reading's source: the question's own words.
QUESTION_WORDING = "question_wording"
#: The families a question's words may state, by the planner's family keys.
FAMILY_READING_FAMILIES = ("association_study", "descriptive_epidemiology")
#: The cues a reading rests on.
FAMILY_READING_ELEMENTS = ("relationship", "proportion")
_MAX_EVIDENCE_CHARS = 300


@dataclass(frozen=True)
class StudyFamilyEvidence:
    """The words of the question one cue rests on, and where they are."""

    element: str
    start: int
    end: int
    evidence: str


@dataclass(frozen=True)
class StudyFamilyReading:
    """Why the host reads a question as asking for one analysis family."""

    family: str
    elements: Tuple[StudyFamilyEvidence, ...]

    def record(self, *, design_family: Optional[str] = None) -> Dict[str, Any]:
        """The reading as the receipt and the study record state it.

        ``design_family`` is the family the study's design states instead,
        when it differs: the record then says so, and the design stays.
        """

        stored: Dict[str, Any] = {
            "family": self.family,
            "source": QUESTION_WORDING,
            "elements": [
                {"element": item.element, "evidence": item.evidence}
                for item in self.elements
            ],
        }
        if design_family is not None:
            stored["conflict"] = {"design_family": design_family}
        return stored


def normalize_study_family_reading(value: Any) -> Optional[Dict[str, Any]]:
    """The stored shape of a reading (:meth:`StudyFamilyReading.record`); ``None`` clears it.

    Raises ``ValueError`` for any other shape: a closed family and source,
    each element once with non-empty evidence, and a conflict only with
    another family.
    """

    if value is None or value == {}:
        return None
    if not isinstance(value, Mapping) or not (
        {"family", "source", "elements"} <= set(value)
        <= {"family", "source", "elements", "conflict"}
    ):
        raise ValueError("a reading states its family, source and elements, and a conflict")
    if value["family"] not in FAMILY_READING_FAMILIES:
        raise ValueError("a reading's family is an association or a description")
    if value["source"] != QUESTION_WORDING:
        raise ValueError("a reading's source is the question's wording")
    elements = value["elements"]
    if not isinstance(elements, list) or not elements:
        raise ValueError("a reading rests on at least one element")
    stored, seen = [], set()
    for item in elements:
        if not isinstance(item, Mapping) or set(item) != {"element", "evidence"}:
            raise ValueError("an element states its name and its evidence only")
        element, evidence = item["element"], item["evidence"]
        if element not in FAMILY_READING_ELEMENTS or element in seen:
            raise ValueError("each element is a known element, named once")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > _MAX_EVIDENCE_CHARS:
            raise ValueError("an element's evidence is the question's words")
        seen.add(element)
        stored.append({"element": element, "evidence": evidence})
    record: Dict[str, Any] = {
        "family": value["family"],
        "source": QUESTION_WORDING,
        "elements": stored,
    }
    if "conflict" in value:
        conflict = value["conflict"]
        family = conflict.get("design_family") if isinstance(conflict, Mapping) else None
        if (
            not isinstance(conflict, Mapping)
            or set(conflict) != {"design_family"}
            or not isinstance(family, str)
            or not family.strip()
            or family == value["family"]
        ):
            raise ValueError("a conflict names the other family the design states")
        record["conflict"] = {"design_family": family}
    return record


def family_reading_rests_on(record: Mapping[str, Any], question: Any) -> bool:
    """Whether every evidence of a stored reading is words of ``question``."""

    text = str(question or "")
    return all(str(item["evidence"]) in text for item in record["elements"])


_CLAUSE = r"[^，。；;!?？！]"
_EN_CLAUSE = r"[^.;!?]"

# ---------------------------------------------------------------- relationship
_RELATIONSHIP = re.compile(
    r"\b(?:is|are|was|were)\b" + _EN_CLAUSE + r"{1,80}?\bassociated\s+with\b"
    r"|\b(?:is|are|was|were)\b" + _EN_CLAUSE + r"{1,80}?\b(?:linked|related)\s+to\b"
    r"|\bassociations?\s+(?:between|of)\b" + _EN_CLAUSE + r"{1,80}?\b(?:and|with)\b"
    r"|\brelationships?\s+between\b" + _EN_CLAUSE + r"{1,80}?\band\b"
    r"|\brisk\s+factors?\s+for\b"
    r"|是否(?:与|和|同|跟)" + _CLAUSE + r"{1,64}?(?:相关|有关|关联)"
    r"|(?:与|和|同|跟)" + _CLAUSE + r"{1,64}?是否(?:相关|有关|关联|存在关联)"
    r"|(?:与|和|同|跟)" + _CLAUSE + r"{1,64}?的(?:关系|关联)"
    r"|的危险因素",
    re.IGNORECASE,
)
# An association the question declines is not asked.
_DECLINED = re.compile(
    r"\b(?:do\s+not|don't|not|without|no)\b" + _EN_CLAUSE + r"{0,40}?"
    r"\b(?:associat\w*|relationships?)\b"
    r"|(?:不(?:再)?(?:研究|分析|估计|评估|检验|探讨)|无需|不要|不必)"
    + _CLAUSE + r"{0,48}(?:关系|关联|相关)",
    re.IGNORECASE,
)
# A relationship the question asks to describe is a descriptive comparison.
_DESCRIBED = re.compile(
    r"\b(?:describe|summari[sz]e|show|compare)\b" + _EN_CLAUSE + r"{0,40}?"
    r"\b(?:associations?|relationships?)\b"
    r"|(?:描述|展示|呈现|比较)" + _CLAUSE + r"{0,48}?(?:关系|关联)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------- proportion
_PROPORTION = re.compile(
    r"\bwhat\s+(?:proportion|percentage|fraction|share)\b"
    r"|\b(?:prevalence|incidence|proportion)\s+of\b"
    r"|\bhow\s+common\b"
    r"|多大比例|比例(?:是|为)?多少|占比(?:是|为)?多少"
    r"|患病率|发生率|多常见",
    re.IGNORECASE,
)
# A question asking for an adjusted, causal or predictive estimate is not a
# description, whatever proportions it reports.
_INFERENTIAL = re.compile(
    r"\b(?:adjust(?:ed|ing|ment)?|regression|hazard|odds|causal(?:ly|ity)?|"
    r"effects?|predict(?:s|ed|ing|ion|ive)?)\b"
    r"|调整|校正|回归|风险比|优势比|比值比|因果|效应|预测",
    re.IGNORECASE,
)

# ---------------------------------------------------------------- the data
# A question about the data themselves is a data-quality question.
_DATA_QUALITY = re.compile(
    r"\b(?:missing(?:ness)?|completeness|data\s+quality|availability|coverage)\b"
    r"|缺失|完整性|数据质量|可用性|覆盖率",
    re.IGNORECASE,
)


def _spans(pattern: re.Pattern[str], text: str) -> Iterator[re.Match[str]]:
    return pattern.finditer(text)


def _first_outside(
    pattern: re.Pattern[str], text: str, masked: Tuple[Tuple[int, int], ...]
) -> Optional[re.Match[str]]:
    for match in _spans(pattern, text):
        if not any(start <= match.start() < end for start, end in masked):
            return match
    return None


def _evidence(element: str, text: str, match: re.Match[str]) -> StudyFamilyEvidence:
    return StudyFamilyEvidence(
        element=element, start=match.start(), end=match.end(), evidence=text[match.start():match.end()]
    )


def study_family_reading(question: Any) -> Optional[StudyFamilyReading]:
    """The analysis family ``question`` asks for in its own words, or ``None``."""

    text = str(question or "")
    if not text.strip() or _DATA_QUALITY.search(text):
        return None
    declined = tuple(
        match.span() for pattern in (_DECLINED, _DESCRIBED) for match in _spans(pattern, text)
    )
    relationship = _first_outside(_RELATIONSHIP, text, declined)
    proportion = _PROPORTION.search(text)
    if relationship is not None:
        elements = [_evidence("relationship", text, relationship)]
        if proportion is not None:
            elements.append(_evidence("proportion", text, proportion))
        return StudyFamilyReading(
            family="association_study",
            elements=tuple(sorted(elements, key=lambda item: item.start)),
        )
    if proportion is not None and not _INFERENTIAL.search(text):
        return StudyFamilyReading(
            family="descriptive_epidemiology",
            elements=(_evidence("proportion", text, proportion),),
        )
    return None
