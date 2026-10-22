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
* ``adjustment``: the question asks for a comparison adjusted for something
  ("请调整性别", "adjusted for age and sex", "控制年龄后").  A description
  adjusts for nothing, so an adjusted comparison is an association study --
  unless the question also asks for a prediction or a causal estimate, which
  the planner's own routing reads, or for a proportion, which may be a
  standardized rate.

A question that states another design in its own words -- survival
("用 Cox 回归"), prediction, a causal estimate or subtype discovery, as the
planner's own routing reads it -- is read as neither, whatever relationship
or proportion it also names.  Otherwise a relationship makes the question an
association study, whatever proportions it also asks for (Table 1 reports them).  A proportion alone makes it
descriptive, unless the question asks for an adjusted, causal or predictive
estimate.  A question about the data themselves (missingness, completeness,
data quality) is read as neither, and so is any other: the host then records
nothing and the study states its family in the conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from easyicu.research_agent.planning.analysis_types import (
    INFERENTIAL_ESTIMATE,
    PROPORTION_QUESTION,
    RELATIONSHIP_QUESTION,
    data_quality_question,
    mask_non_association_phrases,
)

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
FAMILY_READING_ELEMENTS = ("relationship", "proportion", "adjustment")
# "不调整"/"未校正"/"无需控制" and "without adjusting for" decline it.
_ADJUSTMENT_REQUEST = re.compile(
    r"(?<![不未无非需])(?:调整|校正|矫正|控制)(?:了|后)?[^。！？；;!?，,]{1,40}"
    r"|\badjust(?:ed|ing)?\s+for\b|\bafter\s+adjust(?:ment|ing)\s+for\b"
    r"|\bcontroll?(?:ed|ing)\s+for\b",
    re.IGNORECASE,
)
# A prediction or a causal estimate also adjusts; the planner's routing reads it.
_OTHER_ADJUSTED_FAMILY = re.compile(
    r"预测|因果|倾向评分|目标试验|\bpredict\w*|\bcausal\w*|\bpropensity\b|\btarget\s+trial\b",
    re.IGNORECASE,
)
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


def _evidence(element: str, text: str, match: re.Match[str]) -> StudyFamilyEvidence:
    return StudyFamilyEvidence(
        element=element, start=match.start(), end=match.end(), evidence=text[match.start():match.end()]
    )


#: Families a question states by its design or method words ("用 Cox 回归",
#: "预测", "因果", "聚类成亚型").  The planner's own routing reads them, and a
#: relationship, an adjustment or a proportion in the same question does not
#: make it an association study or a description: recording one would plan
#: another analysis than the one asked for (pilot1 10-22: "用 Cox 回归分析…
#: 与 28 天生存的关系" was recorded as an association study).
_SELF_STATED_FAMILIES = frozenset({
    "survival", "prediction_model", "dynamic_prediction", "causal_inference",
    "trajectory_clustering",
})


def _planner_reads_a_stated_design(text: str) -> bool:
    from easyicu.research_agent.planning.analysis_types import infer_analysis_type
    from easyicu.research_agent.schema import CohortDescriptor, ResearchContext

    context = ResearchContext(
        research_question=text,
        cohort=CohortDescriptor(cohort_name="study", database="miiv", n_stays=0),
        variables=[],
    )
    return infer_analysis_type(context).key in _SELF_STATED_FAMILIES


def study_family_reading(question: Any) -> Optional[StudyFamilyReading]:
    """The analysis family ``question`` asks for in its own words, or ``None``."""

    text = str(question or "")
    if not text.strip() or data_quality_question(text):
        return None
    reading = _reading(text)
    if reading is not None and _planner_reads_a_stated_design(text):
        return None
    return reading


def _reading(text: str) -> Optional[StudyFamilyReading]:
    relationship = RELATIONSHIP_QUESTION.search(mask_non_association_phrases(text))
    proportion = PROPORTION_QUESTION.search(text)
    if relationship is not None:
        elements = [_evidence("relationship", text, relationship)]
        if proportion is not None:
            elements.append(_evidence("proportion", text, proportion))
        return StudyFamilyReading(
            family="association_study",
            elements=tuple(sorted(elements, key=lambda item: item.start)),
        )
    adjustment = next(
        (
            match for match in _ADJUSTMENT_REQUEST.finditer(text)
            if not re.search(r"\b(?:without|not|no|un)\s*$", text[max(0, match.start() - 12):match.start()], re.IGNORECASE)
        ),
        None,
    )
    # An adjusted proportion ("校正年龄后的死亡比例") may be a standardized
    # rate, which states no family; an adjusted comparison does.
    if adjustment is not None and proportion is None and not _OTHER_ADJUSTED_FAMILY.search(text):
        return StudyFamilyReading(
            family="association_study",
            elements=(_evidence("adjustment", text, adjustment),),
        )
    if proportion is not None and not INFERENTIAL_ESTIMATE.search(text):
        return StudyFamilyReading(
            family="descriptive_epidemiology",
            elements=(_evidence("proportion", text, proportion),),
        )
    return None
