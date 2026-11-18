"""What a question explicitly asks its plan to answer, and whether the plan does.

Owner
-----
A question can ask for more than its design: a comparison of the model it
builds with an existing score or model, a named subgroup, a named estimand,
another named analysis.  The family-spec Planner states each such requirement
in the words that state it (``FamilyPlanSpec.question_requirements``).  This
module holds that statement to the question's words and to the run's
concepts, judges after the plan is compiled whether the plan answers each
requirement, and names every one it does not, so a requirement is never
dropped silently.

What the host can verify, it verifies:

* compound methods requests are judged clause by clause against recorded plan
  statements and defaults with provenance; a missing clause alone is reported;
* a question-named exposure needs an analysis owner, rather than a definition.
  A single-exposure family leaves an explicit choice for the plan review;
* typed delivery requirements are covered by the same product-file contract
  the browser downloads read, independently of statistical plan steps. Old
  untyped records use a conservative reader for complete delivery clauses;
* the quote matches a contiguous span of the question (``planning.question_quote``);
* every concept is one the run offers;
* a benchmark is answered only by a step whose action compares models on the
  same rows (``BENCHMARK_ACTIONS``) and reads the benchmark's concept.  A
  concept read as a predictor answers no benchmark.  On the family route only
  the prediction template drafts such a step, for an existing score or
  probability the host offers; without one a benchmark is a capability gap
  there, whatever the Planner claims;
* a subgroup is answered only by a subgroup-capable step (``SUBGROUP_ACTIONS``)
  that reads its concept, with the same rule for a family template, or by an
  analysis step that reports the outcome in each level of that very grouping
  (its exposure-outcome distribution's exposure);
* a requirement of any other kind (an estimand, another analysis) is
  unanswered when no analysis step reads one of its concepts.  A sealed
  owner that analyses concepts through its authority rather than naming them
  in its inputs (the signed trajectory candidate owner fits the sealed
  coordinates' representation) reads them, as the authority's disclosure
  says (``sealed_owner_concepts``).

What it cannot verify, it shows as a claim, never as verified, and a note is
not evidence: an estimand or analysis whose concepts analysis steps read, or
that names none, is attested by the plan (reading a concept is not doing the
analysis the question names); a concept stated only to define another element
is shown with what it defines.

A gap is the capability-gap shape every Planner route declares
(``progressive_contract.ProgressiveCapabilityGap``), here with the element it
concerns and why.  The host checks a declared gap against the study
(``planning.capability_gap``, injected as ``check_gap``): the spec parser
returns one the context contradicts to the Planner, and the record states how
each gap was checked.  A gap the host decides itself is verified by the
method sets above.

Every concept the question names (``NamedQuestionConcept``: the Web reader's
reading, bound to the source's columns, with every column the same name or
the concept dictionary relates to it) must be accounted for by a stated
requirement on the family route, where the spec is refused until it is.  The
outline route states no requirements yet, so there each named concept other
than a sealed coordinate is a visible warning naming the steps that read it.

The coverage rows are the requirement-coverage owner's
(``planning.requirement_coverage``), one per requirement and concept.

The conversation's host-verified ``stated_subgroups`` bind every route through
``stated_subgroup_coverage`` in scientific review, even when the outline states
no requirements. Each needs a subgroup-capable analysis reading the concept;
source gaps and families without a subgroup owner remain explicit stops.

The pipeline shapes a plan after planning (measurement companions, product
references, compiled steps, the step cap), so the plan a review request
offers is judged again (:func:`question_requirements_on_plan_under_review`),
from the planning record alone: it carries everything a judgment reads of the
study.  That judgment is a second record, bound to the digest of the plan it
judged; the planning record stays as written.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    get_args,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..canonical_json import canonical_sha256
from ..contracts.association_execution import primary_binary_effect_measure
from ..contracts.model_tokens import BinaryEffectMeasure
from ..contracts.product_files import (
    DeliverableKind, HOST_DELIVERY_REASON, RUN_DELIVERABLES,
    host_delivery_coverage, requested_deliverable_kinds,
)
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from .progressive_contract import (
    CapabilityGapElement,
    CapabilityGapRequirement,
    ProgressiveCapabilityGap,
)
from .question_quote import (
    QUOTE_NOT_IN_QUESTION,
    QuestionQuoteMatcher,
    QuestionQuoteSpan,
    normalize_question_quote,
)
from .requirement_coverage import (
    ANALYSIS_ROLES,
    SUBGROUP_ACTIONS,
    PlanRequirementCoverage,
    PlanRequirementCoverageRecord,
    RequirementCoverageStatus,
    concept_relatives,
)

QUESTION_REQUIREMENTS_SCHEMA_VERSION = "easyicu.question_requirements/1"
QUESTION_REQUIREMENTS_FILENAME = "question_requirements.json"
#: The judgment of the plan a review request offers.
QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION = "easyicu.question_requirements_review/1"
QUESTION_REQUIREMENTS_REVIEW_FILENAME = "question_requirements_review.json"
MAX_QUESTION_REQUIREMENTS = 6
# One requirement can describe the whole selected design. Its roster can
# include up to 24 variables (including fit coordinates and an endpoint).
MAX_REQUIREMENT_CONCEPTS = 24
MAX_NAMED_QUESTION_CONCEPTS = 8
#: The columns one named concept may denote, the dictionary's relatives included.
MAX_DENOTED_COLUMNS = 8

RequirementKind = Literal["benchmark", "subgroup", "estimand", "analysis", "cohort_accounting", "definition", "delivery"]
CoverageClaim = Literal["plan", "capability_gap", "definition_only", "needs_choice"]
Disposition = Literal[
    "covered", "not_covered", "capability_gap", "attested", "definition_only"
]

#: Actions that compare the study's model with another model or score on the same rows.
BENCHMARK_ACTIONS = frozenset(
    {
        "prediction.benchmark_comparison",
        "prediction.delong_ci",
        "prediction.reclassification",
    }
)
#: Why a plan cannot be approved, one code per remedy: a requirement the plan
#: could answer and does not, and one it cannot answer.  Stable: a published
#: code never changes.
QUESTION_REQUIREMENT_APPROVAL_STOPS: Mapping[str, str] = MappingProxyType(
    {
        "not_covered": "question_requirement_not_covered",
        "capability_gap": "question_requirement_capability_gap",
    }
)
#: Visible, non-blocking records.  Stable codes.
ATTESTED_REASON = "question_requirement_attested"
DEFINITION_REASON = "question_requirement_definition_stated"
UNSTATED_REASON = "question_named_concept_unstated"
#: The family spec is refused until each named concept is accounted for.
UNACCOUNTED_REASON = "question_named_concept_unaccounted"
#: The run's record of the named concepts is the host's defect.
MALFORMED_REASON = "question_named_concepts_malformed"
#: The planning record cannot be read where the plan is offered for review, so
#: that plan is not judged: a fresh plan is the remedy.  Stable.
UNREADABLE_REASON = "question_requirements_unreadable"
#: Every code this owner refuses approval with.
QUESTION_REQUIREMENT_STOP_CODES: tuple[str, ...] = (
    *QUESTION_REQUIREMENT_APPROVAL_STOPS.values(),
    UNREADABLE_REASON,
)

#: The rule the Planner is given, case-neutral.
QUESTION_REQUIREMENTS_GUIDE = (
    "Question requirements: state every analysis or delivery the research question explicitly asks "
    "for beyond the sealed design as one question_requirements entry, quoting the "
    "question's own words: a comparison of the model or estimate with an existing score, "
    "model or estimate (benchmark), a named subgroup (subgroup), a named estimand or "
    "performance measure (estimand), counts of included/excluded rows by eligibility "
    "reason (cohort_accounting), or any other named analysis (analysis). Cohort "
    "accounting is an explicit count-reporting requirement, not definition_only; "
    "it is answered by the typed cohort/attrition owner, not an outcome model. Name the "
    "concepts each one reads. Set coverage to plan when the plan answers it, to "
    "capability_gap with the gap when this plan cannot, and use kind definition with "
    "coverage definition_only for a concept the question names only to define another "
    "element, saying in its note which element. A comparison of the model with another score or model is a benchmark, "
    "never a predictor: do not add the benchmark's concept to the model. Every concept "
    "the question names (question_named_concepts) must appear in some entry, unless it "
    "is the sealed exposure or outcome. Return [] when the question asks for nothing "
    "beyond the sealed design. Copy one contiguous span from the question for each quote."
    " Delivery-only requests use kind delivery and deliverable_kinds from the host "
    "delivery contract: " + ", ".join(RUN_DELIVERABLES) + ". "
    "Quote only the requested outputs, keeping scientific requirements "
    "separate. The host covers these outputs independently of analysis steps; never "
    "declare them infeasible because the statistical plan does not generate files. "
    "For other requirements use deliverable_kinds=[]."
    " Split compound analysis requirements into clauses, each quoting a distinct part of "
    "the requirement's quote; use clauses=[] for an atomic requirement. "
    "Only kind=analysis may carry non-empty clauses; all other kinds use clauses=[]. "
    "Non-empty clauses tile the complete quote in order, with only whitespace or CJK/ASCII punctuation between or after them. "
    "For an analysis of multiple question-named exposures, list every exposure "
    "in analysis_exposures and concepts, including the primary. A definition "
    "does not replace an exposure analysis. If this family cannot carry them "
    "together, use coverage needs_choice: the review will ask whether to analyse "
    "both, keep the current primary with the other secondary, or keep only the "
    "primary. Never choose a narrower question for the user. For other items "
    "use analysis_exposures=[]."
)


#: A primary that reports a measure the question names answers it; a measure
#: the question names that no primary reports, and a risk ratio the question
#: does not name, stop approval as ``question_requirement_not_covered``.  The
#: row's reason says which.  Stable codes.
EFFECT_MEASURE_NOT_REPORTED_REASON = "question_effect_measure_not_reported"
EFFECT_MEASURE_UNREQUESTED_REASON = "question_effect_measure_unrequested"


@dataclass(frozen=True)
class StatedEffectMeasure:
    """A binary effect measure the question explicitly asks for, in its words."""

    measure: BinaryEffectMeasure
    quote: str

    def row(self) -> dict[str, str]:
        return {"measure": self.measure, "quote": self.quote}


# The words that name a measure.  A full name counts wherever it stands, except
# the Chinese names that are also everyday phrases: 风险比 is "risk than" in
# 死亡风险比对照组高, 相对风险 is "relatively risky" in 相对风险较高; those count
# only where they read as a measure.  风险比 next to HR names a hazard ratio.
_MEASURE_NAMES: tuple[tuple[BinaryEffectMeasure, "re.Pattern[str]", bool], ...] = (
    ("risk_ratio", re.compile(r"risk[\s\-]+ratios?|relative[\s\-]+risks?", re.IGNORECASE), False),
    ("risk_ratio", re.compile(r"相对危险度"), False),
    ("risk_ratio", re.compile(r"风险比|相对风险|相对危险"), True),
    ("odds_ratio", re.compile(r"odds[\s\-]+ratios?", re.IGNORECASE), False),
    ("odds_ratio", re.compile(r"比值比|优势比|比数比"), False),
)
# An abbreviation counts only as an uppercase token in a measure's context:
# "or" is a conjunction, "RR" a respiratory rate, "OR" an operating room.
_MEASURE_ABBREVIATION = re.compile(r"(?<![A-Za-z0-9])a?(RR|OR)s?(?![A-Za-z0-9])")
_ABBREVIATED = {"RR": "risk_ratio", "OR": "odds_ratio"}
_CJK_OR_FULLWIDTH = re.compile(r"[、-】㐀-鿿！-？]")
_CLAUSE_END = re.compile(r"[。；;！!？?\n]")
_NEGATED = re.compile(
    r"(?:(?:不|非|勿|别|无需|无须|不必|不需要?|不要|不用|而不是|而非|并非|不是|代替|取代|替代|除了?)"
    r"\s*(?:用|以|采用|使用|选用|报告|计算|估计|表示|给出|呈现)?"
    r"|(?:\bnot|n't|\bno|\bnever|\bwithout|\binstead\s+of|\brather\s+than|\bother\s+than"
    r"|\bavoid(?:ing)?|\bexclud(?:e|ing)|\bexcept)"
    r"(?:\s+(?:an?|the|any|use|using|report|reporting|express|expressing|as|in|terms|of|by|via))*)"
    r"\s*$",
    re.IGNORECASE,
)
_COMPARATIVE_AFTER = re.compile(r"\s*(?:较|更|越|高|低|大|小|增|升|降|减|多|少)")
_HAZARD_NEAR = re.compile(r"^\s*[（(]\s*(?:HR|hazard)", re.IGNORECASE)
_HAZARD_BEFORE = re.compile(r"(?:HR|hazard\s+ratio)\s*[（(]\s*$", re.IGNORECASE)
_MEASURE_CN_BEFORE = re.compile(
    r"(?:用|以|按|计算|报告|估计|给出|呈现|表示为|(?:校正|调整)后?的?|及|和|与|或|、|（|\()\s*$"
)
_MEASURE_CN_AFTER = re.compile(
    r"^\s*(?:$|[（()），,。.；;：:、？?！!]|RR|OR|及|和|与|或|为|是|值|估计|表示|作为)"
)
_RESPIRATORY_BEFORE = re.compile(r"respirat|呼吸", re.IGNORECASE)
_RESPIRATORY_AFTER = re.compile(
    r"^\s*(?:[<>=≤≥＜＞≦≧]|\d|次|breaths?|/\s*min|bpm|间期|间隔|interval)", re.IGNORECASE
)
_OPERATING_ROOM_BEFORE = re.compile(
    r"operating|手术室|(?:\bfrom|\bto|\bin|\binto|\bleaving|\bleft|\bentering|\bentered)\s+(?:the\s+|an?\s+)?$",
    re.IGNORECASE,
)
_ENGLISH_MEASURE_BEFORE = re.compile(
    r"(?:\badjusted|\bunadjusted|\bcrude|\breport(?:ed|ing|s)?|\bexpress(?:ed|ing)?|\bpresent(?:ed|ing)?"
    r"|\bestimat(?:e|ed|es|ing)|\bcalculat(?:e|ed|es|ing)|\bcomput(?:e|ed|es|ing)|\bas|\ban|\bthe"
    r"|\busing|\bwith|\bterms\s+of)\s*$|[(\[]\s*$",
    re.IGNORECASE,
)
_ENGLISH_MEASURE_AFTER = re.compile(
    r"^\s*(?:[)\]]|,?\s*\(?\s*95\s*%|with\s+(?:its\s+)?(?:95|confidence|CI)|and\s+(?:its\s+)?(?:95|confidence|CI)"
    r"|of\b|for\b|value|=)",
    re.IGNORECASE,
)
_MEASURE_JOINER = re.compile(r"^\s*(?:/|,|、|and|or|vs\.?|versus|和|与|或|及)\s*$", re.IGNORECASE)


def _clause_before(text: str, start: int) -> str:
    head = text[max(0, start - 40):start]
    ends = list(_CLAUSE_END.finditer(head))
    return head[ends[-1].end():] if ends else head


def _neighbour(text: str, index: int, step: int) -> str:
    while 0 <= index < len(text) and text[index].isspace():
        index += step
    return text[index] if 0 <= index < len(text) else ""


def _abbreviation_reads_as_measure(text: str, match: "re.Match[str]") -> bool:
    letters = match.group(1)
    before = _clause_before(text, match.start())
    after = text[match.end():match.end() + 24]
    if letters == "RR" and (_RESPIRATORY_BEFORE.search(before) or _RESPIRATORY_AFTER.match(after)):
        return False
    if letters == "OR" and _OPERATING_ROOM_BEFORE.search(before):
        return False
    return bool(
        _CJK_OR_FULLWIDTH.match(_neighbour(text, match.start() - 1, -1))
        or _CJK_OR_FULLWIDTH.match(_neighbour(text, match.end(), 1))
        or _ENGLISH_MEASURE_BEFORE.search(before)
        or _ENGLISH_MEASURE_AFTER.match(after)
    )


def _effect_measure_name(value: Any) -> BinaryEffectMeasure:
    if value not in get_args(BinaryEffectMeasure):
        raise ValueError(f"{value!r} is not a binary effect measure")
    return value


def question_effect_measures(question: str) -> tuple[StatedEffectMeasure, ...]:
    """The binary effect measures a question explicitly asks for, in text order.

    Deterministic and case-neutral; the host reads it from the question itself,
    never from a Planner's statement.  One reading per measure: the first
    mention that is not negated ("not", "rather than", "instead of", 不要,
    而不是, 代替 ...).  A question that names both measures asks for both.
    """

    text = str(question or "")
    found: list[tuple[int, int, BinaryEffectMeasure]] = []
    for measure, pattern, everyday in _MEASURE_NAMES:
        for match in pattern.finditer(text):
            after = text[match.end():match.end() + 24]
            before = _clause_before(text, match.start())
            if everyday and (
                _COMPARATIVE_AFTER.match(after)
                or not (_MEASURE_CN_BEFORE.search(before) or _MEASURE_CN_AFTER.match(after))
            ):
                continue
            if _HAZARD_NEAR.match(after) or _HAZARD_BEFORE.search(before):
                continue
            found.append((match.start(), match.end(), measure))
    candidates = [
        (match.start(), match.end(), _ABBREVIATED[match.group(1)], match)
        for match in _MEASURE_ABBREVIATION.finditer(text)
    ]
    for start, end, measure, match in candidates:
        if _abbreviation_reads_as_measure(text, match):
            found.append((start, end, measure))
    # An abbreviation joined to a measure already read ("OR or RR", "RR/OR")
    # is read as one too.
    for start, end, measure, match in candidates:
        if (start, end, measure) in found:
            continue
        if match.group(1) == "RR" and _RESPIRATORY_AFTER.match(text[end:end + 24]):
            continue
        if any(
            _MEASURE_JOINER.match(text[other_end:start]) or _MEASURE_JOINER.match(text[end:other_start])
            for other_start, other_end, _ in found
        ):
            found.append((start, end, measure))
    readings: dict[str, StatedEffectMeasure] = {}
    for start, end, measure in sorted(found, key=lambda item: (item[0], -item[1])):
        if measure in readings or _NEGATED.search(_clause_before(text, start)):
            continue
        readings[measure] = StatedEffectMeasure(measure, text[start:end])
    return tuple(readings.values())


#: How the host checked a gap: ``planning.capability_gap``'s verdicts.
GapVerification = Literal["verified", "unverified", "unverifiable"]
#: Checks one declared gap against the study: (verification, what the study shows).
GapCheck = Callable[[ProgressiveCapabilityGap], tuple[GapVerification, str]]


def _words(value: str) -> str:
    return " ".join(str(value or "").split())


class QuestionRequirementsError(ValueError):
    """A typed refusal of this owner's inputs; ``reason_code`` names it."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code


class QuestionRequirement(BaseModel):
    """One explicit analysis or delivery requirement, in the question's words."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^r[1-6]$")
    kind: RequirementKind
    quote: str = Field(min_length=2, max_length=240)
    concepts: list[str] = Field(
        default_factory=list, max_length=MAX_REQUIREMENT_CONCEPTS
    )
    coverage: CoverageClaim
    gap: Optional[ProgressiveCapabilityGap] = None
    note: Optional[str] = Field(default=None, max_length=300)
    #: The binary effect measure an estimand asks for; the host judges it by
    #: the measure the plan's primary reports.  Omitted when absent.
    effect_measure: Optional[BinaryEffectMeasure] = Field(
        default=None, exclude_if=lambda value: value is None
    )
    deliverable_kinds: list[DeliverableKind] = Field(
        default_factory=list, max_length=5, exclude_if=lambda value: not value
    )
    clauses: list[str] = Field(
        default_factory=list, max_length=12, exclude_if=lambda value: not value
    )
    analysis_exposures: list[str] = Field(
        default_factory=list, max_length=MAX_REQUIREMENT_CONCEPTS,
        exclude_if=lambda value: not value,
    )

    @field_validator("quote")
    @classmethod
    def _collapse_quote(cls, value: str) -> str:
        return _words(value)

    @field_validator("clauses")
    @classmethod
    def _collapse_clauses(cls, values: list[str]) -> list[str]:
        return [_words(value) for value in values]

    @field_validator("note")
    @classmethod
    def _collapse_note(cls, value: Optional[str]) -> Optional[str]:
        return _words(value) or None if value is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> "QuestionRequirement":
        def invalid_clauses(shape: str) -> None:
            # A model-level ValueError located only the entire requirement.
            # Keep the scientific rule and locate the field the Planner can fix.
            raise ValidationError.from_exception_data(type(self).__name__, [{
                "type": "value_error", "loc": ("clauses",), "input": self.clauses,
                "ctx": {"error": ValueError(shape)},
            }])

        concepts = [str(item).strip() for item in self.concepts]
        if any(not item for item in concepts) or len(set(concepts)) != len(concepts):
            raise ValueError("a question requirement names distinct concepts")
        if (self.coverage == "capability_gap") != (self.gap is not None):
            raise ValueError(
                "a requirement states a gap exactly when its coverage is capability_gap"
            )
        if self.gap is not None and (
            self.gap.element is None or self.gap.detail is None
        ):
            # The stop shows the researcher which part of the design and why.
            raise ValueError(
                "a requirement's gap names the design element it concerns and says why"
            )
        if (self.kind == "definition") != (self.coverage == "definition_only"):
            raise ValueError(
                "a definition is the requirement whose coverage is definition_only"
            )
        if self.kind == "definition" and not concepts:
            raise ValueError("a definition names the concept it defines with")
        if self.kind == "definition" and not self.note:
            raise ValueError("a definition says in its note which element it defines")
        if self.effect_measure is not None and self.kind != "estimand":
            raise ValueError("only an estimand requirement names an effect measure")
        if self.deliverable_kinds and (
            self.kind != "delivery" or len(set(self.deliverable_kinds)) != len(self.deliverable_kinds)
        ):
            raise ValueError("only a delivery requirement names distinct deliverable kinds")
        if self.kind == "delivery" and (concepts or (self.gap and self.gap.concept)):
            raise ValueError("a delivery requirement names outputs, not scientific concepts")
        if self.clauses:
            matcher = QuestionQuoteMatcher(self.quote)
            if (
                self.kind != "analysis"
                or len({normalize_question_quote(clause) for clause in self.clauses}) != len(self.clauses)
                or any(matcher.find(clause) is None for clause in self.clauses)
            ):
                invalid_clauses("analysis clauses must quote distinct parts of their requirement; use distinct verbatim parts of quote for kind=analysis, otherwise []")
            cursor = 0
            gaps = []
            for clause in self.clauses:
                match = matcher.find(clause, offset=cursor)
                if match is None:
                    invalid_clauses("clauses must account for the complete requirement quote in order; use [] for an atomic analysis")
                gaps.append(self.quote[cursor:match.start])
                cursor = match.end
            gaps.append(self.quote[cursor:])
            for index, gap in enumerate(gaps):
                # Keep the established connective/method-request allowance;
                # punctuation itself is Unicode-defined, not an incomplete list.
                gap = _METHOD_REQUEST.sub("", gap.strip()) if index == 0 else gap
                gap = re.sub(r"\b(?:and|or|the|also)\b|以及|及|与|并", "", gap, flags=re.IGNORECASE)
                if any(not char.isspace() and not unicodedata.category(char).startswith("P") for char in gap):
                    invalid_clauses("clauses must account for the complete requirement quote; use [] for an atomic analysis")
        if self.analysis_exposures and (
            self.kind != "analysis"
            or len(set(self.analysis_exposures)) != len(self.analysis_exposures)
            or not set(self.analysis_exposures) <= set(concepts)
        ):
            raise ValueError("analysis exposures must be distinct concepts of that analysis")
        if self.coverage == "needs_choice" and (
            self.kind != "analysis" or len(self.analysis_exposures) < 2
        ):
            raise ValueError("an exposure choice names at least two analysis exposures")
        if (
            self.kind in {"benchmark", "subgroup", "cohort_accounting"}
            and self.coverage == "plan"
            and not concepts
        ):
            raise ValueError(
                f"a {self.kind} the plan answers names the concept it reads"
            )
        return self


class NamedQuestionConcept(BaseModel):
    """A concept the question names: the columns it can denote and the words that name it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concepts: list[str] = Field(min_length=1, max_length=MAX_DENOTED_COLUMNS)
    evidence: str = Field(min_length=1, max_length=120)

    @field_validator("concepts")
    @classmethod
    def _distinct(cls, values: list[str]) -> list[str]:
        cleaned = [str(value or "").strip() for value in values]
        if any(not value for value in cleaned) or len(set(cleaned)) != len(cleaned):
            raise ValueError("a named concept denotes distinct non-empty columns")
        return cleaned


@dataclass(frozen=True)
class RequirementProblem:
    """Why a stated requirement list is refused, with the field it is about."""

    code: str
    path: str
    message: str


def question_requirement_problems(
    requirements: Sequence[QuestionRequirement],
    *,
    question: str,
    roster: Collection[str],
    named: Sequence[NamedQuestionConcept],
    sealed: Collection[str] = (),
) -> tuple[RequirementProblem, ...]:
    """Every way a family spec's requirements fail the question and the run.

    ``named`` concepts already list every column the name or the dictionary
    relates to them (``bind_named_question_concepts``), so a named concept is
    accounted for when a requirement, or a sealed coordinate, reads any of
    them.
    """

    problems: list[RequirementProblem] = []
    if len(requirements) > MAX_QUESTION_REQUIREMENTS:
        problems.append(
            RequirementProblem(
                "question_requirement_count_exceeded",
                "question_requirements",
                f"State at most {MAX_QUESTION_REQUIREMENTS} question requirements.",
            )
        )
    ids = [item.id for item in requirements]
    quotes = [normalize_question_quote(item.quote) for item in requirements]
    if len(set(ids)) != len(ids) or len(set(quotes)) != len(quotes):
        problems.append(
            RequirementProblem(
                "question_requirement_duplicate",
                "question_requirements",
                "Each question requirement has its own id and its own quote.",
            )
        )
    offered = set(roster)
    matcher = QuestionQuoteMatcher(question) if requirements else None
    for index, item in enumerate(requirements):
        path = f"question_requirements[{index}]"
        if matcher is not None and matcher.find(item.quote) is None:
            problems.append(
                RequirementProblem(
                    QUOTE_NOT_IN_QUESTION,
                    f"{path}.quote",
                    f"{path}.quote: {item.quote!r} does not match a question span.",
                )
            )
        # A gap's concept is the host check's (``planning.capability_gap``).
        unknown = [concept for concept in item.concepts if concept not in offered]
        if unknown:
            problems.append(
                RequirementProblem(
                    "question_requirement_concept_unknown",
                    f"{path}.concepts",
                    f"{item.id} names {unknown}, which the run does not offer.",
                )
            )
    for concept in unaccounted_named_concepts(named, requirements, sealed=sealed):
        problems.append(
            RequirementProblem(
                UNACCOUNTED_REASON,
                "question_requirements",
                f"The question names {concept.evidence!r} ({', '.join(concept.concepts)}). "
                "State what it asks of the plan as a question requirement, or that it only "
                "defines another element (kind definition).",
            )
        )
    return tuple(problems)


def unaccounted_named_concepts(
    named: Iterable[NamedQuestionConcept],
    requirements: Iterable[QuestionRequirement],
    *,
    sealed: Collection[str] = (),
) -> tuple[NamedQuestionConcept, ...]:
    """The named concepts no requirement and no sealed coordinate reads."""

    read = {concept for item in requirements for concept in item.concepts}
    read.update(
        item.gap.concept
        for item in requirements
        if item.gap is not None and item.gap.concept
    )
    read.update(str(value) for value in sealed if str(value or "").strip())
    return tuple(item for item in named if not read.intersection(item.concepts))


def bind_named_question_concepts(
    named: Iterable[NamedQuestionConcept],
    *,
    roster: Collection[str],
    relatives: Callable[[str], frozenset[str]],
) -> tuple[NamedQuestionConcept, ...]:
    """Each named concept as the run's columns it can denote, the dictionary's relatives included.

    A name that denotes no offered column is dropped: the plan cannot read it.
    """

    offered = set(roster)
    bound: list[NamedQuestionConcept] = []
    seen: set[frozenset[str]] = set()
    for item in named:
        columns = [
            column
            for column in dict.fromkeys(
                relative
                for concept in item.concepts
                for relative in sorted(relatives(concept))
            )
            if column in offered
        ][:MAX_DENOTED_COLUMNS]
        key = frozenset(columns)
        if not columns or key in seen:
            continue
        seen.add(key)
        bound.append(NamedQuestionConcept(concepts=columns, evidence=item.evidence))
    return tuple(bound[:MAX_NAMED_QUESTION_CONCEPTS])


def named_question_concepts(
    context: ResearchContext,
) -> tuple[NamedQuestionConcept, ...]:
    """The concepts the Web reader read in the question, from the run's data constraints.

    Absent on a run the Web did not launch.  A malformed record is the host's
    defect and is refused with :data:`MALFORMED_REASON`, never read as
    "nothing named".
    """

    raw = (
        getattr(context.user_preferences, "data_constraints", None)
        if context.user_preferences
        else None
    )
    if not raw:
        return ()
    try:
        constraints = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as exc:
        raise QuestionRequirementsError(
            MALFORMED_REASON, "the run's data constraints are not JSON"
        ) from exc
    if not isinstance(constraints, Mapping):
        return ()
    entries = constraints.get("question_named_concepts")
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise QuestionRequirementsError(
            MALFORMED_REASON, "question_named_concepts is not a list"
        )
    try:
        return tuple(NamedQuestionConcept.model_validate(entry) for entry in entries)
    except ValidationError as exc:
        raise QuestionRequirementsError(
            MALFORMED_REASON, "a question_named_concepts entry is not a named concept"
        ) from exc


@dataclass(frozen=True)
class JudgedRequirement:
    """One requirement's disposition in the compiled plan."""

    requirement: QuestionRequirement
    disposition: Disposition
    reason_code: str
    owner_step_ids: tuple[str, ...] = ()
    #: The host decided it, whatever the Planner claimed.
    host_judged: bool = False
    #: The gap that stops the plan (the host's, when the host decided it) and
    #: how the host checked it.
    gap: Optional[ProgressiveCapabilityGap] = None
    verification: Optional[GapVerification] = None
    fact: Optional[str] = None
    #: The analysis steps that read its concepts: what an attested claim rests
    #: on, never an owner.
    reading_step_ids: tuple[str, ...] = ()
    delivery: Optional[Mapping[str, Any]] = None
    clauses: tuple[Mapping[str, Any], ...] = ()
    exposure_choice: Optional[Mapping[str, Any]] = None
    quote_span: QuestionQuoteSpan | None = None

    @property
    def quoted_text(self) -> str:
        return self.quote_span.text if self.quote_span is not None else self.requirement.quote

    @property
    def verified_by_host(self) -> bool:
        """Whether the disposition rests on what the host checked, not on a claim."""

        if self.disposition == "capability_gap":
            return self.verification == "verified"
        return self.disposition in {"covered", "not_covered"}

    def row(self) -> dict[str, Any]:
        item = self.requirement
        return {
            "id": item.id,
            "kind": item.kind,
            "quote": self.quoted_text,
            **({"quote_span": {"start": self.quote_span.start, "end": self.quote_span.end}}
               if self.quote_span is not None else {}),
            "concepts": list(item.concepts),
            "claimed_coverage": item.coverage,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "owner_step_ids": list(self.owner_step_ids),
            "reading_step_ids": list(self.reading_step_ids),
            "host_judged": self.host_judged,
            "verified_by_host": self.verified_by_host,
            "gap": self.gap.model_dump(mode="json") if self.gap is not None else None,
            "gap_verification": self.verification,
            "gap_fact": self.fact,
            "note": item.note,
            # Only a requirement that names a measure carries it, so the rows
            # of every other requirement keep their recorded bytes.
            **({"effect_measure": item.effect_measure} if item.effect_measure else {}),
            **({"delivery": dict(self.delivery)} if self.delivery is not None else {}),
            **({"clauses": list(self.clauses)} if self.clauses else {}),
            **({"exposure_choice": dict(self.exposure_choice)} if self.exposure_choice else {}),
        }


#: The kinds a step's method verifies, and the methods that answer each.
_METHOD_SETS: Mapping[str, frozenset[str]] = MappingProxyType(
    {"benchmark": BENCHMARK_ACTIONS, "subgroup": SUBGROUP_ACTIONS}
)
#: Why a family template cannot answer one of them, as its gap says.
_TEMPLATE_GAPS: Mapping[str, tuple[CapabilityGapElement, str]] = MappingProxyType(
    {
        "benchmark": (
            "comparison",
            "This plan's family template drafted no step that compares the model with "
            "this existing score or model on the same rows.",
        ),
        "subgroup": (
            "analysis",
            "This plan's family template has no step that analyses a named subgroup.",
        ),
    }
)


_METHOD_REQUEST = re.compile(
    r"^(?:明确|说明|列明|写明|定义|交代|(?:specify|state|document|define|clarify)\b)\s*",
    re.IGNORECASE,
)
_METHOD_FACETS = {
    "window": r"时间窗|测量窗|观察窗|\bwindow\b|\btime zero\b",
    "unit": r"单位|\bunits?\b",
    "grading": r"分级|评分规则|\bgrading\b|\bthresholds?\b",
    "covariates": r"协变量|调整集|\bcovariates?\b|\badjustment set\b",
    "repeated_stays": r"重复入住|重复入院|\brepeat(?:ed)?[ -](?:icu[ -])?(?:stay|admission)s?\b",
}
_DEFAULT_CLAIM = re.compile(r"\b(?:host|system)(?:'s)?\s+defaults?\b|(?:宿主|系统)(?:默认|缺省)", re.IGNORECASE)


def requirement_clauses(item: QuestionRequirement) -> tuple[str, ...]:
    """Read explicit clause spans; conservatively support older methods lists.

    Scientific conjunctions are never inferred to be methods statements. An
    unknown member of a methods list is retained, so a familiar clause cannot
    hide an unanswered one.
    """
    if item.kind != "analysis" or item.gap or item.analysis_exposures:
        return ()
    match = _METHOD_REQUEST.match(item.quote)
    if not match or not any(re.search(p, item.quote, re.IGNORECASE) for p in _METHOD_FACETS.values()):
        return tuple(item.clauses)
    text = item.quote[match.end():]
    return tuple(
        clause for part in re.split(r"[、,，;；]|以及|与|及|\band\b", text, maxsplit=11)
        if (clause := part.strip(" 。.;；"))
    )


def question_requirement_host_facts(
    *, context: ResearchContext, plan: AnalysisPlan,
    requirements: Sequence[QuestionRequirement],
) -> tuple[dict[str, Any], ...]:
    """Snapshot declared methods and registered defaults with their provenance.

    These are plan-review facts, not permission to execute a clinical rule.
    A later review needs no mutable context or registry: the source and the
    exact plan on which the default was recorded travel in the planning record.
    """
    if not any(requirement_clauses(item) for item in requirements):
        return ()
    facts: list[dict[str, Any]] = []
    digest = analysis_plan_sha256(plan)

    def add(facet: str, concepts: Sequence[str], source_ref: str, statement: Any) -> None:
        facts.append({
            "facet": facet, "concepts": list(concepts), "source": "host_default",
            "source_ref": source_ref, "statement": statement, "plan_sha256": digest,
        })

    for variable in context.variables:
        source = f"research_context.json.variables[{variable.name}]"
        if variable.unit:
            add("unit", [variable.name], source + ".unit", variable.unit)
        if variable.analysis_window:
            add("window", [variable.name], source + ".analysis_window", variable.analysis_window)
        definition = getattr(variable, "clinical_definition", None)
        if definition and definition.source_id and definition.contract_id:
            add("grading", [variable.name], source + ".clinical_definition", {
                "contract_id": definition.contract_id, "version": definition.version,
                "definition": definition.definition, "source_id": definition.source_id,
                "validation_status": definition.validation_status,
            })
    preferences = context.user_preferences
    if preferences:
        for spec in preferences.sensitivity_specs:
            if spec.strategy == "landmark" and spec.landmark_hours is not None:
                add("window", [context.primary_exposure] if context.primary_exposure else [],
                    f"research_context.json.user_preferences.sensitivity_specs[{spec.spec_id}]",
                    {"landmark_hours": spec.landmark_hours, "strategy": spec.strategy})
        try:
            constraints = json.loads(preferences.data_constraints or "{}")
        except (ValueError, TypeError):
            constraints = {}
        if isinstance(constraints, Mapping):
            design = constraints.get("analysis_design")
            grouping = constraints.get("verified_patient_grouping")
            if (
                isinstance(design, Mapping) and isinstance(grouping, Mapping)
                and design.get("analysis_unit") == "icu_stay"
                and design.get("cluster_unit") == "patient"
                and design.get("variance_estimator") == "cluster_robust"
                and grouping.get("authority_ref")
                and re.fullmatch(r"[0-9a-f]{64}", str(grouping.get("mapping_sha256", "")))
            ):
                add("repeated_stays", [],
                    "research_context.json.user_preferences.data_constraints.analysis_design", {
                    "analysis_design": dict(design), "grouping": dict(grouping),
                    "policy": "retain_each_stay_cluster_by_patient",
                })
    proposal = plan.adjustment_proposal
    if proposal and proposal.source_step_id in {step.step_id for step in plan.steps}:
        add("covariates", [], "analysis_plan.json.adjustment_proposal", proposal.model_dump(mode="json"))
    # Read the existing clinical owner registry, never recreate score thresholds.
    grading_concepts = {
        concept for item in requirements
        if re.search(_METHOD_FACETS["grading"], item.quote, re.IGNORECASE)
        for concept in item.concepts
    }
    recorded_grading = {name for fact in facts if fact["facet"] == "grading" for name in fact["concepts"]}
    if grading_concepts - recorded_grading:
        from easyicu.clinical_contracts import load_clinical_contracts

        for contract in load_clinical_contracts().values():
            columns = [
                variable.name for variable in context.variables
                if variable.name in grading_concepts
                and variable.name not in recorded_grading
                and (variable.source_concept or variable.name) in contract.concepts
            ]
            if columns and contract.source_id and contract.reference_implementation:
                add("grading", columns, f"clinical_contract:{contract.contract_id}@{contract.version}", {
                    "definition": contract.definition, "source_id": contract.source_id,
                    "source_table": contract.source_table,
                    "reference_implementation": contract.reference_implementation,
                    "validation_status": contract.validation_status,
                    "contract_id": contract.contract_id,
                    "depends_on_contracts": list(contract.depends_on_contracts),
                })
    return tuple(facts)


def _fact_matches_clause(fact: Mapping[str, Any], clause: str, facet: str) -> bool:
    """A generic default does not override a value or restriction the user stated."""
    statement = json.dumps(fact["statement"], ensure_ascii=False).casefold()
    if facet == "repeated_stays" and re.search(r"\bfirst\b|首次|首住|第一次", clause, re.IGNORECASE):
        if not isinstance(fact["statement"], Mapping) or fact["statement"].get("policy") != "first_stay_only":
            return False
    values = {float(value) for value in re.findall(r"\d+(?:\.\d+)?", clause)}
    recorded = {float(value) for value in re.findall(
        r"\d+(?:\.\d+)?", statement + " " + " ".join(fact.get("concepts", ())),
    )}
    if not values <= recorded:
        return False
    if facet == "unit":
        unit = r"[a-zµμ]+/[a-zµμ]+"
        requested_units = set(re.findall(unit, clause.casefold()))
        if not requested_units <= set(re.findall(unit, statement)):
            return False
    return True


def _judge_clauses(
    item: QuestionRequirement, *, requirements: Sequence[QuestionRequirement],
    plan: AnalysisPlan, host_facts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    clauses = requirement_clauses(item)
    if not clauses:
        return ()
    rows = []
    digest = analysis_plan_sha256(plan) if host_facts else ""
    for index, clause in enumerate(clauses):
        # A bare "units" inherits the previous window's subject; another
        # variable's unit statement must not accidentally answer this clause.
        sources = [
            other for other in requirements if other.id != item.id and not other.gap
            and len(other.quote) < len(item.quote)
            and other.note and other.coverage in {"plan", "definition_only"}
            and _words(clause).casefold() in _words(other.quote).casefold()
            and (not index or clause.casefold() not in {"unit", "units", "单位"}
                 or not other.concepts
                 or _words(clauses[index - 1]).casefold() in _words(other.quote).casefold())
        ]
        concepts = {concept for source in sources for concept in source.concepts} or set(item.concepts)
        facet = next((key for key, pattern in _METHOD_FACETS.items()
                      if re.search(pattern, clause, re.IGNORECASE)), None)
        candidates = [dict(value) for value in host_facts if (
            facet is not None and value.get("facet") == facet
            and isinstance(value.get("source_ref"), str) and value["source_ref"].strip()
            and value.get("statement") and value.get("plan_sha256") == digest
            and (not value.get("concepts") or concepts.intersection(value["concepts"]))
            and _fact_matches_clause(value, clause, facet)
        )]
        if facet == "grading":
            # Prefer a component's own rule over an aggregate that depends on
            # that same component, without naming any particular score.
            contract_ids = {value["statement"].get("contract_id") for value in candidates
                            if isinstance(value["statement"], Mapping)}
            candidates = [value for value in candidates if not (
                isinstance(value["statement"], Mapping)
                and contract_ids.intersection(value["statement"].get("depends_on_contracts", ()))
            )]
        fact = next(iter(candidates), None)
        stated = next((source for source in sources if not _DEFAULT_CLAIM.search(source.note or "")), None)
        evidence = fact or ({
            "source": "plan_declared", "source_ref": f"question_requirements.json.stated[{stated.id}]",
            "statement": stated.note,
        } if stated else _plan_clause_statement(plan, clause))
        rows.append({
            "quote": clause, "disposition": "covered" if evidence else "not_covered",
            "evidence": evidence,
        })
    return tuple(rows)


def _plan_clause_statement(plan: AnalysisPlan, clause: str) -> dict[str, str] | None:
    """Credit an actual plan statement, retaining the field it was read from."""
    statements = [(f"analysis_plan.json.steps[{step.step_id}].intent", getattr(step, "intent", "")) for step in plan.steps]
    if getattr(plan, "design_selection", None):
        statements.extend(
            (f"analysis_plan.json.design_selection.candidates[{candidate.design_id}].reviewable_plan[{index}]", text)
            for candidate in plan.design_selection.candidates if candidate.disposition == "selected"
            for index, text in enumerate(candidate.reviewable_plan or ())
        )
    for source, statement in statements:
        if _words(clause).casefold() in _words(statement).casefold() and not _DEFAULT_CLAIM.search(statement) and not re.search(
            r"\b(?:no|not|never|cannot|without|unsupported)\b|不(?:做|进行|支持|可)|未(?:计划|安排)",
            statement, re.IGNORECASE,
        ):
            return {"source": "plan_declared", "source_ref": source, "statement": statement}
    return None


def question_analysis_exposures(
    item: QuestionRequirement, requirements: Sequence[QuestionRequirement],
    *, sealed: Sequence[str],
) -> tuple[str, ...]:
    """The exposures this analysis asks to study, retaining old definition demotions.

    New specs name their exposure roles explicitly. The legacy reader requires
    a joint association requirement with both sealed coordinates, plus a
    separate definition-only demotion; it never treats every covariate as an
    exposure merely because a model reads it.
    """
    if item.kind != "analysis" or item.gap:
        return ()
    if len(item.analysis_exposures) > 1:
        return tuple(item.analysis_exposures)
    if len(sealed) != 2 or not set(sealed) <= set(item.concepts):
        return ()
    if not re.search(r"association|relationship|against|关联|关系", item.quote, re.IGNORECASE):
        return ()
    defined = {concept for other in requirements if other.kind == "definition" for concept in other.concepts}
    additional = [concept for concept in item.concepts if concept in defined and concept not in sealed]
    return (sealed[0], *additional) if additional else ()


def _judge_exposures(
    item: QuestionRequirement, exposures: Sequence[str], *, plan: AnalysisPlan,
    sealed: Sequence[str], relatives: Callable[[str], frozenset[str]],
) -> JudgedRequirement:
    outcome = sealed[1] if len(sealed) == 2 else (plan.endpoint.name if plan.endpoint else "")
    owners: dict[str, tuple[str, ...]] = {}
    for exposure in exposures:
        columns = relatives(exposure)
        owners[exposure] = tuple(step.step_id for step in plan.steps if (
            step.planned_analysis_role in ANALYSIS_ROLES
            and any(req.exposure_source in columns and req.outcome == outcome
                    and req.analysis_role == step.planned_analysis_role
                    and req.exposure_source in step.inputs and req.outcome in step.inputs
                    and req.required_for_step_success
                    for req in step.model_requirements)
        ) or (
            len(sealed) == 2 and exposure == sealed[0]
            and step.planned_analysis_role == "primary" and step.runtime_outcome_contract is not None
            and step.runtime_outcome_contract.owner_ref in step.icu_rule_refs
            and outcome in step.runtime_outcome_contract.outcomes
            and str(step.method or "").startswith("signed_")
            and exposure in step.inputs
        ))
    if item.coverage != "needs_choice" and all(owners.values()):
        return JudgedRequirement(item, "covered", "typed_plan_owner_present",
                                 tuple(dict.fromkeys(owner for group in owners.values() for owner in group)),
                                 host_judged=True)
    primary = sealed[0] if sealed and sealed[0] in exposures else exposures[0]
    return JudgedRequirement(
        item, "not_covered", "question_named_exposure_choice_required", host_judged=True,
        exposure_choice={
            "primary_exposure": primary,
            "additional_exposures": [name for name in exposures if name != primary],
            "options": ["analyse_both", "primary_and_secondary", "only_primary"],
            "question": "Analyse both, keep the current primary and the other secondary, or analyse only the current primary?",
        },
    )


def judge_question_requirements(
    requirements: Iterable[QuestionRequirement],
    *,
    plan: AnalysisPlan,
    family_template: bool,
    relatives: Callable[[str], frozenset[str]],
    check_gap: GapCheck,
    sealed_owner_concepts: Mapping[str, Collection[str]] | None = None,
    host_facts: Sequence[Mapping[str, Any]] = (),
    sealed: Sequence[str] = (),
    question: str | None = None,
) -> tuple[JudgedRequirement, ...]:
    """Decide how the compiled plan answers each stated requirement.

    ``sealed_owner_concepts`` maps a sealed owner's method to the concepts it
    analyses through its authority, which its inputs do not name. Methods
    clauses read recorded statements and digest-bound ``host_facts``; they
    never grant an analysis owner to a scientific claim.
    """

    requirements = tuple(requirements)
    judged = tuple(
        _judge(
            item,
            plan=plan,
            family_template=family_template,
            relatives=relatives,
            check_gap=check_gap,
            sealed_owner_concepts=sealed_owner_concepts or {},
            requirements=requirements, host_facts=host_facts, sealed=sealed,
        )
        for item in requirements
    )
    if question is not None and judged:
        matcher = QuestionQuoteMatcher(question)
        judged = tuple(
            replace(entry, quote_span=matcher.find(entry.requirement.quote)) for entry in judged
        )
    return judged


def _judge(
    item: QuestionRequirement,
    *,
    plan: AnalysisPlan,
    family_template: bool,
    relatives: Callable[[str], frozenset[str]],
    check_gap: GapCheck,
    sealed_owner_concepts: Mapping[str, Collection[str]],
    requirements: Sequence[QuestionRequirement],
    host_facts: Sequence[Mapping[str, Any]],
    sealed: Sequence[str],
) -> JudgedRequirement:
    delivery = requirement_delivery_coverage(item)
    if delivery is not None:
        return JudgedRequirement(
            item, "covered", HOST_DELIVERY_REASON, host_judged=True,
            gap=item.gap, verification="unverified" if item.gap else None,
            fact=str(delivery["reason"]), delivery=delivery,
        )
    exposures = question_analysis_exposures(item, requirements, sealed=sealed)
    if exposures:
        return _judge_exposures(item, exposures, plan=plan, sealed=sealed, relatives=relatives)
    clauses = _judge_clauses(item, requirements=requirements, plan=plan, host_facts=host_facts)
    if clauses:
        complete = all(row["disposition"] == "covered" for row in clauses)
        return JudgedRequirement(
            item, "covered" if complete else "not_covered",
            "question_methods_clauses_covered" if complete else QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"],
            host_judged=True, clauses=clauses,
        )
    denoted = [relatives(concept) for concept in item.concepts]
    analysing = [
        step for step in plan.steps if step.planned_analysis_role in ANALYSIS_ROLES
    ]

    def reads(step: Any) -> set[str]:
        method = str(getattr(step, "method", None) or "").strip()
        return {*step.inputs, *sealed_owner_concepts.get(method, ())}

    reading = tuple(
        step.step_id
        for step in analysing
        if any(columns & reads(step) for columns in denoted)
    )
    if item.kind == "definition":
        return JudgedRequirement(
            item, "definition_only", DEFINITION_REASON, reading_step_ids=reading
        )
    if item.effect_measure is not None:
        return _judge_effect_measure(
            item, plan=plan, family_template=family_template,
            check_gap=check_gap, reading=reading,
        )
    if item.kind == "cohort_accounting" and item.gap is None:
        from ..contracts.primary_cohort import primary_analysis_cohort_producer_uses_universe

        # Attrition is produced by the unique typed cohort owner, often an
        # auxiliary step. Keep ordinary auxiliary reads, models and figures
        # outside this authority; neither prose nor an input alone grants it.
        # Its typed row identity is also a consumed concept: accounting does
        # not require treating the identity as an ordinary clinical variable.
        owners = tuple(
            step.step_id for step in plan.steps
            if denoted and step.cohort_definition_spec is not None
            and primary_analysis_cohort_producer_uses_universe(step=step, plan=plan)
            and all(
                columns & (set(step.inputs) | {step.cohort_definition_spec.identity_column})
                for columns in denoted
            )
        )
        return JudgedRequirement(
            item, "covered" if owners else "not_covered",
            "typed_plan_owner_present" if owners else QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"],
            owner_step_ids=owners, host_judged=bool(owners), reading_step_ids=owners,
        )
    actions = _METHOD_SETS.get(item.kind)
    if actions is not None:
        # One comparing (or subgroup) step reads every concept it names.
        owners = tuple(
            step.step_id
            for step in plan.steps
            if denoted
            and step.scientific_action_id in actions
            and all(columns & set(step.inputs) for columns in denoted)
        )
        if owners:
            return JudgedRequirement(
                item, "covered", "typed_plan_owner_present", owners
            )
        if item.kind == "subgroup":
            # "Describe the outcome by age group": the subgroup is the grouping
            # an analysis step reports the outcome in, level by level, and
            # every concept it names denotes that grouping ("age" and its
            # "age_group_x1" alike).
            owners = tuple(
                step.step_id
                for step in analysing
                if denoted
                and step.exposure_outcome_distribution_spec is not None
                and all(
                    step.exposure_outcome_distribution_spec.exposure in columns
                    for columns in denoted
                )
            )
            if owners:
                return JudgedRequirement(
                    item, "covered", "typed_plan_owner_present", owners
                )
        if family_template:
            element, detail = _TEMPLATE_GAPS[item.kind]
            return JudgedRequirement(
                item,
                "capability_gap",
                f"question_{item.kind}_step_unavailable_in_family_template",
                host_judged=True,
                gap=ProgressiveCapabilityGap(
                    requirement="design_element_unsupported",
                    element=element,
                    detail=detail,
                ),
                verification="verified",
                fact=(
                    f"no step of the plan uses {', '.join(sorted(actions))}, "
                    "and its family template drafts none"
                ),
                reading_step_ids=reading,
            )
    if item.gap is not None:
        verification, fact = check_gap(item.gap)
        return JudgedRequirement(
            item,
            "capability_gap",
            "question_requirement_capability_gap_declared",
            gap=item.gap,
            verification=verification,
            fact=fact,
            reading_step_ids=reading,
        )
    if actions is None and all(
        any(columns & reads(step) for step in analysing) for columns in denoted
    ):
        # Reading its concepts is not doing the analysis the question names.
        return JudgedRequirement(
            item, "attested", ATTESTED_REASON, reading_step_ids=reading
        )
    return JudgedRequirement(
        item,
        "not_covered",
        QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"],
        reading_step_ids=reading,
    )


def requirement_delivery_coverage(item: QuestionRequirement) -> dict[str, object] | None:
    """Host-owned outputs bypass scientific gap checks, on every planning boundary."""
    if item.gap is not None and item.gap.concept is not None:
        return None
    # Stored untyped requirements need a conservative compatibility reader.
    # New requirements resolve the typed kind, independently of wording.
    kinds = item.deliverable_kinds or (
        requested_deliverable_kinds(item.quote) if item.kind in {"analysis", "delivery"} else ()
    )
    return host_delivery_coverage(kinds)


def _judge_effect_measure(
    item: QuestionRequirement,
    *,
    plan: AnalysisPlan,
    family_template: bool,
    check_gap: GapCheck,
    reading: tuple[str, ...],
) -> JudgedRequirement:
    """An effect measure is answered by a primary step that reports it.

    The host reads the measure from the primary's typed model contract
    (``primary_binary_effect_measure``), never from a claim.  On the family
    route the spec contract admits a gap only for a family whose primary
    cannot report the measure, so a gap there is the host's verdict.
    """

    owners = tuple(
        step.step_id
        for step in plan.steps
        if step.planned_analysis_role == "primary"
        and primary_binary_effect_measure(step) == item.effect_measure
    )
    if owners:
        return JudgedRequirement(
            item, "covered", "typed_plan_owner_present", owners,
            host_judged=True, reading_step_ids=owners,
        )
    if item.gap is not None:
        if family_template:
            measure = str(item.effect_measure).replace("_", " ")
            return JudgedRequirement(
                item,
                "capability_gap",
                "question_effect_measure_unavailable_in_family_template",
                host_judged=True,
                gap=ProgressiveCapabilityGap(
                    requirement="estimand_unsupported",
                    element="analysis",
                    detail=(
                        f"This plan's family template has no primary model that reports a {measure}."
                    ),
                ),
                verification="verified",
                fact=(
                    f"no primary step of the plan reports a {item.effect_measure}, and "
                    "its family template's primary cannot"
                ),
                reading_step_ids=reading,
            )
        verification, fact = check_gap(item.gap)
        return JudgedRequirement(
            item,
            "capability_gap",
            "question_requirement_capability_gap_declared",
            gap=item.gap,
            verification=verification,
            fact=fact,
            reading_step_ids=reading,
        )
    return JudgedRequirement(
        item,
        "not_covered",
        QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"],
        host_judged=True,
        reading_step_ids=reading,
    )


@dataclass(frozen=True)
class EffectMeasureJudgment:
    """The host's judgment of a binary effect measure on the plan.

    Read from the question (``question_effect_measures``) and from the plan's
    typed primary estimators (``primary_binary_effect_measure``), never from a
    Planner's claim: a measure the question names that no primary reports, and
    a risk ratio the question does not name, stop approval.
    """

    measure: BinaryEffectMeasure
    #: The question's words; ``None`` for a measure the question does not name.
    quote: Optional[str]
    disposition: Literal["covered", "not_covered"]
    reason_code: str
    owner_step_ids: tuple[str, ...] = ()
    #: The question names both measures, and a primary reports one.
    both_named: bool = False

    def row(self) -> dict[str, Any]:
        """A row of the record's ``judged`` table, in a requirement row's shape."""

        return {
            "id": "measure",
            "kind": "estimand",
            "quote": self.quote or "",
            "concepts": [],
            "claimed_coverage": None,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "owner_step_ids": list(self.owner_step_ids),
            "reading_step_ids": [],
            "host_judged": True,
            "verified_by_host": True,
            "gap": None,
            "gap_verification": None,
            "gap_fact": None,
            "note": None,
            "effect_measure": self.measure,
            "read_from_question": self.quote is not None,
        }

    def sentence(self) -> str:
        name = self.measure.replace("_", " ")
        if self.reason_code == EFFECT_MEASURE_UNREQUESTED_REASON:
            return (
                f"Primary step {', '.join(self.owner_step_ids)} reports a {name}, which the "
                "question does not ask for; a binary primary reports the odds ratio unless "
                "the question names the risk ratio."
            )
        return (
            f"The question asks for a {name} ({self.quote!r}), but no primary step reports one"
            + (
                "; a primary model reports one measure, so ask for one of the two."
                if self.both_named
                else "."
            )
        )


def judge_stated_effect_measures(
    stated: Sequence[StatedEffectMeasure],
    *,
    plan: AnalysisPlan,
    requirements: Sequence[QuestionRequirement],
) -> tuple[EffectMeasureJudgment, ...]:
    """Hold the plan's binary primary to the effect measure the question names.

    Only a plan whose primary reports a binary measure is judged: the outcome
    type is the study's.  A measure the question names is answered by a primary
    that reports it; a Planner requirement naming that measure is judged on its
    own (covered, a host-verified gap, or not covered), so it is not judged
    twice.  Otherwise it stops approval.  A risk ratio the question does not
    name stops approval too: the odds ratio is the default and the Planner does
    not switch the estimator on its own.
    """

    reported: dict[str, str] = {}
    for step in plan.steps:
        if step.planned_analysis_role != "primary":
            continue
        measure = primary_binary_effect_measure(step)
        if measure is not None:
            reported[step.step_id] = measure
    if not reported:
        return ()
    named = {item.measure: item for item in stated}
    stated_by_planner = {item.effect_measure for item in requirements if item.effect_measure}
    judged: list[EffectMeasureJudgment] = []
    for measure, item in named.items():
        owners = tuple(step_id for step_id, value in reported.items() if value == measure)
        if owners:
            judged.append(EffectMeasureJudgment(
                measure, item.quote, "covered", "typed_plan_owner_present", owners,
            ))
        elif measure not in stated_by_planner:
            judged.append(EffectMeasureJudgment(
                measure, item.quote, "not_covered", EFFECT_MEASURE_NOT_REPORTED_REASON,
                both_named=len(named) > 1,
            ))
    risk_ratio_owners = tuple(
        step_id for step_id, value in reported.items() if value == "risk_ratio"
    )
    if risk_ratio_owners and "risk_ratio" not in named:
        judged.append(EffectMeasureJudgment(
            "risk_ratio", None, "not_covered", EFFECT_MEASURE_UNREQUESTED_REASON,
            risk_ratio_owners,
        ))
    return tuple(judged)


@dataclass(frozen=True)
class UnstatedConcept:
    """A named concept a route that states no requirements leaves to review."""

    named: NamedQuestionConcept
    #: The steps that read it, as an analysis input or otherwise.
    reading_step_ids: tuple[str, ...]
    #: Whether a cohort criterion reads it.
    cohort_criterion: bool

    def row(self) -> dict[str, Any]:
        return {
            **self.named.model_dump(mode="json"),
            "reading_step_ids": list(self.reading_step_ids),
            "cohort_criterion": self.cohort_criterion,
        }


def outline_route_unstated(
    named: Iterable[NamedQuestionConcept],
    *,
    plan: AnalysisPlan,
    sealed: Collection[str] = (),
) -> tuple[UnstatedConcept, ...]:
    """On a route that states no requirements: every named concept but a sealed coordinate.

    What a step reads says nothing of what the question asks of it (a
    benchmark read as a predictor is read), so each one is a visible warning
    with the steps and cohort criteria that read it, until that route states
    requirements.
    """

    sealed_columns = {str(value) for value in sealed if str(value or "").strip()}
    cohort = getattr(plan, "cohort", None)
    criteria = {
        str(getattr(predicate, "concept_id", None) or "")
        for predicate in (
            *(getattr(cohort, "inclusion", None) or ()),
            *(getattr(cohort, "exclusion", None) or ()),
        )
    }
    return tuple(
        UnstatedConcept(
            named=item,
            reading_step_ids=tuple(
                step.step_id
                for step in plan.steps
                if set(item.concepts) & {str(value) for value in step.inputs}
            ),
            cohort_criterion=bool(set(item.concepts) & criteria),
        )
        for item in named
        if not set(item.concepts) & sealed_columns
    )


def question_requirement_coverage(
    judged: Iterable[JudgedRequirement],
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    measures: Sequence[EffectMeasureJudgment] = (),
) -> PlanRequirementCoverage:
    """The requirement-coverage rows: one per verifiable requirement and concept."""

    from .requirement_coverage import with_requirement_data_availability

    return with_requirement_data_availability(
        _coverage(
            judged, context_sha256=_sha256(context.model_dump(mode="json")), plan=plan,
            measures=measures,
        ),
        context,
    )


def _coverage(
    judged: Iterable[JudgedRequirement],
    *,
    context_sha256: str,
    plan: AnalysisPlan,
    measures: Sequence[EffectMeasureJudgment] = (),
) -> PlanRequirementCoverage:
    status: Mapping[str, RequirementCoverageStatus] = {
        "covered": "covered",
        "not_covered": "missing",
        "capability_gap": "unsupported",
    }
    records: list[PlanRequirementCoverageRecord] = []
    for entry in judged:
        if entry.disposition not in status:
            continue
        item = entry.requirement
        if entry.clauses:
            for index, clause in enumerate(entry.clauses, 1):
                evidence = clause["evidence"]
                records.append(PlanRequirementCoverageRecord(
                    requirement_id=f"question:{item.id}:clause:{index}",
                    kind="question", question_kind=item.kind,
                    concept_identity=f"{item.id}:clause:{index}",
                    source_ref="research_context.json.research_question",
                    status=status[clause["disposition"]],
                    statement_ref=str(evidence["source_ref"]) if evidence else None,
                    reason_code="question_method_clause_stated" if evidence else QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"],
                ))
            continue
        delivery_kinds = entry.delivery["deliverable_kinds"] if entry.delivery else ()
        for concept in delivery_kinds or item.concepts or [item.id]:
            records.append(
                PlanRequirementCoverageRecord(
                    requirement_id=f"question:{item.id}:{concept}",
                    kind="question",
                    question_kind=item.kind,
                    concept_identity=concept,
                    source_ref="research_context.json.research_question",
                    status=status[entry.disposition],
                    owner_step_ids=entry.owner_step_ids
                    if entry.disposition == "covered"
                    else (),
                    reason_code=entry.reason_code,
                    **({"delivery_kind": concept} if delivery_kinds else {}),
                )
            )
    for measure in measures:
        records.append(
            PlanRequirementCoverageRecord(
                requirement_id=f"question:effect_measure:{measure.measure}",
                kind="question",
                question_kind="estimand",
                concept_identity=measure.measure,
                source_ref="research_context.json.research_question",
                status=status[measure.disposition],
                owner_step_ids=measure.owner_step_ids if measure.disposition == "covered" else (),
                reason_code=measure.reason_code,
            )
        )
    return PlanRequirementCoverage(
        context_sha256=context_sha256,
        # The digest a review request's authority binds, as the record's.
        plan_sha256=analysis_plan_sha256(plan),
        records=tuple(records),
        complete=all(record.status == "covered" for record in records),
    )


def question_requirement_findings(
    judged: Sequence[JudgedRequirement],
    *,
    unstated: Sequence[UnstatedConcept] = (),
    measures: Sequence[EffectMeasureJudgment] = (),
) -> list[ValidationFinding]:
    """The approval stops, then the visible records: claims, definitions, unstated names."""

    findings: list[ValidationFinding] = []
    causes = {
        "not_covered": "no step of the plan answers",
        "capability_gap": "this plan cannot answer",
    }
    unrequested = [
        entry for entry in measures if entry.reason_code == EFFECT_MEASURE_UNREQUESTED_REASON
    ]
    for disposition, reason in QUESTION_REQUIREMENT_APPROVAL_STOPS.items():
        rows: list[JudgedRequirement | EffectMeasureJudgment] = [
            entry for entry in judged if entry.disposition == disposition
        ]
        rows += [
            entry for entry in measures
            if entry.disposition == disposition and entry not in unrequested
        ]
        if not rows:
            continue
        findings.append(
            ValidationFinding(
                validator="question_requirements",
                severity="error",
                message=(
                    "This plan cannot be approved: the research question asks for "
                    + ("an analysis " if len(rows) == 1 else "analyses ")
                    + f"that {causes[disposition]}. "
                    + " ".join(
                        entry.sentence()
                        if isinstance(entry, EffectMeasureJudgment)
                        else _stated(entry)
                        for entry in rows
                    )
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason": reason,
                    "human_review_required": True,
                    "approval_allowed": False,
                    "requirements": [row for entry in rows for row in _unanswered_rows(entry)],
                },
            )
        )
    if unrequested:
        # The plan answers with a measure the question did not ask for: the
        # same remedy as an unanswered requirement, a fresh plan.
        findings.append(
            ValidationFinding(
                validator="question_requirements",
                severity="error",
                message=(
                    "This plan cannot be approved: its primary model reports an effect "
                    "measure the research question does not ask for. "
                    + " ".join(entry.sentence() for entry in unrequested)
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason": QUESTION_REQUIREMENT_APPROVAL_STOPS["not_covered"],
                    "reason_code": EFFECT_MEASURE_UNREQUESTED_REASON,
                    "human_review_required": True,
                    "approval_allowed": False,
                    "requirements": [entry.row() for entry in unrequested],
                },
            )
        )
    attested = [entry for entry in judged if entry.disposition == "attested"]
    if attested:
        findings.append(
            ValidationFinding(
                validator="question_requirements",
                severity="warning",
                message=(
                    "The plan states that it answers what the question asks, which the "
                    "host cannot verify: "
                    + "; ".join(
                        f"{entry.requirement.id} {entry.quoted_text!r} "
                        + (
                            f"(read by {', '.join(entry.reading_step_ids)})"
                            if entry.reading_step_ids
                            else "(names no concept)"
                        )
                        for entry in attested
                    )
                    + "."
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason_code": ATTESTED_REASON,
                    "verified_by_host": False,
                    "requirements": [entry.row() for entry in attested],
                },
            )
        )
    defined = [entry for entry in judged if entry.disposition == "definition_only"]
    if defined:
        findings.append(
            ValidationFinding(
                validator="question_requirements",
                severity="warning",
                message=(
                    "The plan states that the question names these only to define "
                    "another element, which the host cannot verify: "
                    + "; ".join(
                        f"{entry.requirement.id} {entry.quoted_text!r} "
                        f"({', '.join(entry.requirement.concepts)}): "
                        f"{entry.requirement.note}"
                        for entry in defined
                    )
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason_code": DEFINITION_REASON,
                    "verified_by_host": False,
                    "requirements": [entry.row() for entry in defined],
                },
            )
        )
    if unstated:
        findings.append(
            ValidationFinding(
                validator="question_requirements",
                severity="warning",
                message=(
                    "This plan states no requirements of the question; check what the "
                    "question asks of each concept it names: "
                    + "; ".join(
                        f"{item.named.evidence!r} ({', '.join(item.named.concepts)}), "
                        + (
                            f"read by {', '.join(item.reading_step_ids)}"
                            if item.reading_step_ids
                            else "read by no step"
                        )
                        + (" and a cohort criterion" if item.cohort_criterion else "")
                        for item in unstated
                    )
                    + "."
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason_code": UNSTATED_REASON,
                    "verified_by_host": False,
                    "named_concepts": [item.row() for item in unstated],
                },
            )
        )
    return findings


def _stated(entry: JudgedRequirement) -> str:
    item = entry.requirement
    if entry.clauses:
        return " ".join(
            f"{item.id} {clause['quote']!r}: no recorded statement or host default answers this clause."
            for clause in entry.clauses if clause["disposition"] != "covered"
        )
    if entry.exposure_choice:
        choice = entry.exposure_choice
        return (f"{item.id} {entry.quoted_text!r}: {choice['question']} "
                f"Primary: {choice['primary_exposure']}; additional: "
                + ", ".join(choice["additional_exposures"]) + ".")
    why = (
        entry.gap.detail
        if entry.gap is not None and entry.gap.detail
        else f"no primary step reports a {item.effect_measure.replace('_', ' ')}"
        if item.effect_measure is not None
        else (
            f"no step that compares models on the same rows reads {', '.join(item.concepts)}"
            if item.kind == "benchmark"
            else (
                f"no subgroup step reads {', '.join(item.concepts)}"
                if item.kind == "subgroup"
                else f"no analysis step reads {', '.join(item.concepts)}"
            )
        )
    )
    # A gap's detail is a sentence of its own: the line ends with one full stop.
    return f"{item.id} {entry.quoted_text!r} ({item.kind}): {why.rstrip(' .。')}."


def _unanswered_rows(entry: JudgedRequirement | EffectMeasureJudgment) -> list[dict[str, Any]]:
    row = entry.row()
    if isinstance(entry, JudgedRequirement) and entry.clauses:
        if "quote_span" in row:
            row["parent_quote_span"] = row.pop("quote_span")
        return [
            {**row, "quote": clause["quote"], "parent_quote": entry.quoted_text}
            for clause in entry.clauses if clause["disposition"] != "covered"
        ]
    return [row]


def analysis_plan_sha256(plan: AnalysisPlan) -> str:
    """The plan's digest as a review request's authority binds it (``PlanReviewAuthority``)."""

    return canonical_sha256(plan.model_dump(mode="json"))


def question_requirements_record(
    *,
    route: Literal["family_template", "outline"],
    plan: AnalysisPlan,
    requirements: Sequence[QuestionRequirement],
    named: Sequence[NamedQuestionConcept],
    judged: Sequence[JudgedRequirement],
    unstated: Sequence[UnstatedConcept],
    coverage: PlanRequirementCoverage,
    denoted: Mapping[str, Collection[str]],
    sealed: Sequence[str],
    stated_measures: Sequence[StatedEffectMeasure] = (),
    measures: Sequence[EffectMeasureJudgment] = (),
    sealed_owner_concepts: Mapping[str, Collection[str]] | None = None,
    host_facts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The run's record of what the question asked and how the compiled plan answers it.

    It carries everything the judgment read of the study (the columns each
    concept denotes, the sealed coordinates, the concepts a sealed owner
    analyses, how each declared gap was checked, the study's digest), so
    another plan is judged from it alone.
    """

    owner_concepts = {
        str(method): sorted({str(concept) for concept in concepts})
        for method, concepts in sorted((sealed_owner_concepts or {}).items())
    }
    return {
        "schema_version": QUESTION_REQUIREMENTS_SCHEMA_VERSION,
        "route": route,
        "compiled_plan_sha256": analysis_plan_sha256(plan),
        "stated": [item.model_dump(mode="json") for item in requirements],
        "named_concepts": [item.model_dump(mode="json") for item in named],
        "denoted_columns": {
            str(concept): sorted(str(column) for column in columns)
            for concept, columns in sorted(denoted.items())
        },
        "sealed": [str(value) for value in sealed],
        # Only a run with a sealed owner carries it, so every other record
        # keeps its bytes; a record without it credits no sealed owner.
        **({"sealed_owner_concepts": owner_concepts} if owner_concepts else {}),
        **({"host_facts": [dict(fact) for fact in host_facts]} if host_facts else {}),
        # Always present on a record this owner writes, so a plan under review
        # is judged on it; a record written before it was read lacks it.
        "question_effect_measures": [item.row() for item in stated_measures],
        "judged": [
            *(entry.row() for entry in judged),
            *(entry.row() for entry in measures),
        ],
        "unstated": [item.row() for item in unstated],
        "coverage": coverage.model_dump(mode="json"),
    }


@dataclass(frozen=True)
class RecordedJudgment:
    """A planning record's requirements judged on one plan, and the record of it."""

    judged: tuple[JudgedRequirement, ...]
    unstated: tuple[UnstatedConcept, ...]
    record: dict[str, Any]
    measures: tuple[EffectMeasureJudgment, ...] = ()

    def findings(self) -> list[ValidationFinding]:
        return question_requirement_findings(
            self.judged, unstated=self.unstated, measures=self.measures
        )


def _gap_key(gap: ProgressiveCapabilityGap) -> str:
    return json.dumps(gap.model_dump(mode="json"), sort_keys=True)


def judge_recorded_requirements(
    record: Mapping[str, Any], *, plan: AnalysisPlan
) -> RecordedJudgment:
    """Judge a planning record's requirements on another plan, from the record alone.

    The same rules judge it, reading nothing of the study but what the record
    carries, so the judgment cannot drift with later changes to the study.  A
    scientific gap keeps the verdict it was given: it is a check of the study,
    not of the plan. Host delivery is resolved against the current export
    contract, including for older records that declared it infeasible.
    A record that cannot be read is refused with
    :data:`UNREADABLE_REASON`.
    """

    try:
        if not isinstance(record, Mapping):
            raise ValueError("it is not an object")
        if record.get("schema_version") != QUESTION_REQUIREMENTS_SCHEMA_VERSION:
            raise ValueError("it is not a record of this owner's schema")
        route = record["route"]
        if route not in ("family_template", "outline"):
            raise ValueError(f"route {route!r} is not a planning route")
        requirements = tuple(
            QuestionRequirement.model_validate(item) for item in record["stated"]
        )
        named = tuple(
            NamedQuestionConcept.model_validate(item)
            for item in record["named_concepts"]
        )
        denoted = {
            str(concept): frozenset({str(concept), *(str(c) for c in columns)})
            for concept, columns in dict(record["denoted_columns"]).items()
        }
        sealed = tuple(str(value) for value in record["sealed"])
        host_facts = record.get("host_facts", [])
        if not isinstance(host_facts, list) or not all(isinstance(fact, Mapping) for fact in host_facts):
            raise ValueError("its host facts are not a list of recorded statements")
        owner_concepts = record.get("sealed_owner_concepts", {})
        if not isinstance(owner_concepts, Mapping) or not all(
            isinstance(concepts, list) for concepts in owner_concepts.values()
        ):
            raise ValueError("its sealed owners' concepts are not lists by method")
        sealed_owner_concepts = {
            str(method): frozenset(str(concept) for concept in concepts)
            for method, concepts in owner_concepts.items()
        }
        verdicts = {
            str(row["id"]): (row["gap_verification"], row["gap_fact"])
            for row in record["judged"]
        }
        quote_spans = {
            row["id"]: QuestionQuoteSpan(text=row["quote"], **row["quote_span"])
            for row in record["judged"] if "quote_span" in row
        }
        read_measures = record.get("question_effect_measures")
        stated_measures = (
            None
            if read_measures is None
            else tuple(
                StatedEffectMeasure(
                    _effect_measure_name(item["measure"]), str(item["quote"])
                )
                for item in read_measures
            )
        )
        context_sha256 = str(record["coverage"]["context_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", context_sha256):
            raise ValueError("its study digest is not a digest")
        record_sha256 = canonical_sha256(dict(record))
        missing = sorted(
            {concept for item in requirements for concept in item.concepts}
            - set(denoted)
        )
        if missing:
            raise ValueError(f"it does not say which columns {missing} denote")
        checks: dict[str, tuple[GapVerification, str]] = {}
        for item in requirements:
            if item.gap is None:
                continue
            verification, fact = verdicts.get(item.id, (None, None))
            if verification not in get_args(GapVerification):
                raise ValueError(f"it does not say how {item.id}'s gap was checked")
            checks[_gap_key(item.gap)] = (verification, str(fact or ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise QuestionRequirementsError(
            UNREADABLE_REASON, f"the planning record cannot be read: {exc}"
        ) from exc
    # Only reading the record is refused as unreadable: a defect in judging
    # is not the record's, and a fresh plan would meet it again.
    judged = judge_question_requirements(
        requirements,
        plan=plan,
        family_template=route == "family_template",
        relatives=lambda column: denoted[column],
        check_gap=lambda gap: checks[_gap_key(gap)],
        sealed_owner_concepts=sealed_owner_concepts,
        host_facts=host_facts, sealed=sealed,
    )
    if quote_spans:
        judged = tuple(
            replace(entry, quote_span=quote_spans.get(entry.requirement.id)) for entry in judged
        )
    unstated = (
        ()
        if route == "family_template"
        else outline_route_unstated(named, plan=plan, sealed=sealed)
    )
    # A record written before the question's measures were read judges none.
    measures = (
        ()
        if stated_measures is None
        else judge_stated_effect_measures(
            stated_measures, plan=plan, requirements=requirements
        )
    )
    coverage = _coverage(judged, context_sha256=context_sha256, plan=plan, measures=measures)
    return RecordedJudgment(
        judged=judged,
        unstated=unstated,
        measures=measures,
        record={
            "schema_version": QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
            "plan_sha256": analysis_plan_sha256(plan),
            "planning_record_sha256": record_sha256,
            "route": route,
            "judged": [
                *(entry.row() for entry in judged),
                *(entry.row() for entry in measures),
            ],
            "unstated": [item.row() for item in unstated],
            "coverage": coverage.model_dump(mode="json"),
        },
    )


def question_requirements_on_plan_under_review(
    findings: Iterable[ValidationFinding],
    *,
    plan: AnalysisPlan,
    run_dir: Path,
) -> list[ValidationFinding]:
    """The plan phase's findings, with this owner's judged again on the plan offered for review.

    Without a planning record nothing was judged and the findings stand.  With
    one, its requirements are judged on ``plan`` and that judgment, in either
    direction, replaces the findings this owner made on the compiled plan; it
    is written beside the planning record, bound to the plan's digest
    (:data:`QUESTION_REQUIREMENTS_REVIEW_FILENAME`).  The same record and plan
    give the same bytes, so deriving the requests again writes nothing new.  A
    planning record that cannot be read refuses approval
    (:data:`UNREADABLE_REASON`).
    """

    given = list(findings)
    kept = [
        finding
        for finding in given
        if getattr(finding, "validator", None) != "question_requirements"
    ]
    source = Path(run_dir) / QUESTION_REQUIREMENTS_FILENAME
    if not source.is_file():
        # The plan phase records whatever it judges; a judgment whose record
        # is gone cannot be made on this plan.
        if len(kept) < len(given):
            return [*kept, _unreadable_finding("the planning record is missing")]
        return given
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [
            *kept,
            _unreadable_finding(f"the planning record cannot be read: {exc}"),
        ]
    try:
        judgment = judge_recorded_requirements(record, plan=plan)
    except QuestionRequirementsError as exc:
        return [*kept, _unreadable_finding(str(exc))]
    _write_record(run_dir, QUESTION_REQUIREMENTS_REVIEW_FILENAME, judgment.record)
    return [*kept, *judgment.findings()]


def _unreadable_finding(cause: str) -> ValidationFinding:
    return ValidationFinding(
        validator="question_requirements",
        severity="error",
        message=(
            "This plan cannot be approved: the run's record of what the research "
            "question asks cannot be read, so the plan offered for review is not "
            "judged against it. Generate the plan again."
        ),
        evidence_ids=["analysis_plan"],
        detail={
            "reason": UNREADABLE_REASON,
            "human_review_required": True,
            "approval_allowed": False,
            "cause": _words(cause)[:300],
        },
    )


def write_question_requirements(run_dir: Path, record: Mapping[str, Any]) -> Path:
    """Write the planning record beside the plan.  It is a run fact, not evidence."""

    return _write_record(run_dir, QUESTION_REQUIREMENTS_FILENAME, record)


def _write_record(run_dir: Path, filename: str, record: Mapping[str, Any]) -> Path:
    target = Path(run_dir) / filename
    target.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def question_requirements_schema(
    concepts: Sequence[str], *, effect_measures: Sequence[str] = ()
) -> dict[str, Any]:
    """The strict transport shape of the requirement list over the offered concepts.

    ``effect_measures`` are offered to an estimand only where a binary primary
    reports one; elsewhere the field is not offered at all.
    """

    names = list(dict.fromkeys(str(value) for value in concepts if str(value))) or [
        "__no_concept__"
    ]
    concept = {"type": "string", "enum": names}
    # The capability-gap shape: its concept is checked by the host, not enumerated.
    gap = {
        "type": "object",
        "additionalProperties": False,
        "required": ["requirement", "concept", "element", "detail"],
        "properties": {
            "requirement": {
                "type": "string",
                "enum": list(get_args(CapabilityGapRequirement)),
            },
            "concept": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "element": {"type": "string", "enum": list(get_args(CapabilityGapElement))},
            "detail": {"type": "string"},
        },
    }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "id", "kind", "quote", "concepts", "coverage", "gap", "note",
                "deliverable_kinds", "clauses", "analysis_exposures",
                *(["effect_measure"] if effect_measures else []),
            ],
            "properties": {
                "id": {
                    "type": "string",
                    "enum": [
                        f"r{index}" for index in range(1, MAX_QUESTION_REQUIREMENTS + 1)
                    ],
                },
                "kind": {
                    "type": "string",
                    "enum": list(get_args(RequirementKind)),
                },
                "quote": {"type": "string"},
                "concepts": {"type": "array", "items": concept, "maxItems": MAX_REQUIREMENT_CONCEPTS},
                "deliverable_kinds": {
                    "type": "array", "items": {"type": "string", "enum": list(get_args(DeliverableKind))},
                },
                "clauses": {
                    "type": "array", "items": {"type": "string"},
                    "description": "For kind=analysis only: clauses tile the complete quote in order, with only whitespace or CJK/ASCII punctuation between or after them; otherwise [].",
                },
                "analysis_exposures": {"type": "array", "items": concept, "maxItems": MAX_REQUIREMENT_CONCEPTS},
                "coverage": {
                    "type": "string",
                    "enum": list(get_args(CoverageClaim)),
                },
                "gap": {"anyOf": [gap, {"type": "null"}]},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                **(
                    {
                        "effect_measure": {
                            "anyOf": [
                                {"type": "string", "enum": list(effect_measures)},
                                {"type": "null"},
                            ]
                        }
                    }
                    if effect_measures
                    else {}
                ),
            },
        },
    }


def question_requirements_shape(
    concepts: Sequence[str], *, effect_measures: Sequence[str] = ()
) -> str:
    """The same shape in words, for a route without a schema."""

    measure = (
        ', "effect_measure": null or one of '
        + json.dumps(list(effect_measures))
        + " (an estimand that asks for one)"
        if effect_measures
        else ""
    )
    return (
        '- "question_requirements": array of 0-6 objects, each exactly {"id": "r1".."r6", '
        '"kind": "benchmark"|"subgroup"|"estimand"|"analysis"|"cohort_accounting"|"definition"|"delivery", "quote": "<the '
        f'question\'s own words, 2-240 characters>", "concepts": [0-{MAX_REQUIREMENT_CONCEPTS} distinct names from '
        + json.dumps(list(concepts), ensure_ascii=False)
        + '], "deliverable_kinds": [0-5 distinct kinds from '
        + json.dumps(list(get_args(DeliverableKind)))
        + '], "clauses": [distinct quoted parts of this requirement, or []], '
        '"analysis_exposures": [distinct exposure names from concepts, or []], '
        '"coverage": "plan"|"capability_gap"|"definition_only"|"needs_choice", "gap": null or '
        '{"requirement": <one of '
        + ", ".join(get_args(CapabilityGapRequirement))
        + '>, "concept": "<the threshold variable or stated adjustment>" or null, "element": <one '
        "of "
        + ", ".join(get_args(CapabilityGapElement))
        + '>, "detail": "<why this plan cannot, 8-300 characters>"}, "note": null or "<at '
        'most 300 characters>"'
        + measure
        + '}. "gap" is set exactly when coverage is capability_gap, and '
        "its concept only for levels_from_thresholds_unavailable or a stated adjustment's "
        "design_element_unsupported; kind definition goes with "
        "coverage definition_only and a note saying which element it defines; a benchmark "
        "or subgroup answered by the plan names its concept. "
        "Only kind=analysis may carry non-empty clauses; otherwise clauses=[]. "
        "Clauses tile the complete quote in order, with only whitespace or CJK/ASCII punctuation between or after them."
    )


def _sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "ANALYSIS_ROLES",
    "ATTESTED_REASON",
    "BENCHMARK_ACTIONS",
    "DEFINITION_REASON",
    "MALFORMED_REASON",
    "MAX_NAMED_QUESTION_CONCEPTS",
    "MAX_QUESTION_REQUIREMENTS",
    "QUESTION_REQUIREMENTS_FILENAME",
    "QUESTION_REQUIREMENTS_GUIDE",
    "QUESTION_REQUIREMENTS_REVIEW_FILENAME",
    "QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION",
    "QUESTION_REQUIREMENTS_SCHEMA_VERSION",
    "QUESTION_REQUIREMENT_APPROVAL_STOPS",
    "QUESTION_REQUIREMENT_STOP_CODES",
    "EFFECT_MEASURE_NOT_REPORTED_REASON",
    "EFFECT_MEASURE_UNREQUESTED_REASON",
    "SUBGROUP_ACTIONS",
    "UNACCOUNTED_REASON",
    "UNREADABLE_REASON",
    "UNSTATED_REASON",
    "GapCheck",
    "GapVerification",
    "JudgedRequirement",
    "NamedQuestionConcept",
    "QuestionRequirement",
    "QuestionRequirementsError",
    "EffectMeasureJudgment",
    "RecordedJudgment",
    "RequirementProblem",
    "StatedEffectMeasure",
    "UnstatedConcept",
    "analysis_plan_sha256",
    "bind_named_question_concepts",
    "concept_relatives",
    "judge_question_requirements",
    "judge_recorded_requirements",
    "judge_stated_effect_measures",
    "named_question_concepts",
    "outline_route_unstated",
    "question_requirement_coverage",
    "question_requirement_findings",
    "question_requirement_host_facts",
    "question_analysis_exposures",
    "requirement_clauses",
    "question_effect_measures",
    "question_requirement_problems",
    "question_requirements_on_plan_under_review",
    "question_requirements_record",
    "question_requirements_schema",
    "question_requirements_shape",
    "requirement_delivery_coverage",
    "unaccounted_named_concepts",
    "write_question_requirements",
]
