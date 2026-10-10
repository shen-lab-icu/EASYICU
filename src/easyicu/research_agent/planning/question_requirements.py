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

* the quote is written in the question (``population_spec.quote_written_in``);
* every concept is one the run offers;
* a benchmark is answered only by a step whose action compares models on the
  same rows (``BENCHMARK_ACTIONS``) and reads the benchmark's concept.  A
  concept read as a predictor answers no benchmark.  On the family route only
  the prediction template drafts such a step, for an existing score or
  probability the host offers; without one a benchmark is a capability gap
  there, whatever the Planner claims;
* a subgroup is answered only by a subgroup-capable step (``SUBGROUP_ACTIONS``)
  that reads its concept, with the same rule for a family template;
* a requirement of any other kind (an estimand, another analysis) is
  unanswered when no analysis step reads one of its concepts.

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
from dataclasses import dataclass
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
from ..schema import AnalysisPlan, ResearchContext, ValidationFinding
from .population_spec import quote_written_in
from .progressive_contract import (
    CapabilityGapElement,
    CapabilityGapRequirement,
    ProgressiveCapabilityGap,
)
from .requirement_coverage import (
    PlanRequirementCoverage,
    PlanRequirementCoverageRecord,
    RequirementCoverageStatus,
)

QUESTION_REQUIREMENTS_SCHEMA_VERSION = "easyicu.question_requirements/1"
QUESTION_REQUIREMENTS_FILENAME = "question_requirements.json"
#: The judgment of the plan a review request offers.
QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION = "easyicu.question_requirements_review/1"
QUESTION_REQUIREMENTS_REVIEW_FILENAME = "question_requirements_review.json"
MAX_QUESTION_REQUIREMENTS = 6
MAX_REQUIREMENT_CONCEPTS = 4
MAX_NAMED_QUESTION_CONCEPTS = 8
#: The columns one named concept may denote, the dictionary's relatives included.
MAX_DENOTED_COLUMNS = 8

RequirementKind = Literal["benchmark", "subgroup", "estimand", "analysis", "definition"]
CoverageClaim = Literal["plan", "capability_gap", "definition_only"]
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
#: Actions that analyse a named subgroup.
SUBGROUP_ACTIONS = frozenset(
    {
        "association.effect_modification",
        "prediction.subgroup_fairness",
        "time_to_event.subgroup_hr",
    }
)
#: The step roles that analyse.  Cohort accounting, Table 1, audits, figures
#: and the report are auxiliary: reading a concept there answers nothing.
ANALYSIS_ROLES = frozenset({"primary", "secondary", "sensitivity"})

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
    "Question requirements: state every analysis the research question explicitly asks "
    "for beyond the sealed design as one question_requirements entry, quoting the "
    "question's own words: a comparison of the model or estimate with an existing score, "
    "model or estimate (benchmark), a named subgroup (subgroup), a named estimand or "
    "performance measure (estimand), or any other named analysis (analysis). Name the "
    "concepts each one reads. Set coverage to plan when the plan answers it, to "
    "capability_gap with the gap when this plan cannot, and use kind definition with "
    "coverage definition_only for a concept the question names only to define another "
    "element, saying in its note which element. A comparison of the model with another score or model is a benchmark, "
    "never a predictor: do not add the benchmark's concept to the model. Every concept "
    "the question names (question_named_concepts) must appear in some entry, unless it "
    "is the sealed exposure or outcome. Return [] when the question asks for nothing "
    "beyond the sealed design."
)


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
    """One analysis the question asks for, in its words."""

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

    @field_validator("quote")
    @classmethod
    def _collapse_quote(cls, value: str) -> str:
        return _words(value)

    @field_validator("note")
    @classmethod
    def _collapse_note(cls, value: Optional[str]) -> Optional[str]:
        return _words(value) or None if value is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> "QuestionRequirement":
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
        if (
            self.kind in {"benchmark", "subgroup"}
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
    quotes = [_words(item.quote).casefold() for item in requirements]
    if len(set(ids)) != len(ids) or len(set(quotes)) != len(quotes):
        problems.append(
            RequirementProblem(
                "question_requirement_duplicate",
                "question_requirements",
                "Each question requirement has its own id and its own quote.",
            )
        )
    offered = set(roster)
    for index, item in enumerate(requirements):
        path = f"question_requirements[{index}]"
        if not quote_written_in(item.quote, "question", (question,)):
            problems.append(
                RequirementProblem(
                    "question_requirement_quote_not_in_question",
                    f"{path}.quote",
                    f"{item.id} quotes {item.quote!r}, which the research question does "
                    "not say: quote its own words.",
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


def concept_relatives(context: ResearchContext) -> Callable[[str], frozenset[str]]:
    """The run's columns the concept dictionary relates to a column.

    A column's identity is its name, its source concept and the concepts it
    derives from; two columns are related when their identities meet.
    """

    identities = {
        str(variable.name): frozenset(
            str(value)
            for value in (
                variable.name,
                getattr(variable, "source_concept", None),
                *(getattr(variable, "derived_from_concepts", None) or ()),
            )
            if str(value or "").strip()
        )
        for variable in context.variables
    }

    def relatives(column: str) -> frozenset[str]:
        own = identities.get(column, frozenset({column}))
        return frozenset(
            {column, *(name for name, ids in identities.items() if ids & own)}
        )

    return relatives


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
            "quote": item.quote,
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


def judge_question_requirements(
    requirements: Iterable[QuestionRequirement],
    *,
    plan: AnalysisPlan,
    family_template: bool,
    relatives: Callable[[str], frozenset[str]],
    check_gap: GapCheck,
) -> tuple[JudgedRequirement, ...]:
    """Decide how the compiled plan answers each stated requirement."""

    return tuple(
        _judge(
            item,
            plan=plan,
            family_template=family_template,
            relatives=relatives,
            check_gap=check_gap,
        )
        for item in requirements
    )


def _judge(
    item: QuestionRequirement,
    *,
    plan: AnalysisPlan,
    family_template: bool,
    relatives: Callable[[str], frozenset[str]],
    check_gap: GapCheck,
) -> JudgedRequirement:
    denoted = [relatives(concept) for concept in item.concepts]
    analysing = [
        step for step in plan.steps if step.planned_analysis_role in ANALYSIS_ROLES
    ]
    reading = tuple(
        step.step_id
        for step in analysing
        if any(columns & set(step.inputs) for columns in denoted)
    )
    if item.kind == "definition":
        return JudgedRequirement(
            item, "definition_only", DEFINITION_REASON, reading_step_ids=reading
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
        any(columns & set(step.inputs) for step in analysing) for columns in denoted
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
) -> PlanRequirementCoverage:
    """The requirement-coverage rows: one per verifiable requirement and concept."""

    return _coverage(
        judged, context_sha256=_sha256(context.model_dump(mode="json")), plan=plan
    )


def _coverage(
    judged: Iterable[JudgedRequirement],
    *,
    context_sha256: str,
    plan: AnalysisPlan,
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
        for concept in item.concepts or [item.id]:
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
) -> list[ValidationFinding]:
    """The approval stops, then the visible records: claims, definitions, unstated names."""

    findings: list[ValidationFinding] = []
    causes = {
        "not_covered": "no step of the plan answers",
        "capability_gap": "this plan cannot answer",
    }
    for disposition, reason in QUESTION_REQUIREMENT_APPROVAL_STOPS.items():
        rows = [entry for entry in judged if entry.disposition == disposition]
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
                    + " ".join(_stated(entry) for entry in rows)
                ),
                evidence_ids=["analysis_plan"],
                detail={
                    "reason": reason,
                    "human_review_required": True,
                    "approval_allowed": False,
                    "requirements": [entry.row() for entry in rows],
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
                        f"{entry.requirement.id} {entry.requirement.quote!r} "
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
                        f"{entry.requirement.id} {entry.requirement.quote!r} "
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
    why = (
        entry.gap.detail
        if entry.gap is not None and entry.gap.detail
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
    return f"{item.id} {item.quote!r} ({item.kind}): {why.rstrip(' .。')}."


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
) -> dict[str, Any]:
    """The run's record of what the question asked and how the compiled plan answers it.

    It carries everything the judgment read of the study (the columns each
    concept denotes, the sealed coordinates, how each declared gap was
    checked, the study's digest), so another plan is judged from it alone.
    """

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
        "judged": [entry.row() for entry in judged],
        "unstated": [item.row() for item in unstated],
        "coverage": coverage.model_dump(mode="json"),
    }


@dataclass(frozen=True)
class RecordedJudgment:
    """A planning record's requirements judged on one plan, and the record of it."""

    judged: tuple[JudgedRequirement, ...]
    unstated: tuple[UnstatedConcept, ...]
    record: dict[str, Any]

    def findings(self) -> list[ValidationFinding]:
        return question_requirement_findings(self.judged, unstated=self.unstated)


def _gap_key(gap: ProgressiveCapabilityGap) -> str:
    return json.dumps(gap.model_dump(mode="json"), sort_keys=True)


def judge_recorded_requirements(
    record: Mapping[str, Any], *, plan: AnalysisPlan
) -> RecordedJudgment:
    """Judge a planning record's requirements on another plan, from the record alone.

    The same rules judge it, reading nothing of the study but what the record
    carries, so the judgment cannot drift with later changes to the study.  A
    declared gap keeps the verdict it was given: it is a check of the study,
    not of the plan.  A record that cannot be read is refused with
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
        verdicts = {
            str(row["id"]): (row["gap_verification"], row["gap_fact"])
            for row in record["judged"]
        }
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
    )
    unstated = (
        ()
        if route == "family_template"
        else outline_route_unstated(named, plan=plan, sealed=sealed)
    )
    coverage = _coverage(judged, context_sha256=context_sha256, plan=plan)
    return RecordedJudgment(
        judged=judged,
        unstated=unstated,
        record={
            "schema_version": QUESTION_REQUIREMENTS_REVIEW_SCHEMA_VERSION,
            "plan_sha256": analysis_plan_sha256(plan),
            "planning_record_sha256": record_sha256,
            "route": route,
            "judged": [entry.row() for entry in judged],
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


def question_requirements_schema(concepts: Sequence[str]) -> dict[str, Any]:
    """The strict transport shape of the requirement list over the offered concepts."""

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
            "required": ["id", "kind", "quote", "concepts", "coverage", "gap", "note"],
            "properties": {
                "id": {
                    "type": "string",
                    "enum": [
                        f"r{index}" for index in range(1, MAX_QUESTION_REQUIREMENTS + 1)
                    ],
                },
                "kind": {
                    "type": "string",
                    "enum": [
                        "benchmark",
                        "subgroup",
                        "estimand",
                        "analysis",
                        "definition",
                    ],
                },
                "quote": {"type": "string"},
                "concepts": {"type": "array", "items": concept},
                "coverage": {
                    "type": "string",
                    "enum": ["plan", "capability_gap", "definition_only"],
                },
                "gap": {"anyOf": [gap, {"type": "null"}]},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
    }


def question_requirements_shape(concepts: Sequence[str]) -> str:
    """The same shape in words, for a route without a schema."""

    return (
        '- "question_requirements": array of 0-6 objects, each exactly {"id": "r1".."r6", '
        '"kind": "benchmark"|"subgroup"|"estimand"|"analysis"|"definition", "quote": "<the '
        'question\'s own words, 2-240 characters>", "concepts": [0-4 distinct names from '
        + json.dumps(list(concepts), ensure_ascii=False)
        + '], "coverage": "plan"|"capability_gap"|"definition_only", "gap": null or '
        '{"requirement": <one of '
        + ", ".join(get_args(CapabilityGapRequirement))
        + '>, "concept": "<the variable to group by thresholds>" or null, "element": <one '
        "of "
        + ", ".join(get_args(CapabilityGapElement))
        + '>, "detail": "<why this plan cannot, 8-300 characters>"}, "note": null or "<at '
        'most 300 characters>"}. "gap" is set exactly when coverage is capability_gap, and '
        "its concept only for levels_from_thresholds_unavailable; kind definition goes with "
        "coverage definition_only and a note saying which element it defines; a benchmark "
        "or subgroup answered by the plan names its concept"
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
    "RecordedJudgment",
    "RequirementProblem",
    "UnstatedConcept",
    "analysis_plan_sha256",
    "bind_named_question_concepts",
    "concept_relatives",
    "judge_question_requirements",
    "judge_recorded_requirements",
    "named_question_concepts",
    "outline_route_unstated",
    "question_requirement_coverage",
    "question_requirement_findings",
    "question_requirement_problems",
    "question_requirements_on_plan_under_review",
    "question_requirements_record",
    "question_requirements_schema",
    "question_requirements_shape",
    "unaccounted_named_concepts",
    "write_question_requirements",
]
