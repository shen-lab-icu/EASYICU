"""Opt-in comparison beside the live v1 judgment, with no approval side effects."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from ...schema import AnalysisPlan, ResearchContext
    from ..question_quote import QuestionQuoteMatcher, QuestionQuoteSpan
    from ..question_requirements import JudgedRequirement
    from .contract import Decision, FacetName, PartResult
    from .intake import RepairField

FLAG = "EASYICU_REQUIREMENT_COVERAGE_V2"
SHADOW_METRICS_FIELD = "requirement_coverage_v2_shadow"
_SHADOW_SCHEMA = "easyicu.requirement_coverage_v2_shadow/1"


class _StateCounts(TypedDict):
    covered: int
    not_covered: int
    unknown: int


class _DisagreementCounts(TypedDict):
    true_false_pass: int
    prose_proven: int
    v2_under_typing: int


class ShadowSummary(TypedDict):
    schema_version: str
    proposal_count: int
    parts_total: int
    by_facet: dict[str, _StateCounts]
    by_language: dict[str, _StateCounts]
    disagreements: _DisagreementCounts
    proposal_rejections: dict[str, int]


class ShadowFailure(TypedDict):
    schema_version: str
    proposal_count: int
    shadow_error: Literal["shadow_evaluation_failed"]


def _proposal_count(proposals: object) -> int:
    return len(proposals) if isinstance(proposals, list) else 0


def shadow_failure(proposals: object) -> ShadowFailure:
    return {
        "schema_version": _SHADOW_SCHEMA,
        "proposal_count": _proposal_count(proposals),
        "shadow_error": "shadow_evaluation_failed",
    }


def typing_shadow_enabled() -> bool:
    return os.getenv(FLAG) == "shadow"


@dataclass(frozen=True)
class _Observation:
    part_id: str
    quote: str | None
    v1_state: str
    v1_reason: str
    facet_hint: FacetName | None = None
    parent_quote: str | None = None
    output_id: str | None = None


def _requirements(judged: Sequence[JudgedRequirement]) -> Iterable[_Observation]:
    from ...contracts.product_files import requested_deliverable_kinds
    from .source import has_source_alternatives, repeat_policy_in_source

    for entry in judged:
        item = entry.requirement
        stay_policy = repeat_policy_in_source(item.quote, "retain_each_stay_cluster_by_patient")
        if entry.clauses and not stay_policy:
            for index, clause in enumerate(entry.clauses, 1):
                evidence = clause.get("evidence") or {}
                source = evidence.get("source")
                yield _Observation(
                    part_id=item.id if len(entry.clauses) == 1 else f"{item.id}/clause/{index}",
                    quote=item.quote if len(entry.clauses) == 1 else clause["quote"],
                    v1_state=clause["disposition"],
                    v1_reason=source if source in ("host_default", "plan_declared") else entry.reason_code,
                    parent_quote=item.quote if len(entry.clauses) > 1 else None,
                )
            continue
        outputs = requested_deliverable_kinds(item.quote) if item.kind == "delivery" else ()
        if outputs and not has_source_alternatives(item.quote):
            for output_id in outputs:
                yield _Observation(
                    part_id=item.id if len(outputs) == 1 else f"{item.id}/delivery/{output_id}",
                    quote=item.quote,
                    v1_state=entry.disposition, v1_reason=entry.reason_code,
                    facet_hint="delivery", output_id=output_id,
                )
            continue
        yield _Observation(
            part_id=item.id, quote=item.quote, v1_state=entry.disposition,
            v1_reason=entry.reason_code,
            facet_hint=("delivery" if item.kind == "delivery" else "method" if item.effect_measure
                        else "repeat_stay_policy" if stay_policy else None),
        )


def _clause_spans(
    source: QuestionQuoteMatcher, parent_quote: str, observations: Sequence[_Observation],
) -> dict[str, QuestionQuoteSpan]:
    from ..question_quote import normalize_question_quote

    container = source.find(parent_quote)
    if container is None or source.find(parent_quote, offset=container.start + 1):
        return {}

    # Resolve longer clauses first so a shorter prefix can use a separate
    # occurrence when one exists.
    quotes = {item.part_id: normalize_question_quote(item.quote or "") for item in observations}
    spans: dict[str, QuestionQuoteSpan] = {}
    used: set[tuple[int, int]] = set()
    for item in sorted(observations, key=lambda item: len(quotes[item.part_id]), reverse=True):
        if not item.quote:
            continue
        offset = container.start
        overlap = None
        while (span := source.find(item.quote, offset=offset)) and span.end <= container.end:
            offset = span.start + 1
            bounds = (span.start, span.end)
            if bounds in used:
                continue
            if all(span.end <= start or end <= span.start for start, end in used):
                break
            if overlap is None:
                overlap = span
        else:
            span = overlap
        if span is not None:
            spans[item.part_id] = span
            used.add((span.start, span.end))

    # The preferred non-overlapping occurrence may come after an overlapping
    # one. Assign each group's matched occurrences in declared order.
    groups: dict[str, list[str]] = defaultdict(list)
    for item in observations:
        groups[quotes[item.part_id]].append(item.part_id)
    for part_ids in groups.values():
        matched = sorted(
            (spans.pop(part_id) for part_id in part_ids if part_id in spans),
            key=lambda span: span.start,
        )
        spans.update(zip(part_ids[:len(matched)], matched, strict=True))
    return spans


@dataclass(frozen=True)
class _ComparedPart:
    v1: str
    v1_reason: str
    decision: Decision
    facet: FacetName | None
    language: Literal["en", "zh"]
    result: PartResult | None
    repair: tuple[RepairField, ...]
    source_bound_miss: bool


@dataclass(frozen=True)
class _Comparison:
    parts: tuple[_ComparedPart, ...]
    repair: tuple[RepairField, ...]


def _observe(
    observations: Sequence[_Observation], *,
    plan: AnalysisPlan, context: ResearchContext | None, question: str | None,
    proposals: object,
) -> _Comparison:
    from ...schema import AnalysisPlan
    from ..family_spec.plan_language import plan_language
    from ..question_quote import QuestionQuoteMatcher
    from .contract import AtomicPart, PartResult, SourceSpan, Unknown
    from .facts import project_facts
    from .intake import TypingIntake
    from .registry import registered_dictionary
    from .source import source_supports

    dictionary = registered_dictionary(context.cohort.database if context else None)
    projected = (
        project_facts(plan, context=context, dictionary=dictionary)
        if isinstance(plan, AnalysisPlan) else None
    )
    question = question if question is not None else getattr(plan, "research_question", "")
    source = QuestionQuoteMatcher(question)
    intake = TypingIntake(
        question=question, dictionary=dictionary, projected=projected, context=context,
        proposals=proposals, part_ids=frozenset(item.part_id for item in observations),
    ) if projected else None
    groups: dict[tuple[str, str], list[_Observation]] = defaultdict(list)
    for item in observations:
        if item.parent_quote:
            parent = item.part_id.split("/clause/", 1)[0]
            groups[parent, item.parent_quote].append(item)
    clause_spans = {
        part_id: span for (_, parent_quote), group in groups.items()
        for part_id, span in _clause_spans(source, parent_quote, group).items()
    }
    rows: list[_ComparedPart] = []
    for item in observations:
        if item.parent_quote:
            span = clause_spans.get(item.part_id)
        else:
            span = source.find(item.quote) if item.quote else None
            if span and source.find(item.quote, offset=span.end):
                span = None
        result = None
        repair = ()
        decision: Decision = Unknown(
            reason="plan_contract_unavailable" if projected is None else "source_span_unavailable"
        )
        facet = item.facet_hint
        source_bound_miss = False
        if span is not None:
            part = AtomicPart(
                part_id=item.part_id, source_span=SourceSpan(start=span.start, end=span.end),
                key=None, facet_hint=facet,
            )
            if intake is not None:
                result, repair = intake.judge(part, output_id=item.output_id)
                part, decision = result.part, result.decision
            else:
                result = PartResult(part=part, decision=decision)
            facet = part.key.facet.kind if part.key else part.facet_hint
            if (
                item.v1_state == "covered"
                and decision.state in {"not_covered", "capability_gap"}
                and part.key is not None
            ):
                source_bound_miss = source_supports(span.text, part.key, context=context, dictionary=dictionary)
        elif intake is not None:
            repair = intake.missing_span_repair(item.part_id)
        rows.append(_ComparedPart(
            item.v1_state, item.v1_reason, decision, facet,
            plan_language(span.text if span else item.quote or question), result, repair, source_bound_miss,
        ))
    if intake is not None:
        intake.record(tuple(row.result for row in rows if row.result is not None))
    return _Comparison(tuple(rows), tuple(intake.repair) if intake is not None else ())


def _aggregate(comparison: _Comparison, proposals: object) -> ShadowSummary:
    by_facet: dict[str, _StateCounts] = {}
    by_language: dict[str, _StateCounts] = {}
    disagreements: _DisagreementCounts = {"true_false_pass": 0, "prose_proven": 0, "v2_under_typing": 0}
    repairs = set(comparison.repair)
    for row in comparison.parts:
        state = row.decision.state
        if state == "capability_gap":
            state = "not_covered"
        by_facet.setdefault(row.facet or "untyped", {"covered": 0, "not_covered": 0, "unknown": 0})[state] += 1
        by_language.setdefault(row.language, {"covered": 0, "not_covered": 0, "unknown": 0})[state] += 1
        if row.v1 == "covered" and state != "covered":
            if row.v1_reason == "plan_declared":
                disagreements["prose_proven"] += 1
            elif row.source_bound_miss:
                disagreements["true_false_pass"] += 1
            else:
                disagreements["v2_under_typing"] += 1
        repairs.update(row.repair)

    # One count per submitted entry and reason, regardless of how many
    # fields triggered it. An invalid batch is one batch-level rejection.
    rejected = {(issue.path.partition("]")[0], issue.code) for issue in repairs}
    return {
        "schema_version": _SHADOW_SCHEMA,
        "proposal_count": _proposal_count(proposals),
        "parts_total": len(comparison.parts),
        "by_facet": by_facet,
        "by_language": by_language,
        "disagreements": disagreements,
        "proposal_rejections": dict(Counter(code for _, code in rejected)),
    }


def observe_requirements(
    judged: Sequence[JudgedRequirement], *, plan: AnalysisPlan, question: str | None,
    context: ResearchContext | None, proposals: object,
) -> ShadowSummary:
    """Compare the Planner's atomic parts in memory and return counts only."""
    comparison = _observe(
        tuple(_requirements(judged)), plan=plan, context=context, question=question, proposals=proposals,
    )
    return _aggregate(comparison, proposals)
