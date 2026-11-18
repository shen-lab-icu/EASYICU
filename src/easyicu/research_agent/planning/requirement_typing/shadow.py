"""Opt-in comparison beside the live v1 judgment, with no approval side effects."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...schema import AnalysisPlan, ResearchContext
    from ..question_requirements import EffectMeasureJudgment, JudgedRequirement, UnstatedConcept
    from .contract import FacetName

FLAG = "EASYICU_REQUIREMENT_COVERAGE_V2"
LOGGER = logging.getLogger(__name__)


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


def _observe(
    observations: Sequence[_Observation], *,
    plan: AnalysisPlan, context: ResearchContext | None, question: str | None,
    boundary: str, proposals: object = (),
) -> None:
    from ...schema import AnalysisPlan
    from ..question_quote import QuestionQuoteMatcher
    from .contract import AtomicPart, CoverageRecord, PartResult, SourceSpan, Unknown
    from .facts import project_facts
    from .intake import TypingIntake
    from .registry import registered_dictionary

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
    # Resolve repeated source wording in order within each requirement. V1 may
    # reuse quotes across requirements, which is kept visible by separate IDs.
    offsets: dict[str, int] = {}
    rows = []
    results = []
    for item in observations:
        parent = item.part_id.split("/clause/", 1)[0]
        container = source.find(item.parent_quote) if item.parent_quote else None
        if container and source.find(item.parent_quote, offset=container.end):
            container = None
        start = max(offsets.get(parent, 0), container.start if container else 0)
        span = source.find(item.quote, offset=start) if item.quote else None
        if container and item.quote and (span is None or span.end > container.end):
            span = source.find(item.quote, offset=container.start)
        if span and item.parent_quote and (container is None or span.end > container.end):
            span = None
        if span and not item.parent_quote and source.find(item.quote, offset=span.end):
            span = None
        part = None
        decision = Unknown(
            reason="plan_contract_unavailable" if projected is None else "source_span_unavailable"
        )
        if span is not None:
            if item.parent_quote:
                offsets[parent] = max(offsets.get(parent, 0), span.end)
            part = AtomicPart(
                part_id=item.part_id, source_span=SourceSpan(start=span.start, end=span.end),
                key=None, facet_hint=item.facet_hint,
            )
        repair = ()
        result = None
        if part is not None:
            if intake is not None:
                result, repair = intake.judge(part, output_id=item.output_id)
                part, decision = result.part, result.decision
            else:
                result = PartResult(part=part, decision=decision)
            results.append(result)
        rows.append({
            "part_id": item.part_id,
            "v1": item.v1_state,
            "v1_reason": item.v1_reason,
            "v2": decision.state,
            "v2_reason": getattr(decision, "reason", None),
            "facet": (part.key.facet.kind if part and part.key else part.facet_hint if part else item.facet_hint),
            "source_span": part.source_span.model_dump() if part else None,
            "result": result.model_dump(mode="json") if result else None,
            "repair": [asdict(issue) for issue in repair],
        })
    record = intake.record(tuple(results)) if intake else None
    LOGGER.debug(
        "Typed requirement coverage shadow judgment",
        extra={"requirement_coverage_shadow": {
            "schema_version": CoverageRecord.model_fields["schema_version"].default,
            "boundary": boundary,
            "authority": record.authority.model_dump() if record else None,
            "parts": rows,
            "repair": [asdict(issue) for issue in intake.repair] if intake else [],
        }},
    )


def _run(
    observations: Iterable[_Observation], *,
    plan: AnalysisPlan, context: ResearchContext | None = None, question: str | None = None,
    boundary: str, proposals: object = (),
) -> None:
    try:
        items = tuple(observations)
        if items:
            _observe(items, plan=plan, context=context, question=question, boundary=boundary, proposals=proposals)
        else:
            LOGGER.debug(
                "No legacy requirement parts to compare",
                extra={"requirement_coverage_shadow": {"boundary": boundary, "parts": []}},
            )
    except Exception as exc:
        # Shadow remains observational even if a legacy shape fails projection.
        # Keep the diagnostic free of plan, question, and provider text.
        LOGGER.warning(
            "Typed requirement coverage shadow failed at %s (%s)", boundary, type(exc).__name__,
            extra={"requirement_coverage_shadow_error": {"boundary": boundary, "error": type(exc).__name__}},
        )


def observe_requirements(
    judged: Sequence[JudgedRequirement], *, plan: AnalysisPlan, question: str | None,
    context: ResearchContext | None, proposals: object = (),
) -> None:
    if typing_shadow_enabled():
        _run(
            _requirements(judged), plan=plan, context=context, question=question,
            boundary="question_requirements", proposals=proposals,
        )


def observe_effect_measures(
    judged: Sequence[EffectMeasureJudgment], *, plan: AnalysisPlan,
) -> None:
    if os.getenv(FLAG) == "shadow":
        _run(
            (_Observation(
                part_id=f"effect_measure/{item.measure}", quote=item.quote,
                v1_state=item.disposition, v1_reason=item.reason_code, facet_hint="method",
            ) for item in judged),
            plan=plan, boundary="stated_effect_measures",
        )


def observe_outline(
    unstated: Sequence[UnstatedConcept], *, plan: AnalysisPlan,
) -> None:
    if os.getenv(FLAG) == "shadow":
        _run(
            (_Observation(
                part_id=f"unstated/{index}", quote=item.named.evidence,
                v1_state="unstated", v1_reason="question_named_concept_unstated",
            ) for index, item in enumerate(unstated, 1)),
            plan=plan, boundary="outline_unstated",
        )
