"""Opt-in comparison beside the live v1 judgment, with no approval side effects."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...schema import AnalysisPlan, ResearchContext
    from ..question_requirements import EffectMeasureJudgment, JudgedRequirement, UnstatedConcept
    from .contract import FacetName, RequirementKey

FLAG = "EASYICU_REQUIREMENT_COVERAGE_V2"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Observation:
    part_id: str
    quote: str | None
    v1_state: str
    v1_reason: str
    key: RequirementKey | None = None
    facet_hint: FacetName | None = None
    parent_quote: str | None = None


def _requirements(
    judged: Sequence[JudgedRequirement], context: ResearchContext | None,
) -> Iterable[_Observation]:
    from .contract import Binding, Delivery, Method, NotApplicable, Parameter, RequirementKey
    from .registry import STUDY_POPULATION, registered_concept_id, registered_dictionary

    for entry in judged:
        item = entry.requirement
        if entry.clauses:
            # V1's segmentation only identifies which legacy result to compare.
            # It supplies neither coordinates nor evidence to v2.
            for index, clause in enumerate(entry.clauses, 1):
                evidence = clause.get("evidence") or {}
                source = evidence.get("source")
                yield _Observation(
                    part_id=f"{item.id}/clause/{index}", quote=clause["quote"],
                    v1_state=clause["disposition"],
                    v1_reason=source if source in ("host_default", "plan_declared") else entry.reason_code,
                    parent_quote=item.quote,
                )
            continue
        if item.deliverable_kinds:
            for output_id in item.deliverable_kinds:
                yield _Observation(
                    part_id=f"{item.id}/delivery/{output_id}", quote=item.quote,
                    v1_state=entry.disposition, v1_reason=entry.reason_code,
                    key=RequirementKey(
                        bindings=(STUDY_POPULATION,), time=NotApplicable(),
                        facet=Delivery(output_id=output_id),
                    ),
                )
            continue
        key = None
        if item.effect_measure and context and context.primary_exposure and context.target_outcome:
            dictionary = registered_dictionary(context.cohort.database)
            exposure = context.variable(context.primary_exposure)
            outcome = context.variable(context.target_outcome)
            exposure_id = registered_concept_id(context.primary_exposure, exposure, dictionary)
            outcome_id = registered_concept_id(context.target_outcome, outcome, dictionary)
            if exposure_id is not None and outcome_id is not None:
                key = RequirementKey(
                    bindings=(
                        Binding(concept_id=exposure_id, role="exposure"),
                        Binding(concept_id=outcome_id, role="outcome"),
                    ),
                    time=NotApplicable(),
                    facet=Method(action_id="binary_effect_measure", parameters=(
                        Parameter(name="measure", value=item.effect_measure),
                    )),
                )
        yield _Observation(
            part_id=item.id, quote=item.quote, v1_state=entry.disposition,
            v1_reason=entry.reason_code, key=key,
            facet_hint="delivery" if item.kind == "delivery" else "method" if item.effect_measure else None,
        )


def _observe(
    observations: Iterable[_Observation], *,
    plan: AnalysisPlan, context: ResearchContext | None, question: str | None,
    boundary: str,
) -> None:
    from ...schema import AnalysisPlan
    from ..question_quote import QuestionQuoteMatcher
    from .contract import AtomicPart, CoverageRecord, PartResult, SourceSpan, Unknown
    from .facts import project_facts
    from .matcher import match
    from .registry import registered_dictionary

    dictionary = registered_dictionary(context.cohort.database if context else None)
    projected = (
        project_facts(plan, context=context, dictionary=dictionary)
        if isinstance(plan, AnalysisPlan) else None
    )
    source = QuestionQuoteMatcher(
        question if question is not None else getattr(plan, "research_question", "")
    )
    # Resolve repeated source wording in order within each requirement. V1 may
    # reuse quotes across requirements, which is kept visible by separate IDs.
    offsets: dict[str, int] = {}
    rows = []
    for item in observations:
        parent = item.part_id.split("/clause/", 1)[0]
        container = source.find(item.parent_quote) if item.parent_quote else None
        start = max(offsets.get(parent, 0), container.start if container else 0)
        span = source.find(item.quote, offset=start) if item.quote else None
        if span and item.parent_quote and (container is None or span.end > container.end):
            span = None
        part = None
        decision = Unknown(
            reason="plan_contract_unavailable" if projected is None else "source_span_unavailable"
        )
        if span is not None:
            offsets[parent] = span.end
            part = AtomicPart(
                part_id=item.part_id, source_span=SourceSpan(start=span.start, end=span.end),
                key=item.key, facet_hint=item.facet_hint,
            )
            if projected is not None:
                decision = match(
                    part.key, authority=projected.authority, dictionary=dictionary, facts=projected.facts,
                )
        result = PartResult(part=part, decision=decision) if part is not None else None
        rows.append({
            "part_id": item.part_id,
            "v1": item.v1_state,
            "v1_reason": item.v1_reason,
            "v2": decision.state,
            "v2_reason": getattr(decision, "reason", None),
            "facet": item.key.facet.kind if item.key else item.facet_hint,
            "source_span": part.source_span.model_dump() if part else None,
            "result": result.model_dump(mode="json") if result else None,
        })
    LOGGER.debug(
        "Typed requirement coverage shadow judgment",
        extra={"requirement_coverage_shadow": {
            "schema_version": CoverageRecord.model_fields["schema_version"].default,
            "boundary": boundary,
            "authority": projected.authority.model_dump() if projected is not None else None,
            "parts": rows,
        }},
    )


def _run(
    observations: Iterable[_Observation], *,
    plan: AnalysisPlan, context: ResearchContext | None = None, question: str | None = None,
    boundary: str,
) -> None:
    try:
        items = tuple(observations)
        if items:
            _observe(items, plan=plan, context=context, question=question, boundary=boundary)
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
    context: ResearchContext | None,
) -> None:
    if os.getenv(FLAG) == "shadow":
        _run(
            _requirements(judged, context), plan=plan, context=context, question=question,
            boundary="question_requirements",
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
