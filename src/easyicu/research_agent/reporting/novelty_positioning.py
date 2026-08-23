"""Unsigned, source-bound novelty-positioning review packet.

This module owns one dependency-neutral reporting boundary: it turns the
sealed study context, final plan, and retained direct-comparator records into
an inspectable packet for independent appraisal.  It intentionally does not
claim that novelty is established.  A retrieved abstract cannot by itself
prove that populations, time zero, estimands, or analyses differ, and the
Agent must not fill those comparator cells from intuition.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..literature import LiteratureBundle
from ..planning.novelty_contract import NOVELTY_REVIEW_DIMENSIONS
from ..schema import AnalysisPlan, ResearchContext


class NoveltyComparisonDimension(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    study: str
    comparator: Optional[str] = None
    difference: Optional[str] = None
    source_status: Literal["study_authority_only", "independent_reviewed"] = (
        "study_authority_only"
    )


class NoveltyComparatorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str
    title: str
    year: Optional[str] = None
    venue: Optional[str] = None
    source_excerpt: Optional[str] = None
    population_match: bool
    exposure_match: bool
    outcome_match: bool
    screening_rationale: str


class NoveltyPositioningPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.novelty_positioning/2"] = (
        "easyicu.novelty_positioning/2"
    )
    status: Literal[
        "not_established", "review_required", "supported", "not_supported"
    ]
    review_disposition: Literal[
        "not_available",
        "independent_pre_review_pass",
        "human_review_pass",
        "requires_changes",
    ] = "not_available"
    reviewer_owner: Optional[str] = None
    reviewed_at: Optional[str] = None
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    literature_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    direct_comparator_keys: list[str] = Field(default_factory=list)
    comparators: list[NoveltyComparatorRecord] = Field(default_factory=list)
    comparison_dimensions: dict[str, NoveltyComparisonDimension]
    reviewer_questions: list[str]
    claim_boundary: str = (
        "The generated packet is unsigned. A novelty claim is authorized only "
        "when an external reviewer completes every comparison dimension, binds "
        "the exact input digests, and issues an accepted review disposition."
    )

    @model_validator(mode="after")
    def _review_state_is_coherent(self) -> "NoveltyPositioningPacket":
        missing_dimensions = sorted(
            set(NOVELTY_REVIEW_DIMENSIONS) - set(self.comparison_dimensions)
        )
        if missing_dimensions:
            raise ValueError(
                "novelty positioning requires every prespecified dimension: "
                + ", ".join(missing_dimensions)
            )
        accepted = self.review_disposition in {
            "independent_pre_review_pass",
            "human_review_pass",
        }
        if self.status == "supported":
            if not accepted or not str(self.reviewer_owner or "").strip():
                raise ValueError(
                    "supported novelty requires an accepted external review "
                    "disposition and reviewer_owner"
                )
            if not self.direct_comparator_keys:
                raise ValueError("supported novelty requires a direct comparator")
            incomplete = [
                name
                for name, dimension in self.comparison_dimensions.items()
                if not str(dimension.comparator or "").strip()
                or not str(dimension.difference or "").strip()
                or dimension.source_status != "independent_reviewed"
            ]
            if incomplete:
                raise ValueError(
                    "supported novelty requires independently reviewed comparator "
                    "and difference cells for every dimension: "
                    + ", ".join(sorted(incomplete))
                )
        if accepted and self.status != "supported":
            raise ValueError("an accepted review disposition requires status=supported")
        if self.review_disposition == "not_available" and self.status not in {
            "not_established",
            "review_required",
        }:
            raise ValueError("an unsigned packet cannot declare a review conclusion")
        return self


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def novelty_input_sha256(value: Any) -> str:
    """Hash one canonical input governed by a novelty review."""

    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def novelty_authority_digests(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    literature: Any,
) -> dict[str, str]:
    """Bind a review to the exact context, plan, and literature bundle."""

    return {
        "context_sha256": novelty_input_sha256(context),
        "plan_sha256": novelty_input_sha256(plan),
        "literature_sha256": novelty_input_sha256(literature),
    }


def _study_dimensions(
    context: ResearchContext,
    plan: AnalysisPlan,
) -> dict[str, NoveltyComparisonDimension]:
    selected_design = (
        plan.design_selection.selected if plan.design_selection is not None else None
    )
    exposure = context.variable(context.primary_exposure or "")
    outcome = context.variable(context.target_outcome or "")
    primary = next(
        (step for step in plan.steps if step.planned_analysis_role == "primary"),
        None,
    )
    covariates = sorted(
        {
            value
            for requirement in (primary.model_requirements if primary else ())
            for value in requirement.covariates
        }
    )
    cohort = context.cohort
    population = "; ".join(
        value
        for value in (
            f"database={cohort.database}",
            f"analysis_unit={'ICU stay' if cohort.n_stays is not None else 'unspecified'}",
            (
                "inclusion=" + " | ".join(cohort.inclusion_criteria)
                if cohort.inclusion_criteria
                else ""
            ),
            (
                "exclusion=" + " | ".join(cohort.exclusion_criteria)
                if cohort.exclusion_criteria
                else ""
            ),
        )
        if value
    )
    time_zero = (
        "; ".join(
            [
                _text(getattr(selected_design, "time_zero", None)),
                _text(getattr(selected_design, "observation_window", None)),
                *[_text(item) for item in context.time_windows],
                *[_text(item) for item in context.temporal_constraints],
                _text(getattr(exposure, "analysis_window", None)),
            ]
        )
        or "No complete time-zero/follow-up authority was declared."
    )
    exposure_text = "; ".join(
        value
        for value in (
            f"name={context.primary_exposure or 'not declared'}",
            _text(getattr(exposure, "description", None)),
            _text(getattr(exposure, "source_concept", None)),
        )
        if value
    )
    endpoint_text = "; ".join(
        value
        for value in (
            f"name={context.target_outcome or 'not declared'}",
            (
                f"selected_estimand={_text(selected_design.estimand)}"
                if selected_design is not None
                else ""
            ),
            _text(getattr(outcome, "description", None)),
            _text(getattr(outcome, "source_concept", None)),
            _text(context.endpoint),
        )
        if value
    )
    analysis = "; ".join(
        value
        for value in (
            f"analysis_type={plan.analysis_type or 'not declared'}",
            (
                f"selected_method={_text(selected_design.primary_method)}"
                if selected_design is not None
                else ""
            ),
            f"method={_text(primary.method) if primary else 'no primary step'}",
            f"intent={_text(primary.intent) if primary else 'no primary step'}",
            "covariates=" + (", ".join(covariates) if covariates else "none"),
        )
        if value
    )
    transportability = "; ".join(
        value
        for value in (
            f"primary_database={cohort.database}",
            (
                "prespecified_external_databases="
                + ", ".join(context.cross_database_validation)
                if context.cross_database_validation
                else "prespecified_external_databases=none"
            ),
        )
        if value
    )
    contribution = "; ".join(
        value
        for value in (
            (
                f"planner_positioning={_text(selected_design.novelty_positioning)}"
                if selected_design is not None
                else ""
            ),
            (
                f"supports={_text(selected_design.supports)}"
                if selected_design is not None
                else ""
            ),
            (
                f"cannot_prove={_text(selected_design.cannot_prove)}"
                if selected_design is not None
                else ""
            ),
            (
                "No contribution claim is pre-authorized. The independent "
                "reviewer must distinguish a clinically meaningful or "
                "methodological advance from a new database/concept "
                "instantiation using retained source evidence."
            ),
        )
        if value
    )
    return {
        "population_and_setting": NoveltyComparisonDimension(study=population),
        "exposure_definition_and_time_zero": NoveltyComparisonDimension(
            study="; ".join((exposure_text, time_zero))
        ),
        "outcome_and_estimand": NoveltyComparisonDimension(study=endpoint_text),
        "analysis_and_robustness_route": NoveltyComparisonDimension(
            study=analysis
        ),
        "data_source_and_transportability": NoveltyComparisonDimension(
            study=transportability
        ),
        "clinical_decision_or_methodological_contribution": (
            NoveltyComparisonDimension(study=contribution)
        ),
    }


def build_unsigned_novelty_positioning_packet(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    literature: Optional[LiteratureBundle],
) -> NoveltyPositioningPacket:
    records_by_key = {
        record.key: record for record in (literature.citations if literature else ())
    }
    decisions = [
        decision
        for decision in (literature.screening_decisions if literature else ())
        if decision.disposition == "include"
        and decision.evidence_role == "direct_comparator"
        and decision.publication_type_eligible
        and decision.citation_key in records_by_key
    ]
    comparators = []
    for decision in decisions:
        record = records_by_key[decision.citation_key]
        comparators.append(
            NoveltyComparatorRecord(
                citation_key=record.key,
                title=record.title,
                year=record.year,
                venue=record.venue,
                source_excerpt=_text(record.relevance) or None,
                population_match=decision.population_match,
                exposure_match=decision.exposure_match,
                outcome_match=decision.outcome_match,
                screening_rationale=decision.rationale,
            )
        )
    keys = [row.citation_key for row in comparators]
    return NoveltyPositioningPacket(
        status="review_required" if keys else "not_established",
        **novelty_authority_digests(
            context=context,
            plan=plan,
            literature=(
                literature
                or LiteratureBundle(
                    research_question=context.research_question,
                    citations=[],
                )
            ),
        ),
        direct_comparator_keys=keys,
        comparators=comparators,
        comparison_dimensions=_study_dimensions(context, plan),
        reviewer_questions=[
            "Does each retained source truly study a comparable ICU population?",
            "Are time zero, exposure ascertainment, and outcome follow-up comparable?",
            "Is the estimand materially different, rather than only the database name?",
            "Were analysis choices borrowed appropriately without copying incompatible eligibility rules?",
            "Does the data-source and validation strategy support any transportability claim?",
            "Is any claimed innovation clinically meaningful and current enough for the target journal?",
        ],
    )


__all__ = [
    "NoveltyComparatorRecord",
    "NoveltyComparisonDimension",
    "NoveltyPositioningPacket",
    "build_unsigned_novelty_positioning_packet",
    "novelty_authority_digests",
    "novelty_input_sha256",
]
