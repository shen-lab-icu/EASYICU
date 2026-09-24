"""Preserve a reviewed candidate's primary-analysis inputs across metadata-to-data planning.

A metadata-only candidate names its primary-analysis inputs as catalog
concepts (``hr``, ``wbc``).  After materialization each concept becomes a
family of columns, and the package-bound Planner sees them only through a
bounded retrieval roster, which can omit an accepted input entirely.  This
contract keeps every accepted input concept visible with at least one value
representation.  It is not a choice of aggregation: every value
representation stays eligible, and measurement-process companions (counts,
measured flags, observation times) never stand in for the value.  Only the
host accepting an exact candidate may issue it.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schema import ResearchContext


_PROVENANCE_KEY = "accepted_analysis_inputs"
_NON_VALUE_ROLES = frozenset({"meta", "time", "id", "index"})


class AcceptedAnalysisInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.accepted_analysis_inputs/1"] = (
        "easyicu.accepted_analysis_inputs/1"
    )
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_step_ids: tuple[str, ...] = Field(min_length=1)
    concepts: tuple[str, ...] = Field(min_length=1)

    @field_validator("source_step_ids", "concepts")
    @classmethod
    def _unique_nonblank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(str(value or "").strip() for value in values)
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("accepted analysis inputs must be unique non-empty values")
        return cleaned


def candidate_analysis_inputs(
    *,
    plan: Mapping[str, Any],
    source_plan_sha256: str,
    selected_concepts: Sequence[str],
    excluded: Sequence[str],
) -> AcceptedAnalysisInputs | None:
    """Project the primary steps' catalog-concept inputs of a verified candidate.

    The caller owns verification of the candidate and its zero-row catalog.
    Artifact references and the identity, exposure and outcome coordinates
    have their own owners; an input that is not a concept the catalog
    selected (an already operationalized column) is exact elsewhere.
    """

    concepts = set(selected_concepts)
    skip = {str(value or "").strip() for value in excluded}
    step_ids: list[str] = []
    names: list[str] = []
    for step in plan.get("steps") or ():
        if not isinstance(step, Mapping) or step.get("planned_analysis_role") != "primary":
            continue
        accepted = [
            name
            for name in (str(value or "").strip() for value in step.get("inputs") or ())
            if name and ":" not in name and name not in skip and name in concepts
        ]
        if accepted:
            step_ids.append(str(step.get("step_id") or "").strip())
            names.extend(accepted)
    if not names:
        return None
    return AcceptedAnalysisInputs(
        source_plan_sha256=source_plan_sha256,
        source_step_ids=tuple(dict.fromkeys(step_ids)),
        concepts=tuple(dict.fromkeys(names)),
    )


def context_analysis_inputs(context: ResearchContext) -> AcceptedAnalysisInputs | None:
    payload = context.cohort.provenance.get(_PROVENANCE_KEY)
    return AcceptedAnalysisInputs.model_validate(payload) if payload is not None else None


def bind_analysis_inputs(
    context: ResearchContext, payload: Mapping[str, Any] | None,
    *, restoring: bool = False,
) -> ResearchContext:
    """Bind once before context sealing; never retrofit a resumed context."""

    required = (
        AcceptedAnalysisInputs.model_validate(payload) if payload is not None else None
    )
    existing = context_analysis_inputs(context)
    if restoring or existing is not None:
        if existing != required:
            raise ValueError("accepted_analysis_inputs_binding_drift")
        return context
    if required is None:
        return context
    provenance = dict(context.cohort.provenance)
    provenance[_PROVENANCE_KEY] = required.model_dump(mode="json")
    return context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"provenance": provenance}),
    })


def analysis_input_value_columns(context: ResearchContext) -> dict[str, tuple[str, ...]]:
    """Every value representation the sealed context offers per accepted concept.

    Uses the host's source/role metadata, not column-name suffixes; an empty
    tuple means the materialized universe lost an accepted input.
    """

    accepted = context_analysis_inputs(context)
    if accepted is None:
        return {}
    return {
        concept: tuple(
            variable.name
            for variable in context.variables
            if variable.role.value not in _NON_VALUE_ROLES
            and (variable.name == concept or variable.source_concept == concept)
        )
        for concept in accepted.concepts
    }


__all__ = [
    "AcceptedAnalysisInputs",
    "analysis_input_value_columns",
    "bind_analysis_inputs",
    "candidate_analysis_inputs",
    "context_analysis_inputs",
]
