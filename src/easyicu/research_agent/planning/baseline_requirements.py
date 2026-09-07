"""Preserve accepted baseline rosters across metadata-to-data planning.

This is a completeness contract, not a choice of aggregation or a claim that
an executed table is correct. Only the host accepting an exact candidate may
issue it; available variables or mentions in prose never create requirements.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from easyicu.concept_output_sources import COMPOSITE_CONCEPT_OUTPUT_SOURCES

from ..schema import AnalysisPlan, ResearchContext, TableOneSpec


_PROVENANCE_KEY = "accepted_baseline_requirements"
_NON_VALUE_ROLES = frozenset({"meta", "time", "id", "index"})


class BaselineCoordinate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    # A catalog concept may need a Planner-selected value aggregation after
    # materialization. An already operationalized coordinate remains exact.
    source_concept: str | None = None


class BaselineTableRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_step_id: str = Field(min_length=1)
    group_by: BaselineCoordinate
    variables: tuple[BaselineCoordinate, ...] = Field(min_length=1)


class AcceptedBaselineRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.accepted_baseline_requirements/1"] = (
        "easyicu.accepted_baseline_requirements/1"
    )
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tables: tuple[BaselineTableRequirement, ...] = Field(min_length=1)


def candidate_baseline_requirements(
    *, plan: Mapping[str, Any], source_plan_sha256: str,
    selected_concepts: Sequence[str], catalog_columns: Sequence[str],
) -> AcceptedBaselineRequirements | None:
    """Project every typed baseline table from a digest-verified candidate.

    The caller owns verification of the candidate and zero-row catalog. No
    truncation is permitted here: dropping a long tail would recreate the bug.
    """

    concepts, columns = set(selected_concepts), set(catalog_columns)

    def coordinate(name: str) -> BaselineCoordinate:
        if name not in columns:
            raise ValueError(f"accepted baseline coordinate is outside the catalog: {name}")
        return BaselineCoordinate(
            name=name, source_concept=name if name in concepts else None,
        )

    tables = []
    for step in plan.get("steps") or ():
        if not isinstance(step, Mapping) or step.get("table_one_spec") is None:
            continue
        spec = TableOneSpec.model_validate(step["table_one_spec"])
        tables.append(BaselineTableRequirement(
            source_step_id=step["step_id"],
            group_by=coordinate(spec.group_by),
            variables=tuple(coordinate(variable.name) for variable in spec.variables),
        ))
    if not tables:
        return None
    return AcceptedBaselineRequirements(
        source_plan_sha256=source_plan_sha256, tables=tuple(tables),
    )


def context_baseline_requirements(
    context: ResearchContext,
) -> AcceptedBaselineRequirements | None:
    payload = context.cohort.provenance.get(_PROVENANCE_KEY)
    return (
        AcceptedBaselineRequirements.model_validate(payload)
        if payload is not None else None
    )


def bind_baseline_requirements(
    context: ResearchContext, payload: Mapping[str, Any] | None,
    *, restoring: bool = False,
) -> ResearchContext:
    """Bind once before context sealing; never retrofit a resumed context."""

    required = (
        AcceptedBaselineRequirements.model_validate(payload)
        if payload is not None else None
    )
    existing = context_baseline_requirements(context)
    if restoring or existing is not None:
        if existing != required:
            raise ValueError("accepted_baseline_requirements_binding_drift")
        return context
    if required is None:
        return context
    provenance = dict(context.cohort.provenance)
    provenance[_PROVENANCE_KEY] = required.model_dump(mode="json")
    return context.model_copy(update={
        "cohort": context.cohort.model_copy(update={"provenance": provenance}),
    })


def _available_columns(
    requirement: BaselineCoordinate, context: ResearchContext,
) -> list[str]:
    source = requirement.source_concept
    value_sources = {
        variable.source_concept for variable in context.variables
        if variable.role.value not in _NON_VALUE_ROLES
    }
    if source is not None and source not in value_sources:
        # The concept owner declares composite output identities (not string
        # aliases). Accept a unique materialized family, never choose between
        # competing outputs such as two different comorbidity scores.
        outputs = {
            output for output, parent in COMPOSITE_CONCEPT_OUTPUT_SOURCES.items()
            if parent == source and output in value_sources
        }
        source = next(iter(outputs)) if len(outputs) == 1 else None
    matches = []
    for variable in context.variables:
        if requirement.source_concept is not None and variable.role.value in _NON_VALUE_ROLES:
            # Even an exact name cannot turn a concept's metadata into its
            # clinical value. Explicit operationalized coordinates stay exact.
            continue
        if variable.name == requirement.name:
            matches.append(variable.name)
        elif (
            source is not None
            and variable.source_concept == source
            and variable.role.value not in _NON_VALUE_ROLES
        ):
            # Use the host's source/role metadata, not column-name suffixes.
            # Measurement counts/timestamps are not the required clinical value.
            matches.append(variable.name)
    return sorted(set(matches))


def baseline_requirement_projection(context: ResearchContext) -> dict[str, Any]:
    required = context_baseline_requirements(context)
    if required is None:
        return {"status": "not_bound", "tables": []}
    return {
        "status": "bound",
        "source_plan_sha256": required.source_plan_sha256,
        "scope": "accepted baseline roster; not aggregation, execution or publication approval",
        "tables": [
            {
                "source_step_id": table.source_step_id,
                "group_by": {
                    "required": table.group_by.name,
                    "available_columns": _available_columns(table.group_by, context),
                },
                "variables": [
                    {"required": variable.name,
                     "available_columns": _available_columns(variable, context)}
                    for variable in table.variables
                ],
            }
            for table in required.tables
        ],
    }


def baseline_requirement_coverage(
    context: ResearchContext, plan: AnalysisPlan,
) -> dict[str, Any]:
    """Check actual typed table rows, not step inputs or a baseline label."""

    projection = baseline_requirement_projection(context)
    for table in projection["tables"]:
        candidates = [
            step for step in plan.steps
            if step.table_one_spec is not None
            and step.table_one_spec.group_by in table["group_by"]["available_columns"]
        ]
        missing_by_step = [
            (
                step.step_id,
                [row["required"] for row in table["variables"]
                 if not set(row["available_columns"]).intersection(
                     variable.name for variable in step.table_one_spec.variables
                 )],
            )
            for step in candidates
        ]
        best_step, missing = min(missing_by_step, key=lambda pair: len(pair[1]), default=(
            None, [row["required"] for row in table["variables"]],
        ))
        table.update({
            "matched_step_id": best_step,
            "missing_variables": missing,
            "unavailable_coordinates": [
                row["required"] for row in [table["group_by"], *table["variables"]]
                if not row["available_columns"]
            ],
            "complete": best_step is not None and not missing,
        })
    if projection["tables"]:
        projection["status"] = (
            "complete" if all(table["complete"] for table in projection["tables"])
            else "incomplete"
        )
    return projection
