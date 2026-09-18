"""Preserve accepted baseline rosters across metadata-to-data planning.

This is a completeness contract, not a choice of aggregation or a claim that
an executed table is correct. Only the host accepting an exact candidate may
issue it; available variables or mentions in prose never create requirements.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.concept_output_sources import resolve_composite_concept_output

from ..schema import AnalysisPlan, AnalysisStep, ResearchContext, TableOneSpec
from ..contracts.cohort_summary import declared_summary_columns, is_descriptive_cohort_summary_step


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
    group_by: BaselineCoordinate | None
    variables: tuple[BaselineCoordinate, ...] = Field(min_length=1)


class AcceptedBaselineRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.accepted_baseline_requirements/1", "easyicu.accepted_baseline_requirements/2"] = (
        "easyicu.accepted_baseline_requirements/1"
    )
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tables: tuple[BaselineTableRequirement, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _versioned_grouping(self) -> "AcceptedBaselineRequirements":
        if self.schema_version.endswith("/1") and any(table.group_by is None for table in self.tables):
            raise ValueError("ungrouped baseline requirements require schema version 2")
        return self


def candidate_baseline_requirements(
    *, plan: Mapping[str, Any], source_plan_sha256: str,
    selected_concepts: Sequence[str], catalog_columns: Sequence[str],
) -> AcceptedBaselineRequirements | None:
    """Project every closed baseline owner from a digest-verified candidate.

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
        if not isinstance(step, Mapping):
            continue
        if step.get("table_one_spec") is None:
            # Only the closed executable owner declares this roster. Neither
            # a baseline label nor arbitrary custom inputs create authority.
            if not set(step.get("expected_outputs") or ()).intersection({"table:baseline_table", "table:cohort_summary"}):
                continue
            parsed = AnalysisStep.model_validate(step)
            if is_descriptive_cohort_summary_step(parsed):
                tables.append(BaselineTableRequirement(
                    source_step_id=parsed.step_id, group_by=None,
                    variables=tuple(coordinate(name) for name in declared_summary_columns(parsed)),
                ))
            elif set(parsed.expected_outputs).intersection({"table:baseline_table", "table:cohort_summary"}):
                raise ValueError("accepted baseline lacks a closed descriptive or Table One owner")
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
        schema_version=("easyicu.accepted_baseline_requirements/2"
                        if any(table.group_by is None for table in tables)
                        else "easyicu.accepted_baseline_requirements/1"),
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
        and variable.source_concept is not None
    }
    if source is not None:
        # The concept owner declares composite output identities (not string
        # aliases). Accept a unique materialized family, never choose between
        # competing outputs such as two different comorbidity scores.
        source = resolve_composite_concept_output(source, value_sources)
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
                    "required": table.group_by.name if table.group_by else None,
                    "available_columns": _available_columns(table.group_by, context) if table.group_by else [],
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
    """Check rows declared by closed table owners, never arbitrary inputs."""

    rosters = [
        (step.step_id, {step.table_one_spec.group_by},
         {variable.name for variable in step.table_one_spec.variables})
        for step in plan.steps if step.table_one_spec is not None
    ]
    rosters.extend(
        (step.step_id, set(), set(declared_summary_columns(step)))
        for step in plan.steps if is_descriptive_cohort_summary_step(step)
    )
    return _baseline_roster_coverage(context, rosters)


def baseline_outline_coverage(
    context: ResearchContext, steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check a prospective baseline owner before its outline becomes binding.

    An outline has not chosen the exact stratum or summary yet. It must expose
    the accepted grouping and a clinical representation of every required row
    in ONE baseline step. Ungrouped requirements can use an auxiliary summary;
    final coverage must still prove its closed executable owner.
    """

    return _baseline_roster_coverage(context, [
        (str(step["step_id"]), (set(step.get("variable_names", ()))
                               if step.get("module_id") == "table_one" else set()),
         set(step.get("variable_names", ())))
        for step in steps if step.get("module_id") == "table_one"
        or (step.get("module_id") == "custom_analysis"
            and step.get("planned_analysis_role") == "auxiliary"
            and step.get("scientific_action_id") is None)
    ])


def _baseline_roster_coverage(
    context: ResearchContext,
    rosters: Sequence[tuple[str, set[str], set[str]]],
) -> dict[str, Any]:
    projection = baseline_requirement_projection(context)
    for table in projection["tables"]:
        candidates = [
            (step_id, variables | groups if table["group_by"]["required"] is None else variables)
            for step_id, groups, variables in rosters
            if table["group_by"]["required"] is None
            or groups.intersection(table["group_by"]["available_columns"])
        ]
        missing_by_step = [
            (
                step_id,
                [row["required"] for row in table["variables"]
                 if not set(row["available_columns"]).intersection(variables)],
            )
            for step_id, variables in candidates
        ]
        best_step, missing = min(missing_by_step, key=lambda pair: len(pair[1]), default=(
            None, [row["required"] for row in table["variables"]],
        ))
        table.update({
            "matched_step_id": best_step,
            "missing_variables": missing,
            "unavailable_coordinates": [
                row["required"] for row in [table["group_by"], *table["variables"]]
                if row["required"] is not None and not row["available_columns"]
            ],
            "complete": best_step is not None and not missing,
        })
    if projection["tables"]:
        projection["status"] = (
            "complete" if all(table["complete"] for table in projection["tables"])
            else "incomplete"
        )
    return projection
