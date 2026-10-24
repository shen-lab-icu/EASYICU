"""Preserve accepted baseline rosters across metadata-to-data planning.

This is a completeness contract, not a choice of aggregation or a claim that
an executed table is correct. Only the host accepting an exact candidate may
issue it; available variables or mentions in prose never create requirements.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from easyicu.concept_output_sources import resolve_composite_concept_output

from ..schema import AnalysisPlan, AnalysisStep, ResearchContext, TableOneSpec
from ..contracts.cohort_summary import declared_summary_columns
from ..contracts.cohort_summary_products import (
    auxiliary_summary_product,
    is_descriptive_summary_product_step,
)
from .progressive_contract import ProgressivePlanCompileError


_PROVENANCE_KEY = "accepted_baseline_requirements"
_NON_VALUE_ROLES = frozenset({"meta", "time", "id", "index"})
_LINEAGE_ROLES = frozenset({"id", "index"})


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
    # The accepted cohort owner can name an identity that is renamed during
    # extraction. Retain its lineage meaning without guessing column aliases.
    identities = {
        AnalysisStep.model_validate(step).cohort_definition_spec.identity_column
        for step in plan.get("steps") or ()
        if isinstance(step, Mapping) and step.get("cohort_definition_spec") is not None
    }

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
            baseline_products = {
                "table:baseline_table", "table:cohort_summary",
                auxiliary_summary_product(str(step.get("step_id") or "")),
            }
            if not set(step.get("expected_outputs") or ()).intersection(baseline_products):
                continue
            parsed = AnalysisStep.model_validate(step)
            # Primary results have their own scientific action/product contract.
            # Sharing a summary product never makes them baseline obligations.
            if parsed.planned_analysis_role != "auxiliary":
                continue
            if is_descriptive_summary_product_step(parsed):
                variables = tuple(coordinate(name) for name in declared_summary_columns(parsed) if name not in identities)
                if variables:
                    tables.append(BaselineTableRequirement(
                        source_step_id=parsed.step_id, group_by=None, variables=variables,
                    ))
            elif set(parsed.expected_outputs).intersection(baseline_products):
                raise ValueError("accepted baseline lacks a closed descriptive or Table One owner")
            continue
        spec = TableOneSpec.model_validate(step["table_one_spec"])
        variables = tuple(coordinate(variable.name) for variable in spec.variables if variable.name not in identities)
        if variables:
            tables.append(BaselineTableRequirement(
                source_step_id=step["step_id"],
                group_by=coordinate(spec.group_by) if spec.group_by not in identities else None,
                variables=variables,
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
    from ..methods.descriptive_inputs import context_columns_without_data

    return sorted(set(matches) - set(context_columns_without_data(context, matches)))


def baseline_requirement_projection(context: ResearchContext) -> dict[str, Any]:
    required = context_baseline_requirements(context)
    if required is None:
        return {"status": "not_bound", "tables": []}
    lineage = set(context.cohort.id_columns) | {
        variable.name for variable in context.variables if variable.role.value in _LINEAGE_ROLES
    }
    # A source concept with only identity/index outputs is lineage too. An
    # identity sharing a source with clinical measurements cannot erase them.
    clinical_sources = {
        name for variable in context.variables
        if variable.role.value not in _NON_VALUE_ROLES
        for name in (variable.name, variable.source_concept)
    }
    lineage.update(
        variable.source_concept for variable in context.variables
        if variable.role.value in _LINEAGE_ROLES and variable.source_concept is not None
        and variable.source_concept not in clinical_sources
    )
    tables = []
    for table in required.tables:
        variables = [variable for variable in table.variables if variable.name not in lineage]
        if not variables:
            continue
        group = table.group_by if table.group_by and table.group_by.name not in lineage else None
        tables.append({
            "source_step_id": table.source_step_id,
            "group_by": {
                "required": group.name if group else None,
                "available_columns": _available_columns(group, context) if group else [],
            },
            "variables": [
                {"required": variable.name, "available_columns": _available_columns(variable, context)}
                for variable in variables
            ],
        })
    return {
        "status": "bound",
        "source_plan_sha256": required.source_plan_sha256,
        "scope": "accepted baseline roster; not aggregation, execution or publication approval",
        "tables": tables,
    }


def validate_baseline_materialization(context: ResearchContext) -> None:
    """Refuse unavailable accepted coordinates before any provider request.

    A Planner can repair an omitted available row, but cannot materialize a
    missing clinical coordinate. Keep that host failure outside parse retries.
    """

    coverage = baseline_outline_coverage(context, ())
    unavailable = [table for table in coverage["tables"] if table["unavailable_coordinates"]]
    if unavailable:
        names = sorted({name for table in unavailable for name in table["unavailable_coordinates"]})
        raise ProgressivePlanCompileError(
            "progressive_accepted_baseline_materialization_missing",
            "Accepted baseline coordinates are unavailable after extraction: " + ", ".join(names),
            step_id=unavailable[0]["source_step_id"], path="accepted_baseline_requirements",
            concepts=names, findings=unavailable,
        )


def validate_baseline_outline(context: ResearchContext, steps: Sequence[Mapping[str, Any]]) -> None:
    """Keep repairable roster omissions inside the outline's local retry."""

    baseline = baseline_outline_coverage(context, steps)
    if baseline["status"] == "incomplete":
        raise ProgressivePlanCompileError(
            "progressive_outline_accepted_baseline_incomplete",
            "The outline must preserve each accepted baseline roster in one "
            "table_one step (or an auxiliary summary for an ungrouped roster) "
            "before foundation or step materialization. Include the required "
            "grouping and one available clinical-value column per required variable; "
            "measurement metadata and mentions in an audit or prose do not satisfy "
            "this requirement. " + json.dumps(baseline, ensure_ascii=False, sort_keys=True),
            path="steps", findings=(baseline,),
        )


def baseline_requirement_coverage(
    context: ResearchContext,
    plan: AnalysisPlan,
    *,
    signed_rosters: Sequence[tuple[str, set[str], set[str]]] = (),
) -> dict[str, Any]:
    """Check rows declared by closed table owners, never arbitrary inputs.

    ``signed_rosters`` are Table 1 rosters a signed runtime owner declares for
    one of the plan's steps -- ``(step_id, group columns, variables)`` -- which
    the caller resolves from that owner's contract, not from the step's inputs.
    """

    rosters = [
        (step.step_id, {step.table_one_spec.group_by},
         {variable.name for variable in step.table_one_spec.variables})
        for step in plan.steps if step.table_one_spec is not None
    ]
    rosters.extend(
        (step.step_id, set(), set(declared_summary_columns(step)))
        for step in plan.steps if step.planned_analysis_role == "auxiliary"
        and is_descriptive_summary_product_step(step)
    )
    rosters.extend(signed_rosters)
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
    if projection["status"] == "bound":
        projection["status"] = (
            "complete" if all(table["complete"] for table in projection["tables"])
            else "incomplete"
        )
    return projection
