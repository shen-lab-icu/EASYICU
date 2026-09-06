"""Shared semantic boundary for planned and locally bound Table 1 specs."""

from __future__ import annotations

from ..schema import ResearchContext, TableOneSpec, VariableRole


def table_one_identity_columns(context: ResearchContext) -> frozenset[str]:
    """Use owner-declared roles and cohort keys, never column-name guesses."""

    return frozenset(context.cohort.id_columns) | frozenset(
        variable.name
        for variable in context.variables
        if variable.role in {VariableRole.ID, VariableRole.INDEX}
    )


def validate_table_one_semantic_roles(
    spec: TableOneSpec, context: ResearchContext
) -> None:
    """Reject summaries or grouping by identity without rewriting the plan.

    IDs remain valid lineage/cohort inputs. They cannot become continuous
    clinical rows or arbitrary strata merely because their dtype is numeric.
    """

    selected = {spec.group_by, *(row.name for row in spec.variables)}
    invalid = sorted(selected & table_one_identity_columns(context))
    if invalid:
        raise ValueError(
            "table_one_identity_coordinate_ineligible: Table 1 rows and "
            f"grouping cannot use owner-declared identity/index columns {invalid!r}"
        )
