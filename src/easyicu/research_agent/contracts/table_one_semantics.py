"""Shared semantic boundary for planned and locally bound Table 1 specs."""

from __future__ import annotations

from typing import Iterable

from ..schema import ResearchContext, TableOneSpec, VariableRole


def table_one_identity_columns(context: ResearchContext) -> frozenset[str]:
    """Use owner-declared roles and cohort keys, never column-name guesses."""

    return frozenset(context.cohort.id_columns) | frozenset(
        variable.name
        for variable in context.variables
        if variable.role in {VariableRole.ID, VariableRole.INDEX}
    )


def table_one_measurement_columns(context: ResearchContext) -> frozenset[str]:
    """Auxiliary measurement metadata is not a default clinical descriptor.

    A question explicitly bound to a measurement-process exposure or outcome
    may describe that anchor. Merely selecting an auxiliary count in an
    outline does not grant that scientific meaning.
    """

    anchors = {context.primary_exposure, context.target_outcome}
    return frozenset(
        variable.name
        for variable in context.variables
        if variable.name not in anchors
        and (
            variable.role == VariableRole.META
            or (
                variable.role == VariableRole.TIME
                and variable.unit_normalization
                in {"window_first_time", "window_last_time"}
            )
        )
    )


def validate_table_one_column_roles(
    columns: Iterable[str], context: ResearchContext
) -> None:
    """Validate selected rows/strata before levels or statistics are compiled.

    IDs remain valid lineage/cohort inputs. They cannot become continuous
    clinical rows or arbitrary strata merely because their dtype is numeric.
    """

    selected = set(columns)
    invalid = sorted(selected & table_one_identity_columns(context))
    if invalid:
        raise ValueError(
            "table_one_identity_coordinate_ineligible: Table 1 rows and "
            f"grouping cannot use owner-declared identity/index columns {invalid!r}"
        )
    measurement = sorted(selected & table_one_measurement_columns(context))
    if measurement:
        raise ValueError(
            "table_one_measurement_metadata_ineligible: auxiliary observation "
            f"counts, availability or observation-time fields {measurement!r} belong in a "
            "measurement audit, not automatic clinical baseline rows/strata. "
            "Select clinical-value representations from the declared roster; "
            "only an explicitly bound measurement-process exposure or outcome "
            "can use this Table 1 exception."
        )


def validate_table_one_semantic_roles(
    spec: TableOneSpec, context: ResearchContext
) -> None:
    validate_table_one_column_roles(
        (spec.group_by, *(row.name for row in spec.variables)), context
    )
