"""The columns that identify an analysis input's rows.

A predicate over one of them states no population: every row has an
identifier, so the predicate keeps every row or an arbitrary subset of them.
Planning and the population compiler both refuse such a predicate.
"""

from __future__ import annotations

from ..schema import ResearchContext


def cohort_identity_columns(context: ResearchContext) -> frozenset[str]:
    """The columns that identify the analysis input's rows.

    Every row has its identifier, so a predicate over one keeps every row (a
    value that is not missing, a count of at least one) or an arbitrary
    subset of them: it states no population.
    """

    return frozenset(
        {
            *(str(name) for name in context.cohort.id_columns),
            *(
                variable.name
                for variable in context.variables
                if variable.role.value == "id"
            ),
        }
    )


__all__ = ["cohort_identity_columns"]
