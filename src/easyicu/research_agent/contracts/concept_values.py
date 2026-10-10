"""Which concepts and columns of a study hold no value, by one rule.

An export can list a concept whose column holds no value: its producer ran and
wrote none, or the source cannot hold the concept and the export wrote a
placeholder (``intake.export_package.concepts_without_values``).  The column
exists, so a plan can name it, yet whatever reads it reads nothing: an
exclusion over it excludes no stay, and an imputer drops a predictor without a
value, so nothing fails while the analysis is not the one planned.  Every owner
that reads such a column refuses it:

* planning reads :func:`names_without_values` of its context.  A metadata-only
  planning context reads no row, so the host states its source's concepts
  without values beside it, under :data:`CONCEPTS_WITHOUT_VALUES_KEY`
  (``cohort.provenance``); a context built from rows shows its own, as the
  columns every row of which is missing;
* an owner that filters or fits rows asks :func:`columns_without_values` of
  the rows it uses, and stops with its own typed reason.

Pure: it reads what it is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

#: Where a metadata-only planning context states the concepts its source holds
#: no value of (``cohort.provenance``), as a sorted list of concept ids.
CONCEPTS_WITHOUT_VALUES_KEY = "concepts_without_values"


def stated_concepts_without_values(provenance: Any) -> tuple[str, ...]:
    """The concepts ``provenance`` states its source holds no value of.

    Only a list of non-empty names is a statement; anything else states none.
    """

    stated = (
        provenance.get(CONCEPTS_WITHOUT_VALUES_KEY)
        if isinstance(provenance, Mapping)
        else None
    )
    if not isinstance(stated, (list, tuple)) or not all(
        isinstance(name, str) and name.strip() for name in stated
    ):
        return ()
    return tuple(sorted({name.strip() for name in stated}))


def names_without_values(context: Any) -> frozenset[str]:
    """The concepts and columns ``context`` knows to hold no value.

    What the host states of the source, and every variable whose every row,
    of at least one, is missing.
    """

    cohort = getattr(context, "cohort", None)
    names = set(stated_concepts_without_values(getattr(cohort, "provenance", None)))
    for variable in getattr(context, "variables", None) or ():
        profile = getattr(variable, "missingness", None)
        total = getattr(profile, "n_total", None)
        if (
            isinstance(total, int)
            and total > 0
            and getattr(profile, "n_missing", None) == total
        ):
            names.add(str(variable.name))
    return frozenset(names)


def columns_without_values(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    """The ``columns`` of ``frame`` no row of which holds a value, in their order.

    A frame without rows holds no value in any column: whether rows are
    required is its caller's own rule.
    """

    return tuple(
        column for column in dict.fromkeys(columns) if not bool(frame[column].notna().any())
    )


__all__ = [
    "CONCEPTS_WITHOUT_VALUES_KEY",
    "columns_without_values",
    "names_without_values",
    "stated_concepts_without_values",
]
