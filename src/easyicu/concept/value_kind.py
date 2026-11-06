"""Declared value kind of a concept: numeric, logical or text.

The kind comes from the concept dictionary (class and numeric bounds), never
from the values a batch happens to hold. A sampled decision ("most of the
first rows parse as numbers") let the stays sharing a batch decide whether a
stay's hour was a median or a first row.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .errors import ConceptError

ValueKind = Literal["numeric", "logical", "text"]

_CLASS_KINDS: dict[str, ValueKind] = {
    "num_cncpt": "numeric",
    "unt_cncpt": "numeric",
    "lgl_cncpt": "logical",
    "fct_cncpt": "text",
    "chr_cncpt": "text",
}


class ConceptValueKindUndeclared(ConceptError):
    """The dictionary does not settle whether a concept's values are numeric."""

    def __init__(self, *, concept_id: str, database: str, class_name: object) -> None:
        self.concept_id = str(concept_id)
        self.database = str(database)
        self.class_name = class_name
        super().__init__(
            f"concept {self.concept_id!r} on {self.database!r} has a non-numeric "
            f"value column, and its declaration (class {class_name!r}, no numeric "
            "bounds) does not say whether it is numeric, logical or text. "
            "Refusing to decide from the batch's values."
        )


def declared_value_kind(definition, *, database: str) -> ValueKind:
    """Numeric, logical or text from the class and bounds of ``definition``.

    No class is ricu's default ``num_cncpt``. Numeric bounds declare a number
    whatever the class (a bounded ``rec_cncpt`` is a score). A class list must
    agree on one kind.
    """

    if definition.minimum is not None or definition.maximum is not None:
        return "numeric"
    declared = definition.class_name
    classes = declared if isinstance(declared, (list, tuple)) else [declared or "num_cncpt"]
    kinds = {_CLASS_KINDS.get(str(name)) for name in classes}
    if len(kinds) != 1 or None in kinds:
        raise ConceptValueKindUndeclared(
            concept_id=definition.name, database=database, class_name=declared,
        )
    return kinds.pop()


def coerce_declared_values(values: pd.Series, kind: ValueKind) -> pd.Series:
    """Parse a numeric or logical column; a text column stays as it is.

    Unparseable cells of a numeric or logical concept are missing values.
    The parser is the one the sampled route used (``pd.to_numeric``).
    """

    if kind == "text" or pd.api.types.is_numeric_dtype(values):
        return values
    return pd.to_numeric(values, errors="coerce")
