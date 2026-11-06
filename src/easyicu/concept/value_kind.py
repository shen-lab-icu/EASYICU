"""Declared value kind of a concept: numeric, logical or text.

The kind comes from the concept dictionary (class, numeric bounds, unit) or,
for an undeclared recipe, from the recipe's documented output type. It never
comes from the values a batch happens to hold: a sampled decision ("most of
the first rows parse as numbers") let the stays sharing a batch decide
whether a stay's hour was a median, a logical any or a first row.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
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

#: Output type of each unbounded recipe whose dictionary entry declares no
#: unit. The sealed dictionary does not carry it; each entry is the type its
#: callback publishes (exported Arrow type on the demos where exported).
_RECIPE_KINDS: dict[str, ValueKind] = {
    "acute_rrt_input": "logical",
    "avpu": "text",
    "delirium_tx": "logical",
    "delirium_tx_evidence": "text",
    "gcs": "numeric",
    "kdigo_aki": "numeric",
    "kdigo_creat": "numeric",
    "kdigo_creatinine_input": "numeric",
    "kdigo_uo": "numeric",
    "kdigo_urine_input": "numeric",
    "mews": "numeric",
    "news": "numeric",
    "pafi": "numeric",
    "qsofa": "numeric",
    "safi": "numeric",
    "sep3": "logical",
    "sep3_sofa2": "logical",
    "sirs": "numeric",
    "sofa": "numeric",
    "sofa2": "numeric",
    "sofa2_cardio": "numeric",
    "sofa2_cns": "numeric",
    "sofa2_cns_ascertainment": "text",
    "sofa2_cns_delirium_tx_ascertainment": "text",
    "sofa2_cns_proxy_sensitivity": "numeric",
    "sofa2_coag": "numeric",
    "sofa2_liver": "numeric",
    "sofa2_renal": "numeric",
    "sofa2_resp": "numeric",
    "sofa_cardio": "numeric",
    "sofa_cns": "numeric",
    "sofa_coag": "numeric",
    "sofa_liver": "numeric",
    "sofa_renal": "numeric",
    "sofa_resp": "numeric",
    "supp_o2": "logical",
    "susp_inf": "logical",
    "uo_12h": "numeric",
    "uo_24h": "numeric",
    "uo_6h": "numeric",
    "vaso_ind": "logical",
    "vent_ind": "logical",
}


class ConceptValueKindUndeclared(ConceptError):
    """The declaration does not settle a concept's value kind.

    Raised for a declaration that names no kind, and for a classless,
    unbounded concept (numeric only by default) whose values do not parse:
    such a concept must declare its class rather than lose its text.
    """

    def __init__(self, *, concept_id: str, database: str, class_name: object, detail: str = "") -> None:
        self.concept_id = str(concept_id)
        self.database = str(database)
        self.class_name = class_name
        super().__init__(
            f"concept {self.concept_id!r} on {self.database!r}: "
            + (detail or (
                f"its declaration (class {class_name!r}, no numeric bounds, no unit, "
                "no recipe output type) does not say whether it is numeric, logical "
                "or text"
            ))
            + ". Refusing to decide from the batch's values; declare class "
            "(num_cncpt, lgl_cncpt, fct_cncpt or chr_cncpt)."
        )


def _units(definition) -> list[str]:
    units = getattr(definition, "units", None)
    if units is None:
        return []
    return [str(unit).strip().lower() for unit in (units if isinstance(units, (list, tuple)) else [units])]


def implicitly_numeric(definition) -> bool:
    """True for a classless, unbounded concept: numeric only by default."""

    declared = definition.class_name
    return not declared and definition.minimum is None and definition.maximum is None


def declared_value_kind(definition, *, database: str) -> ValueKind:
    """Numeric, logical or text from the declaration of ``definition``.

    Numeric bounds declare a number whatever the class. No class is ricu's
    default ``num_cncpt``. A recipe takes its kind from a declared unit
    ('boolean' is logical, any other unit numeric) or its documented output.
    A class list must agree on one kind.
    """

    if definition.minimum is not None or definition.maximum is not None:
        return "numeric"
    declared = definition.class_name
    classes = [str(name) for name in declared] if isinstance(declared, (list, tuple)) else [declared or "num_cncpt"]
    if classes == ["rec_cncpt"]:
        units = _units(definition)
        if units:
            return "logical" if units == ["boolean"] else "numeric"
        kind = _RECIPE_KINDS.get(definition.name)
        if kind is None:
            raise ConceptValueKindUndeclared(concept_id=definition.name, database=database, class_name=declared)
        return kind
    kinds = {_CLASS_KINDS.get(name) for name in classes}
    if len(kinds) != 1 or None in kinds:
        raise ConceptValueKindUndeclared(concept_id=definition.name, database=database, class_name=declared)
    return kinds.pop()


def coerce_declared_values(values: pd.Series, kind: ValueKind, *, refuse_text_for=None) -> pd.Series:
    """Give a value column its declared type; a text column stays as it is.

    Numeric cells parse with ``pd.to_numeric`` (the parser of the sampled
    route). Logical cells keep a boolean, read a number as nonzero, and lose
    anything else. Unparseable cells of a declared numeric concept (a class or
    numeric bounds) are missing values. ``refuse_text_for=(definition,
    database)`` marks a concept that is numeric only by default: a value that
    does not parse raises instead of becoming missing.
    """

    if kind == "text":
        return values
    if kind == "numeric":
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            return values
        parsed = pd.to_numeric(values, errors="coerce")
        if refuse_text_for is not None:
            unparsed = int((parsed.isna() & values.notna()).sum())
            if unparsed:
                definition, database = refuse_text_for
                raise ConceptValueKindUndeclared(
                    concept_id=definition.name, database=database, class_name=definition.class_name,
                    detail=(
                        f"{unparsed} value(s) do not parse as numbers, and the concept is "
                        "numeric only by default (no class, no numeric bounds)"
                    ),
                )
        return parsed
    if pd.api.types.is_bool_dtype(values):
        return values
    flags = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    numbers = pd.to_numeric(values.where(~flags), errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="boolean")
    out[flags] = values[flags].astype(bool)
    out[numbers.notna()] = numbers[numbers.notna()].ne(0)
    return out


def default_aggregate(kind: ValueKind, *, nullable_status: bool) -> str:
    """ricu's default hourly aggregate for a value kind."""

    if kind == "numeric":
        return "median"
    if kind == "logical":
        return "max" if nullable_status else "any"
    return "first"
