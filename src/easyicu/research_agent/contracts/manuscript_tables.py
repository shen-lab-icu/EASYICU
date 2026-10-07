"""Reader tables an owner declares for its own products.

A signed owner that writes a reader-facing table (its Table 1, its risk-set
accounting) declares it in its step summary (``manuscript_tables``): which of
its products, which generic layout, and the reader words for the layout's
identities.  The reporting owner formats the declared product's recorded
cells and never recomputes one.  Table cells stay out of the bound manuscript
text, so they need no numeric binding: the product's evidence digest is their
provenance.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

MANUSCRIPT_TABLES_KEY = "manuscript_tables"
MANUSCRIPT_TABLE_SCHEMA_VERSION = "easyicu.manuscript_table/1"

#: Reader words: no Markdown, placeholder or table syntax can enter a cell.
_READER_TEXT = r"^[^{}\[\]<>`\\|*_#\n]{1,200}$"


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TableGroup(_Closed):
    """One column group: its field prefix in the product and its reader name.

    ``events`` and ``events_percent`` are the group's recorded outcome events
    and their share of ``n``, when the layout names its outcome.
    """

    prefix: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    label: str = Field(pattern=_READER_TEXT)
    n: int = Field(ge=0)
    events: int | None = Field(default=None, ge=0)
    events_percent: float | None = Field(default=None, ge=0.0, le=100.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _events_are_a_share_of_the_group(self) -> "TableGroup":
        if (self.events is None) != (self.events_percent is None):
            raise ValueError("group events need their recorded share")
        if self.events is not None and (
            self.events > self.n
            or abs(self.events_percent - (100.0 * self.events / self.n if self.n else 0.0)) > 1e-6
        ):
            raise ValueError("group events contradict their recorded share of the group")
        return self


class GroupedSummaryLayout(_Closed):
    """Variables, or variable levels, summarized in each group.

    A categorical row records ``<prefix>_n``, ``<prefix>_denominator`` and
    ``<prefix>_percent``; a continuous row ``<prefix>_mean``, ``<prefix>_sd``,
    ``<prefix>_median``, ``<prefix>_q1`` and ``<prefix>_q3``; every row of
    two or more groups a ``standardized_mean_difference``.  One group describes
    a whole population and compares nothing.  A variable reads by the plan's
    display label for it.  ``events_label`` names the outcome whose recorded
    events every group carries; its row closes the table.
    """

    layout: Literal["grouped_summary"]
    groups: list[TableGroup] = Field(min_length=1, max_length=6)
    events_label: str | None = Field(default=None, pattern=_READER_TEXT)

    @model_validator(mode="after")
    def _distinct_groups(self) -> "GroupedSummaryLayout":
        if len({group.prefix for group in self.groups}) != len(self.groups):
            raise ValueError("table groups must have distinct field prefixes")
        if any((group.events is None) != (self.events_label is None) for group in self.groups):
            raise ValueError("an events row needs every group's events, and only then")
        return self


class StageFlowLayout(_Closed):
    """Consecutive stages with their counts and the records each excluded.

    The product records ``stage_order``, ``stage``, ``count`` and
    ``excluded_since_prior_stage``; every stage it records needs a label.
    """

    layout: Literal["stage_flow"]
    stage_labels: dict[
        Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,80}$")],
        Annotated[str, Field(pattern=_READER_TEXT)],
    ] = Field(min_length=1)


class ManuscriptTableDeclaration(_Closed):
    schema_version: Literal["easyicu.manuscript_table/1"]
    product: str = Field(pattern=r"^table:[a-z][a-z0-9_]{0,79}$")
    caption: str = Field(pattern=_READER_TEXT)
    body: Annotated[
        Union[GroupedSummaryLayout, StageFlowLayout], Field(discriminator="layout")
    ]
    notes: list[Annotated[str, Field(pattern=_READER_TEXT)]] = Field(
        default_factory=list, max_length=8
    )


_DECLARATIONS: TypeAdapter[Any] = TypeAdapter(list[ManuscriptTableDeclaration])


def validate_manuscript_table_declarations(payload: object) -> list[ManuscriptTableDeclaration]:
    """Parse an owner's declarations into their closed types (fail closed)."""

    declarations = _DECLARATIONS.validate_python(payload)
    products = [declaration.product for declaration in declarations]
    if len(products) != len(set(products)):
        raise ValueError("a product is declared as more than one table")
    return declarations


__all__ = [
    "GroupedSummaryLayout",
    "MANUSCRIPT_TABLES_KEY",
    "MANUSCRIPT_TABLE_SCHEMA_VERSION",
    "ManuscriptTableDeclaration",
    "StageFlowLayout",
    "TableGroup",
    "validate_manuscript_table_declarations",
]
