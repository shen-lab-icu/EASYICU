"""A dictionary factor stored as numeric codes is a category in Table 1.

pilot1 round 2 (10-22), A4: an eICU plan put ``icu_unit_type`` -- a
``fct_cncpt`` whose dictionary entry lists no levels, stored as float codes
-- in its Table 1 with ``count_percent``.  Table 1 read the float dtype as a
continuous quantity and refused the summary; the planner's three repairs
toggled every row's summary and planning stopped.  A dictionary factor is now
a category whatever its storage type, and a category with no closed levels
before extraction is refused with the repair that can succeed: leave it out.
"""

from __future__ import annotations

from easyicu.research_agent.contracts.table_one_semantics import table_one_variable_kind
from easyicu.research_agent.research_context.typed import declares_factor_values
from easyicu.research_agent.schema import ConceptDescriptor, VariableRole


def _descriptor(name: str, **fields: object) -> ConceptDescriptor:
    return ConceptDescriptor(
        name=name, role=VariableRole.DEMOGRAPHIC, dtype="float64", source_concept=name, **fields,
    )


def test_a_coded_factor_without_listed_levels_is_categorical() -> None:
    unit_type = _descriptor("icu_unit_type")

    assert declares_factor_values(unit_type)
    assert table_one_variable_kind(unit_type, []) == "categorical"
    assert table_one_variable_kind(unit_type, [1.0, 2.0, 3.0]) == "categorical"


def test_a_measured_quantity_and_a_derived_count_stay_continuous() -> None:
    lactate = ConceptDescriptor(
        name="lact", role=VariableRole.LAB, dtype="float64", source_concept="lact",
    )
    unit_count = _descriptor("icu_unit_type_n", unit_normalization="window_nonnull_count")
    unit_count = unit_count.model_copy(update={"source_concept": "icu_unit_type"})

    assert table_one_variable_kind(lactate, []) == "continuous"
    assert not declares_factor_values(unit_count)
    assert table_one_variable_kind(unit_count, []) == "continuous"
