"""A passed-through source value outside a factor's declared levels is missing.

``sex`` declares ``Female``/``Male``.  eICU's source maps nothing, so its raw
``Other`` and ``Unknown`` reached models as levels the codebook does not have
(hosted pilot: a landmark model stopped because ``Other`` had no events).  A
typed export now reads them as missing with a receipt; a source whose own map
leaves a value unmapped is a mapping defect and is left alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from easyicu.research_agent.cohort import materializer

_TYPED = SimpleNamespace(column_metadata_sha256="a" * 64)


def _read(values, *, concept="sex", database="eicu", package=_TYPED):
    frame = pd.DataFrame({"stay_id": range(len(values)), concept: values})
    return materializer._missing_outside_declared_source_factor(
        frame, concept=concept, database=database, package=package,
    )


def test_an_unmapped_source_value_outside_the_levels_is_missing_with_a_receipt() -> None:
    read, receipt = _read(["Female", "Other", "Male", "Unknown", None, "Unknown"])

    assert read["sex"].dropna().tolist() == ["Female", "Male"]
    assert read["sex"].isna().tolist() == [False, True, False, True, True, True]
    assert receipt == {
        "policy": "source_value_outside_declared_factor_levels_missing",
        "concept": "sex",
        "database": "eicu",
        "declared_levels": ["Female", "Male"],
        "values_read_as_missing": {"Unknown": 2, "Other": 1},
        "rows_read_as_missing": 3,
    }


def test_values_within_the_levels_are_left_as_they_are() -> None:
    frame, receipt = _read(["Female", "Male", None])

    assert receipt is None
    assert frame["sex"].tolist() == ["Female", "Male", None]


def test_a_source_with_its_own_map_keeps_its_mapping_defect_visible() -> None:
    # MIMIC-IV maps M/F; a value its map leaves is not decided here.
    read, receipt = _read(["Female", "X"], database="miiv")

    assert receipt is None
    assert read["sex"].tolist() == ["Female", "X"]


def test_an_untyped_export_or_a_non_factor_concept_is_not_read_here() -> None:
    assert _read(["Female", "Other"], package=SimpleNamespace(column_metadata_sha256=None))[1] is None
    assert _read(["Female", "Other"], package=None)[1] is None
    assert _read([50.0, 60.0], concept="age")[1] is None
