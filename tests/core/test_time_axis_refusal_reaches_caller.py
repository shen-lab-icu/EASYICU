"""An undeclared or unknown time-axis unit refuses at the public entry point.

Codex delta4 D4-AXIS-CATCH: the unaligned-axis guard added by the lead-r13
time-axis sweep raised inside the window-expansion ``try`` whose broad
``except Exception`` logged and continued with the raw rows, so a load that
must refuse returned unexpanded windows instead. An unknown declared unit took
the same fallback. Both are typed value failures now
(``TimeAxisUnitError``, reasons ``time_axis_unit_undeclared`` and
``time_axis_unit_unknown``). Synthetic rows only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.config import DataSourceConfig
from easyicu.io.ts_utils import TimeAxisUnitError
from easyicu.table import ICUTable
from easyicu.value_failures import TYPED_VALUE_FAILURES


def _source(declared):
    class Source:
        config = DataSourceConfig(
            name="eicu",
            tables={"infusion": {"defaults": {
                "id_var": "patientunitstayid", "index_var": "startoffset",
                "val_var": "value", "time_vars": ["startoffset", "endtime"],
            }}},
        )
        base_path = None

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            frame = pd.DataFrame({
                "patientunitstayid": [1, 1], "startoffset": [30.0, 400.0],
                "endtime": [150.0, 520.0], "kind": ["x", "x"], "value": [2.0, 3.0],
            })
            for spec in filters or []:
                frame = spec.apply(frame)
            if declared is not None:
                frame.attrs["easyicu_time_columns"] = {"startoffset": declared}
            return ICUTable(
                data=frame, id_columns=["patientunitstayid"], index_column="startoffset",
                value_column="value", time_columns=["startoffset", "endtime"],
            )

    return Source()


def _load(declared):
    dictionary = ConceptDictionary({"rate": ConceptDefinition(
        name="rate", class_name="num_cncpt",
        sources={"eicu": [ConceptSource(
            table="infusion", sub_var="kind", ids=["x"], value_var="value",
            index_var="startoffset",
        )]},
    )})
    return ConceptResolver(dictionary).load_concepts(
        ["rate"], _source(declared), merge=False, interval=pd.Timedelta(hours=1),
        align_to_admission=False, r_compatible=False, verbose=False, concept_workers=1,
    )["rate"]


@pytest.mark.parametrize(
    "declared, reason",
    [(None, "time_axis_unit_undeclared"), ("fortnights", "time_axis_unit_unknown")],
)
def test_an_unaligned_window_without_a_known_unit_refuses(declared, reason):
    with pytest.raises(TimeAxisUnitError) as raised:
        _load(declared)
    assert raised.value.reason == reason


def test_a_declared_unaligned_window_expands():
    expanded = _load("icu_relative_hours")
    assert len(expanded.data) > 2  # the two windows were spread over hours


def test_time_axis_refusals_are_typed_value_failures():
    assert TimeAxisUnitError in TYPED_VALUE_FAILURES
    assert issubclass(TimeAxisUnitError, ValueError)
