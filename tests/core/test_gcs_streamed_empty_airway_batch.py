"""A batch without any ett_gcs window must still score GCS (synthetic rows only).

When a stay batch has no RASS row, eICU ett_gcs comes from the empty
fwd_concept branch, whose placeholder index is datetime64. The GCS callback
cast that empty frame onto the float hour clock and the streamed eICU export
failed for every GCS-dependent module ("Cannot cast DatetimeArray to dtype
float64", r13gcsfix). No window means no sedated hour.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.table import WinTbl
from easyicu.table.duration import UNIT_MINUTES, set_dur_var_unit
from tests.support.eicu_source import _eicu_source
from tests.support.gcs_ricu import ID, TIME, component, run_gcs

pytestmark = pytest.mark.ricu_parity


def _empty_airway_windows():
    frame = pd.DataFrame({ID: pd.Series([], dtype="int64"), TIME: pd.Series([], dtype="datetime64[s]"),
                          "dur_var": pd.Series([], dtype=float), "ett_gcs": pd.Series([], dtype=bool)})
    set_dur_var_unit(frame, UNIT_MINUTES)  # as the empty fwd_concept branch declares
    return WinTbl(data=frame, id_vars=[ID], index_var=TIME, dur_var="dur_var")


def test_an_empty_datetime_indexed_window_table_marks_no_hour():
    got = run_gcs(
        component("egcs", [0, 3], [2, 3]),
        component("mgcs", [0, 3], [4, 5]),
        component("vgcs", [0, 3], [1, 2]),
        _empty_airway_windows(),
    )
    assert got == {0.0: 7.0, 3.0: 10.0}


def test_a_streamed_batch_without_rass_exports_gcs(tmp_path):
    rows = []
    for offset in (10, 190):
        for name, value in (("Eyes", "3"), ("Motor", "5"), ("Verbal", "4")):
            rows.append((offset, "Neurologic", "Glasgow coma score", name, value))
    nurse = pd.DataFrame(rows, columns=[
        "nursingchartoffset", "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue",
    ])
    nurse["nursingchartentryoffset"] = nurse["nursingchartoffset"]
    nurse["nursingchartid"] = np.arange(len(nurse), dtype=float)
    source = tmp_path / "src"
    _eicu_source(source, "eicu", nursecharting=nurse)
    output = tmp_path / "export"
    output.mkdir()
    api._run_module_extraction(
        "neurological", ["gcs"], "eicu", str(source), {"patientunitstayid": [1]}, 1,
        str(output), stream_output_batches=True,
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["errors"] == []
    frame = pd.read_parquet(manifest["saved"]["neurological"]["path"])
    assert frame["gcs"].dropna().tolist() == [12.0, 12.0]
