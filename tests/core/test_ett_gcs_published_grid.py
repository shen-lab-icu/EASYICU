"""ett_gcs is published once: the public API and a module export are identical.

Both read ``window_expansion.publish_inclusive_any_window`` (ricu ``expand(x,
aggregate = "any")``): each window covers its start hour through
``floor(start + dur)`` inclusive, TRUE wins a shared hour, FALSE windows stay.
Before r13gcs the eICU export kept start hours only (FALSE rows had zero
duration) while the public API expanded TRUE windows half-open and dropped
FALSE ones. Synthetic rows only.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

import easyicu
from easyicu.api import extraction as api
from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.table.duration import resolve_dur_var_hours
from tests.support.eicu_source import DICTIONARY, _eicu_source

pytestmark = pytest.mark.ricu_parity

_PAIR = "Sedation Scale/Score/Goal"
_KEYS = ["patientunitstayid", "charttime"]


def _nursecharting():
    rows = []
    # deep sedation at minute 5 and minute 725, light at minute 605: the
    # second TRUE window (hours 12..18) overlaps the FALSE one (10..16).
    for offset, score in ((5, "-4"), (605, "-1"), (725, "-5")):
        rows += [(offset, "Scores", _PAIR, "Sedation Scale", "RASS"),
                 (offset, "Scores", _PAIR, "Sedation Score", score)]
    nurse = pd.DataFrame(rows, columns=[
        "nursingchartoffset", "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue",
    ])
    nurse["nursingchartentryoffset"] = nurse["nursingchartoffset"]
    nurse["nursingchartid"] = np.arange(len(nurse), dtype=float)
    return nurse


def _bytes(frame: pd.DataFrame) -> bytes:
    frame = frame[[*_KEYS, "ett_gcs"]].dropna(subset=["ett_gcs"])
    frame = frame.sort_values(_KEYS, kind="mergesort").reset_index(drop=True)
    frame = frame.astype({"patientunitstayid": "int64", "charttime": "float64", "ett_gcs": "bool"})
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue()


def test_public_api_and_module_export_publish_identical_ett_gcs(tmp_path):
    source = tmp_path / "src"
    _eicu_source(source, "eicu", nursecharting=_nursecharting())

    public = easyicu.load_concepts(
        ["ett_gcs"], database="eicu", data_path=str(source), merge=True, verbose=False,
    )
    output = tmp_path / "export"
    output.mkdir()
    api._run_module_extraction(
        module_name="respiratory", concepts=list(api.EXTRACT_MODULES["respiratory"]),
        database="eicu", data_path=str(source),
        patient_ids_filter={"patientunitstayid": [1]}, batch_size=10,
        output_dir=str(output),
    )
    manifest = json.loads((output / "_manifest.json").read_text())
    exported = pd.read_parquet(manifest["saved"]["respiratory"]["path"])

    assert public["ett_gcs"].dtype == bool
    expected = {**{float(h): True for h in range(7)},
                **{float(h): False for h in (10, 11)},
                **{float(h): True for h in range(12, 19)}}
    assert dict(zip(public["charttime"], public["ett_gcs"])) == expected
    assert _bytes(public) == _bytes(exported)


def test_false_rows_keep_the_window_duration(tmp_path):
    # ricu ts_to_win_tbl sets dur_var := win_dur on every row, TRUE or FALSE.
    source = _eicu_source(tmp_path, "eicu", nursecharting=_nursecharting())
    table = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["ett_gcs"], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": [1]},
        concept_workers=1, verbose=False,
    )["ett_gcs"]
    frame = table.data
    assert frame["ett_gcs"].tolist() == [True, False, True]
    assert resolve_dur_var_hours(frame).tolist() == [6.0, 6.0, 6.0]


def test_sicdb_ett_gcs_keeps_its_declared_time_index(tmp_path):
    # The single win_tbl output picked its time column from a name list that
    # lacks SICdb's ``Offset``, so the public ett_gcs had no time at all.
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "sic"
    generate_source(root, "sic", stays=3)
    frame = easyicu.load_concepts(
        ["ett_gcs"], database="sic", data_path=str(root), merge=True, verbose=False,
    )
    assert "Offset" in frame.columns
    assert frame["ett_gcs"].dtype == bool
