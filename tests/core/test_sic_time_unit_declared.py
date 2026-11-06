"""SICdb time columns carry a declared unit; the values never decide it.

The SIC branch of ``_align_time_to_admission`` used to treat any numeric time
column whose largest absolute value exceeded 5,000 as raw seconds. The DuckDB
loaders already publish ICU-relative hours, so one record lying more than
5,000 h from ICU admission rescaled the whole frame by 1/3600 and collapsed
every stay in the batch onto hour 0 (lead-r13 G1 preflight: SICdb 1.0.8 RASS,
3,000-stay batch). A 60-stay batch without that record was correct, so the
result depended on which stays shared the batch. Synthetic rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.aggregation_route import force_raw_aggregation, record_duckdb_routes
from easyicu.resources import load_dictionary
from tests.support.urine_raw_source import _source

pytestmark = pytest.mark.filterwarnings("ignore")

_RASS = 3123  # SICdb 1.0.8 d_references: RASS - Richmond Agitation-Sedation Scale
_LONG_STAY = 1  # ICU stay of 6,000 h, RASS recorded at 5,500 h
_N = 3000


def _sic(root):
    ids = np.arange(1, _N + 1)
    cases = pd.DataFrame({
        "CaseID": ids, "PatientID": ids, "ICUOffset": 0.0,
        "TimeOfStay": np.where(ids == _LONG_STAY, 6000.0, 72.0) * 3600,
        "WeightOnAdmission": 70000.0,
    })
    hours = np.where(ids == _LONG_STAY, 5500.0, 30.25)
    rows = pd.concat([
        pd.DataFrame({"CaseID": ids, "DataID": _RASS, "Offset": 2.5 * 3600, "Val": -2.0}),
        pd.DataFrame({"CaseID": ids, "DataID": _RASS, "Offset": hours * 3600, "Val": 1.0}),
    ], ignore_index=True)
    return _source(root, "sic", cases=cases, data_float_h=rows)


def _rass(source, ids):
    frame = ConceptResolver(load_dictionary(include_sofa2=True)).load_concepts(
        ["rass"], source, merge=True, interval=pd.Timedelta(hours=1),
        patient_ids={"CaseID": list(ids)}, r_compatible=True, verbose=False, concept_workers=1,
    )
    keys = [c for c in frame.columns if c != "rass" and not c.startswith("_")]
    return frame[keys + ["rass"]].sort_values(keys, kind="mergesort").reset_index(drop=True)


def test_a_far_record_does_not_rescale_the_rest_of_the_batch(tmp_path):
    source = _sic(tmp_path / "sic")
    with record_duckdb_routes() as routes:
        batch = _rass(source, range(1, _N + 1))
    assert routes, "rass did not take the DuckDB aggregation route"

    # Every ordinary stay keeps both hours, exactly as a 60-stay batch without
    # the long stay publishes them.
    small = _rass(source, range(2, 62))
    id_col = small.columns[0]
    pd.testing.assert_frame_equal(
        batch[batch[id_col].isin(range(2, 62))].reset_index(drop=True), small, check_exact=True,
    )
    per_stay = small.groupby(id_col)["charttime"].apply(lambda t: tuple(sorted(t.astype(float))))
    assert len(per_stay) == 60 and set(per_stay) == {(2.0, 30.0)}

    # The long stay's late value stays at its own hour.
    long_stay = batch[batch[id_col] == _LONG_STAY]
    assert sorted(long_stay["charttime"].astype(float)) == [2.0, 5500.0]


def test_the_raw_path_converts_only_the_named_second_columns(tmp_path):
    source = _sic(tmp_path / "sic")
    ids = [_LONG_STAY, 2, 3]
    duck = _rass(source, ids)
    with force_raw_aggregation(), record_duckdb_routes() as raw_routes:
        raw = _rass(source, ids)
    assert not raw_routes
    pd.testing.assert_frame_equal(duck, raw, check_exact=True)
    assert 5500.0 in set(raw["charttime"].astype(float))
