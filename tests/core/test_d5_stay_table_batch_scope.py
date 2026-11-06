"""A stay-table frame and its binding live for one unit, never past it (Codex delta5, test_duckdb_raw_parity).

The process-level stay-table owner keyed a frame by the stays alone, and
only a streamed export's batch boundary released it. Outside an export,
a frame read by one top-level ``load_concepts`` stayed held after the call
returned; a later call for the same stays (another read path, a rerun
after the installation was rewritten) met it and was refused as
``stay_table_source_changed``, although no batch was live any more. Now
a top-level call outside a unit is its own unit and drops its binding
and payload on return, while a run's ``stay_table_unit`` still shares one
read across its loads and still refuses a file that changes inside it.
Rows are synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.stay_tables import (
    StayTableSourceChanged,
    stay_table_reads,
    stay_table_unit,
)
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from tests.support.bucket_storage import write_fixture_parquet

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"


def _install(root: Path, discharge_minutes: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_fixture_parquet(pd.DataFrame({
        "patientunitstayid": [1], "patienthealthsystemstayid": [1],
        "unitdischargeoffset": [discharge_minutes], "hospitaladmitoffset": [0],
        "admissionweight": [70.0],
    }), root / "patient.parquet", index=False)
    (root / "vitalperiodic").mkdir(exist_ok=True)
    write_fixture_parquet(pd.DataFrame({
        "vitalperiodicid": [1, 2, 3], "patientunitstayid": [1, 1, 1],
        "observationoffset": [30, 90, 13 * 60], "heartrate": [70.0, 80.0, 90.0],
    }), root / "vitalperiodic" / "part.parquet", index=False)
    return root


def _hr(root: Path) -> pd.DataFrame:
    return ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["hr"], ICUDataSource(load_src_cfg("eicu"), base_path=root, enable_cache=False),
        merge=True, r_compatible=False, interval=pd.Timedelta(hours=1),
        patient_ids={"patientunitstayid": [1]}, concept_workers=1, verbose=False,
    )


def test_a_finished_call_holds_nothing(tmp_path):
    root = _install(tmp_path / "eicu", 10 * 60)
    before = stay_table_reads()
    _hr(root)
    _hr(root)
    # Each top-level call outside a unit is its own unit.
    assert stay_table_reads() - before == 2


def test_a_rewritten_installation_is_read_fresh_by_the_next_call(tmp_path):
    root = _install(tmp_path / "eicu", 10 * 60)
    _hr(root)
    _install(root, 20 * 60)  # the stay now lasts 20 h: a later hour is in the episode
    later = _hr(root)
    fresh = _hr(_install(tmp_path / "fresh", 20 * 60))
    pd.testing.assert_frame_equal(later, fresh, check_exact=True)


def test_a_change_inside_one_batch_is_still_refused(tmp_path):
    root = _install(tmp_path / "eicu", 10 * 60)
    with stay_table_unit():
        _hr(root)
        _install(root, 20 * 60)
        with pytest.raises(StayTableSourceChanged) as error:
            _hr(root)
    assert error.value.reason == "stay_table_source_changed"
    # The unit is over: its binding is dropped and the new bytes are read.
    pd.testing.assert_frame_equal(
        _hr(root), _hr(_install(tmp_path / "fresh", 20 * 60)), check_exact=True)


def test_nested_units_keep_the_outer_binding(tmp_path):
    root = _install(tmp_path / "eicu", 10 * 60)
    before = stay_table_reads()
    with stay_table_unit() as outer:
        with stay_table_unit() as inner:
            assert inner is outer
            _hr(root)
        _hr(root)
    assert stay_table_reads() - before == 1
