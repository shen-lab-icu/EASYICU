"""Every stay-table read of a call has one owning batch; batches never evict each other (Codex delta7c).

D7C-PRELOAD: wide/long-table preloads read the stay table before the call's
batch was opened, so their entries had no owner: one stayed held after the
call, and when the call's later reads switched to the new token the
preloaded entry was evicted and a stay table rewritten in between was
accepted instead of refused. The batch is now opened before anything in
the call can read a stay table.

D7C-CROSS-BATCH: a lookup evicted every entry of another scope, so an
independent batch (another thread's, or an interleaved one) evicted this
batch's entry and a rewritten file was then accepted. Eviction now stays
within the owning batch, and a streamed batch boundary releases only its
own batch. Rows are synthetic.
"""

from __future__ import annotations

import contextvars
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import stay_tables
from easyicu.concept.stay_tables import (
    StayTableSourceChanged,
    clear_stay_tables,
    held_stay_tables,
    open_stay_table_batch,
    release_stay_table_batch,
    stay_table_reads,
)
from tests.support.urine_raw_source import _load, _source

pytestmark = pytest.mark.filterwarnings("ignore")

_IDS = np.arange(1, 4)


def _patient() -> pd.DataFrame:
    return pd.DataFrame({
        "patientunitstayid": _IDS, "patienthealthsystemstayid": _IDS,
        "unitdischargeoffset": 6000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
    })


def _eicu(root: Path):
    return _source(
        root, "eicu", patient=_patient(),
        # pap_mean is a single-source wide concept (vitalperiodic preload);
        # crea a long-table one (lab).
        vitalperiodic=pd.DataFrame({
            "vitalperiodicid": _IDS, "patientunitstayid": _IDS, "observationoffset": 150,
            "pasystolic": 25.0, "padiastolic": 15.0, "pamean": 30.0, "heartrate": 80.0,
        }),
        lab=pd.DataFrame({
            "labid": _IDS, "patientunitstayid": _IDS, "labresultoffset": 200,
            "labname": "creatinine", "labresult": 1.0, "labresultrevisedoffset": 200,
            "labmeasurenamesystem": "mg/dL",
        }),
    )


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def _load_batch(source, names):
    return _load(source, names, patient_ids={"patientunitstayid": list(_IDS)})


def _load_ids(source, ids):
    return _load(source, ["pap_mean"], patient_ids=ids)


@pytest.fixture(autouse=True)
def _fresh_owner():
    clear_stay_tables()
    yield
    clear_stay_tables()


def test_a_call_with_a_preload_holds_nothing_afterwards(tmp_path):
    _load_batch(_eicu(tmp_path / "eicu"), ["pap_mean"])
    assert held_stay_tables() == 0


def test_a_stay_table_rewritten_between_preload_and_a_later_read_is_refused(tmp_path, monkeypatch):
    source = _eicu(tmp_path / "eicu")
    patient = tmp_path / "eicu" / "patient.parquet"
    real = stay_tables.stay_table_frame
    calls = []

    def rewrite_after_the_first_read(binding, scope, read):
        calls.append(binding.table)
        frame = real(binding, scope, read)
        if len(calls) == 1:
            _touch(patient)  # the installation changes inside the call
        return frame

    monkeypatch.setattr(stay_tables, "stay_table_frame", rewrite_after_the_first_read)
    with pytest.raises(StayTableSourceChanged) as error:
        _load_batch(source, ["pap_mean", "crea"])
    assert error.value.reason == "stay_table_source_changed"
    assert len(calls) >= 2  # a later read of the same call met the change
    assert held_stay_tables() == 0


def _in(context: contextvars.Context, fn, *args):
    return context.run(fn, *args)


def test_interleaved_batches_do_not_evict_each_other(tmp_path):
    source = _eicu(tmp_path / "eicu")
    patient = tmp_path / "eicu" / "patient.parquet"
    a, b = contextvars.copy_context(), contextvars.copy_context()
    _token_a, close_a = _in(a, open_stay_table_batch)
    _token_b, close_b = _in(b, open_stay_table_batch)
    before = stay_table_reads()
    _in(a, _load_batch, source, ["pap_mean"])
    _in(b, _load_batch, source, ["pap_mean"])  # an independent batch, same stays
    assert stay_table_reads() - before == 2  # independent batches share nothing
    assert held_stay_tables() == 2
    _touch(patient)
    # Each live batch still holds its binding and refuses the change itself.
    for context in (a, b):
        with pytest.raises(StayTableSourceChanged) as error:
            _in(context, _load_batch, source, ["pap_mean"])
        assert error.value.reason == "stay_table_source_changed"
    _in(a, close_a)
    assert held_stay_tables() == 1  # closing one batch leaves the other's entry
    _in(b, close_b)
    assert held_stay_tables() == 0


def test_a_batch_boundary_releases_only_its_own_batch(tmp_path):
    source = _eicu(tmp_path / "eicu")
    a, b = contextvars.copy_context(), contextvars.copy_context()
    _, close_a = _in(a, open_stay_table_batch)
    _, close_b = _in(b, open_stay_table_batch)
    _in(a, _load_batch, source, ["pap_mean"])
    _in(b, _load_batch, source, ["pap_mean"])
    _in(a, release_stay_table_batch)
    assert held_stay_tables() == 1
    _in(a, close_a), _in(b, close_b)
    assert held_stay_tables() == 0


def test_an_entry_needs_an_owning_batch():
    with pytest.raises(ValueError):
        stay_tables.stay_table_frame(None, (None, ("patientunitstayid", frozenset())), pd.DataFrame)


def test_another_batch_on_other_stays_does_not_evict_ours(tmp_path):
    source = _eicu(tmp_path / "eicu")
    patient = tmp_path / "eicu" / "patient.parquet"
    a, b = contextvars.copy_context(), contextvars.copy_context()
    _, close_a = _in(a, open_stay_table_batch)
    _, close_b = _in(b, open_stay_table_batch)
    ours = {"patientunitstayid": [1, 2]}
    _in(a, _load_ids, source, ours)
    _in(b, _load_ids, source, {"patientunitstayid": [2, 3]})
    _touch(patient)
    with pytest.raises(StayTableSourceChanged):
        _in(a, _load_ids, source, ours)
    _in(a, close_a), _in(b, close_b)
    assert held_stay_tables() == 0
