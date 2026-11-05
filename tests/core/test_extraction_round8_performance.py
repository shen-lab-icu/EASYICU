"""Round-8 performance fixes that must not change extracted values."""
import copy
import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept.immutable_receipts import FrozenDict, FrozenList, freeze, freeze_sic_attrs


def _receipt():
    return {"overlap_seconds": [1.0, 2.5], "rows": [{"CaseID": 1, "_h": 3.0}], "label": "x"}


def test_frozen_receipt_is_shared_by_copies_and_refuses_mutation():
    frozen = freeze(_receipt())
    assert frozen == _receipt()
    assert copy.deepcopy(frozen) is frozen and copy.copy(frozen) is frozen
    assert json.dumps(frozen, sort_keys=True) == json.dumps(_receipt(), sort_keys=True)
    with pytest.raises(TypeError):
        frozen["label"] = "y"
    with pytest.raises(TypeError):
        frozen["overlap_seconds"].append(3.0)
    with pytest.raises(TypeError):
        frozen["rows"][0]["_h"] = 4.0


def test_frozen_receipt_survives_pickling_between_worker_processes():
    # The default list/dict reduction replays items through denied methods.
    restored = pickle.loads(pickle.dumps(freeze(_receipt())))
    assert restored == _receipt()
    assert isinstance(restored, FrozenDict) and isinstance(restored["overlap_seconds"], FrozenList)
    with pytest.raises(TypeError):
        restored["overlap_seconds"].append(3.0)


def test_derived_frames_share_the_sic_receipt_without_deep_copies():
    frame = pd.DataFrame({"CaseID": [1, 1], "Offset": [0.0, 1.0]})
    frame.attrs["sicdb_interval_coverage"] = _receipt()
    frame.attrs["other"] = [1]
    freeze_sic_attrs(frame)
    derived = frame.loc[frame.Offset.ge(0)].copy()
    assert derived.attrs["sicdb_interval_coverage"] is frame.attrs["sicdb_interval_coverage"]
    assert type(frame.attrs["other"]) is list  # only sicdb_* receipts are frozen
    restored = pickle.loads(pickle.dumps(derived))
    assert restored.attrs["sicdb_interval_coverage"] == _receipt()


class _Diagnoses:
    """eICU diagnosis table honouring the IN filter the loader pushes down."""

    def __init__(self, frame):
        self.frame = frame
        self.filters = []

    def load_table(self, name, columns=None, filters=None):
        assert name == "diagnosis"
        self.filters.append(filters)
        frame = self.frame
        for spec in filters or ():
            frame = frame.loc[frame[spec.column].isin(spec.value)]
        return SimpleNamespace(data=frame.reset_index(drop=True))


DIAGNOSES = pd.DataFrame({
    "patientunitstayid": [1, 1, 2, 3, 3, 4],
    "icd9code": ["428.0, I50.9", "250.00", "585.9, N18.9", "496, J44.9", "", "571.5, K74.60"],
})


@pytest.mark.parametrize("system", ["charlson", "elixhauser"])
@pytest.mark.parametrize("ids", [None, [1, 3], [2, 4, 99], [99]])
def test_eicu_comorbidity_reads_only_the_requested_stays(monkeypatch, system, ids):
    from easyicu.scores import comorbidity

    full_source = _Diagnoses(DIAGNOSES)
    monkeypatch.setattr(comorbidity, "_build_datasource", lambda *_a, **_k: full_source)
    whole = comorbidity.load_comorbidity("eicu", system=system)
    sample_source = _Diagnoses(DIAGNOSES)
    monkeypatch.setattr(comorbidity, "_build_datasource", lambda *_a, **_k: sample_source)
    got = comorbidity.load_comorbidity("eicu", system=system, patient_ids=ids)

    expected = whole if ids is None else whole.loc[whole.patientunitstayid.isin(ids)]
    pd.testing.assert_frame_equal(got.sort_values("patientunitstayid").reset_index(drop=True),
                                  expected.sort_values("patientunitstayid").reset_index(drop=True))
    if ids is None:
        assert sample_source.filters == [None]
    else:
        assert [spec.value for spec in sample_source.filters[0]] == [list(ids)]
    if ids == [99]:
        assert got.empty
