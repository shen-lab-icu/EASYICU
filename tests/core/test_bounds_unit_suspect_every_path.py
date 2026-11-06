"""One bounds-unit-suspect rule on every bounded read path (Codex delta4 V6).

At least 100 values in one read and none within the declared bounds means the
values carry another unit. The single-concept SQL owner refused such a read
with ``UndeclaredUnitError(bounds_unit_suspect)``; the multi-concept and wide
SQL owners, the raw pandas path and the native export contract dropped the same
values and published the concept as absent. The rule now lives in
``utils.declared_units`` and every path calls it. Zeros carry no unit (every
conversion maps 0 to 0) and are not counted: eICU monitors record 0 mmHg
end-tidal CO2 for no signal, and the eICU demo at batch 50 holds a read of
only those. Synthetic rows only.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.aggregation_route import force_raw_aggregation, record_duckdb_routes
from easyicu.resources import load_dictionary
from easyicu.utils.declared_units import (
    BOUNDS_UNIT_SUSPECT_MIN_VALUES,
    UndeclaredUnitError,
    bounds_unit_suspect,
    refuse_bounds_unit_suspect,
)
from tests.support.urine_raw_source import _load, _source

pytestmark = pytest.mark.filterwarnings("ignore")

_N = 150  # stays; one value each, so one read holds _N values
_SIC_TEMP = 709  # temp, bounds [32, 42] degC
_SIC_SPO2 = 710  # spo2, bounds [50, 100] %
_SIC_ETCO2 = 716  # etco2, bounds [10, 60] mmHg


def _sic(root, *, temp=98.6, etco2=38.0):
    ids = np.arange(1, _N + 1)
    cases = pd.DataFrame({
        "CaseID": ids, "PatientID": ids, "ICUOffset": 0.0,
        "TimeOfStay": 72.0 * 3600, "WeightOnAdmission": 70000.0,
    })
    rows = pd.concat([
        pd.DataFrame({"CaseID": ids, "DataID": _SIC_TEMP, "Offset": 2.5 * 3600, "Val": temp}),
        pd.DataFrame({"CaseID": ids, "DataID": _SIC_SPO2, "Offset": 2.5 * 3600, "Val": 96.0}),
        pd.DataFrame({"CaseID": ids, "DataID": _SIC_ETCO2, "Offset": 2.5 * 3600, "Val": etco2}),
    ], ignore_index=True)
    return _source(root, "sic", cases=cases, data_float_h=rows)


def _eicu(root):
    ids = np.arange(1, _N + 1)
    return _source(
        root, "eicu",
        patient=pd.DataFrame({
            "patientunitstayid": ids, "patienthealthsystemstayid": ids,
            "unitdischargeoffset": 6000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
        }),
        # pap_sys bounds [0, 120] mmHg: 500 is another unit; pap_mean/pap_dia are valid.
        vitalperiodic=pd.DataFrame({
            "vitalperiodicid": ids, "patientunitstayid": ids, "observationoffset": 150,
            "pasystolic": 500.0, "padiastolic": 15.0, "pamean": 30.0, "etco2": 0.0,
        }),
    )


def _outcome(source, names, *, raw=False):
    route = force_raw_aggregation() if raw else contextlib.nullcontext()
    with route, record_duckdb_routes() as routes:
        try:
            out = _load(source, names)
        except UndeclaredUnitError as error:
            return ("refused", error.reason), dict(routes)
    return ("published", {k: int(v.data[k].notna().sum()) for k, v in out.items()}), dict(routes)


def _values(source, names):
    out = _load(source, names)
    frames = {}
    for name, table in out.items():
        data = table.data
        keys = [c for c in data.columns if c != name]
        frames[name] = data.sort_values(keys, kind="mergesort").reset_index(drop=True)
    return frames


@pytest.mark.parametrize(("raw", "bounded", "refused"), [
    (BOUNDS_UNIT_SUSPECT_MIN_VALUES - 1, 0, False),
    (BOUNDS_UNIT_SUSPECT_MIN_VALUES, 0, True),
    (BOUNDS_UNIT_SUSPECT_MIN_VALUES, 1, False),
    (0, 0, False),
])
def test_the_rule_needs_100_nonzero_values_and_none_in_bounds(raw, bounded, refused):
    assert bounds_unit_suspect(raw, bounded) is refused
    assert bounds_unit_suspect(raw, 0, 1) is False  # an in-bounds aggregate survived
    if refused:
        with pytest.raises(UndeclaredUnitError) as caught:
            refuse_bounds_unit_suspect(quantity="temp", site="t", nonzero_values=raw,
                                       bounded_non_null=bounded, value_min=32, value_max=42)
        assert caught.value.reason == "bounds_unit_suspect"
        assert caught.value.quantity == "temp"
    else:
        refuse_bounds_unit_suspect(quantity="temp", site="t", nonzero_values=raw,
                                   bounded_non_null=bounded)


def test_single_multi_and_raw_long_table_paths_refuse_alike(tmp_path):
    source = _sic(tmp_path / "sic")

    single, single_routes = _outcome(source, ["temp"])
    multi, multi_routes = _outcome(source, ["temp", "spo2"])
    raw, raw_routes = _outcome(source, ["temp"], raw=True)

    assert single == multi == raw == ("refused", "bounds_unit_suspect")
    assert not raw_routes
    # The multi request went through the multi-concept owner, not one read per concept.
    _, partner_routes = _outcome(source, ["spo2", "etco2"])
    assert partner_routes == {"long_table_batch": 1}


def test_a_valid_concept_in_the_multi_owner_is_unchanged(tmp_path):
    source = _sic(tmp_path / "sic")
    alone = _values(source, ["spo2"])["spo2"]
    with record_duckdb_routes() as routes:
        paired = _values(source, ["spo2", "etco2"])
    assert routes == {"long_table_batch": 1}
    pd.testing.assert_frame_equal(paired["spo2"], alone, check_exact=True)
    assert int(alone["spo2"].notna().sum()) == _N


def test_in_bounds_values_still_publish_on_every_path(tmp_path):
    source = _sic(tmp_path / "sic", temp=37.0)
    single, _ = _outcome(source, ["temp"])
    multi, _ = _outcome(source, ["temp", "spo2"])
    raw, _ = _outcome(source, ["temp"], raw=True)
    assert single == ("published", {"temp": _N})
    assert multi == ("published", {"temp": _N, "spo2": _N})
    assert raw == ("published", {"temp": _N})


def test_wide_table_owner_refuses_like_the_raw_path(tmp_path):
    source = _eicu(tmp_path / "eicu")

    single, single_routes = _outcome(source, ["pap_sys"])
    multi, _ = _outcome(source, ["pap_sys", "pap_mean"])
    raw, raw_routes = _outcome(source, ["pap_sys"], raw=True)

    assert single == multi == raw == ("refused", "bounds_unit_suspect")
    assert not raw_routes

    # Valid columns of the same wide table are unchanged by a paired read.
    alone = _values(source, ["pap_mean"])["pap_mean"]
    with record_duckdb_routes() as routes:
        paired = _values(source, ["pap_mean", "pap_dia"])
    assert routes == {"wide_table_batch": 1}
    pd.testing.assert_frame_equal(paired["pap_mean"], alone, check_exact=True)
    assert int(alone["pap_mean"].notna().sum()) == _N


def test_native_export_contract_refuses_instead_of_nulling_every_value():
    from easyicu.api.extraction import _enforce_native_export_concept_bounds

    dictionary = load_dictionary(include_sofa2=True)
    suspect = pd.DataFrame({"temp": np.full(_N, 98.6), "spo2": np.full(_N, 96.0)})
    with pytest.raises(UndeclaredUnitError) as caught:
        _enforce_native_export_concept_bounds(
            suspect, requested_concepts=["temp", "spo2"], dictionary=dictionary,
        )
    assert caught.value.reason == "bounds_unit_suspect"
    assert caught.value.quantity == "temp"

    # Ordinary implausible values are still nulled one by one.
    mixed = pd.DataFrame({"temp": np.r_[np.full(_N - 3, 37.0), 98.6, 98.6, 98.6]})
    audit = _enforce_native_export_concept_bounds(
        mixed, requested_concepts=["temp"], dictionary=dictionary,
    )
    assert int(mixed["temp"].notna().sum()) == _N - 3
    assert audit["temp"]["excluded_out_of_bounds"] == 3


def test_zeros_alone_are_not_unit_evidence_on_any_path(tmp_path):
    # 150 zeros of etco2 (bounds [10, 60] mmHg) and nothing else: dropped as
    # implausible on every path, exactly as before the shared rule.
    source = _sic(tmp_path / "sic", etco2=0.0)
    single, _ = _outcome(source, ["etco2"])
    multi, multi_routes = _outcome(source, ["etco2", "spo2"])
    raw, _ = _outcome(source, ["etco2"], raw=True)
    assert single == ("published", {"etco2": 0})
    assert multi == ("published", {"etco2": 0, "spo2": _N})
    assert multi_routes == {"long_table_batch": 1}
    assert raw == ("published", {"etco2": 0})

    eicu = _eicu(tmp_path / "eicu")
    wide, wide_routes = _outcome(eicu, ["etco2", "pap_mean"])
    eicu_raw, _ = _outcome(eicu, ["etco2"], raw=True)
    assert wide == ("published", {"etco2": 0, "pap_mean": _N})
    assert wide_routes == {"wide_table_batch": 1}
    assert eicu_raw == ("published", {"etco2": 0})

    from easyicu.api.extraction import _enforce_native_export_concept_bounds

    zeros = pd.DataFrame({"etco2": np.zeros(_N)})
    audit = _enforce_native_export_concept_bounds(
        zeros, requested_concepts=["etco2"], dictionary=load_dictionary(include_sofa2=True),
    )
    assert int(zeros["etco2"].notna().sum()) == 0
    assert audit["etco2"]["excluded_out_of_bounds"] == _N
