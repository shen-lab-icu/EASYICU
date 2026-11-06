"""Out-of-bounds values are dropped per row and counted, the same for any batch size.

lead-r13 bounds ruling: no bounds-evidence refusal and no unit guess. A value
outside a concept's declared ``[min, max]`` is dropped per row, as ricu's
``filter_bounds`` drops it, whatever else shares its read. The bounds-unit-
suspect refusal (valsc #14, extended in Codex delta4 V6 and delta5
D5V-WIDE-SINGLE) refused a read of 100+ values with none in bounds, so a stay
whose values were all out of bounds was refused when read alone (batch 1) and
dropped when read with ordinary stays (eICU cvp_raw, etco2 on full data).

Every bounded path is checked with one synthetic all-out-of-bounds stay (120
values) and ordinary stays that each carry one out-of-bounds value: the
published values and the recorded counts (``easyicu.source_bounds``) are the
same read one stay at a time as read all at once. Synthetic rows only.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from easyicu.source_bounds import collect_bounds_drops
from easyicu.concept.aggregation_route import force_raw_aggregation, record_duckdb_routes
from easyicu.resources import load_dictionary
from tests.support.urine_raw_source import _load, _source

pytestmark = pytest.mark.filterwarnings("ignore")

_STAYS = 6
_SUSPECT = 1  # every value out of bounds, 120 of them
_SUSPECT_VALUES = 120
_ORDINARY_VALUES = 5  # plus one out-of-bounds value per ordinary stay


def _hours(n):
    return np.arange(n, dtype=float) + 1.25


def _sic(root):
    ids = np.arange(1, _STAYS + 1)
    cases = pd.DataFrame({
        "CaseID": ids, "PatientID": ids, "ICUOffset": 0.0,
        "TimeOfStay": 200.0 * 3600, "WeightOnAdmission": 70000.0,
    })
    rows = []
    for stay in ids:
        if stay == _SUSPECT:
            hours, temp = _hours(_SUSPECT_VALUES), np.full(_SUSPECT_VALUES, 98.6)
        else:
            hours = _hours(_ORDINARY_VALUES + 1)
            temp = np.r_[np.full(_ORDINARY_VALUES, 36.5 + stay / 10), 45.0]
        rows.append(pd.DataFrame({"CaseID": stay, "DataID": 709, "Offset": hours * 3600, "Val": temp}))
        rows.append(pd.DataFrame({"CaseID": stay, "DataID": 710, "Offset": hours * 3600, "Val": 96.0}))
    return _source(root, "sic", cases=cases, data_float_h=pd.concat(rows, ignore_index=True))


def _eicu(root):
    ids = np.arange(1, _STAYS + 1)
    rows = []
    for stay in ids:
        if stay == _SUSPECT:
            minutes, pas = _hours(_SUSPECT_VALUES) * 60, np.full(_SUSPECT_VALUES, 500.0)
        else:
            minutes = _hours(_ORDINARY_VALUES + 1) * 60
            pas = np.r_[np.full(_ORDINARY_VALUES, 20.0 + stay), 300.0]
        rows.append(pd.DataFrame({
            "patientunitstayid": stay, "observationoffset": minutes,
            "pasystolic": pas, "padiastolic": 15.0, "pamean": 30.0,
        }))
    vitals = pd.concat(rows, ignore_index=True)
    vitals.insert(0, "vitalperiodicid", np.arange(1, len(vitals) + 1))
    return _source(
        root, "eicu",
        patient=pd.DataFrame({
            "patientunitstayid": ids, "patienthealthsystemstayid": ids,
            "unitdischargeoffset": 12000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
        }),
        vitalperiodic=vitals,
    )


def _values(out, names):
    frames = {}
    for name in names:
        data = out[name].data
        data = data.loc[data[name].notna()]
        keys = [c for c in data.columns if c != name]
        frames[name] = data.sort_values(keys, kind="mergesort").reset_index(drop=True)
    return frames


def _read(source, names, id_column, stays, *, raw=False, **kwargs):
    route = force_raw_aggregation() if raw else contextlib.nullcontext()
    with route, record_duckdb_routes() as routes, collect_bounds_drops() as drops:
        out = _load(source, names, patient_ids={id_column: list(stays)}, **kwargs)
    return _values(out, names), drops, dict(routes)


def _batch_one_vs_all(source, names, id_column, **kwargs):
    whole, whole_drops, routes = _read(source, names, id_column, range(1, _STAYS + 1), **kwargs)
    parts, part_drops = {name: [] for name in names}, {}
    from easyicu.source_bounds import merge_bounds_drops

    for stay in range(1, _STAYS + 1):
        values, drops, _ = _read(source, names, id_column, [stay], **kwargs)
        for name in names:
            if len(values[name]):
                parts[name].append(values[name])
        part_drops = merge_bounds_drops(part_drops, drops)
    for name in names:
        one_at_a_time = pd.concat(parts[name], ignore_index=True)
        keys = [c for c in one_at_a_time.columns if c != name]
        one_at_a_time = one_at_a_time.sort_values(keys, kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(one_at_a_time, whole[name], check_exact=True)
    assert part_drops == whole_drops
    return whole, whole_drops, routes


_EXPECTED = {
    "dropped_values": _SUSPECT_VALUES + (_STAYS - 1),
    "stays_all_out_of_bounds": 1,
    "stays_read": _STAYS,
}


def test_single_concept_sql_owner(tmp_path):
    whole, drops, routes = _batch_one_vs_all(_sic(tmp_path / "sic"), ["temp"], "CaseID")
    assert routes == {"bucketed_single": 1}
    assert drops == {"temp": {"data_float_h": _EXPECTED}}
    assert set(whole["temp"]["CaseID"]) == set(range(2, _STAYS + 1))


def test_multi_concept_sql_owner(tmp_path):
    whole, drops, routes = _batch_one_vs_all(_sic(tmp_path / "sic"), ["temp", "spo2"], "CaseID")
    assert routes == {"long_table_batch": 1}
    assert drops["temp"] == {"data_float_h": _EXPECTED}
    assert drops["spo2"]["data_float_h"]["dropped_values"] == 0
    assert len(whole["spo2"]) > 0


def test_raw_pandas_path(tmp_path):
    whole, drops, routes = _batch_one_vs_all(_sic(tmp_path / "sic"), ["temp"], "CaseID", raw=True)
    assert routes == {}
    assert drops == {"temp": {"data_float_h": _EXPECTED}}


def test_wide_table_sql_owner(tmp_path):
    whole, drops, routes = _batch_one_vs_all(
        _eicu(tmp_path / "eicu"), ["pap_sys", "pap_mean"], "patientunitstayid",
    )
    assert routes == {"wide_table_batch": 1}
    assert drops["pap_sys"] == {"vitalperiodic": _EXPECTED}
    assert drops["pap_mean"]["vitalperiodic"]["dropped_values"] == 0


@pytest.mark.parametrize("aggregate", ["min", "mean"])
def test_wide_single_column_sql(tmp_path, aggregate):
    whole, drops, routes = _batch_one_vs_all(
        _eicu(tmp_path / "eicu"), ["pap_sys"], "patientunitstayid", aggregate=aggregate,
    )
    assert routes == {"wide_single_column": 1}
    assert drops == {"pap_sys": {"vitalperiodic": _EXPECTED}}


def test_native_export_contract():
    from easyicu.api.extraction import _enforce_native_export_concept_bounds

    dictionary = load_dictionary(include_sofa2=True)
    rows = []
    for stay in range(1, _STAYS + 1):
        n = _SUSPECT_VALUES if stay == _SUSPECT else _ORDINARY_VALUES + 1
        temp = np.full(n, 98.6) if stay == _SUSPECT else np.r_[np.full(n - 1, 37.0), 45.0]
        rows.append(pd.DataFrame({"stay_id": stay, "charttime": np.arange(n, dtype=float), "temp": temp}))
    frame = pd.concat(rows, ignore_index=True)

    whole = frame.copy()
    whole_audit = _enforce_native_export_concept_bounds(
        whole, requested_concepts=["temp"], dictionary=dictionary,
    )
    parts, totals = [], {"excluded_out_of_bounds": 0, "stays_all_out_of_bounds": 0}
    for _stay, part in frame.groupby("stay_id", sort=True):
        part = part.copy()
        audit = _enforce_native_export_concept_bounds(
            part, requested_concepts=["temp"], dictionary=dictionary,
        )
        parts.append(part)
        for field in totals:
            totals[field] += audit["temp"][field]
    pd.testing.assert_frame_equal(pd.concat(parts), whole, check_exact=True)
    assert totals == {
        "excluded_out_of_bounds": _EXPECTED["dropped_values"],
        "stays_all_out_of_bounds": 1,
    }
    assert {k: whole_audit["temp"][k] for k in totals} == totals


def test_qc_warns_on_a_source_with_many_all_out_of_bounds_stays():
    from easyicu.io.data_quality import (
        SOURCE_BOUNDS_WARN_MIN_STAYS,
        source_bounds_warnings,
    )

    # Full-eICU counts of 2026-11-06 (cvp_raw, hgb) and this file's synthetic temp.
    cvp = {"dropped_values": 18299, "stays_all_out_of_bounds": 262, "stays_read": 28698}
    drops = {
        "circulatory": {"cvp_raw": {"vitalperiodic": cvp}},
        "vitals": {"cvp_raw": {"vitalperiodic": {**cvp, "stays_read": 100}}},
        "blood_gas": {"hgb": {"lab": {"dropped_values": 215, "stays_all_out_of_bounds": 126, "stays_read": 189012}}},
        "labs": {"temp": {"data_float_h": _EXPECTED}},
    }
    rows = source_bounds_warnings(drops)
    assert [(r["concept"], r["source_table"], r["module"]) for r in rows] == [
        ("cvp_raw", "vitalperiodic", "circulatory"),
    ]
    assert rows[0]["stays_all_out_of_bounds"] >= SOURCE_BOUNDS_WARN_MIN_STAYS


def test_declared_unit_refusals_still_fire():
    from easyicu.utils import declared_units
    from easyicu.utils.declared_units import UndeclaredUnitError, declared_fio2_unit

    with pytest.raises(UndeclaredUnitError) as caught:
        declared_fio2_unit("ratio", site="test")
    assert caught.value.reason == "fio2_unit_invalid"
    assert declared_fio2_unit(None, site="test") is None
    # Only the bounds evidence rule is withdrawn.
    assert not hasattr(declared_units, "refuse_bounds_unit_suspect")
    assert not hasattr(declared_units, "bounds_unit_suspect")


def test_export_bytes_and_manifest_counts_do_not_depend_on_batch_size(tmp_path):
    import hashlib
    import json

    from easyicu.api.concepts import clear_global_loader
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "sic"
    generate_source(root, "sic", stays=_STAYS)
    cases = pd.read_parquet(root / "cases.parquet")
    cases["TimeOfStay"] = 200.0 * 3600
    cases.to_parquet(root / "cases.parquet", index=False)
    _sic(tmp_path / "rows")
    pd.read_parquet(tmp_path / "rows" / "data_float_h.parquet").to_parquet(
        root / "data_float_h.parquet", index=False,
    )

    published = {}
    try:
        for batch_size in (1, _STAYS):
            clear_global_loader()
            out = tmp_path / f"batch_{batch_size}"
            extract_database(
                "sic", data_path=str(root), output_dir=str(out), modules=["outcome", "vitals"],
                patient_ids={"CaseID": list(range(1, _STAYS + 1))}, batch_size=batch_size,
                max_workers=1, native_export_v2=True, stream_output_batches=True, verbose=False,
            )
            manifest = json.loads((out / "_manifest.json").read_text())
            vitals = next(f for f in manifest["files"] if f["module"] == "vitals")
            published[batch_size] = (
                hashlib.sha256((out / vitals["file"]).read_bytes()).hexdigest(),
                vitals["concept_status"]["temp"],
                manifest["source_bounds_drops"],
                manifest["qc_warnings"],
            )
    finally:
        clear_global_loader()
    assert published[1] == published[_STAYS]
    _sha, temp_status, receipt, qc = published[1]
    assert temp_status["source_bounds"] == {"data_float_h": _EXPECTED}
    assert receipt["vitals"]["temp"] == {"data_float_h": _EXPECTED}
    assert qc == {"source_bounds_suspect": []}  # one stay is below the warning floor


def test_a_failed_batch_and_its_resume_count_each_stay_once(tmp_path, monkeypatch):
    """A batch that fails mid-load commits no counts; the resumed run sums the ledger."""
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    source = _sic(tmp_path / "sic")
    kwargs = dict(
        data_path=str(source.base_path), database="sic", concepts=["temp", "spo2"],
        verbose=False, merge=True, concept_workers=1, _defer_empty_columns_to_arrow=True,
    )
    cohort = {"CaseID": list(range(1, _STAYS + 1))}

    def stream(output_dir):
        clear_global_loader()
        output_dir.mkdir(parents=True, exist_ok=True)
        return extraction._stream_module_batches_to_parquet(
            "vitals", ["temp", "spo2"], dict(kwargs), cohort, 2, str(output_dir),
        )

    clean = stream(tmp_path / "clean")["source_bounds"]
    assert clean["temp"] == {"data_float_h": _EXPECTED}

    real = extraction._load_stream_module_batch
    calls = {"n": 0}

    def fail_second_batch_once(*args, **kw):
        calls["n"] += 1
        result = real(*args, **kw)  # the reads are made (and counted) ...
        if calls["n"] == 2:
            raise MemoryError("synthetic")  # ... then the batch fails
        return result

    monkeypatch.setattr(extraction, "_load_stream_module_batch", fail_second_batch_once)
    with pytest.raises(MemoryError):
        stream(tmp_path / "resumed")
    resumed = stream(tmp_path / "resumed")
    assert calls["n"] == 4  # batch 2 ran twice, batch 1 was not redone
    assert resumed["source_bounds"] == clean
    clear_global_loader()


def test_isolated_batch_processes_hand_their_counts_back(tmp_path, monkeypatch):
    from easyicu.api import extraction
    from easyicu.api.concepts import clear_global_loader

    source = _sic(tmp_path / "sic")
    kwargs = dict(
        data_path=str(source.base_path), database="sic", concepts=["temp", "spo2"],
        verbose=False, merge=True, concept_workers=1, _defer_empty_columns_to_arrow=True,
    )
    cohort = {"CaseID": list(range(1, _STAYS + 1))}
    monkeypatch.setattr(extraction, "_requires_isolated_stream_batch", lambda *args: True)
    out = tmp_path / "isolated"
    out.mkdir()
    clear_global_loader()
    info = extraction._stream_module_batches_to_parquet(
        "vitals", ["temp", "spo2"], kwargs, cohort, 2, str(out),
    )
    clear_global_loader()
    assert info["batch_process_isolation"] is True
    assert info["source_bounds"]["temp"] == {"data_float_h": _EXPECTED}
    assert not list(out.glob(".*.bounds.json"))
