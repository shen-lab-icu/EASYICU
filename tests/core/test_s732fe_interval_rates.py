"""Scoped rate grids retain native changes without changing hourly callbacks."""

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.aumc_intervals import expand_intervals
from easyicu.concept.minute_resolution import minute_rate_grid, grid_bin_hours
from easyicu.utils.callback_utils import mimic_rate_mv


def _rate_sources():
    from easyicu.resources import load_dictionary
    return [(name, database, source) for name, definition in load_dictionary().items()
            for database, sources in definition.sources.items()
            for source in sources if database in {"aumc", "miiv", "mimic", "mimic_demo"}
            and source.callback in {"aumc_rate_kg", "mimic_rate_mv"}]


@pytest.mark.parametrize("name,database,source", _rate_sources(),
                         ids=lambda value: value if isinstance(value, str) else value.callback)
def test_every_declared_interval_rate_source_keeps_its_hourly_default(name, database, source):
    """Every dictionary consumer retains the pre-change half-open hourly bins."""
    from dataclasses import replace
    from types import SimpleNamespace
    from easyicu.concept.callback_apply import _apply_callback
    from easyicu.resources import load_dictionary

    if database == "aumc":
        origin = 1234567.25
        raw = pd.DataFrame({"admissionid": 1, "start": origin + np.array([7., 31.]),
                            "stop": origin + np.array([31., 65.]), name: [1., 2.],
                            "doseunit": "mcg", "doserateunit": "min", "doserateperkg": 1,
                            "weight": 70., "iscontinuous": 1, "action": "Nieuwe zak",
                            "ordercategory": "2. Spuitpompen"})
        origins = pd.DataFrame({"admissionid": [1], "admittedat": [origin]})
        source = replace(source, value_var=name)
        index = "start"
        unit = "doseunit"
        expected_time = [origin, origin, origin + 60]
    else:
        origin = pd.Timestamp("2150-01-01 00:00:17")
        ident = "stay_id" if database == "miiv" else "icustay_id"
        unit = "rateuom"
        target = load_dictionary()[name].units[0]
        raw = pd.DataFrame({ident: 1, "starttime": origin + pd.to_timedelta([7, 31], unit="m"),
                            "endtime": origin + pd.to_timedelta([31, 65], unit="m"),
                            "rate": [1., 2.], unit: target, "patientweight": 70.,
                            "ordercategorydescription": "Continuous Med",
                            "statusdescription": "FinishedRunning"})
        origins = pd.DataFrame({ident: [1], "intime": [origin]})
        index = "starttime"
        expected_time = [origin, origin, origin + pd.Timedelta(hours=1)]
    data = SimpleNamespace(load_table=lambda *a, **k: origins)
    result = _apply_callback(raw, source, name, unit_column=unit, data_source=data)
    assert result[index].tolist() == expected_time
    assert result[name].tolist() == [1., 2., 2.]


@pytest.mark.parametrize("database", ["aumc", "miiv"])
def test_haemo_native_export_changes_at_record_minutes(tmp_path, database):
    from easyicu.api import extraction as api
    from easyicu.api.concepts import clear_global_loader

    source = tmp_path / "source"
    if database == "aumc":
        from easyicu.synthetic_sources import generate_source
        generate_source(source, "aumc", stays=3)
        admissions = pd.read_parquet(source / "admissions.parquet")
        origin = admissions.loc[admissions.admissionid.eq(1), "admittedat"].iloc[0]
        pd.DataFrame({"admissionid": 1, "itemid": 7229,
                      "start": origin + np.array([7, 31]) * 60000,
                      "stop": origin + np.array([31, 65]) * 60000,
                      "dose": [0.1, 0.3], "doseunit": "mcg", "doserateunit": "min",
                      "doserateperkg": True, "iscontinuous": 1,
                      "action": "Nieuwe zak", "ordercategory": "2. Spuitpompen"}).to_parquet(source / "drugitems.parquet", index=False)
        patients = {"admissionid": [1]}
    else:
        from tests.support.mimic_synthetic_source import mimic_synthetic_source
        source.mkdir()
        origin = pd.Timestamp("2150-01-01")
        raw = pd.DataFrame({"stay_id": 1, "subject_id": 11, "hadm_id": 21, "itemid": 221906,
                            "starttime": origin + pd.to_timedelta([7, 31], unit="m"),
                            "endtime": origin + pd.to_timedelta([31, 65], unit="m"),
                            "rate": [0.1, 0.3], "rateuom": "mcg/kg/min",
                            "amount": 1., "amountuom": "mg",
                            "statusdescription": "FinishedRunning", "patientweight": 70.,
                            "orderid": [1, 2], "linkorderid": [1, 2], "storetime": origin})
        mimic_synthetic_source(source, "miiv", inputevents=raw)
        patients = {"stay_id": [1]}
    try:
        for module in ("haemo_native", "vasopressors"):
            target = tmp_path / module
            target.mkdir()
            api._run_module_extraction(module_name=module, concepts=["norepi_equiv"], database=database,
                                       data_path=str(source), patient_ids_filter=patients, batch_size=3,
                                       output_dir=str(target))
            import json
            assert not json.loads((target / "_manifest.json").read_text())["errors"]
            frame = pd.read_parquet(target / f"{module}.parquet")
            time = frame.iloc[:, 1]
            present = frame.norepi_equiv.notna()
            minutes = (time[present] * 60).round(6).tolist()
            if module == "haemo_native":
                assert minutes == list(range(7, 65))
                rates = dict(zip(minutes, frame.loc[present, "norepi_equiv"]))
                assert rates[7] == pytest.approx(.1)
                assert rates[31] == pytest.approx(.3)
            else:
                assert minutes == [0., 60.]
                assert frame.loc[present, "norepi_equiv"].tolist() == pytest.approx([.2, .3])
    finally:
        clear_global_loader()


@pytest.mark.parametrize("database", ["aumc", "miiv"])
def test_interval_owner_uses_scoped_grid_and_restores_hourly_default(database):
    starts = np.array([7, 31, 62]) / 60
    stops = np.array([31, 62, 65]) / 60
    if database == "aumc":
        origin = 1234567.25
        frame = pd.DataFrame({"admissionid": 1, "admittedat": origin,
                              "start": origin + starts * 60, "stop": origin + stops * 60,
                              "rate": [1., 2., 3.]})

        def run():
            from easyicu.concept.minute_resolution import active_minute_rate_grid
            out = expand_intervals(frame, start_col="start", stop_col="stop", value_col="rate",
                                   grid=active_minute_rate_grid())
            return (out.start - origin) / 60, out.rate
    else:
        origin = pd.Timestamp("2150-01-01 00:00:17")
        frame = pd.DataFrame({"stay_id": 1, "starttime": origin + pd.to_timedelta(starts, unit="h"),
                              "endtime": origin + pd.to_timedelta(stops, unit="h"),
                              "rate": [1., 2., 3.]})

        def run():
            out = mimic_rate_mv(frame, "rate", stop_var="endtime",
                                admission_times=pd.DataFrame({"stay_id": [1], "intime": [origin]}))
            return (out.starttime - origin).dt.total_seconds() / 3600, out.rate

    with minute_rate_grid({"_minute_resolution": True}, pd.Timedelta(minutes=1)):
        time, rate = run()
    assert time.to_numpy() == pytest.approx(grid_bin_hours(np.arange(7, 65), pd.Timedelta(minutes=1)))
    assert rate.tolist() == [1.] * 24 + [2.] * 31 + [3.] * 3
    time, rate = run()
    assert time.tolist() == [0., 0., 1., 1.]
    assert rate.tolist() == [1., 2., 2., 3.]
