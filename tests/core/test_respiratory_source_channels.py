"""Synthetic events through real selection, callbacks, and hourly aggregation."""
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary


DICTIONARY = Path(__file__).parents[2] / "src/easyicu/data/concept-dict.json"
CHANNELS = ["spo2", "o2sat", "peep_set", "peep_total", "fio2_chart", "fio2_lab"]


def synthetic_source(tmp_path, database):
    id_column = "stay_id" if database == "miiv" else "icustay_id"
    admission = pd.Timestamp("2180-01-01 00:15:00")
    # Stay 1 has alarms only, stays 2 and 3 have identical actual pulse oximetry.
    # Extra alarms at stay 2 must not lower its measured saturation median.
    chart_rows = [(1, 5, 226253, 80.), (2, 5, 220277, 96.),
                  (2, 25, 220277, 98.), (3, 5, 220277, 96.),
                  (3, 25, 220277, 98.)]
    chart_rows += [(2, minute, 226253, 80.) for minute in [6, 10, 15, 20]]
    chart_rows += [(2, 5, 220339, 8.), (2, 25, 220339, 10.),
                   (2, 15, 224700, 16.), (4, 5, 224700, 20.),
                   (5, 5, 220339, 6.), (2, 5, 223835, .4),
                   (2, 25, 223835, 40.), (5, 5, 223835, .5),
                   (2, 65, 223835, .21), (2, 125, 223835, 1.),
                   (2, 185, 223835, 10.), (2, 245, 223835, 101.),
                   (3, 5, 506, 12.), (3, 25, 506, 14.),
                   (3, 15, 505, 35.),
                   (6, 5, 220277, 60.), (6, 5, 220339, 35.)]
    lab_rows = [(2, 5, 50816, .8), (2, 25, 50816, 80.),
                (4, 5, 50816, .9), (2, 65, 50816, .3),
                (2, 125, 50816, 100.), (2, 185, 50816, 10.),
                (2, 245, 50816, 101.)]
    for table, rows in [("chartevents", chart_rows), ("labevents", lab_rows)]:
        frame = pd.DataFrame(rows, columns=[id_column, "minute", "itemid", "valuenum"])
        frame["charttime"] = admission + pd.to_timedelta(frame.pop("minute"), unit="m")
        frame["storetime"] = frame.charttime
        frame["value"] = frame.valuenum.astype(str)
        frame["valueuom"] = "%"
        frame.loc[frame.itemid.isin([220339, 224700, 506]), "valueuom"] = "cmH2O"
        frame["subject_id"] = frame[id_column] + 100
        frame["hadm_id"] = frame[id_column] + 200
        if table == "labevents":
            frame = frame.drop(columns=id_column)
        folder = tmp_path / table
        folder.mkdir()
        frame.to_parquet(folder / "part.parquet", index=False)
    stays = pd.DataFrame({id_column: range(1, 7), "subject_id": range(101, 107),
                          "hadm_id": range(201, 207), "intime": [admission] * 6,
                          "outtime": [admission + pd.Timedelta(hours=48)] * 6})
    stays.to_parquet(tmp_path / "icustays.parquet", index=False)
    return ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False), id_column


@pytest.mark.parametrize("database", ["miiv", "mimic", "mimic_demo"])
@pytest.mark.parametrize("use_duckdb", [False, True])
def test_actual_loader_keeps_alarms_and_respiratory_channels_separate(
    tmp_path, monkeypatch, database, use_duckdb,
):
    source, id_column = synthetic_source(tmp_path, database)
    import easyicu.datasource as datasource_module

    calls = []
    original = datasource_module.load_bucketed_table_aggregated
    original_multi = datasource_module.load_bucketed_table_multi_aggregated

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args[1], tuple(args[3])))
        return result

    def capture_multi(*args, **kwargs):
        result = original_multi(*args, **kwargs)
        for ids in args[2].values():
            calls.append((args[1], tuple(ids)))
        return result

    monkeypatch.setattr(datasource_module, "load_bucketed_table_aggregated", capture)
    monkeypatch.setattr(datasource_module, "load_bucketed_table_multi_aggregated", capture_multi)
    if not use_duckdb:
        monkeypatch.setattr(source, "resolve_bucket_directory", lambda _table: None)
        monkeypatch.setattr(source, "resolve_flat_parquet_directory", lambda _table: None)
    loaded = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        CHANNELS, source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={id_column: [1, 2, 3, 4, 5]},
        concept_workers=1, verbose=False,
    )

    def values(name, patient):
        frame = loaded[name].data
        assert set(frame[id_column]) <= {1, 2, 3, 4, 5}
        return frame.loc[frame[id_column].eq(patient)].set_index("charttime")[name].dropna().to_dict()

    for saturation in ["spo2", "o2sat"]:
        assert values(saturation, 1) == {}
        assert values(saturation, 2) == values(saturation, 3) == {0.: 97.}
    assert values("peep_set", 2) == {0.: 9.}
    assert values("peep_total", 2) == {0.: 16.}
    assert values("peep_set", 4) == {}
    assert values("peep_total", 4) == {0.: 20.}
    assert values("peep_set", 5) == {0.: 6.}
    assert values("peep_total", 5) == {}
    assert values("peep_set", 3) == ({} if database == "miiv" else {0.: 13.})
    assert values("fio2_chart", 2) == {0.: 40., 1.: 21., 2.: 100.}
    assert values("fio2_lab", 2) == {0.: 80., 1.: 30., 2.: 100.}
    assert values("fio2_chart", 4) == {}
    assert values("fio2_lab", 4) == {0.: 90.}
    assert values("fio2_chart", 5) == {0.: 50.}
    assert values("fio2_lab", 5) == {}
    assert bool(calls) is use_duckdb
    if use_duckdb:
        assert any(table == "labevents" and 50816 in ids for table, ids in calls)
        for item in [220277, 220339, 224700, 223835]:
            assert any(table == "chartevents" and item in ids for table, ids in calls)
    assert all(226253 not in ids for _, ids in calls)


def test_source_contracts_do_not_invent_cross_database_support():
    dictionary = load_dictionary(include_sofa2=True)
    for name in CHANNELS[2:]:
        assert set(dictionary.get(name).sources) == {"miiv", "mimic", "mimic_demo"}
    for database in ["miiv", "mimic", "mimic_demo"]:
        for saturation in ["spo2", "o2sat"]:
            assert all(226253 not in source.ids for source in dictionary.get(saturation).sources[database])
        assert {source.table for source in dictionary.get("fio2_chart").sources[database]} == {"chartevents"}
        assert {source.table for source in dictionary.get("fio2_lab").sources[database]} == {"labevents"}


def test_generic_peep_and_fio2_keep_their_mixed_source_contracts():
    dictionary = load_dictionary()
    assert set(dictionary.get("peep").sources["miiv"][0].ids) == {220339, 224700}
    assert {source.table for source in dictionary.get("fio2").sources["miiv"]} == {"chartevents", "labevents"}


@pytest.mark.parametrize("database", ["miiv", "mimic", "mimic_demo"])
@pytest.mark.parametrize("use_duckdb", [False, True])
@pytest.mark.parametrize("selected", [[10, 20], [20]])
def test_laboratory_channels_preserve_repeated_stay_identity(
    tmp_path, monkeypatch, database, use_duckdb, selected,
):
    id_column = "stay_id" if database == "miiv" else "icustay_id"
    origin = pd.Timestamp("2180-01-01 00:15:00")
    # Two separate ICU stays in one hospital admission; none of the subject,
    # admission, and stay identifiers share numeric values.
    starts = [origin, origin + pd.Timedelta(days=2), origin]
    stays = pd.DataFrame({id_column: [10, 20, 30], "subject_id": [101, 101, 102],
                          "hadm_id": [501, 501, 502], "intime": starts,
                          "outtime": [t + pd.Timedelta(hours=12) for t in starts]})
    stays.to_parquet(tmp_path / "icustays.parquet", index=False)
    records = []
    for subject, admission, start, fraction, oxygen in zip(
        [101, 101, 102], [501, 501, 502], starts, [.4, .8, .99], [70., 90., 120.],
    ):
        for item, value, unit in [(50816, fraction, "%"), (50821, oxygen, "mm Hg")]:
            records.append(dict(subject_id=subject, hadm_id=admission,
                                charttime=start + pd.Timedelta(minutes=5),
                                itemid=item, valuenum=value, value=str(value), valueuom=unit))
    lab = pd.DataFrame(records)
    lab["storetime"] = lab.charttime
    (tmp_path / "labevents").mkdir()
    lab.to_parquet(tmp_path / "labevents/part.parquet", index=False)
    source = ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)
    if not use_duckdb:
        monkeypatch.setattr(source, "resolve_bucket_directory", lambda _table: None)
        monkeypatch.setattr(source, "resolve_flat_parquet_directory", lambda _table: None)
    outputs = ConceptResolver(load_dictionary()).load_concepts(
        ["fio2_lab", "po2"], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={id_column: selected},
        concept_workers=1, verbose=False,
    )
    for name, expected in [("fio2_lab", {10: 40., 20: 80.}), ("po2", {10: 70., 20: 90.})]:
        table = outputs[name]
        assert table.id_columns == [id_column]
        frame = table.data
        assert frame.charttime.tolist() == [0.] * len(selected)
        assert frame.set_index(id_column)[name].to_dict() == {key: expected[key] for key in selected}
