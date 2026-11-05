"""Loader failures surface instead of silently emptying a batch (lead-r13 sweep).

Each case used to catch an unexpected exception, log at most a warning, and
then publish NA or an empty table for the whole batch.  Genuine absence keeps
its existing empty/NA contract; a failure now raises with a reason code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback


class _FailingResolver:
    def _load_single_concept(self, *_args, **_kwargs):
        raise OSError("read failed")

    def load_concepts(self, *_args, **_kwargs):
        raise OSError("read failed")


def _db(name):
    return SimpleNamespace(config=SimpleNamespace(name=name))


def test_hirid_rate_kg_weight_load_failure_raises():
    frame = pd.DataFrame({"patientid": [1], "givenat": [0.0], "givendose": [5.0],
                          "doseunit": ["mg"], "infusionid": [9]})
    with pytest.raises(RuntimeError, match="hirid_rate_kg_weight_load_failed"):
        _apply_callback(frame, ConceptSource(callback="hirid_rate_kg"), concept_name="norepi_rate",
                        resolver=_FailingResolver(), data_source=_db("hirid"))


def test_sic_rate_kg_weight_load_failure_raises():
    frame = pd.DataFrame({"CaseID": [1], "Offset": [0.0], "OffsetDrugEnd": [60.0], "norepi_rate": [0.001]})
    with pytest.raises(RuntimeError, match="sic_rate_kg_weight_load_failed"):
        _apply_callback(frame, ConceptSource(callback="sic_rate_kg"), concept_name="norepi_rate",
                        resolver=_FailingResolver(), data_source=_db("sic"))


def test_vent_mode_rows_without_a_map_raise_but_empty_frames_pass():
    from easyicu.concept.callbacks import apply_vent_mode_frame

    rows = pd.DataFrame({"value": ["PC"]})
    with pytest.raises(ValueError, match="vent_mode_map_unavailable"):
        apply_vent_mode_frame(rows, "value", None, "control", "vent_mode")
    assert apply_vent_mode_frame(rows.iloc[:0], "value", None, "control", "vent_mode").empty


def test_callback_concept_load_failure_raises():
    from easyicu.concept.callbacks import _load_concept_for_callback

    ctx = SimpleNamespace(resolver=_FailingResolver(), data_source=None, patient_ids=None)
    with pytest.raises(RuntimeError, match="callback_concept_load_failed: weight"):
        _load_concept_for_callback(ctx, "weight")


def test_component_merge_without_keys_raises():
    from easyicu.concept.callbacks import _merge_tables
    from easyicu.table import ICUTable

    timed = ICUTable(pd.DataFrame({"stay_id": [1], "charttime": [0.0], "weight": [70.0]}),
                     id_columns=["stay_id"], index_column="charttime", value_column="weight")
    timed2 = ICUTable(pd.DataFrame({"stay_id": [1], "charttime": [0.0], "height": [170.0]}),
                      id_columns=["stay_id"], index_column="charttime", value_column="height")
    original_merge = pd.DataFrame.merge

    def failing(frame, right, *args, **kwargs):
        if "height" in right.columns:
            raise ValueError("You are trying to merge on float64 and object columns")
        return original_merge(frame, right, *args, **kwargs)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd.DataFrame, "merge", failing)
        with pytest.raises(ValueError, match="concept_component_merge_failed: 'height'"):
            _merge_tables({"weight": timed, "height": timed2}, ctx=SimpleNamespace(), how="inner")


def test_eicu_optional_channel_absent_file_is_empty_but_key_error_raises():
    from easyicu.concept.eicu_source_qc import load_optional

    class _Source:
        def __init__(self, exc):
            self.exc = exc

        def load_table(self, *_args, **_kwargs):
            raise self.exc

    assert load_optional(_Source(FileNotFoundError("x")), "respiratorycare", [1], ["a"]).empty
    with pytest.raises(RuntimeError, match="eicu_optional_channel_read_failed"):
        load_optional(_Source(KeyError("patientunitstayid")), "respiratorycare", [1], ["a"])


def test_circ_failure_dependency_failure_raises(monkeypatch):
    from easyicu.scores import circ_failure

    calls = []

    def failing(concepts, **_kwargs):
        calls.append(list(concepts))
        raise OSError("read failed")

    monkeypatch.setattr("easyicu.api.load_concepts", failing, raising=False)
    monkeypatch.setattr(circ_failure, "load_concepts", failing, raising=False)
    with pytest.raises(RuntimeError, match="circ_failure_dependency_load_failed"):
        circ_failure.load_circ_failure(database="miiv", data_path="/nonexistent", patient_ids=[1])


def test_partitioned_read_failure_raises_instead_of_empty(tmp_path, monkeypatch):
    import pyarrow.dataset as pads

    from easyicu.config import DataSourceConfig
    from easyicu.datasource import ICUDataSource

    table = tmp_path / "tbl"
    table.mkdir()
    pd.DataFrame({"stay_id": [1, 2], "a": [1.0, 2.0]}).to_parquet(table / "1.parquet")
    source = ICUDataSource(DataSourceConfig(name="miiv"), base_path=tmp_path)

    def boom(*_args, **_kwargs):
        raise OSError("disk read failed")

    monkeypatch.setattr(pads, "dataset", boom)
    monkeypatch.setattr(pd, "read_parquet", boom)
    with pytest.raises(RuntimeError, match="partitioned_read_failed: tbl"):
        source._read_partitioned_data_optimized(table, ["a"], None)


def test_urine24_urine_load_failure_raises():
    from easyicu.concept.callbacks import _callback_urine24

    class _Resolver:
        def load_concepts(self, *_args, **_kwargs):
            raise KeyError("urine")

    ctx = SimpleNamespace(resolver=_Resolver(), data_source=None, patient_ids=None, interval=None)
    with pytest.raises(RuntimeError, match="urine24_urine_load_failed"):
        _callback_urine24({}, ctx)


def test_sicdb_cvp_reference_read_failure_raises(monkeypatch):
    from easyicu.concept import sicdb_sources
    from easyicu.table import ICUTable

    cvp = ICUTable(pd.DataFrame({"CaseID": [1], "Offset": [0.0], "cvp_raw": [8.0]}),
                   id_columns=["CaseID"], index_column="Offset", value_column="cvp_raw")

    def boom(*_args, **_kwargs):
        raise OSError("disk read failed")

    monkeypatch.setattr(sicdb_sources, "_table", boom)
    with pytest.raises(RuntimeError, match="sicdb_cvp_arterial_reference_read_failed"):
        sicdb_sources.filter_cvp({"cvp_raw": cvp}, SimpleNamespace(data_source=None, interval=None))


def _ctx(name, resolver, **kwargs):
    from easyicu.concept.callbacks import ConceptCallbackContext

    return ConceptCallbackContext(
        concept_name=name, target=None, interval=pd.Timedelta(hours=1),
        resolver=resolver, data_source=_db("eicu"), patient_ids=None, kwargs=kwargs,
    )


def test_vaso60_weight_load_failure_raises():
    from easyicu.concept.callbacks import _callback_vaso60
    from easyicu.table import ICUTable

    rate = ICUTable(
        pd.DataFrame({"stay_id": [1], "charttime": [0.0], "norepi_rate": [8.0],
                      "rateuom": ["mcg/min"]}),
        id_columns=["stay_id"], index_column="charttime",
        value_column="norepi_rate", unit_column="rateuom",
    )
    dur = ICUTable(
        pd.DataFrame({"stay_id": [1], "charttime": [0.0], "norepi_dur": [pd.Timedelta(hours=2)]}),
        id_columns=["stay_id"], index_column="charttime", value_column="norepi_dur",
    )
    with pytest.raises(RuntimeError, match="vasopressor_weight_load_failed: weight"):
        _callback_vaso60({"norepi_rate": rate, "norepi_dur": dur},
                         _ctx("norepi60", _FailingKeyResolver()))


class _FailingKeyResolver:
    def load_concepts(self, *_args, **_kwargs):
        raise KeyError("weight")


def test_rrt_criteria_dependency_load_failure_raises():
    from easyicu.concept.callbacks import _callback_rrt_criteria

    with pytest.raises(RuntimeError, match="rrt_dependency_load_failed"):
        _callback_rrt_criteria({}, _ctx("rrt_criteria", _FailingKeyResolver()))


def test_raw_concept_load_failure_raises_instead_of_returning_none(monkeypatch):
    from easyicu.concept import ConceptResolver
    from easyicu.resources import load_dictionary

    resolver = ConceptResolver(load_dictionary())

    def fail(*_args, **_kwargs):
        raise OSError("read failed")

    monkeypatch.setattr(resolver, "load_concepts", fail)
    with pytest.raises(RuntimeError, match="raw_concept_load_failed: urine"):
        resolver.get_raw_concept("urine", _db("miiv"), None)
