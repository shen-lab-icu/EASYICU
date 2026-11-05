"""CVP quality uses invasive DBP, with invented source rows only."""

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.callbacks import ConceptCallbackContext, _callback_cvp_artefact_filter
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable


DATABASES = ["miiv", "mimic", "mimic_demo", "eicu", "eicu_demo", "aumc", "hirid", "sic"]
CVP = [40., 40., 40., 40., 40., 20., 20., -5., 50.]
INVASIVE = {0: 42., 2: 80., 3: 42., 7: 80., 8: 80., 11: 80.}
CUFF = {1: 41., 2: 41., 3: 80., 5: 22.}


def _source(root, database, *, invasive_only=False, arterial_channel="invasive"):
    cfg = load_src_cfg(database)
    dictionary = load_dictionary(source_name=database)
    identifier = cfg.id_configs["icustay"].id
    start = pd.Timestamp("2100-01-01 00:30")
    if database in {"miiv", "mimic", "mimic_demo"}:
        anchor = {identifier: [1, 2], "subject_id": [1, 2], "hadm_id": [11, 12],
                  "intime": start, "outtime": start + pd.Timedelta(days=1), "los": 1.}
    elif database in {"eicu", "eicu_demo"}:
        anchor = {identifier: [1, 2], "patienthealthsystemstayid": [11, 12],
                  "uniquepid": ["invented-one", "invented-two"],
                  "hospitaladmitoffset": -30, "unitdischargeoffset": 1440}
    elif database == "aumc":
        anchor = {identifier: [1, 2], "patientid": [11, 12],
                  "admittedat": 1800000, "dischargedat": 88200000}
    elif database == "hirid":
        anchor = {identifier: [1, 2], "admissiontime": str(start), "age": 60, "sex": "M"}
    else:
        anchor = {identifier: [1, 2], "PatientID": [11, 12], "ICUOffset": 1800,
                  "TimeOfStay": 88200}

    frames = defaultdict(list)
    frames[cfg.id_configs["icustay"].table].append(pd.DataFrame(anchor))
    sources = dictionary["dbp"].for_data_source(cfg)
    arterial = next(s for s in sources if s.params["source_channel"] == arterial_channel)
    cuff = next(s for s in sources if s.params["source_channel"] == "nibp")
    inputs = [(dictionary["cvp_raw"].for_data_source(cfg)[0], dict(enumerate(CVP)), "cvp_raw"),
              (arterial, INVASIVE, "dbp")]
    if not invasive_only:
        inputs.append((cuff, CUFF, "dbp"))
    for source, observations, name in inputs:
        defaults = cfg.get_table(source.table).defaults
        time = source.index_var or defaults.index_var
        value = source.value_var or defaults.val_var
        hours = np.array(list(observations), dtype=float)
        frame = pd.DataFrame({identifier: 1, value: list(observations.values())})
        if name == "cvp_raw":
            # A second stay must not borrow the first stay's cuff evidence.
            frame = pd.concat([frame, pd.DataFrame({identifier: [2], value: [40.]})], ignore_index=True)
            hours = np.append(hours, 1.)
        if source.ids:
            frame[source.sub_var] = source.ids[0]
        if database in {"miiv", "mimic", "mimic_demo"}:
            frame[time] = start + pd.to_timedelta(hours, unit="h")
            frame = frame.assign(subject_id=frame[identifier], hadm_id=frame[identifier] + 10,
                                 valueuom="mmHg", error=0)
        elif database in {"eicu", "eicu_demo"}:
            frame[time] = hours * 60
        elif database == "aumc":
            frame[time] = hours * 3600000 + 1800000
            frame = frame.assign(unit="mmHg", tag="+")
        elif database == "hirid":
            frame[time] = start + pd.to_timedelta(hours, unit="h")
            frame = frame.assign(entertime=frame[time], status=8, type="N", stringvalue=None)
        else:
            frame[time] = hours * 3600 + 1800
        frames[source.table].append(frame)
    for table, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True)
        if database in {"eicu", "eicu_demo"} and table == "vitalperiodic":
            frame = frame.groupby([identifier, "observationoffset"], as_index=False).first()
        declared = cfg.get_table(table).columns
        frame = frame[[c for c in frame if c.lower() in {d.lower() for d in declared}]]
        for column in declared:
            if column.lower() not in {c.lower() for c in frame}:
                frame[column] = np.nan
        filename = "general_table" if database == "hirid" and table == "general" else table
        frame.to_parquet(root / f"{filename}.parquet", index=False)
    source = ICUDataSource(cfg, base_path=root, enable_cache=False)
    return source, dictionary, identifier


def _load(source, dictionary, identifier, names):
    return ConceptResolver(dictionary).load_concepts(
        names, source, merge=False, r_compatible=False, concept_workers=1,
        interval=pd.Timedelta(hours=1), patient_ids={identifier: [1, 2]}, verbose=False,
    )


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("mode", ["single", "batch"])
def test_cvp_quality_distinguishes_invasive_cuff_both_and_missing(tmp_path, database, mode):
    source, dictionary, identifier = _source(tmp_path, database)
    names = ["cvp", "cvp_raw", "cvp_qc", "dbp"]
    before = _load(source, dictionary, identifier, ["dbp"])["dbp"].data
    if mode == "single":
        result = {name: _load(source, dictionary, identifier, [name])[name] for name in names}
    else:
        result = _load(source, dictionary, identifier, names)
    main, qc = result["cvp"], result["cvp_qc"]
    pd.testing.assert_frame_equal(result["dbp"].data, before)
    pd.testing.assert_frame_equal(main.data.rename(columns={"cvp": "cvp_raw"}), result["cvp_raw"].data)
    assert list(qc.data.columns) == [identifier, qc.index_column, "cvp_qc"]
    pd.testing.assert_frame_equal(main.data[[identifier, main.index_column]], qc.data[[identifier, qc.index_column]])
    assert main.data.loc[main.data[identifier].eq(1), "cvp"].tolist() == CVP
    flags = qc.data.loc[qc.data[identifier].eq(1)].set_index(qc.index_column).cvp_qc.to_dict()
    assert flags == {
        0.: "high_30_to_50|arterial_diastolic_match",
        1.: "high_30_to_50|cuff_dbp_only",
        2.: "high_30_to_50",
        3.: "high_30_to_50|arterial_diastolic_match",
        4.: "high_30_to_50|arterial_evidence_unavailable",
        5.: "kept",
        6.: "kept",
        7.: "kept",
        8.: "high_30_to_50",
    }
    assert qc.data.loc[qc.data[identifier].eq(2), "cvp_qc"].tolist() == [
        "high_30_to_50|arterial_evidence_unavailable",
    ]


@pytest.mark.parametrize("database", DATABASES)
def test_without_cuff_sources_quality_and_values_are_identical(tmp_path, database):
    source, dictionary, identifier = _source(tmp_path, database, invasive_only=True)
    names = ["cvp", "cvp_raw", "cvp_qc", "dbp"]
    with_fallback = _load(source, dictionary, identifier, names)
    definition = dictionary["dbp"]
    sources = [s for s in definition.for_data_source(source.config)
               if s.params["source_channel"].startswith("invasive")]
    narrowed = dictionary.copy()
    narrowed.update(ConceptDictionary({"dbp": replace(definition, sources={database: sources})}))
    without_fallback = _load(source, narrowed, identifier, names)
    for name in names:
        pd.testing.assert_frame_equal(with_fallback[name].data, without_fallback[name].data)
    flags = without_fallback["cvp_qc"].data.cvp_qc.tolist()
    assert flags[0] == "high_30_to_50|arterial_diastolic_match"
    assert "cuff_dbp_only" not in "|".join(flags)


def test_secondary_invasive_channel_is_arterial_evidence(tmp_path):
    source, dictionary, identifier = _source(tmp_path, "aumc", arterial_channel="invasive_secondary")
    result = _load(source, dictionary, identifier, ["cvp", "cvp_qc"])
    qc = result["cvp_qc"]
    flags = qc.data.loc[qc.data[identifier].eq(1)].set_index(qc.index_column).cvp_qc
    assert flags.loc[0.] == "high_30_to_50|arterial_diastolic_match"
    assert flags.loc[2.] == "high_30_to_50"
    assert flags.loc[1.] == "high_30_to_50|cuff_dbp_only"


@pytest.mark.parametrize("database", ["miiv", "mimic"])
def test_cuff_only_source_definition_never_supplies_arterial_evidence(tmp_path, database):
    source, dictionary, identifier = _source(tmp_path, database)
    definition = dictionary["dbp"]
    cuff = [s for s in definition.for_data_source(source.config) if s.params["source_channel"] == "nibp"]
    narrowed = dictionary.copy()
    narrowed.update(ConceptDictionary({"dbp": replace(definition, sources={database: cuff})}))
    result = _load(source, narrowed, identifier, ["cvp", "cvp_qc"])
    qc = result["cvp_qc"]
    flags = qc.data.loc[qc.data[identifier].eq(1)].set_index(qc.index_column).cvp_qc
    assert flags.loc[0.] == "high_30_to_50|arterial_evidence_unavailable"
    assert flags.loc[3.] == "high_30_to_50|cuff_dbp_only"
    assert "arterial_diastolic_match" not in "|".join(flags)
    assert result["cvp"].data.loc[lambda d: d[identifier].eq(1), "cvp"].tolist() == CVP


def test_unlabelled_dbp_is_not_arterial_evidence_for_direct_callback():
    tables = {
        name: ICUTable(pd.DataFrame({"stay_id": [1], "hour": [0.], name: [value]}),
                       id_columns=["stay_id"], index_column="hour", value_column=name)
        for name, value in [("cvp_raw", 40.), ("dbp", 42.)]
    }
    ctx = ConceptCallbackContext(
        concept_name="cvp_qc", target=None, interval=pd.Timedelta(hours=1), resolver=None,
        data_source=SimpleNamespace(config=SimpleNamespace(name="miiv")), patient_ids=[1],
    )
    result = _callback_cvp_artefact_filter(tables, ctx)
    assert result.data.cvp_qc.tolist() == ["high_30_to_50|arterial_evidence_unavailable"]


def test_cvp_values_do_not_require_an_additional_arterial_source_load(monkeypatch):
    tables = {
        name: ICUTable(pd.DataFrame({"stay_id": [1], "hour": [0.], name: [value]}),
                       id_columns=["stay_id"], index_column="hour", value_column=name)
        for name, value in [("cvp_raw", 40.), ("dbp", 42.)]
    }
    ctx = ConceptCallbackContext(
        concept_name="cvp", target=None, interval=pd.Timedelta(hours=1),
        resolver=SimpleNamespace(dictionary=load_dictionary(source_name="miiv")),
        data_source=SimpleNamespace(config=load_src_cfg("miiv")), patient_ids=[1],
    )

    def reject_additional_load(*args, **kwargs):
        pytest.fail("CVP retention must not depend on the extra QC source read")

    monkeypatch.setattr(ConceptResolver, "_load_single_concept", reject_additional_load)
    result = _callback_cvp_artefact_filter(tables, ctx)
    assert result.data.to_dict("list") == {"stay_id": [1], "hour": [0.], "cvp": [40.]}
