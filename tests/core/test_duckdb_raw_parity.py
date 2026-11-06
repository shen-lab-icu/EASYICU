"""The DuckDB aggregation path publishes exactly what the pandas path publishes.

Each case loads one concept twice on an invented source: once normally (the
test asserts a DuckDB aggregation route was taken) and once inside
``force_raw_aggregation``.  The two frames must be byte-identical: rows,
values and dtypes.  The rows straddle the ICU episode bound (discharge +
24 h), where flooring the hour before clipping kept rows up to 59 minutes
past the bound, and hourly urine totals above the per-value maximum, which
the DuckDB path used to bound a second time (lead-r13 parity census).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.aggregation_route import (
    force_raw_aggregation,
    record_duckdb_routes,
)
from tests.support.aumc_source import aumc_db
from tests.support.hirid_source import hirid_source_root
from easyicu.concept import ConceptResolver
from easyicu.resources import load_dictionary
from tests.support.urine_raw_source import _source

pytestmark = pytest.mark.filterwarnings("ignore")


def _published(source_factory, concept, **kwargs) -> pd.DataFrame:
    # The merged frame is what extraction publishes (per-concept tables may
    # still carry source-clock helper columns).
    frame = ConceptResolver(load_dictionary(include_sofa2=True)).load_concepts(
        [concept], source_factory(), merge=True, interval=pd.Timedelta(hours=1),
        r_compatible=True, verbose=False, concept_workers=1, **kwargs,
    )
    keys = [c for c in frame.columns if c != concept and not c.startswith("_")]
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _both_paths(source_factory, concept, *, require_route=True, **kwargs):
    with record_duckdb_routes() as routes:
        duck = _published(source_factory, concept, **kwargs)
    with force_raw_aggregation(), record_duckdb_routes() as raw_routes:
        raw = _published(source_factory, concept, **kwargs)
    if require_route:
        assert routes, f"{concept} did not take a DuckDB aggregation route"
    assert not raw_routes
    pd.testing.assert_frame_equal(duck, raw, check_exact=True)
    return duck


def _hours(frame, concept):
    time = next(c for c in frame.columns if c != concept and not c.startswith("_") and "id" not in c.lower())
    return dict(zip(frame[time].astype(float), frame[concept]))


def test_eicu_urine_hour_total_and_post_discharge_rows(tmp_path):
    # Discharge at 100 h, so the episode ends at 124 h.
    offsets = np.array([10.1, 10.2, 10.3, 123.5, 124.25]) * 60
    amounts = [2000.0, 2000.0, 2000.0, 300.0, 400.0]

    def source():
        return _source(
            tmp_path / "eicu_urine", "eicu",
            patient=pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                                  "unitdischargeoffset": [6000], "hospitaladmitoffset": [0],
                                  "admissionweight": [70.0]}),
            intakeoutput=pd.DataFrame({
                "patientunitstayid": 1, "intakeoutputid": range(len(offsets)),
                "intakeoutputoffset": offsets, "intakeoutputentryoffset": offsets,
                "cellpath": "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine",
                "celllabel": "Urine", "cellvaluenumeric": amounts,
            }),
        )

    frame = _both_paths(source, "urine", patient_ids={"patientunitstayid": [1]})
    hours = _hours(frame, "urine")
    # Three in-bound values sum above the 5000 mL per-value maximum.
    assert hours[10.0] == 6000.0
    assert hours[123.0] == 300.0
    assert 124.0 not in hours


def test_eicu_creatinine_either_side_of_the_episode_bound(tmp_path):
    def source():
        return _source(
            tmp_path / "eicu_crea", "eicu",
            patient=pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                                  "unitdischargeoffset": [6000], "hospitaladmitoffset": [0]}),
            lab=pd.DataFrame({
                "labid": [1, 2, 3], "patientunitstayid": 1,
                "labresultoffset": np.array([5.0, 123.5, 124.5]) * 60,
                "labtypeid": 1, "labname": "creatinine", "labresult": [1.0, 1.5, 2.0],
                "labresulttext": ["1.0", "1.5", "2.0"], "labmeasurenamesystem": "mg/dL",
                "labmeasurenameinterface": "mg/dL",
                "labresultrevisedoffset": np.array([5.0, 123.5, 124.5]) * 60,
            }),
        )

    hours = _hours(_both_paths(source, "crea", patient_ids={"patientunitstayid": [1]}), "crea")
    assert hours == {5.0: 1.0, 123.0: 1.5}


def _miiv_stays(origin):
    return pd.DataFrame({"stay_id": [1], "subject_id": [1], "hadm_id": [1], "intime": [origin],
                         "outtime": [origin + pd.Timedelta(hours=96)], "los": [4.0]})


def test_miiv_integer_urine_publishes_float64_on_both_paths(tmp_path):
    origin = pd.Timestamp("2180-01-01 00:17")
    stamps = origin + pd.to_timedelta([1.25, 1.5, 119.5, 120.5], unit="h")

    def source():
        return _source(
            tmp_path / "miiv_urine", "miiv", icustays=_miiv_stays(origin),
            outputevents=pd.DataFrame({"stay_id": 1, "subject_id": 1, "hadm_id": 1, "charttime": stamps,
                                       "itemid": 226559, "value": np.array([100, 250, 40, 60], dtype="int64"),
                                       "valueuom": "mL"}),
        )

    frame = _both_paths(source, "urine", patient_ids={"stay_id": [1]})
    assert frame["urine"].dtype == np.float64
    assert _hours(frame, "urine") == {1.0: 350.0, 119.0: 40.0}


def test_miiv_creatinine_lab_and_chart_channels(tmp_path):
    origin = pd.Timestamp("2180-01-01 00:00")
    lab_times = origin + pd.to_timedelta([2.0, 119.5, 120.5], unit="h")

    def source():
        return _source(
            tmp_path / "miiv_crea", "miiv", icustays=_miiv_stays(origin),
            admissions=pd.DataFrame({"subject_id": [1], "hadm_id": [1],
                                     "admittime": [origin - pd.Timedelta(hours=5)],
                                     "dischtime": [origin + pd.Timedelta(days=9)]}),
            labevents=pd.DataFrame({"labevent_id": [1, 2, 3], "subject_id": 1, "hadm_id": 1,
                                    "specimen_id": [1, 2, 3], "itemid": 50912, "charttime": lab_times,
                                    "valuenum": [1.0, 1.4, 1.8], "value": ["1.0", "1.4", "1.8"],
                                    "valueuom": "mg/dL"}),
            chartevents=pd.DataFrame({"stay_id": 1, "subject_id": 1, "hadm_id": 1,
                                      "charttime": origin + pd.to_timedelta([3.0, 120.75], unit="h"),
                                      "itemid": 220615, "valuenum": [1.1, 2.2], "value": ["1.1", "2.2"],
                                      "valueuom": "mg/dL"}),
        )

    hours = _hours(_both_paths(source, "crea", patient_ids={"stay_id": [1]}), "crea")
    assert 120.0 not in hours and hours[119.0] == 1.4


def test_mimic3_creatinine_rolling_join_branch(tmp_path):
    origin = pd.Timestamp("2150-01-01 00:00")

    def source():
        return _source(
            tmp_path / "mimic_crea", "mimic",
            icustays=pd.DataFrame({"icustay_id": [1], "subject_id": [1], "hadm_id": [1], "intime": [origin],
                                   "outtime": [origin + pd.Timedelta(hours=96)], "los": [4.0]}),
            labevents=pd.DataFrame({"row_id": [1, 2, 3], "subject_id": 1, "hadm_id": 1, "itemid": 50912,
                                    "charttime": origin + pd.to_timedelta([2.0, 119.5, 120.5], unit="h"),
                                    "valuenum": [1.0, 1.4, 1.8], "value": ["1.0", "1.4", "1.8"],
                                    "valueuom": "mg/dL"}),
        )

    hours = _hours(_both_paths(source, "crea", patient_ids={"icustay_id": [1]}), "crea")
    assert 120.0 not in hours and hours[119.0] == 1.4


def test_sic_urine_episode_bound(tmp_path):
    # ICUOffset 0 and TimeOfStay 100 h: the episode ends at 124 h.
    def source():
        return _source(
            tmp_path / "sic_urine", "sic",
            cases=pd.DataFrame({"CaseID": [1], "PatientID": [1], "ICUOffset": [0.0],
                                "TimeOfStay": [360000.0], "WeightOnAdmission": [70000.0]}),
            data_float_h=pd.DataFrame({"CaseID": 1, "DataID": 725,
                                       "Offset": np.array([3.0, 123.5, 124.5]) * 3600,
                                       "Val": [100.0, 50.0, 70.0]}),
        )

    hours = _hours(_both_paths(source, "urine", patient_ids={"CaseID": [1]}), "urine")
    assert 124.0 not in hours


def test_aumc_creatinine_episode_bound(tmp_path):
    # aumc_db discharges 48 h after admission: the episode ends at 72 h.
    def source():
        return aumc_db(tmp_path / "aumc", numeric=[
            (2 * 60, 6836, 90.0, "µmol/l"), (71.5 * 60, 6836, 100.0, "µmol/l"),
            (72.5 * 60, 6836, 110.0, "µmol/l"),
        ])

    hours = _hours(_both_paths(source, "crea", patient_ids={"admissionid": [1]}), "crea")
    assert 72.0 not in hours and 71.0 in hours


def test_hirid_creatinine_sanity_bound(tmp_path):
    # HiRID has no discharge clock; the 366-day sanity bound applies.
    root = hirid_source_root(tmp_path, "observations", [
        {"patientid": 1, "variableid": 20000600, "value": 90.0, "datetime": 2.0},
        {"patientid": 1, "variableid": 20000600, "value": 100.0, "datetime": 8783.5},
        {"patientid": 1, "variableid": 20000600, "value": 110.0, "datetime": 8784.5},
    ])

    def source():
        from easyicu.config import load_src_cfg
        from easyicu.datasource import ICUDataSource

        return ICUDataSource(load_src_cfg("hirid"), base_path=root, enable_cache=False)

    hours = _hours(_both_paths(source, "crea", patient_ids={"patientid": [1]}), "crea")
    assert 8784.0 not in hours and 8783.0 in hours


def _eicu_patient():
    return pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                         "unitdischargeoffset": [6000], "hospitaladmitoffset": [0]})


def test_eicu_two_sources_pool_into_one_median(tmp_path):
    # Monitor 30/31/32 and a charted 50 in one hour: the pooled median is 31.5
    # (ricu); a median of the two per-source medians would be 40.5.
    def source():
        return _source(
            tmp_path / "eicu_etco2", "eicu", patient=_eicu_patient(),
            respiratorycharting=pd.DataFrame({
                "respchartid": [1], "patientunitstayid": [1], "respchartoffset": [130.0],
                "respchartentryoffset": [130.0], "respcharttypecat": ["Vent Details"],
                "respchartvaluelabel": ["EtCO2"], "respchartvalue": ["50"],
            }),
            vitalperiodic=pd.DataFrame({
                "vitalperiodicid": [1, 2, 3], "patientunitstayid": 1,
                "observationoffset": [125.0, 135.0, 145.0], "etco2": [30.0, 31.0, 32.0],
            }),
        )

    hours = _hours(_both_paths(source, "etco2", patient_ids={"patientunitstayid": [1]},
                               require_route=False), "etco2")
    assert hours == {2.0: 31.5}


def test_eicu_wide_batch_publishes_only_hours_with_a_value(tmp_path):
    def source():
        return _source(
            tmp_path / "eicu_co", "eicu", patient=_eicu_patient(),
            vitalaperiodic=pd.DataFrame({
                "vitalaperiodicid": [1, 2, 3], "patientunitstayid": 1,
                "observationoffset": [65.0, 185.0, 305.0],
                "noninvasivesystolic": [120.0, 118.0, 121.0],
                "cardiacoutput": [np.nan, 5.5, np.nan],
            }),
        )

    hours = _hours(_both_paths(source, "co", patient_ids={"patientunitstayid": [1]}), "co")
    assert hours == {3.0: 5.5}
