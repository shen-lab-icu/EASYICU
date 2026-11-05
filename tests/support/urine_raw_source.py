"""Raw-schema MIMIC/MIIV/eICU/SICdb sources for urine-window tests (synthetic rows only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.concept import ConceptResolver
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.resources import load_dictionary


def _source(root, database, *, layout="flat", **frames):
    """Real loader, projection and selectors; only on-disk raw rows are synthetic."""
    import easyicu

    schema = next(c for c in json.loads(
        (Path(easyicu.__file__).parent / "data/data-sources.json").read_text()
    ) if c["name"] == database)
    root.mkdir(parents=True, exist_ok=True)
    for table, frame in frames.items():
        # SIC's source config declares canonical names in each column's name.
        declared = schema["tables"][table]["cols"]
        allowed = {c.lower() for c in declared}
        allowed.update(c.get("name", key).lower() for key, c in declared.items())
        assert {c.lower() for c in frame} <= allowed, (database, table, set(frame) - allowed)
        path = root / f"{table}.parquet"
        if layout == "bucket" and table not in {"patient", "icustays", "general", "cases"}:
            path = root / f"{table}_bucket/bucket_id=0/part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return ICUDataSource(load_src_cfg(database), base_path=root, enable_cache=False)


def _load(source, names, **kwargs):
    return ConceptResolver(load_dictionary(include_sofa2=True)).load_concepts(
        names, source, merge=False, interval=pd.Timedelta(hours=1),
        r_compatible=False, verbose=False, concept_workers=1, **kwargs,
    )


def _stays(database, origin):
    id_col = "stay_id" if database == "miiv" else "icustay_id"
    return pd.DataFrame({id_col: [1], "subject_id": [1], "hadm_id": [1],
                         "intime": [origin], "outtime": [origin + pd.Timedelta(days=4)]})


def _urine_source(root, database, times, amounts, *, layout="flat", weight=70., intime=None):
    origin = pd.Timestamp("2180-01-01 00:17")
    if database in {"miiv", "mimic"}:
        key = "stay_id" if database == "miiv" else "icustay_id"
        stamps = origin + pd.to_timedelta(times, unit="h")
        return _source(root, database, layout=layout, icustays=_stays(database, origin if intime is None else intime),
                       outputevents=pd.DataFrame({key: 1, "subject_id": 1, "hadm_id": 1,
                           "charttime": stamps, "itemid": 226559 if database == "miiv" else 40055,
                           "value": amounts, "valueuom": "mL"}),
                       chartevents=pd.DataFrame({key: [1], "subject_id": [1], "hadm_id": [1],
                           "charttime": [origin], "itemid": [226512], "valuenum": [weight],
                           "value": [str(weight)], "valueuom": ["kg"]}))
    if database == "eicu":
        return _source(root, database, layout=layout,
                       patient=pd.DataFrame({"patientunitstayid": [1], "patienthealthsystemstayid": [1],
                           "unitdischargeoffset": [6000], "hospitaladmitoffset": [0],
                           "admissionweight": [weight]}),
                       intakeoutput=pd.DataFrame({"patientunitstayid": 1,
                           "intakeoutputid": range(len(times)), "intakeoutputoffset": np.asarray(times) * 60,
                           "intakeoutputentryoffset": np.asarray(times) * 60,
                           "cellpath": "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine",
                           "celllabel": "Urine", "cellvaluenumeric": amounts}))
    return _source(root, database, layout=layout,
                   cases=pd.DataFrame({"CaseID": [1], "PatientID": [1], "ICUOffset": [0.],
                       "TimeOfStay": [400000.], "WeightOnAdmission": [weight * 1000]}),
                   data_float_h=pd.DataFrame({"CaseID": 1, "DataID": 725,
                       "Offset": np.asarray(times) * 3600, "Val": amounts}))
