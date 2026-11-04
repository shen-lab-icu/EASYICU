"""Small invented eICU source tables for source-contract regression tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)

def _eicu_source(tmp_path: Path, database: str, **tables) -> ICUDataSource:
    _write(pd.DataFrame({
        "patientunitstayid": [1], "patienthealthsystemstayid": [11], "uniquepid": ["p1"],
        "age": ["60"], "gender": ["Male"], "hospitalid": [1],
        "unitvisitnumber": [1],
        "hospitaladmitoffset": [0], "hospitaldischargeoffset": [72 * 60],
        "unitdischargeoffset": [48 * 60], "unitdischargestatus": ["Alive"],
        "hospitaldischargestatus": ["Alive"], "admissionweight": [80.0],
    }), tmp_path / "patient.parquet")
    for name, frame in tables.items():
        frame = frame.copy()
        frame["patientunitstayid"] = 1
        directory = name in {"nursecharting", "vitalperiodic"}
        _write(frame, tmp_path / name / "part.parquet" if directory else tmp_path / f"{name}.parquet")
    return ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)

def _hourly(source, concept, aggregate="median") -> dict:
    loaded = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        [concept], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": [1]},
        aggregate=aggregate, concept_workers=1, verbose=False,
    )[concept]
    frame = loaded.data
    if frame.empty:
        return {}
    time = frame[loaded.index_column]
    if pd.api.types.is_timedelta64_dtype(time):
        time = time / pd.Timedelta(hours=1)
    return {
        float(hour): (value if isinstance(value, (bool, np.bool_)) else round(float(value), 3))
        for hour, value in zip(time, frame[concept]) if pd.notna(value)
    }
