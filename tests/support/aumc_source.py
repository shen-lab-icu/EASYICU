"""Tiny AmsterdamUMCdb parquet sources of invented rows for extraction tests."""

from pathlib import Path

import pandas as pd

import easyicu
from easyicu.concept import ConceptDictionary, ConceptResolver
from tests.support.blood_gas_sources import _source, _write

DATA = Path(easyicu.__file__).parent / "data"
MINUTE = 60_000


def aumc_db(tmp_path, *, origin=17.25, numeric=(), drugs=(), processes=(), lists=(), texts=(), layout="file"):
    _write(pd.DataFrame({
        "admissionid": [1], "patientid": [11], "admittedat": [origin * MINUTE],
        "dischargedat": [(origin + 48 * 60) * MINUTE], "weightgroup": ["80-89"],
    }), tmp_path / "admissions.parquet")
    for table, rows, cols in (
        ("numericitems", numeric, ["measuredat", "itemid", "value", "unit"]),
        ("drugitems", drugs, ["start", "stop", "itemid", "dose", "doseunit", "doserateunit", "iscontinuous", "action", "ordercategory"]),
        ("processitems", processes, ["start", "stop", "itemid", "item"]),
        ("listitems", lists, ["measuredat", "itemid", "value"]),
        ("freetextitems", texts, ["measuredat", "itemid", "value"]),
    ):
        frame = pd.DataFrame(rows, columns=cols)
        frame["admissionid"] = 1
        for col in ("start", "stop", "measuredat"):
            if col in frame:
                frame[col] = (frame[col] + origin) * MINUTE
        if "measuredat" in frame:
            frame["registeredat"] = frame["measuredat"]
        if table == "numericitems":
            frame["tag"] = None
        if table == "drugitems":
            frame["doserateperkg"] = True
            frame["orderid"] = range(len(frame))
            frame["duration"] = (frame["stop"] - frame["start"]) / MINUTE
        path = tmp_path / f"{table}.parquet"
        if layout == "bucket":
            path = tmp_path / f"{table}_bucket" / "bucket_id=0" / "part.parquet"
        _write(frame, path)
    return _source(tmp_path, "aumc")


def aumc_load(db, name, *, sofa=False, interval=pd.Timedelta(hours=1)):
    dictionary = ConceptDictionary.from_json(DATA / "concept-dict.json")
    if sofa:
        dictionary.update(ConceptDictionary.from_json(DATA / "sofa2-dict.json"))
    result = ConceptResolver(dictionary).load_concepts(
        [name], db, merge=False, r_compatible=False, interval=interval,
        patient_ids={"admissionid": [1]}, concept_workers=1, verbose=False,
    )[name]
    frame = result.data
    return frame.dropna(subset=[name]).set_index(result.index_column)[name].to_dict()


def aumc_drug(start, stop, item, dose=1.0, unit="mcg", rate="min", continuous=True, action="Toedienen", category="IV"):
    return (start, stop, item, dose, unit, rate, continuous, action, category)
