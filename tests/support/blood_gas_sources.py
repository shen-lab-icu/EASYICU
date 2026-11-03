"""Synthetic raw blood-gas sources with recorded specimen types.

Each builder writes one ICU stay whose gases differ only in specimen type, so
a loader that ignores the specimen record returns extra (venous) values.
Times are minutes after ICU admission; one gas per hour keeps the expected
hourly values unambiguous.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"
ADMISSION = pd.Timestamp("2180-01-01 00:00:00")


def _source(tmp_path: Path, database: str) -> ICUDataSource:
    return ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _mimic_family(tmp_path: Path, database: str, lab_rows, chart_rows, specimen_column):
    id_column = "stay_id" if database == "miiv" else "icustay_id"
    lab = pd.DataFrame(lab_rows, columns=["minute", "itemid", "value", specimen_column])
    lab["charttime"] = ADMISSION + pd.to_timedelta(lab.pop("minute"), unit="m")
    lab["storetime"] = lab["charttime"]
    lab["valuenum"] = pd.to_numeric(lab["value"], errors="coerce")
    lab["valueuom"] = None
    lab["subject_id"] = 101
    lab["hadm_id"] = 201
    if specimen_column != "specimen_id":
        lab = lab.drop(columns=specimen_column)
    _write(lab, tmp_path / "labevents" / "part.parquet")
    chart = pd.DataFrame(chart_rows, columns=["minute", "itemid", "valuenum"])
    chart["charttime"] = ADMISSION + pd.to_timedelta(chart.pop("minute"), unit="m")
    chart["storetime"] = chart["charttime"]
    chart["value"] = chart["valuenum"].astype(str)
    chart["valueuom"] = None
    chart[id_column] = 1
    chart["subject_id"] = 101
    chart["hadm_id"] = 201
    _write(chart, tmp_path / "chartevents" / "part.parquet")
    _write(
        pd.DataFrame({
            id_column: [1], "subject_id": [101], "hadm_id": [201],
            "intime": [ADMISSION], "outtime": [ADMISSION + pd.Timedelta(hours=48)],
        }),
        tmp_path / "icustays.parquet",
    )
    return _source(tmp_path, database), id_column


def miiv_source(tmp_path: Path):
    """Specimen type is lab item 52033 sharing the gas's ``specimen_id``."""

    lab = [
        (5, 50821, "90", 1), (5, 52033, "ART.", 1), (5, 50820, "7.45", 1),
        (65, 50821, "35", 2), (65, 52033, "VEN.", 2), (65, 50820, "7.30", 2),
        (125, 50821, "40", 3),  # no specimen record
        (185, 50821, "95", 4), (185, 52033, "ART.", 4), (185, 52033, "VEN.", 4),
        (245, 50821, "38", 5), (245, 52033, "CENTRAL VENOUS.", 5),
    ]
    chart = [(70, 220274, 7.31), (130, 223830, 7.40)]
    return _mimic_family(tmp_path, "miiv", lab, chart, "specimen_id")


def mimic_source(tmp_path: Path, database: str = "mimic"):
    """Specimen type is lab item 50800 at the same subject and charttime."""

    lab = [
        (5, 50821, "90", None), (5, 50800, "ART", None), (5, 50820, "7.45", None),
        (65, 50821, "37", None), (65, 50800, "MIX", None), (65, 50820, "7.33", None),
        (125, 50821, "80", None),  # no specimen record
        (185, 50821, "95", None), (185, 50800, "ART", None), (185, 50800, "VEN", None),
    ]
    chart = [
        (245, 490, 300.), (305, 779, 110.),  # 490 records alveolar PAO2
        (70, 860, 7.31), (130, 780, 7.40), (190, 4202, 7.35),
    ]
    return _mimic_family(tmp_path, database, lab, chart, "unused")


def aumc_source(tmp_path: Path):
    """Specimen type is freetext 11646 at the same admission and measuredat."""

    minute = 60_000
    numeric = pd.DataFrame(
        [
            (5, 7433, 90., "mmHg"), (65, 7433, 35., "mmHg"),
            (125, 21214, 12., "kPa"), (185, 7433, 80., "mmHg"),
            (245, 9996, 95., "mmHg"),
        ],
        columns=["minute", "itemid", "value", "unit"],
    )
    numeric["measuredat"] = numeric.pop("minute") * minute
    numeric["admissionid"] = 1
    numeric["tag"] = None
    numeric["registeredat"] = numeric["measuredat"]
    _write(numeric, tmp_path / "numericitems" / "part.parquet")
    freetext = pd.DataFrame(
        [(5, "ART."), (65, "Ven."), (185, "Vervalt"), (245, "Art.")],
        columns=["minute", "value"],
    )
    freetext["measuredat"] = freetext.pop("minute") * minute
    freetext["admissionid"] = 1
    freetext["itemid"] = 11646
    freetext["item"] = "Afname (bloed)"
    _write(freetext, tmp_path / "freetextitems.parquet")
    _write(
        pd.DataFrame({
            "admissionid": [1], "patientid": [11], "admittedat": [0],
            "dischargedat": [48 * 60 * minute], "admissioncount": [1],
        }),
        tmp_path / "admissions.parquet",
    )
    return _source(tmp_path, "aumc"), "admissionid"


def sic_source(tmp_path: Path):
    """Specimen type is the laboratory row's own LaboratoryType (2296)."""

    lab = pd.DataFrame(
        [
            # Offsets stay above the loader's 5000 s seconds-vs-hours cut.
            (1, 125, 689, 90., 2296), (2, 95, 689, 35., None),
            (3, 185, 664, 70., 2296), (4, 185, 688, 7.40, 2296),
            (5, 245, 688, 7.30, None),
        ],
        columns=["id", "minute", "LaboratoryID", "LaboratoryValue", "LaboratoryType"],
    )
    lab["Offset"] = lab.pop("minute") * 60
    lab["CaseID"] = 1
    lab["LaboratoryType"] = lab["LaboratoryType"].astype("Int64")
    _write(lab, tmp_path / "laboratory" / "part.parquet")
    _write(
        pd.DataFrame({
            "CaseID": [1], "PatientID": [11], "ICUOffset": [0],
            "TimeOfStay": [48 * 3600], "OffsetOfDeath": pd.array([None], dtype="Int64"),
        }),
        tmp_path / "cases.parquet",
    )
    return _source(tmp_path, "sic"), "CaseID"


def hourly(source, id_column: str, concept: str, *, parquet_scan: bool = True) -> dict:
    """Load one concept from the shipped dictionary as {hour: value}.

    ``parquet_scan=False`` hides the parquet directories so the loader takes
    its pandas table path; specimen evidence must survive both readers.
    """

    if not parquet_scan:
        source.resolve_bucket_directory = lambda _table: None
        source.resolve_flat_parquet_directory = lambda _table: None
    loaded = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        [concept], source, merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={id_column: [1]},
        aggregate="min", concept_workers=1, verbose=False,
    )[concept]
    frame = loaded.data
    time_column = loaded.index_column
    return {
        float(hour): round(float(value), 3)
        for hour, value in zip(frame[time_column], frame[concept])
        if pd.notna(value)
    }
