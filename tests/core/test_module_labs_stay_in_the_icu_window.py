"""Labs loaded together keep the ICU window a single lab load keeps.

Audit round 2 (R2-05): MIMIC-III labevents are stay-mapped through the
hospital admission.  Loading one lab clipped it to [-24 h, LOS + 24 h], but a
module loads several labevents concepts through one batched DuckDB query that
skipped the window: chemistry and haematology kept labs from -2,487 h to well
past discharge and fed them to the creatinine baseline and KDIGO.
Invented MIMIC-III CSVs only.
"""

from __future__ import annotations

import csv

import pytest

from easyicu.base import BaseICULoader
from easyicu.config import load_src_cfg
from easyicu.datasource import _close_duckdb_connections
from easyicu.io.data_converter import DataConverter

# ICU stay 2100-01-01 01:00 .. 2100-01-03 01:00 (48 h): the window is
# [-24 h, 72 h].  The hospital admission spans 2099-12-25 .. 2100-01-10.
# itemid: a plausible value
_LABS = {51006: 20, 50983: 140, 50971: 4}
_TIMES = {
    "2099-12-27 00:00:00": "outside",  # -97 h
    "2100-01-01 02:00:00": "inside",  # 1 h
    "2100-01-08 00:00:00": "outside",  # 167 h
}


@pytest.fixture(params=["mimic", "mimic_demo"])
def loader(tmp_path, request, monkeypatch):
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    cfg = load_src_cfg(request.param)
    stay = {"subject_id": 1, "hadm_id": 11, "icustay_id": 101}
    rows = {
        "patients": [{"subject_id": 1, "gender": "F", "dob": "2060-01-01 00:00:00"}],
        "admissions": [{"subject_id": 1, "hadm_id": 11, "admittime": "2099-12-25 00:00:00",
                        "dischtime": "2100-01-10 00:00:00", "hospital_expire_flag": 0}],
        "icustays": [{**stay, "intime": "2100-01-01 01:00:00",
                      "outtime": "2100-01-03 01:00:00", "los": 2}],
        "labevents": [
            {"subject_id": 1, "hadm_id": 11, "charttime": when, "itemid": item,
             "value": str(value), "valuenum": value, "valueuom": ""}
            for item, value in _LABS.items() for when in _TIMES
        ],
    }
    raw = tmp_path / "mimic-iii-clinical-database-demo-1.4"
    raw.mkdir()
    for name, table in cfg.tables.items():
        columns = list(table.columns)
        with (raw / f"{name.upper()}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writerow({column: column.upper() for column in columns})
            writer.writerows(rows.get(name, []))
    conversion = DataConverter(tmp_path, database=request.param, parallel_workers=1,
                               verbose=False).convert_all()
    assert all(row["status"] == "completed" for row in conversion.values())
    instance = BaseICULoader(data_path=tmp_path, database=request.param)
    yield instance
    instance.clear_cache()
    _close_duckdb_connections()


def test_labs_loaded_as_a_module_keep_only_the_icu_window(loader):
    result = loader.load_concepts(
        ["bun", "na", "k"], merge=False, concept_workers=1, parallel_workers=1,
    )

    for concept in ("bun", "na", "k"):
        frame = getattr(result[concept], "data", result[concept])
        assert frame["charttime"].tolist() == [1.0], concept

