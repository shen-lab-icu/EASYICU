from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from easyicu import load_concepts
from easyicu.base import BaseICULoader
from easyicu.databases.detection import detect_database_identity
from easyicu.io.community_databases import (
    ensure_community_stays,
    prepare_community_archives,
)
from easyicu.io.data_converter import ConversionStatus, DataConverter
from easyicu.resources import load_dictionary


def test_community_concept_overlay_is_loaded_by_default() -> None:
    dictionary = load_dictionary()
    assert {"miiv", "mimic", "eicu", "aumc", "hirid", "sic"}.issubset(
        dictionary["hr"].sources
    )
    assert dictionary["hr"].sources["nwicu"][0].ids == [320045]
    assert dictionary["crea"].sources["zigong"][0].table == "dtlab"
    assert dictionary["death"].sources["jinhua"][0].value_var == "expire_flag"


@pytest.mark.parametrize(
    ("database", "relative_marker"),
    [
        ("nwicu", "data/nw_icu/icustays.csv.gz"),
        ("zhejiang_eicu", "OMIX005817-01.zip"),
        ("jinhua", "OMIX007493-02.zip"),
        ("zigong", "DataTables/dtBaseline.csv"),
    ],
)
def test_raw_community_layout_detection(
    tmp_path: Path, database: str, relative_marker: str
) -> None:
    marker = tmp_path / relative_marker
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    assert detect_database_identity(tmp_path) == database


def test_identity_receipt_wins_over_mimic_compatible_schema(tmp_path: Path) -> None:
    pd.DataFrame({"stay_id": [1], "intime": ["2026-01-01"]}).to_parquet(
        tmp_path / "icustays.parquet", index=False
    )
    (tmp_path / "community_preparation_manifest.json").write_text(
        json.dumps({"database": "nwicu"}), encoding="utf-8"
    )
    assert detect_database_identity(tmp_path) == "nwicu"


def test_base_loader_accepts_registered_community_path(tmp_path: Path) -> None:
    (tmp_path / "community_preparation_manifest.json").write_text(
        json.dumps({"database": "jinhua"}), encoding="utf-8"
    )
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = False
    assert loader._setup_data_path(tmp_path, "jinhua") == tmp_path


def test_community_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "OMIX005817-01.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../PtAdmiTable.csv", "patient_SN\n1\n")
    with pytest.raises(ValueError, match="outside extraction root"):
        prepare_community_archives(tmp_path, "zhejiang_eicu")
    assert not (tmp_path.parent / "PtAdmiTable.csv").exists()


def test_zhejiang_stay_index_records_proxy_semantics(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "patient_SN": [11, 12],
            "Hospital_ID": [101, 101],
            "Discharge_DateTime": [5.0, 4.0],
            "DaysHospitalStay": [4, 2],
        }
    ).to_parquet(tmp_path / "ptadmitable.parquet", index=False)
    receipt = ensure_community_stays(tmp_path, "zhejiang_eicu")
    stays = pd.read_parquet(tmp_path / "stays.parquet")
    assert receipt is not None
    assert receipt["stay_count"] == 2
    assert "proxy" in " ".join(receipt["limitations"]).casefold()
    assert stays["stay_id"].tolist() == [11, 12]
    assert stays["patient_SN"].tolist() == [11, 12]
    assert (stays["outtime"] - stays["intime"]).tolist() == [96.0, 48.0]


def test_jinhua_stay_index_collapses_multiple_icu_transfers(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 10, 10],
            "outtime_base": [None, 2880, None],
            "out_department": ["Ward", "Intensive Care Unit", "Ward"],
            "out_department_en": ["Ward", "Intensive Care Unit", "Ward"],
            "intime_base": [60, 1440, 4320],
            "transfer_department": ["重症监护住院部", "Ward", "重症监护住院部"],
            "transfer_department_en": [
                "Intensive Care Unit",
                "Ward",
                "Intensive Care Unit",
            ],
        }
    ).to_parquet(tmp_path / "transfer.parquet", index=False)
    pd.DataFrame({"hadm_id": [10], "dischtime_base": [7200]}).to_parquet(
        tmp_path / "medical_record_front_page.parquet", index=False
    )
    receipt = ensure_community_stays(tmp_path, "jinhua")
    stays = pd.read_parquet(tmp_path / "stays.parquet")
    assert receipt is not None and receipt["stay_count"] == 1
    assert stays.loc[0, "stay_id"] == 10
    assert stays.loc[0, "intime"] == pytest.approx(1.0)
    assert stays.loc[0, "outtime"] == pytest.approx(48.0)


def test_zigong_stay_index_uses_transfer_bounds(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "PATIENT_ID": [7, 7],
            "INP_NO": [70, 70],
            "StartTime": [0.0, 24.0],
            "StopTime": [24.0, 72.0],
        }
    ).to_parquet(tmp_path / "dttransfer.parquet", index=False)
    receipt = ensure_community_stays(tmp_path, "zigong")
    stays = pd.read_parquet(tmp_path / "stays.parquet")
    assert receipt is not None and receipt["stay_count"] == 1
    assert stays.loc[0, "INP_NO"] == 70
    assert stays.loc[0, "intime"] == 0.0
    assert stays.loc[0, "outtime"] == 72.0


def test_nwicu_nested_raw_csv_converts_and_writes_identity_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data" / "nw_icu" / "icustays.csv.gz"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [2],
            "stay_id": [3],
            "intime": ["2026-01-01 00:00:00"],
            "outtime": ["2026-01-02 00:00:00"],
            "los": [1.0],
        }
    ).to_csv(source, index=False, compression="gzip")

    results = DataConverter(tmp_path, parallel_workers=1, verbose=False).convert_all()

    assert results["icustays.csv.gz"]["status"] == ConversionStatus.COMPLETED
    assert results["community_stay_index"]["row_count"] == 1
    receipt = json.loads(
        (tmp_path / "community_preparation_manifest.json").read_text(encoding="utf-8")
    )
    assert receipt["database"] == "nwicu"
    assert detect_database_identity(tmp_path) == "nwicu"


def test_jinhua_string_ids_and_minute_offsets_load_through_public_api(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": ["IP1"],
            "stay_id": ["IP1"],
            "intime": [1.0],
            "outtime": [12.0],
        }
    ).to_parquet(tmp_path / "stays.parquet", index=False)
    events = tmp_path / "vital_signs"
    events.mkdir()
    pd.DataFrame(
        {
            "hadm_id": ["IP1"],
            "charttime_base": [120],
            "subcategory_name_en": ["Pulse"],
            "value": [88.0],
            "subcategory_unit": ["bpm"],
        }
    ).to_parquet(events / "part-00000.parquet", index=False)
    (tmp_path / "community_preparation_manifest.json").write_text(
        json.dumps({"database": "jinhua"}), encoding="utf-8"
    )

    result = load_concepts(
        "hr",
        patient_ids={"hadm_id": ["IP1"]},
        database="jinhua",
        data_path=tmp_path,
        interval="1h",
        parallel_workers=1,
        concept_workers=1,
    )

    assert result[["hadm_id", "charttime", "hr"]].to_dict("records") == [
        {"hadm_id": "IP1", "charttime": 1.0, "hr": 88.0}
    ]


def test_zigong_wide_table_casts_strings_and_uses_hour_offsets(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [70],
            "stay_id": [70],
            "INP_NO": [70],
            "intime": [1.0],
            "outtime": [12.0],
        }
    ).to_parquet(tmp_path / "stays.parquet", index=False)
    events = tmp_path / "dtnursingchart"
    events.mkdir()
    pd.DataFrame(
        {
            "INP_NO": [70],
            "ChartTime": [2.25],
            "heart_rate": ["88"],
            "temperature": ["37.2"],
        }
    ).to_parquet(events / "part-00000.parquet", index=False)
    (tmp_path / "community_preparation_manifest.json").write_text(
        json.dumps({"database": "zigong"}), encoding="utf-8"
    )

    result = load_concepts(
        ["hr", "temp"],
        patient_ids={"INP_NO": [70]},
        database="zigong",
        data_path=tmp_path,
        interval="1h",
        parallel_workers=1,
        concept_workers=1,
    )

    assert result.loc[0, "INP_NO"] == 70
    assert result.loc[0, "charttime"] == 1.0
    assert result.loc[0, "hr"] == 88.0
    assert result.loc[0, "temp"] == pytest.approx(37.2)
