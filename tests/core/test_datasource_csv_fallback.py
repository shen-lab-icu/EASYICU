from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.config import DataSourceConfig, DataSourceRegistry
from easyicu.datasource import FilterOp, FilterSpec, ICUDataSource


def _repo_config(name: str):
    config_path = Path(__file__).resolve().parents[1] / "src/easyicu/data/data-sources.json"
    return DataSourceRegistry.from_json(config_path).get(name)


def test_mimic_services_prefers_csv_over_parquet(tmp_path: Path) -> None:
    (tmp_path / "services.parquet").write_bytes(b"not a parquet file")
    pd.DataFrame(
        {
            "ROW_ID": [1, 2],
            "SUBJECT_ID": [10, 11],
            "HADM_ID": [100, 101],
            "TRANSFERTIME": ["2100-01-01 00:00:00", "2100-01-02 00:00:00"],
            "CURR_SERVICE": ["MED", "SURG"],
        }
    ).to_csv(tmp_path / "SERVICES.csv.gz", index=False)

    config = _repo_config("mimic")
    source = ICUDataSource(config=config, base_path=tmp_path)

    table = source.load_table(
        "services",
        columns=["hadm_id", "curr_service"],
        filters=[FilterSpec(column="hadm_id", op=FilterOp.IN, value=[100])],
    )

    assert table.data["hadm_id"].tolist() == [100]
    assert table.data["curr_service"].tolist() == ["MED"]


def test_empty_item_filter_short_circuits_partition_read(tmp_path: Path) -> None:
    config = _repo_config("hirid")
    source = ICUDataSource(config=config, base_path=tmp_path)

    result = source._read_partitioned_data_duckdb(
        tmp_path,
        columns=["patientid", "variableid", "value"],
        itemid_filter_config=("variableid", set()),
    )

    assert list(result.columns) == ["patientid", "variableid", "value"]
    assert result.empty


def test_partitioned_duckdb_pushes_string_selector_with_quote(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 2],
            "nursingchartvalue": ["ventilator/ECMO", "O'Brien", "O'Brien"],
            "value": [1.0, 2.0, 3.0],
        }
    ).to_parquet(tmp_path / "part.parquet", index=False)
    source = ICUDataSource(config=DataSourceConfig(name="unit"), base_path=tmp_path)

    result = source._read_partitioned_data_duckdb(
        tmp_path,
        columns=["patientunitstayid", "nursingchartvalue", "value"],
        patient_ids_filter=FilterSpec(
            column="patientunitstayid",
            op=FilterOp.IN,
            value=[1],
        ),
        itemid_filter_config=("nursingchartvalue", {"O'Brien"}),
    )

    assert result.to_dict("records") == [
        {
            "patientunitstayid": 1,
            "nursingchartvalue": "O'Brien",
            "value": 2.0,
        }
    ]


def test_load_table_pushes_any_non_patient_exact_selector(monkeypatch) -> None:
    config = DataSourceConfig(
        name="unit",
        tables={
            "events": {
                "defaults": {
                    "id_var": "patientunitstayid",
                    "index_var": "offset",
                    "val_var": "value",
                }
            }
        },
    )
    source = ICUDataSource(config=config)
    captured = {}

    def _fake_load_raw_frame(
        table_name,
        columns,
        patient_ids_filter=None,
        concept_itemid_filter=None,
        wide_table_value_columns=None,
    ):
        del table_name, columns, wide_table_value_columns
        captured["patient"] = patient_ids_filter
        captured["selector"] = concept_itemid_filter
        return pd.DataFrame(
            {
                "patientunitstayid": [1, 1],
                "offset": [0, 1],
                "custom_label": ["keep", "drop"],
                "value": [2.0, 3.0],
            }
        )

    monkeypatch.setattr(source, "_load_raw_frame", _fake_load_raw_frame)
    table = source.load_table(
        "events",
        filters=[
            FilterSpec(column="custom_label", op=FilterOp.IN, value=["keep"]),
            FilterSpec(column="patientunitstayid", op=FilterOp.IN, value=[1]),
        ],
    )

    assert captured["selector"] == ("custom_label", {"keep"})
    assert captured["patient"].column == "patientunitstayid"
    assert table.data["custom_label"].tolist() == ["keep"]


def test_known_value_selector_keeps_value_to_itemid_fast_path(monkeypatch) -> None:
    config = DataSourceConfig(
        name="miiv",
        tables={
            "chartevents": {
                "defaults": {
                    "id_var": "stay_id",
                    "index_var": "charttime",
                    "val_var": "value",
                }
            }
        },
    )
    source = ICUDataSource(config=config)
    captured = {}

    def _fake_load_raw_frame(
        table_name,
        columns,
        patient_ids_filter=None,
        concept_itemid_filter=None,
        wide_table_value_columns=None,
    ):
        del table_name, columns, wide_table_value_columns
        captured["selector"] = concept_itemid_filter
        return pd.DataFrame(
            {
                "stay_id": [1],
                "charttime": [0],
                "itemid": [223900],
                "value": ["No Response-ETT"],
            }
        )

    monkeypatch.setattr(source, "_load_raw_frame", _fake_load_raw_frame)
    result = source.load_table(
        "chartevents",
        filters=[
            FilterSpec(
                column="value",
                op=FilterOp.IN,
                value=["No Response-ETT"],
            ),
            FilterSpec(column="stay_id", op=FilterOp.IN, value=[1]),
        ],
    )

    assert captured["selector"] == ("itemid", {223900})
    assert result.data["value"].tolist() == ["No Response-ETT"]
