from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.config import DataSourceRegistry
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
