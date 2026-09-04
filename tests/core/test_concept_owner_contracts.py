from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import easyicu.api.concepts as concept_api
import easyicu.concept as concept_module
from easyicu.concept.data_source_contract import ConceptDataSourceStorage
from easyicu.concept.source_duration import (
    drop_negative_source_end_durations,
    source_duration_is_end,
)
from easyicu.concept.window_expansion import expand_wintbl_vectorized
from easyicu.datasource import ICUDataSource


def test_wintbl_expansion_is_independently_owned_and_preserves_row_shape() -> None:
    source = pd.DataFrame(
        {
            "stay_id": [7, 8],
            "time": [0.5, 2.0],
            "duration": [2.0, 0.0],
            "dose": [1.5, 2.5],
        }
    )

    expanded = expand_wintbl_vectorized(
        source,
        idx_col="time",
        dur_col="duration",
        id_cols=["stay_id"],
        value_columns=["dose"],
        interval_hours=1.0,
        duration_zero_single=True,
    )

    assert expanded.to_dict("records") == [
        {"time": 0.0, "stay_id": 7, "dose": 1.5},
        {"time": 1.0, "stay_id": 7, "dose": 1.5},
        {"time": 2.0, "stay_id": 7, "dose": 1.5},
        {"time": 2.0, "stay_id": 8, "dose": 2.5},
    ]
    assert concept_module._expand_wintbl_vectorized is expand_wintbl_vectorized


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_wintbl_expansion_rejects_a_nonpositive_interval(interval: float) -> None:
    with pytest.raises(ValueError, match="interval_hours must be positive"):
        expand_wintbl_vectorized(
            pd.DataFrame({"time": [0.0], "duration": [1.0]}),
            idx_col="time",
            dur_col="duration",
            id_cols=[],
            value_columns=[],
            interval_hours=interval,
        )


def test_source_duration_owner_uses_schema_semantics_and_quarantines_bad_rows() -> None:
    source = SimpleNamespace(dur_var="drugstopoffset", params={"dur_is_end": False})
    frame = pd.DataFrame({"dur_var": [2.0, -1.0], "value": [10, 20]})

    assert source_duration_is_end(source) is False
    assert drop_negative_source_end_durations(
        frame,
        concept_name="norepi_rate",
        source_table="infusiondrug",
    ).to_dict("records") == [{"dur_var": 2.0, "value": 10}]


def test_icu_data_source_satisfies_the_public_concept_storage_contract() -> None:
    source = object.__new__(ICUDataSource)

    assert isinstance(source, ConceptDataSourceStorage)


def test_concept_consumers_no_longer_reach_into_private_storage_layout() -> None:
    forbidden = (
        "._resolve_bucket_directory(",
        "._resolve_flat_parquet_directory(",
        "._resolve_loader_from_disk(",
        "._get_bucket_files_for_ids(",
    )
    source = inspect.getsource(concept_module) + inspect.getsource(concept_api)

    assert all(token not in source for token in forbidden)


def test_public_storage_methods_preserve_the_existing_layout_implementation(
    tmp_path: Path,
) -> None:
    source = object.__new__(ICUDataSource)
    source._resolve_bucket_directory = lambda _name: tmp_path / "bucket"
    source._resolve_flat_parquet_directory = lambda _name: tmp_path / "flat"
    source._resolve_loader_from_disk = lambda _name: tmp_path / "table.parquet"
    source._get_bucket_files_for_ids = lambda *_args: (
        {2},
        4,
        (tmp_path / "2.parquet",),
    )

    assert source.resolve_bucket_directory("events") == tmp_path / "bucket"
    assert source.resolve_flat_parquet_directory("events") == tmp_path / "flat"
    assert source.resolve_loader_from_disk("events") == tmp_path / "table.parquet"
    assert source.get_bucket_files_for_ids(tmp_path, [2], object()) == (
        {2},
        4,
        (tmp_path / "2.parquet",),
    )
