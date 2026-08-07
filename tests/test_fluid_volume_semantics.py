"""Regression contracts for cross-database hourly fluid volume semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptSource, _apply_callback
from easyicu.concept.callbacks import _callback_fluid_balance_cumulative
from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary
from easyicu.config import DataSourceConfig
from easyicu.table import ICUTable
from easyicu.utils.callback_utils import (
    distribute_volume_hourly,
    normalize_volume_to_ml,
)


def test_partial_hour_allocation_conserves_and_preaggregates_volume():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "starttime": [0.25, 0.50],
            "endtime": [2.00, 1.50],
            "amount": [175.0, 100.0],
        }
    )

    result = distribute_volume_hourly(
        frame,
        val_col="amount",
        end_col="endtime",
        index_col="starttime",
    )

    assert result[["stay_id", "starttime"]].values.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert result["amount"].tolist() == pytest.approx([125.0, 150.0])
    assert result["amount"].sum() == pytest.approx(frame["amount"].sum())

    one_row_chunks = distribute_volume_hourly(
        frame,
        val_col="amount",
        end_col="endtime",
        index_col="starttime",
        row_chunk_size=1,
    )
    pd.testing.assert_frame_equal(result, one_row_chunks)


def test_zero_or_missing_end_is_a_single_bolus_and_negative_duration_is_dropped():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "starttime": [3.2, 4.8, 6.0],
            "endtime": [3.2, np.nan, 5.0],
            "amount": [20.0, 30.0, 999.0],
        }
    )

    result = distribute_volume_hourly(
        frame,
        val_col="amount",
        end_col="endtime",
        index_col="starttime",
    )

    assert result["starttime"].tolist() == [3.0, 4.0]
    assert result["amount"].tolist() == [20.0, 30.0]


def test_datetime_bins_are_anchored_to_icu_admission_not_wall_clock():
    frame = pd.DataFrame(
        {
            "stay_id": [7],
            "starttime": [pd.Timestamp("2025-01-01 10:30:00")],
            "endtime": [pd.Timestamp("2025-01-01 12:00:00")],
            "amount": [150.0],
        }
    )
    origins = pd.DataFrame(
        {"stay_id": [7], "intime": [pd.Timestamp("2025-01-01 10:00:00")]}
    )

    result = distribute_volume_hourly(
        frame,
        val_col="amount",
        end_col="endtime",
        index_col="starttime",
        origin_times=origins,
        origin_col="intime",
    )

    assert result["starttime"].tolist() == [0.0, 1.0]
    assert result["amount"].tolist() == pytest.approx([50.0, 100.0])


def test_explicit_metric_volume_units_are_normalized_to_ml():
    values = pd.Series([1.5, 20.0, 300.0, 4000.0, 5.0, 99.0])
    units = pd.Series(["L", "mL", "cm3", "µL", "mm^3", "unknown"])

    result = normalize_volume_to_ml(values, units)

    assert result.iloc[:5].tolist() == pytest.approx(
        [1500.0, 20.0, 300.0, 4.0, 0.005]
    )
    assert pd.isna(result.iloc[5])


def test_aumc_bins_return_absolute_minutes_for_single_generic_alignment():
    frame = pd.DataFrame(
        {
            "admissionid": [9],
            "start": [1030.0],
            "stop": [1120.0],
            "fluidin": [150.0],
        }
    )
    origins = pd.DataFrame({"admissionid": [9], "admittedat": [1000.0]})

    result = distribute_volume_hourly(
        frame,
        val_col="fluidin",
        end_col="stop",
        index_col="start",
        id_col="admissionid",
        origin_times=origins,
        origin_col="admittedat",
        numeric_time_unit="minutes",
        output_time_unit="absolute_minutes",
    )

    assert result["start"].tolist() == [1000.0, 1060.0]
    assert result["fluidin"].tolist() == pytest.approx([50.0, 100.0])
    assert result["fluidin"].sum() == pytest.approx(150.0)


class _AumcSource:
    config = SimpleNamespace(name="aumc")

    def load_table(self, table, columns=None, verbose=False):
        assert table == "admissions"
        return ICUTable(
            data=pd.DataFrame({"admissionid": [9], "admittedat": [1000.0]}),
            id_columns=["admissionid"],
            index_column="admittedat",
        )


def test_dispatcher_uses_larger_aumc_fluid_or_solution_total():
    frame = pd.DataFrame(
        {
            "admissionid": [9],
            "start": [1000.0],
            "stop": [1060.0],
            "total_input_ml": [100.0],
            "solutionadministered": [120.0],
        }
    )
    source = ConceptSource.from_mapping(
        {
            "table": "drugitems",
            "val_var": "fluidin",
            "index_var": "start",
            "callback": "distribute_volume_hourly",
            "end_var": "stop",
            "alternate_value_var": "solutionadministered",
        }
    )

    result = _apply_callback(
        frame,
        source,
        concept_name="total_input_ml",
        data_source=_AumcSource(),
    )

    assert result["charttime"].tolist() == [1000.0]
    assert result["total_input_ml"].tolist() == [120.0]


def test_loader_preserves_canonical_time_metadata_after_callback_renames_index():
    class DataSource:
        config = DataSourceConfig(
            name="miiv",
            tables={
                "inputevents": {
                    "defaults": {
                        "id_var": "stay_id",
                        "index_var": "starttime",
                        "val_var": "amount",
                        "time_vars": ["starttime", "endtime"],
                    }
                },
                "icustays": {
                    "defaults": {
                        "id_var": "stay_id",
                        "index_var": "intime",
                        "time_vars": ["intime"],
                    }
                },
            },
        )
        base_path = None

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            del verbose
            if table_name == "icustays":
                frame = pd.DataFrame(
                    {
                        "stay_id": [1],
                        "intime": [pd.Timestamp("2025-01-01 10:00:00")],
                    }
                )
                return ICUTable(
                    data=frame,
                    id_columns=["stay_id"],
                    index_column="intime",
                )
            frame = pd.DataFrame(
                {
                    "stay_id": [1],
                    "itemid": [10],
                    "starttime": [pd.Timestamp("2025-01-01 10:30:00")],
                    "endtime": [pd.Timestamp("2025-01-01 12:00:00")],
                    "amount": [0.15],
                    "amountuom": ["L"],
                }
            )
            for filter_spec in filters or []:
                frame = filter_spec.apply(frame)
            return ICUTable(
                data=frame,
                id_columns=["stay_id"],
                index_column="starttime",
                value_column="amount",
                time_columns=["starttime", "endtime"],
            )

    dictionary = ConceptDictionary(
        {
            "total_input_ml": ConceptDefinition(
                name="total_input_ml",
                aggregate="sum",
                sources={
                    "miiv": [
                        ConceptSource.from_mapping(
                            {
                                "table": "inputevents",
                                "sub_var": "itemid",
                                "ids": [10],
                                "val_var": "amount",
                                "unit_var": "amountuom",
                                "index_var": "starttime",
                                "callback": "distribute_volume_hourly",
                                "end_var": "endtime",
                                "extra_vars": ["endtime"],
                            }
                        )
                    ]
                },
            )
        }
    )

    loaded = ConceptResolver(dictionary).load_concepts(
        ["total_input_ml"],
        DataSource(),
        merge=False,
        interval=pd.Timedelta(hours=1),
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )["total_input_ml"]

    assert loaded.index_column == "charttime"
    assert loaded.data["charttime"].tolist() == [0.0, 1.0]
    assert loaded.data["total_input_ml"].tolist() == pytest.approx([50.0, 100.0])


def test_total_input_dictionary_declares_interval_semantics():
    dictionary_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "easyicu"
        / "data"
        / "concept-dict.json"
    )
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    sources = dictionary["total_input_ml"]["sources"]

    expected = {
        "miiv": ("starttime", "endtime"),
        "mimic": ("starttime", "endtime"),
        "mimic_demo": ("starttime", "endtime"),
        "aumc": ("start", "stop"),
    }
    for database, (start, end) in expected.items():
        source = sources[database][0]
        assert source["callback"] == "distribute_volume_hourly"
        assert source["index_var"] == start
        assert source["end_var"] == end
        assert end in source["extra_vars"]

    for database in ("miiv", "mimic", "mimic_demo"):
        assert sources[database][0]["unit_var"] == "amountuom"

    assert sources["aumc"][0]["alternate_value_var"] == "solutionadministered"


def test_cumulative_balance_starts_at_hour_zero_without_hidden_prehistory():
    fluid_balance = ICUTable(
        data=pd.DataFrame(
            {
                "stay_id": [1, 1, 1, 2, 2],
                "charttime": [-2.0, 0.0, 1.0, -1.0, 2.0],
                "fluid_balance": [-500.0, 100.0, -25.0, 900.0, 40.0],
            }
        ),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column="fluid_balance",
    )

    result = _callback_fluid_balance_cumulative(
        {"fluid_balance": fluid_balance},
        SimpleNamespace(),
    ).data

    assert result[["stay_id", "charttime"]].values.tolist() == [
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    assert result["fluid_balance_cumulative"].tolist() == [100.0, 75.0, 40.0]
