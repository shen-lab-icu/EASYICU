from __future__ import annotations

import warnings

import pandas as pd

from easyicu.base import BaseICULoader
from easyicu.config import DataSourceConfig
from easyicu.concept import ConceptResolver
from easyicu.concept.callbacks import CALLBACK_REGISTRY, register_callback
from easyicu.concept.expr_parser import _default_aggregator_for_dtype
from easyicu.concept.parser import default_aggregator_for_dtype
from easyicu.concept.schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.table import ICUTable
from easyicu.io.ts_utils import change_interval


def test_auto_aggregator_for_bool_uses_any_semantics() -> None:
    series = pd.Series([False, False, True], name="flag")

    assert _default_aggregator_for_dtype(series) == "any"
    assert default_aggregator_for_dtype(series) == "any"
    assert ConceptResolver._resolve_aggregator(series, "auto") == "any"


def test_bool_auto_aggregation_is_occurrence_not_count_or_majority() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": pd.to_timedelta([0, 0, 0], unit="h"),
            "rrt": [False, False, True],
        }
    )
    table = ICUTable(
        data=frame,
        id_columns=["stay_id"],
        index_column="charttime",
        value_column="rrt",
    )

    aggregation = ConceptResolver._resolve_aggregator(frame["rrt"], "auto")
    result = change_interval(
        table,
        interval=pd.Timedelta(hours=1),
        aggregation=aggregation,
    ).data

    assert result["rrt"].tolist() == [True]
    assert str(result["rrt"].dtype) == "bool"


def test_recursive_bool_callback_load_concepts_uses_any_after_callback() -> None:
    def callback(_tables, _ctx) -> ICUTable:
        return ICUTable(
            data=pd.DataFrame(
                {
                    "stay_id": [1, 1, 1],
                    "charttime": pd.to_timedelta([0, 0, 0], unit="h"),
                    "derived_flag": [False, False, True],
                }
            ),
            id_columns=["stay_id"],
            index_column="charttime",
            value_column="derived_flag",
        )

    class DataSource:
        config = DataSourceConfig(
            name="unit",
            tables={
                "marker_events": {
                    "defaults": {
                        "id_var": "stay_id",
                        "index_var": "charttime",
                        "val_var": "marker",
                    }
                }
            },
        )
        base_path = None

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            del verbose
            frame = pd.DataFrame(
                {
                    "stay_id": [1],
                    "charttime": [pd.Timedelta(0)],
                    "marker": [1.0],
                }
            )
            for filter_spec in filters or []:
                frame = filter_spec.apply(frame)
            if columns:
                required = ["stay_id", "charttime", *columns]
                frame = frame[[col for col in required if col in frame.columns]]
            return ICUTable(
                data=frame,
                id_columns=["stay_id"],
                index_column="charttime",
                value_column="marker",
            )

    dictionary = ConceptDictionary(
        {
            "marker": ConceptDefinition(
                name="marker",
                sources={
                    "unit": [
                        ConceptSource(
                            table="marker_events",
                            value_var="marker",
                        )
                    ]
                },
            ),
            "derived_flag": ConceptDefinition(
                name="derived_flag",
                sources={},
                sub_concepts=["marker"],
                callback="test_bool_occurrence_callback",
            ),
        }
    )
    previous = CALLBACK_REGISTRY.get("test_bool_occurrence_callback")
    register_callback("test_bool_occurrence_callback", callback)
    try:
        loaded = ConceptResolver(dictionary).load_concepts(
            ["derived_flag"],
            DataSource(),
            merge=False,
            interval=pd.Timedelta(hours=1),
            r_compatible=False,
            verbose=False,
            concept_workers=1,
        )
    finally:
        if previous is None:
            CALLBACK_REGISTRY.pop("test_bool_occurrence_callback", None)
        else:
            CALLBACK_REGISTRY["test_bool_occurrence_callback"] = previous

    result = loaded["derived_flag"].data if isinstance(loaded, dict) else loaded.data

    assert result["derived_flag"].tolist() == [True]
    assert str(result["derived_flag"].dtype) == "bool"


def test_r_style_merge_preserves_requested_empty_event_concept_column() -> None:
    class DataSource:
        config = DataSourceConfig(name="sic", tables={})

    resolver = ConceptResolver(ConceptDictionary({}))
    tables = {
        "urine": ICUTable(
            data=pd.DataFrame(
                {
                    "CaseID": [1, 1],
                    "charttime": [0.0, 1.0],
                    "urine": [25.0, 40.0],
                }
            ),
            id_columns=["CaseID"],
            index_column="charttime",
            value_column="urine",
        ),
        "rrt": ICUTable(
            data=pd.DataFrame(columns=["CaseID", "charttime", "rrt"]),
            id_columns=["CaseID"],
            index_column="charttime",
            value_column="rrt",
        ),
    }

    result = resolver._to_r_format_merged_enhanced(
        tables,
        ["urine", "rrt"],
        data_source=DataSource(),
    )

    assert "urine" in result.columns
    assert "rrt" in result.columns
    assert result["rrt"].isna().all()


def test_base_merge_preserves_requested_empty_event_concept_column() -> None:
    loader = BaseICULoader.__new__(BaseICULoader)
    results = {
        "urine": pd.DataFrame(
            {
                "CaseID": [1, 1],
                "charttime": [0.0, 1.0],
                "urine": [25.0, 40.0],
            }
        ),
        "rrt": pd.DataFrame(columns=["CaseID", "charttime", "rrt"]),
    }

    result = loader._merge_concepts(results, keep_components=False)

    assert "urine" in result.columns
    assert "rrt" in result.columns
    assert result["rrt"].isna().all()


def test_base_batched_merge_aligns_sparse_concepts_without_dataframe_set_index(
    monkeypatch,
) -> None:
    """The wide fallback must retain only one transient key index at a time."""
    loader = BaseICULoader.__new__(BaseICULoader)
    results = {
        "hr": pd.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "charttime": [0.0, 1.0, 0.0],
                "hr": [80.0, 90.0, 70.0],
            }
        ),
        "map": pd.DataFrame(
            {
                "stay_id": [1, 2, 2],
                "charttime": [1.0, 0.0, 1.0],
                "map": [65.0, 75.0, 72.0],
            }
        ),
        "rrt": pd.DataFrame(
            {
                "stay_id": [2],
                "charttime": [1.0],
                "rrt": [True],
            }
        ),
    }

    def forbidden_set_index(*args, **kwargs):
        raise AssertionError("batched merge must not build per-concept DataFrame indexes")

    monkeypatch.setattr(pd.DataFrame, "set_index", forbidden_set_index)

    result = loader._merge_concepts(results, keep_components=False).sort_values(
        ["stay_id", "charttime"],
        ignore_index=True,
    )

    assert result[["stay_id", "charttime"]].values.tolist() == [
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 1.0],
    ]
    assert result["hr"].tolist()[:3] == [80.0, 90.0, 70.0]
    assert pd.isna(result["hr"].iloc[3])
    assert pd.isna(result["map"].iloc[0])
    assert result["map"].tolist()[1:] == [65.0, 75.0, 72.0]
    assert result["rrt"].isna().tolist() == [True, True, True, False]
    assert bool(result["rrt"].iloc[3]) is True
    assert str(result["rrt"].dtype) == "boolean"


def test_mimic_fio2_carevue_torr_unit_is_not_reported_as_mismatch() -> None:
    class DataSource:
        config = DataSourceConfig(
            name="mimic",
            tables={
                "chartevents": {
                    "defaults": {
                        "id_var": "icustay_id",
                        "index_var": "charttime",
                        "val_var": "valuenum",
                        "unit_var": "valueuom",
                    }
                },
                "icustays": {
                    "defaults": {
                        "id_var": "icustay_id",
                        "index_var": "intime",
                    }
                }
            },
        )
        base_path = None

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            del verbose
            if table_name == "icustays":
                return ICUTable(
                    data=pd.DataFrame(
                        {
                            "icustay_id": [1],
                            "intime": [pd.Timestamp("2020-01-01")],
                        }
                    ),
                    id_columns=["icustay_id"],
                    index_column="intime",
                )
            frame = pd.DataFrame(
                {
                    "icustay_id": [1],
                    "charttime": [pd.Timestamp("2020-01-01")],
                    "itemid": [189],
                    "valuenum": [0.5],
                    "valueuom": ["torr"],
                }
            )
            for filter_spec in filters or []:
                frame = filter_spec.apply(frame)
            if columns:
                required = ["icustay_id", "charttime", *columns]
                frame = frame[[col for col in required if col in frame.columns]]
            return ICUTable(
                data=frame,
                id_columns=["icustay_id"],
                index_column="charttime",
                value_column="valuenum",
                unit_column="valueuom",
            )

    dictionary = ConceptDictionary(
        {
            "fio2": ConceptDefinition(
                name="fio2",
                units=["%"],
                minimum=21,
                maximum=100,
                sources={
                    "mimic": [
                        ConceptSource(
                            table="chartevents",
                            sub_var="itemid",
                            ids=[189],
                            value_var="valuenum",
                            unit_var="valueuom",
                            callback="transform_fun(percent_as_numeric)",
                        )
                    ]
                },
            )
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = ConceptResolver(dictionary).load_concepts(
            ["fio2"],
            DataSource(),
            merge=False,
            r_compatible=False,
            verbose=False,
            concept_workers=1,
        )

    assert not [
        warning
        for warning in captured
        if "不是所有单位都在允许列表中" in str(warning.message)
    ]
    frame = result["fio2"].data
    assert frame["fio2"].tolist() == [50.0]
