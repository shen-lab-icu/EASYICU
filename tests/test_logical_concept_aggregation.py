from __future__ import annotations

import pandas as pd

from easyicu.config import DataSourceConfig
from easyicu.concept import ConceptResolver
from easyicu.concept_callbacks import CALLBACK_REGISTRY, register_callback
from easyicu.concept_expr_parser import _default_aggregator_for_dtype
from easyicu.concept_parser import default_aggregator_for_dtype
from easyicu.concept_schema import ConceptDefinition, ConceptDictionary, ConceptSource
from easyicu.table import ICUTable
from easyicu.ts_utils import change_interval


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
                frame = frame[[col for col in columns if col in frame.columns]]
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
