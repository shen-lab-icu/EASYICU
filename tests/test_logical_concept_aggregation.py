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
from easyicu.resources import load_data_sources
from easyicu.table import ICUTable
from easyicu.io.ts_utils import change_interval
from easyicu.utils import compat


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


def test_miiv_rrt_keeps_charted_points_and_expands_procedure_windows() -> None:
    """A mixed MIIV RRT source must not erase either source representation."""

    class DataSource:
        config = load_data_sources().get("miiv")
        base_path = None

        def __init__(self) -> None:
            intime = pd.Timestamp("2180-01-01 00:00:00")
            self.frames = {
                "chartevents": pd.DataFrame(
                    {
                        "stay_id": [1],
                        "charttime": [intime + pd.Timedelta(hours=2)],
                        "itemid": [224149],
                        "valuenum": [10.0],
                    }
                ),
                "procedureevents": pd.DataFrame(
                    {
                        "stay_id": [1],
                        "starttime": [intime + pd.Timedelta(hours=4)],
                        "endtime": [intime + pd.Timedelta(hours=6)],
                        "itemid": [225441],
                        "value": [120.0],
                    }
                ),
                "icustays": pd.DataFrame(
                    {
                        "stay_id": [1],
                        "intime": [intime],
                        "outtime": [intime + pd.Timedelta(days=1)],
                        "los": [1.0],
                    }
                ),
            }

        def load_table(self, table_name, columns=None, filters=None, verbose=False):
            del columns, verbose
            frame = self.frames[table_name].copy()
            for filter_spec in filters or []:
                frame = filter_spec.apply(frame)
            defaults = self.config.get_table(table_name).defaults
            return ICUTable(
                data=frame,
                id_columns=[defaults.id_var],
                index_column=(
                    defaults.index_var
                    if defaults.index_var in frame.columns
                    else None
                ),
                value_column=(
                    defaults.val_var if defaults.val_var in frame.columns else None
                ),
                time_columns=[
                    column
                    for column in defaults.time_vars or []
                    if column in frame.columns
                ],
            )

    dictionary = ConceptDictionary(
        {
            "rrt": ConceptDefinition(
                name="rrt",
                class_name="lgl_cncpt",
                sources={
                    "miiv": [
                        ConceptSource(
                            table="chartevents",
                            sub_var="itemid",
                            ids=[224149],
                            value_var="valuenum",
                            index_var="charttime",
                            callback="transform_fun(set_val(TRUE))",
                        ),
                        ConceptSource(
                            table="procedureevents",
                            sub_var="itemid",
                            ids=[225441],
                            value_var="value",
                            index_var="starttime",
                            callback="transform_fun(set_val(TRUE))",
                        ),
                    ]
                },
            )
        }
    )

    loaded = ConceptResolver(dictionary).load_concepts(
        ["rrt"],
        DataSource(),
        merge=False,
        interval=pd.Timedelta(hours=1),
        r_compatible=False,
        verbose=False,
        concept_workers=1,
    )
    result = loaded["rrt"].data.sort_values("charttime").reset_index(drop=True)

    assert result["charttime"].tolist() == [2.0, 4.0, 5.0, 6.0]
    assert result["rrt"].tolist() == [True, True, True, True]


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


def test_r_style_merge_uses_each_table_declared_time_index() -> None:
    """Uncommon source index names must not turn event concepts static."""

    class DataSource:
        config = DataSourceConfig(name="eicu", tables={})

    resolver = ConceptResolver(ConceptDictionary({}))
    tables = {
        "abx": ICUTable(
            data=pd.DataFrame(
                {
                    "patientunitstayid": [1],
                    "charttime": [0.0],
                    "abx": [True],
                }
            ),
            id_columns=["patientunitstayid"],
            index_column="charttime",
            value_column="abx",
        ),
        "phenytoin": ICUTable(
            data=pd.DataFrame(
                {
                    "patientunitstayid": [2],
                    "drugoffset": [6.0],
                    "phenytoin": [True],
                }
            ),
            id_columns=["patientunitstayid"],
            index_column="drugoffset",
            value_column="phenytoin",
        ),
    }

    result = resolver._to_r_format_merged_enhanced(
        tables,
        ["abx", "phenytoin"],
        data_source=DataSource(),
    ).sort_values(["patientunitstayid", "charttime"])

    assert result[["patientunitstayid", "charttime"]].to_dict("records") == [
        {"patientunitstayid": 1, "charttime": 0.0},
        {"patientunitstayid": 2, "charttime": 6.0},
    ]
    assert bool(result.loc[result["patientunitstayid"].eq(2), "phenytoin"].iloc[0])


def test_r_style_merge_drops_value_less_static_outer_artifact() -> None:
    class DataSource:
        config = DataSourceConfig(name="eicu", tables={})

    resolver = ConceptResolver(ConceptDictionary({}))
    tables = {
        "qsofa": ICUTable(
            data=pd.DataFrame(
                {
                    "patientunitstayid": [1],
                    "charttime": [0.0],
                    "qsofa": [1.0],
                }
            ),
            id_columns=["patientunitstayid"],
            index_column="charttime",
            value_column="qsofa",
        ),
        "apache_iv": ICUTable(
            data=pd.DataFrame(
                {
                    "patientunitstayid": [2],
                    "apache_iv": [None],
                }
            ),
            id_columns=["patientunitstayid"],
            index_column=None,
            value_column="apache_iv",
        ),
    }

    result = resolver._to_r_format_merged_enhanced(
        tables,
        ["qsofa", "apache_iv"],
        data_source=DataSource(),
    )

    assert result["patientunitstayid"].tolist() == [1]
    assert result["charttime"].tolist() == [0.0]


def test_r_style_sparse_merge_does_not_materialise_unused_time_grid(
    monkeypatch,
) -> None:
    def _unexpected_grid(*_args, **_kwargs):
        raise AssertionError("optimized merge must not build an unused key grid")

    monkeypatch.setattr(compat, "build_time_grid", _unexpected_grid)
    source = {
        "hr": pd.DataFrame(
            {"id": [1, 1], "time": [0.0, 2.0], "hr": [80.0, 82.0]}
        ),
        "map": pd.DataFrame(
            {"id": [1, 2], "time": [2.0, 1.5], "map": [65.0, 70.0]}
        ),
    }

    result = compat.merge_concepts_r_style(source).sort_values(
        ["stay_id", "charttime"]
    ).reset_index(drop=True)

    assert result[["stay_id", "charttime"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 0.0},
        {"stay_id": 1, "charttime": 2.0},
        {"stay_id": 2, "charttime": 1.0},
    ]
    assert result.loc[0, "hr"] == 80.0
    assert result.loc[1, "map"] == 65.0


def test_r_style_consume_input_releases_owned_source_mapping() -> None:
    source = {
        "hr": pd.DataFrame(
            {"id": [1, 1], "time": [0.0, 1.0], "hr": [80.0, 82.0]}
        ),
        "map": pd.DataFrame(
            {"id": [1], "time": [1.0], "map": [65.0]}
        ),
    }

    result = compat.merge_concepts_r_style(source, consume_input=True)

    assert source == {}
    assert result[["stay_id", "charttime"]].to_dict("records") == [
        {"stay_id": 1, "charttime": 0.0},
        {"stay_id": 1, "charttime": 1.0},
    ]
    assert result.loc[1, "map"] == 65.0


def test_r_style_public_default_preserves_source_mapping() -> None:
    source = {
        "hr": pd.DataFrame({"id": [1], "time": [0.0], "hr": [80.0]})
    }

    compat.merge_concepts_r_style(source)

    assert list(source) == ["hr"]


def test_r_style_no_grid_branch_preserves_all_null_timed_frame() -> None:
    source = {
        "flag": pd.DataFrame(
            {"id": [None], "time": [None], "flag": [True]}
        )
    }

    result = compat.merge_concepts_r_style(source)

    assert list(result.columns) == ["stay_id", "charttime", "flag"]
    assert len(result) == 1
    assert pd.isna(result.loc[0, "stay_id"])
    assert pd.isna(result.loc[0, "charttime"])
    assert bool(result.loc[0, "flag"]) is True


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


def test_partial_wide_merge_preserves_exact_fractional_charttime_keys() -> None:
    resolver = ConceptResolver(ConceptDictionary({}))
    wide = pd.DataFrame(
        {
            "admissionid": [16075, 16075],
            "charttime": [-0.383333333333, 0.616666666667],
            "hr": [89.0, 95.0],
        }
    )
    pulse = ICUTable(
        data=pd.DataFrame(
            {
                "admissionid": [16075, 16075],
                "charttime": [0.0, 0.616666666667],
                "pulse_pressure": [48.5, 97.0],
            }
        ),
        id_columns=["admissionid"],
        index_column="charttime",
        value_column="pulse_pressure",
    )

    observed = resolver._merge_partial_wide_result(
        wide,
        {"pulse_pressure": pulse},
        ["hr", "pulse_pressure"],
        {"hr"},
    ).sort_values("charttime", kind="mergesort")

    assert observed["charttime"].tolist() == [
        -0.383333333333,
        0.0,
        0.616666666667,
    ]
    by_time = observed.set_index("charttime")
    assert by_time.loc[-0.383333333333, "hr"] == 89.0
    assert pd.isna(by_time.loc[-0.383333333333, "pulse_pressure"])
    assert pd.isna(by_time.loc[0.0, "hr"])
    assert by_time.loc[0.0, "pulse_pressure"] == 48.5
    assert by_time.loc[0.616666666667, "hr"] == 95.0
    assert by_time.loc[0.616666666667, "pulse_pressure"] == 97.0


def test_partial_wide_merge_falls_back_on_duplicate_exact_keys() -> None:
    resolver = ConceptResolver(ConceptDictionary({}))
    wide = pd.DataFrame(
        {"admissionid": [1], "charttime": [0.25], "hr": [80.0]}
    )
    duplicate = ICUTable(
        data=pd.DataFrame(
            {
                "admissionid": [1, 1],
                "charttime": [0.25, 0.25],
                "map": [70.0, 72.0],
            }
        ),
        id_columns=["admissionid"],
        index_column="charttime",
        value_column="map",
    )

    observed = resolver._merge_partial_wide_result(
        wide,
        {"map": duplicate},
        ["hr", "map"],
        {"hr"},
    )

    assert observed is None


def test_partial_wide_merge_falls_back_instead_of_coercing_text_to_null() -> None:
    resolver = ConceptResolver(ConceptDictionary({}))
    wide = pd.DataFrame(
        {"stay_id": [1], "charttime": [0.0], "hr": [80.0]}
    )
    mode = ICUTable(
        data=pd.DataFrame(
            {"stay_id": [1], "charttime": [0.0], "vent_mode": ["PCV"]}
        ),
        id_columns=["stay_id"],
        index_column="charttime",
        value_column="vent_mode",
    )

    observed = resolver._merge_partial_wide_result(
        wide,
        {"vent_mode": mode},
        ["hr", "vent_mode"],
        {"hr"},
    )

    assert observed is None
