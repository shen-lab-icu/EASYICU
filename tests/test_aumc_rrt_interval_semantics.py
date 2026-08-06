"""Regression tests for AmsterdamUMCdb RRT episode semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept.schema import ConceptSource
from easyicu.load_concepts import ConceptLoader
from easyicu.utils.compat import POINT_EVENT_CONCEPTS, WINDOW_CONCEPTS, expand_interval_rows


ROOT = Path(__file__).resolve().parents[1]
DICTIONARIES = (
    ROOT / "src/easyicu/data/concept-dict.json",
    ROOT / "src/easyicu/data/sofa2-dict.json",
)


@pytest.mark.parametrize("dictionary_path", DICTIONARIES)
def test_aumc_rrt_uses_treatment_process_intervals_only(
    dictionary_path: Path,
) -> None:
    dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
    sources = dictionary["rrt"]["sources"]["aumc"]

    assert sources == [
        {
            "table": "processitems",
            "sub_var": "itemid",
            "ids": [12465, 16363],
            "index_var": "start",
            "dur_var": "stop",
            "callback": "transform_fun(set_val(TRUE))",
            "_comment": sources[0]["_comment"],
        }
    ]
    assert not ({9161, 9162, 9163, 16352} & set(sources[0]["ids"]))


def test_compatibility_loader_projects_explicit_duration_end_column() -> None:
    loader = ConceptLoader.__new__(ConceptLoader)
    loader._infer_required_columns = (  # type: ignore[method-assign]
        lambda _table, _id_type, extra: list(extra)
    )
    source = ConceptSource(
        table="processitems",
        sub_var="itemid",
        index_var="start",
        dur_var="stop",
    )

    assert loader._columns_for_source(source, "icustay") == [
        "itemid",
        "start",
        "stop",
    ]


def test_rrt_episode_expands_to_active_hourly_status() -> None:
    assert "rrt" in WINDOW_CONCEPTS
    assert "rrt" not in POINT_EVENT_CONCEPTS
    frame = pd.DataFrame(
        {
            "id": [11, 11],
            "time": [4.17, 14.57],
            "duration": [7.90, 13.33],
            "rrt": [True, True],
        }
    )

    expanded = expand_interval_rows(
        frame,
        "rrt",
        id_col="id",
        time_col="time",
        value_col="rrt",
        interval_hours=1.0,
    )

    assert expanded.loc[expanded["rrt"], "time"].tolist() == [
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
        26.0,
        27.0,
    ]
