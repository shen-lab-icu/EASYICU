"""Ventilator axes must describe one observed native mode, including ties."""

from dataclasses import replace

import pandas as pd
import pytest

from easyicu.api import EXTRACT_MODULES
from easyicu.concept.callbacks import apply_vent_mode_frame, _load_vent_mode_map
from easyicu.io.ts_utils import change_interval
from easyicu.resources import load_dictionary
from easyicu.table import ICUTable


def hourly_axis(frame, database, axis, column):
    mapped = apply_vent_mode_frame(frame, "value", database, axis, column)
    table = ICUTable(
        data=mapped[["stay_id", "charttime", column]],
        id_columns=["stay_id"], index_column="charttime", value_column=column,
    )
    return change_interval(
        table, interval=pd.Timedelta(hours=1), aggregation="first", time_unit="hours"
    ).data


@pytest.mark.parametrize(
    "database,labels",
    [
        ("aumc", ["CPPV", "SIMV_ASB"]),
        ("aumc", ["CPAP/ASB", "IPPV/ASSIST"]),
        ("aumc", ["PC ", "VC (trig)"]),
        ("mimic", ["CMV", "Pressure Support"]),
        ("miiv", ["Standby", "VOL/AC"]),
        ("hirid", ["1", "2"]),
    ],
)
def test_exact_time_conflicts_never_create_a_hybrid_mode(database, labels):
    frame = pd.DataFrame({
        "stay_id": [1, 1, 2], "charttime": [0.25, 0.25, 0.25],
        "value": [*labels, labels[-1]],
    })
    expected = _load_vent_mode_map()[database]["map"][min(v.strip() for v in labels)]
    outputs = []
    for source in [frame, frame.iloc[::-1]]:
        left = hourly_axis(source, database, "control", "vent_mode")
        right = hourly_axis(source, database, "seq", "vent_breath_seq")
        joined = left.merge(right, on=["stay_id", "charttime"]).sort_values("stay_id")
        row = joined.iloc[0]
        assert (row.vent_mode, row.vent_breath_seq) == (expected["control"], expected["seq"])
        assert len(joined) == 2
        outputs.append(joined.reset_index(drop=True))
    pd.testing.assert_frame_equal(*outputs)


def test_source_time_precedes_native_label_order_and_partition_does_not_change_it():
    frame = pd.DataFrame({
        "stay_id": [1, 1, 1, 2], "charttime": [0.75, 0.25, 0.25, 0.25],
        "value": ["CPPV", "SIMV_ASB", "VC ", "CPPV"],
    })
    # CPPV sorts first lexically but occurs later. The 0.25-h conflict is
    # resolved to SIMV_ASB before either derived axis is hourly-aggregated.
    for axis, column, expected in [
        ("control", "vent_mode", "unspecified"),
        ("seq", "vent_breath_seq", "simv"),
    ]:
        full = hourly_axis(frame, "aumc", axis, column).sort_values("stay_id").reset_index(drop=True)
        parts = pd.concat([
            hourly_axis(group, "aumc", axis, column)
            for _, group in frame.groupby("stay_id")
        ]).sort_values("stay_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(full, parts)
        assert full.iloc[0][column] == expected


def test_unmapped_earlier_native_label_does_not_hide_valid_mode():
    frame = pd.DataFrame({
        "stay_id": [1, 1], "charttime": [0.25, 0.25], "value": ["AAA unknown", "VC "],
    })
    result = apply_vent_mode_frame(frame, "value", "aumc", "control", "vent_mode")
    assert result.vent_mode.tolist() == ["volume"]
    assert set(result.columns) == set(frame.columns) | {"vent_mode"}


def test_both_axes_use_identical_native_sources_in_all_supported_databases():
    dictionary = load_dictionary(include_sofa2=True)
    control = dictionary.get("vent_mode").sources
    sequence = dictionary.get("vent_breath_seq").sources
    # mimic_demo is an alias, not a seventh database. eICU/SIC have no source
    # for these canonical mode axes; this does not describe their IMV support.
    assert set(control) == set(sequence) == {"aumc", "hirid", "miiv", "mimic", "mimic_demo"}
    for database in control:
        assert all(source.callback == "vent_mode_control" for source in control[database])
        assert all(source.callback == "vent_mode_seq" for source in sequence[database])
        assert [replace(source, callback=None) for source in control[database]] == [
            replace(source, callback=None) for source in sequence[database]
        ]


@pytest.mark.parametrize(
    "concept", ["vent_mode", "vent_breath_seq", "driving_pres_controlled", "tidal_vol"],
)
def test_recent_semantic_fixes_only_feed_ventilator_module(concept):
    """A new downstream consumer requires explicitly revisiting refresh scope."""
    dictionary = load_dictionary(include_sofa2=True)

    def dependency_closure(names):
        seen = set()
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            definition = dictionary.get(name)
            if definition is not None:
                pending.extend(definition.sub_concepts)
                pending.extend(definition.depends_on)
        return seen

    consumers = {
        module for module, names in EXTRACT_MODULES.items()
        if concept in dependency_closure(names)
    }
    assert consumers == {"ventilator"}
