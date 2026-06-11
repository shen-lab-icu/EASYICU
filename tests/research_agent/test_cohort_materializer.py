"""Unit tests for the cohort materializer's deterministic core.

These do not touch a real database; they exercise the per-stay summarisation,
windowing, predicate-column, binary-outcome helpers, and the CTAS 纳排
integration on synthetic frames so the logic is covered in CI.
"""
import pandas as pd
import pytest

from easyicu.research_agent import cohort_materializer as M
from easyicu.research_agent.cohort_schema import CohortDefinition, build_cohort


def test_summarize_timeseries_basic():
    df = pd.DataFrame(
        {"stay_id": [1, 1, 2], "charttime": [1, 5, 2], "lact": [3.0, 5.0, 1.0]}
    )
    out = M._summarize_timeseries(df, "lact", (0.0, 24.0))
    r1 = out[out.stay_id == 1].iloc[0]
    assert r1.lact_max == 5.0
    assert r1.lact_min == 3.0
    assert r1.lact_first == 3.0
    assert r1.lact_n == 2
    assert r1.lact_measured == 1


def test_window_excludes_out_of_window_records():
    df = pd.DataFrame({"stay_id": [1, 1], "charttime": [1, 30], "lact": [3.0, 9.0]})
    out = M._summarize_timeseries(df, "lact", (0.0, 24.0))
    # the 30h record is outside the first-24h window and must not count
    assert out[out.stay_id == 1].iloc[0].lact_max == 3.0


def test_predicate_column_uses_declared_aggregation():
    df = pd.DataFrame({"stay_id": [1, 1, 2], "charttime": [1, 2, 1], "sofa": [2, 7, 1]})
    out = M._predicate_column(df, "sofa", (0.0, 24.0), "max")
    assert set(out.columns) == {"stay_id", "sofa"}
    assert out[out.stay_id == 1].iloc[0].sofa == 7


def test_predicate_column_rejects_unknown_aggregation():
    df = pd.DataFrame({"stay_id": [1], "charttime": [1], "sofa": [2]})
    with pytest.raises(ValueError, match="unsupported cohort predicate aggregation"):
        M._predicate_column(df, "sofa", (0.0, 24.0), "mode")


def test_predicate_column_supports_last_aggregation():
    df = pd.DataFrame({"stay_id": [1, 1], "charttime": [2, 1], "sofa": [7, 2]})
    first = M._predicate_column(df, "sofa", (0.0, 24.0), "first")
    out = M._predicate_column(df, "sofa", (0.0, 24.0), "last")
    assert first[first.stay_id == 1].iloc[0].sofa == 2
    assert out[out.stay_id == 1].iloc[0].sofa == 7


def test_predicate_column_all_requires_observed_truthy_values():
    df = pd.DataFrame(
        {"stay_id": [1, 1, 2, 2], "charttime": [1, 2, 1, 2], "flag": [None, None, "1", "true"]}
    )
    out = M._predicate_column(df, "flag", (0.0, 24.0), "all")
    got = dict(zip(out.stay_id.tolist(), out.flag.tolist()))
    assert got == {1: False, 2: True}


def test_binary_event_column_is_whole_stay():
    # a death at 200h must still count as 1 (not windowed away)
    df = pd.DataFrame({"stay_id": [1, 3], "charttime": [200, 5], "death": [True, True]})
    out = M._binary_event_column(df, "death")
    assert set(out.stay_id) == {1, 3}
    assert out.death.tolist() == [1, 1]


def test_binary_event_column_respects_zero_one_values():
    df = pd.DataFrame({"stay_id": [1, 2, 2], "death": [0, 0, 1]})
    out = M._binary_event_column(df, "death")
    got = dict(zip(out.stay_id.tolist(), out.death.tolist()))
    assert got == {1: 0, 2: 1}


def test_binary_event_column_respects_numeric_strings():
    df = pd.DataFrame({"stay_id": [1, 2, 2, 3], "death": ["0.0", "0", "1.0", "false"]})
    out = M._binary_event_column(df, "death")
    got = dict(zip(out.stay_id.tolist(), out.death.tolist()))
    assert got == {1: 0, 2: 1, 3: 0}


def test_binary_event_column_treats_event_rows_without_value_as_positive():
    df = pd.DataFrame({"stay_id": [1, 3], "charttime": [200, 5]})
    out = M._binary_event_column(df, "death")
    assert dict(zip(out.stay_id.tolist(), out.death.tolist())) == {1: 1, 3: 1}


def test_ctas_inclusion_filters_cohort():
    wide = pd.DataFrame({"stay_id": [1, 2, 3], "age": [40, 10, 80], "lact": [5.0, 9.0, 1.0]})
    cdef = CohortDefinition.from_dict(
        {
            "name": "adult_hyperlactatemia",
            "inclusion": [
                {
                    "concept_id": "age",
                    "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},
                    "aggregation": "max",
                    "op": ">=",
                    "value": 18,
                },
                {
                    "concept_id": "lact",
                    "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},
                    "aggregation": "max",
                    "op": ">=",
                    "value": 2,
                },
            ],
            "exclusion": [],
        }
    )
    out = build_cohort(cdef, wide)
    # stay 2 excluded (age 10 < 18); stay 3 excluded (lact 1 < 2); only stay 1 remains
    assert out.stay_id.tolist() == [1]


def test_ctas_builder_accepts_schema_declared_aggregations_on_materialised_columns():
    wide = pd.DataFrame({"stay_id": [1, 2], "lact": [3.0, 1.0]})
    cdef = CohortDefinition.from_dict(
        {
            "name": "mean_lactate_gate",
            "inclusion": [
                {
                    "concept_id": "lact",
                    "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},
                    "aggregation": "mean",
                    "op": ">=",
                    "value": 2,
                },
            ],
            "exclusion": [],
        }
    )
    out = build_cohort(cdef, wide)
    assert out.stay_id.tolist() == [1]
