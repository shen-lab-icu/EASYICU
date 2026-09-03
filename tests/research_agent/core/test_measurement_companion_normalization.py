from __future__ import annotations

import duckdb
import pandas as pd

from easyicu.research_agent.case_plugins.builder import (
    _normalize_measurement_count_pair,
)
from tools.build_discovery_universe import normalize_measurement_companions


def test_discovery_universe_normalizes_left_joined_measurement_pairs() -> None:
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TABLE u(stay_id INTEGER, signal_n BIGINT, signal_measured INTEGER)"
        )
        con.execute("INSERT INTO u VALUES (1, 2, 1), (2, NULL, NULL), (3, 0, 1)")

        normalize_measurement_companions(
            con,
            table="u",
            concepts=["signal"],
        )

        assert con.execute(
            "SELECT signal_n, signal_measured FROM u ORDER BY stay_id"
        ).fetchall() == [(2, 1), (0, 0), (0, 0)]
    finally:
        con.close()


def test_case_builder_normalizes_and_derives_measurement_pair() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "signal_n_24h": [2, None, 0],
            "signal_measured_24h": [1, None, 1],
        }
    )

    _normalize_measurement_count_pair(
        frame,
        measured_column="signal_measured_24h",
        count_column="signal_n_24h",
    )

    assert frame["signal_n_24h"].tolist() == [2, 0, 0]
    assert frame["signal_measured_24h"].tolist() == [1, 0, 0]


def test_case_builder_adds_absent_measurement_pair_as_zero() -> None:
    frame = pd.DataFrame({"stay_id": [1, 2]})

    _normalize_measurement_count_pair(
        frame,
        measured_column="signal_measured_24h",
        count_column="signal_n_24h",
    )

    assert frame["signal_n_24h"].tolist() == [0, 0]
    assert frame["signal_measured_24h"].tolist() == [0, 0]
