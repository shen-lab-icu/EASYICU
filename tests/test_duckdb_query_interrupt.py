from __future__ import annotations

import threading
import time

import duckdb
import pytest

from easyicu import datasource


def test_duckdb_interrupt_is_mapped_to_typed_cancellation() -> None:
    error = duckdb.InterruptException("INTERRUPT Error: Interrupted!")

    with pytest.raises(datasource.DuckDBQueryInterrupted) as exc_info:
        datasource._raise_if_duckdb_interrupted(error)

    assert "interrupted" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is error


def test_non_interrupt_duckdb_error_is_not_reclassified() -> None:
    datasource._raise_if_duckdb_interrupted(
        duckdb.IOException("IO Error: No space left on device")
    )


def test_duckdb_interrupt_callback_stops_an_active_query() -> None:
    ready = threading.Event()
    outcome: dict[str, object] = {}

    def run_query() -> None:
        connection = datasource._get_duckdb_connection()
        outcome["interrupt"] = datasource.get_duckdb_interrupt_callback()
        ready.set()
        try:
            connection.execute(
                "SELECT sum(i) FROM range(1000000000000) values_table(i)"
            ).fetchone()
        except Exception as error:  # the assertion below checks the exact type
            outcome["error"] = error

    worker = threading.Thread(target=run_query, daemon=True)
    worker.start()
    assert ready.wait(timeout=2)
    time.sleep(0.05)

    interrupt = outcome["interrupt"]
    assert callable(interrupt)
    interrupt()
    worker.join(timeout=3)

    assert worker.is_alive() is False
    assert isinstance(outcome.get("error"), duckdb.InterruptException)
