"""Contract smoke gate for ``easyicu.io.data_load`` (A-P2-12).

``data_load`` is a supported but lightly-travelled entry point: nothing inside
EasyICU calls it, so the extraction suite never exercises it. This gate pins
the key function signatures and failure types so signature drift or fail-open
regressions are caught in CI without needing real ICU extracts.
"""

from __future__ import annotations

import inspect

import pytest

from easyicu.io import data_load
from easyicu.io.data_load import (
    TimeOriginError,
    VALID_TIME_UNITS,
    load_difftime,
    load_id,
    load_src,
    load_ts,
    load_win,
)


def test_public_functions_keep_signatures() -> None:
    assert list(inspect.signature(load_src).parameters)[:4] == ["x", "rows", "cols", "src"]
    assert "time_unit" in inspect.signature(load_difftime).parameters
    assert "id_var" in inspect.signature(load_id).parameters
    assert "index_var" in inspect.signature(load_ts).parameters
    assert "dur_var" in inspect.signature(load_win).parameters
    assert "duration_unit" in inspect.signature(load_win).parameters
    assert set(VALID_TIME_UNITS) == {"seconds", "minutes", "hours", "days"}


def test_load_src_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError):
        load_src("events", src="mimic_demo", bogus_kwarg=1)


def test_load_src_rejects_bad_first_argument() -> None:
    with pytest.raises(TypeError):
        load_src(123, src="mimic_demo")


def test_load_src_requires_src_for_table_name() -> None:
    with pytest.raises(ValueError, match="src argument required"):
        load_src("events")


def test_load_difftime_rejects_unknown_time_unit() -> None:
    with pytest.raises(ValueError, match="unknown time_unit"):
        load_difftime("events", src="mimic_demo", time_unit="fortnights")


def test_time_origin_error_is_value_error() -> None:
    assert issubclass(TimeOriginError, ValueError)
    assert issubclass(data_load.TimeOriginError, ValueError)
