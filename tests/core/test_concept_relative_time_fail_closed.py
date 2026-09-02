"""A mixed number/timestamp time column must not be NaN'd into silence.

The concept resolver concatenates DuckDB frames (relative hours, float64) with
non-DuckDB frames (absolute datetime64); the concat yields object dtype and the
column has to be unified onto one numeric scale. Unifying needs each stay's
``intime`` as the anchor.

The old code wrote the coerced column back unconditionally. When the anchor was
missing — no ``intime`` in the frame and an unreadable ``icustays`` table — the
timestamp rows stayed NaN and were written anyway. NaN in a time column is
indistinguishable downstream from "no measurement here", so a missing anchor
table silently became missing clinical data.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptExtractionUnavailable
from easyicu.concept.relative_time import coerce_mixed_time_column


def _mixed_frame(**extra) -> pd.DataFrame:
    """Two DuckDB rows (relative hours) and two datetime rows, as object dtype."""
    return pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2],
            "charttime": pd.Series(
                [
                    3.0,
                    5.0,
                    pd.Timestamp("2026-01-02 06:00:00"),
                    pd.Timestamp("2026-01-02 12:00:00"),
                ],
                dtype=object,
            ),
            **extra,
        }
    )


class _IcuStays:
    """An ``icustays`` source that can be told to fail or to answer."""

    def __init__(self, *, frame=None, error=None) -> None:
        self._frame = frame
        self._error = error

    def load_table(self, name, columns=None, **kwargs):
        assert name == "icustays"
        if self._error is not None:
            raise self._error
        return SimpleNamespace(data=self._frame)


def _coerce(frame, data_source=None):
    return coerce_mixed_time_column(
        frame,
        "charttime",
        concept_id="map",
        database="miiv",
        id_columns=["stay_id"],
        data_source=data_source,
    )


def test_inline_intime_anchor_converts_to_relative_hours() -> None:
    frame = _mixed_frame(
        intime=pd.to_datetime(
            [
                "2026-01-01 00:00:00",
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-02 00:00:00",
            ]
        )
    )
    _, values = _coerce(frame)
    assert list(values) == [3.0, 5.0, 6.0, 12.0]


def test_icustays_anchor_is_loaded_when_the_frame_has_no_intime() -> None:
    stays = _IcuStays(
        frame=pd.DataFrame(
            {
                "stay_id": [1, 2],
                "intime": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-02 00:00:00"]),
            }
        )
    )
    _, values = _coerce(_mixed_frame(), data_source=stays)
    assert list(values) == [3.0, 5.0, 6.0, 12.0]


def test_all_numeric_column_is_passed_through_untouched() -> None:
    frame = pd.DataFrame({"stay_id": [1, 2], "charttime": pd.Series([1.0, 2.0], dtype=object)})
    _, values = coerce_mixed_time_column(
        frame, "charttime", concept_id="map", database="miiv"
    )
    assert list(values) == [1.0, 2.0]


def test_unreadable_icustays_refuses_instead_of_writing_nan() -> None:
    stays = _IcuStays(error=PermissionError("icustays: permission denied"))
    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _coerce(_mixed_frame(), data_source=stays)
    assert excinfo.value.stage == "relative_time_anchor"
    assert excinfo.value.concept_id == "map"
    assert "permission denied" in str(excinfo.value)


def test_absent_anchor_refuses_instead_of_writing_nan() -> None:
    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _coerce(_mixed_frame(), data_source=None)
    assert excinfo.value.stage == "relative_time_anchor"


def test_partial_anchor_coverage_refuses_rather_than_dropping_those_rows() -> None:
    """Stay 2 has no ``intime``; its timestamps must not become NaN."""
    stays = _IcuStays(
        frame=pd.DataFrame(
            {"stay_id": [1], "intime": pd.to_datetime(["2026-01-01 00:00:00"])}
        )
    )
    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _coerce(_mixed_frame(), data_source=stays)
    assert "no usable 'intime' anchor" in str(excinfo.value)


def test_unparseable_value_refuses_rather_than_becoming_missing() -> None:
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": pd.Series([3.0, "not a time at all"], dtype=object),
            "intime": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        }
    )
    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _coerce(frame)
    assert "neither a number nor a parseable timestamp" in str(excinfo.value)


def test_duplicated_anchor_rows_refuse_rather_than_misaligning() -> None:
    """A duplicated stay in ``icustays`` fans the merge out.

    The mask that selects the timestamp rows is positional, so a fan-out would
    apply each anchor to the wrong row rather than fail. Refuse instead.
    """
    stays = _IcuStays(
        frame=pd.DataFrame(
            {
                "stay_id": [1, 2, 2],
                "intime": pd.to_datetime(
                    ["2026-01-01 00:00:00", "2026-01-02 00:00:00", "2026-01-03 00:00:00"]
                ),
            }
        )
    )
    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _coerce(_mixed_frame(), data_source=stays)
    assert "more than one" in str(excinfo.value)
