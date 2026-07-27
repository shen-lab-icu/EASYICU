"""An outcome concept must not report zero events because it failed to read.

``hirid_death`` swallowed every read error into an empty result. An empty
result from a mortality concept is indistinguishable, downstream, from "nobody
in this cohort died" — so a permissions change, a renamed upstream column, a
missing parquet directory or a DuckDB error silently produced a HiRID mortality
of zero and the analysis continued on it.

The same logic exists in two places (the source-level callback and the
load-time fast path in the resolver), and both had the defect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptExtractionUnavailable
from easyicu.concept.callback_apply import _apply_callback
from easyicu.concept.schema import ConceptSource


def _source() -> ConceptSource:
    return ConceptSource(
        table="observations",
        callback="hirid_death",
        class_name="hrd_itm",
        ids=[110, 200],
        sub_var="variableid",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patientid": [1, 1, 2],
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "variableid": [110, 200, 110],
        }
    )


class _GeneralSource:
    """A data source whose ``general`` table can be told how to misbehave."""

    def __init__(self, *, general=None, error=None, base_path=None) -> None:
        self._general = general
        self._error = error
        if base_path is not None:
            self.base_path = base_path

    def load_table(self, name, columns=None, **kwargs):
        if self._error is not None:
            raise self._error
        return SimpleNamespace(data=self._general)


def _apply(data_source):
    return _apply_callback(
        _frame(),
        _source(),
        concept_name="death",
        data_source=data_source,
    )


def test_an_unreadable_general_table_raises_instead_of_reporting_no_deaths():
    source = _GeneralSource(error=PermissionError("Operation not permitted"))

    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _apply(source)

    assert excinfo.value.concept_id == "death"
    assert excinfo.value.database == "hirid"
    assert excinfo.value.stage == "load_general"
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_a_renamed_discharge_status_column_raises():
    """Upstream schema drift is the quiet version of the same failure."""

    general = pd.DataFrame({"patientid": [1, 2], "discharge_state": ["dead", "alive"]})

    with pytest.raises(ConceptExtractionUnavailable, match="discharge_status"):
        _apply(_GeneralSource(general=general))


def test_a_missing_data_source_raises():
    with pytest.raises(ConceptExtractionUnavailable, match="no data source"):
        _apply(None)


def test_a_cohort_with_no_deaths_still_returns_an_empty_result():
    """The negative control: a real zero must stay a real zero.

    Fail-closed is only useful if it distinguishes cases. A general table that
    reads cleanly and contains no deceased patient is a legitimate empty.
    """

    general = pd.DataFrame(
        {"patientid": [1, 2], "discharge_status": ["alive", "alive"]}
    )

    result = _apply(_GeneralSource(general=general))

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_deaths_that_cannot_be_timed_are_not_reported_as_no_deaths(tmp_path):
    """The general table says these patients died; the result said they did not."""

    general = pd.DataFrame({"patientid": [7, 8], "discharge_status": ["dead", "dead"]})
    # Patients 7 and 8 are deceased but appear nowhere in the observations
    # frame, so no last-observation time can be derived for either.
    source = _GeneralSource(general=general, base_path=tmp_path)

    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _apply(source)

    assert excinfo.value.stage == "last_observation"
    assert "2 patient(s) are recorded as deceased" in str(excinfo.value)


def test_a_successful_read_still_produces_the_death_rows(tmp_path):
    """The whole path still works when the source is readable."""

    general = pd.DataFrame({"patientid": [1, 2], "discharge_status": ["dead", "alive"]})
    result = _apply(_GeneralSource(general=general, base_path=tmp_path))

    assert list(result["patientid"]) == [1]
    assert bool(result["death"].iloc[0]) is True


def test_both_copies_of_the_logic_fail_closed():
    """The resolver fast path is a second implementation of the same callback.

    It was added as an optimisation over the callback above and reproduced the
    swallow. A fix to only one of them would leave the real HiRID load path —
    which takes the fast path — still reporting zero.
    """

    import inspect

    import easyicu.concept as concept_module

    source = inspect.getsource(concept_module)
    start = source.index("hirid_death 快速路径")
    fast_path = source[start : start + 4_000]

    assert (
        "ConceptExtractionUnavailable" in fast_path
    ), "the resolver fast path must fail closed too, not just the callback"
    assert (
        "dead_pids = set()" not in fast_path
    ), "swallowing the general-table read back into an empty set is the bug"
