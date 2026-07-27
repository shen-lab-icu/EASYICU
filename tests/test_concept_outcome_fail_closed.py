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


def _apply(data_source, *, frame=None, patient_ids=None):
    return _apply_callback(
        _frame() if frame is None else frame,
        _source(),
        concept_name="death",
        patient_ids=patient_ids,
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


def test_a_cohort_of_survivors_is_zero_even_when_the_database_records_deaths(tmp_path):
    """The guard has to be about *this* cohort, not about the whole source.

    A guard written against every death in the database answers a question
    nobody asked: for a cohort of survivors it reports a failure while the
    correct answer, zero, was sitting right there. Asking about one living
    patient in a database where somebody else died must return an empty
    result, not raise.
    """

    general = pd.DataFrame({"patientid": [1, 2], "discharge_status": ["alive", "dead"]})
    frame = pd.DataFrame(
        {"patientid": [1, 1], "datetime": [10, 20], "variableid": [110, 200]}
    )

    result = _apply(
        _GeneralSource(general=general, base_path=tmp_path),
        frame=frame,
        patient_ids=[1],
    )

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_a_dead_patient_in_the_cohort_still_fails_closed(tmp_path):
    """The other half of the same narrowing: it must not weaken the guard."""

    general = pd.DataFrame({"patientid": [1, 2], "discharge_status": ["alive", "dead"]})
    # Patient 2 died but has no observation to time the death.
    frame = pd.DataFrame(
        {"patientid": [1, 1], "datetime": [10, 20], "variableid": [110, 200]}
    )

    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _apply(
            _GeneralSource(general=general, base_path=tmp_path),
            frame=frame,
            patient_ids=[2],
        )

    assert excinfo.value.stage == "last_observation"
    assert "1 patient(s) in this cohort" in str(excinfo.value)


def test_a_dict_cohort_selector_narrows_the_same_way(tmp_path):
    """`patient_ids` also arrives as a {id_column: ids} mapping."""

    general = pd.DataFrame({"patientid": [1, 2], "discharge_status": ["alive", "dead"]})
    frame = pd.DataFrame(
        {"patientid": [1, 1], "datetime": [10, 20], "variableid": [110, 200]}
    )

    result = _apply(
        _GeneralSource(general=general, base_path=tmp_path),
        frame=frame,
        patient_ids={"patientid": [1]},
    )

    assert result.empty


def test_deaths_dropped_for_want_of_a_timestamp_fail_closed(tmp_path):
    """Partial loss is the same defect as total loss, only smaller.

    A death the extraction cannot time does not appear in the result, so the
    mortality computed from it is lower than the source says and nothing
    downstream can see that a number went missing. An earlier version warned
    instead, on the assumption that a patient could legitimately lack the
    observation that times the death. Measured against the real HiRID export
    that assumption is false — all 2,062 recorded deaths are timeable from
    variables 110/200 — so a shortfall is a fault, and raising costs a correct
    run nothing. A RuntimeWarning also cannot be read by any gate: it reaches
    stderr and no caller.
    """

    general = pd.DataFrame({"patientid": [1, 2], "discharge_status": ["dead", "dead"]})
    # Patient 1 can be timed; patient 2 cannot.
    frame = pd.DataFrame(
        {"patientid": [1, 1], "datetime": [10, 20], "variableid": [110, 200]}
    )

    with pytest.raises(ConceptExtractionUnavailable) as excinfo:
        _apply(_GeneralSource(general=general, base_path=tmp_path), frame=frame)

    assert excinfo.value.stage == "last_observation"
    message = str(excinfo.value)
    assert "1 of 2 recorded deaths" in message
    # The undercount it refused to report, stated as a number.
    assert "mortality of 1/2" in message


def test_an_explicitly_empty_cohort_is_empty_not_everybody(tmp_path):
    """Only absence means "all patients"; `[]` means nobody.

    Collapsing an empty selector into "no filter" makes every guard below
    answer for the whole database when the caller asked about nobody — the
    same mistake as guarding on every death in the source. The package's own
    normalizers already keep `None` and `[]` apart, so this one must too.
    """

    from easyicu.concept.callback_apply import (
        cohort_patient_ids,
        deaths_within_cohort,
    )

    assert cohort_patient_ids(None) is None
    assert cohort_patient_ids([]) == set()
    assert cohort_patient_ids({}) == set()
    assert cohort_patient_ids({"patientid": []}) == set()
    assert cohort_patient_ids({"patientid": [1]}) == {1}

    # The consequence that matters: a death in the database is not a death in
    # an empty cohort.
    assert deaths_within_cohort({2}, []) == set()
    assert deaths_within_cohort({2}, None) == {2}


def test_a_dict_holding_no_ids_selects_nothing_rather_than_everybody():
    """`{col: None}` is refused, because it has no settled meaning.

    Reading it as "all" widens a filtered request to the whole database;
    reading it as "none" empties it. Neither reference normalizer accepts it —
    both reach `list(None)` and raise — so a helper that quietly answered
    "everybody" would be the single place in the package where this shape
    acquires a population.
    """

    import pytest

    from easyicu.concept.callback_apply import (
        cohort_patient_ids,
        deaths_within_cohort,
    )

    with pytest.raises(ValueError, match="does not select a cohort"):
        cohort_patient_ids({"patientid": None})
    with pytest.raises(ValueError, match="does not select a cohort"):
        deaths_within_cohort({2}, {"patientid": None})

    # The refusal is specific to the missing ids, not to the dict spelling.
    assert cohort_patient_ids({"patientid": [2]}) == {2}


def test_an_empty_cohort_returns_empty_rather_than_failing_on_someone_elses_death(
    tmp_path,
):
    """The end-to-end consequence of the selector fix."""

    general = pd.DataFrame({"patientid": [1, 2], "discharge_status": ["alive", "dead"]})
    # Patient 2 died and cannot be timed — which would raise for a cohort that
    # contained them, and must not for a cohort that contains nobody.
    frame = pd.DataFrame(
        {"patientid": [1, 1], "datetime": [10, 20], "variableid": [110, 200]}
    )

    result = _apply(
        _GeneralSource(general=general, base_path=tmp_path),
        frame=frame,
        patient_ids=[],
    )

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
    assert "2 patient(s) in this cohort are recorded as deceased" in str(excinfo.value)


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
    assert "deaths_within_cohort" in fast_path, (
        "the fast path must narrow to the cohort through the same shared helper, "
        "or the two copies drift on which population their guards are about"
    )
    assert "_refuse_untimed_deaths" in fast_path, (
        "a death the query cannot time silently lowers the mortality reported, "
        "and both copies must refuse it through the same helper"
    )
    assert "_warn_untimed_deaths" not in fast_path, (
        "warning was the old behaviour: it reaches stderr, not the caller, so "
        "nothing downstream can act on the shortfall"
    )
