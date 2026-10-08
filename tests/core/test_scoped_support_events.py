"""Invented temporary Parquet only. Never access a clinical export or roster."""

from pathlib import Path
import pandas as pd
import pytest

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.io import scoped_support_events as api
from easyicu.io.source_events import SourceEventContractError

ORIGIN = pd.Timestamp("2180-01-01")


def stamp(hour):
    return None if hour is None else str(ORIGIN + pd.Timedelta(hours=hour))


def event(**kwargs):
    row = dict(
        subject_id=1,
        hadm_id=10,
        stay_id=100,
        itemid=225792,
        starttime=stamp(1),
        endtime=stamp(3),
        storetime=stamp(4),
        value=120.0,
        valueuom="min",
        statusdescription="FinishedRunning",
        orderid=50,
        linkorderid=40,
        continueinnextdept=0,
    )
    row.update(kwargs)
    return row


def source(tmp_path, rows=None, *, clocks=None, nested=False):
    clocks = clocks or [
        (1, 10, 100, stamp(0), stamp(20)),
        (1, 10, 101, stamp(22), None),
        (2, 20, 200, stamp(0), stamp(20)),
    ]
    folder = tmp_path / "icu" if nested else tmp_path
    folder.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(
        clocks, columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
    ).to_parquet(folder / "icustays.parquet", index=False)
    pd.DataFrame([event()] if rows is None else rows).to_parquet(
        folder / "procedureevents.parquet", index=False
    )
    return ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)


def extract(tmp_path, rows=None, allowed=(100, 101), **kwargs):
    return api.extract_miiv_support_events(
        source(tmp_path, rows, **kwargs), allowed_stay_ids=allowed
    )


def evidence(result, cutoff=5):
    return api.support_window_evidence(
        result.events,
        window_start=stamp(1),
        window_end=stamp(2),
        decision_time=stamp(cutoff),
    )


def test_exact_items_native_identity_duplicates_and_zero_event_coverage(tmp_path):
    rows = [
        event(),
        event(),
        event(itemid=225794),
        event(stay_id=200, subject_id=2, hadm_id=20),
        event(itemid=999),
        event(subject_id=999),
    ]
    r = extract(tmp_path, rows)
    assert len(r.events) == 3 and r.events.event_key.nunique() == 3
    assert set(r.events.stay_id) == {100}
    assert set(r.events.support_type) == {"invasive", "noninvasive"}
    assert r.clock_context.stay_id.tolist() == [100, 101]
    assert r.clock_context.outtime_status.tolist() == ["known", "unknown"]
    assert r.events.duration_value_minus_interval_hours.eq(0).all()
    assert r.receipt["source_filter_applied_before_python"]
    assert sum(x["count"] for x in r.receipt["source_filter_counts"]) == 4
    assert evidence(r).completed_window_record_evidence.all()


@pytest.mark.parametrize("nested", [False, True])
def test_flat_and_icu_single_file_layout(tmp_path, nested):
    assert len(extract(tmp_path, nested=nested).events) == 1


@pytest.mark.parametrize(
    "end,status", [(100, "Rewritten"), (None, None), (0, "Cancelled"), (3, "unknown")]
)
def test_start_claim_invariant_to_future_end_and_final_status(tmp_path, end, status):
    r = extract(
        tmp_path,
        [event(endtime=stamp(end), statusdescription=status, storetime=stamp(2))],
    )
    x = r.events.iloc[0]
    assert x.recorded_start_claim_at == pd.Timestamp(stamp(2))
    assert evidence(r, 3).recorded_start_claim_visible.all()
    assert not evidence(r, 3).completed_window_record_evidence.any()


@pytest.mark.parametrize(
    "store,visible",
    [(3, True), (4, True), (5, False), (2, False), (None, False), (0, False)],
)
def test_completed_requires_store_after_end_and_strict_cutoff(tmp_path, store, visible):
    r = extract(tmp_path, [event(storetime=stamp(store))])
    assert bool(evidence(r).completed_window_record_evidence.iloc[0]) == visible


@pytest.mark.parametrize("status", ["Cancelled", "Rewritten", "unknown", None])
def test_final_nonaffirmative_status_never_proves_no_support(tmp_path, status):
    r = extract(tmp_path, [event(statusdescription=status)])
    e = evidence(r)
    assert e.completed_interval_record_visible.all()
    assert e.support_evidence_status.tolist() == ["unknown"]
    assert "availability_certified" not in r.events


@pytest.mark.parametrize("name", ["starttime", "endtime", "storetime"])
@pytest.mark.parametrize(
    "value,status",
    [(None, "missing"), ("2180-01-01 01:00:00 UTC", "invalid"), ("bad", "invalid")],
)
def test_invalid_event_clocks_retained(tmp_path, name, value, status):
    r = extract(tmp_path, [event(**{name: value})])
    assert len(r.events) == 1 and r.events.iloc[0][name + "_status"] == status
    assert not evidence(r).completed_window_record_evidence.any()


@pytest.mark.parametrize(
    "end,expected", [(1, "zero_length"), (0, "reversed"), (None, "missing_end")]
)
def test_nonpositive_unknown_intervals_not_removed(tmp_path, end, expected):
    r = extract(tmp_path, [event(endtime=stamp(end))])
    assert r.events.interval_status.tolist() == [expected]
    assert not evidence(r).completed_window_record_evidence.any()


def test_no_selected_event_preserves_schema_and_full_roster(tmp_path):
    r = extract(tmp_path, [event(itemid=999)])
    assert r.events.empty and len(r.clock_context) == 2
    assert evidence(r).empty
    assert "recorded_start_claim_at" in r.events


@pytest.mark.parametrize("allowed", [[], [100, 100], [True], [100.5], [0], [-1], [999]])
def test_invalid_identity_request_fails(tmp_path, allowed):
    with pytest.raises(SourceEventContractError):
        extract(tmp_path, allowed=allowed)


def test_unknown_exit_kept_bad_nonempty_exit_rejected(tmp_path):
    with pytest.raises(
        SourceEventContractError, match="invalid_requested_native_clock_context"
    ):
        extract(
            tmp_path,
            clocks=[
                (1, 10, 100, stamp(0), "2180-01-01 20:00:00 UTC"),
                (1, 10, 101, stamp(22), None),
            ],
        )


def test_native_assignment_does_not_depend_on_sibling_exit_ties(tmp_path):
    r = extract(
        tmp_path,
        clocks=[(1, 10, 100, stamp(0), stamp(20)), (1, 10, 101, stamp(0), stamp(20))],
    )
    assert len(r.events) == 1 and len(r.clock_context) == 2


@pytest.mark.parametrize("unit", [None, "None", "mystery"])
def test_unit_unknown_preserved_no_duration_imputation(tmp_path, unit):
    r = extract(tmp_path, [event(valueuom=unit)])
    assert r.events.duration_value_hours.isna().all()
    assert r.events.duration_hours.tolist() == [2]


def test_missing_optional_columns_explicit_in_receipt(tmp_path):
    row = event()
    for col in api.OPTIONAL_COLUMNS:
        row.pop(col)
    r = extract(tmp_path, [row])
    assert not any(r.receipt["optional_source_columns"].values())
    assert r.events[list(api.OPTIONAL_COLUMNS)].isna().all().all()


def test_invalid_window_rejected(tmp_path):
    r = extract(tmp_path)
    with pytest.raises(SourceEventContractError):
        api.support_window_evidence(
            r.events, window_start=stamp(2), window_end=stamp(4), decision_time=stamp(3)
        )
