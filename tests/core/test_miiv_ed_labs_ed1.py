"""ED1/ED2 lab attribution regressions, using small invented records only."""

import pandas as pd
import pytest

from tests.support.bucket_storage import write_fixture_parquet

from easyicu.config import load_src_cfg
from easyicu.datasource import (
    FilterOp,
    FilterSpec,
    ICUDataSource,
    _close_duckdb_connections,
    _get_duckdb_connection,
    _lab_query,
    finish_miiv_lab_frame,
    load_bucketed_table_aggregated,
    load_bucketed_table_multi_aggregated,
)


@pytest.fixture(autouse=True)
def bounded_duckdb(monkeypatch, tmp_path):
    monkeypatch.setenv("EASYICU_DUCKDB_THREADS", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_MEMORY_LIMIT_GB", "1")
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(tmp_path / "spill"))
    yield
    _close_duckdb_connections()


def _stays():
    # Deliberately reverse stay order; the second subject shares the clock.
    return pd.DataFrame({
        "subject_id": [1, 1, 2], "hadm_id": [12, 11, 21], "stay_id": [102, 101, 201],
        "intime": pd.to_datetime(["2140-01-03", "2140-01-02", "2140-01-02"]),
        "outtime": pd.to_datetime(["2140-01-04", "2140-01-03", "2140-01-03"]),
    })


def _labs(rows):
    frame = pd.DataFrame(rows, columns=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"])
    frame["charttime"] = pd.to_datetime(frame.charttime, format="ISO8601")
    frame["hadm_id"] = frame.hadm_id.astype("Int64")
    frame["valueuom"] = "mmol/L"
    return frame


def _source(tmp_path, stays, labs, path, admissions=None):
    stays.to_parquet(tmp_path / "icustays.parquet", index=False)
    if admissions is not None:
        admissions.to_parquet(tmp_path / "admissions.parquet", index=False)
    directory = tmp_path / "labevents_bucket" / "bucket_id=0"
    directory.mkdir(parents=True)
    write_fixture_parquet(labs, directory / "part.parquet", index=False)
    source = ICUDataSource(load_src_cfg("miiv"), base_path=tmp_path, enable_cache=False)
    if path == "preloaded":
        source._preloaded_tables["labevents"] = labs
        if admissions is not None:
            source._preloaded_tables["admissions"] = admissions
    return source


def _load(source, path, targets=None):
    if path in {"source", "preloaded"}:
        filters = [] if targets is None else [FilterSpec("stay_id", FilterOp.IN, targets)]
        frame = source.load_table("labevents", columns=["valuenum"], filters=filters).data
        starts = source.load_table("icustays").data.set_index("stay_id").intime
        hours = (frame.charttime - frame.stay_id.map(starts)) / pd.Timedelta(hours=1)
        frame = frame.assign(charttime=hours)
    elif path == "bucket":
        frame = load_bucketed_table_aggregated(
            source, "labevents", "valuenum", [50813], patient_ids=targets, agg_func="sum",
        )
    else:
        frame = load_bucketed_table_multi_aggregated(
            source, "labevents", {"lact": [50813], "alb": [50862]},
            patient_ids=targets, agg_func="sum",
        ).rename(columns={"lact": "valuenum"})
    return frame


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
@pytest.mark.parametrize("targets", [None, [101]])
def test_ed_lab_reaches_every_loader_without_cross_subject_or_window_leak(tmp_path, path, targets):
    labs = _labs([
        (1, None, "2140-01-01 12:00", 50813, 1.25),
        (1, 11, "2140-01-02 01:00", 50813, 2.),
        (1, None, "2139-12-31 23:59:59", 50813, 90.),
        (1, None, "2140-01-05 00:00:01", 50813, 91.),
        (99, None, "2140-01-01 12:00", 50813, 92.),
    ])
    source = _source(tmp_path, _stays(), labs, path)
    frame = _load(source, path, targets)
    assert list(frame[["stay_id", "charttime", "valuenum"]].itertuples(index=False, name=None)) == [
        (101, -12., 1.25), (101, 1., 2.),
    ]


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
@pytest.mark.parametrize("targets", [None, [101], [102]])
@pytest.mark.parametrize("same_admission", [False, True])
def test_overlapping_windows_roll_on_outtime_before_target_filter(tmp_path, path, targets, same_admission):
    stays = _stays()
    if same_admission:
        stays.loc[stays.subject_id.eq(1), "hadm_id"] = 11
    labs = _labs([
        (1, None, "2140-01-02 12:00", 50813, 1.),
        (1, None, "2140-01-03 00:00", 50813, 2.),
        (1, None, "2140-01-03 00:00:01", 50813, 3.),
        (1, None, "2140-01-04 00:00", 50813, 4.),
    ])
    source = _source(tmp_path, stays, labs, path)
    frame = _load(source, path, targets)
    expected = [(101, 1.), (101, 2.), (102, 3.), (102, 4.)]
    if targets is not None:
        expected = [row for row in expected if row[0] in targets]
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == expected
    assert frame.attrs["easyicu_miiv_lab_attribution"]["ambiguous_null_hadm_rows"] == 4


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_null_hadm_window_edges_and_post_discharge_rollend(tmp_path, path):
    labs = _labs([
        (1, None, "2140-01-01 00:00", 50813, 1.),
        (1, None, "2140-01-05 00:00", 50813, 2.),
    ])
    source = _source(tmp_path, _stays(), labs, path)
    frame = _load(source, path)
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == [(101, 1.), (102, 2.)]


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_overlapping_post_discharge_windows_roll_to_the_last_outtime(tmp_path, path):
    stays = _stays()
    stays.loc[stays.stay_id.eq(102), "outtime"] = pd.Timestamp("2140-01-03 12:00")
    labs = _labs([(1, None, "2140-01-03 18:00", 50813, 1.)])
    source = _source(tmp_path, stays, labs, path)
    frame = _load(source, path)
    assert frame.stay_id.tolist() == [102]
    assert frame.valuenum.tolist() == [1.]


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_equal_outtimes_keep_the_existing_lowest_stay_id_tiebreak(tmp_path, path):
    stays = _stays()
    stays.loc[stays.stay_id.eq(102), "outtime"] = pd.Timestamp("2140-01-03")
    source = _source(tmp_path, stays, _labs([(1, None, "2140-01-02 12:00", 50813, 1.)]), path)
    frame = _load(source, path)
    assert frame.stay_id.tolist() == [101]


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_other_subject_cannot_win_an_earlier_outtime(tmp_path, path):
    stays = _stays()
    stays.loc[stays.stay_id.eq(201), "outtime"] = pd.Timestamp("2140-01-02 18:00")
    labs = _labs([
        (1, None, "2140-01-02 12:00", 50813, 1.),
        (2, 21, "2140-01-02 12:00", 50813, 2.),
    ])
    source = _source(tmp_path, stays, labs, path)
    frame = _load(source, path)
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == [(101, 1.), (201, 2.)]


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_equal_independent_null_labs_keep_multiplicity_in_an_overlap(tmp_path, path):
    labs = _labs([(1, None, "2140-01-02 12:00", 50813, 1.)] * 2)
    source = _source(tmp_path, _stays(), labs, path)
    frame = _load(source, path)
    assert set(frame.stay_id) == {101}
    assert frame.valuenum.sum() == 2.
    assert frame.attrs["easyicu_miiv_lab_attribution"]["attributed_null_hadm_rows"] == 2


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_admission_linked_labs_keep_their_original_assignment_without_null_fallback(tmp_path, path):
    labs = _labs([
        (1, 11, "2139-12-30", 50813, 1.),  # Existing hadm rows have no new window clamp.
        (1, 11, "2140-01-04 12:00", 50813, 2.),  # Keep admission 11, even nearer stay 102.
        (1, 12, "2140-01-02 12:00", 50813, 3.),
        (1, 999, "2140-01-02 12:00", 50813, 90.),  # Non-NULL unknown admission never falls back.
        (2, 11, "2140-01-02 12:00", 50813, 91.),  # Nor a different subject's admission.
    ])
    source = _source(tmp_path, _stays(), labs, path)
    frame = _load(source, path)
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == [(101, 1.), (101, 2.), (102, 3.)]


@pytest.mark.parametrize("path", ["source", "preloaded"])
@pytest.mark.parametrize("target_hadm, expected", [(11, [1.]), (12, [])])
def test_hadm_pushdown_retains_null_labs_but_filters_after_subject_attribution(tmp_path, path, target_hadm, expected):
    labs = _labs([(1, None, "2140-01-02 12:00", 50813, 1.)])
    source = _source(tmp_path, _stays(), labs, path)
    frame = source.load_table("labevents", filters=[FilterSpec("hadm_id", FilterOp.IN, [target_hadm])]).data
    assert frame.valuenum.tolist() == expected
    if expected:
        assert frame.hadm_id.isna().all()
        assert frame.attributed_hadm_id.tolist() == [11]


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_rolling_null_lab_still_deduplicates_only_against_its_assigned_admission(tmp_path, path):
    labs = _labs([
        (1, None, "2140-01-02 12:00", 50813, 1.),
        (1, 11, "2140-01-02 12:00", 50813, 1.),
        (1, 12, "2140-01-02 12:00", 50813, 1.),
    ])
    source = _source(tmp_path, _stays(), labs, path)
    frame = _load(source, path)
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == [(101, 1.), (102, 1.)]
    assert frame.attrs["easyicu_miiv_lab_attribution"]["deduplicated_null_hadm_rows"] == 1


def test_multi_concept_sql_recovers_each_lab_in_an_overlapping_panel(tmp_path):
    labs = _labs([
        (1, None, "2140-01-02 12:00", 50813, 1.),
        (1, None, "2140-01-02 12:00", 50862, 3.),
    ])
    source = _source(tmp_path, _stays(), labs, "multi")
    frame = _load(source, "multi", [101])
    assert frame.stay_id.tolist() == [101]
    assert frame.valuenum.tolist() == [1.]
    assert frame.alb.tolist() == [3.]


def _admissions(rows):
    # Keep source timestamps as strings: the shared SQL owner must TRY_CAST.
    return pd.DataFrame(rows, columns=["subject_id", "hadm_id", "admittime", "dischtime"])


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
@pytest.mark.parametrize("targets", [None, [101], [102]])
def test_admission_intervals_restrict_null_labs_before_target_selection(tmp_path, path, targets):
    stays = pd.concat([_stays(), pd.DataFrame({
        "subject_id": [1], "hadm_id": [14], "stay_id": [104],
        "intime": pd.to_datetime(["2140-01-10"]), "outtime": pd.to_datetime(["2140-01-11"]),
    })], ignore_index=True)
    admissions = _admissions([
        (1, 11, "2140-01-01 18:00", "2140-01-02 08:00"),
        (1, 12, "2140-01-02 10:00", "2140-01-04 06:00"),
        (1, 13, "2140-01-04 08:00", "2140-01-05 00:00"),  # No ICU stay.
        (1, 14, "2140-01-01 00:00", "2140-01-01 06:00"),  # ICU window misses the lab.
        (2, 21, "2140-01-01 00:00", "2140-01-05 00:00"),  # A different subject cannot restrict.
    ])
    labs = _labs([
        (1, None, "2140-01-01 12:00", 50813, 1.),  # No admission: accepted ED1 rule.
        (1, None, "2140-01-02 06:00", 50813, 2.),  # Candidate's own admission.
        (1, None, "2140-01-02 12:00", 50813, 3.),  # Another admission's ICU stay wins.
        (1, 11, "2140-01-02 13:00", 50813, 4.),  # Non-NULL linkage still takes precedence.
        (1, None, "2140-01-04 09:00", 50813, 90.),
        (1, None, "2140-01-01 03:00", 50813, 91.),
    ])
    frame = _load(_source(tmp_path, stays, labs, path, admissions), path, targets)
    expected = [(101, -12., 1.), (101, 6., 2.), (101, 13., 4.), (102, -12., 3.)]
    if targets is not None:
        expected = [row for row in expected if row[0] in targets]
    assert list(frame[["stay_id", "charttime", "valuenum"]].itertuples(index=False, name=None)) == expected
    assert frame.attrs["easyicu_miiv_lab_attribution"] == {
        "ambiguous_null_hadm_rows": 2, "unmatched_null_hadm_rows": 0,
        "deduplicated_null_hadm_rows": 0, "attributed_null_hadm_rows": 3, "hadm_rows": 1,
        "null_hadm_rows_in_other_admission_reassigned": 1,
        "null_hadm_rows_in_other_admission_dropped": 2,
        "window_before_hours": 24, "window_after_hours": 24,
    }


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
@pytest.mark.parametrize("targets", [None, [101], [102]])
def test_overlapping_admissions_roll_only_among_their_icu_windows(tmp_path, path, targets):
    stays = _stays()
    stays.loc[stays.stay_id.eq(102), "outtime"] = pd.Timestamp("2140-01-03 12:00")
    stays = pd.concat([stays, pd.DataFrame({
        "subject_id": [1], "hadm_id": [13], "stay_id": [103],
        "intime": pd.to_datetime(["2140-01-02 06:00"]),
        "outtime": pd.to_datetime(["2140-01-02 18:00"]),
    })], ignore_index=True)
    admissions = _admissions([
        (1, 11, "2140-01-02 10:00", "2140-01-04 00:00"),
        (1, 12, "2140-01-02 11:00", "2140-01-04 00:00"),
        (1, 11, "2140-01-02 10:00", "2140-01-04 00:00"),  # Duplicate interval cannot duplicate a lab.
    ])
    labs = _labs([
        (1, None, "2140-01-02 12:00", 50813, 1.),
        (1, None, "2140-01-03 00:00", 50813, 2.),
        (1, None, "2140-01-03 00:00:01", 50813, 3.),
        (1, None, "2140-01-03 18:00", 50813, 4.),
    ])
    frame = _load(_source(tmp_path, stays, labs, path, admissions), path, targets)
    expected = [(101, 1.), (101, 2.), (102, 3.), (102, 4.)]
    if targets is not None:
        expected = [row for row in expected if row[0] in targets]
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == expected
    audit = frame.attrs["easyicu_miiv_lab_attribution"]
    assert audit["ambiguous_null_hadm_rows"] == 4
    assert audit["null_hadm_rows_in_other_admission_reassigned"] == 1
    assert audit["null_hadm_rows_in_other_admission_dropped"] == 0


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_admission_bounds_are_inclusive_and_neighbouring_seconds_keep_ed_rule(tmp_path, path):
    admissions = _admissions([(1, 12, "2140-01-02 12:00", "2140-01-02 13:00")])
    labs = _labs([
        (1, None, "2140-01-02 11:59:59", 50813, 1.),
        (1, None, "2140-01-02 12:00:00", 50813, 2.),
        (1, None, "2140-01-02 13:00:00", 50813, 3.),
        (1, None, "2140-01-02 13:00:01", 50813, 4.),
    ])
    frame = _load(_source(tmp_path, _stays(), labs, path, admissions), path)
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == [
        (101, 1.), (101, 4.), (102, 2.), (102, 3.),
    ]
    assert frame.attrs["easyicu_miiv_lab_attribution"]["null_hadm_rows_in_other_admission_reassigned"] == 2


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_other_admission_drop_is_counted_even_when_no_result_survives(tmp_path, path):
    admissions = _admissions([(1, 99, "2140-01-02 00:00", "2140-01-02 23:00")])
    labs = _labs([(1, None, "2140-01-02 12:00", 50813, 1.)])
    frame = _load(_source(tmp_path, _stays(), labs, path, admissions), path)
    assert frame.empty
    audit = frame.attrs["easyicu_miiv_lab_attribution"]
    assert audit["null_hadm_rows_in_other_admission_dropped"] == 1
    assert audit["null_hadm_rows_in_other_admission_reassigned"] == 0
    assert audit["attributed_null_hadm_rows"] == audit["unmatched_null_hadm_rows"] == 0


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_reassigned_null_lab_deduplicates_against_the_admission_that_contains_it(tmp_path, path):
    admissions = _admissions([(1, 12, "2140-01-02 00:00", "2140-01-04 00:00")])
    labs = _labs([
        (1, None, "2140-01-02 12:00", 50813, 1.),
        (1, 11, "2140-01-02 12:00", 50813, 1.),
        (1, 12, "2140-01-02 12:00", 50813, 1.),
    ])
    frame = _load(_source(tmp_path, _stays(), labs, path, admissions), path)
    assert list(frame[["stay_id", "valuenum"]].itertuples(index=False, name=None)) == [(101, 1.), (102, 1.)]
    audit = frame.attrs["easyicu_miiv_lab_attribution"]
    assert audit["deduplicated_null_hadm_rows"] == 1
    assert audit["null_hadm_rows_in_other_admission_reassigned"] == 1


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
@pytest.mark.parametrize("storage", ["preloaded", "registered"])
def test_admission_restriction_uses_the_source_table_without_a_disk_copy(tmp_path, path, storage):
    admissions = _admissions([(1, 12, "2140-01-02 00:00", "2140-01-04 00:00")])
    labs = _labs([(1, None, "2140-01-02 12:00", 50813, 1.)])
    source = _source(tmp_path, _stays(), labs, path)
    if storage == "preloaded":
        source._preloaded_tables["admissions"] = admissions
    else:
        source.register_table_source("admissions", lambda: admissions.copy())
    frame = _load(source, path)
    assert frame.stay_id.tolist() == [102]
    assert frame.valuenum.tolist() == [1.]


@pytest.mark.parametrize("path", ["source", "preloaded", "bucket", "multi"])
def test_invalid_or_incomplete_admission_intervals_do_not_claim_a_null_lab(tmp_path, path):
    admissions = _admissions([
        (1, 96, "2140-01-02 00:00", "unrecorded"),
        (1, 97, "unrecorded", "2140-01-04 00:00"),
        (1, 98, "2140-01-02 00:00", None),
        (1, 99, "2140-01-04 00:00", "2140-01-02 00:00"),
    ])
    labs = _labs([(1, None, "2140-01-02 12:00", 50813, 1.)])
    frame = _load(_source(tmp_path, _stays(), labs, path, admissions), path)
    assert frame.stay_id.tolist() == [101]
    assert frame.valuenum.tolist() == [1.]
    assert frame.attrs["easyicu_miiv_lab_attribution"]["null_hadm_rows_in_other_admission_dropped"] == 0


@pytest.mark.parametrize("invalid_column", ["admittime", "dischtime"])
def test_shared_sql_owner_tolerates_malformed_admission_strings(invalid_column):
    # Exercise raw VARCHARs independently of the public loader's datetime normalization.
    admissions = _admissions([(1, 99, "2140-01-02 00:00", "2140-01-04 00:00")])
    admissions[invalid_column] = "unrecorded"
    labs = _labs([(1, None, "2140-01-02 12:00", 50813, 1.)])
    for column in ("value", "specimen_id", "labevent_id"):
        labs[column] = None
    conn = _get_duckdb_connection()
    conn.register("__ed2_owner_labs", labs)
    conn.register("__ed2_owner_stays", _stays())
    conn.register("__miiv_lab_admissions", admissions)
    try:
        query = _lab_query(
            "SELECT * FROM __ed2_owner_labs", "SELECT * FROM __ed2_owner_stays",
            "SELECT * EXCLUDE (__lab_row) FROM assigned",
        )
        frame = finish_miiv_lab_frame(conn.execute(query).fetchdf())
    finally:
        for table in ("__ed2_owner_labs", "__ed2_owner_stays", "__miiv_lab_admissions"):
            conn.unregister(table)
    assert frame.stay_id.tolist() == [101]
    assert frame.valuenum.tolist() == [1.]
    assert frame.attrs["easyicu_miiv_lab_attribution"]["null_hadm_rows_in_other_admission_dropped"] == 0


@pytest.mark.parametrize("path", ["source", "preloaded"])
@pytest.mark.parametrize("target_hadm, expected", [(11, []), (12, [1.])])
def test_hadm_selection_uses_the_containing_admission_and_retains_null_provenance(
    tmp_path, path, target_hadm, expected,
):
    admissions = _admissions([(1, 12, "2140-01-02 00:00", "2140-01-04 00:00")])
    labs = _labs([(1, None, "2140-01-02 12:00", 50813, 1.)])
    source = _source(tmp_path, _stays(), labs, path, admissions)
    frame = source.load_table("labevents", filters=[FilterSpec("hadm_id", FilterOp.IN, [target_hadm])]).data
    assert frame.valuenum.tolist() == expected
    if expected:
        assert frame.hadm_id.isna().all()
        assert frame.attributed_hadm_id.tolist() == [12]
