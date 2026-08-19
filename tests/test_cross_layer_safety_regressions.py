"""Cross-layer data, runner, relay and Web runtime safety regressions.

Origin: 2026-07-24 external code review. One test per confirmed finding,
named after it, so a future refactor that reintroduces the defect fails with
the finding id in the test name.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import (
    ConceptResolver,
    _drop_negative_source_end_durations,
    _source_duration_is_end,
)
from easyicu.api import (
    MAX_WINDOW_EXPANSION_POINTS,
    SOFA_FIXED_CHUNK_SIZE,
    WindowExpansionError,
    _database_profile_or_default,
    _expand_public_numeric_win_tbl_output,
)
from easyicu.table.duration import (
    ALLOW_GUESS_ENV_VAR,
    UNIT_HOURS,
    UNIT_MINUTES,
    DurationUnitError,
    DurationValueError,
    get_dur_var_unit,
    resolve_dur_var_hours,
    set_dur_var_unit,
)


def _win_frame(dur_values, *, index=(0.0,)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1] * len(dur_values),
            "charttime": list(index) * len(dur_values),
            "dur_var": list(dur_values),
            "norepi_rate": [0.1] * len(dur_values),
        }
    )


def test_aumc_later_admissions_use_icu_relative_hours():
    """AUMC's database-wide clock must not leak into native charttime."""

    resolver = ConceptResolver.__new__(ConceptResolver)
    resolver._aumc_admissions_cache = None
    source = SimpleNamespace(
        config=SimpleNamespace(name="aumc"),
        load_table=lambda *_args, **_kwargs: pd.DataFrame(
            {
                "admissionid": [1, 2],
                # ICUDataSource exposes AUMC's database-wide clock in minutes.
                "admittedat": [0, 300],
            }
        ),
    )
    frame = pd.DataFrame(
        {
            "admissionid": [1, 2],
            # bucketed sources reach the resolver as absolute minutes
            "measuredat_minutes": [120.0, 540.0],
            "stop": [180.0, 600.0],
            "value": [1.0, 2.0],
        }
    )

    out = resolver._align_time_to_admission(
        frame,
        source,
        ["admissionid"],
        "measuredat_minutes",
        time_columns=["stop"],
    )

    assert out["measuredat_minutes"].tolist() == pytest.approx([2.0, 4.0])
    assert out["stop"].tolist() == pytest.approx([3.0, 5.0])
    assert "admittedat" not in out.columns


def test_aumc_large_offsets_preserve_exact_integer_hour_boundaries():
    """Subtract the admission origin before scaling to avoid cancellation."""

    resolver = ConceptResolver.__new__(ConceptResolver)
    resolver._aumc_admissions_cache = None
    source = SimpleNamespace(
        config=SimpleNamespace(name="aumc"),
        load_table=lambda *_args, **_kwargs: pd.DataFrame(
            {
                "admissionid": [14301],
                # Converted source clock in integer minutes.
                "admittedat": [28_926.0],
                "dischargedat": [75_892.0],
            }
        ),
    )
    frame = pd.DataFrame(
        {
            "admissionid": [14301],
            "start": [67_026.0],
            "stop": [69_096.0],
            "rrt": [True],
        }
    )

    out = resolver._align_time_to_admission(
        frame,
        source,
        ["admissionid"],
        "start",
        time_columns=["stop"],
    )

    assert out["start"].tolist() == [635.0]
    assert out["stop"].tolist() == [669.5]


def test_aumc_admission_table_time_does_not_merge_its_origin_twice():
    """Stay-level AUMC concepts already carry admittedat in their frame."""

    resolver = ConceptResolver.__new__(ConceptResolver)
    source = SimpleNamespace(
        config=SimpleNamespace(name="aumc"),
        load_table=lambda *_args, **_kwargs: pytest.fail(
            "an admissions frame must not reload and duplicate admittedat"
        ),
    )
    frame = pd.DataFrame(
        {
            "admissionid": [2],
            "admittedat": [300.0],
            "dischargedat": [540.0],
            "los_icu": [4.0],
        }
    )

    out = resolver._align_time_to_admission(
        frame,
        source,
        ["admissionid"],
        "dischargedat",
    )

    assert out["dischargedat"].tolist() == [4.0]
    assert "admittedat" not in out.columns


def test_aumc_source_times_outside_the_icu_episode_are_quarantined():
    resolver = ConceptResolver.__new__(ConceptResolver)
    resolver._aumc_admissions_cache = None
    source = SimpleNamespace(
        config=SimpleNamespace(name="aumc"),
        load_table=lambda *_args, **_kwargs: pd.DataFrame(
            {
                "admissionid": [2],
                "admittedat": [300.0],
                "dischargedat": [600.0],
            }
        ),
    )
    frame = pd.DataFrame(
        {
            "admissionid": [2, 2],
            "measuredat_minutes": [540.0, 30_300.0],
            "value": [1.0, 2.0],
        }
    )

    out = resolver._align_time_to_admission(
        frame,
        source,
        ["admissionid"],
        "measuredat_minutes",
    )

    assert out["measuredat_minutes"].tolist() == [4.0]
    assert out["value"].tolist() == [1.0]
    assert "dischargedat" not in out.columns


def test_aumc_kdigo_history_can_use_168h_without_widening_default_window():
    """The phenotype window keeps seven-day history; generic concepts do not."""

    resolver = ConceptResolver.__new__(ConceptResolver)
    resolver._aumc_admissions_cache = None
    admittedat = 20_000.0
    source = SimpleNamespace(
        config=SimpleNamespace(name="aumc"),
        load_table=lambda *_args, **_kwargs: pd.DataFrame(
            {
                "admissionid": [1],
                "admittedat": [admittedat],
                "dischargedat": [admittedat + 48 * 60],
            }
        ),
    )
    frame = pd.DataFrame(
        {
            "admissionid": [1, 1, 1, 1],
            "measuredat_minutes": [
                admittedat - 169 * 60,
                admittedat - 168 * 60,
                admittedat - 120 * 60,
                admittedat,
            ],
            "crea": [0.8, 0.9, 1.0, 1.6],
        }
    )

    generic = resolver._align_time_to_admission(
        frame.copy(), source, ["admissionid"], "measuredat_minutes"
    )
    kdigo = resolver._align_time_to_admission(
        frame.copy(),
        source,
        ["admissionid"],
        "measuredat_minutes",
        pre_admission_hours=168,
    )

    assert generic["measuredat_minutes"].tolist() == [0.0]
    assert kdigo["measuredat_minutes"].tolist() == [-168.0, -120.0, 0.0]


def test_sic_interval_duration_seconds_are_declared_as_hours():
    """SIC data_range durations must not expand seconds as hours."""

    resolver = ConceptResolver.__new__(ConceptResolver)
    source = SimpleNamespace(config=SimpleNamespace(name="sic"))
    frame = pd.DataFrame(
        {
            "CaseID": [1],
            # The bucketed SIC loader already emitted an hour index.
            "charttime": [2.0],
            # data_range OffsetEnd - Offset is still seconds here.
            "dur_var": [7200.0],
        }
    )

    out = resolver._align_time_to_admission(
        frame,
        source,
        ["CaseID"],
        "charttime",
    )

    assert out["charttime"].tolist() == [2.0]
    assert out["dur_var"].tolist() == [2.0]
    assert get_dur_var_unit(out) == UNIT_HOURS


def test_sic_extreme_source_interval_is_quarantined_before_expansion():
    resolver = ConceptResolver.__new__(ConceptResolver)
    source = SimpleNamespace(config=SimpleNamespace(name="sic"))
    frame = pd.DataFrame(
        {
            "CaseID": [1, 1],
            # Raw SIC offsets are seconds; these correspond to -31,138 h and
            # +2 h respectively.
            "charttime": [-31_138.0 * 3_600.0, 2.0 * 3_600.0],
            "dur_var": [7_200.0, 7_200.0],
        }
    )

    out = resolver._align_time_to_admission(
        frame,
        source,
        ["CaseID"],
        "charttime",
    )

    assert out["charttime"].tolist() == [2.0]
    assert out["dur_var"].tolist() == [2.0]


# --------------------------------------------------------------------------
# P0-2 — dur_var unit was guessed from the value distribution (60x inflation)
# --------------------------------------------------------------------------


def test_p0_2_declared_minutes_are_not_read_as_hours():
    """A 10-minute infusion must not expand as a 10-hour one.

    The old heuristic (q95 <= 48 and median <= 24 => hours) classified
    10-minute durations as hours, inflating every window 60x.
    """

    frame = _win_frame([10.0, 10.0, 10.0], index=(0.0,))
    set_dur_var_unit(frame, UNIT_MINUTES)
    minutes = _expand_public_numeric_win_tbl_output(frame, "norepi_rate", "1h")

    hours_frame = _win_frame([10.0, 10.0, 10.0], index=(0.0,))
    set_dur_var_unit(hours_frame, UNIT_HOURS)
    hours = _expand_public_numeric_win_tbl_output(hours_frame, "norepi_rate", "1h")

    assert len(minutes) == 1, "10 minutes at a 1h interval is a single point"
    assert len(hours) == 11, "10 hours at a 1h interval is 11 points"
    assert len(hours) > len(minutes)


@pytest.mark.parametrize(
    ("unit", "value", "expected_hours"),
    [
        (UNIT_MINUTES, 10.0, 10.0 / 60.0),
        (UNIT_MINUTES, 120.0, 2.0),
        (UNIT_HOURS, 0.5, 0.5),
        (UNIT_HOURS, 2.0, 2.0),
    ],
)
def test_p0_2_declared_unit_converts_exactly(unit, value, expected_hours):
    frame = pd.DataFrame({"dur_var": [value]})
    set_dur_var_unit(frame, unit)
    assert resolve_dur_var_hours(frame).iloc[0] == pytest.approx(expected_hours)


def test_p0_2_timedelta_is_self_describing():
    """A timedelta dur_var needs no declaration and never hits the guess."""

    frame = pd.DataFrame({"dur_var": pd.to_timedelta([10], unit="m")})
    assert get_dur_var_unit(frame) is None
    assert resolve_dur_var_hours(frame).iloc[0] == pytest.approx(10.0 / 60.0)


def test_p0_2_undeclared_unit_fails_closed_by_default(monkeypatch):
    """Round 2: a warning does not stop a wrong number reaching a manuscript."""

    monkeypatch.delenv(ALLOW_GUESS_ENV_VAR, raising=False)
    with pytest.raises(DurationUnitError, match="no declared unit"):
        _expand_public_numeric_win_tbl_output(_win_frame([10.0]), "norepi_rate", "1h")


def test_p0_2_guess_requires_explicit_opt_out(monkeypatch, caplog):
    """The legacy guess survives only for a caller who asks for it by name."""

    monkeypatch.setenv(ALLOW_GUESS_ENV_VAR, "1")
    with caplog.at_level("WARNING", logger="easyicu.table.duration"):
        out = _expand_public_numeric_win_tbl_output(
            _win_frame([10.0]), "norepi_rate", "1h"
        )
    assert not out.empty
    assert "no declared unit" in caplog.text


# --- Round 2: the datetime branch used to ignore the declaration entirely ---


@pytest.mark.parametrize(
    ("unit", "expected_rows"),
    [(UNIT_MINUTES, 1), (UNIT_HOURS, 11)],
)
def test_p0_2_datetime_index_honours_declared_unit(unit, expected_rows):
    """A datetime index with a numeric dur_var used to be forced to minutes.

    That is the same 60x error as the distribution guess, on the other branch:
    a frame declaring hours came out 60x too short.
    """

    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": pd.to_datetime(["2026-01-01 00:00"]),
            "dur_var": [10.0],
            "norepi_rate": [0.1],
        }
    )
    set_dur_var_unit(frame, unit)
    out = _expand_public_numeric_win_tbl_output(frame, "norepi_rate", "1h")
    assert len(out) == expected_rows


def test_p0_2_datetime_and_numeric_branches_agree():
    """The two index branches must not disagree about the same duration."""

    numeric = _win_frame([90.0])
    set_dur_var_unit(numeric, UNIT_MINUTES)

    datetime_frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": pd.to_datetime(["2026-01-01 00:00"]),
            "dur_var": [90.0],
            "norepi_rate": [0.1],
        }
    )
    set_dur_var_unit(datetime_frame, UNIT_MINUTES)

    assert len(_expand_public_numeric_win_tbl_output(numeric, "norepi_rate", "1h")) == (
        len(_expand_public_numeric_win_tbl_output(datetime_frame, "norepi_rate", "1h"))
    )


# --- Round 2: corrupt durations used to become valid zero-length exposures ---


@pytest.mark.parametrize("bad", [-5.0, float("inf"), float("-inf")])
def test_p0_2_corrupt_duration_fails_closed(bad):
    """A negative/infinite duration became a valid exposure point via max(x, 0)."""

    frame = _win_frame([bad])
    set_dur_var_unit(frame, UNIT_HOURS)
    with pytest.raises(DurationValueError):
        _expand_public_numeric_win_tbl_output(frame, "norepi_rate", "1h")


def test_p0_2_known_source_end_before_start_is_quarantined(caplog):
    """A raw end-minus-start anomaly is dropped before the generic contract."""

    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "dur_var": [60.0, -15.0, float("nan")],
        }
    )
    with caplog.at_level("WARNING", logger="easyicu.concept"):
        cleaned = _drop_negative_source_end_durations(
            frame,
            concept_name="dex",
            source_table="inputevents_mv",
        )

    assert cleaned["stay_id"].tolist() == [1, 3]
    assert "dropping 1 raw end-before-start" in caplog.text


@pytest.mark.parametrize(
    "column",
    ["endtime", "stop", "drugstopoffset", "OffsetEnd"],
)
def test_p0_2_duration_end_semantics_come_from_schema_not_patient_sample(column):
    source = SimpleNamespace(dur_var=column, params={})
    assert _source_duration_is_end(source) is True


def test_p0_2_explicit_duration_semantics_override_column_name():
    assert (
        _source_duration_is_end(
            SimpleNamespace(
                dur_var="drugstopoffset",
                params={"dur_is_end": False},
            )
        )
        is False
    )
    assert (
        _source_duration_is_end(
            SimpleNamespace(
                dur_var="duration_minutes",
                params={"dur_is_end": True},
            )
        )
        is True
    )


def test_p0_2_missing_duration_is_dropped_not_zeroed(caplog):
    """NaN must not silently become a zero-length window that still emits a point."""

    frame = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [0.0, 0.0],
            "dur_var": [float("nan"), 2.0],
            "norepi_rate": [0.1, 0.2],
        }
    )
    set_dur_var_unit(frame, UNIT_HOURS)
    with caplog.at_level("WARNING", logger="easyicu.table.duration"):
        out = _expand_public_numeric_win_tbl_output(frame, "norepi_rate", "1h")

    assert set(out["stay_id"]) == {2}, "the NaN-duration stay must not appear"
    assert "missing dur_var" in caplog.text


def test_p0_2_producers_declare_their_unit():
    """The ts_to_win_tbl callback must tag what it wrote."""

    from easyicu.utils.callback_utils import ts_to_win_tbl

    numeric_index = pd.DataFrame({"charttime": [0.0, 1.0], "v": [1, 2]})
    tagged = ts_to_win_tbl(pd.Timedelta(hours=1))(numeric_index)
    assert get_dur_var_unit(tagged) == UNIT_HOURS

    datetime_index = pd.DataFrame(
        {"charttime": pd.to_datetime(["2026-01-01", "2026-01-02"]), "v": [1, 2]}
    )
    tagged_dt = ts_to_win_tbl(pd.Timedelta(hours=1))(datetime_index)
    assert pd.api.types.is_timedelta64_dtype(tagged_dt["dur_var"])


def test_p0_2_hirid_vent_declares_hour_duration_after_conversion():
    """HiRID ventilation converts padded minutes to hours and must label them."""

    from easyicu.utils.callback_utils import hirid_vent

    frame = pd.DataFrame(
        {
            "patientid": [1, 1],
            "datetime": [0.0, 2.0],
            "mech_vent": [1, 1],
        }
    )
    result = hirid_vent(
        frame,
        concept_name="mech_vent",
        val_col="mech_vent",
        index_col="datetime",
        expand_to_hourly=False,
    )

    assert get_dur_var_unit(result) == UNIT_HOURS
    assert result["dur_var"].tolist() == pytest.approx([2.0, 4.0])


def test_p0_2_source_projection_preserves_and_normalizes_duration_unit():
    """A projected medication source must not lose its duration declaration."""

    from easyicu.concept import _normalize_source_dur_var_hours

    source = pd.DataFrame({"dur_var": [120.0], "dex": [1.0]})
    set_dur_var_unit(source, UNIT_MINUTES)
    projected = source.loc[:, ["dur_var", "dex"]].copy()
    projected.attrs.clear()  # model pandas concat/projection metadata loss

    normalized = _normalize_source_dur_var_hours(
        projected,
        concept_name="dex",
        source_frame=source,
    )

    assert normalized["dur_var"].iloc[0] == pytest.approx(2.0)
    assert get_dur_var_unit(normalized) == UNIT_HOURS


def test_p0_2_callback_chain_preserves_unchanged_duration_contract():
    """A value-only nested callback must not erase an existing duration unit."""

    from easyicu.concept.callback_apply import _preserve_callback_dur_var_unit

    before = pd.DataFrame({"dur_var": [10.0], "dex": [1.0]})
    set_dur_var_unit(before, UNIT_MINUTES)
    after = before.copy()
    after.attrs.clear()

    carried = _preserve_callback_dur_var_unit(before, after)

    assert get_dur_var_unit(carried) == UNIT_MINUTES


def test_p0_2_expansion_is_capped_instead_of_exhausting_memory():
    """An implausible duration fails loudly rather than allocating forever."""

    frame = _win_frame([1e9])
    set_dur_var_unit(frame, UNIT_HOURS)
    with pytest.raises(WindowExpansionError) as excinfo:
        _expand_public_numeric_win_tbl_output(frame, "norepi_rate", "1h")
    assert str(MAX_WINDOW_EXPANSION_POINTS) in str(excinfo.value)
    # The row is described, not reproduced (2026-07-29): field names and a
    # digest identify it for debugging; its values would put a patient
    # identifier and event times into every log this message reaches.
    assert "stay_id" in str(excinfo.value), "the offending row must be reported"
    assert "sha256" in str(excinfo.value)


# --------------------------------------------------------------------------
# P0-3 — chunk size (an execution parameter) must not depend on host memory
# --------------------------------------------------------------------------


def test_p0_3_sofa_chunk_size_is_memory_independent(monkeypatch):
    """The same cohort must chunk identically on a laptop and a workstation.

    Chunk size can still change SOFA window expansion, so letting free RAM pick
    it made the score host-dependent.
    """

    from easyicu import api

    sizes = set()
    for memory_mb in (2 * 1024, 4 * 1024, 8 * 1024, 32 * 1024):
        monkeypatch.setattr(
            "easyicu.runtime.memory_manager.get_available_memory_mb",
            lambda mb=memory_mb: mb,
        )
        strategy = api._get_auto_chunk_strategy(
            ["sofa"],
            50_000,
            merge=True,
            chunk_size=None,
            batch_size=None,
            parallel_workers=None,
            concept_workers=None,
        )
        if strategy is not None:
            sizes.add(strategy["chunk_size"])

    assert sizes in (
        {SOFA_FIXED_CHUNK_SIZE},
        set(),
    ), f"SOFA chunk size varied with available memory: {sorted(sizes)}"


# --------------------------------------------------------------------------
# P0-4 — the global loader was a check-then-act race across databases
# --------------------------------------------------------------------------


def test_p0_4_concurrent_loader_requests_never_cross_databases(monkeypatch):
    """Two threads asking for different databases must not get each other's loader."""

    from easyicu import api

    class _FakeLoader:
        def __init__(self, database=None, data_path=None, dict_path=None, **kwargs):
            self.database = database
            self.data_path = data_path

        def clear_cache(self):
            pass

    from easyicu.api import concepts as concept_api

    monkeypatch.setattr(concept_api, "BaseICULoader", _FakeLoader)
    api.clear_global_loader()

    requests = [("miiv", "/data/mimiciv"), ("eicu", "/data/eicu")] * 40
    barrier = threading.Barrier(len(requests))
    mismatches: list[tuple[str, str]] = []
    lock = threading.Lock()

    def _worker(database: str, data_path: str) -> None:
        barrier.wait()
        loader = api._get_global_loader(database=database, data_path=Path(data_path))
        if loader.database != database:
            with lock:
                mismatches.append((database, loader.database))

    threads = [threading.Thread(target=_worker, args=pair) for pair in requests]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    api.clear_global_loader()
    assert not mismatches, f"loader handed back the wrong database: {mismatches[:5]}"


# --------------------------------------------------------------------------
# P1-1 — MIMIC-III was detected as MIMIC-IV (different stay-id column)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/data/mimiciii",
        "/data/mimic-iii",
        "/data/mimic_iii",
        "/data/mimic3",
        "/data/miii",
    ],
)
def test_p1_1_mimic_iii_paths_are_not_labelled_mimic_iv(path):
    from easyicu.webserver.dataio import _detect_database

    assert _detect_database(Path(path)) == "miii"


@pytest.mark.parametrize("path", ["/data/mimiciv", "/data/mimic-iv", "/data/miiv"])
def test_p1_1_mimic_iv_paths_still_detected(path):
    from easyicu.webserver.dataio import _detect_database

    assert _detect_database(Path(path)) == "miiv"


def test_p1_1_ambiguous_mimic_path_refuses_to_guess():
    """ "/data/mimic" says nothing about the version — do not pick one."""

    from easyicu.webserver.dataio import _detect_database

    assert _detect_database(Path("/nonexistent/mimic")) == "unknown"


def test_p1_1_miii_resolves_to_the_mimic_iii_profile():
    profile = _database_profile_or_default("miii")
    assert profile.display_name == "MIMIC-III"
    assert profile.stay_id_col == "icustay_id"


def test_p1_1_miii_has_core_tables_configured():
    from easyicu.webserver.dataio import _CORE_TABLES, _DB_LABELS

    assert "miii" in _DB_LABELS
    assert _CORE_TABLES.get("miii"), "miii had a label but no core-table list"


# --------------------------------------------------------------------------
# P1-2 — an unknown database name silently became MIMIC-IV
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mimic-42", "aumdb", "not-a-database"])
def test_p1_2_unknown_database_raises_instead_of_defaulting(name):
    with pytest.raises(ValueError, match="Unsupported database"):
        _database_profile_or_default(name)


@pytest.mark.parametrize("name", [None, ""])
def test_p1_2_unspecified_database_still_defaults(name):
    """Only an *unspecified* database keeps the legacy MIIV default."""

    assert _database_profile_or_default(name).key == "miiv"


@pytest.mark.parametrize(
    "name", ["miiv", "mimic-iv", "eicu", "hirid", "aumc", "mimic3"]
)
def test_p1_2_known_aliases_still_resolve(name):
    assert _database_profile_or_default(name) is not None


# --------------------------------------------------------------------------
# P1-3 — the Docker runner build context was not shipped in the package
# --------------------------------------------------------------------------


def test_p1_3_runner_image_is_declared_as_package_data():
    """DockerRunner's docstring points users at these files; ship them."""

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    for asset in ("runner_image/Dockerfile", "runner_image/requirements.lock"):
        assert (
            asset in pyproject
        ), f"{asset} missing from [tool.setuptools.package-data]"

    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text(
        encoding="utf-8"
    )
    assert "runner_image" in manifest, "runner_image missing from the sdist manifest"


# --------------------------------------------------------------------------
# P1-4 — the PID file could SIGTERM a recycled, unrelated PID
# --------------------------------------------------------------------------


def test_p1_4_pid_record_rejects_a_process_that_is_not_the_webserver(tmp_path):
    from easyicu.webserver import __main__ as webserver_main

    pid_path = tmp_path / "easyicu_webserver.pid"
    # This interpreter is a live PID, but it is not uvicorn.
    webserver_main._write_pid_record(pid_path, os.getpid(), 8765)
    record = webserver_main._read_pid_record(pid_path)

    assert record["pid"] == os.getpid()
    assert webserver_main._pid_matches_record(os.getpid(), record) is False


def test_p1_4_legacy_bare_pid_file_is_still_read(tmp_path):
    from easyicu.webserver import __main__ as webserver_main

    pid_path = tmp_path / "legacy.pid"
    pid_path.write_text("4242", encoding="utf-8")
    assert webserver_main._read_pid_record(pid_path) == {"pid": 4242}


def test_p1_4_runtime_dir_is_private_and_user_scoped(monkeypatch, tmp_path):
    from easyicu.webserver import __main__ as webserver_main

    monkeypatch.delenv("EASYICU_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    runtime_dir = webserver_main._runtime_dir()
    assert str(os.getuid()) in runtime_dir.name, "runtime dir must be UID-scoped"
    assert runtime_dir.stat().st_mode & 0o777 == 0o700


# --------------------------------------------------------------------------
# P1-5 — nested generated output never reached the evidence store
# --------------------------------------------------------------------------


def test_p1_5_nested_artifacts_are_collected(tmp_path):
    from easyicu.research_agent.execution.runner import _collect_safe_output_artifacts

    (tmp_path / "top.csv").write_text("a", encoding="utf-8")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "fig1.png").write_text("p", encoding="utf-8")
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "table1.csv").write_text("t", encoding="utf-8")

    collected = {
        path.relative_to(tmp_path).as_posix()
        for path in _collect_safe_output_artifacts(tmp_path)
    }
    assert collected == {"top.csv", "figures/fig1.png", "tables/table1.csv"}


def test_p1_5_symlinks_are_still_rejected(tmp_path):
    from easyicu.research_agent.execution.runner import _collect_safe_output_artifacts

    (tmp_path / "nested").mkdir()
    target = tmp_path / "nested" / "real.csv"
    target.write_text("x", encoding="utf-8")
    os.symlink("/etc/passwd", tmp_path / "nested" / "escape.link")

    collected = _collect_safe_output_artifacts(tmp_path)
    assert [p.name for p in collected] == ["real.csv"]
    assert not (tmp_path / "nested" / "escape.link").exists()


# --------------------------------------------------------------------------
# P1-6 — the hosted relay bounded request *count* but not request *size*
# --------------------------------------------------------------------------


@pytest.fixture()
def relay_client(monkeypatch):
    from fastapi.testclient import TestClient

    import easyicu.hosted_llm_server as relay

    monkeypatch.setattr(relay, "HOSTED_SERVER_TOKEN", "test-token")
    monkeypatch.setattr(relay, "OPENROUTER_API_KEY", "test-key")
    return TestClient(relay.app), {"Authorization": "Bearer test-token"}


def test_p1_6_rejects_oversized_max_tokens(relay_client):
    client, headers = relay_client
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "hosted-default",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200_000,
        },
    )
    assert response.status_code == 413


def test_p1_6_rejects_too_many_messages(relay_client):
    client, headers = relay_client
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "hosted-default",
            "messages": [{"role": "user", "content": "x"}] * 5_000,
        },
    )
    assert response.status_code == 413


def test_p1_6_rejects_oversized_body(relay_client):
    client, headers = relay_client
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "hosted-default",
            "messages": [{"role": "user", "content": "x" * 3_000_000}],
        },
    )
    assert response.status_code == 413


def test_p1_6_forwards_only_allowlisted_fields():
    import easyicu.hosted_llm_server as relay

    upstream = relay._build_upstream_payload(
        {
            "model": "hosted-default",
            "messages": [{"role": "user", "content": "hi"}],
            "provider": {"order": ["some-provider"]},
            "route": "fallback",
            "transforms": ["middle-out"],
        }
    )
    assert set(upstream) <= relay.HOSTED_FORWARDED_FIELDS
    for smuggled in ("provider", "route", "transforms"):
        assert smuggled not in upstream


def test_p1_6_rate_limit_state_is_bounded(monkeypatch):
    import easyicu.hosted_llm_server as relay

    monkeypatch.setattr(relay, "HOSTED_RATE_LIMIT", 100)
    monkeypatch.setattr(relay, "HOSTED_RATE_LIMIT_MAX_TRACKED_IPS", 16)
    relay._RATE_LIMIT_STATE.clear()

    for octet in range(300):
        relay._check_rate_limit(f"10.0.{octet // 256}.{octet % 256}")

    assert len(relay._RATE_LIMIT_STATE) <= 16
    relay._RATE_LIMIT_STATE.clear()


# --------------------------------------------------------------------------
# P2-4 — three different version numbers for one product
# --------------------------------------------------------------------------


def test_p2_4_web_api_version_matches_the_package():
    from importlib.metadata import version

    from easyicu.webserver.app import app

    assert app.version == version("easyicu")


# ==========================================================================
# Round 2 (2026-07-25) — findings raised against the first remediation
# ==========================================================================


def test_r2_loader_is_not_torn_down_while_another_thread_uses_it(monkeypatch):
    """Switching database must not clear a loader another thread is using.

    The first remediation released the previous loader when the config changed,
    which emptied its ConceptResolver and DataSource caches underneath a caller
    that was still mid-extraction.
    """

    from easyicu import api

    cleared: list[str] = []

    class _FakeLoader:
        def __init__(self, database=None, data_path=None, dict_path=None, **kwargs):
            self.database = database

        def clear_cache(self):
            cleared.append(self.database)

    from easyicu.api import concepts as concept_api

    monkeypatch.setattr(concept_api, "BaseICULoader", _FakeLoader)
    api.clear_global_loader()
    cleared.clear()

    holder: dict[str, object] = {}
    acquired = threading.Event()
    may_finish = threading.Event()

    def _long_running_consumer():
        holder["loader"] = api._get_global_loader(
            database="miiv", data_path=Path("/data/mimiciv")
        )
        acquired.set()
        may_finish.wait(timeout=5)

    worker = threading.Thread(target=_long_running_consumer)
    worker.start()
    assert acquired.wait(timeout=5)

    other = api._get_global_loader(database="eicu", data_path=Path("/data/eicu"))

    assert other.database == "eicu"
    assert holder["loader"].database == "miiv"
    assert "miiv" not in cleared, "loader was cleared while still in use"

    may_finish.set()
    worker.join(timeout=5)
    api.clear_global_loader()


def test_r2_loader_cache_returns_the_same_instance_per_config(monkeypatch):
    """Alternating databases must not rebuild (and re-scan) a loader each time."""

    from easyicu import api

    class _FakeLoader:
        def __init__(self, database=None, data_path=None, dict_path=None, **kwargs):
            self.database = database

        def clear_cache(self):
            pass

    from easyicu.api import concepts as concept_api

    monkeypatch.setattr(concept_api, "BaseICULoader", _FakeLoader)
    api.clear_global_loader()

    first = api._get_global_loader(database="miiv", data_path=Path("/data/mimiciv"))
    api._get_global_loader(database="eicu", data_path=Path("/data/eicu"))
    again = api._get_global_loader(database="miiv", data_path=Path("/data/mimiciv"))

    assert again is first
    api.clear_global_loader()


def test_r2_flat_mimic_iv_is_not_mistaken_for_mimic_iii(tmp_path):
    """Both generations ship icustays/patients/admissions — use the schema.

    A converted MIMIC-IV in a flat parquet layout, in a directory whose name
    carries no version token, was detected as MIMIC-III.
    """

    from easyicu.webserver.dataio import _detect_database

    for name, columns, expected in (
        ("prepared_v4", ["stay_id", "subject_id"], "miiv"),
        ("prepared_v3", ["icustay_id", "subject_id"], "miii"),
    ):
        root = tmp_path / name
        root.mkdir()
        for table in ("admissions", "patients"):
            pd.DataFrame(columns=["x"]).to_parquet(root / f"{table}.parquet")
        pd.DataFrame(columns=columns).to_parquet(root / "icustays.parquet")
        assert _detect_database(root) == expected


def test_r2_unidentifiable_mimic_layout_returns_unknown(tmp_path):
    root = tmp_path / "prepared"
    root.mkdir()
    for table in ("admissions", "patients", "icustays"):
        pd.DataFrame(columns=["foo", "bar"]).to_parquet(root / f"{table}.parquet")

    from easyicu.webserver.dataio import _detect_database

    assert _detect_database(root) == "unknown"


def test_r2_output_deeper_than_the_sweep_limit_fails_closed(tmp_path):
    """An artefact the sweep cannot register must not coexist with success."""

    from easyicu.research_agent.execution import runner

    deep = tmp_path
    for level in range(runner.MAX_OUTPUT_ARTIFACT_DEPTH + 2):
        deep = deep / f"level{level}"
    deep.mkdir(parents=True)
    (deep / "result.csv").write_text("x", encoding="utf-8")

    with pytest.raises(runner.OutputArtifactPolicyError, match="nested deeper"):
        runner._collect_safe_output_artifacts(tmp_path)


def test_r2_oversized_output_file_fails_closed(tmp_path, monkeypatch):
    from easyicu.research_agent.execution import runner

    monkeypatch.setattr(runner, "MAX_OUTPUT_ARTIFACT_FILE_BYTES", 16)
    (tmp_path / "big.csv").write_text("x" * 64, encoding="utf-8")

    with pytest.raises(runner.OutputArtifactPolicyError, match="per-file limit"):
        runner._collect_safe_output_artifacts(tmp_path)


def test_r2_too_many_output_files_fails_closed(tmp_path, monkeypatch):
    from easyicu.research_agent.execution import runner

    monkeypatch.setattr(runner, "MAX_OUTPUT_ARTIFACT_FILES", 3)
    for index in range(6):
        (tmp_path / f"f{index}.csv").write_text("x", encoding="utf-8")

    with pytest.raises(runner.OutputArtifactPolicyError, match="more than 3"):
        runner._collect_safe_output_artifacts(tmp_path)


@pytest.mark.parametrize("bad", [8192.9, "8192", True])
def test_r2_relay_rejects_non_integer_numerics(relay_client, bad):
    """Validating a coerced copy while forwarding the raw value is not enough."""

    client, headers = relay_client
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "hosted-default",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": bad,
        },
    )
    assert response.status_code == 400


def test_r2_relay_forwards_the_normalised_value():
    import easyicu.hosted_llm_server as relay

    validated = relay._validate_request_shape(
        {
            "model": "hosted-default",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
        }
    )
    assert relay._build_upstream_payload(validated)["max_tokens"] == 4096


def test_r2_relay_ceiling_is_not_used_as_the_default():
    """An omitted max_tokens must not bill at the maximum allowed length."""

    import easyicu.hosted_llm_server as relay

    upstream = relay._build_upstream_payload(
        {"model": "hosted-default", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert upstream["max_tokens"] == relay.HOSTED_DEFAULT_OUTPUT_TOKENS
    assert upstream["max_tokens"] < relay.HOSTED_MAX_OUTPUT_TOKENS


def test_r2_relay_does_not_block_the_event_loop_on_upstream():
    """A slow upstream call must not stall other routes on the same worker."""

    import inspect

    import easyicu.hosted_llm_server as relay

    source = inspect.getsource(relay.chat_completions)
    assert (
        "run_in_threadpool" in source
    ), "synchronous _post_upstream must not be awaited directly on the loop"
    assert "_upstream_slot" in source, "in-flight upstream calls must be bounded"


# ==========================================================================
# Round 3 (2026-07-25) — findings raised against the second remediation
# ==========================================================================


def test_r3_change_dur_unit_updates_the_declaration():
    """Converting the values without relabelling them is a 60x error.

    120 minutes -> 2 hours must also stop saying "minutes", otherwise the next
    consumer divides by 60 again and reads 0.033 h.
    """

    from easyicu.io.data_tools import change_dur_unit
    from easyicu.table import WinTbl

    frame = pd.DataFrame({"pid": [1], "t": [0.0], "dur": [120.0], "v": [1.0]})
    set_dur_var_unit(frame, UNIT_MINUTES)
    table = WinTbl(frame, id_vars=["pid"], index_var="t", dur_var="dur")

    converted = change_dur_unit(table, "hours")

    assert converted.data["dur"].iloc[0] == pytest.approx(2.0)
    assert get_dur_var_unit(converted.data) == UNIT_HOURS
    assert converted.dur_unit == UNIT_HOURS
    assert resolve_dur_var_hours(converted.data, column="dur").iloc[0] == pytest.approx(
        2.0
    )


@pytest.mark.parametrize(
    ("start_unit", "target", "value", "expected"),
    [
        (UNIT_MINUTES, "hours", 120.0, 2.0),
        (UNIT_HOURS, "minutes", 2.0, 120.0),
        (UNIT_MINUTES, "seconds", 2.0, 120.0),
        (UNIT_HOURS, "days", 48.0, 2.0),
    ],
)
def test_r3_change_dur_unit_converts_between_all_units(
    start_unit, target, value, expected
):
    from easyicu.io.data_tools import change_dur_unit
    from easyicu.table import WinTbl

    frame = pd.DataFrame({"pid": [1], "t": [0.0], "dur": [value], "v": [1.0]})
    set_dur_var_unit(frame, start_unit)
    table = WinTbl(frame, id_vars=["pid"], index_var="t", dur_var="dur")

    converted = change_dur_unit(table, target)
    assert converted.data["dur"].iloc[0] == pytest.approx(expected)
    assert get_dur_var_unit(converted.data) == target


def test_r3_change_dur_unit_round_trips():
    from easyicu.io.data_tools import change_dur_unit
    from easyicu.table import WinTbl

    frame = pd.DataFrame({"pid": [1], "t": [0.0], "dur": [90.0], "v": [1.0]})
    set_dur_var_unit(frame, UNIT_MINUTES)
    table = WinTbl(frame, id_vars=["pid"], index_var="t", dur_var="dur")

    back = change_dur_unit(change_dur_unit(table, "hours"), "minutes")
    assert back.data["dur"].iloc[0] == pytest.approx(90.0)
    assert get_dur_var_unit(back.data) == UNIT_MINUTES


def test_r3_change_dur_unit_converts_timedelta_and_labels_it():
    from easyicu.io.data_tools import change_dur_unit
    from easyicu.table import WinTbl

    frame = pd.DataFrame(
        {
            "pid": [1],
            "t": [0.0],
            "dur": pd.to_timedelta([90], unit="m"),
            "v": [1.0],
        }
    )
    table = WinTbl(frame, id_vars=["pid"], index_var="t", dur_var="dur")

    converted = change_dur_unit(table, "hours")
    assert converted.data["dur"].iloc[0] == pytest.approx(1.5)
    assert get_dur_var_unit(converted.data) == UNIT_HOURS


def test_r3_change_dur_unit_refuses_undeclared_numeric():
    """ "Already numeric, assume minutes" is the guess this contract removes."""

    from easyicu.io.data_tools import change_dur_unit
    from easyicu.table import WinTbl

    frame = pd.DataFrame({"pid": [1], "t": [0.0], "dur": [120.0], "v": [1.0]})
    table = WinTbl(frame, id_vars=["pid"], index_var="t", dur_var="dur")

    with pytest.raises(DurationUnitError, match="CURRENT unit"):
        change_dur_unit(table, "hours")


def test_r3_wintbl_carries_the_unit_as_a_structural_field():
    """attrs alone is too fragile a carrier; WinTbl records it too."""

    from easyicu.table import WinTbl

    frame = pd.DataFrame({"pid": [1], "t": [0.0], "dur": [30.0], "v": [1.0]})
    table = WinTbl(
        frame, id_vars=["pid"], index_var="t", dur_var="dur", dur_unit=UNIT_MINUTES
    )
    assert table.dur_unit == UNIT_MINUTES
    assert get_dur_var_unit(table.data) == UNIT_MINUTES

    timedelta_frame = pd.DataFrame(
        {"pid": [1], "t": [0.0], "dur": pd.to_timedelta([30], unit="m"), "v": [1.0]}
    )
    self_describing = WinTbl(
        timedelta_frame, id_vars=["pid"], index_var="t", dur_var="dur"
    )
    assert self_describing.dur_unit == "timedelta"


def test_r3_evidence_scan_fails_closed_on_unreadable_directory(tmp_path, monkeypatch):
    """An unenumerable output directory is a failure, not a gap to skip."""

    from easyicu.research_agent.execution import runner

    nested = tmp_path / "figures"
    nested.mkdir()
    (nested / "fig.png").write_text("x", encoding="utf-8")

    real_iterdir = Path.iterdir

    def _explode(self):
        if self.name == "figures":
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _explode)

    with pytest.raises(runner.OutputArtifactPolicyError, match="cannot enumerate"):
        runner._collect_safe_output_artifacts(tmp_path)


def test_r3_evidence_scan_fails_closed_on_real_unreadable_directory(tmp_path):
    """Same contract against a real chmod-000 directory where the OS allows it."""

    if os.getuid() == 0:
        pytest.skip("root ignores directory permissions")

    from easyicu.research_agent.execution import runner

    nested = tmp_path / "locked"
    nested.mkdir()
    (nested / "fig.png").write_text("x", encoding="utf-8")
    nested.chmod(0o000)
    try:
        with pytest.raises(runner.OutputArtifactPolicyError, match="cannot enumerate"):
            runner._collect_safe_output_artifacts(tmp_path)
    finally:
        nested.chmod(0o700)


def test_r3_cache_manager_does_not_pin_evicted_loader_components():
    """Registering a cache must not keep it alive after its owner is gone."""

    import gc

    from easyicu.runtime.cache_manager import CacheManager

    class _Resolver:
        def clear(self):
            pass

    manager = CacheManager()
    resolver = _Resolver()
    manager.register_memory_cache(resolver)
    tracked = __import__("weakref").ref(resolver)

    assert tracked() is not None
    del resolver
    gc.collect()

    assert (
        tracked() is None
    ), "CacheManager still holds a strong reference to an evicted component"


def test_r3_cache_manager_still_clears_live_caches():
    from easyicu.runtime.cache_manager import CacheManager

    cleared = []

    class _Resolver:
        def clear(self):
            cleared.append(True)

    manager = CacheManager()
    resolver = _Resolver()
    manager.register_memory_cache(resolver)
    manager.clear_memory_cache()

    assert cleared, "a live registered cache must still be cleared"


def test_r3_streaming_holds_its_upstream_slot_for_the_whole_stream():
    """Not a source grep: actually run concurrent streams and count them.

    Releasing the slot when requests.post returns bounded only the connect,
    so unbounded long-lived streams could run at once.
    """

    import asyncio
    import time as _time

    import easyicu.hosted_llm_server as relay

    original_limit = relay.HOSTED_MAX_CONCURRENT_UPSTREAM
    original_semaphore = relay._UPSTREAM_SEMAPHORE
    original_post = relay._post_upstream

    relay.HOSTED_MAX_CONCURRENT_UPSTREAM = 2
    relay._UPSTREAM_SEMAPHORE = None
    state = {"live": 0, "peak": 0}

    class _FakeResponse:
        status_code = 200

        def iter_content(self, chunk_size=1024):
            for _ in range(3):
                _time.sleep(0.02)
                yield b"chunk"

        def close(self):
            state["live"] -= 1

    def _fake_post(request, payload, *, stream):
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        return _FakeResponse()

    relay._post_upstream = _fake_post

    async def _drive():
        async def _one():
            response = await relay._bounded_streaming_response(None, {"model": "m"})
            async for _ in response.body_iterator:
                pass

        await asyncio.gather(*[_one() for _ in range(8)])

    try:
        asyncio.run(_drive())
        assert (
            state["peak"] <= 2
        ), f"{state['peak']} concurrent upstream streams exceeded the limit of 2"
        assert relay._UPSTREAM_SEMAPHORE._value == 2, "a slot leaked"
    finally:
        relay.HOSTED_MAX_CONCURRENT_UPSTREAM = original_limit
        relay._UPSTREAM_SEMAPHORE = original_semaphore
        relay._post_upstream = original_post


def test_r3_api_has_no_stale_chunk_invariance_claim():
    """The measurement replaced the claim; the claim must not survive anywhere."""

    import inspect

    from easyicu import api

    source = inspect.getsource(api._get_auto_chunk_strategy)
    assert "can still change SOFA" not in source
    assert "measured" in source.lower() or "invariance" in source.lower()
