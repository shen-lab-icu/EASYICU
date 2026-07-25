"""Regressions for the 2026-07-24 external code review.

One test per confirmed finding, named after it, so a future refactor that
reintroduces the defect fails with the finding id in the test name.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading

import pandas as pd
import pytest

from easyicu.api import (
    MAX_WINDOW_EXPANSION_POINTS,
    SOFA_FIXED_CHUNK_SIZE,
    WindowExpansionError,
    _database_profile_or_default,
    _expand_public_numeric_win_tbl_output,
)
from easyicu.table.duration import (
    UNIT_HOURS,
    UNIT_MINUTES,
    DurationUnitError,
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


def test_p0_2_strict_mode_refuses_to_guess():
    frame = pd.DataFrame({"dur_var": [10.0]})
    with pytest.raises(DurationUnitError, match="no declared unit"):
        resolve_dur_var_hours(frame, concept="norepi_rate", strict=True)


def test_p0_2_undeclared_unit_warns_but_still_works(caplog):
    """No hard break for untagged legacy paths — but the guess is announced."""

    frame = _win_frame([10.0])
    with caplog.at_level("WARNING", logger="easyicu.table.duration"):
        out = _expand_public_numeric_win_tbl_output(frame, "norepi_rate", "1h")
    assert not out.empty
    assert "no declared unit" in caplog.text


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


def test_p0_2_expansion_is_capped_instead_of_exhausting_memory():
    """An implausible duration fails loudly rather than allocating forever."""

    frame = _win_frame([1e9])
    set_dur_var_unit(frame, UNIT_HOURS)
    with pytest.raises(WindowExpansionError) as excinfo:
        _expand_public_numeric_win_tbl_output(frame, "norepi_rate", "1h")
    assert str(MAX_WINDOW_EXPANSION_POINTS) in str(excinfo.value)
    assert "stay_id" in str(excinfo.value), "the offending row must be reported"


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

    monkeypatch.setattr(api, "BaseICULoader", _FakeLoader)
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
