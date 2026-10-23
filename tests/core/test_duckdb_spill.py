"""Connection spill paths must never share DuckDB's cwd block files."""

from pathlib import Path

import pytest

from easyicu import datasource


@pytest.fixture(autouse=True)
def isolated_connection(monkeypatch, tmp_path):
    datasource._close_duckdb_connections()
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("EASYICU_DUCKDB_TEMP_DIR", raising=False)
    monkeypatch.delenv("EASYICU_DUCKDB_MAX_TEMP_SIZE", raising=False)
    yield
    datasource._close_duckdb_connections()


def test_datasource_always_uses_private_state_spill(tmp_path):
    connection = datasource._get_duckdb_connection()
    path = Path(
        connection.execute("SELECT current_setting('temp_directory')").fetchone()[0]
    )
    assert path.is_absolute()
    assert path.is_relative_to(tmp_path / "home" / ".easyicu")
    assert path.is_dir()


def test_datasource_rejects_unwritable_parent(monkeypatch, tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied")
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(parent))
    with pytest.raises(RuntimeError, match=str(parent)):
        datasource._get_duckdb_connection()


def test_override_is_a_parent_with_distinct_private_names(monkeypatch, tmp_path):
    import os
    import re
    from easyicu.runtime import duckdb_spill as spill

    parent = tmp_path / "custom ' parent"
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(parent))
    monkeypatch.setenv("EASYICU_DUCKDB_MAX_TEMP_SIZE", "64MiB")
    with (
        spill.connection(config={"temp_directory": ".tmp"}) as first,
        spill.connection() as second,
    ):
        paths = [
            Path(c.execute("SELECT current_setting('temp_directory')").fetchone()[0])
            for c in (first, second)
        ]
        assert paths[0] != paths[1]
        for path in paths:
            assert path.parent == parent / "duckdb-spill"
            assert re.fullmatch(rf"pid-{os.getpid()}-[0-9a-f]{{32}}", path.name)
            assert path.stat().st_mode & 0o777 == 0o700
        assert (
            first.execute(
                "SELECT current_setting('max_temp_directory_size')"
            ).fetchone()[0]
            == "64.0 MiB"
        )
    assert not any(path.exists() for path in paths)
    assert parent.is_dir()


def test_maximum_applies_without_parent_override(monkeypatch):
    monkeypatch.setenv("EASYICU_DUCKDB_MAX_TEMP_SIZE", "48MiB")
    connection = datasource._get_duckdb_connection()
    assert (
        connection.execute(
            "SELECT current_setting('max_temp_directory_size')"
        ).fetchone()[0]
        == "48.0 MiB"
    )


def test_unwritable_parent_has_typed_path(monkeypatch, tmp_path):
    from easyicu.runtime import duckdb_spill as spill

    parent = tmp_path / "readonly"
    parent.mkdir(mode=0o500)
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(parent))
    try:
        with pytest.raises(spill.DuckDBSpillError) as caught:
            spill.connect()
        assert caught.value.path.is_relative_to(parent)
        assert str(caught.value.path) in str(caught.value)
        assert isinstance(caught.value.__cause__, PermissionError)
    finally:
        parent.chmod(0o700)


def test_context_cleanup_on_query_failure():
    from easyicu.runtime import duckdb_spill as spill

    with pytest.raises(ValueError, match="synthetic failure"):
        with spill.connection() as connection:
            path = Path(
                connection.execute(
                    "SELECT current_setting('temp_directory')"
                ).fetchone()[0]
            )
            (path / "synthetic-block").write_bytes(b"temporary")
            raise ValueError("synthetic failure")
    assert not path.exists()


def test_creation_failure_does_not_leak_directory(monkeypatch, tmp_path):
    from easyicu.runtime import duckdb_spill as spill

    monkeypatch.setenv("EASYICU_DUCKDB_MAX_TEMP_SIZE", "not-a-size")
    with pytest.raises(spill.DuckDBSpillError) as caught:
        spill.connect()
    assert not caught.value.path.exists()


def test_reaper_only_removes_old_dead_private_directories(monkeypatch, tmp_path):
    import os
    import time
    from easyicu.runtime import duckdb_spill as spill

    parent = tmp_path / "override"
    root = parent / "duckdb-spill"
    root.mkdir(parents=True)
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(parent))
    dead_pid, live_pid = 12345, 12346
    monkeypatch.setattr(spill, "_pid_alive", lambda pid: pid == live_pid)
    old = time.time() - 7200
    stale = root / f"pid-{dead_pid}-{'a' * 32}"
    young = root / f"pid-{dead_pid}-{'b' * 32}"
    live = root / f"pid-{live_pid}-{'c' * 32}"
    unknown = root / "unrelated"
    sibling = parent / f"pid-{dead_pid}-{'d' * 32}"
    for path in (stale, young, live, unknown, sibling):
        path.mkdir()
        (path / "keep").write_text("synthetic")
        if path != young:
            os.utime(path, (old, old))
    link = root / f"pid-{dead_pid}-{'e' * 32}"
    link.symlink_to(sibling, target_is_directory=True)
    with spill.connection():
        assert not stale.exists()
        assert all(
            (path / "keep").is_file() for path in (young, live, unknown, sibling, link)
        )


def test_symlink_root_is_not_reaped(monkeypatch, tmp_path):
    from easyicu.runtime import duckdb_spill as spill

    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "duckdb-spill").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("EASYICU_DUCKDB_TEMP_DIR", str(parent))
    with pytest.raises(spill.DuckDBSpillError):
        spill.connect()
    assert list(outside.iterdir()) == []


def test_pid_probe_preserves_live_or_inaccessible_processes(monkeypatch):
    import errno
    import os
    from easyicu.runtime import duckdb_spill as spill

    if os.name == "nt":
        pytest.skip("POSIX process probe")
    assert spill._pid_alive(os.getpid())
    for code, expected in (
        (errno.ESRCH, False),
        (errno.EPERM, True),
        (errno.EINVAL, True),
    ):

        def fail(pid, signal):
            raise OSError(code, "synthetic probe")

        monkeypatch.setattr(spill.os, "kill", fail)
        assert spill._pid_alive(12345) is expected


def test_cleanup_never_deletes_an_inherited_directory(monkeypatch):
    import os
    from easyicu.runtime import duckdb_spill as spill

    with spill.connection() as connection:
        path = Path(
            connection.execute("SELECT current_setting('temp_directory')").fetchone()[0]
        )
        pid = os.getpid()
        with monkeypatch.context() as patch:
            patch.setattr(spill.os, "getpid", lambda: pid + 1)
            spill._cleanup()
        assert path.is_dir()
        assert connection.execute("SELECT 42").fetchone() == (42,)


def test_all_production_duckdb_connections_have_one_owner():
    import ast

    source = Path(__file__).resolve().parents[2] / "src" / "easyicu"
    unmanaged = []
    tools = [
        "tools/verify_aumc_ventilator_sources.py",
        "tools/run_idea_mining_real_discovery.py",
        "tools/run_concept_proposer.py",
        "tools/build_discovery_universe.py",
        "scripts/build_itemid_bucket_cache.py",
        "scripts/releases/EX-A01_seal_full6_release.py",
        "scripts/releases/EX-A03_refresh_selected_modules.py",
    ]
    repo = source.parents[1]
    for path in [*source.rglob("*.py"), *(repo / name for name in tools)]:
        if path == source / "runtime" / "duckdb_spill.py":
            continue
        tree = ast.parse(path.read_text())
        aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == "duckdb"
        }
        direct = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "duckdb"
            for alias in node.names
            if alias.name == "connect"
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name)
                and node.func.id in direct
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in aliases
            ):
                unmanaged.append(f"{path.relative_to(repo)}:{node.lineno}")
    assert unmanaged == []


def test_exit_cleanup_closes_live_engine():
    from easyicu.runtime import duckdb_spill as spill
    import duckdb

    connection = spill.connect()
    path = Path(
        connection.execute("SELECT current_setting('temp_directory')").fetchone()[0]
    )
    spill._cleanup()
    assert not path.exists()
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_export_does_not_repoint_process_spill_state(monkeypatch, tmp_path):
    import os
    import tempfile
    from easyicu.api import extraction

    original_temp = tempfile.tempdir
    original_tmpdir = os.environ.get("TMPDIR")
    output = tmp_path / "output"

    def stop_before_loading(*args):
        raise ValueError("synthetic boundary")

    monkeypatch.setattr(extraction, "_get_all_patient_ids", stop_before_loading)
    with pytest.raises(ValueError, match="synthetic boundary"):
        extraction.extract_database(
            "miiv", data_path=str(tmp_path), output_dir=output, modules=["vitals"]
        )
    assert "EASYICU_DUCKDB_TEMP_DIR" not in os.environ
    assert os.environ.get("TMPDIR") == original_tmpdir
    assert tempfile.tempdir == original_temp
    assert not (output / ".easyicu_spill").exists()


def test_normal_exit_cleans_a_live_connection(tmp_path):
    import os
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from easyicu.runtime.duckdb_spill import connect; "
            "connection = connect(); "
            "print(connection.execute(\"SELECT current_setting('temp_directory')\").fetchone()[0])",
        ],
        env=dict(os.environ, PYTHONPATH=str(source), EASYICU_HOME=str(tmp_path)),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    path = Path(result.stdout.strip())
    assert path.is_relative_to(tmp_path / ".easyicu")
    assert not path.exists()
