from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from easyicu.research_agent.execution.raw_snapshot_sandbox import run_snapshot_worker


@pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which("sandbox-exec"),
    reason="requires macOS sandbox-exec",
)
def test_native_readers_are_confined_to_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "output"
    snapshot.mkdir()
    output.mkdir()
    inside = snapshot / "table.parquet"
    outside = tmp_path / "outside.parquet"
    monkeypatch.setenv("EASYICU_TEST_EXTRA_PATH", str(tmp_path))
    pq.write_table(pa.table({"value": [1]}), inside)
    pq.write_table(pa.table({"value": [2]}), outside)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import errno, os, socket, sys\n"
        "import duckdb, pyarrow.parquet as pq\n"
        "good, bad = sys.argv[1:]\n"
        "assert pq.read_table(good).num_rows == 1\n"
        "assert duckdb.connect().execute('SELECT count(*) FROM read_parquet(?)', [good]).fetchone()[0] == 1\n"
        "os.symlink(bad, 'external-link.parquet')\n"
        "for read in (lambda: pq.read_table(bad), lambda: duckdb.connect().execute('SELECT * FROM read_parquet(?)', [bad]).fetchall(), lambda: pq.read_table('external-link.parquet')):\n"
        "    try:\n"
        "        read()\n"
        "    except Exception:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('outside data was readable')\n"
        "try:\n"
        "    open(good, 'ab')\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('snapshot was writable')\n"
        "assert 'EASYICU_TEST_EXTRA_PATH' not in os.environ\n"
        "try:\n"
        "    socket.socket().connect(('127.0.0.1', 9))\n"
        "except OSError as error:\n"
        "    assert error.errno in (errno.EPERM, errno.EACCES), error\n"
        "else:\n"
        "    raise AssertionError('network was allowed')\n"
        "print('confined')\n",
        encoding="utf-8",
    )
    result = run_snapshot_worker(
        snapshot=snapshot,
        output=output,
        python=Path(sys.executable),
        worker=worker,
        runtime_roots=[Path(sys.prefix), Path(sys.base_prefix)],
        arguments=[str(inside), str(outside)],
    )
    assert result.returncode == 0, result.stderr
    assert "confined" in result.stdout


def test_runtime_cannot_contain_snapshot_or_output(tmp_path: Path) -> None:
    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        pytest.skip("requires macOS sandbox-exec")
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "output"
    snapshot.mkdir()
    output.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("pass", encoding="utf-8")
    with pytest.raises(ValueError, match="outside runtime roots"):
        run_snapshot_worker(
            snapshot=snapshot,
            output=output,
            python=Path(sys.executable),
            worker=worker,
            runtime_roots=[Path(sys.prefix), Path(sys.base_prefix), tmp_path],
        )
    with pytest.raises(ValueError, match="raw source must be outside"):
        run_snapshot_worker(
            snapshot=snapshot,
            output=output,
            python=Path(sys.executable),
            worker=worker,
            runtime_roots=[Path(sys.prefix), Path(sys.base_prefix)],
            forbidden_source=Path(sys.prefix) / "bin",
        )
