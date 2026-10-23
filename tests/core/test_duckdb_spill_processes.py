"""Two independent engines sorting disk data from one working directory."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import duckdb

from tests.support.duckdb_spill_worker import digest

ROOT = Path(__file__).resolve().parents[2]


def test_concurrent_datasource_spills_are_private_and_exact():
    root = ROOT / ".codex_lane" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spill-proof-", dir=root) as temporary:
        work = Path(temporary)
        parquet = work / "synthetic.parquet"
        with duckdb.connect(
            config={
                "threads": 1,
                "memory_limit": "512MB",
                "temp_directory": str(work / "reference-spill"),
            }
        ) as reference:
            reference.execute(
                "COPY (SELECT i, hash(i) AS k, i % 97 AS payload "
                "FROM range(2000000) t(i)) TO ? (FORMAT PARQUET)",
                [str(parquet)],
            )
            expected = []
            for offset in (0, 2000000):
                reference.execute(
                    "SELECT i + ?, k, payload FROM read_parquet(?) ORDER BY k, i",
                    [offset, str(parquet)],
                )
                expected.append(digest(reference))
        env = dict(
            os.environ,
            EASYICU_HOME=str(work / "home"),
            EASYICU_DUCKDB_MEMORY_LIMIT="32MB",
            EASYICU_DUCKDB_THREADS="1",
        )
        env.pop("EASYICU_DUCKDB_TEMP_DIR", None)
        env.pop("EASYICU_DUCKDB_MAX_TEMP_SIZE", None)
        env["PYTHONPATH"] = os.environ.get("PYTHONPATH", str(ROOT / "src"))
        workers = [
            subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tests/support/duckdb_spill_worker.py"),
                    str(parquet),
                    str(offset),
                ],
                cwd=work,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for offset in (0, 2000000)
        ]
        try:
            # Pipes synchronize both writers without sockets or a server.
            with ThreadPoolExecutor(max_workers=2) as pool:

                def receive(worker):
                    line = worker.stdout.readline()
                    assert line, worker.stderr.read()
                    return json.loads(line)

                def receive_both():
                    futures = [pool.submit(receive, worker) for worker in workers]
                    try:
                        return [future.result(timeout=60) for future in futures]
                    except BaseException:
                        for worker in workers:
                            worker.kill()
                        raise

                paths = [Path(row["directory"]) for row in receive_both()]
                for worker in workers:
                    worker.stdin.write("start\n")
                    worker.stdin.flush()
                spills = receive_both()
                assert all(row["spilled_bytes"] > 0 for row in spills), spills
                for worker in workers:
                    worker.stdin.write("read\n")
                    worker.stdin.flush()
                results = receive_both()
            for worker in workers:
                assert worker.wait(timeout=15) == 0, worker.stderr.read()
            assert [row["digest"] for row in results] == expected
            print(
                {
                    "directories": [str(p) for p in paths],
                    "spills": spills,
                    "exact_results": True,
                }
            )
            assert paths[0] != paths[1], paths
            assert all(
                path.is_relative_to(work / "home" / ".easyicu") for path in paths
            )
            assert all(not path.exists() for path in paths)
            assert not (work / ".tmp").exists()
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate(timeout=15)
