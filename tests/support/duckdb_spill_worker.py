"""Pipe-synchronized synthetic spill worker; invoked only by the regression test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def digest(connection) -> str:
    checksum = hashlib.sha256()
    while rows := connection.fetchmany(8192):
        checksum.update(repr(rows).encode())
    return checksum.hexdigest()


def main() -> None:
    from easyicu import datasource

    connection = datasource._get_duckdb_connection()
    try:
        connection.execute("SET preserve_insertion_order = false")
        directory = connection.execute(
            "SELECT current_setting('temp_directory')"
        ).fetchone()[0]
        print(json.dumps({"directory": str(Path(directory).absolute())}), flush=True)
        assert sys.stdin.readline().strip() == "start"
        connection.execute(
            "CREATE TEMP TABLE sorted AS SELECT i + ? AS i, k, payload "
            "FROM read_parquet(?) ORDER BY k, i",
            [int(sys.argv[2]), sys.argv[1]],
        )
        spilled = connection.execute(
            "SELECT coalesce(sum(size), 0) FROM duckdb_temporary_files()"
        ).fetchone()[0]
        print(json.dumps({"spilled_bytes": spilled}), flush=True)
        assert sys.stdin.readline().strip() == "read"
        connection.execute("SELECT i, k, payload FROM sorted ORDER BY k, i")
        print(json.dumps({"digest": digest(connection)}), flush=True)
    finally:
        datasource._close_duckdb_connections()


if __name__ == "__main__":
    main()
