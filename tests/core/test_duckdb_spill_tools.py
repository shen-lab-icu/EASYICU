"""Exercise utility connection owners without starting discovery or release runs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

from easyicu.runtime import duckdb_spill

ROOT = Path(__file__).resolve().parents[2]


def _functions(relative, names, **bindings):
    # Load the actual functions without executing the CLI's module-level
    # discovery setup or importing a real-run harness.
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"Path": Path, "argparse": argparse, "pq": pq, **bindings}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def private_engines(monkeypatch, tmp_path):
    paths = []
    original = duckdb_spill.connect

    def record(*args, **kwargs):
        conn = original(*args, **kwargs)
        path = Path(
            conn.execute("SELECT current_setting('temp_directory')").fetchone()[0]
        )
        assert path.is_relative_to(tmp_path / "home" / ".easyicu")
        paths.append(path)
        return conn

    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("EASYICU_DUCKDB_TEMP_DIR", raising=False)
    monkeypatch.setattr(duckdb_spill, "connect", record)
    yield paths
    duckdb_spill._cleanup()
    assert paths and all(not path.exists() for path in paths)


def test_ventilator_verifier_opens_private_engine(
    monkeypatch, tmp_path, private_engines
):
    calls = []
    namespace = _functions(
        "tools/verify_aumc_ventilator_sources.py",
        ["main"],
        verify=lambda args, output, conn: calls.append(
            conn.execute("SELECT 42").fetchone()
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--before",
            "unused",
            "--after",
            "unused",
            "--raw",
            "unused",
            "--output",
            str(tmp_path / "receipt.json"),
        ],
    )
    namespace["main"]()
    assert calls == [(42,)]
    assert len(private_engines) == 1


def test_concept_probe_opens_private_engine(tmp_path, private_engines):
    namespace = _functions(
        "tools/run_concept_proposer.py", ["make_duckdb_probe"], _TABLE_SOURCES={}
    )
    probe = namespace["make_duckdb_probe"](tmp_path)
    assert probe([], "synthetic") == {}
    assert len(private_engines) == 1


def test_full_wide_builder_opens_private_engine(tmp_path, private_engines):
    source = tmp_path / "export"
    source.mkdir()
    pd.DataFrame({"stay_id": [1, 1, 2], "value": [3, 4, 5]}).to_parquet(
        source / "demographics.parquet"
    )
    namespace = _functions(
        "tools/run_idea_mining_real_discovery.py", ["build_full_wide_cohort"]
    )
    output = tmp_path / "wide.parquet"
    result = namespace["build_full_wide_cohort"](source, output)
    assert result["n_stays"] == 2
    assert pd.read_parquet(output).sort_values("stay_id").value.tolist() == [4, 5]
    assert len(private_engines) == 1


def test_universe_builder_opens_private_engine(tmp_path, private_engines):
    source = tmp_path / "synthetic.parquet"
    pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}).to_parquet(source)
    output = tmp_path / "universe.parquet"
    namespace = _functions(
        "tools/build_discovery_universe.py",
        ["main", "normalize_measurement_companions"],
        OUT=output,
        file_for=lambda concept: source,
        MEASUREMENT=[],
        EVENT=[],
        DURATION_ZERO=[],
        STATIC=[],
        OUTCOME=["death"],
    )
    namespace["main"]()
    assert pd.read_parquet(output).sort_values("stay_id").death.tolist() == [0, 1]
    assert len(private_engines) == 1
