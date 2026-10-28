"""Synthetic file binding and strict parity assertions for reference scripts."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


def run_host_executor(tmp_path, monkeypatch, frame, step, code_factory) -> Path:
    run = tmp_path / "host"
    run.mkdir()
    source = run / "synthetic.parquet"
    frame.to_parquet(source, index=False)
    typed_input = next(item for item in step.inputs if ":" in item)
    kind, product = typed_input.split(":", 1)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = run / "resolved_inputs.json"
    manifest.write_text(json.dumps({"step_id": step.step_id, "inputs": {typed_input: {
        "relative_path": source.name, "sha256": digest,
        "declared_kind": kind, "product": product, "evidence_kind": "table",
        "evidence_id": f"sha256:{digest}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
    }}}))
    out = run / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest))
    monkeypatch.setenv("COHORT_PARQUET", str(source))
    monkeypatch.setenv("STEP_OUT_DIR", str(out))
    exec(code_factory(step), {})
    return out


def assert_numeric_parity(actual, expected, *, counts, values):
    """Counts are exact; finite floating inference uses the requested 1e-8 bound."""
    assert_frame_equal(actual[counts], expected[counts], check_exact=True)
    assert_frame_equal(actual[values], expected[values], check_exact=False, atol=1e-8, rtol=1e-8)


def assert_csv_cells_equal(actual: Path, expected: Path):
    assert_frame_equal(
        pd.read_csv(actual, dtype=str, keep_default_na=False),
        pd.read_csv(expected, dtype=str, keep_default_na=False), check_exact=True,
    )
