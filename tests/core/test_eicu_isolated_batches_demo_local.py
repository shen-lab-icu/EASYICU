"""Optional offline check: per-batch process isolation must not change eICU output.

``(eicu, circulatory)`` and ``(eicu, vitals)`` run each streamed batch in a
fresh interpreter (``_ISOLATED_STREAM_BATCH_TARGETS``). Isolation changes only
the process boundary, so the exported parquet must be byte-identical to the
in-process run at the same batch size.

Set EASYICU_EICU_DEMO_RAW to the converted public eICU demo (v2.0.1) parquet
directory. It is read only; outputs go to pytest's temporary directory. Only
hashes, row counts and column names are compared or reported, never rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.api import extraction as api
from easyicu.api.extraction import EXTRACT_MODULES


RAW = Path(os.environ.get(
    "EASYICU_EICU_DEMO_RAW",
    str(Path.home() / ".easyicu" / "demo_sources" / "eicu_demo_v2_0_1" / "raw"),
))
# 2,520 demo stays in four batches: enough boundaries to expose state carried
# from one batch to the next, few enough spawns to keep the check short.
BATCH_SIZE = 700


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export(module: str, out: Path, *, isolated: bool, monkeypatch) -> Path:
    targets = set(api._ISOLATED_STREAM_BATCH_TARGETS)
    if isolated:
        targets.add(("eicu", module))
    else:
        targets.discard(("eicu", module))
    monkeypatch.setattr(api, "_ISOLATED_STREAM_BATCH_TARGETS", frozenset(targets))
    ids = sorted(
        pd.read_parquet(RAW / "patient.parquet", columns=["patientunitstayid"])
        ["patientunitstayid"].tolist()
    )
    out.mkdir(parents=True)
    api._run_module_extraction(
        module,
        list(EXTRACT_MODULES[module]),
        "eicu",
        str(RAW),
        {"patientunitstayid": ids},
        BATCH_SIZE,
        str(out),
        stream_output_batches=True,
    )
    manifest = json.loads((out / "_manifest.json").read_text())
    assert manifest["errors"] == [], f"{module} isolated={isolated}: {len(manifest['errors'])} errors"
    assert manifest["batch_process_isolation"] is isolated
    return Path(manifest["saved"][module]["path"])


@pytest.mark.slow
@pytest.mark.skipif(
    not (RAW / "patient.parquet").is_file(),
    reason="Converted public eICU demo parquet directory is absent",
)
@pytest.mark.parametrize("module", ["circulatory", "vitals"])
def test_isolated_batches_export_the_same_bytes_as_in_process(module, tmp_path, monkeypatch):
    isolated = _export(module, tmp_path / "isolated", isolated=True, monkeypatch=monkeypatch)
    in_process = _export(module, tmp_path / "in_process", isolated=False, monkeypatch=monkeypatch)

    left = pd.read_parquet(isolated)
    right = pd.read_parquet(in_process)
    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right) > 0
    assert _sha256(isolated) == _sha256(in_process)
