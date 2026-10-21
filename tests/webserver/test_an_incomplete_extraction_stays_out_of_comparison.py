"""An extraction the comparison cannot line up is not selected for it.

Every finished extraction is registered and joined the cross-database
selection. An extraction of a few modules, without demographics or
outcome, then made the default comparison fail with crossdb_incompatible
(found on the hospital server, 10-21). It is still registered and active,
but only an export holding the gate's core modules joins the comparison
selection. Fixtures are synthetic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import crossdb_review, dataio
from easyicu.webserver import sources as source_store
from easyicu.webserver.app import app


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source_store, "_CONFIG_DIR", tmp_path / "source-cfg")
    monkeypatch.setattr(source_store, "_CONFIG_PATH", tmp_path / "source-cfg" / "sources.json")
    monkeypatch.setattr(source_store, "_autodiscovered_paths", lambda: [])


def _fake_export(monkeypatch: pytest.MonkeyPatch) -> None:
    def make_export_runner(**kwargs: Any):
        def runner(job: Any) -> dict[str, Any]:
            out = Path(kwargs["out_dir"])
            out.mkdir(parents=True)
            files = []
            for module in kwargs["modules"]:
                (out / f"{module}.csv").write_text("stay_id,value\n1,0.5\n2,1.5\n", encoding="utf-8")
                files.append({"file": f"{module}.csv", "rows": 2, "module": module})
            manifest = {"schema_version": "easyicu_native_export_v2", "database": "miiv",
                        "format": "csv", "patient_count": 2, "files": files}
            (out / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return {"out_dir": str(out), "manifest": "_manifest.json", "files": files}

        return runner

    monkeypatch.setattr(dataio, "make_export_runner", make_export_runner)


def _extract(tmp_path: Path, name: str, modules: list[str]) -> dict[str, Any]:
    client = TestClient(app)
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    job = client.post("/api/jobs/extract", json={
        "path": str(raw), "database": "miiv", "modules": modules,
        "out_dir": str(tmp_path / name), "label": name,
    }).json()
    deadline = time.time() + 10
    while time.time() < deadline:
        snapshot = client.get(f"/api/jobs/{job['job_id']}").json()
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.02)
    raise AssertionError("extraction did not finish")


def test_only_an_export_with_the_core_modules_joins_the_comparison(tmp_path, monkeypatch):
    _fake_export(monkeypatch)
    core = ["demographics", "outcome", "vitals"]
    for name in ("prepared_eicu", "prepared_miiv"):
        assert _extract(tmp_path, name, core)["status"] == "done"
    assert _extract(tmp_path, "vitals_only", ["vitals"])["status"] == "done"

    registry = source_store.load_registry()
    path = {row["label"]: row["path"] for row in registry["sources"]}
    assert set(path) == {"prepared_eicu", "prepared_miiv", "vitals_only"}
    # The latest extraction is active for the review pages...
    assert registry["active_path"] == path["vitals_only"]
    # ...but the comparison keeps the two exports it can line up.
    assert registry["crossdb_paths"] == [path["prepared_eicu"], path["prepared_miiv"]]

    assert _extract(tmp_path, "my_cohort", core)["status"] == "done"
    registry = source_store.load_registry()
    path = {row["label"]: row["path"] for row in registry["sources"]}
    assert registry["crossdb_paths"] == [path["prepared_eicu"], path["prepared_miiv"], path["my_cohort"]]


@pytest.mark.parametrize(("modules", "joins"), [
    (["demographics", "outcome"], True),
    (["outcome", "vitals", "demographics"], True),
    (["demographics", "vitals"], False),
    ([], False),
    (None, False),
])
def test_the_gate_core_decides_who_joins(modules, joins):
    assert crossdb_review.can_join_comparison(modules) is joins
