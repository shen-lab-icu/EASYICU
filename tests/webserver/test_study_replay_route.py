"""The replay download names a run only by its history-issued ref (TestClient: run on the Mac)."""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from easyicu.webserver import run_refs
from easyicu.webserver.routes import study_replay

FIXTURE = Path(__file__).resolve().parents[2] / "website/replay-fixtures/synthetic-group-comparison"
ROUTE = "/api/agent-runs/download-replay"


def client() -> TestClient:
    app = FastAPI()
    app.include_router(study_replay.router)
    return TestClient(app)


def test_a_listed_run_downloads_a_byte_stable_replay(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    ref = run_refs.issue(str(run))
    with client() as http:
        first = http.post(ROUTE, json={"run_ref": ref})
        second = http.post(ROUTE, json={"run_ref": ref})
    assert first.status_code == 200, first.text
    assert first.headers["content-type"] == "application/zip"
    assert first.headers["cache-control"] == "no-store"
    assert first.content == second.content
    archive = zipfile.ZipFile(io.BytesIO(first.content))
    names = set(archive.namelist())
    assert {"easyicu-replay/index.html", "easyicu-replay/replay.json", "easyicu-replay/manifest.json"} <= names
    replay = archive.read("easyicu-replay/replay.json").decode("utf-8")
    assert json.loads(replay)["run_id"]
    assert str(tmp_path) not in replay


def test_only_a_listed_ref_names_a_run():
    with client() as http:
        assert http.post(ROUTE, json={"run_ref": "rr_" + "0" * 32}).json()["detail"]["error"] == "run_ref_unknown"
        # A client path is never a coordinate here.
        assert http.post(ROUTE, json={"project_dir": str(FIXTURE)}).status_code == 404


def test_an_exporter_refusal_returns_its_stable_code(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(FIXTURE, run)
    table = run / "evidence/aggregate.csv"
    table.write_bytes(table.read_bytes() + b"\n")
    with client() as http:
        refused = http.post(ROUTE, json={"run_ref": run_refs.issue(str(run))})
    assert refused.status_code == 409
    assert refused.json()["detail"]["error"].startswith("replay_")
    assert str(tmp_path) not in refused.text
