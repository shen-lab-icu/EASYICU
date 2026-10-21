"""The conversation's run files name a run by an opaque ref, never its folder.

Counted eICU Q1' attempts 7-10 (2026-10-20/21) left absolute host folders in
the conversation's ``data-run-files`` attributes.  The run history now issues
a ``run_ref`` for each run it lists when asked for path-free rows, and the
review, artifact, download and sign-off routes resolve that ref on the server;
an unknown ref is refused rather than read as some other folder.  The classic
workspace's local ``project_dir`` coordinate keeps working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import agent_runs, run_refs
from easyicu.webserver.app import app


def _run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "projects" / "study-1" / "run_job-1"
    run_dir.mkdir(parents=True)
    return run_dir.resolve()


def _row(run_dir: Path) -> dict[str, Any]:
    return {"run_id": "run_candidate", "study_id": "study-1", "project_dir": str(run_dir),
            "run_status": "completed", "updated_at_epoch": 1}


@pytest.fixture
def listed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = _run(tmp_path)
    monkeypatch.setattr(agent_runs, "list_run_history", lambda **_kw: {
        "ok": True, "project_root": str(tmp_path / "projects"), "study_id": "study-1",
        "runs": [_row(run_dir)], "count": 1,
    })
    read: list[str] = []

    def review(project_dir: str) -> dict[str, Any]:
        read.append(project_dir)
        return {"ok": True, "project_dir": project_dir, "run_id": "run_candidate", "study_id": "study-1",
                "artifacts": [{"name": "result_tables.json", "path": f"{project_dir}/result_tables.json",
                               "relative_path": "result_tables.json", "sha256": "a" * 64}],
                "artifact_payloads": {"run_context.json": {"source": {"label": "eICU", "path": "/Volumes/x/eicu"}}}}

    monkeypatch.setattr(agent_runs, "read_run_review", review)
    monkeypatch.setattr(agent_runs, "read_run_artifact", lambda project_dir, name: {
        "ok": True, "project_dir": project_dir,
        "artifact": {"name": name, "path": f"{project_dir}/{name}", "relative_path": name},
        "payload": {"tables": []},
    })
    return run_dir, read


def test_path_free_history_names_each_run_by_its_ref(tmp_path: Path, listed) -> None:
    run_dir, _read = listed
    client = TestClient(app)

    body = client.post("/api/agent-runs/history", json={
        "study_id": "study-1", "project_root": str(tmp_path / "projects"), "path_free": True,
    }).json()

    [row] = body["runs"]
    assert row["run_id"] == "run_candidate"
    assert row["run_ref"].startswith("rr_") and len(row["run_ref"]) == 35
    assert run_refs.resolve(row["run_ref"]) == str(run_dir)
    dumped = json.dumps(body)
    assert str(tmp_path) not in dumped and str(run_dir) not in dumped
    assert not set(run_refs.PATH_FIELDS) & set(body) and "project_dir" not in row

    # Without the flag the classic workspace keeps its local coordinate.
    legacy = client.post("/api/agent-runs/history", json={
        "study_id": "study-1", "project_root": str(tmp_path / "projects"),
    }).json()
    assert legacy["runs"][0]["project_dir"] == str(run_dir)


def test_a_ref_is_resolved_on_the_server_and_answered_by_ref(tmp_path: Path, listed) -> None:
    run_dir, read = listed
    client = TestClient(app)
    ref = run_refs.issue(str(run_dir))

    review = client.post("/api/agent-runs/review", json={"run_ref": ref})
    assert review.status_code == 200
    assert read == [str(run_dir)]
    shown = review.json()
    assert shown["run_ref"] == ref and "project_dir" not in shown
    assert shown["artifacts"] == [{"name": "result_tables.json", "relative_path": "result_tables.json", "sha256": "a" * 64}]
    assert shown["artifact_payloads"]["run_context.json"]["source"] == {"label": "eICU"}

    artifact = client.post("/api/agent-runs/artifact", json={"run_ref": ref, "artifact": "result_tables.json"})
    assert artifact.status_code == 200
    assert artifact.json()["artifact"] == {"name": "result_tables.json", "relative_path": "result_tables.json"}
    for response in (review, artifact):
        assert str(tmp_path) not in response.text and "/Volumes/" not in response.text


@pytest.mark.parametrize("ref", ["rr_" + "0" * 32, "rr_short", "../../etc", "/Users/someone/run"])
def test_an_unknown_ref_is_refused_not_read(listed, ref: str) -> None:
    _run_dir, read = listed
    client = TestClient(app)
    for endpoint in ("/api/agent-runs/review", "/api/agent-runs/artifact",
                     "/api/agent-runs/download-artifact", "/api/agent-runs/download-bundle",
                     "/api/agent-runs/signoff"):
        response = client.post(endpoint, json={"run_ref": ref, "artifact": "result_tables.json"})
        assert response.status_code == 404, endpoint
        assert response.json()["detail"] == {"ok": False, "error": "run_ref_unknown"}
    assert read == []
