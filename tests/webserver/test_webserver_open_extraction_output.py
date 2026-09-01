from __future__ import annotations

import time
from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import jobs as job_store
from easyicu.webserver.app import app
from easyicu.webserver.jobs import JobManager
from easyicu.webserver.routes import jobs as job_routes


def _completed_extract_job(manager: JobManager, out_dir: Path):
    result = {
        "out_dir": str(out_dir),
        "files": [{"file": "vitals.parquet", "rows": 12}],
        "definition_files": [
            {"file": "feature_definitions.json", "records": 2},
            {"file": "feature_definitions.csv", "records": 2},
        ],
        "manifest": "_manifest.json",
        "readme": "README.md",
        "column_metadata": "column_metadata.sha256-demo.json",
    }
    job = manager.submit("extract", lambda _job: result)
    deadline = time.time() + 2
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert job.status == "done"
    return job


def test_open_extraction_output_uses_completed_job_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "easyicu_export"
    out_dir.mkdir()
    for name in (
        "vitals.parquet",
        "feature_definitions.json",
        "feature_definitions.csv",
        "_manifest.json",
        "README.md",
        "column_metadata.sha256-demo.json",
    ):
        (out_dir / name).write_text("test", encoding="utf-8")

    manager = JobManager(max_completed=10)
    job = _completed_extract_job(manager, out_dir)
    opened: list[Path] = []
    monkeypatch.setattr(job_store, "MANAGER", manager)
    monkeypatch.setattr(job_routes, "_launch_local_path", opened.append)
    client = TestClient(app)

    folder = client.post(f"/api/jobs/{job.id}/open-output", json={})
    file = client.post(
        f"/api/jobs/{job.id}/open-output",
        json={"file": "vitals.parquet"},
    )
    metadata = client.post(
        f"/api/jobs/{job.id}/open-output",
        json={"file": "column_metadata.sha256-demo.json"},
    )

    assert folder.status_code == 200
    assert folder.json() == {
        "ok": True,
        "target": "folder",
        "name": "easyicu_export",
        "method": "finder",
    }
    assert file.status_code == 200
    assert file.json() == {
        "ok": True,
        "target": "file",
        "name": "vitals.parquet",
        "method": "application",
    }
    assert metadata.status_code == 200
    assert opened == [
        out_dir.resolve(),
        (out_dir / "vitals.parquet").resolve(),
        (out_dir / "column_metadata.sha256-demo.json").resolve(),
    ]


@pytest.mark.parametrize(
    ("kind", "status", "file_name", "expected_status", "error"),
    [
        ("convert", "done", "vitals.parquet", 409, "extraction_output_unavailable"),
        ("extract", "running", "vitals.parquet", 409, "extraction_output_unavailable"),
        ("extract", "done", "../private.txt", 400, "invalid_extraction_output_file"),
        ("extract", "done", "unlisted.json", 404, "extraction_output_file_not_declared"),
    ],
)
def test_open_extraction_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    status: str,
    file_name: str,
    expected_status: int,
    error: str,
) -> None:
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    (out_dir / "vitals.parquet").write_text("test", encoding="utf-8")
    manager = JobManager(max_completed=10)
    job = _completed_extract_job(manager, out_dir)
    job.kind = kind
    job.status = status
    monkeypatch.setattr(job_store, "MANAGER", manager)
    monkeypatch.setattr(job_routes, "_launch_local_path", lambda _path: None)

    response = TestClient(app).post(
        f"/api/jobs/{job.id}/open-output",
        json={"file": file_name},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"]["error"] == error


def test_open_extraction_output_rejects_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    outside = tmp_path / "outside.parquet"
    outside.write_text("private", encoding="utf-8")
    (out_dir / "vitals.parquet").symlink_to(outside)
    manager = JobManager(max_completed=10)
    job = _completed_extract_job(manager, out_dir)
    monkeypatch.setattr(job_store, "MANAGER", manager)
    monkeypatch.setattr(job_routes, "_launch_local_path", lambda _path: None)

    response = TestClient(app).post(
        f"/api/jobs/{job.id}/open-output",
        json={"file": "vitals.parquet"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "extraction_output_path_escape"


def test_local_open_falls_back_to_finder_for_unassociated_macos_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "vitals.parquet"
    target.write_text("test", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> None:
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(job_routes.sys, "platform", "darwin")
    monkeypatch.setattr(job_routes.subprocess, "run", fake_run)

    method = job_routes._launch_local_path(target)

    assert method == "finder"
    assert commands == [
        ["/usr/bin/open", str(target)],
        ["/usr/bin/open", "-R", str(target)],
    ]
