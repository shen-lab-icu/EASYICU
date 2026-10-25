from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from easyicu.webserver import agent_pipeline_runs, agent_runs, run_refs
from easyicu.webserver.routes.agent import post_agent_run_download_artifact


def _figure_run(root: Path) -> tuple[dict, list[dict]]:
    (root / "figures").mkdir(parents=True)
    (root / "evidence").mkdir()
    records = []
    exports = {}
    for fmt in ("pdf", "svg", "tiff", "png"):
        raw = f"synthetic aggregate figure {fmt}".encode()
        relative = f"figures/result.{fmt}"
        (root / relative).write_bytes(raw)
        registered = f"evidence/registered.{fmt}"
        (root / registered).write_bytes(raw)
        records.append({
            "evidence_id": f"synthetic_{fmt}", "kind": "figure",
            "relative_path": registered, "sha256": hashlib.sha256(raw).hexdigest(),
        })
        exports[fmt] = relative
    row = {
        "relative_path": "figures/result.png", "exports": exports,
        "label": "Aggregate result", "caption": "Synthetic aggregate comparison.",
        "data_url": "data:image/png;base64," + base64.b64encode(b"synthetic png").decode(),
    }
    _write(root, row, records)
    return row, records


def _write(root: Path, row: dict, records: list[dict]) -> None:
    (root / "figure_gallery.json").write_text(json.dumps({"figures": [row]}))
    (root / "evidence/evidence_index.json").write_text(json.dumps(records))


def _request(root: Path, fmt: str = "pdf") -> dict:
    records = json.loads((root / "evidence/evidence_index.json").read_text())
    record = next(record for record in records if record["evidence_id"] == f"synthetic_{fmt}")
    return {
        "run_ref": run_refs.issue(str(root)),
        "figure_export": {"format": fmt, **{key: record[key] for key in ("evidence_id", "sha256", "kind")}},
    }


def _refused(body: dict, code: str) -> None:
    with pytest.raises(HTTPException) as refused:
        post_agent_run_download_artifact(body)
    assert refused.value.status_code == 400
    assert refused.value.detail == {"ok": False, "error": code, "run_ref": body["run_ref"]}


@pytest.mark.parametrize("fmt,media_type", [
    ("pdf", "application/pdf"), ("svg", "image/svg+xml"),
    ("tiff", "image/tiff"), ("png", "image/png"),
])
def test_download_route_accepts_only_the_runs_validated_figure_locator(tmp_path, fmt, media_type):
    _figure_run(tmp_path)
    locator = agent_pipeline_runs._figure_projection(tmp_path)["figures"][0]["export_locators"][fmt]
    body = {"run_ref": run_refs.issue(str(tmp_path)), "figure_export": {"format": fmt, **locator}}

    response = post_agent_run_download_artifact(body)

    assert response.body == (tmp_path / f"figures/result.{fmt}").read_bytes()
    assert response.media_type == media_type
    assert response.headers["content-disposition"] == f'attachment; filename="synthetic_{fmt}.{fmt}"'


@pytest.mark.parametrize("location", ["alias", "record"])
@pytest.mark.parametrize("unsafe", ["../outside.pdf", "/outside.pdf", "figures/../result.pdf"])
def test_download_refuses_paths_outside_the_run_or_noncanonical_paths(tmp_path, location, unsafe):
    row, records = _figure_run(tmp_path)
    body = _request(tmp_path)
    if location == "alias":
        row["exports"]["pdf"] = unsafe
    else:
        records[0]["relative_path"] = unsafe
    _write(tmp_path, row, records)
    _refused(body, "figure_export_not_validated" if location == "alias" else "figure_export_path_unsafe")


@pytest.mark.parametrize("field,value,code", [
    ("format", "html", "figure_export_format_not_allowed"),
    ("format", "PDF", "figure_export_format_not_allowed"),
    ("kind", "table", "figure_export_locator_invalid"),
    ("sha256", "invalid", "figure_export_locator_invalid"),
    ("evidence_id", "../private", "figure_export_locator_invalid"),
    ("relative_path", "figures/result.pdf", "figure_export_locator_invalid"),
    ("sha256", "0" * 64, "figure_export_digest_mismatch"),
    ("evidence_id", "unregistered", "figure_export_not_validated"),
])
def test_download_refuses_invalid_or_stale_locators(tmp_path, field, value, code):
    _figure_run(tmp_path)
    body = _request(tmp_path)
    body["figure_export"][field] = value
    _refused(body, code)


@pytest.mark.parametrize("changed", ["alias", "record", "registry", "kind", "gallery"])
def test_download_revalidates_the_gallery_registry_and_actual_bytes(tmp_path, changed):
    row, records = _figure_run(tmp_path)
    body = _request(tmp_path)
    code = "figure_export_not_validated"
    if changed in {"alias", "record"}:
        path = row["exports"]["pdf"] if changed == "alias" else records[0]["relative_path"]
        (tmp_path / path).write_bytes(b"changed aggregate figure")
        if changed == "record":
            code = "figure_export_digest_mismatch"
    elif changed == "registry":
        records[0]["sha256"] = "0" * 64
        code = "figure_export_digest_mismatch"
    elif changed == "kind":
        records[0]["kind"] = "table"
    else:
        row["exports"] = {}
    _write(tmp_path, row, records)
    _refused(body, code)


@pytest.mark.parametrize("location", ["alias", "record", "index_directory"])
def test_download_does_not_follow_figure_or_registry_symlinks(tmp_path, location):
    row, records = _figure_run(tmp_path)
    body = _request(tmp_path)
    if location == "index_directory":
        (tmp_path / "evidence").rename(tmp_path / "original_evidence")
        (tmp_path / "evidence").symlink_to(tmp_path / "original_evidence", target_is_directory=True)
    else:
        relative = row["exports"]["pdf"] if location == "alias" else records[0]["relative_path"]
        path = tmp_path / relative
        raw = path.read_bytes()
        path.unlink()
        (tmp_path / "replacement.pdf").write_bytes(raw)
        path.symlink_to(tmp_path / "replacement.pdf")
    _refused(body, "figure_export_not_validated" if location == "alias" else "figure_export_path_unsafe")


def test_download_does_not_accept_another_runs_locator(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _figure_run(first)
    row, records = _figure_run(second)
    body = _request(first)
    records[0]["evidence_id"] = "other_pdf"
    _write(second, row, records)
    body["run_ref"] = run_refs.issue(str(second))
    _refused(body, "figure_export_not_validated")


@pytest.mark.parametrize("fmt", ["html", "csv"])
def test_even_a_registered_gallery_export_requires_an_allowed_format(tmp_path, fmt):
    row, records = _figure_run(tmp_path)
    raw = b"synthetic aggregate export"
    relative = f"figures/result.{fmt}"
    (tmp_path / relative).write_bytes(raw)
    records.append({
        "evidence_id": f"synthetic_{fmt}", "kind": "figure", "relative_path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
    })
    row["exports"][fmt] = relative
    _write(tmp_path, row, records)
    _refused(_request(tmp_path, fmt), "figure_export_format_not_allowed")


def test_a_registered_export_cannot_escape_to_matching_bytes_outside_the_run(tmp_path):
    root = tmp_path / "run"
    row, records = _figure_run(root)
    body = _request(root)
    (tmp_path / "outside.pdf").write_bytes((root / records[0]["relative_path"]).read_bytes())
    records[0]["relative_path"] = "../outside.pdf"
    _write(root, row, records)
    _refused(body, "figure_export_path_unsafe")


def test_registry_extension_must_match_the_selected_format(tmp_path):
    row, records = _figure_run(tmp_path)
    body = _request(tmp_path)
    relative = "evidence/pdf_bytes.svg"
    (tmp_path / relative).write_bytes((tmp_path / records[0]["relative_path"]).read_bytes())
    records[0]["relative_path"] = relative
    _write(tmp_path, row, records)
    _refused(body, "figure_export_format_not_allowed")


def test_registration_alone_does_not_authorize_a_figure_download(tmp_path):
    row, records = _figure_run(tmp_path)
    body = _request(tmp_path)
    row["exports"]["pdf"] = "figures/unlisted.pdf"
    (tmp_path / "figures/unlisted.pdf").write_bytes((tmp_path / "figures/result.pdf").read_bytes())
    _write(tmp_path, row, records)
    _refused(body, "figure_export_not_validated")


def test_duplicate_evidence_ids_do_not_authorize_a_download(tmp_path):
    row, records = _figure_run(tmp_path)
    body = _request(tmp_path)
    records.append(dict(records[0]))
    _write(tmp_path, row, records)
    _refused(body, "figure_export_not_validated")


@pytest.mark.parametrize("value", [None, [], "figures/result.pdf", {}])
def test_malformed_figure_selection_is_refused_with_a_typed_reason(tmp_path, value):
    _figure_run(tmp_path)
    _refused(
        {"run_ref": run_refs.issue(str(tmp_path)), "figure_export": value},
        "figure_export_format_not_allowed" if isinstance(value, dict) else "figure_export_locator_invalid",
    )


def test_hosted_figure_download_rechecks_instance_ownership(tmp_path, monkeypatch):
    _figure_run(tmp_path)
    body = _request(tmp_path)
    instance_home = tmp_path / "instance_home"
    instance_home.mkdir()
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_HOME", str(instance_home))
    monkeypatch.setattr(agent_runs.run_figure_exports, "read_figure_export_bytes", lambda *_: pytest.fail("read outside instance"))
    with pytest.raises(HTTPException) as refused:
        post_agent_run_download_artifact(body)
    assert refused.value.status_code == 403
    assert refused.value.detail["error"] == "hosted_data_read_only"


def test_wrapper_gallery_downloads_through_its_declared_pipeline_run(tmp_path):
    pipeline = tmp_path / "pipeline/run_synthetic"
    _figure_run(pipeline)
    gallery = agent_pipeline_runs._figure_projection(pipeline)
    (tmp_path / "figure_gallery.json").write_text(json.dumps(gallery))
    (tmp_path / "source_run_manifest.json").write_text(json.dumps({"run_id": "run_synthetic"}))

    shown = agent_runs.read_run_artifact(str(tmp_path), "figure_gallery.json")["payload"]
    assert shown["run_ref"] == run_refs.issue(str(tmp_path))
    assert "relative_path" not in json.dumps(shown["figures"][0]["export_locators"])
    response = post_agent_run_download_artifact({
        "run_ref": shown["run_ref"],
        "figure_export": {"format": "pdf", **shown["figures"][0]["export_locators"]["pdf"]},
    })
    assert response.body == (pipeline / "figures/result.pdf").read_bytes()


def test_plain_paths_and_patient_row_artifacts_remain_refused(tmp_path):
    _figure_run(tmp_path)
    with pytest.raises(HTTPException) as path:
        post_agent_run_download_artifact({"project_dir": str(tmp_path), "artifact": "figures/result.pdf"})
    assert path.value.detail["error"] == "artifact_not_allowed"
    (tmp_path / "result_tables.json").write_text(json.dumps({"patient_rows": [{"age": 54}]}))
    with pytest.raises(HTTPException) as rows:
        post_agent_run_download_artifact({"project_dir": str(tmp_path), "artifact": "result_tables.json"})
    assert rows.value.status_code == 409
    assert rows.value.detail["error"] == "artifact_privacy_scan_failed"
