"""Only existing, bounded run metadata may feed the review materials."""

import json


def test_metadata_retains_current_packages_notes_and_source_identity(tmp_path):
    from easyicu.webserver.scientific_review_metadata import recorded_review_metadata

    (tmp_path / "manifest.json").write_text(json.dumps({"per_step_records": [
        {"step_id": "model", "status": "ok", "step_summary": {"package_versions": {"synthetic-lib": "0.1"}}},
        {"step_id": "model", "status": "ok", "step_summary": {
            "scientific_runtime_receipt": {"package_versions": {"synthetic-lib": "0.2"}},
            "data_property_notes": ["Recorded synthetic limitation."],
            "private_path": "/outside/never-open", "rows": [{"patient_id": "not projected"}],
        }},
    ]}))
    source = {"database": "synthetic", "version": "0.3", "export_id": "synthetic-export", "path": "/outside/never-open"}
    result = recorded_review_metadata(tmp_path, source)
    assert result["source"] == {key: source[key] for key in ("database", "version", "export_id")}
    assert [row["value"] for row in result["runtime"] if row["field"] == "package_versions"] == [{"synthetic-lib": "0.2"}]
    assert result["runtime"][0]["step_id"] == "model"
    assert "patient_id" not in json.dumps(result)
    assert "never-open" not in json.dumps(result)


def test_missing_or_symlink_metadata_is_explicit(tmp_path):
    from easyicu.webserver.scientific_review_metadata import recorded_review_metadata

    result = recorded_review_metadata(tmp_path, {})
    assert result["status"] == "unavailable"
    target = tmp_path / "target.json"
    target.write_text('{}')
    (tmp_path / "manifest.json").symlink_to(target)
    assert recorded_review_metadata(tmp_path, {})["status"] == "unavailable"


def test_latest_failed_step_does_not_resurrect_earlier_metadata(tmp_path):
    from easyicu.webserver.scientific_review_metadata import recorded_review_metadata

    (tmp_path / "manifest.json").write_text(json.dumps({"per_step_records": [
        {"step_id": "model", "status": "ok", "step_summary": {"package_versions": {"lib": "0.1"}}},
        {"step_id": "model", "status": "failed", "step_summary": {"package_versions": {"lib": "0.2"}}},
    ]}))
    assert recorded_review_metadata(tmp_path, {})["runtime"] == []
