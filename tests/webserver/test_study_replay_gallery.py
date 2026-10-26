"""Gallery registration pins public or synthetic recorded inputs, without runs."""

import json
from pathlib import Path

import pytest


WEBSITE = Path(__file__).resolve().parents[2] / "website"


def test_gallery_builds_registered_fixture_replays_byte_stably(tmp_path):
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    first = build_replay_gallery(WEBSITE, "replay_registry.json")
    assert first == build_replay_gallery(WEBSITE, "replay_registry.json")
    gallery = json.loads(first["gallery.json"])
    assert len(gallery["entries"]) == 2
    for entry in gallery["entries"]:
        assert entry["source_kind"] == "synthetic"
        assert entry["real_data_acceptance"] == "not_accepted"
        assert entry["question"] and entry["title"] and entry["study_type"]
        assert entry["replay"] in first
        assert entry["database"] == "synthetic"
        assert entry["replay_manifest_sha256"]


def test_gallery_rejects_unregistered_or_changed_run(tmp_path):
    import shutil
    from easyicu.webserver.study_replay import ReplayExportError
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    shutil.copytree(WEBSITE / "replay-fixtures", tmp_path / "replay-fixtures")
    registry = json.loads((WEBSITE / "replay_registry.json").read_bytes())
    registry["entries"][0]["source_manifest_sha256"] = "0" * 64
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    with pytest.raises(ReplayExportError, match="gallery_registration_mismatch"):
        build_replay_gallery(tmp_path, "registry.json")


def test_gallery_never_admits_credentialed_runs(tmp_path):
    import hashlib
    import shutil
    from easyicu.webserver.study_replay import ReplayExportError
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    shutil.copytree(WEBSITE / "replay-fixtures", tmp_path / "replay-fixtures")
    registry = json.loads((WEBSITE / "replay_registry.json").read_bytes())
    registry["entries"] = registry["entries"][:1]
    row = registry["entries"][0]
    row["source_kind"] = "synthetic"  # Caller declaration cannot upgrade the run.
    root = tmp_path / row["run_dir"]
    for name in ("run_context.json", "source_run_manifest.json"):
        record = json.loads((root / name).read_bytes())
        record["source"] = {"database": "miiv", "label": "Credentialed fixture"}
        (root / name).write_text(json.dumps(record))
    row["source_manifest_sha256"] = hashlib.sha256(
        (root / "source_run_manifest.json").read_bytes()
    ).hexdigest()
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    with pytest.raises(ReplayExportError, match="gallery_source_forbidden"):
        build_replay_gallery(tmp_path, "registry.json")


def test_gallery_refuses_duplicates_and_escaping_paths(tmp_path):
    from easyicu.webserver.study_replay import ReplayExportError
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    registry = json.loads((WEBSITE / "replay_registry.json").read_bytes())
    row = registry["entries"][0]
    row["run_dir"] = "../escape"
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    with pytest.raises(ReplayExportError, match="replay_path_forbidden"):
        build_replay_gallery(tmp_path, "registry.json")


def test_gallery_refuses_empty_registry(tmp_path):
    from easyicu.webserver.study_replay import ReplayExportError
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    (tmp_path / "registry.json").write_text(
        '{"schema_version":"easyicu.replay-gallery-registry/1","entries":[]}'
    )
    with pytest.raises(ReplayExportError, match="gallery_registry_invalid"):
        build_replay_gallery(tmp_path, "registry.json")


def test_registration_is_checked_before_any_evidence_read(tmp_path, monkeypatch):
    import shutil
    from easyicu.webserver.study_replay import ReplayExportError
    import easyicu.webserver.study_replay_gallery as gallery

    shutil.copytree(WEBSITE / "replay-fixtures", tmp_path / "replay-fixtures")
    registry = json.loads((WEBSITE / "replay_registry.json").read_bytes())
    registry["entries"][0]["source_manifest_sha256"] = "0" * 64
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    monkeypatch.setattr(
        gallery,
        "build_study_replay",
        lambda *_: pytest.fail("evidence read before registration"),
    )
    with pytest.raises(ReplayExportError, match="gallery_registration_mismatch"):
        gallery.build_replay_gallery(tmp_path, "registry.json")


def test_gallery_rechecks_the_snapshot_actually_exported(monkeypatch):
    import hashlib
    from easyicu.webserver.study_replay import ReplayExportError
    import easyicu.webserver.study_replay_gallery as gallery

    original = gallery.build_study_replay

    def changed(root):
        bundle = original(root)
        manifest = json.loads(bundle["manifest.json"])
        for row in manifest["inputs"]:
            if row["record"] == "source_run_manifest.json":
                row["sha256"] = hashlib.sha256(b"changed snapshot").hexdigest()
        bundle["manifest.json"] = json.dumps(manifest).encode()
        return bundle

    monkeypatch.setattr(gallery, "build_study_replay", changed)
    with pytest.raises(ReplayExportError, match="gallery_registration_mismatch"):
        gallery.build_replay_gallery(WEBSITE, "replay_registry.json")


def test_duplicate_gallery_identity_is_refused(tmp_path):
    import shutil
    from easyicu.webserver.study_replay import ReplayExportError
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    shutil.copytree(WEBSITE / "replay-fixtures", tmp_path / "replay-fixtures")
    registry = json.loads((WEBSITE / "replay_registry.json").read_bytes())
    registry["entries"].append(registry["entries"][0])
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    with pytest.raises(ReplayExportError, match="gallery_registry_invalid"):
        build_replay_gallery(tmp_path, "registry.json")


def test_gallery_admits_registered_public_demo_contracts(tmp_path):
    import hashlib
    import shutil
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    shutil.copytree(WEBSITE / "replay-fixtures", tmp_path / "replay-fixtures")
    registry = json.loads((WEBSITE / "replay_registry.json").read_bytes())
    registry["entries"] = registry["entries"][:1]
    row = registry["entries"][0]
    root = tmp_path / row["run_dir"]
    for name in ("run_context.json", "source_run_manifest.json"):
        value = json.loads((root / name).read_bytes())
        value["source"] = {
            "database": "eicu_demo",
            "label": "Demo contract fixture",
            "demo_source_id": "eicu_demo_v2_0_1",
        }
        (root / name).write_text(json.dumps(value))
    row["source_manifest_sha256"] = hashlib.sha256(
        (root / "source_run_manifest.json").read_bytes()
    ).hexdigest()
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    entry = json.loads(build_replay_gallery(tmp_path, "registry.json")["gallery.json"])[
        "entries"
    ][0]
    assert entry["source_kind"] == "public_demo"
    assert entry["real_data_acceptance"] == "not_accepted"


def test_gallery_refuses_a_credentialed_validation_source(tmp_path):
    import shutil
    from easyicu.webserver.study_replay import ReplayExportError
    from easyicu.webserver.study_replay_gallery import build_replay_gallery

    shutil.copytree(WEBSITE / "replay-fixtures", tmp_path / "replay-fixtures")
    registry = json.loads((WEBSITE / "replay_registry.json").read_bytes())
    root = tmp_path / registry["entries"][0]["run_dir"]
    plan = json.loads((root / "agent_plan.json").read_bytes())
    plan["external_validation"] = {"database": "eicu"}
    (root / "agent_plan.json").write_text(json.dumps(plan))
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    with pytest.raises(ReplayExportError, match="gallery_source_forbidden"):
        build_replay_gallery(tmp_path, "registry.json")
