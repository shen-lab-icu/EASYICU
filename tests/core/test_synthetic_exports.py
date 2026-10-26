"""Synthetic clearance follows only the host producer's instance-owned receipt."""

import json
from pathlib import Path

import pytest

from easyicu.synthetic_exports import record_generated_export, verified_export_source
from easyicu.synthetic_sources import generate_source


@pytest.fixture
def produced(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "app"))
    source, export = tmp_path / "source", tmp_path / "export"
    proof = generate_source(source, "sic", stays=3)
    export.mkdir()
    manifest = {"database": "sic", "source_kind": "synthetic",
                "files": [{"file": "fixture.csv"}], "column_metadata": {"file": "columns.json"}}
    (export / "_manifest.json").write_text(json.dumps(manifest))
    (export / "fixture.csv").write_text("stay_id,age\n1,65\n")
    (export / "columns.json").write_text("{}")
    (export / "README.md").write_text(proof["notice"])
    return source, export, proof


def test_manifest_claims_alone_grant_nothing(produced):
    source, export, proof = produced
    assert verified_export_source(export) is None
    record_generated_export(source, export, source_proof=None)
    assert verified_export_source(export) is None
    record_generated_export(source, export, source_proof=proof)
    assert verified_export_source(export) == proof


@pytest.mark.parametrize("defect", ["source", "export", "manifest", "extra", "symlink", "different_instance", "receipt_schema", "receipt_path"])
def test_receipt_rechecks_all_bound_content(produced, monkeypatch, tmp_path, defect):
    source, export, proof = produced
    record_generated_export(source, export, source_proof=proof)
    if defect == "source":
        (source / "README.md").write_text("changed input")
    elif defect == "export":
        (export / "fixture.csv").write_text("stay_id,age\n1,66\n")
    elif defect == "manifest":
        manifest = json.loads((export / "_manifest.json").read_text())
        manifest["database"] = "hirid"
        (export / "_manifest.json").write_text(json.dumps(manifest))
    elif defect == "extra":
        (export / "additional.csv").write_text("unverified data")
    elif defect == "symlink":
        member = export / "fixture.csv"
        moved = tmp_path / "moved.csv"
        member.rename(moved)
        member.symlink_to(moved)
    elif defect.startswith("receipt_"):
        from easyicu.state_paths import state_root
        path = next((state_root() / "synthetic_export_receipts").glob("*.json"))
        receipt = json.loads(path.read_text())
        receipt["export" if defect == "receipt_path" else "schema_version"] = "not this export"
        path.write_text(json.dumps(receipt))
    else:
        monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "other_app"))
    assert verified_export_source(export) is None


@pytest.mark.parametrize("defect", ["source", "extra", "database", "kind"])
def test_producer_refuses_changed_input_and_unowned_outputs(produced, defect):
    source, export, proof = produced
    if defect == "source":
        (source / "README.md").write_text("changed during extraction")
    elif defect == "extra":
        (export / "additional.csv").write_text("not produced here")
    else:
        path = export / "_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["database" if defect == "database" else "source_kind"] = "hirid" if defect == "database" else "credentialed"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        record_generated_export(source, export, source_proof=proof)
    assert verified_export_source(export) is None


def test_receipt_is_persistent_instance_state_outside_the_export(produced):
    import importlib
    import easyicu.synthetic_exports as owner
    from easyicu.state_paths import state_root

    source, export, proof = produced
    record_generated_export(source, export, source_proof=proof)
    receipts = list((state_root() / "synthetic_export_receipts").glob("*.json"))
    assert len(receipts) == 1
    assert Path(receipts[0]).stat().st_mode & 0o777 == 0o600
    importlib.reload(owner)
    assert owner.verified_export_source(export) == proof
