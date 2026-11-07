"""The web producer seals the exact auxiliary bytes consumed by downloads."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu import export_publication as publication
from easyicu.webserver import dataio, export_download


@pytest.fixture
def web_export(tmp_path, monkeypatch):
    from easyicu.api.concepts import clear_global_loader
    from easyicu.synthetic_sources import generate_source

    source = tmp_path / "source"
    generate_source(source, "hirid", stays=6)
    root = tmp_path / "export"
    sealed = []
    finish = dataio.finish_native_export

    def seal(folder):
        manifest = json.loads((folder / "_manifest.json").read_text(encoding="utf-8"))
        sealed.append((manifest, (folder / "README.md").read_bytes()
                       if (folder / "README.md").exists() else None))
        finish(folder)

    monkeypatch.setattr(dataio, "finish_native_export", seal)
    try:
        dataio.make_export_runner(
            data_path=str(source), database="hirid", modules=["demographics"],
            concepts={"demographics": ["age"]}, export_format="csv", out_dir=str(root),
            include_feature_definitions=True,
        )(SimpleNamespace(emit=lambda payload: None))
    finally:
        clear_global_loader()
    return root, sealed


@pytest.mark.parametrize("changed_member", [
    None, "README.md", "feature_definitions.json", "feature_definitions.csv", "unrecorded_auxiliary",
])
def test_writer_auxiliary_bytes_are_verified_in_the_download(web_export, tmp_path, changed_member):
    root, sealed = web_export
    registry = {"sources": [{"id": "src_0123456789ab", "path": str(root), "ok": True}]}

    def download():
        return export_download.prepare_registered_export_bundle(
            "src_0123456789ab", registry=registry, temp_dir=tmp_path / "downloads",
        )

    # Require the actual producer output to work before exercising corruption.
    # No test helper fills in missing manifest digests.
    with publication.formal_integrity_scope(propagate_to_children=False):
        bundle = download()
    payload = b"".join(export_download.iter_bundle_and_cleanup(bundle))
    manifest, readme_at_seal = sealed[0]
    assert readme_at_seal == (root / "README.md").read_bytes()
    records = publication.manifest_member_records(manifest)
    definitions = manifest["feature_definitions"]["files"]
    for entry in definitions:
        raw = (root / entry["file"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["bytes"] == len(raw)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        receipt = json.loads(archive.read(export_download.INTEGRITY_RECEIPT_NAME))
        assert receipt["status"] == "verified"
        for name in ("README.md", "feature_definitions.json", "feature_definitions.csv"):
            raw = archive.read(name)
            digest = hashlib.sha256(raw).hexdigest()
            assert raw == (root / name).read_bytes()
            assert records[name] == (digest, len(raw))
            assert receipt["members"][name] == "verified"
            assert receipt["snapshots"][name] == {"sha256": digest, "bytes": len(raw)}
    if changed_member == "unrecorded_auxiliary":
        # Reproduce the earlier web writer's auxiliary declarations, retaining
        # the real data-member digests. There is no download compatibility path.
        manifest.pop("exported_files")
        for entry in manifest["feature_definitions"]["files"]:
            entry.pop("sha256")
            entry.pop("bytes")
        (root / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(export_download.ExportDownloadError) as caught:
            download()
        assert caught.value.__cause__.reason == publication.REASON_DIGEST_UNRECORDED
        assert caught.value.message == publication.INTEGRITY_UNRECORDED_LABEL + ": export/README.md"
    elif changed_member is not None:
        member = root / changed_member
        raw = bytearray(member.read_bytes())
        raw[len(raw) // 2] ^= 1  # Same size: require a digest comparison.
        member.write_bytes(raw)
        with pytest.raises(export_download.ExportDownloadError) as caught:
            download()
        assert caught.value.code == publication.INTEGRITY_FAILED_CODE
        assert caught.value.__cause__.reason == publication.REASON_SHA256_MISMATCH
        assert caught.value.__cause__.member == changed_member


@pytest.mark.parametrize("coordinate", ["inside", "outside", "unknown_root"])
def test_member_refusal_never_exposes_an_absolute_coordinate(tmp_path, coordinate):
    root = tmp_path / "export"
    member = (root / "nested" / "data.csv" if coordinate == "inside"
              else tmp_path / "private" / "data.csv")
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.require_listed_members_present(root, {"files": [{"file": str(member)}]})
    assert caught.value.reason == publication.REASON_MEMBER_PATH_UNSAFE
    refusal = dataio._member_refusal(caught.value, None if coordinate == "unknown_root" else root)
    text = json.dumps(refusal)
    assert str(tmp_path) not in text
    assert str(member) not in text
    assert not Path(refusal["member"]).is_absolute()
    assert not Path(refusal["export_path"]).is_absolute()
    if coordinate == "inside":
        assert refusal["member"] == "nested/data.csv"
        assert refusal["export_path"] == "export/nested/data.csv"
    else:
        assert refusal["member"] == "member-" + hashlib.sha256(str(member).encode()).hexdigest()[:12]
