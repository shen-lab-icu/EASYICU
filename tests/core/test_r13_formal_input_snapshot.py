"""Formal runs prove their inputs did not change: hash before and after (delta9b ruling).

``easyicu.formal_inputs`` snapshots every input of a database (the conversion
manifest's outputs and existing inputs, the chosen bucket installations, a
declared view's source manifest and outputs) and hashes the same set after the
run. Any difference stops the run. Rows are synthetic.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from easyicu import formal_inputs
from tests.support.bucket_storage import TABLE_CONFIGS, b6_source, install_layout


def _manifest(root: Path) -> None:
    """A conversion manifest listing every flat parquet output under ``root``."""
    tables: dict = {}
    for path in sorted(root.rglob("*.parquet")):
        relative = path.relative_to(root)
        if relative.parts[0].endswith("_bucket"):
            continue
        table = relative.parts[0] if len(relative.parts) > 1 else path.stem
        tables.setdefault(table, []).append(
            {"relative_path": relative.as_posix(), "size_bytes": path.stat().st_size}
        )
    (root / "conversion_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "tables": [{"table": t, "status": "completed", "outputs": o} for t, o in tables.items()],
    }))


def _installation(tmp_path: Path) -> Path:
    source = b6_source(tmp_path / "hirid", "hirid")
    _manifest(source.base_path)
    install_layout(source, "hash")
    return source.base_path


def _rewrite_same_stat(path: Path) -> None:
    stat = path.stat()
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    path.write_bytes(bytes(payload))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_the_snapshot_covers_manifest_outputs_and_the_chosen_bucket(tmp_path):
    root = _installation(tmp_path)
    table = TABLE_CONFIGS["hirid"][0]

    snapshot = formal_inputs.take_snapshot([f"hirid:{root}:144-145"])

    entry = snapshot["databases"]["hirid"]
    roles = {role for f in entry["files"] for role in f["roles"]}
    assert {"conversion_manifest", "conversion_output", "bucket_file"} <= roles
    assert [b["table"] for b in entry["buckets"]] == [table]
    bucket = entry["buckets"][0]
    assert all(s["sha256_matches_snapshot"] for s in bucket["derived_from"])
    for item in entry["files"]:
        assert item["sha256"] == hashlib.sha256(Path(item["realpath"]).read_bytes()).hexdigest()
    assert entry["bytes"] == sum(f["size"] for f in entry["files"])
    assert entry["hash_seconds"] >= 0
    assert entry["digest"] == formal_inputs.snapshot_digest(entry["files"])
    # The same inputs hash to the same set afterwards.
    post = formal_inputs.verify_snapshot(snapshot)
    assert post["databases"]["hirid"]["digest"] == entry["digest"]


def test_an_input_modified_during_the_run_stops_it(tmp_path):
    root = _installation(tmp_path)
    snapshot = formal_inputs.take_snapshot([f"hirid:{root}"])
    other = next(
        Path(f["realpath"]) for f in snapshot["databases"]["hirid"]["files"]
        if f["roles"] == ["conversion_output"]
        and "observations" not in f["realpath"]
    )
    _rewrite_same_stat(other)

    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.verify_snapshot(snapshot)
    assert error.value.code == "input_changed_during_run"
    assert error.value.details["changed"] == [str(other.resolve())]


def test_a_same_size_rewrite_with_mtime_restored_while_hashed_is_refused(tmp_path, monkeypatch):
    # delta25d stat identity: utime restores mtime but ctime still moves.
    import time

    root = _installation(tmp_path)
    target = next(root.rglob("*.parquet"))
    real = formal_inputs.hashlib.file_digest
    done = []

    def rewrite_while_hashing(handle, name):
        digest = real(handle, name)
        if not done and os.path.realpath(handle.name) == str(target.resolve()):
            done.append(True)
            time.sleep(0.02)
            _rewrite_same_stat(target)
        return digest

    monkeypatch.setattr(formal_inputs.hashlib, "file_digest", rewrite_while_hashing)
    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.take_snapshot([f"hirid:{root}"])
    assert done and error.value.code == "input_changed_while_hashed"


def test_an_input_added_during_the_run_stops_it(tmp_path):
    root = _installation(tmp_path)
    snapshot = formal_inputs.take_snapshot([f"hirid:{root}"])
    extra = root / "extra" / "1.parquet"
    extra.parent.mkdir()
    extra.write_bytes((root / "observations" / "1.parquet").read_bytes())
    _manifest_entries = json.loads((root / "conversion_manifest.json").read_text())
    _manifest_entries["tables"].append(
        {"table": "extra", "outputs": [{"relative_path": "extra/1.parquet"}]}
    )
    (root / "conversion_manifest.json").write_text(json.dumps(_manifest_entries))

    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.verify_snapshot(snapshot)
    assert error.value.code == "input_changed_during_run"


def test_an_input_modified_and_restored_passes(tmp_path):
    # Accepted, documented limitation: equality is by content, not history.
    root = _installation(tmp_path)
    snapshot = formal_inputs.take_snapshot([f"hirid:{root}"])
    target = Path(snapshot["databases"]["hirid"]["files"][-1]["realpath"])
    original = target.read_bytes()
    target.write_bytes(original + b"x")
    target.write_bytes(original)

    post = formal_inputs.verify_snapshot(snapshot)

    assert post["databases"]["hirid"]["digest"] == snapshot["databases"]["hirid"]["digest"]


def test_a_bucket_whose_source_no_longer_has_the_builders_content_is_refused(tmp_path):
    root = _installation(tmp_path)
    table = TABLE_CONFIGS["hirid"][0]
    manifest = json.loads((root / f"{table}_bucket" / "_BUCKET_MANIFEST.json").read_text())
    _rewrite_same_stat(root / manifest["derived_from"]["files"][0]["path"])

    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.take_snapshot([f"hirid:{root}"])
    assert error.value.code == "bucket_derived_from_sha_mismatch"


def test_an_input_without_a_conversion_manifest_is_refused(tmp_path):
    source = b6_source(tmp_path / "hirid", "hirid")

    with pytest.raises(formal_inputs.InputSnapshotError) as error:
        formal_inputs.take_snapshot([f"hirid:{source.base_path}"])
    assert error.value.code == "input_manifest_missing"


def test_the_cli_writes_snapshots_outside_the_input_only(tmp_path):
    root = _installation(tmp_path)

    assert formal_inputs.main(["snapshot", "--out", str(root / "PRE.json"), f"hirid:{root}"]) == 3
    assert not (root / "PRE.json").exists()

    pre, post = tmp_path / "INPUT_SNAPSHOT.json", tmp_path / "INPUT_SNAPSHOT_POST.json"
    assert formal_inputs.main(["snapshot", "--out", str(pre), f"hirid:{root}:1-2"]) == 0
    assert formal_inputs.main(["verify", "--pre", str(pre), "--out", str(post)]) == 0
    assert (json.loads(post.read_text())["databases"]["hirid"]["digest"]
            == json.loads(pre.read_text())["databases"]["hirid"]["digest"])


def _view_farm(tmp_path: Path) -> tuple[Path, Path]:
    """A source installation and a symlink farm whose manifest declares it."""
    source = b6_source(tmp_path / "source", "hirid").base_path
    _manifest(source)
    farm = tmp_path / "farm"
    farm.mkdir()
    for item in source.iterdir():
        if item.name != "conversion_manifest.json":
            (farm / item.name).symlink_to(item, target_is_directory=item.is_dir())
    manifest = json.loads((source / "conversion_manifest.json").read_text())
    raw = (source / "conversion_manifest.json").read_bytes()
    manifest["derived_from"] = {
        "path": str(source / "conversion_manifest.json"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (farm / "conversion_manifest.json").write_text(json.dumps(manifest))
    return source, farm


def test_a_declared_view_snapshot_carries_its_chain_and_the_source_outputs(tmp_path):
    source, farm = _view_farm(tmp_path)

    entry = formal_inputs.take_snapshot([f"hirid:{farm}"])["databases"]["hirid"]

    chain = entry["declared_view_chain"]
    assert chain["source_manifest_path"] == str(source / "conversion_manifest.json")
    assert chain["source_manifest_sha256"] == hashlib.sha256(
        (source / "conversion_manifest.json").read_bytes()).hexdigest()
    roles = {role for f in entry["files"] for role in f["roles"]}
    assert {"view_source_manifest", "view_source_output"} <= roles
    # Farm outputs and source outputs are the same real files, hashed once.
    reals = [f["realpath"] for f in entry["files"]]
    assert len(reals) == len(set(reals))


@pytest.mark.parametrize("change", [False, True])
def test_the_extract_cli_opt_in_proves_the_inputs_did_not_change(tmp_path, monkeypatch, change):
    from easyicu.scripts import extract_features

    root = _installation(tmp_path)
    target = root / "observations" / "1.parquet"

    def run(resolver, datasource, groups):
        if change:
            _rewrite_same_stat(target)
        return {}

    monkeypatch.setattr(extract_features, "extract_groups", run)
    out = tmp_path / "exports"

    code = extract_features.main([
        "extract", "--sources", "hirid", "--data-dirs", str(root),
        "--output", str(out), "--input-snapshot",
    ])

    assert (out / "INPUT_SNAPSHOT.json").is_file()
    assert code == (3 if change else 0)
    assert (out / "INPUT_SNAPSHOT_POST.json").is_file() is (not change)


def test_a_built_layout_records_the_code_and_engine_that_wrote_it(tmp_path):
    import duckdb

    from easyicu import code_identity

    root = _installation(tmp_path)
    table = TABLE_CONFIGS["hirid"][0]
    manifest = json.loads((root / f"{table}_bucket" / "_BUCKET_MANIFEST.json").read_text())

    package = Path(formal_inputs.__file__).resolve().parent
    identity = code_identity.resolve_code_identity(package)
    assert manifest["builder_code_identity"]["source"] == identity["source"]
    assert manifest["builder_code_identity"]["commit"] == identity["commit"]
    assert manifest["builder_code_identity"]["version"] == identity["version"]
    assert manifest["duckdb_version"] == duckdb.__version__
    assert manifest["builder_version"].startswith("easyicu.bucket_builder/")
    # The layout with its extra metadata still opens.
    assert formal_inputs.take_snapshot([f"hirid:{root}"])["databases"]["hirid"]["buckets"]


def test_the_cli_opt_in_says_it_is_not_the_formal_guarantee():
    # Supervisor ruling (D10D-CONSUMED-DELIVERY NB): no reader may take the CLI
    # snapshot for the formal guarantee.
    from easyicu.scripts import extract_features

    parser = extract_features.build_parser()
    extract = next(a for a in parser._subparsers._group_actions[0].choices.values()
                   if any("--input-snapshot" in o.option_strings for o in a._actions))
    (option,) = [o for o in extract._actions if "--input-snapshot" in o.option_strings]
    for text in (option.help, formal_inputs.__doc__):
        flat = " ".join(text.split())
        assert "listed input files are unchanged before and after the run" in flat
        assert "does NOT publish a consumed set or a RUN_RECEIPT" in flat
        assert "does NOT prove that every file the run read was in the snapshot" in flat
        assert "formal guarantee comes only from bench7v2 / " in flat


@pytest.mark.parametrize("scheme", ["variable", "hash"])
def test_every_layout_scheme_records_its_builder_in_manifest_and_receipt(tmp_path, scheme):
    import duckdb

    from easyicu import code_identity
    from tests.support.bucket_storage import install_layout

    source = b6_source(tmp_path / "eicu", "eicu")
    install_layout(source, scheme)
    bucket = source.base_path / f"{TABLE_CONFIGS['eicu'][0]}_bucket"
    manifest = json.loads((bucket / "_BUCKET_MANIFEST.json").read_text())
    build = json.loads((bucket / "_BUCKET_BUILD_RECEIPT.json").read_text())

    identity = code_identity.resolve_code_identity(Path(formal_inputs.__file__).resolve().parent)
    expected = {k: identity[k] for k in ("source", "commit", "dirty", "version")}
    for record in (manifest, build):
        assert {k: record["builder_code_identity"][k] for k in expected} == expected
        assert record["duckdb_version"] == duckdb.__version__
