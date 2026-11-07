"""A damaged export member is refused by every reader, never read or emptied.

B-SHA-UNVERIFIED: the manifest records each member's size and SHA-256, but no
reader compared them, so a replaced file was consumed as published data.
B-TRUNCATED-SILENT: webserver readers turned an unreadable member into an
empty column list, a zero count or a dropped module.

Each case starts from a real published synthetic export (``tests.support.
published_export``) and damages its outcome member three ways: truncated,
one byte flipped (still readable) and rewritten with one value +1 (readable,
same schema and row count, different data).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from easyicu import export_publication as publication
from tests.support.published_export import (
    DAMAGES,
    damaged_copy,
    make_pre_integrity_web_export,
    member_file,
    publish_synthetic_export,
    strip_digests,
)

REPO = Path(__file__).resolve().parents[2]
INTEGRITY = "export_member_integrity_failed"
UNREADABLE = "export_member_unreadable"


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        return publish_synthetic_export(tmp_path_factory.mktemp("published"), patch)


@pytest.fixture(params=DAMAGES)
def damaged(request, published, tmp_path):
    return damaged_copy(published, tmp_path / request.param / "hirid", request.param)


@pytest.fixture
def truncated_undigested(published, tmp_path):
    """A pre-integrity web export whose outcome member is truncated."""
    export = damaged_copy(published, tmp_path / "legacy" / "hirid", "truncated")
    make_pre_integrity_web_export(export)
    return export


@pytest.fixture
def unrecorded(published, tmp_path):
    """A readable pre-integrity web export (its writer recorded no digests)."""
    import shutil

    export = tmp_path / "unrecorded" / "hirid"
    shutil.copytree(published, export)
    make_pre_integrity_web_export(export)
    return export


def _rules():
    spec = importlib.util.spec_from_file_location(
        "r13exportsha_plausibility_cli", REPO / "tools/report_export_plausibility.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SCREENING_RULES


# --- the owner check ---------------------------------------------------------

def test_the_owner_check_names_the_cause(published, damaged):
    member = member_file(published)
    assert publication.verify_member_integrity(published, member) == publication.INTEGRITY_VERIFIED
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(damaged, member)
    assert caught.value.code == INTEGRITY
    assert caught.value.reason in {publication.REASON_SIZE_MISMATCH,
                                   publication.REASON_SHA256_MISMATCH}
    assert caught.value.member == member


def test_an_unlisted_member_or_missing_digest_is_refused(published, tmp_path):
    import shutil

    export = tmp_path / "copy"
    shutil.copytree(published, export)
    (export / "stray.parquet").write_bytes((export / member_file(published)).read_bytes())
    with pytest.raises(publication.ExportMemberIntegrityError) as unlisted:
        publication.verify_member_integrity(export, "stray.parquet")
    assert unlisted.value.reason == publication.REASON_MEMBER_UNRECORDED
    manifest = json.loads((export / "_manifest.json").read_text(encoding="utf-8"))
    manifest["files"][0].pop("parquet_sha256")
    (export / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(publication.ExportMemberIntegrityError) as undigested:
        publication.verify_member_integrity(export, manifest["files"][0]["file"])
    assert undigested.value.reason == publication.REASON_DIGEST_UNRECORDED


def _flip_in_place_keeping_mtime(member: Path) -> None:
    import os

    before = member.stat()
    data = bytearray(member.read_bytes())
    data[len(data) // 3] ^= 0x01
    member.write_bytes(bytes(data))
    os.utime(member, ns=(before.st_atime_ns, before.st_mtime_ns))  # forge mtime only


def test_a_freshly_changed_member_is_never_memoised(published, tmp_path):
    """Racily clean: a same-tick rewrite can keep ctime on a coarse clock."""
    import shutil

    export = tmp_path / "racy"
    shutil.copytree(published, export)
    member = export / member_file(published)
    assert publication.verify_member_integrity(export, member) == publication.INTEGRITY_VERIFIED
    _flip_in_place_keeping_mtime(member)
    with pytest.raises(publication.ExportMemberIntegrityError):
        publication.verify_member_integrity(export, member)


def test_the_memo_reuses_a_digest_and_rekeys_on_ctime(published, tmp_path, monkeypatch):
    import shutil
    import time

    export = tmp_path / "memo"
    shutil.copytree(published, export)
    member = export / member_file(published)
    monkeypatch.setattr(publication, "_RACY_WINDOW_NS", -(10**18))  # memoise at once
    hashed = []
    real_hash = publication._hash_file
    monkeypatch.setattr(publication, "_hash_file",
                        lambda path: hashed.append(path) or real_hash(path))
    publication.clear_integrity_memo()
    for _ in range(3):
        publication.verify_member_integrity(export, member)
    assert len(hashed) == 1  # hashed once per stat identity, not per read
    time.sleep(0.05)  # past a coarse ctime tick
    _flip_in_place_keeping_mtime(member)
    with pytest.raises(publication.ExportMemberIntegrityError):
        publication.verify_member_integrity(export, member)
    publication.clear_integrity_memo()


def test_a_pre_integrity_manifest_is_unrecorded_but_still_refuses_unreadable(
        truncated_undigested):
    member = member_file(truncated_undigested)
    assert (publication.verify_member_integrity(truncated_undigested, member)
            == publication.INTEGRITY_UNRECORDED == "unrecorded")


# --- intake (research agent) -------------------------------------------------

@pytest.mark.parametrize("reader", ["inspect_export_layout", "open_export_package",
                                    "index_export_package"])
def test_intake_refuses_a_damaged_member(published, damaged, reader):
    from easyicu.research_agent.intake import export_package as intake

    clean = getattr(intake, reader)(published)
    if hasattr(clean, "close"):
        clean.close()
    with pytest.raises(intake.ExportPackageError) as caught:
        result = getattr(intake, reader)(damaged)
        if hasattr(result, "close"):
            result.close()
    assert caught.value.code == INTEGRITY


# --- aggregate tools ---------------------------------------------------------

def test_plausibility_refuses_a_damaged_member(published, damaged):
    from easyicu.io import export_plausibility

    rules = _rules()
    assert export_plausibility.build_report(published, rules=rules, threads=1,
                                            memory_limit="256MB")
    with pytest.raises(publication.ExportMemberIntegrityError):
        export_plausibility.build_report(damaged, rules=rules, threads=1,
                                         memory_limit="256MB")


def test_the_download_never_zips_a_damaged_member(published, damaged, tmp_path):
    from easyicu.webserver import export_download

    def registry(root):
        return {"sources": [{"ok": True, "id": "src_0123456789ab", "path": str(root)}]}

    bundle = export_download.prepare_registered_export_bundle(
        "src_0123456789ab", registry=registry(published), temp_dir=tmp_path / "zips")
    bundle.path.unlink()
    with pytest.raises(export_download.ExportDownloadError) as caught:
        export_download.prepare_registered_export_bundle(
            "src_0123456789ab", registry=registry(damaged), temp_dir=tmp_path / "zips")
    assert (caught.value.code, caught.value.status_code) == (INTEGRITY, 409)
    assert not list((tmp_path / "zips").glob("*.zip"))


# --- webserver readers -------------------------------------------------------

@pytest.mark.parametrize("reader", ["scan_path", "describe_export_source",
                                    "summarize_export_workspace"])
def test_webserver_folder_readers_refuse_a_damaged_member(published, damaged, reader):
    from easyicu.webserver import dataio

    assert getattr(dataio, reader)(str(published))["ok"] is True
    answer = getattr(dataio, reader)(str(damaged))
    assert answer["ok"] is False
    assert answer["error"] == INTEGRITY


def test_the_projection_reader_refuses_a_damaged_member(published, damaged):
    from easyicu.webserver import dataio

    member = member_file(published)
    assert len(dataio.read_export_projection(published / member, columns=["stay_id"])) == 6
    with pytest.raises(publication.ExportMemberIntegrityError):
        dataio.read_export_projection(damaged / member, columns=["stay_id"])


def _agent_source(export: Path) -> dict:
    manifest = json.loads((export / "_manifest.json").read_text(encoding="utf-8"))
    return {"files": [{"module": e["module"], "file": e["file"]} for e in manifest["files"]]}


def test_agent_outputs_refuse_a_damaged_member(published, damaged):
    from easyicu.webserver import agent_outputs

    clean = agent_outputs._load_context(str(published), _agent_source(published))
    assert set(clean["frames"]) == {"demographics", "outcome"}
    with pytest.raises(publication.ExportMemberIntegrityError):
        agent_outputs._load_context(str(damaged), _agent_source(published))


def _concept_item(export: Path, concept: str = "death") -> dict:
    manifest = json.loads((export / "_manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["files"] if e["module"] == "outcome")
    return {"file": entry["file"], "module": "outcome", "rows": entry["rows"],
            "columns": list(entry["physical_schema"])}


def test_idea_mining_labels_a_damaged_member_instead_of_dropping_it(published, damaged):
    from easyicu.webserver.ideas import mining

    index = {"concept_to_file": {"death": _concept_item(published)}}
    clean = mining._feature_stats(published, ["death"], index, {"1"})
    assert len(clean) == 1 and clean[0].get("status") != "sample_unavailable"
    stats = mining._feature_stats(damaged, ["death"], index, {"1"})
    assert len(stats) == 1
    assert stats[0]["status"] == "sample_unavailable"
    assert stats[0]["error_code"] == INTEGRITY


def test_drilldown_coverage_marks_a_damaged_member_unreadable(published, damaged):
    from easyicu.webserver.patient_drilldown import coverage

    description = {"files": [_concept_item(published)]}
    coverage.clear_cache()
    clean = coverage.build_feature_coverage(published, description)
    outcome = next(m for m in clean["modules"] if m["module"] == "outcome")
    assert outcome["member_error"] is None
    coverage.clear_cache()
    result = coverage.build_feature_coverage(damaged, description)
    outcome = next(m for m in result["modules"] if m["module"] == "outcome")
    assert outcome["member_error"]["code"] == INTEGRITY
    materialized = [f for f in outcome["features"] if f["materialized"]]
    assert materialized and all(f["status"] == "member_unreadable" for f in materialized)
    assert not any(f["loadable"] for f in materialized)
    coverage.clear_cache()


# --- no silent empties (a listing-only manifest cannot catch the damage) ------

def test_unreadable_member_helpers_raise_instead_of_returning_empty(truncated_undigested):
    from easyicu.webserver import dataio

    path = truncated_undigested / member_file(truncated_undigested)
    for call in (
        lambda: dataio._read_columns(path),
        lambda: dataio._count_rows(path),
        lambda: dataio._read_stay_ids(path),
        lambda: dataio._count_matching_stay_rows(path, {"1"}),
        lambda: dataio.read_export_projection(path, columns=["stay_id"]),
    ):
        with pytest.raises(publication.ExportMemberUnreadableError) as caught:
            call()
        assert caught.value.code == UNREADABLE
        assert caught.value.__cause__ is not None


def test_webserver_summaries_refuse_an_unreadable_member(truncated_undigested):
    from easyicu.webserver import dataio

    for reader in (dataio.describe_export_source, dataio.summarize_export_workspace):
        answer = reader(str(truncated_undigested))
        assert (answer["ok"], answer["error"]) == (False, UNREADABLE), reader.__name__


def test_agent_outputs_never_drop_an_unreadable_or_missing_module(truncated_undigested, published):
    from easyicu.webserver import agent_outputs

    with pytest.raises(publication.ExportMemberUnreadableError):
        agent_outputs._load_context(str(truncated_undigested), _agent_source(published))
    (truncated_undigested / member_file(published)).unlink()
    with pytest.raises((publication.ExportMemberUnreadableError,
                        publication.ExportMemberIntegrityError)) as missing:
        agent_outputs._load_context(str(truncated_undigested), _agent_source(published))
    assert missing.value.__cause__ is not None or getattr(
        missing.value, "reason", "") == publication.REASON_MEMBER_MISSING


def test_mining_and_coverage_label_an_unreadable_member(truncated_undigested, published):
    from easyicu.webserver.ideas import mining
    from easyicu.webserver.patient_drilldown import coverage

    index = {"concept_to_file": {"death": _concept_item(published)}}
    stats = mining._feature_stats(truncated_undigested, ["death"], index, {"1"})
    assert [(s["status"], s["error_code"]) for s in stats] == [("sample_unavailable", UNREADABLE)]
    coverage.clear_cache()
    result = coverage.build_feature_coverage(
        truncated_undigested, {"files": [_concept_item(published)]})
    outcome = next(m for m in result["modules"] if m["module"] == "outcome")
    assert outcome["member_error"]["code"] == UNREADABLE
    coverage.clear_cache()


# --- delta27e: the unrecorded policy -------------------------------------------
#
# 1 visible, never silent; 2 formal paths refuse; 3 anti-downgrade per manifest;
# 4 formal tool consumers (see also tests/governance/test_export_reader_gates.py).

UNRECORDED_LABEL = "integrity unrecorded · re-export to verify"


def test_unrecorded_members_are_named_by_the_api_and_the_source_row(unrecorded):
    from easyicu.webserver import dataio, sources

    for reader in (dataio.describe_export_source, dataio.summarize_export_workspace):
        answer = reader(str(unrecorded))
        assert answer["ok"] is True, reader.__name__
        assert answer["member_integrity"] == {
            "status": "unrecorded", "label": UNRECORDED_LABEL,
            "unrecorded_members": len(answer["files"]),
        }, reader.__name__
        assert {row["integrity"] for row in answer["files"]} == {"unrecorded"}
    row = sources._source_from_path(str(unrecorded))
    assert row["member_integrity"]["label"] == UNRECORDED_LABEL
    script = (REPO / "src/easyicu/webserver/static/js/screens-viz.js").read_text(encoding="utf-8")
    assert "s.member_integrity.status === 'unrecorded'" in script
    assert UNRECORDED_LABEL in script


def test_a_verified_export_shows_verified(published):
    from easyicu.webserver import dataio

    answer = dataio.describe_export_source(str(published))
    assert answer["member_integrity"]["status"] == "verified"
    assert answer["member_integrity"]["label"] is None


def test_intake_packages_and_receipts_record_unrecorded_per_member(unrecorded):
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.intake import export_package as intake

    layout = intake.inspect_export_layout(unrecorded)
    assert layout["member_integrity"]["status"] == "unrecorded"
    with intake.open_export_package(unrecorded) as package:
        assert set(package.member_integrity.values()) == {"unrecorded"}
        receipt = materializer._export_authority_provenance(package)
    assert {row["integrity"] for row in receipt["files"]} == {"unrecorded"}


def test_agent_outputs_record_unrecorded_per_member(unrecorded):
    from easyicu.webserver import agent_outputs

    context = agent_outputs._load_context(str(unrecorded), _agent_source(unrecorded))
    assert set(context["member_integrity"].values()) == {"unrecorded"}
    assert len(context["member_integrity"]) == 2


def test_formal_scope_refuses_an_unrecorded_manifest(unrecorded):
    from easyicu.research_agent.intake import export_package as intake

    member = member_file(unrecorded)
    assert publication.verify_member_integrity(unrecorded, member) == "unrecorded"
    with pytest.raises(publication.ExportMemberIntegrityError) as explicit:
        publication.verify_member_integrity(unrecorded, member, formal=True)
    assert explicit.value.reason == publication.REASON_INTEGRITY_UNRECORDED
    with publication.formal_integrity_scope():
        with pytest.raises(intake.ExportPackageError) as scoped:
            intake.open_export_package(unrecorded)
        assert scoped.value.code == INTEGRITY
        import os
        assert os.environ.get("EASYICU_FORMAL_EXPORT_INTEGRITY") == "1"  # children inherit
    assert publication.formal_integrity_active() is False
    with intake.open_export_package(unrecorded) as package:  # UI path still accepts
        assert set(package.member_integrity.values()) == {"unrecorded"}


def test_publication_qc_and_benchmark_inputs_refuse_an_unrecorded_manifest(unrecorded, tmp_path):
    import shutil

    root = tmp_path / "full6"
    qc_a02 = _load_tool("r13exportsha_qc_a02",
                        "scripts/figures/QC-A02_easyicu_cross_database_reliability_audit.py")
    for database in qc_a02.DATABASES:
        shutil.copytree(unrecorded, root / database)
    with pytest.raises(publication.ExportMemberIntegrityError) as qc:
        qc_a02._require_published_exports(root)
    assert qc.value.reason == publication.REASON_INTEGRITY_UNRECORDED
    qc_a01 = _load_tool("r13exportsha_qc_a01",
                        "scripts/figures/QC-A01_cross_database_distributions.py")
    with pytest.raises(publication.ExportMemberIntegrityError):
        qc_a01.module_names(root)


def test_the_e3_benchmark_input_refuses_an_unrecorded_member(unrecorded, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO))
    tool = _load_tool("r13exportsha_e3", "tools/materialize_e3_miiv_kdigo_gradient.py")
    with pytest.raises(publication.ExportMemberIntegrityError):
        tool._one_row_per_stay(unrecorded / member_file(unrecorded), columns=["stay_id"],
                               label="full6 outcome")


@pytest.mark.parametrize("declared_web", [False, True])
def test_deleting_digests_never_downgrades_a_digest_writing_manifest(
        published, tmp_path, declared_web):
    """Native v2, or a web export declaring member_integrity, must keep digests."""
    import shutil

    from easyicu.research_agent.intake import export_package as intake
    from easyicu.webserver import dataio

    export = tmp_path / "stripped" / "hirid"
    shutil.copytree(published, export)
    strip_digests(export)
    if declared_web:
        manifest = json.loads((export / "_manifest.json").read_text(encoding="utf-8"))
        manifest["export_kind"] = "web_module_extraction"
        manifest["member_integrity"] = {"algorithm": "sha256", "scope": "every_member"}
        (export / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    member = member_file(published)
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(export, member)
    assert caught.value.reason == publication.REASON_DIGEST_UNRECORDED
    with pytest.raises(intake.ExportPackageError) as package:
        intake.open_export_package(export)
    assert package.value.code == INTEGRITY
    answer = dataio.describe_export_source(str(export))
    assert (answer["ok"], answer["reason"]) == (False, publication.REASON_DIGEST_UNRECORDED)


def test_writers_declare_member_integrity(published, web_written):
    for export in (published, web_written):
        manifest = json.loads((export / "_manifest.json").read_text(encoding="utf-8"))
        assert manifest["member_integrity"] == {"algorithm": "sha256", "scope": "every_member"}
        assert publication.manifest_requires_digests(manifest)
        assert all(publication.recorded_member_digest(e) for e in manifest["files"])


@pytest.fixture
def web_written(tmp_path):
    from easyicu.webserver import dataio
    from easyicu.webserver.demo_source_contracts import SOURCES
    from tests.support.windows_laptop_demo import isolated_demo_paths, synthetic_demo_download

    class _Job:
        cancel_requested = False

        def emit(self, _payload):
            pass

    source = next(item for item in SOURCES if item.database == "eicu_demo")
    paths = isolated_demo_paths(tmp_path, source)
    synthetic_demo_download(source, paths, _Job())
    assert dataio.make_convert_runner(str(paths.raw), "eicu")(_Job())["failed"] == 0
    dataio.make_export_runner(
        str(paths.raw), "eicu", modules=["demographics"], out_dir=str(paths.export),
        export_format="parquet", include_feature_definitions=False,
    )(_Job())
    return paths.export


def _load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- delta27e: missing members, hash race, relative paths, legacy inventories --

def _copy(published: Path, destination: Path) -> Path:
    import shutil

    shutil.copytree(published, destination)
    return destination


def _registry(root: Path) -> dict:
    return {"sources": [{"ok": True, "id": "src_0123456789ab", "path": str(root)}]}


def test_a_deleted_listed_member_is_refused_by_every_reader(published, tmp_path):
    """D27E-MISSING: a digest-bearing export is checked member by member."""
    from easyicu.research_agent.intake import export_package as intake
    from easyicu.webserver import dataio, export_download

    export = _copy(published, tmp_path / "missing" / "hirid")
    manifest = json.loads((export / "_manifest.json").read_text(encoding="utf-8"))
    deleted = next(e["file"] for e in manifest["files"] if e["module"] == "demographics")
    (export / deleted).unlink()
    answer = dataio.describe_export_source(str(export))
    assert (answer["ok"], answer["reason"]) == (False, publication.REASON_MEMBER_MISSING)
    assert str(tmp_path) not in json.dumps(answer)  # no host path in the refusal
    with pytest.raises(intake.ExportPackageError) as package:
        intake.open_export_package(export)
    assert package.value.code == INTEGRITY
    with pytest.raises(export_download.ExportDownloadError) as download:
        export_download.prepare_registered_export_bundle(
            "src_0123456789ab", registry=_registry(export), temp_dir=tmp_path / "zips")
    # The download's own coordinate check already refused a missing file.
    assert download.value.code in {INTEGRITY, "registered_export_manifest_file_missing"}
    with pytest.raises(publication.ExportMemberIntegrityError) as package_gate:
        publication.require_verified_export_package(export)
    assert package_gate.value.reason == publication.REASON_MEMBER_MISSING


def test_a_member_changed_during_its_hash_is_refused_and_not_memoised(
        published, tmp_path, monkeypatch):
    """D27E-HASH-RACE: the same fd is fstat-ed before and after the hash."""
    import os
    import time

    export = _copy(published, tmp_path / "race" / "hirid")
    member = export / member_file(published)
    monkeypatch.setattr(publication, "_RACY_WINDOW_NS", -(10**18))  # memoise at once
    real_chunks = publication._digest_chunks

    def racing(handle):
        chunks = list(real_chunks(handle))  # the original bytes are hashed
        time.sleep(0.05)  # past a coarse ctime tick
        before = member.stat()
        with member.open("r+b") as writer:  # same size, in place
            writer.seek(len(chunks[0]) // 3)
            byte = writer.read(1)
            writer.seek(-1, os.SEEK_CUR)
            writer.write(bytes([byte[0] ^ 0x01]))
        os.utime(member, ns=(before.st_atime_ns, before.st_mtime_ns))
        yield from chunks

    monkeypatch.setattr(publication, "_digest_chunks", racing)
    publication.clear_integrity_memo()
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(export, member)
    assert caught.value.reason == publication.REASON_CHANGED_DURING_HASH
    assert not any(key[0] == str(member.resolve()) for key in publication._DIGEST_MEMO)
    monkeypatch.setattr(publication, "_digest_chunks", real_chunks)
    with pytest.raises(publication.ExportMemberIntegrityError) as later:
        publication.verify_member_integrity(export, member)  # the changed bytes
    assert later.value.reason == publication.REASON_SHA256_MISMATCH
    publication.clear_integrity_memo()


def test_the_projection_reader_gives_the_same_answer_for_relative_paths(
        published, monkeypatch):
    """D27E-RELATIVE: a caller-relative path is never joined to its root twice."""
    from easyicu.webserver import dataio

    member = member_file(published)
    absolute = dataio.read_export_projection(published / member, columns=["stay_id"])
    monkeypatch.chdir(published.parent)
    from_parent = dataio.read_export_projection(
        Path(published.name) / member, columns=["stay_id"])
    monkeypatch.chdir(published)
    from_root = dataio.read_export_projection(Path(member), columns=["stay_id"])
    assert absolute.equals(from_parent) and absolute.equals(from_root)


def _legacy_inventories(export: Path, *, conflict: bool) -> list[str]:
    """Rewrite as a legacy manifest: names in ``files``, digests in ``modules``."""
    path = export / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.pop("files")
    for native_only in ("schema_version", "member_integrity", "column_metadata",
                        "export_kind", "contract_revision"):
        manifest.pop(native_only, None)
    manifest["files"] = (
        [dict(e) for e in entries] if conflict else [e["file"] for e in entries])
    manifest["modules"] = [
        {"module": e["module"], "file": e["file"],
         "sha256": ("0" * 64) if conflict else e["parquet_sha256"],
         "bytes": e["parquet_bytes"]}
        for e in entries
    ]
    path.unlink()  # an explicitly legacy package: the legacy manifest name
    (export / publication.LEGACY_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return [e["file"] for e in entries]


def test_legacy_inventories_merge_compatible_digests(published, tmp_path):
    """D27E-LEGACY: one normaliser; a later inventory's digest is not ignored."""
    from easyicu.research_agent.intake import export_package as intake

    export = _copy(published, tmp_path / "legacy_merge" / "hirid")
    members = _legacy_inventories(export, conflict=False)
    manifest = json.loads(
        (export / publication.LEGACY_MANIFEST).read_text(encoding="utf-8"))
    assert publication.manifest_integrity_contract(manifest) == "legacy"
    assert all(publication.manifest_member_records(manifest)[m] for m in members)
    for member in members:
        assert publication.verify_member_integrity(export, member) == "verified"
    with intake.open_export_package(export) as package:
        assert set(package.member_integrity.values()) == {"verified"}


def test_legacy_inventories_with_conflicting_digests_are_refused(published, tmp_path):
    from easyicu.research_agent.intake import export_package as intake
    from easyicu.webserver import dataio

    export = _copy(published, tmp_path / "legacy_conflict" / "hirid")
    member = _legacy_inventories(export, conflict=True)[0]
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.verify_member_integrity(export, member)
    assert caught.value.reason == publication.REASON_DIGEST_CONFLICT
    with pytest.raises(intake.ExportPackageError) as package:
        intake.open_export_package(export)
    assert package.value.code == INTEGRITY
    answer = dataio.describe_export_source(str(export))
    assert (answer["ok"], answer["reason"]) == (False, publication.REASON_DIGEST_CONFLICT)


def test_an_unrecorded_truncated_member_never_reaches_a_zip(truncated_undigested, tmp_path):
    """D27E-LEGACY-ZIP: PAR1 magic alone is not readability."""
    from easyicu.webserver import export_download

    with pytest.raises(export_download.ExportDownloadError) as caught:
        export_download.prepare_registered_export_bundle(
            "src_0123456789ab", registry=_registry(truncated_undigested),
            temp_dir=tmp_path / "zips")
    assert (caught.value.code, caught.value.status_code) == (UNREADABLE, 409)
    assert not list((tmp_path / "zips").glob("*.zip"))


def test_an_unrecorded_zip_states_it_and_formal_scope_refuses_it(unrecorded, tmp_path):
    import zipfile

    from easyicu.webserver import export_download

    bundle = export_download.prepare_registered_export_bundle(
        "src_0123456789ab", registry=_registry(unrecorded), temp_dir=tmp_path / "zips")
    try:
        # Data members remain unrecorded. The receipt now also covers sidecars,
        # whose independently recorded digests can still verify.
        data_members = {entry["file"] for entry in _manifest(unrecorded)["files"]}
        assert {name: bundle.member_integrity[name] for name in data_members} == {
            name: "unrecorded" for name in data_members}
        with zipfile.ZipFile(bundle.path) as archive:
            receipt = json.loads(archive.read(export_download.INTEGRITY_RECEIPT_NAME))
            assert set(receipt["members"]) == set(archive.namelist()) - {
                "_manifest.json", export_download.INTEGRITY_RECEIPT_NAME}
        assert receipt["status"] == "unrecorded"
        assert receipt["label"] == UNRECORDED_LABEL
        assert receipt["members"] == dict(bundle.member_integrity)
    finally:
        bundle.path.unlink(missing_ok=True)
    with publication.formal_integrity_scope(propagate_to_children=False):
        with pytest.raises(export_download.ExportDownloadError) as formal:
            export_download.prepare_registered_export_bundle(
                "src_0123456789ab", registry=_registry(unrecorded),
                temp_dir=tmp_path / "zips")
    assert formal.value.code == INTEGRITY


@pytest.mark.parametrize("name,relative,entry", [
    ("r13exportsha_real_discovery", "tools/run_idea_mining_real_discovery.py",
     "build_full_wide_cohort"),
    ("r13exportsha_s6_harness", "tools/run_idea_mining_s6_validation_harness.py",
     "build_wide_cohort"),
])
def test_formerly_pending_tools_refuse_an_unrecorded_export(
        unrecorded, tmp_path, monkeypatch, name, relative, entry):
    """D27E-TOOLS: the package gate refuses what no digest can vouch for."""
    monkeypatch.syspath_prepend(str(REPO))
    tool = _load_tool(name, relative)
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        getattr(tool, entry)(unrecorded, tmp_path / "out.parquet")
    assert caught.value.reason == publication.REASON_INTEGRITY_UNRECORDED
    assert not (tmp_path / "out.parquet").exists()
    assert publication.require_verified_export_package(unrecorded, formal=None)[
        "status"] == "unrecorded"


# --- the pre-6064eaf6a web format (supervisor ruling, option a) ---------------

_SIGNATURE_PRESENT = ("data_path", "cohort_contract", "column_metadata")
_SIGNATURE_ABSENT = {"export_kind": "web_module_extraction",
                     "contract_revision": "native_v2_clinical_contract_provenance_20260812",
                     "member_integrity": {"algorithm": "sha256", "scope": "every_member"}}


def _manifest(export: Path) -> dict:
    return json.loads((export / "_manifest.json").read_text(encoding="utf-8"))


def _refused_without_digest(manifest: dict, member: str) -> None:
    assert publication.manifest_integrity_contract(manifest) == "required"
    assert publication.legacy_manifest_format(manifest) is None
    with pytest.raises(publication.ExportMemberIntegrityError) as caught:
        publication.require_member_digest(manifest, member, sha256="0" * 64, size=1, path=member)
    assert caught.value.reason == publication.REASON_DIGEST_UNRECORDED


def test_the_exact_pre_kind_web_signature_is_unrecorded_and_formally_refused(unrecorded):
    """(3) Exactly the pre-6064eaf6a web signature: UNRECORDED, labelled, never certified."""
    manifest = _manifest(unrecorded)
    assert publication.legacy_manifest_format(manifest) == publication.LEGACY_WEB_PRE_KIND
    member = member_file(unrecorded)
    assert publication.verify_member_integrity(unrecorded, member) == "unrecorded"
    summary = publication.require_verified_export_package(unrecorded, formal=None)
    assert summary["status"] == "unrecorded"
    assert summary["label"] == UNRECORDED_LABEL
    assert summary["legacy_format"] == "legacy_web_pre_6064eaf6a"
    with pytest.raises(publication.ExportMemberIntegrityError) as formal:
        publication.verify_member_integrity(unrecorded, member, formal=True)
    assert formal.value.reason == publication.REASON_INTEGRITY_UNRECORDED


@pytest.mark.parametrize("native_key", ["contract_revision", "export_kind"])
def test_a_native_manifest_with_its_digests_stripped_is_not_legacy(published, tmp_path, native_key):
    """(1)/(2) Digests stripped, a native or kinded key kept: refused, never legacy."""
    from easyicu.research_agent.intake import export_package as intake

    export = _copy(published, tmp_path / "stripped_native" / "hirid")
    make_pre_integrity_web_export(export)  # every other signature field matches
    manifest = _manifest(export)
    manifest[native_key] = _SIGNATURE_ABSENT[native_key]
    (export / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _refused_without_digest(manifest, member_file(published))
    with pytest.raises(intake.ExportPackageError) as package:
        intake.open_export_package(export)
    assert package.value.code == INTEGRITY


def test_the_real_native_manifest_with_digests_stripped_is_refused(published, tmp_path):
    """(1) The unmodified native writer's manifest, digests deleted: refused."""
    export = _copy(published, tmp_path / "native_stripped" / "hirid")
    strip_digests(export)
    manifest = _manifest(export)
    assert "contract_revision" in manifest or "export_kind" in manifest
    _refused_without_digest(manifest, member_file(published))


@pytest.mark.parametrize("change", [
    *(f"missing:{key}" for key in _SIGNATURE_PRESENT),
    *(f"extra:{key}" for key in _SIGNATURE_ABSENT),
    "extra:one_member_digest",
])
def test_one_extra_or_missing_signature_field_is_refused(unrecorded, change):
    """(4) Any deviation from the exact signature stays required."""
    manifest = _manifest(unrecorded)
    kind, key = change.split(":")
    if kind == "missing":
        manifest.pop(key)
    elif key == "one_member_digest":
        manifest["files"][0]["parquet_sha256"] = "0" * 64
    else:
        manifest[key] = _SIGNATURE_ABSENT[key]
    assert publication.legacy_manifest_format(manifest) is None
    assert publication.manifest_integrity_contract(manifest) == "required"
    other = next(e["file"] for e in manifest["files"][1:]) if len(manifest["files"]) > 1 else None
    if key != "one_member_digest":
        _refused_without_digest(manifest, member_file(unrecorded))
    elif other:
        _refused_without_digest(manifest, other)


def test_the_aumc_checker_never_certifies_an_unrecorded_export(unrecorded):
    source = (REPO / "tools/verify_aumc_ventilator_sources.py").read_text(encoding="utf-8")
    assert "'integrity_certified':set(member_integrity.values())=={'verified'}" in source
    assert "uncertified: export integrity unrecorded" in source


# --- ruling (i): a v0.1.x web export is refused with the exact re-export text --

LABEL = "integrity unrecorded · re-export to verify"


@pytest.fixture
def v01_web(published, tmp_path):
    """A web export as v0.1.x wrote it: export_kind, no digests, no declaration."""
    export = _copy(published, tmp_path / "exports" / "hirid")
    strip_digests(export)
    manifest = _manifest(export)
    manifest.pop("member_integrity", None)
    manifest["export_kind"] = "web_module_extraction"
    (export / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return export


def _no_host_path(payload, tmp_path) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    assert str(tmp_path) not in text
    assert '"/' not in text  # no string value starts with an absolute path


@pytest.mark.parametrize("reader", ["describe_export_source", "scan_path",
                                    "summarize_export_workspace"])
def test_web_readers_refuse_a_v01_export_with_the_exact_text(v01_web, tmp_path, reader):
    from easyicu.webserver import dataio

    answer = getattr(dataio, reader)(str(v01_web))
    assert answer["ok"] is False
    assert answer["reason"] == publication.REASON_DIGEST_UNRECORDED
    assert answer["message"] == LABEL
    assert answer["export_path"].startswith("hirid/") and answer["export_path"].endswith(".parquet")
    _no_host_path(answer, tmp_path)


def test_intake_refuses_a_v01_export_with_the_exact_text(v01_web, tmp_path):
    from easyicu.research_agent.intake import export_package as intake
    from easyicu.webserver import dataio

    with pytest.raises(intake.ExportPackageError) as caught:
        intake.open_export_package(v01_web)
    assert str(caught.value).startswith(f"{LABEL}: ")
    assert str(tmp_path) not in str(caught.value)
    detail = dataio._research_pipeline_intake_diagnostic(caught.value)
    assert detail["intake_error_message"].startswith(LABEL)
    _no_host_path(detail, tmp_path)


def test_the_download_refuses_a_v01_export_with_the_exact_text(v01_web, tmp_path):
    from easyicu.webserver import export_download

    with pytest.raises(export_download.ExportDownloadError) as caught:
        export_download.prepare_registered_export_bundle(
            "src_0123456789ab", registry=_registry(v01_web), temp_dir=tmp_path / "zips")
    assert caught.value.message.startswith(f"{LABEL}: hirid/")
    assert str(tmp_path) not in caught.value.message
    assert "re-run the extraction" not in caught.value.message


def test_the_package_gate_refuses_a_v01_or_unrecorded_export_with_the_exact_text(
        v01_web, unrecorded):
    with pytest.raises(publication.ExportMemberIntegrityError) as v01:
        publication.require_verified_export_package(v01_web)
    assert str(v01.value).startswith(f"{LABEL}: ")
    with pytest.raises(publication.ExportMemberIntegrityError) as legacy:
        publication.require_verified_export_package(unrecorded)
    assert str(legacy.value).startswith(f"{LABEL}: ")
    assert publication.refusal_message(v01.value) == LABEL
