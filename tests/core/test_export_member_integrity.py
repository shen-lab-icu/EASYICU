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
    """A pre-digest (listing-only) manifest whose outcome member is truncated."""
    export = damaged_copy(published, tmp_path / "legacy" / "hirid", "truncated")
    strip_digests(export)
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


def test_a_pre_digest_manifest_is_listing_free_but_still_refuses_unreadable(
        truncated_undigested):
    member = member_file(truncated_undigested)
    assert (publication.verify_member_integrity(truncated_undigested, member)
            == publication.INTEGRITY_UNRECORDED)


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
    with pytest.raises(publication.ExportMemberUnreadableError):
        agent_outputs._load_context(str(truncated_undigested), _agent_source(published))


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
