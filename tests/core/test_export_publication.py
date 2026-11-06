"""An interrupted native export is never read as a finished one (B-EXPORT-PARTIAL).

``extract_database`` writes raw module outputs (native ids, raw dtypes, rows
outside the episode window) into the export folder before publication
canonicalises them. A comparison made on such a folder once produced a false
batch-dependence finding. The ``_IN_PROGRESS`` marker now covers the whole
unpublished interval, and every in-repo reader refuses a folder carrying it
(or lacking ``_manifest.json``) with ``native_export_unpublished``.
Synthetic rows only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from easyicu import export_publication as publication
from easyicu.api import extraction as api

REPO = Path(__file__).resolve().parents[2]
DATABASE = "hirid"
MODULE = "outcome"


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _InProcess:
    def __init__(self, *, target, args, daemon):
        self.target, self.args = target, args
        self.exitcode = None

    def start(self):
        try:
            self.target(*self.args)
            self.exitcode = 0
        except BaseException:
            self.exitcode = 1

    def join(self):
        return None


class _Context:
    @staticmethod
    def Process(*, target, args, daemon):
        return _InProcess(target=target, args=args, daemon=daemon)


def _extract(source: Path, output: Path):
    from easyicu.api.concepts import clear_global_loader

    try:
        return api.extract_database(
            DATABASE, data_path=str(source), output_dir=str(output), modules=[MODULE],
            native_export_v2=True, verbose=False,
        )
    finally:
        clear_global_loader()


@pytest.fixture
def source(tmp_path, monkeypatch):
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / DATABASE
    generate_source(root, DATABASE, stays=6)
    monkeypatch.setattr(api, "_get_extraction_mp_context", lambda _mp: _Context())
    return root


@pytest.fixture
def interrupted(source, tmp_path, monkeypatch):
    """A run stopped after its module outputs, before publication."""

    def stop(**_kwargs):
        raise KeyboardInterrupt  # the process is interrupted, not a publish error

    output = tmp_path / "export"
    with monkeypatch.context() as patch:
        patch.setattr(api, "_publish_native_export_v2", stop)
        with pytest.raises(KeyboardInterrupt):
            _extract(source, output)
    return output


# --- the marker contract ---------------------------------------------------

def test_the_marker_covers_the_unpublished_interval(tmp_path):
    root = tmp_path / "export"
    publication.begin_native_export(root)
    assert publication.unpublished_reason(root) == publication.REASON_IN_PROGRESS
    with pytest.raises(publication.NativeExportUnpublishedError) as early:
        publication.finish_native_export(root)  # no manifest yet: the marker stays
    assert early.value.reason == publication.REASON_MANIFEST_MISSING
    assert publication.unpublished_reason(root) == publication.REASON_IN_PROGRESS
    (root / "_manifest.json").write_text("{}", encoding="utf-8")
    assert publication.unpublished_reason(root) == publication.REASON_IN_PROGRESS
    publication.finish_native_export(root)
    assert publication.unpublished_reason(root) is None
    assert publication.require_published_native_export(root) == root / "_manifest.json"


def test_a_folder_without_a_root_manifest_is_unpublished(tmp_path):
    (tmp_path / "outcome.parquet").write_bytes(b"not read")
    with pytest.raises(publication.NativeExportUnpublishedError) as caught:
        publication.require_published_native_export(tmp_path)
    assert caught.value.code == "native_export_unpublished"
    assert caught.value.reason == publication.REASON_MANIFEST_MISSING


# --- extraction ------------------------------------------------------------

def test_an_interrupted_run_leaves_the_marker_with_its_raw_files(interrupted):
    assert (interrupted / f"{MODULE}.parquet").is_file()
    assert not (interrupted / "_manifest.json").exists()
    assert (interrupted / publication.IN_PROGRESS_MARKER).is_file()
    with pytest.raises(publication.NativeExportUnpublishedError) as caught:
        publication.require_published_native_export(interrupted)
    assert caught.value.reason == publication.REASON_IN_PROGRESS


def test_a_finished_run_is_accepted(source, tmp_path):
    from easyicu.research_agent.intake.export_package import inspect_export_layout

    output = tmp_path / "export"
    _extract(source, output)
    assert not (output / publication.IN_PROGRESS_MARKER).exists()
    assert publication.require_published_native_export(output) == output / "_manifest.json"
    assert inspect_export_layout(output)["database"] == DATABASE


def test_a_rerun_over_a_stale_partial_export_is_refused_not_cleaned(interrupted, source):
    before = sorted(path.name for path in interrupted.iterdir())
    with pytest.raises(publication.NativeExportUnpublishedError) as caught:
        _extract(source, interrupted)
    assert caught.value.reason == publication.REASON_STALE_PARTIAL
    assert sorted(path.name for path in interrupted.iterdir()) == before
    # The refusal names the remedy in plain words and never suggests deletion.
    message = str(caught.value)
    assert "unfinished export from an interrupted run" in message
    assert "Choose a new output folder" in message
    assert "move this one aside after checking it" in message
    assert "remove" not in message.lower() and "delete it" not in message.lower()


def test_a_publication_failure_quarantines_the_marker_with_the_files(source, tmp_path, monkeypatch):
    def fail(**_kwargs):
        raise ValueError("synthetic publication failure")

    output = tmp_path / "export"
    monkeypatch.setattr(api, "_publish_native_export_v2", fail)
    with pytest.raises(api.NativeExportPublicationError) as caught:
        _extract(source, output)
    quarantine = Path(caught.value.quarantine)
    assert (quarantine / publication.IN_PROGRESS_MARKER).is_file()
    assert (quarantine / f"{MODULE}.parquet").is_file()
    assert not (output / publication.IN_PROGRESS_MARKER).exists()
    assert not (output / f"{MODULE}.parquet").exists()


# --- every reader refuses ----------------------------------------------------

@pytest.fixture
def marked(interrupted):
    """The folder an interrupted run left (marker, raw files, no manifest)."""
    return interrupted


def test_intake_refuses_an_unpublished_export(marked, source, tmp_path):
    from easyicu.research_agent.intake.export_package import (
        ExportPackageError,
        inspect_export_layout,
        open_export_package,
    )

    for reader in (inspect_export_layout, open_export_package):
        with pytest.raises(ExportPackageError) as caught:
            reader(marked)
        assert caught.value.code == "native_export_unpublished"

    # Crash window: the manifest is written but the marker was not yet removed.
    finished = tmp_path / "finished"
    _extract(source, finished)
    (finished / publication.IN_PROGRESS_MARKER).write_text("", encoding="utf-8")
    with pytest.raises(ExportPackageError) as window:
        inspect_export_layout(finished)
    assert window.value.code == "native_export_unpublished"


def test_webserver_folder_readers_refuse_an_unpublished_export(marked):
    from easyicu.webserver import dataio

    for reader in (
        dataio.scan_path,
        dataio.describe_export_source,
        dataio.describe_source_tables,
        dataio.summarize_export_workspace,
    ):
        answer = reader(str(marked))
        assert answer["ok"] is False, reader.__name__
        assert answer["error"] == "native_export_unpublished", reader.__name__
    assert dataio.read_prepared_export_manifest(str(marked)) is None


def test_webserver_download_and_raw_source_binding_refuse(marked):
    from easyicu.webserver import export_download, raw_source_authority

    registry = {"sources": [{"ok": True, "id": "src_0123456789ab", "path": str(marked)}]}
    with pytest.raises(export_download.ExportDownloadError) as download:
        export_download.prepare_registered_export_bundle("src_0123456789ab", registry=registry)
    assert download.value.code == "native_export_unpublished"
    with pytest.raises(raw_source_authority.RawSourceAuthorityError) as binding:
        raw_source_authority._read_export_manifest(marked)
    assert binding.value.code == "native_export_unpublished"


def test_export_plausibility_refuses(marked):
    from easyicu.io import export_plausibility

    with pytest.raises(publication.NativeExportUnpublishedError):
        export_plausibility._views(None, marked, None)


def test_release_and_refresh_scripts_refuse(marked, tmp_path, monkeypatch):
    reextract = _load_script("r13partial_reextract", "tools/reextract_native_export_v2.py")
    with pytest.raises(reextract.ExtractionRunError, match="native_export_unpublished"):
        reextract._validate_export_package(marked, "0" * 40)

    republisher = _load_script(
        "r13partial_republisher", "scripts/releases/EX-A02_republish_full6_candidate.py"
    )
    with pytest.raises(republisher.RepublicationError, match="native_export_unpublished"):
        republisher._source_database_receipt(marked)

    refresh = _load_script(
        "r13partial_refresh", "scripts/releases/EX-A03_refresh_selected_modules.py"
    )
    with pytest.raises(refresh.ModuleRefreshError, match="native_export_unpublished"):
        refresh._validate_existing_staging_native_manifest(
            database=DATABASE, staging_root=marked, modules=[MODULE]
        )

    sealer = _load_script("r13partial_sealer", "scripts/releases/EX-A01_seal_full6_release.py")
    monkeypatch.setattr(sealer, "_validate_foundation_lock", lambda: {})
    paths = dict(sealer.FOUNDATION_RESOURCE_PATHS)
    paths["concept_dictionary_sha256"] = tmp_path / "no-overlay" / "concept-dict.json"
    monkeypatch.setattr(sealer, "FOUNDATION_RESOURCE_PATHS", paths)
    run_root = tmp_path / "run"
    for database in sealer.DATABASES:
        root = run_root / "exports" / database
        root.mkdir(parents=True)
        (root / publication.IN_PROGRESS_MARKER).write_text("", encoding="utf-8")
    with pytest.raises(sealer.ReleaseValidationError, match="native_export_unpublished"):
        sealer.validate_release(run_root)


# --- the web cohort export ---------------------------------------------------

class _Job:
    def emit(self, _payload):
        pass


@pytest.fixture
def web_source(tmp_path):
    from easyicu.webserver import dataio
    from easyicu.webserver.demo_source_contracts import SOURCES
    from tests.support.windows_laptop_demo import isolated_demo_paths, synthetic_demo_download

    source = next(item for item in SOURCES if item.database == "eicu_demo")
    paths = isolated_demo_paths(tmp_path, source)
    synthetic_demo_download(source, paths, _Job())
    assert dataio.make_convert_runner(str(paths.raw), "eicu")(_Job())["failed"] == 0
    return paths


def _web_export(paths):
    from easyicu.webserver import dataio

    return dataio.make_export_runner(
        str(paths.raw), "eicu", modules=["demographics"], out_dir=str(paths.export),
        export_format="parquet", include_feature_definitions=False,
    )(_Job())


def test_a_web_export_interrupted_after_the_manifest_delete_is_refused(web_source, monkeypatch):
    from easyicu.webserver import dataio

    assert _web_export(web_source)["manifest"] == "_manifest.json"
    export = web_source.export
    assert not (export / publication.IN_PROGRESS_MARKER).exists()
    assert dataio.scan_path(str(export))["ok"] is True

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(dataio, "_write_frame", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _web_export(web_source)
    assert not (export / "_manifest.json").exists()  # the old manifest is gone
    assert (export / publication.IN_PROGRESS_MARKER).is_file()
    for reader in (
        dataio.scan_path,
        dataio.describe_export_source,
        dataio.describe_source_tables,
        dataio.summarize_export_workspace,
    ):
        answer = reader(str(export))
        assert answer["ok"] is False, reader.__name__
        assert answer["error"] == "native_export_unpublished", reader.__name__
        assert answer["reason"] == publication.REASON_IN_PROGRESS, reader.__name__

    # A new web export into the same folder is refused in plain words.
    with pytest.raises(dataio.ExportCohortError) as again:
        _web_export(web_source)
    assert again.value.error == "native_export_unpublished"
    assert "Choose a new output folder" in again.value.detail["message"]


def test_an_export_folder_without_a_manifest_is_refused_but_a_source_is_not(web_source):
    from easyicu.webserver import dataio

    # A prepared (converted) source folder is not an export: still served.
    assert dataio.scan_path(str(web_source.raw))["ok"] is True
    assert publication.folder_unpublished_reason(web_source.raw) is None

    # An export folder (producer manifests) without a root manifest and with no
    # marker, as a pre-marker interrupted run left it, is refused.
    _web_export(web_source)
    export = web_source.export
    (export / "_manifest.json").unlink()
    (export / "demographics.manifest.json").write_text("{}", encoding="utf-8")
    answer = dataio.describe_export_source(str(export))
    assert answer["ok"] is False
    assert (answer["error"], answer["reason"]) == (
        "native_export_unpublished", publication.REASON_MANIFEST_MISSING,
    )
